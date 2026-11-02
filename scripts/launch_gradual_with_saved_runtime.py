"""Restore recorded CPU concurrency before executing the unchanged gradual CLI.

The metadata reader is a separate, GPU-hidden process. It authenticates an
existing development checkpoint and returns primitive runtime metadata only.
The training process sets CPU threads before importing the scientific engine;
all original preparation, checkpoint and training validations still execute.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys

VERSION = "gradual_saved_runtime_launcher_v1.0"
THREAD_KEYS = ("cpu_threads", "interop_threads")
RUNTIME_TYPES = {"python": str, "torch": str, "cuda": str, "device": str,
    "cpu_threads": int, "interop_threads": int, "deterministic": bool,
    "deterministic_warn_only": bool, "cublas_workspace_config": str,
    "matmul_allow_tf32": bool, "cudnn_allow_tf32": bool, "cudnn_benchmark": bool,
    "cudnn_deterministic": bool, "device_name": str, "device_capability": list}


def _require(condition, code):
    if not condition:
        raise ValueError(code)


def validate_runtime(value):
    _require(isinstance(value, dict) and set(value) == set(RUNTIME_TYPES),
        "exact_saved_cuda_runtime_metadata_required")
    _require(all(type(value[key]) is kind for key, kind in RUNTIME_TYPES.items()),
        "primitive_saved_runtime_metadata_required")
    _require(all(1 <= value[key] <= 1024 for key in THREAD_KEYS)
        and value["device"] == "cuda:0" and len(value["device_capability"]) == 2
        and all(type(item) is int and 0 <= item <= 100 for item in value["device_capability"]),
        "bounded_saved_cpu_threads_and_cuda_device_required")
    return json.loads(json.dumps(value, allow_nan=False))


def differences(saved, current):
    return {key: {"saved": saved.get(key), "current": current.get(key)}
        for key in sorted(set(saved) | set(current)) if saved.get(key) != current.get(key)}


def configure_threads(torch, saved):
    """Called in a fresh process before any model, seed, probe or tensor work."""
    before = {key: getter() for key, getter in (
        ("cpu_threads", torch.get_num_threads), ("interop_threads", torch.get_num_interop_threads))}
    for key, setter in (("interop_threads", torch.set_num_interop_threads),
                         ("cpu_threads", torch.set_num_threads)):
        if before[key] != saved[key]:
            try:
                setter(saved[key])
            except RuntimeError as error:
                raise RuntimeError("fresh_process_required_to_restore_saved_cpu_threads") from error
    after = {"cpu_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads()}
    _require(after == {key: saved[key] for key in THREAD_KEYS}, "saved_cpu_threads_not_restored")
    return before


def _read_json(path, maximum=64 * 1024**2):
    from knee.colab_data import checked_path
    path = checked_path(path)
    _require(path.is_file() and 0 < path.stat().st_size <= maximum, "bounded_plain_runtime_metadata_required")
    return json.loads(path.read_text(encoding="utf-8"))


def find_campaign(records, identity, core):
    """Same bounded identity lookup as the core, without creating a campaign."""
    from knee.colab_data import checked_path
    matches = []
    for count, candidate in enumerate(records.iterdir(), 1):
        _require(count <= core.transport.MAX_DIRECT_CHILDREN, "bounded_gradual_resume_inventory_required")
        if not candidate.name.startswith("gradual_image_campaign_"):
            continue
        candidate = checked_path(candidate)
        if not candidate.is_dir() or not (candidate / "campaign_identity.json").is_file():
            continue
        manifest = _read_json(candidate / "campaign_identity.json")
        _require(manifest.get("version") == core.VERSION, "registered_gradual_campaign_manifest_required")
        if manifest.get("identity") == identity:
            matches.append(candidate)
        elif manifest.get("identity", {}).get("development_package_sha256") == identity["development_package_sha256"]:
            raise ValueError("existing_gradual_campaign_incompatible_do_not_restart_or_change_recipe")
    _require(len(matches) <= 1, "ambiguous_gradual_campaigns_must_not_restart_silently")
    return matches[0] if matches else None


def checkpoint_metadata(checkpoint, latest):
    """Full identity/inventory/SHA checks precede safe, lazy CPU deserialization."""
    import torch
    epoch, path, manifest = latest
    print(f"Verificando el estado guardado de la época {epoch}; no se ejecutan actualizaciones…",
        file=sys.stderr, flush=True)
    state_path = checkpoint._checked_state_path(path, manifest)
    state = torch.load(state_path, map_location="cpu", weights_only=True, mmap=True)
    try:
        _require(isinstance(state, dict) and state.get("identity") == checkpoint.identity
            and type(state.get("epoch")) is int and state["epoch"] == epoch,
            "authenticated_saved_runtime_identity_and_epoch_required")
        return {"epoch": epoch, "runtime": validate_runtime(state.get("runtime"))}
    finally:
        del state
        gc.collect()


def inspect_saved_runtime(args):
    from knee import gradual_campaign as core, gradual_campaign_colab as orchestration
    from knee.colab_data import checked_path, sha_file
    from knee.gradual_epoch_resume import GradualEpochCheckpoint
    from knee.training_authority import digest
    repository, records, proof = map(checked_path,
        (args.repository_root, args.records_root, args.preparation_dir))
    orchestration.validate_preparation_receipt(repository, proof, code_revision=args.code_revision)
    prepared = _read_json(proof / core.PREPARATION_FILENAME)
    _, _, plan, registry = core.release.load_release(repository)
    evidence_hash = sha_file(proof / core.SELECTION_FILENAME)
    identity = {"code_revision": args.code_revision, "stage": "stage_2", "seed": core.SEED,
        "model_order": list(core.MODELS), "training_plan_sha256": core.release.PINS[core.release.PLAN_PATH],
        "modeling_release_sha256": core.release.PINS[core.release.RELEASE_PATH],
        "development_package_sha256": prepared["development_package_sha256"],
        "model_input_reference_sha256": sha_file(checked_path(args.reference_path)),
        "selection_evidence_sha256": evidence_hash, "source_files_sha256": core._sources(repository)}
    _require(prepared["record_sha256"] == digest({k: v for k, v in prepared.items() if k != "record_sha256"})
        and prepared["source_files_sha256"] == identity["source_files_sha256"]
        and prepared["model_input_reference_sha256"] == identity["model_input_reference_sha256"],
        "unchanged_authenticated_gradual_preparation_required")
    campaign = find_campaign(records, identity, core)
    if campaign is None:
        return {"version": VERSION, "status": "no_saved_checkpoint", "checkpoints": []}
    allowed = {f"clinical_main__{model}__fold_{fold}" for model in core.MODELS for fold in core.FOLDS}
    completed = set()
    for count, path in enumerate(campaign.iterdir(), 1):
        _require(count <= 512, "bounded_gradual_attempts_required")
        if path.name.startswith("job_attempt_") and path.is_dir() and (path / core.transport.JOB_RECEIPT).is_file():
            receipt = _read_json(path / core.transport.JOB_RECEIPT)
            key = receipt.get("key")
            _require(receipt.get("version") == core.VERSION and key in allowed and key not in completed,
                "unique_registered_gradual_completed_job_metadata_required")
            completed.add(key)
    pending, prior = [], []
    for model in core.MODELS:
        batch = campaign / (model + "__batch.json")
        for fold in core.FOLDS:
            key = f"clinical_main__{model}__fold_{fold}"
            directory = campaign / "epoch_checkpoints" / key
            if not directory.exists():
                continue
            micro = _read_json(batch)["micro_batch_size"]
            request = core.make_request(prepared["development_package_sha256"], evidence_hash,
                model, fold, micro, plan, registry)
            checkpoint = GradualEpochCheckpoint(directory, request)
            latest = checkpoint._latest()
            if latest is not None:
                (prior if key in completed else pending).append((key, checkpoint, latest))
    # Pending states must agree. A completed state is only an execution-profile
    # anchor when the next job has no partial checkpoint; its weights are unused.
    selected = pending or prior[:1]
    checked = []
    for key, checkpoint, latest in selected:
        metadata = checkpoint_metadata(checkpoint, latest)
        checked.append({"key": key, **metadata})
    if not checked:
        return {"version": VERSION, "status": "no_saved_checkpoint", "checkpoints": []}
    saved = checked[0]["runtime"]
    _require(all(item["runtime"] == saved for item in checked), "conflicting_saved_gradual_runtime_profiles")
    return {"version": VERSION, "status": "authenticated_saved_runtime", "runtime": saved,
        "completed_job_metadata_count": len(completed),
        "checkpoints": [{"key": item["key"], "epoch": item["epoch"]} for item in checked]}


def read_in_separate_process(argv):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", COLAB_GPU="0", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-u", "-B", str(Path(__file__).resolve()),
        *argv, "--runtime-inspection-only"], env=env, stdout=subprocess.PIPE, text=True, check=False)
    _require(result.returncode == 0, "saved_runtime_inspection_failed_before_training")
    _require(len(result.stdout) <= 1024**2, "bounded_saved_runtime_inspection_output_required")
    value = json.loads(result.stdout)
    _require(value.get("version") == VERSION and value.get("status") in
        {"no_saved_checkpoint", "authenticated_saved_runtime"}, "registered_saved_runtime_inspection_required")
    if value["status"] == "authenticated_saved_runtime":
        validate_runtime(value.get("runtime"))
    return value


def compatibility_before_training(torch, inspection):
    report = {"version": VERSION, "status": inspection["status"],
        "checkpoints": inspection["checkpoints"], "training_started": False}
    if inspection["status"] == "no_saved_checkpoint":
        return report
    saved = validate_runtime(inspection["runtime"])
    before = configure_threads(torch, saved)
    # Imports and deterministic context occur only after concurrency is restored.
    from knee.training_smoke import verify_gpu_environment
    from knee.aligned_training.engine import _seeded_execution
    from knee.training_epoch_resume import _runtime
    verify_gpu_environment("google_colab_gpu")
    with _seeded_execution(2026):
        current = _runtime(torch.device("cuda:0"))
    report.update(cpu_threads_before=before,
        cpu_threads_restored={key: current[key] for key in THREAD_KEYS},
        differences_before=differences(saved, {**current, **before}),
        differences_after=differences(saved, current))
    print("Compatibilidad de recuperación: " + json.dumps(report, ensure_ascii=False, sort_keys=True),
        file=sys.stderr, flush=True)
    _require(not report["differences_after"], "saved_gradual_runtime_incompatible_details_recorded")
    print("Entorno guardado recuperado antes de cargar modelos; se conservan la RAM alta y las comprobaciones originales.",
        file=sys.stderr, flush=True)
    return report


def report_restore_failure(error):
    """Expose individual failed guard groups without printing participant values."""
    if not isinstance(error, ValueError) or str(error) != "same_gradual_recipe_runtime_initial_weights_and_epoch_state_required":
        return
    trace = error.__traceback__
    while trace is not None:
        frame = trace.tb_frame
        if frame.f_globals.get("__name__") == "knee.gradual_epoch_resume" and frame.f_code.co_name == "restore":
            from dataclasses import asdict
            from knee.training_epoch_resume import _runtime
            from knee.aligned_training.engine import epoch_phase, _seeded_execution
            local = frame.f_locals
            state, checkpoint, epoch = local["state"], local["self"], local["epoch"]
            if isinstance(state, dict):
                monitor = state.get("monitor", {})
                history = state.get("history", [])
                checks = {"state_fields": set(state) == local["required"],
                    "identity": state.get("identity") == checkpoint.identity,
                    "epoch": state.get("epoch") == epoch,
                    "binding": state.get("binding") == local["binding"],
                    "initial_weights": state.get("initial_state_sha256") == checkpoint.initial_state_digest,
                    "scheduler_epoch": state.get("scheduler_epoch") == epoch,
                    "phase": state.get("phase") == epoch_phase(checkpoint.request["model_id"], "stage_2", epoch),
                    "history_epochs": [item.get("epoch") for item in history] == list(range(1, epoch + 1)),
                    "history_phases": all(item.get("phase") == epoch_phase(checkpoint.request["model_id"], "stage_2", item["epoch"]) for item in history),
                    "total_updates": state.get("total_updates") == sum(item.get("optimizer_updates", 0) for item in history),
                    "monitor_fields": set(monitor) == set(asdict(local["monitor"])),
                    "patience": monitor.get("patience") == local["monitor"].patience,
                    "monitor_start": monitor.get("monitor_starts_after_epoch") == local["monitor"].monitor_starts_after_epoch == 10,
                    "stopped_type": type(state.get("stopped")) is bool}
                # Capture runtime in the failing frame's deterministic context:
                # unwinding has restored outer deterministic flags already.
                with _seeded_execution(2026):
                    current = _runtime(local["device"])
                detail = {"version": VERSION, "guard_checks": checks,
                    "runtime_differences": differences(state.get("runtime", {}), current),
                    "checkpoint_not_modified": True}
                print("Detalle de recuperación detenida: " + json.dumps(detail, sort_keys=True), file=sys.stderr, flush=True)
            break
        trace = trace.tb_next


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", required=True, choices=("train",))
    parser.add_argument("--campaign", required=True, choices=("gradual_four_models",))
    for key in ("repository-root", "records-root", "preparation-dir", "package-dir",
                "reference-path", "weights-root", "output-dir"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--runtime-inspection-only", action="store_true")
    args = parser.parse_args(argv)
    _require(re.fullmatch(r"[0-9a-f]{40}", args.code_revision or "") is not None,
        "pinned_frozen_gradual_revision_required")
    if not args.runtime_inspection_only:
        inspection = read_in_separate_process(argv)
        import torch
        # No scientific imports/tensor operations have occurred in this process.
        sys.path.insert(0, str(args.repository_root / "src"))
        compatibility_before_training(torch, inspection)
        script = args.repository_root / "scripts/train_gradual_campaign_colab.py"
        previous = sys.argv
        try:
            sys.argv = [str(script), *argv]
            runpy.run_path(str(script), run_name="__main__")
        except BaseException as error:
            try:
                report_restore_failure(error)
            except Exception as diagnostic_error:
                print("No se pudo ampliar el diagnóstico: " + type(diagnostic_error).__name__,
                    file=sys.stderr, flush=True)
            raise
        finally:
            sys.argv = previous
    else:
        _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "runtime_inspection_must_hide_gpu")
        sys.path.insert(0, str(args.repository_root / "src"))
        print(json.dumps(inspect_saved_runtime(args), ensure_ascii=False, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
