"""Development-only stage-two campaign; preparation never trains a model.

Stage-one predictions establish eligibility, not fitted initialization. Every
new fold starts from the registered public weights. Recovery uses the same
request, recipe, selection and implementation, and never opens reserved data.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import uuid

from knee import methodology_campaign as prior
from knee import methodology_campaign_colab as transport
from knee import methodology_release as release
from knee.aligned_training import runner
from knee.aligned_training.authority import digest, validate_request, validate_authority
from knee.colab_data import checked_path, require, sha_file, valid_sha
from knee.training_preparation import read_json

VERSION = "development_gradual_image_campaign_v1.0"
STATUS = "four_model_gradual_development_campaign_complete"
CAMPAIGN = "gradual_four_models"
FILENAME = "gradual_campaign.json"
PREPARATION_FILENAME = "preparation_record.json"
SELECTION_FILENAME = "selection_evidence.json"
AUTHORITY_FILENAME = "gradual_screening_authority.json"
MODELS = ("image_densenet121", "image_vit_b16", "image_skelex", "image_mambaout_tiny")
FOLDS = tuple(range(5))
SEED = 2026
AUTHORIZATION_STATEMENT = (
    "Autorizo el ajuste gradual de DenseNet121, ViT-B/16, Skelex y MambaOut-Tiny "
    "en los mismos cinco pliegues de desarrollo, receta v1.2 y semilla 2026: "
    "cinco épocas de cabeza, cinco del último bloque y después el codificador "
    "completo, con parada temprana y máximo total de 60 épocas. Conservar "
    "trabajos completos y recuperar épocas guardadas; sin abrir la prueba "
    "reservada ni ejecutar fusión, calibración, umbral o ajuste final.")
SOURCE_FILES = (
    "src/knee/gradual_campaign.py", "src/knee/gradual_campaign_colab.py",
    "src/knee/gradual_selection.py", "src/knee/gradual_epoch_resume.py",
    "src/knee/aligned_training/authority.py", "src/knee/aligned_training/engine.py",
    "src/knee/aligned_training/models.py", "src/knee/aligned_training/data.py",
    "src/knee/aligned_training/runner.py", "src/knee/training_augmentation.py",
    "src/knee/clinical_scenarios.py", "src/knee/methodology_release.py",
    "scripts/train_gradual_campaign_colab.py", "requirements-training-gpu.txt")
JOB_FILES = {"training_job.json", "training_record.json", "clinical_state.json",
    "best_validation_predictions.json", "best_model.safetensors", "last_model.safetensors"}
TENSOR_FILES = {"best_model.safetensors", "last_model.safetensors"}
LIMITS = {"image_densenet121": 64 * 1024**2, **transport.IMAGE_CHECKPOINT_LIMITS}


def _sources(repository):
    return {name: sha_file(repository / name) for name in SOURCE_FILES}


def _separate(target, protected):
    require(all(not target.is_relative_to(p) and not p.is_relative_to(target) for p in protected),
        "gradual_outputs_and_cache_must_be_separate_from_inputs")


def _find_stage_one(records, campaign, package_sha256):
    # An earlier failed run may have created its identity before any job was
    # trained. The privately approved completed summary determines the source;
    # directory presence, timestamps and enumeration order grant no eligibility.
    from knee.gradual_selection import find_authorized_campaign
    return find_authorized_campaign(records, campaign, package_sha256,
        authority_path=checked_path(records) / AUTHORITY_FILENAME)


def prepare(repository_root, records_root, package_dir, reference_path, output_dir, *,
            code_revision, cpu_tests=None, runtime=None):
    repository, records, package, reference, output = map(checked_path,
        (repository_root, records_root, package_dir, reference_path, output_dir))
    require(not output.exists() and output.parent.is_dir() and records.is_dir()
        and re.fullmatch(r"[0-9a-f]{40}", code_revision or "") is not None,
        "published_gradual_preparation_and_new_output_required")
    _separate(output, (repository, records, package, reference.parent))
    _, _, plan, registry = release.load_release(repository)
    proof = transport.find_preparation(records)
    _, preflight, _ = prior.validate_preparation(proof, repository)
    require(sha_file(reference) == preflight["sources_sha256"]["notebook32_reference"],
        "same_development_loading_reference_required")
    from knee.aligned_training.data import load_sealed_inputs
    loaded = load_sealed_inputs(package, reference, repository, preflight["development_package_sha256"])
    rows = loaded["manifest"]["rows"]
    require(len(rows) == preflight["development_knees"]
        and len({r["src_subject_id"] for r in rows}) == preflight["development_participants"],
        "same_verified_development_membership_required")
    del loaded
    campaign40 = _find_stage_one(records, "densenet_frozen", preflight["development_package_sha256"])
    campaign41 = _find_stage_one(records, "remaining_encoders_frozen", preflight["development_package_sha256"])
    output.mkdir()
    from knee.gradual_selection import prepare_selection
    selection = prepare_selection(campaign40, campaign41, output / SELECTION_FILENAME, plan, registry, rows,
        authority_path=records / AUTHORITY_FILENAME)
    # The selected model order is a fixed authorization boundary, not a new
    # search to be repeated on a held-out test set.
    require(selection.get("model_order") == list(MODELS), "exact_four_approved_gradual_models_required")
    result = {"version": VERSION, "status": "gradual_campaign_prepared_no_training",
        "scope": "development_only", "stage": "stage_2", "code_revision": code_revision,
        "model_order": list(MODELS), "job_count": 20, "seed": SEED, "folds": list(FOLDS),
        "development_package_sha256": preflight["development_package_sha256"],
        "development_knees": len(rows), "development_participants": preflight["development_participants"],
        "model_input_reference_sha256": sha_file(reference),
        "selection_evidence_sha256": sha_file(output / SELECTION_FILENAME),
        "source_files_sha256": _sources(repository),
        "training_plan_sha256": release.PINS[release.PLAN_PATH],
        "modeling_release_sha256": release.PINS[release.RELEASE_PATH],
        "training_executed": False, "optimizer_created": False,
        "reserved_test_accessed": False, "originals_modified": False, "partitions_modified": False}
    result["record_sha256"] = digest(result)
    runner.save_json(output / PREPARATION_FILENAME, result)
    return result


def make_request(package_sha256, evidence_sha256, model, fold, micro, plan, registry):
    require(model in MODELS and fold in FOLDS and type(fold) is int and micro in (1, 2, 4),
        "only_approved_gradual_job_membership_required")
    from knee.training_augmentation import VERSION as AUGMENTATION_VERSION
    request = {"version": release.REQUEST_VERSION, "operation": "development_fold", "model_id": model,
        "image_encoder_id": model, "stage": "stage_2", "validation_fold": fold, "seed": SEED,
        "development_package_sha256": package_sha256, "image_micro_batch_size": micro,
        "selection_evidence_sha256": evidence_sha256, "fold_image_job_sha256": None,
        "clinical_scenario": "clinical_main", "baseline_sidecar_sha256": None,
        "augmentation_policy": AUGMENTATION_VERSION, "modeling_release_sha256": release.PINS[release.RELEASE_PATH]}
    validate_request(request, plan, registry)
    return request


def _authority(request, plan, registry, probe):
    require(probe.get("model_id") == request["model_id"]
        and probe.get("full_backward_and_optimizer_step_verified") is True
        and probe.get("architecture_mapping_verified") is True
        and probe.get("micro_batch_size") == request["image_micro_batch_size"],
        "actual_model_specific_gradual_gpu_probe_required")
    value = {"version": "development_training_authority_v1.0", "training_authorized": True,
        "reserved_test_access_authorized": False, "approval_source": "direct_investigator_authorization",
        "approval_statement": AUTHORIZATION_STATEMENT, "approved_at_utc": datetime.now(timezone.utc).isoformat(),
        "request_sha256": digest(request), "modeling_release_sha256": release.PINS[release.RELEASE_PATH],
        "eligibility_gates": {key: True for key in registry["eligibility_gates"]},
        "full_encoder_training_authorized": True}
    validate_authority(value, request, plan, registry)
    return value


def _open_campaign(records, identity):
    prefix = "gradual_image_campaign_"
    matches = []
    for count, path in enumerate(records.iterdir(), 1):
        require(count <= transport.MAX_DIRECT_CHILDREN, "bounded_gradual_resume_inventory_required")
        if path.name.startswith(prefix) and path.is_dir() and (path / "campaign_identity.json").is_file():
            manifest = read_json(checked_path(path / "campaign_identity.json"))
            require(manifest.get("version") == VERSION, "registered_gradual_campaign_manifest_required")
            if manifest.get("identity") == identity:
                matches.append(checked_path(path))
            elif manifest.get("identity", {}).get("development_package_sha256") == identity["development_package_sha256"]:
                raise ValueError("existing_gradual_campaign_incompatible_do_not_restart_or_change_recipe")
    require(len(matches) <= 1, "ambiguous_gradual_campaigns_must_not_restart_silently")
    if matches:
        return matches[0]
    target = records / (prefix + uuid.uuid4().hex)
    target.mkdir()
    transport._publish_json(target / "campaign_identity.json", {"version": VERSION, "identity": identity})
    return target


def _validate_history(engine, job, request, rows):
    from knee.aligned_training.engine import VERSION as ENGINE_VERSION, EarlyStopping, epoch_phase, learning_rate_factor
    from knee.densenet_fivefold import _validate_amp
    from knee.gradual_epoch_resume import VERSION as CHECKPOINT_VERSION
    train = [r for r in rows if r["cv_fold"] != request["validation_fold"]]
    valid = [r for r in rows if r["cv_fold"] == request["validation_fold"]]
    identities = sorted([[r["src_subject_id"], r["side"]] for r in train])
    expected_binding = {"validation_fold": request["validation_fold"],
        "development_package_sha256": request["development_package_sha256"],
        "train_identity_sha256": digest(identities), "request_sha256": digest(request),
        "training_knees": len(train), "validation_knees": len(valid),
        "trained_participants": sorted({r["src_subject_id"] for r in train})}
    require(engine.get("version") == ENGINE_VERSION and engine.get("stage") == "stage_2"
        and engine.get("model_id") == request["model_id"] and engine.get("seed") == SEED
        and engine.get("status") == "training_fold_complete" and engine.get("smoke") is False
        and engine.get("scope") == "development_only" and engine.get("training_executed") is True
        and engine.get("optimizer_created") is True and engine.get("eligible_for_model_selection") is True
        and engine.get("reserved_test_accessed") is False and engine.get("restored_best_checkpoint") is True
        and engine.get("resume_supported") is True and engine.get("optimizer_state_saved") is True
        and engine.get("epoch_checkpoint_version") == CHECKPOINT_VERSION
        and engine.get("provenance") == expected_binding
        and engine.get("effective_batch_size") == 16 and engine.get("micro_batch_size") == request["image_micro_batch_size"]
        and engine.get("device") == "cuda:0" and engine.get("precision") == "float16_autocast_grad_scaler"
        and engine.get("gpu_resource_profile_recorded") is True and engine.get("scheduler_horizon_epochs") == 60,
        "exact_completed_gradual_training_engine_required")
    history = engine.get("history")
    require(isinstance(history, list) and 11 <= len(history) <= 60 and engine.get("completed_epochs") == len(history),
        "complete_gradual_history_after_unfreezing_required")
    monitor, stopped = EarlyStopping(patience=8, monitor_starts_after_epoch=10), False
    updates = math.ceil(len(train) / 16)
    for epoch, item in enumerate(history, 1):
        require(not stopped and item.get("epoch") == epoch and item.get("phase") == epoch_phase(request["model_id"], "stage_2", epoch)
            and item.get("training_knees_seen") == len(train) and item.get("optimizer_updates") == updates
            and prior._finite(item.get("weighted_loss_per_knee"))
            and item.get("learning_rate_factor") == learning_rate_factor(epoch)
            and set(item.get("validation", {})) == {"average_precision_score", "roc_auc", "brier"}
            and all(prior._finite(v, maximum=1) for v in item["validation"].values()),
            "whole_registered_gradual_epoch_required")
        improved, stopped = monitor.update(epoch, item["validation"]["average_precision_score"])
        require(item.get("selected_best") is improved, "gradual_epoch_selection_must_start_after_epoch_ten")
    positive = sum(r["PROGRESION_48M"] for r in train)
    require((stopped or len(history) == 60) and engine.get("best_epoch") == monitor.best_epoch
        and engine.get("best_validation_ap") == monitor.best_score
        and job["metrics"] == history[monitor.best_epoch - 1]["validation"]
        and engine.get("optimizer_updates") == updates * len(history)
        and positive > 0 and math.isclose(engine.get("positive_weight", -1), (len(train) - positive) / positive, rel_tol=1e-12)
        and engine.get("checkpoint_sha256") == {name: job["files_sha256"][name] for name in TENSOR_FILES}
        and type(engine.get("resumed_from_epoch")) is int and 0 <= engine["resumed_from_epoch"] <= len(history),
        "authenticated_gradual_best_epoch_and_training_accounting_required")
    _validate_amp(engine)


def read_completed_job(directory, receipt, request, rows, plan, registry):
    """Reauthenticate small records; old tensor payloads are not read/copied."""
    from knee import image_campaign_recovery as recovery
    root = checked_path(directory)
    require(receipt.get("version") == VERSION and receipt.get("key") == prior.job_key(request)
        and set(receipt.get("files_sha256", {})) == JOB_FILES
        and set(receipt.get("checkpoint_sizes", {})) == TENSOR_FILES,
        "complete_gradual_publication_receipt_required")
    paths = {p.name: checked_path(p) for p in root.iterdir()}
    require(set(paths) == JOB_FILES and all(p.is_file() for p in paths.values()), "exact_gradual_job_inventory_required")
    for name in TENSOR_FILES:
        require(0 < paths[name].stat().st_size == receipt["checkpoint_sizes"][name] <= LIMITS[request["model_id"]]
            and valid_sha(receipt["files_sha256"][name]), "bounded_gradual_tensor_stat_required")
    values = {name: recovery._small_json(paths[name], receipt["files_sha256"][name])[0] for name in JOB_FILES - TENSOR_FILES}
    job, engine, state, predictions = (values[name] for name in
        ("training_job.json", "training_record.json", "clinical_state.json", "best_validation_predictions.json"))
    validate_request(request, plan, registry)
    require(job.get("version") == runner.VERSION and job.get("request") == request
        and job.get("scope") == "development_only" and job.get("training_executed") is True
        and job.get("engine_status") == "training_fold_complete" and job.get("eligible_for_model_selection") is True
        and job.get("request_sha256") == digest(request) and job.get("plan_sha256") == release.PINS[release.PLAN_PATH]
        and job.get("reserved_test_accessed") is False and job.get("record_sha256") == receipt.get("job_record_sha256")
        and job["record_sha256"] == digest({k:v for k,v in job.items() if k != "record_sha256"})
        and job.get("files_sha256") == {k:v for k,v in receipt["files_sha256"].items() if k != runner.JOB_FILENAME}
        and job.get("eligibility_gates") == {key:True for key in registry["eligibility_gates"]}
        and isinstance(job.get("selection_evidence"),dict)
        and job["selection_evidence"].get("promoted_modern_models") == list(MODELS[2:])
        and job["selection_evidence"].get("selected_image_encoder_id") is None,
        "authenticated_completed_gradual_job_required")
    from knee import clinical_scenarios as clinical
    require(state == clinical.fit_scenario_fold(rows, request["validation_fold"], "clinical_main"),
        "gradual_clinical_state_must_be_training_fold_only")
    train = [r for r in rows if r["cv_fold"] != request["validation_fold"]]
    valid = [r for r in rows if r["cv_fold"] == request["validation_fold"]]
    participants = sorted({r["src_subject_id"] for r in train})
    require(not set(participants) & {r["src_subject_id"] for r in valid}
        and job.get("weights_provenance") == {"fit_scope":"development_training_fold", "validation_fold":request["validation_fold"],
            "train_identity_sha256":state["train_identity_sha256"], "trained_participants":participants, "reserved_test_accessed":False},
        "participant_disjoint_gradual_provenance_required")
    from knee.aligned_training.engine import binary_metrics
    require(set(predictions) == {"identities", "targets", "logits"}
        and predictions["identities"] == [[r["src_subject_id"],r["side"]] for r in valid]
        and predictions["targets"] == [r["PROGRESION_48M"] for r in valid]
        and len(predictions["logits"]) == len(valid)
        and binary_metrics(predictions["targets"],predictions["logits"]) == job.get("metrics"),
        "exact_gradual_held_out_predictions_required")
    _validate_history(engine, job, request, rows)
    return {"directory":root, "receipt":receipt, "record":job, "engine":engine, "predictions":predictions}


def _completed(campaign, package_hash, evidence_hash, rows, plan, registry):
    result = {}
    for count, path in enumerate(campaign.iterdir(), 1):
        require(count <= 512, "bounded_gradual_attempts_required")
        if not path.name.startswith("job_attempt_") or not path.is_dir() or not (path / transport.JOB_RECEIPT).is_file():
            continue
        receipt = read_json(checked_path(path / transport.JOB_RECEIPT))
        require({p.name for p in path.iterdir()} == {"job", transport.JOB_RECEIPT}, "exact_committed_gradual_attempt_required")
        saved = read_json(checked_path(path / "job" / runner.JOB_FILENAME)).get("request", {})
        request = make_request(package_hash, evidence_hash, saved.get("model_id"), saved.get("validation_fold"),
            saved.get("image_micro_batch_size"), plan, registry)
        key = prior.job_key(request)
        require(key not in result, "duplicate_gradual_completed_jobs_forbidden")
        result[key] = read_completed_job(path / "job", receipt, request, rows, plan, registry)
    return result


def _probe(model_id, catalog, plan, weights, output, *, fixed_micro=None, allow_download=True):
    """Synthetic CUDA backward plus AdamW checks memory before opening a fold.

    A smaller registered microbatch changes only accumulation, never the
    effective batch or learning-rate recipe. An interrupted job retains its
    recorded microbatch, including on a replacement Colab runtime.
    """
    import torch
    from knee.aligned_training import models, engine
    from knee.final_models_preflight import checkpoint_files, synthetic_batch
    paths = checkpoint_files(model_id, catalog, weights, allow_download)
    attempts = (fixed_micro,) if fixed_micro is not None else (4,2,1)
    for micro in attempts:
        model = optimizer = batch = groups = scaler = inputs = None
        try:
            model,_ = models.build_model(model_id, plan, catalog, paths, seed=SEED)
            model.to("cuda:0")
            batch = synthetic_batch(model_id, catalog)[:1].repeat(micro,1,1,1).to("cuda:0")
            torch.cuda.reset_peak_memory_stats()
            models.set_phase(model,"full")
            groups = models.optimizer_groups(model,plan,"full",stage="stage_2")
            optimizer = engine._new_optimizer(groups,plan)
            scaler = torch.amp.GradScaler("cuda", enabled=True)
            phase_counts = {}
            for phase in ("head_only","last_block","full"):
                model.train()
                models.set_phase(model,phase)
                phase_counts[phase] = sum(p.numel() for p in model.parameters() if p.requires_grad)
                require(phase_counts[phase] > 0 and all(not m.training for m in model.encoder.modules()
                    if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)), "mapped_phase_and_frozen_batchnorm_required")
                optimizer.zero_grad(set_to_none=True)
                inputs = [{"image":batch,"target":torch.arange(micro,device="cuda:0").remainder(2).float()}]
                engine._execute_training_epoch(model,inputs,optimizer,scaler,torch.device("cuda:0"),1.0,micro,None)
            require(phase_counts["head_only"] < phase_counts["last_block"] < phase_counts["full"],
                "strictly_progressive_registered_parameter_mapping_required")
            torch.cuda.synchronize()
            value = {"model_id":model_id,"micro_batch_size":micro,"effective_batch_size":16,
                "architecture_mapping_verified":True,"full_backward_and_optimizer_step_verified":True,
                "phase_trainable_parameters":phase_counts,"synthetic_inputs_only":True,
                "peak_memory_allocated_bytes":torch.cuda.max_memory_allocated(),
                "device_name":torch.cuda.get_device_name(),"checkpoint_files_sha256":{
                    item["name"]:item["sha256"] for item in catalog["models"][model_id]["files"]}}
            runner.save_json(output / (model_id + "__gpu_probe.json"),value)
            return value
        except torch.OutOfMemoryError:
            if micro == attempts[-1]:
                raise RuntimeError("La GPU no tiene memoria para este modelo con el lote mínimo registrado; conserva el avance y usa una GPU con más memoria.") from None
            print(f'{model_id}: memoria insuficiente con microlote {micro}; comprobando el siguiente tamaño permitido.',file=sys.stderr,flush=True)
        finally:
            model = optimizer = batch = groups = scaler = inputs = None
            gc.collect()
            torch.cuda.empty_cache()
    raise ValueError("verified_gradual_gpu_probe_required")


def run(repository_root, preparation_dir, package_dir, reference_path, output_dir, *, weights_root,
        persistent_root, code_revision, cpu_tests=None, runtime=None, device="cuda:0", allow_download=True):
    repository,proof,package,reference,output,weights,records = map(checked_path,
        (repository_root,preparation_dir,package_dir,reference_path,output_dir,weights_root,persistent_root))
    require(device == "cuda:0" and type(allow_download) is bool and not output.exists() and output.parent.is_dir()
        and records.is_dir() and weights.is_dir(), "gradual_development_requires_cuda_and_new_local_output")
    _separate(output,(repository,proof,package,reference.parent,records,weights))
    _separate(weights,(repository,proof,package,reference.parent,records))
    _,catalog,plan,registry = release.load_release(repository)
    prepared = read_json(proof / PREPARATION_FILENAME)
    evidence = checked_path(proof / SELECTION_FILENAME)
    sources = _sources(repository)
    require(prepared.get("version") == VERSION and prepared.get("status") == "gradual_campaign_prepared_no_training"
        and prepared.get("code_revision") == code_revision and prepared.get("training_executed") is False
        and prepared.get("record_sha256") == digest({k:v for k,v in prepared.items() if k != "record_sha256"})
        and prepared.get("source_files_sha256") == sources and prepared.get("model_order") == list(MODELS)
        and prepared.get("job_count") == 20 and prepared.get("selection_evidence_sha256") == sha_file(evidence)
        and prepared.get("model_input_reference_sha256") == sha_file(reference),
        "unchanged_authenticated_gradual_preparation_required")
    from knee.aligned_training.data import load_sealed_inputs
    loaded = load_sealed_inputs(package,reference,repository,prepared["development_package_sha256"])
    rows = loaded["manifest"]["rows"]
    del loaded
    from knee.gradual_selection import validate_selection_evidence
    initial = make_request(prepared["development_package_sha256"],sha_file(evidence),MODELS[0],0,1,plan,registry)
    selection = validate_selection_evidence(evidence,initial,plan,registry,rows)
    identity = {"code_revision":code_revision,"stage":"stage_2","seed":SEED,"model_order":list(MODELS),
        "training_plan_sha256":release.PINS[release.PLAN_PATH],"modeling_release_sha256":release.PINS[release.RELEASE_PATH],
        "development_package_sha256":prepared["development_package_sha256"],"model_input_reference_sha256":sha_file(reference),
        "selection_evidence_sha256":sha_file(evidence),"source_files_sha256":sources}
    campaign = _open_campaign(records,identity)
    completed = _completed(campaign,prepared["development_package_sha256"],sha_file(evidence),rows,plan,registry)
    preserved = set(completed)
    output.mkdir()
    print(f'Recuperación: {len(preserved)}/20 trabajos completos conservados; {20-len(preserved)} pendientes.',file=sys.stderr,flush=True)
    probed = []
    resumed = 0
    for model_id in MODELS:
        pending = [fold for fold in FOLDS if f'clinical_main__{model_id}__fold_{fold}' not in completed]
        if not pending:
            continue
        from knee.training_smoke import verify_gpu_environment
        verify_gpu_environment("google_colab_gpu")
        print(f'{model_id}: comprobando las tres fases y memoria de GPU con entradas sintéticas.',file=sys.stderr,flush=True)
        micro_path = campaign / (model_id + "__batch.json")
        fixed = read_json(micro_path)["micro_batch_size"] if micro_path.is_file() else None
        probe = _probe(model_id,catalog,plan,weights,output,fixed_micro=fixed,allow_download=allow_download)
        micro = probe["micro_batch_size"]
        if fixed is None:
            transport._publish_json(micro_path,{"version":VERSION,"model_id":model_id,"micro_batch_size":micro})
        else:
            require(fixed == micro,"resume_microbatch_must_not_change")
        probed.append(model_id)
        for fold in pending:
            request = make_request(prepared["development_package_sha256"],sha_file(evidence),model_id,fold,micro,plan,registry)
            key = prior.job_key(request)
            require(_sources(repository) == sources,"gradual_scientific_sources_changed")
            print(f'Trabajo {len(completed)+1}/20: {key}.',file=sys.stderr,flush=True)
            from knee.gradual_epoch_resume import GradualEpochCheckpoint
            checkpoint = GradualEpochCheckpoint(campaign / "epoch_checkpoints" / key,request)
            authority = _authority(request,plan,registry,probe)
            local = output / key
            job = runner.execute(request,authority,repository_root=repository,package_dir=package,reference_path=reference,
                weights_root=weights,output_dir=local,device=device,allow_download=False,selection_evidence=evidence,
                epoch_checkpoint=checkpoint)
            # Validate new full tensor files once locally; publication streams
            # them once to Drive and verifies the destination by SHA-256.
            require(runner.completed_job(local,request["development_package_sha256"],plan,registry) == job,
                "new_gradual_job_must_be_fully_verified_before_publication")
            receipt = {"version":VERSION,"key":key,"job_record_sha256":job["record_sha256"],
                "files_sha256":{**job["files_sha256"],runner.JOB_FILENAME:sha_file(local / runner.JOB_FILENAME)},
                "checkpoint_sizes":{name:(local / name).stat().st_size for name in TENSOR_FILES}}
            item = read_completed_job(local,receipt,request,rows,plan,registry)
            target = campaign / ("job_attempt_"+uuid.uuid4().hex)
            target.mkdir()
            destination = target / "job"
            destination.mkdir()
            from knee.image_campaign_recovery import _copy_once
            for name,checksum in sorted(receipt["files_sha256"].items()):
                _copy_once(local / name,destination / name,checksum)
            transport._publish_json(target / transport.JOB_RECEIPT,receipt)
            item["directory"] = destination
            completed[key] = item
            resumed += int(item["engine"].get("resumed_from_epoch",0)>0)
            print(f'{key}: completo, guardado en Drive y verificado.',file=sys.stderr,flush=True)
    require(len(completed) == 20,"all_twenty_gradual_development_jobs_required")
    from knee.aligned_training.engine import aggregate_fold_metrics,binary_metrics
    models, oof_hashes = {}, {}
    for model_id in MODELS:
        group = [completed[f'clinical_main__{model_id}__fold_{fold}'] for fold in FOLDS]
        fold_metrics = [{"validation_fold":fold,**item["record"]["metrics"]} for fold,item in zip(FOLDS,group)]
        oof = []
        for fold,item in zip(FOLDS,group):
            p=item["predictions"]
            oof.extend({"src_subject_id":i[0],"side":i[1],"validation_fold":fold,"target":y,"logit":z}
                for i,y,z in zip(p["identities"],p["targets"],p["logits"]))
        require(len(oof)==len(rows) and {(r["src_subject_id"],r["side"]) for r in oof} ==
            {(r["src_subject_id"],r["side"]) for r in rows},"exact_gradual_oof_coverage_required")
        name = "clinical_main__"+model_id
        models[name] = {"fivefold_selection_metrics":aggregate_fold_metrics(fold_metrics),
            "pooled_oof_metrics_descriptive_only":binary_metrics([r["target"] for r in oof],[r["logit"] for r in oof]),
            "out_of_fold_predictions":len(oof)}
        filename = name + "__oof.json"
        runner.save_json(output / filename,{"scope":"development_only","model_id":model_id,"stage":"stage_2", "rows":oof})
        oof_hashes[filename] = sha_file(output / filename)
    result = {"version":VERSION,"status":STATUS,"campaign":CAMPAIGN,"scope":"development_only","stage":"stage_2",
        "seed":SEED,"folds_completed":list(FOLDS),"model_order":list(MODELS),"job_count":20,
        "code_revision":code_revision,"model_results":models,"development_knees":len(rows),
        "development_participants":prepared["development_participants"],"development_package_sha256":prepared["development_package_sha256"],
        "training_plan_sha256":release.PINS[release.PLAN_PATH],"modeling_release_sha256":release.PINS[release.RELEASE_PATH],
        "model_input_reference_sha256":sha_file(reference),"selection_evidence_sha256":sha_file(evidence),
        "source_files_sha256":sources,"persistent_campaign_folder":campaign.name,
        "completed_jobs_preserved":len(preserved),"jobs_executed_this_invocation":20-len(preserved),
        "jobs_resumed_from_complete_epoch":resumed,"within_job_epoch_resume_used":resumed>0,
        "training_executed_this_invocation":len(preserved)<20,"training_executed_in_campaign":True,
        "current_gpu_probed_models":probed,"completed_checkpoint_payloads_copied_on_recovery":False,
        "checkpoint_payloads_rehashed_on_recovery":False,"checkpoint_rehash_required_before_weight_reuse":True,
        "new_job_payloads_verified_on_publication":True,"full_encoder_training_authorized":True,"encoder_unfreezing_executed":True,
        "reserved_test_accessed":False,"reserved_test_predictive_evaluation_opened":False,"originals_modified":False,
        "partitions_modified":False,"fusion_training_executed":False,"calibration_performed":False,"threshold_selected":False,
        "final_refit_executed":False,"unbiased_final_performance_claimed":False,"oof_files_sha256":oof_hashes,
        "job_results":[{"key":key,"job_record_sha256":item["record"]["record_sha256"],"metrics":item["record"]["metrics"],
            "preserved_complete_job":key in preserved} for key,item in sorted(completed.items())],
        "private_job_locations":{key:item["directory"].relative_to(campaign).as_posix() for key,item in completed.items()},
        "artifact_files_sha256":{key+"/"+name:value for key,item in completed.items() for name,value in item["receipt"]["files_sha256"].items()},
        "result_files_saved_and_readback_verified":True,"executed_at_utc":datetime.now(timezone.utc).isoformat()}
    require(_sources(repository)==sources,"gradual_sources_changed_before_completion")
    runner.save_json(output / FILENAME,result)
    closing=campaign / ("completion_"+uuid.uuid4().hex)
    closing.mkdir()
    transport._publish_json(closing / FILENAME,result)
    for filename,checksum in oof_hashes.items():
        from knee.image_campaign_recovery import _copy_once
        _copy_once(output / filename,closing / filename,checksum)
    return result
