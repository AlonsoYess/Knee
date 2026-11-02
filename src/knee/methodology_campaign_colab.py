"""Durable Colab orchestration for bounded methodology v1.2 campaigns."""
from __future__ import annotations

from contextlib import nullcontext, redirect_stdout
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
import shutil
import sys
import uuid

from knee import methodology_campaign as core
from knee import methodology_inputs_colab as inputs
from knee import methodology_release as release
from knee.colab_data import checked_path, require, sha_file, valid_sha
from knee.training_preparation import read_json
from knee.training_preparation_colab import _save

VERSION = "methodology_campaign_colab_orchestration_v1.0"
SUMMARY_FILENAME = "colab_methodology_campaign_check.json"
CAMPAIGN_FILENAME = "campaign_identity.json"
JOB_RECEIPT = "job_transport_receipt.json"
IMAGE_SMOKE_RECEIPT = "image_smoke_transport_receipt.json"
IMAGE_SMOKE_PREFIX = "image_smoke_attempt_"
CAMPAIGN_PREFIX = "methodology_v1_2_campaign_"
APPROVAL_STATEMENTS = core.APPROVAL_STATEMENTS
CONFIRMATIONS = core.CONFIRMATIONS
TEST_MODULES = ("test_clinical_scenarios", "test_training_augmentation", "test_methodology_release",
    "test_methodology_portable_release", "test_methodology_clinical_integration",
    "test_methodology_augmentation_integration", "test_methodology_campaign", "test_methodology_campaign_colab")
IMAGE_TEST_MODULES = TEST_MODULES + ("test_methodology_image_campaign",
    "test_methodology_image_campaign_colab", "test_image_screening_notebook")
SOURCE_FILES = ("src/knee/methodology_campaign.py", "src/knee/methodology_campaign_colab.py",
    "scripts/train_methodology_campaign_colab.py", "src/knee/methodology_release.py",
    "src/knee/clinical_scenarios.py", "src/knee/training_augmentation.py",
    "src/knee/aligned_training/authority.py", "src/knee/aligned_training/runner.py",
    "src/knee/aligned_training/data.py", "src/knee/aligned_training/models.py",
    "src/knee/aligned_training/engine.py")
IMAGE_SOURCE_FILES = SOURCE_FILES + ("scripts/train_image_screening_colab.py",
    "scripts/build_image_screening_notebook.py")
MAX_DIRECT_CHILDREN = 2048
MAX_PREVIOUS_RECORDS = 128
MAX_ATTEMPTS = 512
MAX_SINGLE_FILE_BYTES = 64 * 1024**2
MAX_JOB_BYTES = 256 * 1024**2
# Only the two tensor checkpoints of these registered frozen encoders may use
# larger limits. JSON records and clinical/DenseNet transports keep their caps.
IMAGE_CHECKPOINT_LIMITS = {
    "image_vit_b16": 384 * 1024**2,
    "image_convnextv2_tiny": 160 * 1024**2,
    "image_skelex": 2048 * 1024**2,
    "image_biomedclip": 384 * 1024**2,
    "image_mambaout_tiny": 160 * 1024**2,
}
stage_development = inputs.stage_development
_json = read_json


def runtime_environment(repository, campaign="clinical"):
    require(campaign in core.CAMPAIGNS, "registered_methodology_campaign_required")
    if campaign == "clinical":
        return inputs.runtime_environment(repository)
    from knee.training_gpu_runtime import runtime_environment as gpu_environment
    return gpu_environment(repository)


def setup_runtime(repository_root, destination, campaign="clinical"):
    require(campaign in core.CAMPAIGNS, "registered_methodology_campaign_required")
    if campaign == "clinical":
        return inputs.setup_runtime(repository_root, destination)
    from knee.training_gpu_runtime import setup_runtime as gpu_setup
    return gpu_setup(repository_root, destination)


def test_modules(campaign):
    require(campaign in core.CAMPAIGNS, "registered_methodology_campaign_required")
    return IMAGE_TEST_MODULES if campaign == "remaining_encoders_frozen" else TEST_MODULES


def run_tests(repository_root, campaign="clinical"):
    import torch
    import unittest
    repository = checked_path(repository_root)
    require(checked_path(__file__).resolve() == repository.resolve() / "src/knee/methodology_campaign_colab.py",
        "campaign_tests_must_use_selected_checkout")
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", "campaign_synthetic_tests_must_hide_gpu")
    require(not torch.cuda.is_available(), "campaign_synthetic_tests_must_compute_on_cpu")
    test_runtime = ("gpu_build_cpu_compute_no_training" if torch.version.cuda is not None
        else "synthetic_cpu_no_participant_training")
    sys.path.insert(0, str(repository / "tests"))
    with redirect_stdout(sys.stderr):
        loader = unittest.TestLoader()
        scope = test_modules(campaign)
        suite = loader.loadTestsFromNames(scope)
        require(not loader.errors and suite.countTestCases() > 0, "complete_methodology_campaign_tests_required")
        count = suite.countTestCases()
        result = unittest.TextTestRunner(verbosity=1, stream=sys.stderr).run(suite)
    require(result.wasSuccessful() and result.testsRun == count and not result.skipped
        and not result.expectedFailures and not result.unexpectedSuccesses, "methodology_campaign_tests_not_complete")
    return {"passed": True, "tests_run": count, "skipped": 0, "scope": list(scope),
        "test_runtime": test_runtime}


def find_preparation(persistent_root):
    root = checked_path(persistent_root)
    require(root.is_dir(), "existing_private_records_folder_required")
    candidates, seen = [], 0
    with os.scandir(root) as entries:
        for count, entry in enumerate(entries, 1):
            require(count <= MAX_DIRECT_CHILDREN, "bounded_private_records_folder_required")
            if not entry.name.startswith("methodology_inputs38_"):
                continue
            path = checked_path(root / entry.name)
            if not path.is_dir() or not (path / inputs.SUMMARY_FILENAME).is_file():
                continue
            seen += 1
            require(seen <= MAX_PREVIOUS_RECORDS, "bounded_methodology38_preflights_required")
            summary, _, _ = core.validate_preparation(path)
            stamp = datetime.fromisoformat(summary["executed_at_utc"])
            require(stamp.utcoffset() is not None, "timezone_aware_methodology38_timestamp_required")
            candidates.append((stamp, path))
    require(candidates, "completed_real_colab38_preflight_not_found")
    newest = max(stamp for stamp, _ in candidates)
    matches = [path for stamp, path in candidates if stamp == newest]
    require(len(matches) == 1, "latest_completed_colab38_preflight_is_ambiguous")
    return matches[0]


def _publish_json(target, value):
    target = checked_path(target)
    stage = target.parent / (".publication_" + uuid.uuid4().hex + ".part")
    _save(stage, value)
    if target.exists():
        require(target.is_file() and sha_file(target) == sha_file(stage), "existing_methodology_record_cannot_be_overwritten")
        stage.unlink()
        return
    stage.rename(target)
    require(_json(target) == value, "methodology_record_readback_failed")


def _publish_copy(source, target, checksum):
    source, target = map(checked_path, (source, target))
    require(sha_file(source) == checksum, "local_methodology_publication_source_changed")
    if target.exists():
        require(target.is_file() and sha_file(target) == checksum, "existing_methodology_artifact_cannot_be_overwritten")
        return
    stage = target.parent / (".publication_" + uuid.uuid4().hex + ".part")
    with source.open("rb") as incoming, stage.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=1024**2)
    require(sha_file(stage) == sha_file(source) == checksum, "staged_methodology_artifact_changed")
    stage.rename(target)
    require(sha_file(target) == checksum, "durable_methodology_artifact_readback_failed")


def _inventory(directory):
    root = checked_path(directory)
    paths = list(root.iterdir())
    record_path = checked_path(root / "training_job.json")
    require(record_path.is_file() and 0 < record_path.stat().st_size <= 1024**2,
        "bounded_methodology_job_record_required")
    job = _json(record_path)
    request = job.get("request", {})
    model_id = request.get("model_id")
    large_checkpoint = model_id in IMAGE_CHECKPOINT_LIMITS
    if large_checkpoint:
        require(request.get("version") == release.REQUEST_VERSION and request.get("stage") == "stage_1"
            and request.get("clinical_scenario") == "clinical_main"
            and request.get("image_encoder_id") == model_id,
            "registered_frozen_image_checkpoint_transport_required")
    checkpoint_limit = IMAGE_CHECKPOINT_LIMITS[model_id] if large_checkpoint else MAX_SINGLE_FILE_BYTES
    total_limit = 2 * checkpoint_limit + MAX_SINGLE_FILE_BYTES if large_checkpoint else MAX_JOB_BYTES
    def file_limit(path):
        return checkpoint_limit if path.name in {"best_model.safetensors", "last_model.safetensors"} else MAX_SINGLE_FILE_BYTES
    require(1 <= len(paths) <= 7 and all(checked_path(path).is_file() and not path.is_symlink()
        and 0 < path.stat().st_size <= file_limit(path) for path in paths)
        and sum(path.stat().st_size for path in paths) <= total_limit,
        "bounded_plain_methodology_job_inventory_required")
    expected = {**job.get("files_sha256", {}), "training_job.json": sha_file(root / "training_job.json")}
    require({p.name for p in paths} == set(expected)
        and all(sha_file(root / name) == checksum for name, checksum in expected.items()),
        "methodology_job_transport_inventory_mismatch")
    return expected


def _copy_job(source, destination, expected):
    source, target = map(checked_path, (source, destination))
    require(not target.exists() and target.parent.is_dir() and _inventory(source) == expected,
        "new_verified_methodology_job_transport_required")
    target.mkdir()
    for name, checksum in sorted(expected.items()):
        require(Path(name).name == name and name not in {".", ".."}, "plain_methodology_job_filenames_required")
        with (source / name).open("rb") as incoming, (target / name).open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=1024**2)
        require(sha_file(target / name) == checksum, "methodology_job_copy_readback_failed")
    require(_inventory(source) == _inventory(target) == expected, "methodology_job_transport_changed")


def campaign_identity(repository, proof, reference, revision, campaign):
    require(campaign in core.CAMPAIGNS and isinstance(revision, str)
        and re.fullmatch(r"[0-9a-f]{40}", revision) is not None and revision != "0" * 40,
        "published_methodology_campaign_revision_required")
    _, record, sidecar = core.validate_preparation(proof, repository)
    return {"code_revision": revision, "campaign": campaign, "seed": core.SEED, "stage": "stage_1",
        "training_plan_sha256": release.PINS[release.PLAN_PATH],
        "modeling_release_sha256": release.PINS[release.RELEASE_PATH],
        "preparation38_summary_sha256": sha_file(proof / inputs.SUMMARY_FILENAME),
        "model_input_reference_sha256": sha_file(reference),
        "development_package_sha256": record["development_package_sha256"],
        "baseline_sidecar_sha256": sidecar["record_sha256"],
        "source_files_sha256": {name: sha_file(repository / name) for name in
            (IMAGE_SOURCE_FILES if campaign == "remaining_encoders_frozen" else SOURCE_FILES)}}


def open_campaign(persistent_root, identity):
    root = checked_path(persistent_root)
    require(root.is_dir(), "existing_private_campaign_records_required")
    matches, scanned = [], 0
    with os.scandir(root) as entries:
        for count, entry in enumerate(entries, 1):
            require(count <= MAX_DIRECT_CHILDREN, "bounded_private_campaign_inventory_required")
            if not entry.name.startswith(CAMPAIGN_PREFIX):
                continue
            path = checked_path(root / entry.name)
            if not path.is_dir() or not (path / CAMPAIGN_FILENAME).is_file():
                continue
            scanned += 1
            require(scanned <= MAX_PREVIOUS_RECORDS, "bounded_previous_methodology_campaigns_required")
            manifest = _json(path / CAMPAIGN_FILENAME)
            require(manifest.get("version") == VERSION, "registered_methodology_campaign_manifest_required")
            if manifest.get("identity") == identity:
                matches.append(path)
    require(len(matches) <= 1, "matching_methodology_campaigns_are_ambiguous")
    if matches:
        return matches[0]
    target = root / (CAMPAIGN_PREFIX + identity["campaign"] + "_" + uuid.uuid4().hex)
    target.mkdir()
    _publish_json(target / CAMPAIGN_FILENAME, {"version": VERSION, "identity": identity,
        "created_at_utc": datetime.now(timezone.utc).isoformat()})
    return target


def completed_job_transports(campaign):
    root = checked_path(campaign)
    completed, attempts = {}, 0
    for count, path in enumerate(root.iterdir(), 1):
        require(count <= MAX_ATTEMPTS + 64, "bounded_methodology_campaign_attempts_required")
        if path.name.startswith(".publication_") and path.name.endswith(".part"):
            require(checked_path(path).is_file() and path.stat().st_size <= MAX_SINGLE_FILE_BYTES,
                "bounded_interrupted_methodology_publication_required")
            continue
        if path.name.startswith(IMAGE_SMOKE_PREFIX):
            require(checked_path(path).is_dir(), "bounded_image_smoke_attempt_directory_required")
            continue
        if not path.name.startswith("job_attempt_"):
            require((path.name in {CAMPAIGN_FILENAME, core.FILENAME, SUMMARY_FILENAME}
                or path.name in {f"clinical_main__{model}__oof.json" for model in core.REMAINING_IMAGE_IDS}
                or re.fullmatch(r"clinical_[a-z_]+__(?:clinical_[a-z_]+|image_densenet121)__oof.json", path.name))
                and checked_path(path).is_file(),
                "only_registered_methodology_campaign_artifacts_allowed")
            continue
        attempts += 1
        require(attempts <= MAX_ATTEMPTS and checked_path(path).is_dir(), "bounded_methodology_attempt_directory_required")
        if not (path / JOB_RECEIPT).is_file():
            continue
        receipt = _json(path / JOB_RECEIPT)
        require(set(receipt) == {"version", "key", "files_sha256", "job_record_sha256"}
            and receipt["version"] == VERSION and isinstance(receipt["key"], str)
            and receipt["key"] not in completed and {p.name for p in path.iterdir()} == {"job", JOB_RECEIPT},
            "unique_complete_methodology_job_receipt_required")
        job = _json(path / "job/training_job.json")
        require(_inventory(path / "job") == receipt["files_sha256"]
            and job.get("record_sha256") == receipt["job_record_sha256"]
            and job.get("request", {}).get("version") == release.REQUEST_VERSION
            and job.get("plan_sha256") == release.PINS[release.PLAN_PATH]
            and core.job_key(job["request"]) == receipt["key"],
            "only_exact_v1_2_jobs_may_be_recovered")
        completed[receipt["key"]] = {"directory": path / "job", "receipt": receipt}
    return completed


def persist_completed_job(campaign, key, directory, record):
    root, source = map(checked_path, (campaign, directory))
    require(core.job_key(record["request"]) == key and _json(source / "training_job.json") == record,
        "methodology_record_must_match_completed_job")
    inventory = _inventory(source)
    completed = completed_job_transports(root)
    if key in completed:
        old = completed[key]
        require(old["receipt"]["files_sha256"] == inventory
            and old["receipt"]["job_record_sha256"] == record["record_sha256"],
            "completed_methodology_job_cannot_be_overwritten")
        return old["directory"]
    target = root / ("job_attempt_" + uuid.uuid4().hex)
    target.mkdir()
    _copy_job(source, target / "job", inventory)
    _publish_json(target / JOB_RECEIPT, {"version": VERSION, "key": key,
        "files_sha256": inventory, "job_record_sha256": record["record_sha256"]})
    require(key in completed_job_transports(root), "durable_methodology_job_readback_required")
    print(f'{key}: guardado en Drive y verificado.', file=sys.stderr, flush=True)
    return target / "job"


def _historical_source_matches(repository, name, expected):
    """Only exact registered35 sources may adopt their LF/CRLF byte spelling."""
    from knee.training_smoke_evidence import SOURCE_FILES as allowed
    require(name in allowed and allowed[name] == expected, "registered_historical_gpu_source_pin_required")
    path = checked_path(repository / name)
    if not path.is_file() or not 0 < path.stat().st_size <= 1024**2:
        return False
    raw = path.read_bytes()
    try:
        raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return False
    without_crlf = raw.replace(b"\r\n", b"")
    if b"\r" in without_crlf:
        return False
    if hashlib.sha256(raw).hexdigest() == expected:
        return True
    if b"\r\n" in raw and b"\n" in without_crlf:
        return False
    lf = raw.replace(b"\r\n", b"\n")
    return any(hashlib.sha256(variant).hexdigest() == expected for variant in (lf, lf.replace(b"\n", b"\r\n")))


def _gpu_evidence(records_root, repository):
    """Adopt the original bounded35 proof with only pinned LF/CRLF portability."""
    from knee import training_smoke_evidence as evidence
    from knee import training_smoke_colab as old_colab
    from knee import training_smoke as old_core
    candidates = []
    for count, path in enumerate(checked_path(records_root).iterdir(), 1):
        require(count <= MAX_DIRECT_CHILDREN, "bounded_gpu_evidence_inventory_required")
        if not path.name.startswith("training_smoke_check_") or not path.is_dir():
            continue
        if not (path / old_colab.SUMMARY_FILENAME).is_file():
            continue
        require(len(candidates) < MAX_PREVIOUS_RECORDS, "bounded_previous_gpu_smoke_records_required")
        source = _json(path / old_colab.CORE_FILENAME)
        summary = _json(path / old_colab.SUMMARY_FILENAME)
        request = evidence._validate_scope(source)
        old_colab.validate_result(summary)
        require(summary.get("code_revision") == evidence.SMOKE_REVISION
            and summary.get("smoke_record_sha256") == sha_file(path / old_colab.CORE_FILENAME)
            and summary.get("result_files_saved_and_readback_verified") is True
            and all(summary.get(key) == value for key, value in source.items() if key != "executed_at_utc")
            and summary.get("cpu_tests", {}).get("passed") is True
            and summary["cpu_tests"].get("tests_run") == evidence.SMOKE_TEST_COUNT
            and summary["cpu_tests"].get("skipped") == 0
            and summary["cpu_tests"].get("scope") == list(old_colab.TEST_MODULES),
            "original_completed_gpu35_summary_required")
        job_root = evidence._exact_inventory(path, source)
        evidence._validate_job_links(source, job_root, request)
        _, catalog, _, registry = release.load_release(repository)
        specification = catalog["models"]["image_densenet121"]
        require(source.get("eligibility_gates_before_smoke") == {key: index < 4
                for index, key in enumerate(registry["eligibility_gates"])}
            and source.get("public_checkpoint_files_sha256") == {item["name"]: item["sha256"] for item in specification["files"]}
            and source.get("eligibility_evidence") == {"model_id": "image_densenet121",
                "scope": "private_noncommercial_academic_research",
                "catalog_sha256": evidence.PINS["configs/pretrained_weights_v1_1.json"],
                "license_review": specification["license_review"], "overlap_review": specification["overlap_review"],
                "pretraining_subject_disjointness_proven": False,
                "preprocessing_localizer_oai_overlap_limitation_remains": True,
                "normalization_and_loader_evidence": "authenticated_colab32_reference_and_colab34_frozen_sources"},
            "same_reviewed_densenet_weights_and_academic_use_required")
        require(all(_historical_source_matches(repository, name, checksum)
            for name, checksum in evidence.SOURCE_FILES.items()), "original_gpu_runtime_sources_changed")
        stamp = datetime.fromisoformat(summary["executed_at_utc"])
        require(stamp.utcoffset() is not None, "timezone_aware_gpu_smoke_evidence_required")
        candidates.append((stamp, {"image_eligibility_verified": True,
            "development_package_sha256": source["development_package_sha256"],
            "model_input_reference_sha256": source["model_input_reference_sha256"],
            "historical_smoke_summary_sha256": sha_file(path / old_colab.SUMMARY_FILENAME),
            "historical_smoke_scope": "same_encoder_runtime_only_not_v1_2_augmentation_efficacy"}))
    require(candidates, "completed_actual_densenet_gpu35_smoke_required")
    latest = max(stamp for stamp, _ in candidates)
    matches = [record for stamp, record in candidates if stamp == latest]
    require(len(matches) == 1, "latest_gpu35_smoke_is_ambiguous")
    return matches[0]


def find_image_preflight(records_root, repository):
    """Authenticate completed33 and the exact five public checkpoint records."""
    from knee.training_preparation import validate_preflight
    from knee.training_preparation_colab import find_previous_preflight
    proof = find_previous_preflight(records_root)
    validate_preflight(proof)
    checkpoint = _json(proof / "checkpoint_verification.json")
    _, catalog, _, _ = release.load_release(repository)
    for model_id in core.REMAINING_IMAGE_IDS:
        specification = catalog["models"][model_id]
        record = checkpoint["model_records"][model_id]
        require(specification["verification_enabled"] is True
            and specification["license_review"]["use_scope"] in {
                "private_noncommercial_academic_research", "private_noncommercial_academic_research_with_attribution",
                "private_noncommercial_academic_verification", "academic_verification_and_comparison_no_clinical_validation_claim"}
            and isinstance(specification["overlap_review"], str) and specification["overlap_review"]
            and record.get("checkpoint_loaded") is True
            and record.get("feature_shape") == [2, specification["feature_dimension"]]
            and record.get("files") == [{key: item[key] for key in ("name", "bytes", "sha256")}
                for item in specification["files"]],
            "exact_five_registered_colab33_checkpoint_records_required")
    return proof


def _eval_image_probe(model, batch, feature_dimension, *, autocast_context=None):
    """Eval-only probe shared by the GPU gate and tiny CPU unit fixtures."""
    import torch
    from knee.pretrained_verification import state_digest
    require(not model.training and all(not parameter.requires_grad for parameter in model.encoder.parameters()),
        "frozen_eval_encoder_required_before_synthetic_probe")
    before, source = state_digest(model), batch.clone()
    with torch.inference_mode(), (nullcontext() if autocast_context is None else autocast_context):
        features = model.encoder(batch)
        logits = model(images=batch)
    require(tuple(features.shape) == (2, feature_dimension) and tuple(logits.shape) == (2,)
        and features.device == batch.device and logits.device == batch.device
        and bool(torch.isfinite(features).all()) and bool(torch.isfinite(logits).all()),
        "finite_registered_synthetic_image_forward_required")
    require(state_digest(model) == before and torch.equal(batch, source)
        and all(parameter.grad is None for parameter in model.parameters()),
        "synthetic_image_probe_must_not_change_weights_inputs_or_gradients")
    return {"synthetic_gpu_output_shape": list(features.shape),
        "synthetic_gpu_logits_shape": list(logits.shape), "parameters_and_buffers_unchanged": True,
        "synthetic_input_unchanged": True, "optimizer_created": False, "backward_executed": False,
        "gradients_created": False}


def _probe_image_gpu(model_id, catalog, plan, paths):
    import gc
    import torch
    from knee.aligned_training.models import build_model
    from knee.final_models_preflight import synthetic_batch
    model, _ = build_model(model_id, plan, catalog, paths, seed=core.SEED)
    batch = None
    try:
        model = model.to("cuda:0").eval()
        batch = synthetic_batch(model_id, catalog).to("cuda:0")
        require(batch.device.type == "cuda" and all(p.device.type == "cuda" for p in model.parameters()),
            "actual_cuda_synthetic_image_probe_required_no_cpu_fallback")
        torch.cuda.synchronize()
        result = _eval_image_probe(model, batch, catalog["models"][model_id]["feature_dimension"])
        mixed = _eval_image_probe(model, batch, catalog["models"][model_id]["feature_dimension"],
            autocast_context=torch.autocast("cuda", dtype=torch.float16))
        torch.cuda.synchronize()
        return {**result, "synthetic_gpu_forward_verified": True, "precision_fp16_verified": True,
            "verified_precision_modes": ["float32", "cuda_autocast_float16"],
            "synthetic_gpu_fp16_output_shape": mixed["synthetic_gpu_output_shape"],
            "synthetic_gpu_fp16_logits_shape": mixed["synthetic_gpu_logits_shape"]}
    finally:
        del model, batch
        gc.collect()
        torch.cuda.empty_cache()


def _stored_image_smokes(campaign_dir):
    """Recover small authenticated smoke records; never open smoke checkpoints."""
    root = checked_path(campaign_dir)
    identity_sha = sha_file(root / CAMPAIGN_FILENAME)
    _, _, _, registry = release.load_release(Path(__file__).resolve().parents[2])
    completed = {}
    for count, path in enumerate(root.iterdir(), 1):
        require(count <= MAX_ATTEMPTS + 64, "bounded_image_smoke_inventory_required")
        if not path.name.startswith(IMAGE_SMOKE_PREFIX):
            continue
        require(checked_path(path).is_dir(), "plain_image_smoke_attempt_required")
        receipt_path = path / IMAGE_SMOKE_RECEIPT
        if not receipt_path.is_file():
            continue
        receipt = _json(receipt_path)
        files = receipt.get("files_sha256", {})
        model_id = receipt.get("model_id")
        require(set(receipt) == {"version", "model_id", "campaign_identity_sha256", "files_sha256"}
            and receipt["version"] == VERSION and model_id in core.REMAINING_IMAGE_IDS
            and model_id not in completed and receipt["campaign_identity_sha256"] == identity_sha
            and set(files) == {"encoder_smoke.json", "training_record.json", "training_job.json"}
            and {p.name for p in path.iterdir()} == set(files) | {IMAGE_SMOKE_RECEIPT}
            and all(checked_path(path / name).is_file() and 0 < (path / name).stat().st_size <= MAX_SINGLE_FILE_BYTES
                and valid_sha(checksum) and sha_file(path / name) == checksum for name, checksum in files.items()),
            "exact_completed_current_image_smoke_transport_required")
        smoke = _json(path / "encoder_smoke.json")
        training = _json(path / "training_record.json")
        job = _json(path / "training_job.json")
        core.validate_remaining_smoke_record(model_id, smoke, training, job, registry)
        require(smoke["smoke_training_record_sha256"] == files["training_record.json"],
            "persisted_image_smoke_training_record_hash_mismatch")
        completed[model_id] = smoke
    return completed


def _persist_image_smoke(campaign_dir, model_id, smoke_root):
    root, source = map(checked_path, (campaign_dir, smoke_root))
    completed = _stored_image_smokes(root)
    if model_id in completed:
        return completed[model_id]
    sources = {"encoder_smoke.json": source / "encoder_smoke.json",
        "training_record.json": source / "job/training_record.json",
        "training_job.json": source / "job/training_job.json"}
    smoke, training, job = (_json(sources[name]) for name in
        ("encoder_smoke.json", "training_record.json", "training_job.json"))
    _, _, _, registry = release.load_release(Path(__file__).resolve().parents[2])
    core.validate_remaining_smoke_record(model_id, smoke, training, job, registry)
    require(model_id in core.REMAINING_IMAGE_IDS and all(checked_path(path).is_file()
        and 0 < path.stat().st_size <= MAX_SINGLE_FILE_BYTES for path in sources.values()),
        "bounded_registered_image_smoke_records_required")
    target = root / (IMAGE_SMOKE_PREFIX + uuid.uuid4().hex)
    target.mkdir()
    checksums = {name: sha_file(path) for name, path in sources.items()}
    for name, path in sources.items():
        _publish_copy(path, target / name, checksums[name])
    _publish_json(target / IMAGE_SMOKE_RECEIPT, {"version": VERSION, "model_id": model_id,
        "campaign_identity_sha256": sha_file(root / CAMPAIGN_FILENAME), "files_sha256": checksums})
    return _stored_image_smokes(root)[model_id]


def _image_gpu_evidence(records_root, repository, preparation, reference, weights_root, package, output, campaign_dir):
    """New per-encoder GPU inference gate, never inherited from DenseNet35."""
    from knee.training_smoke import verify_gpu_environment
    from knee.training_preparation import validate_preflight
    from knee.final_models_preflight import checkpoint_files
    verify_gpu_environment("google_colab_gpu")
    proof33 = find_image_preflight(records_root, repository)
    summary33 = validate_preflight(proof33)
    _, record38, _ = core.validate_preparation(preparation, repository)
    require(sha_file(reference) == record38["sources_sha256"]["notebook32_reference"],
        "same_authenticated_development_input_reference_required")
    weights = checked_path(weights_root)
    require(weights.is_dir() and all(not weights.is_relative_to(path) and not path.is_relative_to(weights)
        for path in map(checked_path, (repository, preparation, reference.parent, records_root, package, output))),
        "separate_existing_external_weights_cache_required")
    _, catalog, plan, _ = release.load_release(repository)
    saved_smokes = _stored_image_smokes(campaign_dir)
    models = {}
    for model_id in core.REMAINING_IMAGE_IDS:
        specification = catalog["models"][model_id]
        print(f'{model_id}: verificando pesos y ejecución sintética en GPU antes de entrenar.',
            file=sys.stderr, flush=True)
        paths = checkpoint_files(model_id, catalog, weights, True)
        probe = _probe_image_gpu(model_id, catalog, plan, paths)
        # Recheck the full bytes after loading; the gate does not adopt fold weights.
        checkpoint_files(model_id, catalog, weights, False)
        item = {"model_id": model_id, "image_eligibility_verified": True,
            "verified_checkpoint_files_sha256": {item["name"]: item["sha256"] for item in specification["files"]},
            "authenticated_colab33_record_sha256": summary33["checkpoint_record_sha256"],
            "license_review": specification["license_review"], "overlap_review": specification["overlap_review"],
            "scope": "private_noncommercial_academic_research",
            "pretraining_subject_disjointness_proven": False,
            "preprocessing_localizer_oai_overlap_limitation_remains": True, **probe}
        if model_id not in saved_smokes:
            smoke_root = checked_path(output).parent / ("encoder_smoke_" + uuid.uuid4().hex)
            core.run_remaining_encoder_smoke(repository, preparation, package, reference, weights,
                smoke_root, model_id=model_id, approval_statement=APPROVAL_STATEMENTS["remaining_encoders_frozen"],
                gpu_probe=item)
            saved_smokes[model_id] = _persist_image_smoke(campaign_dir, model_id, smoke_root)
        models[model_id] = {**item, **saved_smokes[model_id]}
    return {"image_eligibility_verified": True,
        "development_package_sha256": record38["development_package_sha256"],
        "model_input_reference_sha256": sha_file(reference), "model_eligibility": models,
        "dense_net35_not_used_as_other_encoder_evidence": True,
        "scope": "authenticated_colab33_current_gpu_forward_and_two_update_smokes_not_predictive_validation"}


def validate_result(result, campaign=None):
    require(isinstance(result, dict) and result.get("version") == core.VERSION and result.get("status") == core.STATUS
        and result.get("campaign") in core.CAMPAIGNS and (campaign is None or result["campaign"] == campaign)
        and result.get("job_count") == (45 if result["campaign"] == "clinical" else
            5 * len(core.image_models(result["campaign"])))
        and result.get("folds_completed") == list(core.FOLDS) and result.get("seed") == core.SEED
        and result.get("training_plan_sha256") == release.PINS[release.PLAN_PATH]
        and result.get("modeling_release_sha256") == release.PINS[release.RELEASE_PATH]
        and result.get("training_executed") is True and result.get("colab_runtime_executed") is True
        and result.get("local_rehearsal") is False, "complete_current_methodology_campaign_result_required")
    for field in ("reserved_test_accessed", "reserved_test_predictive_evaluation_opened", "originals_modified",
        "partitions_modified", "historical_v1_1_jobs_reused", "calibration_performed", "threshold_selected",
        "final_refit_executed", "full_encoder_training_authorized", "encoder_unfreezing_executed",
        "predictive_performance_claimed", "unbiased_final_performance_claimed"):
        require(result.get(field) is False, "methodology_campaign_scope_expansion_forbidden")
    require(len(result.get("job_results", [])) == result["job_count"]
        and len(result.get("model_results", {})) == (9 if result["campaign"] == "clinical" else
            len(core.image_models(result["campaign"]))),
        "complete_current_methodology_algorithm_scenario_coverage_required")
    if result["campaign"] == "remaining_encoders_frozen":
        require(set(result["model_results"]) == {f"clinical_main__{model}" for model in core.REMAINING_IMAGE_IDS},
            "exact_five_registered_image_results_required")
    return True


def train(repository_root, preparation_dir, package_dir, reference_path, output_dir, persistent_root, *,
        campaign, approval_statement, code_revision, cpu_tests, runtime, weights_root=None):
    require(campaign in core.CAMPAIGNS and approval_statement == APPROVAL_STATEMENTS[campaign],
        "future_investigator_confirmation_required_before_campaign_creation")
    require(cpu_tests.get("passed") is True and cpu_tests.get("skipped") == 0
        and type(cpu_tests.get("tests_run")) is int and cpu_tests["tests_run"] > 0
        and cpu_tests.get("scope") == list(test_modules(campaign))
        and cpu_tests.get("test_runtime") == ("synthetic_cpu_no_participant_training"
            if campaign == "clinical" else "gpu_build_cpu_compute_no_training"),
        "complete_methodology_campaign_synthetic_test_receipt_required")
    require(runtime.get("training_executed") is False and runtime.get("host_python_modified") is False
        and runtime.get("pretrained_weights_downloaded") is False
        and runtime.get("cpu_only") is (campaign == "clinical") and runtime.get("python_version") == "3.12.14",
        "verified_isolated_campaign_runtime_required")
    repository, proof, reference, output, persistent = map(checked_path,
        (repository_root, preparation_dir, reference_path, output_dir, persistent_root))
    require(not output.exists() and output.parent.is_dir() and persistent.is_dir()
        and all(not persistent.is_relative_to(path) and not path.is_relative_to(persistent)
            for path in (repository, checked_path(package_dir), reference.parent, output)),
        "separate_code_data_temporary_output_and_durable_records_required")
    if campaign == "clinical":
        from knee import training_preparation_colab as runtime_module
        runtime_sources = ("src/knee/training_preparation_colab.py", "requirements-training-runner.txt")
    else:
        from knee import training_gpu_runtime as runtime_module
        runtime_sources = runtime_module.SOURCE_FILES
    require(runtime.get("version") == runtime_module.VERSION
        and runtime.get("source_sha256") == {name: sha_file(repository / name) for name in runtime_sources},
        "runtime_receipt_must_match_selected_campaign_checkout")
    identity = campaign_identity(repository, proof, reference, code_revision, campaign)
    gpu = _gpu_evidence(persistent, repository) if campaign == "densenet_frozen" else None
    campaign_dir = open_campaign(persistent, identity)
    if campaign == "remaining_encoders_frozen":
        gpu = _image_gpu_evidence(persistent, repository, proof, reference, weights_root,
            package_dir, output, campaign_dir)
    complete = completed_job_transports(campaign_dir)
    result = core.run(repository, proof, package_dir, reference, output, campaign=campaign,
        approval_statement=approval_statement, weights_root=weights_root, gpu_evidence=gpu,
        completed_jobs={key: item["directory"] for key, item in complete.items()},
        job_completed=lambda key, directory, record: persist_completed_job(campaign_dir, key, directory, record))
    validate_result(result, campaign)
    require(campaign_identity(repository, proof, reference, code_revision, campaign) == identity,
        "campaign_code_or_sources_changed_during_execution")
    complete = completed_job_transports(campaign_dir)
    require(len(complete) == result["job_count"], "every_completed_job_must_be_durable")
    for key, checksum in result["artifact_files_sha256"].items():
        if "/" in key:
            job_name, filename = key.split("/", 1)
            require(job_name in complete and sha_file(complete[job_name]["directory"] / filename) == checksum,
                "durable_methodology_job_artifact_mismatch")
        else:
            source, target = output / key, campaign_dir / key
            _publish_copy(source, target, checksum)
    stable = lambda value: {key: val for key, val in value.items() if key not in {"executed_at_utc", "environment", "job_results"}}
    core_path = campaign_dir / core.FILENAME
    if core_path.exists():
        stored = _json(core_path)
        validate_result(stored, campaign)
        require(stable(stored) == stable(result), "recovered_completed_methodology_campaign_changed")
        result = stored
    else:
        _publish_json(core_path, result)
    result = {**result, "code_revision": code_revision, "cpu_tests": cpu_tests,
        "completed_jobs_saved_individually": len(complete), "result_files_saved_and_readback_verified": True,
        "persistent_campaign_folder": campaign_dir.name, "within_job_optimizer_resume_used": False,
        "core_record_sha256": sha_file(core_path), "gpu_runtime_evidence": gpu,
        "private_job_locations": {key: item["directory"].relative_to(campaign_dir).as_posix() for key, item in complete.items()}}
    target = campaign_dir / SUMMARY_FILENAME
    if target.exists():
        stored = _json(target)
        validate_result(stored, campaign)
        require(stable(stored) == stable(result), "recovered_methodology_summary_changed")
        result = stored
    else:
        _publish_json(target, result)
    return result
