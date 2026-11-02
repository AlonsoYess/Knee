"""Explicit recovery of the registered five-encoder development campaign.

Previously published prediction/record JSON is authenticated again. Completed
checkpoint payloads are only stat-checked, never downloaded, copied or loaded:
their original publication hashes remain provenance, not a new bitrot check.
Any future use of those weights must rehash their full payloads first.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import uuid

from knee import methodology_campaign as core
from knee import methodology_campaign_colab as previous
from knee import methodology_release as release
from knee import methodology_inputs_colab as inputs
from knee import clinical_scenarios as clinical
from knee.colab_data import checked_path, require, sha_file, valid_sha
from knee.aligned_training.authority import digest, validate_request
from knee.aligned_training import runner

VERSION = "registered_image_campaign_recovery_v1.0"
STATUS = "remaining_encoder_campaign_complete_after_verified_recovery"
SUMMARY_FILENAME = "colab_image_campaign_recovery_check.json"
COMPATIBILITY_PATH = "configs/image_campaign_legacy_compatibility_v1.json"
CAMPAIGN = "remaining_encoders_frozen"
CONFIRMATIONS = core.CONFIRMATIONS
APPROVAL_STATEMENTS = core.APPROVAL_STATEMENTS
JSON_FILES = {"training_job.json", "training_record.json", "clinical_state.json", "best_validation_predictions.json"}
CHECKPOINT_FILES = {"best_model.safetensors", "last_model.safetensors"}
MIN_CHECKPOINT_BYTES = 64 * 1024
MAX_JSON_BYTES = 4 * 1024**2
JOB_FILES = JSON_FILES | CHECKPOINT_FILES
SOURCE_FILES = previous.IMAGE_SOURCE_FILES + ("src/knee/image_campaign_recovery.py",
    "src/knee/training_epoch_resume.py", COMPATIBILITY_PATH)
setup_runtime = previous.setup_runtime
runtime_environment = previous.runtime_environment
stage_development = previous.stage_development
find_preparation = previous.find_preparation
find_image_preflight = previous.find_image_preflight
_save = previous._save
_json = previous._json


def test_modules(campaign=CAMPAIGN):
    require(campaign == CAMPAIGN, "only_registered_image_recovery_campaign_allowed")
    return previous.IMAGE_TEST_MODULES + ("test_image_campaign_recovery", "test_training_epoch_resume")


def run_tests(repository_root, campaign=CAMPAIGN):
    import torch
    import unittest
    repository = checked_path(repository_root)
    require(checked_path(__file__) == repository / "src/knee/image_campaign_recovery.py"
        and os.environ.get("CUDA_VISIBLE_DEVICES", "") == "" and not torch.cuda.is_available(),
        "recovery_tests_require_selected_checkout_and_hidden_gpu")
    sys.path.insert(0, str(repository / "tests"))
    with redirect_stdout(sys.stderr):
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromNames(test_modules(campaign))
        require(not loader.errors and suite.countTestCases() > 0, "all_recovery_tests_must_load")
        count = suite.countTestCases()
        result = unittest.TextTestRunner(stream=sys.stderr).run(suite)
    require(result.wasSuccessful() and result.testsRun == count and not result.skipped
        and not result.expectedFailures and not result.unexpectedSuccesses, "recovery_tests_not_complete")
    return {"passed": True, "tests_run": count, "skipped": 0, "scope": list(test_modules(campaign)),
        "test_runtime": "gpu_build_cpu_compute_no_training" if torch.version.cuda is not None
            else "synthetic_cpu_no_participant_training"}


def _small_json(path, expected_sha256=None):
    path = checked_path(path)
    require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= MAX_JSON_BYTES,
        "bounded_plain_recovery_json_required")
    raw = path.read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    require(expected_sha256 is None or checksum == expected_sha256, "recovery_json_hash_mismatch")
    def unique(pairs):
        result={}
        for key,value in pairs:
            require(key not in result,"duplicate_recovery_json_key")
            result[key]=value
        return result
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite_json")))
    require(isinstance(value, dict), "recovery_json_object_required")
    return value, checksum


def compatible_identity(identity, current, compatibility):
    """Only the pinned legacy implementation may retain its original folder."""
    require(compatibility.get("version") == "image_campaign_legacy_compatibility_v1.0"
        and compatibility.get("training_plan_sha256") == release.PINS[release.PLAN_PATH]
        and compatibility.get("modeling_release_sha256") == release.PINS[release.RELEASE_PATH],
        "exact_legacy_recovery_compatibility_contract_required")
    require(isinstance(identity, dict) and set(identity) == set(current)
        and identity.get("code_revision") == compatibility["legacy_code_revision"]
        and identity.get("campaign") == CAMPAIGN
        and all(identity[key] == current[key] for key in current if key not in {"code_revision", "source_files_sha256"}),
        "legacy_campaign_recipe_membership_or_reference_mismatch")
    sources = compatibility["legacy_source_files_sha256"]
    require(set(identity.get("source_files_sha256", {})) == set(sources) == set(previous.IMAGE_SOURCE_FILES)
        and all(valid_sha(identity["source_files_sha256"][name])
            and identity["source_files_sha256"][name] in (allowed if isinstance(allowed, list) else [allowed])
            for name, allowed in sources.items()), "unapproved_legacy_campaign_implementation")
    changed = {"src/knee/aligned_training/engine.py", "src/knee/aligned_training/runner.py",
        "scripts/train_image_screening_colab.py", "scripts/build_image_screening_notebook.py"}
    require(all(current["source_files_sha256"][name] in (allowed if isinstance(allowed, list) else [allowed])
        for name,allowed in sources.items() if name not in changed), "unapproved_non_recovery_source_change")
    return True


def find_compatible_campaign(persistent_root, current, compatibility):
    root = checked_path(persistent_root)
    matches = []
    for count, path in enumerate(root.iterdir(), 1):
        require(count <= previous.MAX_DIRECT_CHILDREN, "bounded_recovery_records_folder_required")
        if not path.name.startswith(previous.CAMPAIGN_PREFIX) or not path.is_dir():
            continue
        manifest_path = path / previous.CAMPAIGN_FILENAME
        if not manifest_path.is_file():
            continue
        manifest, _ = _small_json(manifest_path)
        identity = manifest.get("identity", {})
        if identity.get("campaign") != CAMPAIGN:
            continue
        require(manifest.get("version") == previous.VERSION, "legacy_campaign_manifest_version_required")
        compatible_identity(identity, current, compatibility)
        matches.append(path)
    require(len(matches) == 1, "one_exact_compatible_existing_image_campaign_required")
    return matches[0]


def _validate_engine_metadata(engine, request, job, rows):
    """Same epoch/AMP/selection rules; no checkpoint payload is opened here."""
    from knee.aligned_training.engine import EarlyStopping, learning_rate_factor, VERSION as ENGINE_VERSION
    from knee.densenet_fivefold import _validate_amp
    train = [r for r in rows if r["cv_fold"] != request["validation_fold"]]
    validation = [r for r in rows if r["cv_fold"] == request["validation_fold"]]
    identities = sorted([[r["src_subject_id"], r["side"]] for r in train])
    require(engine.get("version") == ENGINE_VERSION and engine.get("status") == "training_fold_complete"
        and engine.get("scope") == "development_only" and engine.get("smoke") is False
        and engine.get("model_id") == request["model_id"] and engine.get("stage") == "stage_1"
        and engine.get("seed") == core.SEED and engine.get("training_executed") is True
        and engine.get("optimizer_created") is True and engine.get("eligible_for_model_selection") is True
        and engine.get("reserved_test_accessed") is False and engine.get("restored_best_checkpoint") is True
        and engine.get("scheduler_horizon_epochs") == 60, "complete_registered_recovery_engine_required")
    old = engine.get("resume_supported") is False and engine.get("optimizer_state_saved") is False
    resumed = (engine.get("resume_supported") is True and engine.get("optimizer_state_saved") is True
        and engine.get("epoch_checkpoint_version") == "frozen_image_epoch_resume_v1.0"
        and type(engine.get("resumed_from_epoch")) is int
        and 0 <= engine["resumed_from_epoch"] <= engine.get("completed_epochs", 0))
    require(old or resumed, "known_legacy_or_explicit_epoch_resume_metadata_required")
    require(engine.get("effective_batch_size") == 16 and engine.get("micro_batch_size") == request["image_micro_batch_size"]
        and engine.get("device") == "cuda:0" and engine.get("precision") == "float16_autocast_grad_scaler"
        and engine.get("gpu_resource_profile_recorded") is True, "unchanged_image_device_precision_batch_required")
    require(engine.get("provenance") == {"validation_fold": request["validation_fold"],
        "development_package_sha256": request["development_package_sha256"], "train_identity_sha256": digest(identities),
        "request_sha256": digest(request), "training_knees": len(train), "validation_knees": len(validation),
        "trained_participants": sorted({x[0] for x in identities})}, "exact_recovery_training_binding_required")
    positive = sum(r["PROGRESION_48M"] for r in train)
    require(0 < positive < len(train) and core._finite(engine.get("positive_weight"), minimum=1e-30)
        and math.isclose(engine["positive_weight"], (len(train)-positive)/positive, rel_tol=1e-12),
        "training_only_recovery_class_weight_required")
    history = engine.get("history")
    require(isinstance(history, list) and 1 <= len(history) <= 60 and engine.get("completed_epochs") == len(history),
        "complete_recovery_history_required")
    monitor, stop = EarlyStopping(patience=8), False
    updates = math.ceil(len(train)/16)
    for epoch, item in enumerate(history, 1):
        require(not stop and item.get("epoch") == epoch and item.get("phase") == "head_only"
            and item.get("training_knees_seen") == len(train) and item.get("optimizer_updates") == updates
            and core._finite(item.get("weighted_loss_per_knee"))
            and item.get("learning_rate_factor") == learning_rate_factor(epoch)
            and isinstance(item.get("validation"), dict)
            and set(item["validation"]) == {"average_precision_score", "roc_auc", "brier"}
            and all(core._finite(v, maximum=1) for v in item["validation"].values()),
            "whole_registered_recovery_epoch_required")
        improved, stop = monitor.update(epoch, item["validation"]["average_precision_score"])
        require(item.get("selected_best") is improved, "unchanged_recovery_ap_selection_required")
    require((stop or len(history) == 60) and engine.get("best_epoch") == monitor.best_epoch
        and engine.get("best_validation_ap") == monitor.best_score
        and job["metrics"] == history[monitor.best_epoch-1]["validation"]
        and engine.get("optimizer_updates") == updates*len(history)
        and engine.get("checkpoint_sha256") == {name: job["files_sha256"][name] for name in CHECKPOINT_FILES},
        "complete_recovery_early_stop_and_checkpoint_metadata_required")
    _validate_amp(engine)


def read_job_metadata(directory, receipt, request, rows, plan, registry):
    """Read four JSON files once; checkpoint files receive stat checks only."""
    root = checked_path(directory)
    require(receipt.get("version") in {previous.VERSION, VERSION} and receipt.get("key") == core.job_key(request)
        and set(receipt) in ({"version", "key", "files_sha256", "job_record_sha256"},
            {"version", "key", "files_sha256", "job_record_sha256", "checkpoint_sizes"})
        and set(receipt.get("files_sha256", {})) == JOB_FILES
        and all(valid_sha(v) for v in receipt["files_sha256"].values()), "exact_complete_recovery_job_receipt_required")
    paths = {p.name: checked_path(p) for p in root.iterdir()}
    require(set(paths) == JOB_FILES and all(p.is_file() and not p.is_symlink() and p.parent == root for p in paths.values()),
        "plain_exact_recovery_job_inventory_required")
    maximum = previous.IMAGE_CHECKPOINT_LIMITS[request["model_id"]]
    sizes = {name: paths[name].stat().st_size for name in CHECKPOINT_FILES}
    require(all(MIN_CHECKPOINT_BYTES <= size <= maximum for size in sizes.values()) and len(set(sizes.values())) == 1
        and ("checkpoint_sizes" not in receipt or receipt["checkpoint_sizes"] == sizes),
        "bounded_existing_recovery_checkpoint_payloads_required")
    values = {name: _small_json(paths[name], receipt["files_sha256"][name])[0] for name in JSON_FILES}
    job, engine, state, predictions = (values[n] for n in
        ("training_job.json", "training_record.json", "clinical_state.json", "best_validation_predictions.json"))
    validate_request(request, plan, registry)
    require(job.get("version") == runner.VERSION and job.get("scope") == "development_only"
        and job.get("request") == request and request["operation"] == "development_fold"
        and job.get("request_sha256") == digest(request) and job.get("plan_sha256") == release.PINS[release.PLAN_PATH]
        and job.get("eligible_for_model_selection") is True and job.get("training_executed") is True
        and job.get("engine_status") == "training_fold_complete" and job.get("reserved_test_accessed") is False
        and job.get("selection_evidence") is None
        and job.get("record_sha256") == receipt["job_record_sha256"] == digest({k:v for k,v in job.items() if k != "record_sha256"})
        and job.get("files_sha256") == {k:v for k,v in receipt["files_sha256"].items() if k != "training_job.json"}
        and job.get("eligibility_gates") == {key: True for key in registry["eligibility_gates"]},
        "authenticated_registered_completed_recovery_job_required")
    expected_state = clinical.fit_scenario_fold(rows, request["validation_fold"], "clinical_main")
    require(state == expected_state, "exact_training_only_recovery_clinical_state_required")
    train = [r for r in rows if r["cv_fold"] != request["validation_fold"]]
    validation = [r for r in rows if r["cv_fold"] == request["validation_fold"]]
    participants = sorted({r["src_subject_id"] for r in train})
    require(not set(participants) & {r["src_subject_id"] for r in validation}
        and job.get("weights_provenance") == {"fit_scope":"development_training_fold", "validation_fold":request["validation_fold"],
            "train_identity_sha256":state["train_identity_sha256"], "trained_participants":participants,"reserved_test_accessed":False},
        "participant_disjoint_recovery_provenance_required")
    from knee.aligned_training.engine import binary_metrics
    require(set(predictions) == {"identities", "targets", "logits"}
        and predictions["identities"] == [[r["src_subject_id"],r["side"]] for r in validation]
        and predictions["targets"] == [r["PROGRESION_48M"] for r in validation]
        and len(predictions["logits"]) == len(validation)
        and binary_metrics(predictions["targets"], predictions["logits"]) == job["metrics"],
        "exact_authenticated_recovery_predictions_and_metrics_required")
    _validate_engine_metadata(engine, request, job, rows)
    return {"directory":root, "receipt":receipt, "record":job, "engine":engine,
        "predictions":predictions, "checkpoint_sizes":sizes, "checkpoint_payloads_rehashed_this_run":False}


def read_completed_jobs(campaign_dir, expected_requests, rows, plan, registry):
    root = checked_path(campaign_dir)
    expected = {core.job_key(r):r for r in expected_requests}
    completed = {}
    for count, path in enumerate(root.iterdir(),1):
        require(count <= previous.MAX_ATTEMPTS+128, "bounded_image_recovery_campaign_required")
        if not path.name.startswith("job_attempt_"):
            continue
        require(checked_path(path).is_dir() and not path.is_symlink(), "plain_recovery_job_attempt_required")
        if not (path/previous.JOB_RECEIPT).is_file():
            continue
        receipt,_ = _small_json(path/previous.JOB_RECEIPT)
        key = receipt.get("key")
        require(key in expected and key not in completed and {p.name for p in path.iterdir()} == {"job", previous.JOB_RECEIPT},
            "unique_expected_completed_recovery_job_required")
        completed[key] = read_job_metadata(path/"job",receipt,expected[key],rows,plan,registry)
    return completed


def _copy_once(source, target, expected_sha256):
    """One source stream and one destination readback; no repeated inventory."""
    source,target=map(checked_path,(source,target))
    require(not target.exists() and target.parent.is_dir(), "new_recovery_copy_destination_required")
    hasher=hashlib.sha256()
    with source.open("rb") as incoming,target.open("xb") as outgoing:
        while block:=incoming.read(1024**2):
            hasher.update(block)
            outgoing.write(block)
        outgoing.flush()
    require(hasher.hexdigest()==expected_sha256 and sha_file(target)==expected_sha256,
        "recovery_copy_and_destination_readback_mismatch")


def persist_completed_job(campaign_dir,key,directory,record,known):
    """Publish one fully validated new job; never rescan completed payloads."""
    require(key not in known and core.job_key(record["request"])==key,"new_unique_recovery_job_required")
    source=checked_path(directory)
    saved, job_sha=_small_json(source/runner.JOB_FILENAME)
    require(saved==record,"new_recovery_job_record_changed_before_publication")
    inventory={**record["files_sha256"],runner.JOB_FILENAME:job_sha}
    require(set(inventory)==JOB_FILES and {p.name for p in source.iterdir()}==JOB_FILES,
        "complete_new_recovery_job_artifacts_required")
    target=checked_path(campaign_dir)/("job_attempt_"+uuid.uuid4().hex)
    target.mkdir()
    destination=target/"job"
    destination.mkdir()
    for name,checksum in sorted(inventory.items()):
        _copy_once(source/name,destination/name,checksum)
    receipt={"version":VERSION,"key":key,"files_sha256":inventory,"job_record_sha256":record["record_sha256"],
        "checkpoint_sizes":{name:(destination/name).stat().st_size for name in CHECKPOINT_FILES}}
    previous._publish_json(target/previous.JOB_RECEIPT,receipt)
    print(f'{key}: guardado y verificado sin releer otros trabajos.',file=sys.stderr,flush=True)
    return destination,receipt


def _pending_gpu_evidence(repository,proof,package,reference,weights,output,persistent,campaign_dir,pending_models):
    """The runtime probes and public weights concern pending encoders only."""
    if not pending_models:
        return {"model_eligibility":{}, "current_gpu_probed_models":[]}
    from knee.training_smoke import verify_gpu_environment
    from knee.training_preparation import validate_preflight
    from knee.final_models_preflight import checkpoint_files
    verify_gpu_environment("google_colab_gpu")
    proof33=previous.find_image_preflight(persistent,repository)
    summary33=validate_preflight(proof33)
    _,catalog,plan,_=release.load_release(repository)
    saved=previous._stored_image_smokes(campaign_dir)
    models={}
    for model_id in pending_models:
        print(f'{model_id}: comprobación GPU y pesos solo porque tiene pliegues pendientes.',file=sys.stderr,flush=True)
        specification=catalog["models"][model_id]
        paths=checkpoint_files(model_id,catalog,weights,True)
        probe=previous._probe_image_gpu(model_id,catalog,plan,paths)
        checkpoint_files(model_id,catalog,weights,False)
        item={"model_id":model_id,"image_eligibility_verified":True,
            "verified_checkpoint_files_sha256":{v["name"]:v["sha256"] for v in specification["files"]},
            "authenticated_colab33_record_sha256":summary33["checkpoint_record_sha256"],
            "license_review":specification["license_review"],"overlap_review":specification["overlap_review"],**probe}
        if model_id not in saved:
            smoke_root=output.parent/("encoder_smoke_"+uuid.uuid4().hex)
            core.run_remaining_encoder_smoke(repository,proof,package,reference,weights,smoke_root,
                model_id=model_id,approval_statement=APPROVAL_STATEMENTS[CAMPAIGN],gpu_probe=item)
            saved[model_id]=previous._persist_image_smoke(campaign_dir,model_id,smoke_root)
        models[model_id]={**item,**saved[model_id]}
    return {"model_eligibility":models,"current_gpu_probed_models":list(pending_models)}


def _current_sources(repository):
    return {name:sha_file(repository/name) for name in SOURCE_FILES}


def _validate_startup(repository, proof, reference, runtime, cpu_tests, campaign):
    from knee import training_gpu_runtime as gpu
    require(campaign==CAMPAIGN and checked_path(__file__)==repository/"src/knee/image_campaign_recovery.py",
        "published_image_recovery_module_required")
    require(cpu_tests.get("passed") is True and cpu_tests.get("skipped")==0
        and type(cpu_tests.get("tests_run")) is int and cpu_tests["tests_run"]>0
        and cpu_tests.get("scope")==list(test_modules())
        and cpu_tests.get("test_runtime")=="gpu_build_cpu_compute_no_training",
        "complete_image_recovery_test_receipt_required")
    require(runtime.get("version")==gpu.VERSION and runtime.get("training_executed") is False
        and runtime.get("host_python_modified") is False and runtime.get("pretrained_weights_downloaded") is False
        and runtime.get("cpu_only") is False and runtime.get("python_version")=="3.12.14"
        and runtime.get("source_sha256")=={name:sha_file(repository/name) for name in gpu.SOURCE_FILES},
        "isolated_unchanged_image_recovery_runtime_required")
    _,core38,sidecar=core.validate_preparation(proof,repository)
    require(sha_file(reference)==core38["sources_sha256"]["notebook32_reference"],
        "exact_prepared_recovery_development_reference_required")
    return core38,sidecar


def _new_job_metadata(directory, record, request, rows, plan, registry, initial_state):
    """New local output still receives actual frozen-weight validation."""
    from knee.training_smoke import _verify_frozen_encoder
    root=checked_path(directory)
    receipt={"version":VERSION,"key":core.job_key(request),"files_sha256":{
        **record["files_sha256"],runner.JOB_FILENAME:sha_file(root/runner.JOB_FILENAME)},
        "job_record_sha256":record["record_sha256"],
        "checkpoint_sizes":{name:(root/name).stat().st_size for name in CHECKPOINT_FILES}}
    item=read_job_metadata(root,receipt,request,rows,plan,registry)
    _verify_frozen_encoder(root,initial_state)
    item["checkpoint_payloads_rehashed_this_run"]=True
    return item


def train(repository_root,preparation_dir,package_dir,reference_path,output_dir,persistent_root,*,
        campaign,approval_statement,code_revision,cpu_tests,runtime,weights_root=None):
    """Resume the same campaign, preserving complete jobs and epoch state."""
    require(campaign==CAMPAIGN and approval_statement==APPROVAL_STATEMENTS[CAMPAIGN],
        "explicit_registered_image_recovery_authority_required")
    repository,proof,package,reference,output,persistent=map(checked_path,
        (repository_root,preparation_dir,package_dir,reference_path,output_dir,persistent_root))
    require(not output.exists() and output.parent.is_dir() and persistent.is_dir()
        and all(not persistent.is_relative_to(p) and not p.is_relative_to(persistent)
            for p in (repository,package,reference.parent,output))
        and all(not output.is_relative_to(p) and not p.is_relative_to(output)
            for p in (repository,proof,package,reference.parent)), "separate_existing_recovery_sources_required")
    core38,sidecar=_validate_startup(repository,proof,reference,runtime,cpu_tests,campaign)
    compatibility,_=_small_json(repository/COMPATIBILITY_PATH)
    current=previous.campaign_identity(repository,proof,reference,code_revision,campaign)
    campaign_dir=find_compatible_campaign(persistent,current,compatibility)
    sources=_current_sources(repository)
    identity_hash=sha_file(campaign_dir/previous.CAMPAIGN_FILENAME)
    _,catalog,plan,registry=release.load_release(repository)
    from knee.aligned_training.data import load_sealed_inputs
    loaded=load_sealed_inputs(package,reference,repository,core38["development_package_sha256"])
    rows=loaded["manifest"]["rows"]
    require(len(rows)==core38["development_knees"] and len({r["src_subject_id"] for r in rows})==core38["development_participants"],
        "exact_recovery_development_membership_required")
    del loaded
    requests=core.requests(CAMPAIGN,core38["development_package_sha256"],sidecar["record_sha256"],plan,registry)
    completed=read_completed_jobs(campaign_dir,requests,rows,plan,registry)
    preserved=set(completed)
    pending=[r for r in requests if core.job_key(r) not in completed]
    pending_models=tuple(model for model in core.REMAINING_IMAGE_IDS if any(r["model_id"]==model for r in pending))
    print(f'Recuperación: {len(completed)}/25 trabajos completos conservados; {len(pending)} pendientes.',file=sys.stderr,flush=True)
    if pending:
        print(f'Primer trabajo pendiente: {core.job_key(pending[0])}.',file=sys.stderr,flush=True)
    output.mkdir()
    execution_id=uuid.uuid4().hex
    authorization={"version":VERSION,"code_revision":code_revision,"legacy_campaign_identity_sha256":identity_hash,
        "source_files_sha256":sources,"training_plan_sha256":release.PINS[release.PLAN_PATH],
        "development_package_sha256":core38["development_package_sha256"],"approval_statement":approval_statement,
        "completed_jobs_preserved":len(completed),"pending_job_keys":[core.job_key(r) for r in pending],
        "checkpoint_payloads_rehashed_on_recovery":False,"reserved_test_accessed":False,
        "created_at_utc":datetime.now(timezone.utc).isoformat()}
    previous._publish_json(campaign_dir/("recovery_execution_"+execution_id+".json"),authorization)
    weights=None
    if pending:
        require(weights_root is not None,"public_weights_cache_required_only_for_pending_jobs")
        weights=checked_path(weights_root)
        require(weights.is_dir() and all(not weights.is_relative_to(p) and not p.is_relative_to(weights)
            for p in (repository,proof,package,reference.parent,persistent,output)),"separate_recovery_public_weights_cache_required")
    gpu=_pending_gpu_evidence(repository,proof,package,reference,weights,output,persistent,campaign_dir,pending_models)
    initial=None
    current_model=None
    resumed_jobs=0
    for index,request in enumerate(pending,1):
        model_id=request["model_id"]
        key=core.job_key(request)
        item=gpu["model_eligibility"][model_id]
        core._validate_encoder_probe(model_id,item,catalog)
        core._require_smoke_receipt(item,model_id,core38["development_package_sha256"])
        if current_model!=model_id:
            initial=None
            initial=core._initial_image_state(model_id,plan,catalog,weights)
            current_model=model_id
        require(_current_sources(repository)==sources and sha_file(campaign_dir/previous.CAMPAIGN_FILENAME)==identity_hash,
            "recovery_sources_or_legacy_identity_changed")
        print(f'Pendiente {index}/{len(pending)}: {key}; pesos iniciales y control de época independientes.',file=sys.stderr,flush=True)
        permit=core.authority(request,CAMPAIGN,approval_statement,plan,registry,
            image_eligibility_verified=True,encoder_smoke_evidence=item)
        from knee.training_epoch_resume import EpochCheckpoint
        checkpoint=EpochCheckpoint(campaign_dir/"epoch_checkpoints"/key,request)
        local=output/key
        job=runner.execute(request,permit,repository_root=repository,package_dir=package,reference_path=reference,
            weights_root=weights,output_dir=local,device="cuda:0",allow_download=False,epoch_checkpoint=checkpoint)
        verified=_new_job_metadata(local,job,request,rows,plan,registry,initial)
        durable,receipt=persist_completed_job(campaign_dir,key,local,job,completed)
        verified.update(directory=durable,receipt=receipt)
        completed[key]=verified
        resumed_jobs+=int(verified["engine"].get("resumed_from_epoch",0)>0)
    initial=None
    require(len(completed)==25 and set(completed)=={core.job_key(r) for r in requests},"all_25_recovery_jobs_required")
    from knee.aligned_training.engine import aggregate_fold_metrics,binary_metrics
    model_results={}
    new_files={}
    for model_id in core.REMAINING_IMAGE_IDS:
        group=[completed[core.job_key(r)] for r in requests if r["model_id"]==model_id]
        metrics=[{"validation_fold":item["record"]["request"]["validation_fold"],**item["record"]["metrics"]} for item in group]
        oof=[]
        for item in group:
            p=item["predictions"]
            fold=item["record"]["request"]["validation_fold"]
            oof.extend({"src_subject_id":identity[0],"side":identity[1],"validation_fold":fold,"target":target,"logit":logit}
                for identity,target,logit in zip(p["identities"],p["targets"],p["logits"]))
        require(len(oof)==len(rows) and {(x["src_subject_id"],x["side"]) for x in oof}==
            {(r["src_subject_id"],r["side"]) for r in rows},"exact_recovery_oof_coverage_required")
        name="clinical_main__"+model_id
        model_results[name]={"fivefold_selection_metrics":aggregate_fold_metrics(metrics),
            "pooled_oof_metrics_descriptive_only":binary_metrics([x["target"] for x in oof],[x["logit"] for x in oof]),
            "out_of_fold_predictions":len(oof)}
        filename=name+"__oof.json"
        value={"scope":"development_only","scenario":"clinical_main","model_id":model_id,"rows":oof,
            "pooled_metric_role":"descriptive_only_not_selection_metric"}
        runner.save_json(output/filename,value)
        new_files[filename]=sha_file(output/filename)
    close_dir=campaign_dir/("recovery_completion_"+execution_id)
    close_dir.mkdir()
    for filename,checksum in new_files.items():
        _copy_once(output/filename,close_dir/filename,checksum)
    require(_current_sources(repository)==sources and sha_file(campaign_dir/previous.CAMPAIGN_FILENAME)==identity_hash,
        "recovery_sources_changed_at_completion")
    result={"version":VERSION,"status":STATUS,"campaign":CAMPAIGN,"code_revision":code_revision,
        "scope":"development_only","seed":core.SEED,"stage":"stage_1","folds_completed":list(core.FOLDS),
        "job_count":25,"model_results":model_results,"development_knees":len(rows),
        "development_participants":core38["development_participants"],
        "development_package_sha256":core38["development_package_sha256"],"model_input_reference_sha256":sha_file(reference),
        "training_plan_sha256":release.PINS[release.PLAN_PATH],"modeling_release_sha256":release.PINS[release.RELEASE_PATH],
        "completed_jobs_preserved":len(preserved),"jobs_executed_this_invocation":len(pending),
        "within_job_epoch_resume_used":resumed_jobs>0,"jobs_resumed_from_complete_epoch":resumed_jobs,
        "training_executed_this_invocation":bool(pending),"training_executed_in_campaign":True,
        "current_gpu_probed_models":list(pending_models),"pretrained_weights_loaded_for_completed_only_models":False,
        "completed_checkpoint_payloads_copied_on_recovery":False,"checkpoint_payloads_rehashed_on_recovery":False,
        "checkpoint_rehash_required_before_weight_reuse":True,"new_job_payloads_verified_on_publication":True,
        "legacy_campaign_identity_preserved":True,"legacy_campaign_identity_sha256":identity_hash,
        "persistent_campaign_folder":campaign_dir.name,"source_files_sha256":sources,"cpu_tests":cpu_tests,
        "job_results":[{"key":core.job_key(r),"job_record_sha256":completed[core.job_key(r)]["record"]["record_sha256"],
            "metrics":completed[core.job_key(r)]["record"]["metrics"],"preserved_complete_job":core.job_key(r) in preserved}
            for r in requests],
        "private_job_locations":{key:item["directory"].relative_to(campaign_dir).as_posix() for key,item in completed.items()},
        "artifact_files_sha256":{core.job_key(r)+"/"+name:checksum for r in requests
            for name,checksum in completed[core.job_key(r)]["receipt"]["files_sha256"].items()},
        "oof_files_sha256":new_files,"result_files_saved_and_readback_verified":True,
        "colab_runtime_executed":True,"local_rehearsal":False,"reserved_test_accessed":False,
        "reserved_test_predictive_evaluation_opened":False,"partitions_modified":False,"originals_modified":False,
        "encoder_unfreezing_executed":False,"full_encoder_training_authorized":False,
        "calibration_performed":False,"threshold_selected":False,"final_refit_executed":False,
        "fusion_training_executed":False,"predictive_performance_claimed":False,
        "unbiased_final_performance_claimed":False,"executed_at_utc":datetime.now(timezone.utc).isoformat()}
    validate_result(result)
    previous._publish_json(close_dir/SUMMARY_FILENAME,result)
    return result


def validate_result(result,campaign=CAMPAIGN):
    require(campaign==CAMPAIGN and result.get("version")==VERSION and result.get("status")==STATUS
        and result.get("campaign")==CAMPAIGN and result.get("job_count")==25
        and result.get("completed_jobs_preserved",-100)+result.get("jobs_executed_this_invocation",-100)==25
        and result.get("folds_completed")==list(core.FOLDS) and result.get("seed")==core.SEED
        and result.get("training_plan_sha256")==release.PINS[release.PLAN_PATH]
        and result.get("modeling_release_sha256")==release.PINS[release.RELEASE_PATH]
        and len(result.get("job_results",[]))==25
        and set(result.get("model_results",{}))=={"clinical_main__"+model for model in core.REMAINING_IMAGE_IDS},
        "complete_bounded_image_recovery_result_required")
    for field in ("reserved_test_accessed","reserved_test_predictive_evaluation_opened","partitions_modified",
        "originals_modified","encoder_unfreezing_executed","full_encoder_training_authorized","calibration_performed",
        "threshold_selected","final_refit_executed","fusion_training_executed","predictive_performance_claimed",
        "unbiased_final_performance_claimed","completed_checkpoint_payloads_copied_on_recovery",
        "checkpoint_payloads_rehashed_on_recovery","pretrained_weights_loaded_for_completed_only_models"):
        require(result.get(field) is False,"recovery_scope_or_integrity_claim_expansion")
    require(result.get("checkpoint_rehash_required_before_weight_reuse") is True
        and result.get("legacy_campaign_identity_preserved") is True
        and result.get("result_files_saved_and_readback_verified") is True,"explicit_recovery_provenance_required")
    return True
