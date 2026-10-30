"""Colab boundary for the authorized four-model gradual development campaign.

Setup, unit checks and preparation cannot enter the training path. The model
selection, fold execution, resource probes and recovery remain in the core.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
import uuid

from knee.colab_data import checked_path, require, sha_file, valid_sha
from knee.training_preparation import read_json as _json

VERSION = "gradual_campaign_colab_orchestration_v1.0"
CAMPAIGN = "gradual_four_models"
MODELS = ("image_densenet121", "image_vit_b16", "image_skelex", "image_mambaout_tiny")
PREPARATION_FILENAME = "preparation_record.json"
SELECTION_FILENAME = "selection_evidence.json"
PREPARATION_RECEIPT = "preparation_colab_receipt.json"
SUMMARY_FILENAME = "colab_gradual_campaign_check.json"
TEST_MODULES = ("test_methodology_release", "test_methodology_portable_release",
    "test_training_augmentation", "test_methodology_augmentation_integration",
    "test_training_models", "test_aligned_training_models", "test_training_epoch_resume",
    "test_gradual_epoch_resume", "test_gradual_selection", "test_gradual_campaign", "test_gradual_campaign_colab",
    "test_gradual_notebook")
SOURCE_FILES = ("src/knee/gradual_campaign.py", "src/knee/gradual_campaign_colab.py",
    "src/knee/gradual_epoch_resume.py", "src/knee/gradual_selection.py",
    "scripts/train_gradual_campaign_colab.py", "scripts/build_gradual_training_notebook.py",
    "src/knee/training_epoch_resume.py", "src/knee/aligned_training/engine.py",
    "src/knee/aligned_training/runner.py", "src/knee/aligned_training/models.py",
    "src/knee/aligned_training/data.py", "src/knee/aligned_training/authority.py",
    "src/knee/methodology_release.py", "configs/training_plan_v1_2.json")


def _save(path, value):
    path = checked_path(path)
    require(not path.exists() and path.parent.is_dir(), "new_gradual_record_destination_required")
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(raw)
    require(path.read_bytes() == raw, "gradual_record_readback_failed")


def _selected(repository_root, campaign=CAMPAIGN):
    repository = checked_path(repository_root).resolve()
    require(campaign == CAMPAIGN and checked_path(__file__).resolve() ==
        repository / "src/knee/gradual_campaign_colab.py", "selected_gradual_checkout_and_campaign_required")
    return repository


def _revision(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None
        and value != "0" * 40, "published_gradual_code_revision_required")


def _source_hashes(repository):
    return {name: sha_file(checked_path(repository / name)) for name in SOURCE_FILES}


def setup_runtime(repository_root, destination, campaign=CAMPAIGN):
    repository = _selected(repository_root, campaign)
    from knee.training_gpu_runtime import setup_runtime as setup
    return setup(repository, destination)


def runtime_environment(repository_root, campaign=CAMPAIGN):
    repository = _selected(repository_root, campaign)
    from knee.training_gpu_runtime import runtime_environment as environment
    return environment(repository)


def stage_development(remote_package_dir, package_dir, reference_path):
    from knee.methodology_inputs_colab import stage_development as stage
    return stage(remote_package_dir, package_dir, reference_path)


def run_tests(repository_root, campaign=CAMPAIGN):
    repository = _selected(repository_root, campaign)
    import torch
    import unittest
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") == "" and not torch.cuda.is_available(),
        "gradual_synthetic_tests_must_hide_gpu")
    sys.path.insert(0, str(repository / "tests"))
    with redirect_stdout(sys.stderr):
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromNames(TEST_MODULES)
        require(not loader.errors and suite.countTestCases() > 0, "all_gradual_test_modules_must_load")
        expected = suite.countTestCases()
        result = unittest.TextTestRunner(stream=sys.stderr).run(suite)
    require(result.wasSuccessful() and result.testsRun == expected and not result.skipped
        and not result.expectedFailures and not result.unexpectedSuccesses, "complete_gradual_tests_required")
    return {"passed": True, "tests_run": expected, "skipped": 0, "scope": list(TEST_MODULES),
        "test_runtime": "gpu_build_cpu_compute_no_training" if torch.version.cuda is not None
            else "synthetic_cpu_no_participant_training"}


def _validate_execution_receipts(repository, cpu_tests, runtime):
    from knee import training_gpu_runtime as gpu
    require(isinstance(cpu_tests, dict) and cpu_tests.get("passed") is True
        and type(cpu_tests.get("tests_run")) is int and cpu_tests["tests_run"] > 0
        and cpu_tests.get("skipped") == 0 and cpu_tests.get("scope") == list(TEST_MODULES)
        and cpu_tests.get("test_runtime") == "gpu_build_cpu_compute_no_training",
        "complete_gradual_hidden_gpu_test_receipt_required")
    require(isinstance(runtime, dict) and runtime.get("version") == gpu.VERSION
        and runtime.get("training_executed") is False and runtime.get("host_python_modified") is False
        and runtime.get("pretrained_weights_downloaded") is False and runtime.get("cpu_only") is False
        and runtime.get("python_version") == "3.12.14"
        and runtime.get("source_sha256") == {name: sha_file(repository / name) for name in gpu.SOURCE_FILES},
        "unchanged_isolated_gradual_gpu_runtime_required")


def _validate_preparation_record(record):
    require(isinstance(record, dict) and record.get("status") == "gradual_campaign_prepared_no_training"
        and record.get("model_order") == list(MODELS) and record.get("job_count") == 20
        and record.get("training_executed") is False
        and valid_sha(record.get("selection_evidence_sha256"))
        and valid_sha(record.get("development_package_sha256")), "exact_untrained_gradual_preparation_required")


def validate_preparation_receipt(repository_root, preparation_dir, *, code_revision):
    repository = _selected(repository_root)
    _revision(code_revision)
    directory = checked_path(preparation_dir)
    receipt = _json(directory / PREPARATION_RECEIPT)
    record = _json(directory / PREPARATION_FILENAME)
    _validate_preparation_record(record)
    require(receipt.get("version") == VERSION and receipt.get("campaign") == CAMPAIGN
        and receipt.get("code_revision") == code_revision and receipt.get("training_executed") is False
        and receipt.get("source_files_sha256") == _source_hashes(repository)
        and receipt.get("preparation_record_sha256") == sha_file(directory / PREPARATION_FILENAME)
        and receipt.get("selection_evidence_sha256") == sha_file(directory / SELECTION_FILENAME)
        and record["selection_evidence_sha256"] == receipt["selection_evidence_sha256"],
        "unchanged_exact_gradual_preparation_receipt_required")
    _validate_execution_receipts(repository, receipt.get("cpu_tests"), receipt.get("runtime"))
    return receipt


def prepare(repository_root, records_root, package_dir, reference_path, output_dir, *,
        code_revision, cpu_tests, runtime, campaign=CAMPAIGN):
    repository = _selected(repository_root, campaign)
    _revision(code_revision)
    _validate_execution_receipts(repository, cpu_tests, runtime)
    records, package, reference, output = map(checked_path, (records_root, package_dir, reference_path, output_dir))
    require(records.is_dir() and not output.exists() and output.parent.is_dir()
        and all(not output.is_relative_to(path) and not path.is_relative_to(output)
            for path in (repository, records, package, reference.parent)), "separate_gradual_preparation_output_required")
    sources = _source_hashes(repository)
    from knee import gradual_campaign as core
    record = core.prepare(repository, records, package, reference, output,
        code_revision=code_revision, cpu_tests=cpu_tests, runtime=runtime)
    _validate_preparation_record(record)
    require(_json(output / PREPARATION_FILENAME) == record
        and sha_file(output / SELECTION_FILENAME) == record["selection_evidence_sha256"]
        and _source_hashes(repository) == sources, "prepared_gradual_sources_or_records_changed")
    receipt = {"version": VERSION, "campaign": CAMPAIGN, "code_revision": code_revision,
        "source_files_sha256": sources, "preparation_record_sha256": sha_file(output / PREPARATION_FILENAME),
        "selection_evidence_sha256": record["selection_evidence_sha256"], "cpu_tests": cpu_tests,
        "runtime": runtime, "training_executed": False, "reserved_test_accessed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat()}
    _save(output / PREPARATION_RECEIPT, receipt)
    validate_preparation_receipt(repository, output, code_revision=code_revision)
    private = records / ("gradual_preparation42_" + uuid.uuid4().hex)
    private.mkdir()
    for name in (PREPARATION_FILENAME, SELECTION_FILENAME, PREPARATION_RECEIPT):
        source = output / name
        require(source.is_file() and 0 < source.stat().st_size <= 64 * 1024**2,
            "bounded_gradual_preparation_record_required")
        with (private / name).open("xb") as stream:
            stream.write(source.read_bytes())
        require(sha_file(private / name) == sha_file(source), "private_gradual_preparation_readback_failed")
    return {**record, "colab_preparation_receipt_sha256": sha_file(output / PREPARATION_RECEIPT),
        "preparation_saved_and_readback_verified": True, "persistent_preparation_folder": private.name}


def train(repository_root, preparation_dir, package_dir, reference_path, output_dir, *,
        weights_root, persistent_root, code_revision, campaign=CAMPAIGN):
    repository = _selected(repository_root, campaign)
    receipt = validate_preparation_receipt(repository, preparation_dir, code_revision=code_revision)
    from knee import training_gpu_runtime as gpu
    gpu._host()
    from knee import gradual_campaign as core
    result = core.run(repository, preparation_dir, package_dir, reference_path, output_dir,
        weights_root=weights_root, persistent_root=persistent_root, code_revision=code_revision,
        cpu_tests=receipt["cpu_tests"], runtime=receipt["runtime"], device="cuda:0", allow_download=True)
    validate_result(result)
    require(_source_hashes(repository) == receipt["source_files_sha256"], "gradual_sources_changed_during_execution")
    result = {**result, "colab_orchestration_version": VERSION, "code_revision": code_revision,
        "colab_runtime_executed": True, "local_rehearsal": False,
        "preparation_colab_receipt_sha256": sha_file(checked_path(preparation_dir) / PREPARATION_RECEIPT)}
    _save(checked_path(output_dir) / SUMMARY_FILENAME, result)
    # Core saves its own complete record in the private campaign folder. This
    # small wrapper record is also durable, without recopying any model payload.
    private = checked_path(persistent_root) / ("gradual_colab42_" + uuid.uuid4().hex)
    private.mkdir()
    _save(private / SUMMARY_FILENAME, result)
    require(sha_file(private / SUMMARY_FILENAME) == sha_file(checked_path(output_dir) / SUMMARY_FILENAME),
        "durable_gradual_colab_summary_readback_failed")
    return result


def validate_result(result):
    require(isinstance(result, dict) and result.get("version") == "development_gradual_image_campaign_v1.0"
        and result.get("status") == "four_model_gradual_development_campaign_complete"
        and result.get("campaign") == CAMPAIGN and result.get("scope") == "development_only"
        and result.get("stage") == "stage_2" and result.get("job_count") == 20
        and result.get("model_order") == list(MODELS) and result.get("folds_completed") == list(range(5))
        and result.get("seed") == 2026 and result.get("training_executed_in_campaign") is True,
        "exact_complete_gradual_campaign_result_required")
    for name in ("reserved_test_accessed", "reserved_test_predictive_evaluation_opened", "calibration_performed",
            "threshold_selected", "final_refit_executed", "fusion_training_executed", "originals_modified",
            "partitions_modified"):
        require(result.get(name) is False, "gradual_campaign_scope_expansion:" + name)
    preserved, executed = result.get("completed_jobs_preserved"), result.get("jobs_executed_this_invocation")
    require(type(preserved) is int and type(executed) is int and 0 <= preserved <= 20
        and 0 <= executed <= 20 and preserved + executed == 20
        and result.get("training_executed_this_invocation") is (executed > 0),
        "exact_gradual_recovery_job_accounting_required")
    require(type(result.get("development_knees")) is int and result["development_knees"] > 0
        and type(result.get("development_participants")) is int
        and 0 < result["development_participants"] <= result["development_knees"],
        "positive_gradual_development_membership_required")
    expected = {f"clinical_main__{model}__fold_{fold}" for model in MODELS for fold in range(5)}
    jobs = result.get("job_results")
    metrics = {"average_precision_score", "roc_auc", "brier"}
    def valid_metrics(value):
        return isinstance(value, dict) and set(value) == metrics and all(type(v) in {int, float}
            and math.isfinite(v) and 0 <= v <= 1 for v in value.values())
    require(isinstance(jobs, list) and len(jobs) == 20 and all(isinstance(job, dict) for job in jobs)
        and {job.get("key") for job in jobs} == expected
        and all(valid_sha(job.get("job_record_sha256")) and valid_metrics(job.get("metrics"))
            and type(job.get("preserved_complete_job")) is bool for job in jobs)
        and sum(job["preserved_complete_job"] for job in jobs) == preserved,
        "unique_complete_gradual_jobs_with_finite_metrics_required")
    job_map = {job["key"]: job for job in jobs}
    summaries = result.get("model_results")
    require(isinstance(summaries, dict) and set(summaries) == {"clinical_main__" + model for model in MODELS},
        "exact_four_gradual_model_summaries_required")
    for model in MODELS:
        entry = summaries["clinical_main__" + model]
        expected_means = {metric: math.fsum(job_map[f"clinical_main__{model}__fold_{fold}"]["metrics"][metric]
            for fold in range(5)) / 5 for metric in metrics}
        require(isinstance(entry, dict) and entry.get("fivefold_selection_metrics") == expected_means
            and valid_metrics(entry.get("pooled_oof_metrics_descriptive_only"))
            and entry.get("out_of_fold_predictions") == result["development_knees"],
            "gradual_selection_must_use_exact_fivefold_means_and_complete_oof")
    files = result.get("oof_files_sha256")
    require(isinstance(files, dict) and set(files) == {"clinical_main__" + model + "__oof.json" for model in MODELS}
        and all(valid_sha(value) for value in files.values()), "four_sealed_gradual_oof_files_required")
    for name in ("completed_checkpoint_payloads_copied_on_recovery", "checkpoint_payloads_rehashed_on_recovery",
            "unbiased_final_performance_claimed"):
        require(result.get(name) is False, "gradual_recovery_or_performance_claim_expansion")
    for name in ("checkpoint_rehash_required_before_weight_reuse", "new_job_payloads_verified_on_publication",
            "full_encoder_training_authorized", "encoder_unfreezing_executed", "result_files_saved_and_readback_verified"):
        require(result.get(name) is True, "required_gradual_execution_provenance:" + name)
    return True
