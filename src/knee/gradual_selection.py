"""Authenticated stage-one selection for the approved gradual image campaign.

No stage-one weights are reused in stage two. This module authenticates small
records and held-out predictions, while only stat-checking their old tensor
files. It cannot authorize fusion or waive full verification when weights are
actually reused. Private summary digests live in a separate private authority.
"""
from __future__ import annotations

import json
import math
from pathlib import Path, PurePosixPath

from knee import clinical_scenarios as clinical
from knee import methodology_campaign as core
from knee import methodology_campaign_colab as previous
from knee import image_campaign_recovery as recovery
from knee import methodology_release as release
from knee.aligned_training import runner
from knee.aligned_training.authority import digest, validate_request
from knee.colab_data import checked_path, require, valid_sha, validate_rows

VERSION = "gradual_image_selection_evidence_v1.0"
AUTHORITY_VERSION = "gradual_screening_authority_v1.0"
AUTHORITY_FILENAME = "gradual_screening_authority.json"
MODEL_ORDER = ("image_densenet121", "image_vit_b16", "image_skelex", "image_mambaout_tiny")
SCREENED_MODELS = ("image_densenet121", *core.REMAINING_IMAGE_IDS)
EXCLUDED = {"image_dinov3_vits16": "authorized_pretrained_checkpoint_access_pending"}
AUTHORITY_FIELDS = {"version", "approved_stage", "source40_summary_digest", "source41_summary_digest",
    "model_order", "training_plan_sha256", "modeling_release_sha256"}
_VALIDATED = {}


def screening_authority(summary40, summary41):
    """Construct a private approval descriptor from the received final records."""
    previous.validate_result(summary40, "densenet_frozen")
    recovery.validate_result(summary41)
    _same_source_scope(summary40, summary41)
    return {"version": AUTHORITY_VERSION, "approved_stage": "stage_2",
        "source40_summary_digest": digest(summary40), "source41_summary_digest": digest(summary41),
        "model_order": list(MODEL_ORDER), "training_plan_sha256": release.PINS[release.PLAN_PATH],
        "modeling_release_sha256": release.PINS[release.RELEASE_PATH]}


def _authority(path):
    value, checksum = recovery._small_json(path)
    require(set(value) == AUTHORITY_FIELDS and value.get("version") == AUTHORITY_VERSION
        and value.get("approved_stage") == "stage_2" and value.get("model_order") == list(MODEL_ORDER)
        and value.get("training_plan_sha256") == release.PINS[release.PLAN_PATH]
        and value.get("modeling_release_sha256") == release.PINS[release.RELEASE_PATH]
        and valid_sha(value.get("source40_summary_digest")) and valid_sha(value.get("source41_summary_digest")),
        "exact_private_gradual_screening_authority_required")
    return value, checksum


def _same_source_scope(summary40, summary41):
    for summary in (summary40, summary41):
        require(summary.get("scope") == "development_only" and summary.get("stage") == "stage_1"
            and summary.get("seed") == 2026 and summary.get("folds_completed") == list(range(5))
            and summary.get("colab_runtime_executed") is True and summary.get("local_rehearsal") is False
            and summary.get("result_files_saved_and_readback_verified") is True
            and summary.get("training_plan_sha256") == release.PINS[release.PLAN_PATH]
            and summary.get("modeling_release_sha256") == release.PINS[release.RELEASE_PATH]
            and summary.get("cpu_tests", {}).get("passed") is True and summary["cpu_tests"].get("skipped") == 0,
            "completed_actual_development_screening_sources_required")
        for key in ("reserved_test_accessed", "reserved_test_predictive_evaluation_opened", "originals_modified",
                "partitions_modified", "encoder_unfreezing_executed", "calibration_performed", "threshold_selected", "final_refit_executed"):
            require(summary.get(key) is False, "screening_source_scope_expansion_forbidden")
    for key in ("development_package_sha256", "model_input_reference_sha256", "training_plan_sha256",
            "modeling_release_sha256", "development_knees", "development_participants"):
        require(summary40.get(key) == summary41.get(key), "screening_sources_must_share_recipe_and_development")
    require(all(valid_sha(summary40.get(key)) for key in ("development_package_sha256", "model_input_reference_sha256")),
        "sealed_screening_development_hashes_required")


def _relative(root, relative):
    require(isinstance(relative, str) and relative and "\\" not in relative and ":" not in relative
        and not relative.startswith("/") and all(p not in {"", ".", ".."} for p in relative.split("/"))
        and str(PurePosixPath(relative)) == relative, "plain_private_screening_relative_path_required")
    path = checked_path(root / relative)
    require(path.is_relative_to(root), "screening_path_must_remain_inside_its_campaign")
    return path


def _source(root, number, authority):
    root = checked_path(root)
    require(root.is_dir(), "existing_screening_campaign_directory_required")
    expected_digest = authority[f"source{number}_summary_digest"]
    if number == "40":
        candidates = [root / previous.SUMMARY_FILENAME]
    else:
        candidates = []
        for count, path in enumerate(root.iterdir(), 1):
            require(count <= previous.MAX_ATTEMPTS + 128, "bounded_screening_campaign_inventory_required")
            if path.name.startswith("recovery_completion_") and checked_path(path).is_dir():
                candidate = path / recovery.SUMMARY_FILENAME
                if candidate.is_file():
                    candidates.append(candidate)
    matches = []
    for path in candidates:
        summary, checksum = recovery._small_json(path)
        if digest(summary) == expected_digest:
            matches.append((path, summary, checksum))
    require(len(matches) == 1, "one_exact_privately_authorized_screening_summary_required:" + number)
    path, summary, checksum = matches[0]
    if number == "40":
        previous.validate_result(summary, "densenet_frozen")
        core_record, core_sha = recovery._small_json(root / core.FILENAME, summary["core_record_sha256"])
        require(all(summary.get(k) == v for k, v in core_record.items()), "screening40_core_summary_mismatch")
    else:
        recovery.validate_result(summary)
        core_sha = None
    manifest, manifest_sha = recovery._small_json(root / previous.CAMPAIGN_FILENAME)
    identity = manifest.get("identity", {})
    require(manifest.get("version") == previous.VERSION and identity.get("campaign") == summary["campaign"]
        and identity.get("seed") == 2026 and identity.get("stage") == "stage_1"
        and all(identity.get(k) == summary.get(k) for k in ("development_package_sha256", "model_input_reference_sha256",
            "training_plan_sha256", "modeling_release_sha256")), "screening_campaign_manifest_binding_required")
    if number == "40":
        require(identity.get("code_revision") == summary.get("code_revision"), "screening40_code_identity_required")
    else:
        require(manifest_sha == summary.get("legacy_campaign_identity_sha256"), "screening41_legacy_identity_required")
    return summary, {"campaign_directory": str(root), "summary_relative_path": path.relative_to(root).as_posix(),
        "summary_sha256": checksum, "summary_digest": expected_digest, "campaign_identity_sha256": manifest_sha,
        "source40_core_sha256": core_sha, "code_revision": summary["code_revision"]}


def _job(root, summary, request, rows, plan, registry):
    """Authenticate all small evidence; do not open any checkpoint payload."""
    key = core.job_key(request)
    directory = _relative(root, summary["private_job_locations"][key])
    require(directory.name == "job" and directory.parent.name.startswith("job_attempt_"), "published_screening_job_directory_required")
    receipt, receipt_sha = recovery._small_json(directory.parent / previous.JOB_RECEIPT)
    receipt_fields = {"version", "key", "files_sha256", "job_record_sha256"}
    require(set(receipt) in (receipt_fields, receipt_fields | {"checkpoint_sizes"})
        and receipt.get("version") in {previous.VERSION, recovery.VERSION} and receipt.get("key") == key
        and set(receipt.get("files_sha256", {})) == recovery.JOB_FILES
        and all(valid_sha(v) for v in receipt["files_sha256"].values())
        and {p.name for p in directory.iterdir()} == recovery.JOB_FILES,
        "exact_published_screening_job_receipt_required")
    for name in recovery.JOB_FILES:
        path = checked_path(directory / name)
        require(path.is_file() and path.parent == directory, "plain_screening_job_files_required")
        require(summary["artifact_files_sha256"].get(key + "/" + name) == receipt["files_sha256"][name],
            "screening_summary_receipt_artifact_mismatch")
    maximum = previous.MAX_SINGLE_FILE_BYTES if request["model_id"] == "image_densenet121" else previous.IMAGE_CHECKPOINT_LIMITS[request["model_id"]]
    sizes = {name: (directory / name).stat().st_size for name in recovery.CHECKPOINT_FILES}
    require(len(set(sizes.values())) == 1 and all(recovery.MIN_CHECKPOINT_BYTES <= v <= maximum for v in sizes.values())
        and ("checkpoint_sizes" not in receipt or receipt["checkpoint_sizes"] == sizes),
        "bounded_existing_screening_checkpoint_sizes_required")
    values = {name: recovery._small_json(directory / name, receipt["files_sha256"][name])[0] for name in recovery.JSON_FILES}
    job, state, predictions, engine = (values[name] for name in
        ("training_job.json", "clinical_state.json", "best_validation_predictions.json", "training_record.json"))
    validate_request(request, plan, registry)
    require(job.get("version") == runner.VERSION and job.get("scope") == "development_only"
        and job.get("request") == request and job.get("request_sha256") == digest(request)
        and job.get("plan_sha256") == release.PINS[release.PLAN_PATH]
        and job.get("eligible_for_model_selection") is True and job.get("training_executed") is True
        and job.get("engine_status") == "training_fold_complete" and job.get("reserved_test_accessed") is False
        and job.get("selection_evidence") is None
        and job.get("record_sha256") == receipt["job_record_sha256"] == digest({k:v for k,v in job.items() if k != "record_sha256"})
        and job.get("files_sha256") == {k:v for k,v in receipt["files_sha256"].items() if k != runner.JOB_FILENAME}
        and job.get("eligibility_gates") == {key: True for key in registry["eligibility_gates"]},
        "authenticated_complete_screening_job_required")
    matching = [item for item in summary["job_results"] if item["key"] == key]
    require(len(matching) == 1 and matching[0]["job_record_sha256"] == job["record_sha256"]
        and matching[0]["metrics"] == job["metrics"]
        and ("request_sha256" not in matching[0] or matching[0]["request_sha256"] == digest(request)),
        "screening_job_summary_metrics_or_identity_mismatch")
    fold = request["validation_fold"]
    expected_state = clinical.fit_scenario_fold(rows, fold, "clinical_main")
    require(state == expected_state, "screening_training_only_clinical_state_mismatch")
    train, validation = ([r for r in rows if r["cv_fold"] != fold], [r for r in rows if r["cv_fold"] == fold])
    participants = sorted({r["src_subject_id"] for r in train})
    require(not set(participants) & {r["src_subject_id"] for r in validation}
        and job.get("weights_provenance") == {"fit_scope": "development_training_fold", "validation_fold": fold,
            "train_identity_sha256": state["train_identity_sha256"], "trained_participants": participants, "reserved_test_accessed": False},
        "participant_disjoint_screening_weight_provenance_required")
    from knee.aligned_training.engine import binary_metrics
    require(set(predictions) == {"identities", "targets", "logits"}
        and predictions["identities"] == [[r["src_subject_id"], r["side"]] for r in validation]
        and predictions["targets"] == [r["PROGRESION_48M"] for r in validation]
        and len(predictions["logits"]) == len(validation)
        and binary_metrics(predictions["targets"], predictions["logits"]) == job["metrics"],
        "exact_held_out_screening_predictions_and_metrics_required")
    recovery._validate_engine_metadata(engine, request, job, rows)
    return {"key": key, "relative_directory": directory.relative_to(root).as_posix(),
        "receipt_sha256": receipt_sha, "job_record_sha256": job["record_sha256"], "request_sha256": digest(request),
        "files_sha256": receipt["files_sha256"], "checkpoint_sizes": sizes, "metrics": job["metrics"],
        "validation_fold": fold, "model_id": request["model_id"]}, predictions


def _build(campaign40_dir, campaign41_dir, authority_path, plan, registry, rows):
    release.validate_plan(plan, registry)
    require(isinstance(rows, list) and rows, "loaded_development_manifest_rows_required")
    validate_rows(rows, "development")
    authority, authority_sha = _authority(authority_path)
    summary40, source40 = _source(campaign40_dir, "40", authority)
    summary41, source41 = _source(campaign41_dir, "41", authority)
    _same_source_scope(summary40, summary41)
    require(len(rows) == summary40["development_knees"]
        and len({r["src_subject_id"] for r in rows}) == summary40["development_participants"],
        "exact_screening_development_membership_required")
    package_sha = summary40["development_package_sha256"]
    all_jobs, model_results = [], {}
    reports = []
    from knee.aligned_training.engine import aggregate_fold_metrics, binary_metrics
    for number, summary, source in (("40", summary40, source40), ("41", summary41, source41)):
        source_root = checked_path(source["campaign_directory"])
        campaign = "densenet_frozen" if number == "40" else "remaining_encoders_frozen"
        expected = core.requests(campaign, package_sha, "0" * 64, plan, registry)
        require(set(summary.get("private_job_locations", {})) == {core.job_key(r) for r in expected}
            and {j["key"] for j in summary["job_results"]} == {core.job_key(r) for r in expected}
            and len(summary["job_results"]) == len(expected), "complete_unique_screening_jobs_required")
        groups = {}
        for request in expected:
            item, predictions = _job(source_root, summary, request, rows, plan, registry)
            item["source"] = number
            all_jobs.append(item)
            groups.setdefault(request["model_id"], []).append((item, predictions))
        require(set(summary["model_results"]) == {"clinical_main__" + model for model in groups},
            "screening_model_summary_coverage_required")
        for model_id, group in groups.items():
            metrics = [{"validation_fold": item["validation_fold"], **item["metrics"]} for item, _ in group]
            means = aggregate_fold_metrics(metrics)
            targets = [target for _, p in group for target in p["targets"]]
            logits = [logit for _, p in group for logit in p["logits"]]
            actual = {"fivefold_selection_metrics": means, "out_of_fold_predictions": len(targets),
                "pooled_oof_metrics_descriptive_only": binary_metrics(targets, logits)}
            require(len(targets) == len(rows) and actual == summary["model_results"]["clinical_main__" + model_id],
                "recomputed_screening_fold_and_oof_metrics_must_match_summary")
            model_results[model_id] = actual
            if model_id in plan["radiographic_models"]["conditional_challengers"]:
                reports.append({"model_id": model_id, "scope": "development_only",
                    "aggregation": "mean_of_five_participant_held_out_fold_metrics", "fold_metrics": metrics,
                    "eligibility_gates": {key: True for key in registry["eligibility_gates"]}})
    promoted = release.select_modern_challengers(reports, plan, registry)
    retained = plan["radiographic_models"]["mandatory"] + promoted
    require(set(model_results) == set(SCREENED_MODELS) and retained == authority["model_order"] == list(MODEL_ORDER)
        and set(plan["radiographic_models"]["conditional_challengers"]) - set(model_results) == set(EXCLUDED),
        "authorized_models_must_match_registered_screening_selection")
    return {"version": VERSION, "scope": "development_only", "authorized_stage": "stage_2",
        "training_plan_sha256": release.PINS[release.PLAN_PATH], "modeling_release_sha256": release.PINS[release.RELEASE_PATH],
        "development_package_sha256": package_sha, "model_input_reference_sha256": summary40["model_input_reference_sha256"],
        "development_rows_sha256": digest(rows), "development_knees": len(rows),
        "development_participants": len({r["src_subject_id"] for r in rows}),
        "authority_path": str(checked_path(authority_path)), "authority_sha256": authority_sha,
        "sources": {"40": source40, "41": source41}, "screening_jobs": all_jobs, "model_results": model_results,
        "promoted_modern_models": promoted, "retained_image_models": retained, "model_order": retained,
        "excluded_modern_models": EXCLUDED,
        "selected_image_encoder_id": None, "stage_one_weights_reused": False,
        "checkpoint_payloads_rehashed_for_selection": False, "future_weight_reuse_requires_full_payload_verification": True,
        "reserved_test_accessed": False, "fusion_authorized": False}


def prepare_selection(campaign40_dir, campaign41_dir, output_path, plan, registry, rows, *, authority_path=None):
    """Write the same deterministic private artifact on every identical rerun."""
    root40, root41, output = map(checked_path, (campaign40_dir, campaign41_dir, output_path))
    authority = checked_path(authority_path or root40.parent / AUTHORITY_FILENAME)
    repository = Path(__file__).resolve().parents[2]
    require(root40 != root41 and not root40.is_relative_to(root41) and not root41.is_relative_to(root40)
        and output.parent.is_dir() and not output.is_relative_to(repository)
        and not output.is_relative_to(root40) and not output.is_relative_to(root41)
        and not root40.is_relative_to(repository) and not root41.is_relative_to(repository)
        and not authority.is_relative_to(repository), "separate_private_screening_sources_and_selection_output_required")
    result = _build(root40, root41, authority, plan, registry, rows)
    if output.exists():
        saved, _ = recovery._small_json(output)
        require(saved == result, "existing_gradual_selection_evidence_changed")
    else:
        runner.save_json(output, result)
    return result


def validate_selection_evidence(path, request, plan, registry, rows=None):
    """Stage-two-only permission; every cached proof originated in full metadata validation."""
    release.validate_plan(plan, registry)
    validate_request(request, plan, registry)
    require(request.get("operation") == "development_fold" and request.get("stage") == "stage_2"
        and request.get("model_id") == request.get("image_encoder_id") in MODEL_ORDER
        and request.get("seed") == 2026 and request.get("clinical_scenario") == "clinical_main",
        "gradual_selection_only_authorizes_registered_stage_two_images")
    require(isinstance(rows, list) and rows, "loaded_development_manifest_rows_required")
    validate_rows(rows, "development")
    value, checksum = recovery._small_json(path, request["selection_evidence_sha256"])
    require(value.get("version") == VERSION and value.get("authorized_stage") == "stage_2"
        and value.get("development_package_sha256") == request["development_package_sha256"]
        and value.get("development_rows_sha256") == digest(rows)
        and value.get("training_plan_sha256") == release.PINS[release.PLAN_PATH]
        and value.get("modeling_release_sha256") == release.PINS[release.RELEASE_PATH],
        "gradual_selection_request_and_development_binding_required")
    authority, authority_sha = _authority(value["authority_path"])
    require(authority_sha == value.get("authority_sha256"), "gradual_selection_private_authority_changed")
    key = (checksum, authority_sha, digest(rows), digest(plan), digest(registry))
    if key not in _VALIDATED:
        verified = _build(value["sources"]["40"]["campaign_directory"], value["sources"]["41"]["campaign_directory"],
            value["authority_path"], plan, registry, rows)
        require(verified == value, "gradual_selection_artifact_must_equal_authenticated_sources")
        # Cache immutable serialized metadata. No patient arrays or tensor weights.
        result = {"promoted_modern_models": verified["promoted_modern_models"], "selected_image_encoder_id": None,
            "retained_image_models": verified["retained_image_models"], "selection_evidence_sha256": checksum,
            "stage_one_weights_reused": False, "reserved_test_accessed": False}
        require(len(_VALIDATED) < 8, "bounded_gradual_selection_process_cache_required")
        _VALIDATED[key] = json.dumps(result, sort_keys=True, allow_nan=False)
    result = json.loads(_VALIDATED[key])
    require(request["model_id"] in result["retained_image_models"], "stage_two_model_not_promoted_or_mandatory")
    return result
