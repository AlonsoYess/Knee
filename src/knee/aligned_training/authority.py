"""Exact v1.2 development authority; input preparation grants no training."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import re

from knee.methodology_release import validate_plan, PINS, RELEASE_PATH

VERSION = "development_training_authority_v1.0"
REQUEST_VERSION = "development_training_request_v1.0"
_SEAL = object()
REQUEST_FIELDS = {"version", "operation", "model_id", "image_encoder_id", "stage",
    "validation_fold", "seed", "development_package_sha256", "image_micro_batch_size",
    "selection_evidence_sha256", "fold_image_job_sha256"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_request(request, plan, registry):
    validate_plan(plan, registry)
    from knee import methodology_release as aligned
    new_recipe = plan.get("version") == aligned.VERSION
    extras = {"clinical_scenario", "baseline_sidecar_sha256", "augmentation_policy", "modeling_release_sha256"}
    require(isinstance(request, dict) and set(request) == (REQUEST_FIELDS | extras if new_recipe else REQUEST_FIELDS)
        and request["version"] == (aligned.REQUEST_VERSION if new_recipe else REQUEST_VERSION), "exact_training_request_required")
    if new_recipe:
        from knee.training_augmentation import VERSION as AUGMENTATION_VERSION
        scenario = request["clinical_scenario"]
        require(scenario in plan["clinical_scenarios"]
            and (request["model_id"].startswith("clinical_") or scenario == "clinical_main"),
            "registered_clinical_scenario_required")
        baseline_hash = request["baseline_sidecar_sha256"]
        require((scenario == "clinical_main" and baseline_hash is None)
            or (scenario != "clinical_main" and isinstance(baseline_hash, str)
                and re.fullmatch(r"[0-9a-f]{64}", baseline_hash) is not None),
            "exact_complementary_baseline_sidecar_hash_required")
        require(request["modeling_release_sha256"] == aligned.PINS[aligned.RELEASE_PATH],
            "aligned_request_release_mismatch")
        expected_policy = "identity" if request["image_encoder_id"] is None else AUGMENTATION_VERSION
        require(request["augmentation_policy"] == expected_policy, "exact_registered_augmentation_policy_required")
        # A conventional fit has no shared two-update smoke boundary. Do not
        # label an entire logistic/XGBoost fit as a bounded neural smoke run.
        require(request["operation"] != "single_fold_smoke"
            or request["model_id"] not in {"clinical_logistic", "clinical_xgboost"},
            "bounded_smoke_requires_neural_model")
    clinical = {v["id"] for v in registry["clinical"]["candidates"]}
    image = {v["id"] for key in ("mandatory_baselines", "modern_challengers")
        for v in registry["radiographic"][key]}
    fusion = set(plan["multimodal_models"]["primary"])
    model = request["model_id"]
    require(model in clinical | image | fusion and model != "image_dinov3_vits16",
        "implemented_accessible_registered_model_required")
    require(request["operation"] in {"single_fold_smoke", "development_fold"}
        and request["stage"] in {"stage_1", "stage_2", "fusion"}, "development_training_stage_required")
    require(type(request["validation_fold"]) is int and request["validation_fold"] in range(5)
        and type(request["seed"]) is int and request["seed"] in plan["selection"]["neural_finalist_seeds"],
        "fixed_development_fold_and_registered_seed_required")
    require(isinstance(request["development_package_sha256"], str)
        and re.fullmatch(r"[0-9a-f]{64}", request["development_package_sha256"]) is not None,
        "sealed_development_package_hash_required")
    require(type(request["image_micro_batch_size"]) is int
        and request["image_micro_batch_size"] in {1, 2, 4}, "registered_micro_batch_required")
    require((model in clinical and request["stage"] == "stage_1" and request["image_encoder_id"] is None)
        or (model in image and request["stage"] in {"stage_1", "stage_2"}
            and request["image_encoder_id"] == model)
        or (model in fusion and request["stage"] == "fusion"
            and request["image_encoder_id"] in image - {"image_dinov3_vits16"}),
        "registered_model_stage_and_encoder_pair_required")
    require(request["operation"] != "single_fold_smoke" or request["stage"] == "stage_1",
        "initial_smoke_requires_stage_one")
    for key, required in (("selection_evidence_sha256", request["stage"] != "stage_1"),
                          ("fold_image_job_sha256", request["stage"] == "fusion")):
        value = request[key]
        require((not required and value is None) or (required and isinstance(value, str)
            and re.fullmatch(r"[0-9a-f]{64}", value) is not None), "exact_conditional_training_evidence_hash_required")
    return request


@dataclass(frozen=True)
class TrainingPermit:
    request_sha256: str
    model_id: str
    stage: str
    smoke: bool
    package_sha256: str
    validation_fold: int
    seed: int
    image_encoder_id: str | None
    full_encoder_training_authorized: bool
    image_micro_batch_size: int
    _seal: object = field(default=None, repr=False, compare=False)
    plan_sha256: str | None = None
    clinical_scenario: str = "clinical_main"
    baseline_sidecar_sha256: str | None = None
    augmentation_policy: str = "identity"


def validate_authority(record, request, plan, registry):
    """Issue a narrow permit only from a separately approved exact request."""
    validate_request(request, plan, registry)
    fields = {"version", "training_authorized", "reserved_test_access_authorized",
        "approval_source", "approval_statement", "approved_at_utc", "request_sha256",
        "modeling_release_sha256", "eligibility_gates", "full_encoder_training_authorized"}
    require(isinstance(record, dict) and set(record) == fields and record["version"] == VERSION,
        "separate_exact_training_authority_required")
    require(record["training_authorized"] is True and record["reserved_test_access_authorized"] is False
        and record["approval_source"] == "direct_investigator_authorization"
        and isinstance(record["approval_statement"], str) and bool(record["approval_statement"].strip()),
        "explicit_investigator_training_authority_required")
    try:
        timestamp = datetime.fromisoformat(record["approved_at_utc"])
        require(timestamp.utcoffset() is not None, "timezone_aware_training_approval_required")
    except (ValueError, TypeError):
        raise ValueError("valid_training_approval_timestamp_required") from None
    from knee.methodology_release import release_sha256, recipe_sha256
    require(record["request_sha256"] == digest(request)
        and record["modeling_release_sha256"] == release_sha256(plan),
        "training_authority_must_match_exact_request_and_release")
    gates = record["eligibility_gates"]
    require(isinstance(gates, dict) and set(gates) == set(registry["eligibility_gates"])
        and all(type(value) is bool for value in gates.values())
        and type(record["full_encoder_training_authorized"]) is bool, "explicit_training_eligibility_gates_required")
    smoke = request["operation"] == "single_fold_smoke"
    if request["image_encoder_id"] is not None:
        needed = registry["eligibility_gates"][:4] if smoke else registry["eligibility_gates"]
        require(all(gates[key] for key in needed), "image_training_eligibility_incomplete")
    require(request["stage"] == "stage_1" or record["full_encoder_training_authorized"] is True,
        "separate_unfreezing_review_required")
    return TrainingPermit(digest(request), request["model_id"], request["stage"], smoke,
        request["development_package_sha256"], request["validation_fold"], request["seed"],
        request["image_encoder_id"], record["full_encoder_training_authorized"],
        request["image_micro_batch_size"], _SEAL,
        plan_sha256=recipe_sha256(plan), clinical_scenario=request.get("clinical_scenario", "clinical_main"),
        baseline_sidecar_sha256=request.get("baseline_sidecar_sha256"),
        augmentation_policy=request.get("augmentation_policy", "identity"))


def require_permit(permit, *, model_id, stage, smoke):
    require(isinstance(permit, TrainingPermit) and permit._seal is _SEAL
        and permit.model_id == model_id and permit.stage == stage and permit.smoke is smoke,
        "explicit_training_permit_required_before_execution")
