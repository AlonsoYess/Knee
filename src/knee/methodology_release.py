"""Methodology alignment release. Preparation is not training authority.

Historical v1.1 jobs and notebooks retain their exact recipe. New comparisons
use v1.2 consistently; old identity-policy image jobs cannot silently enter a
gamma-policy model-selection campaign. No new architecture or cohort is added.
"""
from copy import deepcopy
from pathlib import Path

from knee.training_plan import _digest, _read_json, _require

VERSION = "registered_training_plan_v1.2"
REQUEST_VERSION = "development_training_request_v1.1"
PLAN_PATH = "configs/training_plan_v1_2.json"
RELEASE_PATH = "configs/modeling_release_v1_2.json"
AMENDMENT_PATH = "configs/governance/methodology_alignment_v1.json"
PINS = {'configs/training_plan_v1_2.json': 'a9644a979e4e0f4b1736c78531ea7ccbf8e2e1cfab684bc946334162eeb81b5b', 'configs/modeling_release_v1_2.json': '868744d3673e39656a640cc616d12e23b6540a7d4891a9815ac14d18294f7869', 'configs/governance/methodology_alignment_v1.json': '7ea6e0ff782cf67f907712cc030bd471c651b4ce95ce818c96716077ee0aee82'}


def validate_plan(plan, registry):
    from knee import final_model_release as previous
    _require(isinstance(plan, dict) and _digest(plan) == PINS[PLAN_PATH],
             "methodology_training_plan_changed")
    projected = deepcopy(plan)
    projected["version"] = previous.VERSION
    projected["governing_change"] = "MCR-2026-014"
    projected.pop("clinical_scenarios")
    projected.pop("comparability")
    projected["multimodal_models"].pop("complementary_enabled")
    projected["multimodal_models"].pop("late_fusion")
    projected["multimodal_models"]["complementary_disabled"] = ["fusion_late_fixed_mean"]
    projected["neural_training"]["augmentation"] = {
        "initial_policy": "identity", "geometric_or_intensity_transforms_enabled": False,
        "approval_status": "additional_augmentation_policy_requires_review_before_training"}
    projected["readiness"]["new_architectures_or_predictors"] = "two_explicitly_approved_image_encoders_only_no_new_predictors"
    projected["readiness"]["training_runner_implemented_by_this_plan"] = False
    previous.validate_plan(projected, registry)
    from knee.training_augmentation import validate_policy
    validate_policy(plan["neural_training"]["augmentation"])
    return {"version": VERSION, "status": "methodology_configuration_verified_no_training",
            "training_authorized": False, "reserved_test_access_authorized": False,
            "plan_canonical_sha256": PINS[PLAN_PATH], "registry_canonical_sha256": _digest(registry)}


def load_release(repository_root):
    from knee import final_model_release as previous
    _, catalog, _, registry = previous.load_release(repository_root)
    root = Path(repository_root).resolve()
    values = {}
    for name, expected in PINS.items():
        path = root / name
        _require(not path.is_symlink() and path.resolve().is_relative_to(root), "methodology_release_path_escape")
        value = _read_json(path)
        _require(_digest(value) == expected, "methodology_release_component_changed:" + name)
        values[name] = value
    validate_plan(values[PLAN_PATH], registry)
    return values[RELEASE_PATH], catalog, values[PLAN_PATH], registry


def recipe_sha256(plan):
    from knee import final_model_release as previous
    expected = PINS[PLAN_PATH] if plan.get("version") == VERSION else previous.PINS["configs/training_plan_v1_1.json"]
    _require(_digest(plan) == expected, "approved_training_recipe_required")
    return expected


def release_sha256(plan):
    from knee import final_model_release as previous
    recipe_sha256(plan)
    return PINS[RELEASE_PATH] if plan.get("version") == VERSION else previous.PINS[previous.RELEASE_PATH]


def select_modern_challengers(reports, plan, registry):
    """Reuse the unchanged historical ranking rule after validating v1.2."""
    from knee import final_model_release as previous
    validate_plan(plan, registry)
    historical = _read_json(Path(__file__).resolve().parents[2] / "configs/training_plan_v1_1.json")
    return previous.select_modern_challengers(reports, historical, registry)
