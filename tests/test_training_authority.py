"""Pure future-execution authority checks; no datasets or training are opened."""
from copy import deepcopy
from pathlib import Path
import unittest

from knee import training_authority as authority
from knee.final_model_release import load_release, PINS, RELEASE_PATH

ROOT = Path(__file__).resolve().parents[1]


def request_for(model_id="image_densenet121", *, stage="stage_1", operation="development_fold"):
    image = model_id if model_id.startswith("image_") else (
        "image_densenet121" if model_id.startswith("fusion_") else None)
    return {"version": authority.REQUEST_VERSION, "operation": operation, "model_id": model_id,
        "image_encoder_id": image, "stage": stage, "validation_fold": 0, "seed": 2026,
        "development_package_sha256": "a" * 64, "image_micro_batch_size": 4,
        "selection_evidence_sha256": "b" * 64 if stage != "stage_1" else None,
        "fold_image_job_sha256": "c" * 64 if stage == "fusion" else None}


def authority_for(request, registry):
    gates = {key: True for key in registry["eligibility_gates"]}
    if request["operation"] == "single_fold_smoke":
        for key in registry["eligibility_gates"][4:]:
            gates[key] = False
    return {"version": authority.VERSION, "training_authorized": True,
        "reserved_test_access_authorized": False,
        "approval_source": "direct_investigator_authorization",
        "approval_statement": "Synthetic test fixture only; no training execution.",
        "approved_at_utc": "2026-10-22T16:00:00+00:00", "request_sha256": authority.digest(request),
        "modeling_release_sha256": PINS[RELEASE_PATH], "eligibility_gates": gates,
        "full_encoder_training_authorized": request["stage"] != "stage_1"}


class TrainingAuthorityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, _, cls.plan, cls.registry = load_release(ROOT)

    def issue(self, request=None, record=None):
        request = request_for() if request is None else request
        record = authority_for(request, self.registry) if record is None else record
        return authority.validate_authority(record, request, self.plan, self.registry)

    def test_request_schema_requires_exact_reviewable_fields(self):
        request = request_for()
        self.assertEqual(authority.validate_request(request, self.plan, self.registry), request)
        for kind in ("extra", "missing"):
            changed = deepcopy(request)
            if kind == "extra": changed["open_reserved_test"] = False
            else: changed.pop("development_package_sha256")
            with self.assertRaises(ValueError):
                authority.validate_request(changed, self.plan, self.registry)

    def test_separate_direct_approval_required_and_test_access_never_approved(self):
        request = request_for()
        with self.assertRaises(ValueError):
            self.issue(request, {})
        for key, value in (("training_authorized", False), ("reserved_test_access_authorized", True),
                           ("approval_source", "notebook34_preparation"), ("approval_statement", " ")):
            record = authority_for(request, self.registry)
            record[key] = value
            with self.assertRaises(ValueError):
                self.issue(request, record)

    def test_approval_digest_binds_model_seed_fold_microbatch_and_package(self):
        request = request_for()
        record = authority_for(request, self.registry)
        changes = ({"model_id": "image_vit_b16", "image_encoder_id": "image_vit_b16"},
            {"seed": 2027}, {"validation_fold": 1}, {"image_micro_batch_size": 2},
            {"development_package_sha256": "d" * 64})
        for mutation in changes:
            changed = {**request, **mutation}
            with self.assertRaisesRegex(ValueError, "match_exact_request"):
                self.issue(changed, record)

    def test_reserved_refit_calibration_and_unknown_operations_rejected(self):
        for operation in ("reserved_test", "evaluate_reserved", "final_refit", "calibrate", "train_all"):
            with self.assertRaises(ValueError):
                authority.validate_request(request_for(operation=operation), self.plan, self.registry)

    def test_only_fixed_folds_registered_seeds_and_explicit_microbatch_values(self):
        for key, value in (("validation_fold", -1), ("validation_fold", 5), ("validation_fold", True),
                           ("seed", 2025), ("seed", True), ("image_micro_batch_size", 8),
                           ("image_micro_batch_size", True), ("development_package_sha256", "bad")):
            request = request_for()
            request[key] = value
            with self.assertRaises(ValueError):
                authority.validate_request(request, self.plan, self.registry)

    def test_gated_unregistered_or_complementary_models_fail_closed(self):
        for model_id in ("image_dinov3_vits16", "image_added", "image_bilateral_cross_attention", "fusion_late_fixed_mean"):
            with self.assertRaises(ValueError):
                authority.validate_request(request_for(model_id), self.plan, self.registry)
        request = request_for("fusion_intermediate_concat", stage="fusion")
        request["image_encoder_id"] = "image_dinov3_vits16"
        with self.assertRaises(ValueError):
            authority.validate_request(request, self.plan, self.registry)

    def test_smoke_needs_first_four_gates_and_complete_training_needs_all(self):
        request = request_for(operation="single_fold_smoke")
        record = authority_for(request, self.registry)
        self.assertTrue(self.issue(request, record).smoke)
        for key in self.registry["eligibility_gates"][:4]:
            changed = deepcopy(record)
            changed["eligibility_gates"][key] = False
            with self.assertRaisesRegex(ValueError, "eligibility_incomplete"):
                self.issue(request, changed)
        request["operation"] = "development_fold"
        record["request_sha256"] = authority.digest(request)
        with self.assertRaisesRegex(ValueError, "eligibility_incomplete"):
            self.issue(request, record)

    def test_unfreezing_requires_separate_full_encoder_permission(self):
        request = request_for(stage="stage_2")
        record = authority_for(request, self.registry)
        record["full_encoder_training_authorized"] = False
        with self.assertRaisesRegex(ValueError, "separate_unfreezing"):
            self.issue(request, record)
        record["full_encoder_training_authorized"] = True
        self.assertTrue(self.issue(request, record).full_encoder_training_authorized)

    def test_conditional_selection_and_fold_job_hashes_exactly_required(self):
        for model_id, stage, required in (("image_densenet121", "stage_2", "selection_evidence_sha256"),
                ("fusion_intermediate_concat", "fusion", "selection_evidence_sha256"),
                ("fusion_intermediate_gated_film", "fusion", "fold_image_job_sha256")):
            request = request_for(model_id, stage=stage)
            authority.validate_request(request, self.plan, self.registry)
            for value in (None, "bad"):
                changed = deepcopy(request)
                changed[required] = value
                with self.assertRaisesRegex(ValueError, "conditional_training_evidence"):
                    authority.validate_request(changed, self.plan, self.registry)
        request = request_for()
        request["selection_evidence_sha256"] = "b" * 64
        with self.assertRaises(ValueError):
            authority.validate_request(request, self.plan, self.registry)

    def test_issued_permit_has_all_narrow_bindings_and_scope_cannot_cross(self):
        request = request_for(operation="single_fold_smoke")
        permit = self.issue(request)
        self.assertEqual((permit.package_sha256, permit.validation_fold, permit.seed, permit.image_micro_batch_size,
            permit.image_encoder_id), ("a" * 64, 0, 2026, 4, "image_densenet121"))
        self.assertEqual(permit.request_sha256, authority.digest(request))
        authority.require_permit(permit, model_id=permit.model_id, stage=permit.stage, smoke=True)
        for values in (("image_vit_b16", "stage_1", True), (permit.model_id, "stage_2", True),
                       (permit.model_id, "stage_1", False)):
            with self.assertRaises(ValueError):
                authority.require_permit(permit, model_id=values[0], stage=values[1], smoke=values[2])

    def test_unsealed_or_absent_permit_cannot_authorize_execution(self):
        permit = authority.TrainingPermit("a" * 64, "image_densenet121", "stage_1", False,
            "b" * 64, 0, 2026, "image_densenet121", False, 4)
        for value in (None, {}, permit):
            with self.assertRaisesRegex(ValueError, "explicit_training_permit"):
                authority.require_permit(value, model_id="image_densenet121", stage="stage_1", smoke=False)

    def test_release_timestamp_and_boolean_gate_records_are_explicit(self):
        request = request_for()
        for key, value in (("modeling_release_sha256", "f" * 64),
                           ("approved_at_utc", "2026-10-22T16:00:00"),
                           ("approved_at_utc", "bad")):
            record = authority_for(request, self.registry)
            record[key] = value
            with self.assertRaises(ValueError):
                self.issue(request, record)
        record = authority_for(request, self.registry)
        record["eligibility_gates"][self.registry["eligibility_gates"][0]] = 1
        with self.assertRaises(ValueError):
            self.issue(request, record)


if __name__ == "__main__":
    unittest.main()
