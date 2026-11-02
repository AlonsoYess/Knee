"""Documentary guards for the approved conditional ROI proposal, not an ROI validation."""

import copy
import json
import unittest
from pathlib import Path

from knee.change_control import proposal_is_authorized, validate_registry


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/governance/methodology_change_log.json"


class Phase1RoiRevisionProposalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        cls.proposal = next(
            p for p in cls.registry["proposals"] if p["id"] == "MCR-2026-004"
        )
        cls.document = (ROOT / cls.proposal["document"]).read_text(encoding="utf-8")

    def test_new_proposal_is_approved_with_v04_closure_verified(self):
        self.assertEqual(validate_registry(self.registry), [])
        self.assertEqual(self.proposal["status"], "aprobada_no_aplicada")
        self.assertEqual(self.proposal["approval"]["decision"], "aprobada")
        self.assertEqual(self.proposal["approval"]["date"], "2026-10-01")
        self.assertIn("OK DALE HAGAMOSLO", self.proposal["approval"]["evidence"])
        self.assertTrue(proposal_is_authorized(self.registry, self.proposal["id"]))
        prerequisite = self.proposal["prerequisite_review"]
        self.assertTrue(prerequisite["governance_contracts_updated"])
        self.assertTrue(prerequisite["integration_ready"])
        self.assertTrue(prerequisite["v04_operational_closure_verified"])
        self.assertEqual(prerequisite["v04_review_rows"], 20)
        self.assertEqual(prerequisite["v04_review_rows_with_all_six_answers_blank"], 0)
        self.assertTrue((ROOT / prerequisite["report"]).is_file())

    def test_previous_approval_is_not_rewritten_or_inherited(self):
        old = next(p for p in self.registry["proposals"] if p["id"] == "MCR-2026-003")
        self.assertEqual(old["status"], "retirada")
        self.assertFalse(proposal_is_authorized(self.registry, old["id"]))
        self.assertEqual(old["approval"]["date"], "2026-09-30")
        self.assertEqual(old["supersession"]["superseded_by"], self.proposal["id"])
        self.assertEqual(old["approval"]["decision"], "aprobada")
        self.assertEqual(old["proposed_validation"]["new_qualification_participants"], 60)
        self.assertFalse(old["prerequisite_review"]["integration_ready"])
        self.assertEqual(self.proposal["proposes_to_supersede"], old["id"])
        self.assertTrue(self.proposal["supersession_effective"])
        registry = copy.deepcopy(self.registry)
        new = next(p for p in registry["proposals"] if p["id"] == self.proposal["id"])
        new["status"] = "aprobada_no_aplicada"
        new["approval"]["decision"] = "pendiente"
        self.assertIn("MCR-2026-004 lacks an approved decision", validate_registry(registry))
        self.assertFalse(proposal_is_authorized(registry, new["id"]))

    def test_candidate_is_pinned_not_presented_as_verified(self):
        candidate = self.proposal["candidate"]
        self.assertEqual(candidate["revision"], "c70a2314bdbeed0d2fba3c2c2c652782ce3a2b93")
        self.assertIn(candidate["revision"], self.document)
        self.assertEqual(candidate["license"], "MIT")
        self.assertTrue(candidate["source_inspected"])
        self.assertTrue(candidate["runtime_verified"])
        for field in ("local_performance_verified",
                      "pretrained_weights_required", "local_training_or_fine_tuning_in_scope",
                      "silent_fallback_allowed"):
            self.assertFalse(candidate[field])

    def test_proposed_counts_and_limits_agree(self):
        validation = self.proposal["proposed_validation"]
        for count_key, minimum_key, warning_key in (
            ("historical_pilot_knees", "historical_minimum_acceptable_knees", "historical_maximum_peripheral_warnings"),
            ("new_confirmation_knees_expected", "confirmation_minimum_acceptable_knees", "confirmation_maximum_peripheral_warnings"),
        ):
            total = validation[count_key]
            self.assertEqual(validation[minimum_key] / total, validation["minimum_useful_coverage_fraction"])
            self.assertEqual(validation[warning_key] / total, validation["maximum_peripheral_warning_fraction"])
        self.assertEqual(validation["new_confirmation_knees_expected"], 2 * validation["new_confirmation_participants"])
        self.assertEqual(validation["maximum_incorrect_or_unevaluable_candidates"], 0)
        for literal in ("19/20", "38/40", "2/20", "4/40", "20 participantes", "semilla 2026"):
            self.assertIn(literal, self.document)

    def test_review_is_not_expert_annotation_or_a_population_guarantee(self):
        validation = self.proposal["proposed_validation"]
        self.assertTrue(validation["thresholds_are_proposed_not_clinically_validated"])
        for field in ("expert_landmark_annotations_required", "millimetric_accuracy_claims_allowed",
                      "population_95_percent_reliability_claim_allowed", "reinterpret_v04_results_allowed",
                      "replace_failed_subjects_allowed", "sample_selected"):
            self.assertFalse(validation[field])
        for fragment in ("no una validación radiológica", "La última no equivale a normalidad",
                         "no se convierte retrospectivamente en abstención", "No deriva de potencia estadística",
                         "Texto legible y anonimización siguen siendo defectos críticos"):
            self.assertIn(fragment, self.document)

    def test_no_execution_or_scope_expansion_recorded(self):
        self.assertFalse(self.registry["training_authorized"])
        self.assertFalse(self.proposal["test_block_consulted_before_decision"])
        self.assertTrue(self.proposal["proposed_validation"]["all_technical_subjects_reserved_for_development"])
        implementation = self.proposal["implementation"]
        for field in ("applied", "model_code_changed", "training_executed", "mass_processing_executed",
                      "partitions_created", "reserved_test_opened"):
            self.assertFalse(implementation[field])
        self.assertEqual(implementation["artifacts"], [])
        self.assertIsNone(implementation["verification"])

    def test_document_preserves_critical_limits_and_sources(self):
        for fragment in ("8/20 recortes aceptables", "no verificado según el acta 24",
                         "MCR-2026-001", "MCR-2026-002", "SRC-THESIS-002",
                         "140 mm", "NumPy 2.5.3", "Savitzky", "cv2.imwrite",
                         "La inconsistencia no está resuelta", "No se denominará revisión independiente",
                         "sin sobrescribir los Word originales"):
            self.assertIn(fragment, self.document)
        for heading in range(1, 11):
            self.assertIn(f"## {heading}.", self.document)
        for private_fragment in ("/content/drive/MyDrive/", "C:\\Users\\", "case_"):
            self.assertNotIn(private_fragment, self.document)


if __name__ == "__main__":
    unittest.main()
