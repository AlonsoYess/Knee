"""Tests for the methodological change-control gate."""

import copy
import json
import unittest
from pathlib import Path

from knee.change_control import proposal_is_authorized, validate_registry


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs" / "governance" / "methodology_change_log.json"
TEMPLATE_PATH = ROOT / "docs" / "templates" / "SOLICITUD_CAMBIO_METODOLOGICO.md"


class MethodologyChangeControlTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))

    def test_registry_is_valid_and_prior_decisions_cannot_be_reused(self):
        self.assertEqual(validate_registry(self.registry), [])
        self.assertEqual(self.registry["registry_version"], "1.6")
        self.assertEqual(self.registry["status"], "approved_control_mechanism")
        self.assertEqual(
            self.registry["mechanism_approval"]["approved_by"],
            "researcher",
        )
        self.assertFalse(
            proposal_is_authorized(self.registry, "MCR-2026-001"),
            "A verified historical decision must not authorize another implementation.",
        )
        self.assertFalse(
            proposal_is_authorized(self.registry, "MCR-2026-002"),
            "A verified prospective change must not authorize another implementation.",
        )
        self.assertFalse(self.registry["training_authorized"])

    def test_approved_not_applied_is_the_only_authorization_state(self):
        registry = copy.deepcopy(self.registry)
        proposal = registry["proposals"][0]
        proposal["id"] = "MCR-2026-999"
        proposal["record_kind"] = "prospective"
        proposal["status"] = "aprobada_no_aplicada"
        proposal["implementation"] = {"applied": False}
        self.assertEqual(validate_registry(registry), [])
        self.assertTrue(proposal_is_authorized(registry, "MCR-2026-999"))

    def test_localization_has_explicit_conditional_approval_not_implementation(self):
        proposal = next(
            item for item in self.registry["proposals"]
            if item["id"] == "MCR-2026-004"
        )
        self.assertEqual(proposal["status"], "aprobada_no_aplicada")
        self.assertEqual(proposal["approval"]["decision"], "aprobada")
        self.assertEqual(proposal["approval"]["approved_by"], "researcher")
        self.assertEqual(proposal["approval"]["date"], "2026-10-01")
        self.assertTrue(proposal["approval"]["approved_scope"])
        self.assertTrue(proposal["approval"]["conditions"])
        self.assertFalse(proposal["implementation"]["applied"])
        self.assertTrue(proposal_is_authorized(self.registry, proposal["id"]))
        self.assertTrue(proposal["prerequisite_review"]["integration_ready"])

    def test_promoting_localization_status_without_approval_is_blocked(self):
        registry = copy.deepcopy(self.registry)
        proposal = next(
            item for item in registry["proposals"]
            if item["id"] == "MCR-2026-004"
        )
        proposal["status"] = "aprobada_no_aplicada"
        proposal["approval"]["decision"] = "pendiente"
        self.assertIn(
            "MCR-2026-004 lacks an approved decision", validate_registry(registry)
        )
        self.assertFalse(proposal_is_authorized(registry, proposal["id"]))

    def test_localization_cannot_reuse_approval_after_application(self):
        registry = copy.deepcopy(self.registry)
        proposal = next(p for p in registry["proposals"] if p["id"] == "MCR-2026-004")
        proposal["status"] = "implementada"
        proposal["implementation"].update(
            applied=True, applied_at="2099-01-01", artifacts=["synthetic-test-only"]
        )
        self.assertEqual(validate_registry(registry), [])
        self.assertFalse(proposal_is_authorized(registry, proposal["id"]))

    def test_missing_researcher_approval_blocks_proposal(self):
        registry = copy.deepcopy(self.registry)
        proposal = registry["proposals"][0]
        proposal["status"] = "aprobada_no_aplicada"
        proposal["implementation"] = {"applied": False}
        proposal["approval"]["approved_by"] = "technical_team"
        errors = validate_registry(registry)
        self.assertIn("MCR-2026-001 lacks researcher approval", errors)
        self.assertFalse(proposal_is_authorized(registry, "MCR-2026-001"))

    def test_test_block_consultation_blocks_proposal(self):
        registry = copy.deepcopy(self.registry)
        proposal = registry["proposals"][0]
        proposal["status"] = "aprobada_no_aplicada"
        proposal["implementation"] = {"applied": False}
        proposal["test_block_consulted_before_decision"] = True
        errors = validate_registry(registry)
        self.assertIn(
            "MCR-2026-001 consulted the test block and is blocked",
            errors,
        )
        self.assertFalse(proposal_is_authorized(registry, "MCR-2026-001"))

    def test_template_contains_all_review_sections(self):
        template = TEMPLATE_PATH.read_text(encoding="utf-8")
        for heading in (
            "## 2. Problema observado",
            "## 3. Evidencia",
            "## 5. Alternativas evaluadas",
            "## 6. Impacto académico y metodológico",
            "## 7. Riesgos y recursos",
            "## 9. Decisión del investigador",
            "## 10. Cierre posterior",
        ):
            self.assertIn(heading, template)


if __name__ == "__main__":
    unittest.main()
