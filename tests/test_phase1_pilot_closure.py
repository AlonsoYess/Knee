"""Public safeguards for the Phase 1 pilot closure."""

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLOSURE = ROOT / "docs" / "08_CIERRE_FASE_1_PASO_1.md"
PROTOCOL = ROOT / "docs" / "07_FASE_1_PASO_1.md"
DECISIONS = ROOT / "docs" / "DECISIONES.md"
CONTRACT = ROOT / "configs" / "governance" / "scope_contract.json"


class Phase1PilotClosureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.closure = CLOSURE.read_text(encoding="utf-8")
        cls.protocol = PROTOCOL.read_text(encoding="utf-8")
        cls.decisions = DECISIONS.read_text(encoding="utf-8")
        cls.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    def test_closure_identifies_verified_commit_and_result(self):
        commit = "f540a60f7399f7f97b4f196169c9f5109d017381"
        self.assertIn(commit, self.closure)
        self.assertIn(commit, self.protocol)
        self.assertIn("Estado | Cerrado", self.closure)
        self.assertIn("Adquisiciones únicas seleccionadas | 10", self.closure)

    def test_public_closure_contains_no_private_link_hash_or_acquisition_id(self):
        self.assertNotIn("drive.google.com", self.closure)
        self.assertIsNone(re.search(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", self.closure))
        self.assertIsNone(re.search(r"(?<!\d)\d{8}(?!\d)", self.closure))

    def test_closure_keeps_training_and_reserved_test_blocked(self):
        self.assertFalse(self.contract["training_authorized"])
        self.assertIn("no se entrenó ningún modelo", self.closure)
        self.assertIn("prueba reservada", self.closure)

    def test_decision_log_preserves_next_step(self):
        self.assertIn("Fase 1, paso 1 cerrado", self.decisions)
        self.assertIn("Fase 1, paso 2", self.closure)


if __name__ == "__main__":
    unittest.main()
