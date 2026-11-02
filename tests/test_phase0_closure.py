"""Public closure safeguards for Phase 0."""

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLOSURE = ROOT / "docs" / "05_CIERRE_FASE_0.md"
CONTRACT = ROOT / "configs" / "governance" / "scope_contract.json"


class Phase0ClosureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = CLOSURE.read_text(encoding="utf-8")
        cls.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    def test_closure_identifies_the_executed_public_revision(self):
        self.assertIn(
            "00255b76acb8f6151a839b0fe8d223c217957f68",
            self.text,
        )
        self.assertIn("Cerrada técnicamente", self.text)

    def test_public_closure_contains_no_private_drive_link_or_source_hash(self):
        self.assertNotIn("drive.google.com", self.text)
        self.assertIsNone(re.search(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", self.text))

    def test_closure_keeps_training_and_reserved_test_blocked(self):
        self.assertFalse(self.contract["training_authorized"])
        self.assertIn("No acredita todavía", self.text)
        self.assertIn("partición de prueba reservada", self.text)


if __name__ == "__main__":
    unittest.main()
