"""Public closure checks for Phase 1, step 3."""

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLOSURE = ROOT / "docs" / "13_CIERRE_FASE_1_PASO_3.md"


class Phase1BilateralClosureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = CLOSURE.read_text(encoding="utf-8")

    def test_closure_identifies_verified_execution(self):
        self.assertIn("2489a7ec6253fb4aaa12818416f249780d26ea24", self.text)
        self.assertIn("bilateral_split_v0.2_pilot", self.text)
        self.assertIn("El paso 3 de la Fase 1 queda cerrado técnicamente", self.text)
        self.assertIn("Separaciones aceptadas | 10", self.text)
        self.assertIn("Lateralidades respaldadas | 10", self.text)
        self.assertIn("Exclusiones técnicas | 0", self.text)

    def test_closure_preserves_safety_gates_and_next_step(self):
        self.assertIn("No se ejecutó procesamiento masivo", self.text)
        self.assertIn("No se entrenó ningún modelo", self.text)
        self.assertIn("No se abrió ni creó el conjunto de prueba reservado", self.text)
        self.assertIn("Fase 1, paso 4", self.text)

    def test_public_closure_contains_no_private_locations_or_hashes(self):
        forbidden = (
            "drive.google.com",
            "C:\\Users",
            "/content/drive",
            "s3://",
            "NDA_USERNAME",
            "NDA_PACKAGE_ID",
        )
        for fragment in forbidden:
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, self.text)
        self.assertIsNone(re.search(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", self.text))


if __name__ == "__main__":
    unittest.main()
