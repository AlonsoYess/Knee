"""Public closure checks for Phase 1, step 2."""

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLOSURE = ROOT / "docs" / "11_CIERRE_FASE_1_PASO_2.md"


class Phase1InventoryClosureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = CLOSURE.read_text(encoding="utf-8")

    def test_closure_identifies_final_contract_and_execution(self):
        self.assertIn("1,916", self.text)
        self.assertIn("2,778", self.text)
        self.assertIn("1,906", self.text)
        self.assertIn("8c3be777d3fb3edfbbb82ff4aabb100d59ef4143", self.text)
        self.assertIn("paso 2 de la Fase 1 queda cerrado", self.text)

    def test_closure_keeps_training_and_reserved_test_blocked(self):
        self.assertIn("No se entrenó ningún modelo", self.text)
        self.assertIn("no se abrió ni creó el conjunto de prueba reservado", self.text)
        self.assertIn("Fase 1, paso 3", self.text)

    def test_public_closure_contains_no_private_locations_or_identifiers(self):
        forbidden = (
            "drive.google.com",
            "C:\\\\Users",
            "/content/drive",
            "s3://",
            "NDA_USERNAME",
            "NDA_PACKAGE_ID",
        )
        for fragment in forbidden:
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, self.text)


if __name__ == "__main__":
    unittest.main()
