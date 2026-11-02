"""Contract tests for the public v0.4 blinded-review record."""

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = ROOT / "docs" / "22_REVISION_FASE_1_PASO_4_V04.md"


class Phase1JointV04ReviewDocumentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = DOCUMENT.read_text(encoding="utf-8")

    def test_document_records_the_aggregate_rejection(self):
        for fragment in (
            "8 recortes íntegramente aceptables",
            "12 recortes rechazados",
            "9 líneas articulares no centradas",
            "0 recortes con anatomía tibiofemoral incompleta",
            "6 recortes con puntos de la regla",
            "0 exclusiones técnicas",
        ):
            self.assertIn(fragment, self.text)

    def test_document_preserves_governance_and_blockers(self):
        self.assertIn("parámetros no congelados", self.text)
        self.assertIn("paso 4 abierto", self.text)
        for fragment in (
            "procesamiento masivo",
            "creación de particiones",
            "entrenamiento",
            "prueba reservada",
            "cambio metodológico",
        ):
            self.assertIn(fragment, self.text)

    def test_document_contains_no_private_path_or_case_alias(self):
        self.assertNotIn("/content/drive/MyDrive/", self.text)
        self.assertNotIn("C:\\Users\\", self.text)
        self.assertNotIn("case_", self.text)


if __name__ == "__main__":
    unittest.main()
