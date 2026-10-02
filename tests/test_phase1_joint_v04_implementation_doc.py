"""Contract tests for the prepared v0.4 localization implementation."""

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = ROOT / "docs" / "21_IMPLEMENTACION_FASE_1_PASO_4_V04.md"


class Phase1JointV04ImplementationDocumentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = DOCUMENT.read_text(encoding="utf-8")

    def test_document_records_implementation_and_pending_execution(self):
        self.assertIn("Implementación preparada y no ejecutada", self.text)
        self.assertIn("autorizó implementar `v0.4` y publicar", self.text)
        self.assertIn("libreta 11", self.text)
        self.assertIn("revisión visual ciega", self.text)

    def test_document_describes_the_bounded_v04_controls(self):
        for fragment in (
            "consenso vertical entre familias de señal",
            "soporte óseo bilateral",
            "selección horizontal gobernada por anatomía",
            "artefactos como puertas de rechazo",
            "REVIEW_REQUIRED_NO_FEASIBLE_CROP",
        ):
            self.assertIn(fragment, self.text)

    def test_document_preserves_downstream_blockers(self):
        for fragment in (
            "mass_processing_executed",
            "partitions_created",
            "training_executed",
            "reserved_test_opened",
        ):
            self.assertIn(fragment, self.text)

    def test_document_contains_no_private_path_or_identifier(self):
        self.assertNotIn("/content/drive/MyDrive/", self.text)
        self.assertNotIn("C:\\Users\\", self.text)
        self.assertNotIn("case_", self.text)


if __name__ == "__main__":
    unittest.main()
