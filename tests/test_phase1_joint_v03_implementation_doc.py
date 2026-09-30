"""Contract tests for the prepared v0.3 implementation record."""

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = ROOT / "docs" / "18_IMPLEMENTACION_FASE_1_PASO_4_V03.md"
CONFIG = ROOT / "configs" / "joint_localization.v0.3.example.json"
NOTEBOOK = ROOT / "notebooks" / "09_repeticion_localizacion_tibiofemoral_v03.ipynb"


class Phase1JointV03ImplementationDocumentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = DOCUMENT.read_text(encoding="utf-8")

    def test_record_distinguishes_preparation_from_private_execution(self):
        self.assertIn("Implementación local preparada", self.text)
        self.assertIn("piloto privado aún no ejecutado", self.text)
        self.assertIn("no se ha realizado", self.text)

    def test_record_links_the_public_config_and_notebook(self):
        self.assertTrue(CONFIG.is_file())
        self.assertTrue(NOTEBOOK.is_file())
        self.assertIn("joint_localization.v0.3.example.json", self.text)
        self.assertIn("09_repeticion_localizacion_tibiofemoral_v03.ipynb", self.text)

    def test_record_preserves_downstream_blockers(self):
        for fragment in (
            "mass_processing_executed",
            "partitions_created",
            "training_executed",
            "reserved_test_opened",
        ):
            self.assertIn(fragment, self.text)
        self.assertIn("permanecen en `False`", self.text)

    def test_record_contains_no_private_drive_path(self):
        self.assertNotIn("/content/drive/MyDrive/", self.text)
        self.assertNotIn("C:\\Users\\", self.text)


if __name__ == "__main__":
    unittest.main()
