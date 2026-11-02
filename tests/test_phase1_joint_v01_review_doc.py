"""Public safeguards for the rejected v0.1 joint-localization review."""

import re
import unittest
from pathlib import Path


DOCUMENT = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "15_REVISION_FASE_1_PASO_4_V01.md"
)


class Phase1JointV01ReviewDocTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = DOCUMENT.read_text(encoding="utf-8")

    def test_document_records_the_complete_blinded_result(self):
        self.assertIn("Recortes íntegramente aceptables | 9", self.text)
        self.assertIn("Recortes rechazados | 11", self.text)
        self.assertIn("Líneas articulares no centradas | 9", self.text)
        self.assertIn("Recortes con regla o puntos periféricos | 7", self.text)
        self.assertIn("Exclusiones técnicas de DICOM | 0", self.text)
        self.assertIn("sin consultar el desenlace", self.text)

    def test_document_rejects_v01_and_keeps_downstream_gates_closed(self):
        self.assertIn("parámetros no se congelan", self.text)
        self.assertIn("no se procesarán masivamente", self.text)
        self.assertIn("no se entrenará", self.text)
        self.assertIn("no se abrirá la prueba reservada", self.text)

    def test_document_does_not_publish_private_identifiers(self):
        for fragment in ("drive.google.com", "C:\\Users", "/content/drive", "case_001"):
            self.assertNotIn(fragment, self.text)
        self.assertIsNone(re.search(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", self.text))


if __name__ == "__main__":
    unittest.main()
