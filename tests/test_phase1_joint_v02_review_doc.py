"""Public safeguards for the rejected v0.2 joint-localization review."""

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "docs" / "16_REVISION_FASE_1_PASO_4_V02.md"
PROPOSAL = ROOT / "docs" / "17_PROPUESTA_FASE_1_PASO_4_V03.md"


class Phase1JointV02ReviewDocTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.review = REVIEW.read_text(encoding="utf-8")
        cls.proposal = PROPOSAL.read_text(encoding="utf-8")

    def test_review_records_the_complete_blinded_result(self):
        self.assertIn("Recortes íntegramente aceptables | 8", self.review)
        self.assertIn("Recortes rechazados | 12", self.review)
        self.assertIn("Líneas articulares no centradas | 8", self.review)
        self.assertIn("Recortes con regla o puntos periféricos | 5", self.review)
        self.assertIn("Anatomías tibiofemorales completas | 20", self.review)
        self.assertIn("Exclusiones técnicas de DICOM | 0", self.review)
        self.assertIn("sin consultar el desenlace", self.review)

    def test_review_rejects_v02_and_keeps_downstream_gates_closed(self):
        self.assertIn("queda **rechazado después de revisión visual ciega**", self.review)
        self.assertIn("parámetros no se congelan", self.review)
        self.assertIn("no se procesarán masivamente", self.review)
        self.assertIn("no se entrenará", self.review)
        self.assertIn("no se abrirá la prueba reservada", self.review)

    def test_proposal_is_explicitly_not_implemented(self):
        self.assertIn("no implementada ni autorizada para ejecución", self.proposal)
        self.assertIn("gradiente **con signo**", self.proposal)
        self.assertIn("REVIEW_REQUIRED_VERTICAL_DISAGREEMENT", self.proposal)
        self.assertIn("REVIEW_REQUIRED_ARTIFACT_CLEARANCE", self.proposal)
        self.assertIn("mismas veinte rodillas", self.proposal)

    def test_public_documents_do_not_publish_private_identifiers(self):
        combined = self.review + self.proposal
        for fragment in (
            "drive.google.com",
            "C:\\Users",
            "/content/drive",
            "case_001",
        ):
            self.assertNotIn(fragment, combined)
        self.assertIsNone(re.search(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", combined))


if __name__ == "__main__":
    unittest.main()

