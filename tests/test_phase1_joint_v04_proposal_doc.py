"""Contract tests for the bounded v0.4 localization proposal."""

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = ROOT / "docs" / "20_PROPUESTA_FASE_1_PASO_4_V04.md"


class Phase1JointV04ProposalDocumentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = DOCUMENT.read_text(encoding="utf-8")

    def test_proposal_is_not_implemented_or_authorized(self):
        self.assertIn("no implementada ni autorizada", self.text)
        self.assertIn("Autorización requerida", self.text)
        self.assertNotIn("Implementación preparada", self.text)

    def test_proposal_decouples_anatomy_and_artifact_handling(self):
        self.assertIn("gobernada primero por anatomía", self.text)
        self.assertIn("puertas de rechazo", self.text)
        self.assertIn("no desplazamiento anatómico", self.text)
        self.assertIn("REVIEW_REQUIRED_NO_FEASIBLE_CROP", self.text)

    def test_proposal_requires_cross_family_vertical_consensus(self):
        self.assertIn("máximo multiseñal por compartimento de `v0.2`", self.text)
        self.assertIn("par de bordes dirigidos de `v0.3`", self.text)
        self.assertIn("REVIEW_REQUIRED_VERTICAL_DISAGREEMENT", self.text)
        self.assertIn("no se promedian candidatos incompatibles", self.text)

    def test_proposal_limits_repeated_pilot_tuning(self):
        self.assertIn("última iteración determinista", self.text)
        self.assertIn("no se preparará automáticamente una `v0.5`", self.text)
        self.assertIn("propuesta metodológica separada", self.text)

    def test_proposal_preserves_downstream_blockers(self):
        for fragment in (
            "Procesamiento masivo",
            "particiones",
            "entrenamiento",
            "prueba reservada",
        ):
            self.assertIn(fragment, self.text)

    def test_proposal_contains_no_private_path_or_identifier(self):
        self.assertNotIn("/content/drive/MyDrive/", self.text)
        self.assertNotIn("C:\\Users\\", self.text)
        self.assertNotIn("case_", self.text)


if __name__ == "__main__":
    unittest.main()
