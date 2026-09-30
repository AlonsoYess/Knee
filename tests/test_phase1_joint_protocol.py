"""Public safeguards for the Phase 1 joint-localization protocol."""

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs" / "14_FASE_1_PASO_4.md"


class Phase1JointProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = PROTOCOL.read_text(encoding="utf-8")

    def test_protocol_defines_pilot_and_physical_field(self):
        self.assertIn("diez estudios", self.text)
        self.assertIn("veinte", self.text)
        self.assertIn("140 × 140 mm", self.text)
        self.assertIn("ImagerPixelSpacing", self.text)
        self.assertIn("PixelSpacing", self.text)

    def test_protocol_keeps_mass_processing_and_training_blocked(self):
        self.assertIn("no se procesó masivamente", self.text)
        self.assertIn("no se crearon particiones", self.text)
        self.assertIn("no se entrenó", self.text)
        self.assertIn("no se abrió la prueba reservada", self.text)

    def test_protocol_contains_no_private_locations_or_hashes(self):
        forbidden = ("drive.google.com", "C:\\Users", "/content/drive", "s3://")
        for fragment in forbidden:
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, self.text)
        self.assertIsNone(re.search(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", self.text))


if __name__ == "__main__":
    unittest.main()
