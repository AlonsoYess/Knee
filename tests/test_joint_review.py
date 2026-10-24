"""Tests for blinded review validation of tibiofemoral crops."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from knee.joint_localization import REVIEW_FIELDS
from knee.joint_review import summarize_blinded_joint_review


def write_review(path: Path, accepted: int) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS, delimiter=";")
        writer.writeheader()
        for index in range(4):
            value = "SI" if index < accepted else "NO"
            writer.writerow(
                {
                    "knee_alias": f"case_{index // 2 + 1:03d}_{'RIGHT' if index % 2 == 0 else 'LEFT'}",
                    "case_alias": f"case_{index // 2 + 1:03d}",
                    "patient_side": "RIGHT" if index % 2 == 0 else "LEFT",
                    "preview_file": f"previews/case_{index:03d}.png",
                    "automatic_confidence_level": "HIGH",
                    "automatic_joint_center_y_fraction": "0.5",
                    "automatic_crop_height_mm": "140",
                    "automatic_crop_width_mm": "140",
                    "joint_centered_yes_no": value,
                    "tibiofemoral_anatomy_complete_yes_no": "SI",
                    "text_borders_and_rule_excluded_yes_no": "SI",
                    "crop_acceptable_yes_no": value,
                    "technical_exclusion_yes_no": "NO",
                    "exclusion_reason": "",
                    "reviewed_without_outcome_yes_no": "SI",
                    "reviewer_notes": "revisión ciega",
                }
            )


class JointReviewTest(unittest.TestCase):
    def prepare(self, root: Path, accepted: int) -> None:
        (root / "resumen_localizacion_publico.json").write_text(
            json.dumps(
                {
                    "status": "ready_for_blinded_review",
                    "algorithm_version": "test_v0.1",
                }
            ),
            encoding="utf-8",
        )
        (root / "parametros_candidatos.json").write_text(
            json.dumps(
                {
                    "status": "candidate_not_frozen_until_blinded_visual_review",
                    "algorithm_version": "test_v0.1",
                }
            ),
            encoding="utf-8",
        )
        write_review(root / "revision_visual_ciega.csv", accepted)

    def test_rejected_review_is_complete_but_does_not_freeze_parameters(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.prepare(root, accepted=2)

            result = summarize_blinded_joint_review(root, "test_v0.1", 4)

            self.assertEqual(result["status"], "rejected_after_blinded_review")
            self.assertEqual(result["acceptable_crops"], 2)
            self.assertEqual(result["rejected_crops"], 2)
            self.assertFalse(result["parameters_frozen"])
            self.assertFalse(result["mass_processing_executed"])

    def test_fully_accepted_review_is_reported_without_freezing_here(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.prepare(root, accepted=4)

            result = summarize_blinded_joint_review(root, "test_v0.1", 4)

            self.assertEqual(result["status"], "accepted_after_blinded_review")
            self.assertEqual(result["rejected_crops"], 0)
            self.assertFalse(result["parameters_frozen"])


if __name__ == "__main__":
    unittest.main()
