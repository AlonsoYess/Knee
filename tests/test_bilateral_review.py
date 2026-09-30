"""Tests for blinded bilateral-review validation and parameter freezing."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from knee.bilateral_review import finalize_bilateral_review


REVIEW_FIELDS = (
    "case_alias",
    "preview_file",
    "automatic_confidence_level",
    "automatic_split_fraction",
    "proposed_image_left_side",
    "proposed_image_right_side",
    "split_acceptable_yes_no",
    "laterality_mapping_supported_yes_no",
    "observed_marker_or_anatomy",
    "technical_exclusion_yes_no",
    "exclusion_reason",
    "reviewed_without_outcome_yes_no",
    "reviewer_notes",
)

RESULT_FIELDS = (
    "case_alias",
    "split_fraction",
    "confidence_level",
    "technical_status",
    "proposed_image_left_side",
    "proposed_image_right_side",
    "preview_file",
    "error_code",
    "manifest_key",
)


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


class BilateralReviewTest(unittest.TestCase):
    def prepare(self, root: Path) -> None:
        (root / "resumen_separacion_publico.json").write_text(
            json.dumps(
                {
                    "status": "ready_for_blinded_review",
                    "algorithm_version": "bilateral_split_v0.2_pilot",
                    "processed_unique_studies": 2,
                    "technical_failures": 0,
                    "outcome_data_loaded": False,
                    "mass_processing_executed": False,
                    "training_executed": False,
                    "reserved_test_opened": False,
                }
            ),
            encoding="utf-8",
        )
        (root / "parametros_candidatos.json").write_text(
            json.dumps(
                {
                    "algorithm_version": "bilateral_split_v0.2_pilot",
                    "status": "candidate_not_frozen_until_visual_review",
                    "proposed_mapping": {"image_left": "RIGHT", "image_right": "LEFT"},
                }
            ),
            encoding="utf-8",
        )
        (root / "registro_preparacion.json").write_text(
            json.dumps({"status": "ready_for_blinded_review", "git_commit": "prepare"}),
            encoding="utf-8",
        )
        results = []
        reviews = []
        for index, (confidence, status, fraction) in enumerate(
            (("HIGH", "CANDIDATE_OK", "0.5"), ("LOW", "REVIEW_REQUIRED_BOUNDARY", "0.599")),
            start=1,
        ):
            alias = f"case_{index:03d}"
            preview = f"previews_ciegas/{alias}_separacion.png"
            results.append(
                {
                    "case_alias": alias,
                    "split_fraction": fraction,
                    "confidence_level": confidence,
                    "technical_status": status,
                    "proposed_image_left_side": "RIGHT",
                    "proposed_image_right_side": "LEFT",
                    "preview_file": preview,
                    "error_code": "",
                    "manifest_key": f"private_{index}",
                }
            )
            reviews.append(
                {
                    "case_alias": alias,
                    "preview_file": preview,
                    "automatic_confidence_level": confidence,
                    "automatic_split_fraction": fraction,
                    "proposed_image_left_side": "RIGHT",
                    "proposed_image_right_side": "LEFT",
                    "split_acceptable_yes_no": "SI",
                    "laterality_mapping_supported_yes_no": "SI",
                    "observed_marker_or_anatomy": "evidencia anatómica",
                    "technical_exclusion_yes_no": "NO",
                    "exclusion_reason": "",
                    "reviewed_without_outcome_yes_no": "SI",
                    "reviewer_notes": "aceptado",
                }
            )
        write_csv(root / "resultados_separacion_privados.csv", RESULT_FIELDS, results)
        write_csv(root / "revision_visual_ciega.csv", REVIEW_FIELDS, reviews)

    def test_complete_blinded_review_freezes_parameters(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.prepare(root)

            closure = finalize_bilateral_review(
                root, 2, "bilateral_split_v0.2_pilot", 1, "closure"
            )

            self.assertEqual(closure["status"], "closed")
            self.assertEqual(closure["confidence_counts"], {"HIGH": 1, "LOW": 1})
            self.assertTrue(closure["parameters_frozen"])
            frozen = json.loads((root / "parametros_congelados.json").read_text())
            self.assertEqual(frozen["status"], "frozen_after_blinded_visual_review")
            public_text = (root / "cierre_paso_3_publico.json").read_text()
            self.assertNotIn("private_", public_text)

    def test_incomplete_human_decision_prevents_closure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.prepare(root)
            review_path = root / "revision_visual_ciega.csv"
            text = review_path.read_text(encoding="utf-8-sig")
            review_path.write_text(text.replace(";SI;SI;", ";;SI;", 1), encoding="utf-8-sig")

            with self.assertRaisesRegex(ValueError, "invalid_split_decision"):
                finalize_bilateral_review(
                    root, 2, "bilateral_split_v0.2_pilot", 1, "closure"
                )

    def test_automatic_values_cannot_be_edited_during_review(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.prepare(root)
            review_path = root / "revision_visual_ciega.csv"
            text = review_path.read_text(encoding="utf-8-sig")
            review_path.write_text(text.replace(";HIGH;0.5;", ";LOW;0.5;", 1), encoding="utf-8-sig")

            with self.assertRaisesRegex(ValueError, "automatic_confidence_mismatch"):
                finalize_bilateral_review(
                    root, 2, "bilateral_split_v0.2_pilot", 1, "closure"
                )


if __name__ == "__main__":
    unittest.main()
