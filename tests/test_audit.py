"""Meaningful integrity failures that must stop cohort preparation."""

import csv
import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook

from knee.audit import COLUMNS, MANIFEST_COLUMNS, audit


def example(subject: str, side: str, base: str, follow: str) -> dict[str, str]:
    row = dict.fromkeys(COLUMNS, "")
    row.update({
        "SRC_SUBJECT_ID": subject, "SIDE": side, "KL_BASE": base,
        "KL_48": follow, "PROGRESION_48M": str(int(int(follow) > int(base))),
        "AGEYEARS": "63", "SEX": "F", "BMI": "28.5",
        "BARCODE_BASE": f"barcode_{subject}", "BARCODE_48": "unused",
        "IMAGE_FILE": f"s3://test/{subject}/bilateral.tar.gz",
        "FECHA_V00": "1/01/2005", "FECHA_V06": "2/01/2009",
    })
    return row


class CohortAuditTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rows = [example("example_a", "1", "2", "3"),
                     example("example_a", "2", "3", "3"),
                     example("example_b", "1", "2", "2")]

    def write_sources(self, manifest_override=None):
        cohort_path = self.root / "cohort.csv"
        with cohort_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS, delimiter=";")
            writer.writeheader()
            writer.writerows(self.rows)
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Manifiesto rodillas"
        sheet.append(list(MANIFEST_COLUMNS))
        for row in manifest_override if manifest_override is not None else self.rows:
            sheet.append([row[name] for name in MANIFEST_COLUMNS])
        manifest_path = self.root / "manifest.xlsx"
        workbook.save(manifest_path)
        return cohort_path, manifest_path

    def test_valid_bilateral_pair_and_single_knee(self):
        result = audit(*self.write_sources())
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["counts"]["cohort_rows"], 3)
        self.assertEqual(result["counts"]["unique_bilateral_images"], 2)

    def test_label_disagreement_and_duplicate_knee_stop_audit(self):
        self.rows[0]["PROGRESION_48M"] = "0"
        self.rows.append(self.rows[0].copy())
        result = audit(*self.write_sources())
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["checks"]["label_mismatch"], 2)
        self.assertEqual(result["checks"]["duplicate_subject_side"], 1)

    def test_manifest_mismatch_is_caught_without_disclosing_ids(self):
        manifest = [row.copy() for row in self.rows]
        manifest[0]["IMAGE_FILE"] = "s3://test/incorrect.tar.gz"
        result = audit(*self.write_sources(manifest))
        self.assertEqual(result["checks"]["manifest_value_mismatch"], 1)
        self.assertNotIn("example_a", str(result))


if __name__ == "__main__":
    unittest.main()
