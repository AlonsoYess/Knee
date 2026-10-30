"""Audited baseline projection with synthetic rows; no estimator or real cohort."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from knee import baseline_development_projection as subject
from knee.clinical_scenarios import BASELINE_FIELDS, make_baseline_sidecar
from knee.colab_data import sha_file
from test_training_data import synthetic_inputs


class PoisonTarget(dict):
    def __getitem__(self, key):
        if key == "PROGRESION_48M":
            raise AssertionError("Projection must not inspect development outcomes")
        return super().__getitem__(key)


class BaselineProjectionTests(unittest.TestCase):
    def fixture(self, directory):
        base = Path(directory)
        source_dir = base / "sources"
        source_dir.mkdir()
        rows, _ = synthetic_inputs()
        source_rows = []
        for index, row in enumerate(rows):
            source = {name: "POISON_FUTURE_VALUE_NEVER_PARSE" for name in subject.SOURCE_HEADER}
            source.update(SRC_SUBJECT_ID=row["src_subject_id"], SIDE="1" if row["side"] == "RIGHT" else "2",
                AGEYEARS=str(row["AGEYEARS"]), SEX=row["SEX"], BMI="" if row["BMI"] is None else str(row["BMI"]),
                KL_BASE=str(2 + index % 2), CIRUGIA_PREVIA="" if index % 7 == 0 else str(index % 2),
                WOMAC_TOTAL="" if index % 6 == 0 else str(float(index % 30)))
            source_rows.append(source)
        excluded = {name: "EXCLUDED_POISON_NEVER_PARSE" for name in subject.SOURCE_HEADER}
        excluded.update(SRC_SUBJECT_ID="EXCLUDED_PERSON_NEVER_PROJECT", SIDE="1")
        source_rows.insert(0, excluded)
        fixture = {"rows": rows, "source_rows": source_rows, "path": source_dir / "baseline.csv",
            "output": base / "projection", "package_hash": "a" * 64}
        self.write(fixture)
        return fixture

    def write(self, fixture, *, header=None):
        with fixture["path"].open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream, delimiter=";")
            writer.writerow(header or subject.SOURCE_HEADER)
            for row in fixture["source_rows"]:
                writer.writerow([row[name] for name in subject.SOURCE_HEADER])
        fixture["source_hash"] = sha_file(fixture["path"])

    def project(self, fixture, **kwargs):
        return subject.project_source(fixture["path"], kwargs.get("source_hash", fixture["source_hash"]),
            kwargs.get("rows", fixture["rows"]), development_package_sha256=fixture["package_hash"])

    def test_exact_baseline_only_development_projection_and_downstream_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            rows, record = self.project(fixture)
            self.assertEqual(len(rows), 30)
            self.assertEqual(record["development_participants"], 15)
            self.assertEqual(record["development_knees"], 30)
            self.assertEqual(record["source_rows_scanned"], 31)
            self.assertEqual(record["skipped_non_development_rows"], 1)
            self.assertTrue(all(set(row) == BASELINE_FIELDS for row in rows))
            expected = {(row["src_subject_id"], row["side"]): row for row in fixture["rows"]}
            for row in rows:
                original = expected[(row["src_subject_id"], row["side"])]
                for name in ("AGEYEARS", "SEX", "BMI"):
                    self.assertIs(type(row[name]), type(original[name]))
                    self.assertEqual(row[name], original[name])
                self.assertIn(row["KL_BASE"], {2, 3})
                self.assertIn(row["CIRUGIA_PREVIA"], {0, 1, None})
            sidecar = make_baseline_sidecar(rows, fixture["rows"], fixture["package_hash"], fixture["source_hash"])
            self.assertEqual(len(sidecar["rows"]), 30)
            for name in ("training_executed", "reserved_test_archive_opened", "originals_modified",
                         "partitions_modified", "source_followup_columns_projected", "source_outcome_values_converted"):
                self.assertIs(record[name], False)
            serialized = json.dumps(record)
            self.assertNotIn("EXCLUDED_PERSON_NEVER_PROJECT", serialized)
            self.assertNotIn(fixture["rows"][0]["src_subject_id"], serialized)

    def test_future_and_excluded_clinical_poison_never_converted(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            called = []
            original_integer, original_number = subject._integer, subject._number
            def integer(raw, **kwargs):
                called.append(raw)
                return original_integer(raw, **kwargs)
            def number(raw, **kwargs):
                called.append(raw)
                return original_number(raw, **kwargs)
            with patch.object(subject, "_integer", side_effect=integer), patch.object(subject, "_number", side_effect=number):
                rows, _ = self.project(fixture, rows=[PoisonTarget(row) for row in fixture["rows"]])
            self.assertEqual(len(rows), 30)
            self.assertTrue(called)
            self.assertFalse(any("POISON" in value for value in called))

    def test_missing_counts_zero_womac_and_side_specific_values_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            first = fixture["source_rows"][1]
            first["CIRUGIA_PREVIA"], first["WOMAC_TOTAL"] = "0", "0"
            second = fixture["source_rows"][2]
            second["CIRUGIA_PREVIA"], second["WOMAC_TOTAL"] = "1", "96"
            self.write(fixture)
            rows, record = self.project(fixture)
            found = {(row["src_subject_id"], row["side"]): row for row in rows}
            first_key = first["SRC_SUBJECT_ID"], subject.SIDE_MAP[first["SIDE"]]
            second_key = second["SRC_SUBJECT_ID"], subject.SIDE_MAP[second["SIDE"]]
            self.assertEqual((found[first_key]["CIRUGIA_PREVIA"], found[first_key]["WOMAC_TOTAL"]), (0, 0.0))
            self.assertEqual((found[second_key]["CIRUGIA_PREVIA"], found[second_key]["WOMAC_TOTAL"]), (1, 96.0))
            for name in ("KL_BASE", "CIRUGIA_PREVIA", "WOMAC_TOTAL"):
                self.assertEqual(record["missing_counts"][name], sum(row[name] is None for row in rows))

    def test_source_digest_mismatch_rejected_before_value_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            with patch.object(subject, "_integer", side_effect=AssertionError("Conversion forbidden")):
                with self.assertRaisesRegex(ValueError, "audited_baseline_source_sha256_mismatch"):
                    self.project(fixture, source_hash="b" * 64)

    def test_source_change_during_projection_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            with patch.object(subject, "sha_file", side_effect=[fixture["source_hash"], "b" * 64]):
                with self.assertRaisesRegex(ValueError, "audited_baseline_source_changed_during_projection"):
                    self.project(fixture)

    def test_wrong_header_and_malformed_rows_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            header = list(subject.SOURCE_HEADER)
            header[-1] = "NEW_UNREVIEWED_COLUMN"
            self.write(fixture, header=header)
            with self.assertRaisesRegex(ValueError, "exact_audited_baseline_source_header_required"):
                self.project(fixture)
            self.write(fixture)
            with fixture["path"].open("a", encoding="utf-8", newline="") as stream:
                stream.write("too;few;columns\n")
            fixture["source_hash"] = sha_file(fixture["path"])
            with self.assertRaisesRegex(ValueError, "bounded_well_formed_audited_source_rows_required"):
                self.project(fixture)

    def test_duplicate_source_key_rejected_including_unselected(self):
        for position in (0, 1):
            with self.subTest(position=position), tempfile.TemporaryDirectory() as directory:
                fixture = self.fixture(directory)
                fixture["source_rows"].append(deepcopy(fixture["source_rows"][position]))
                self.write(fixture)
                with self.assertRaisesRegex(ValueError, "duplicate_audited_source_knee"):
                    self.project(fixture)

    def test_missing_development_key_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            fixture["source_rows"].pop()
            self.write(fixture)
            with self.assertRaisesRegex(ValueError, "audited_source_missing_development_baseline_keys"):
                self.project(fixture)

    def test_reserved_assignment_rejected_before_source_access(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            rows = deepcopy(fixture["rows"])
            rows[0]["outer_block"], rows[0]["cv_fold"] = 0, None
            with patch.object(subject, "sha_file", side_effect=AssertionError("Source access forbidden")):
                with self.assertRaises(ValueError):
                    self.project(fixture, rows=rows)

    def test_main_predictor_change_rejected(self):
        for name, value in (("AGEYEARS", "999"), ("SEX", "M"), ("BMI", "99.5")):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                fixture = self.fixture(directory)
                row = fixture["source_rows"][1]
                row[name] = "F" if name == "SEX" and row[name] == "M" else value
                self.write(fixture)
                with self.assertRaisesRegex(ValueError, "source_main_predictors_differ_from_development_package"):
                    self.project(fixture)

    def test_invalid_baseline_domains_rejected(self):
        cases = (("KL_BASE", ""), ("KL_BASE", "4"), ("KL_BASE", "2.5"),
            ("CIRUGIA_PREVIA", "2"), ("CIRUGIA_PREVIA", "unknown"),
            ("WOMAC_TOTAL", "-0.1"), ("WOMAC_TOTAL", "96.1"),
            ("WOMAC_TOTAL", "NaN"), ("WOMAC_TOTAL", "inf"), ("WOMAC_TOTAL", " 1"))
        for name, value in cases:
            with self.subTest(name=name, value=value), tempfile.TemporaryDirectory() as directory:
                fixture = self.fixture(directory)
                fixture["source_rows"][1][name] = value
                self.write(fixture)
                with self.assertRaises(ValueError):
                    self.project(fixture)

    def test_private_output_readback_and_source_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            record = subject.prepare_projection(fixture["path"], fixture["source_hash"], fixture["rows"],
                fixture["output"], development_package_sha256=fixture["package_hash"])
            target = fixture["output"] / subject.PROJECTION_FILENAME
            self.assertEqual(sha_file(target), record["projection_sha256"])
            self.assertEqual(target.stat().st_size, record["projection_bytes"])
            rows = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 30)
            self.assertEqual(sha_file(fixture["path"]), fixture["source_hash"])
            self.assertEqual(json.loads((fixture["output"] / subject.RECORD_FILENAME).read_text(encoding="utf-8")), record)
            with self.assertRaisesRegex(ValueError, "new_disjoint_private_baseline_projection_output_required"):
                subject.prepare_projection(fixture["path"], fixture["source_hash"], fixture["rows"],
                    fixture["output"], development_package_sha256=fixture["package_hash"])

    def test_projection_output_cannot_overlap_source_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            with self.assertRaisesRegex(ValueError, "new_disjoint_private_baseline_projection_output_required"):
                subject.prepare_projection(fixture["path"], fixture["source_hash"], fixture["rows"],
                    fixture["path"].parent / "output", development_package_sha256=fixture["package_hash"])

    def test_source_row_and_byte_budgets_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            with patch.object(subject, "MAX_SOURCE_BYTES", 1):
                with self.assertRaisesRegex(ValueError, "bounded_audited_source_csv_required"):
                    self.project(fixture)
            with patch.object(subject, "MAX_SOURCE_ROWS", 1):
                with self.assertRaisesRegex(ValueError, "bounded_well_formed_audited_source_rows_required"):
                    self.project(fixture)


if __name__ == "__main__":
    unittest.main()
