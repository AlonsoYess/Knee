"""Synthetic safeguards for the final common cohort; no private inputs are read."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from knee import common_cohort as subject


SOURCE_COLUMNS = (
    "SRC_SUBJECT_ID SIDE KL_BASE KL_48 PROGRESION_48M AGEYEARS SEX BMI "
    "BARCODE_BASE BARCODE_48 IMAGE_FILE IMAGE_THUMBNAIL_FILE QC_OUTCOME "
    "CIRUGIA_PREVIA WOMAC_TOTAL FECHA_V00 FECHA_V06 REEMPLAZO_AL_INICIO "
    "FECHA_REEMPLAZO DIAS_REEMPLAZO REEMPLAZO_ANTES_V06"
).split()


def fixture():
    """One bilateral routine subject, one usable and one excluded developer."""
    identities = [
        ("synthetic_routine", "RIGHT", "routine", "routine_batch_001", "accepted"),
        ("synthetic_routine", "LEFT", "routine", "routine_batch_001", "accepted"),
        ("synthetic_dev_used", "RIGHT", "development", "mcr009", "accepted"),
        ("synthetic_dev_excluded", "LEFT", "development", "mcr012", "excluded"),
    ]
    inventory = []
    for sid, sides in (("synthetic_routine", ["RIGHT", "LEFT"]),
                       ("synthetic_dev_used", ["RIGHT"]),
                       ("synthetic_dev_excluded", ["LEFT"])):
        inventory.append({
            "src_subject_id": sid, "acquisition_key": "acq_" + sid,
            "barcode_base": "barcode_" + sid, "image_file": sid + ".dcm",
            "lateralities": "Derecha + Izquierda" if len(sides) == 2 else (
                "Derecha" if sides == ["RIGHT"] else "Izquierda"),
            "associated_knees": str(len(sides)),
        })
    rows, baseline_rows, asset_rows, label_rows = [], [], [], []
    for index, (sid, side, role, block, state) in enumerate(identities):
        inv = next(row for row in inventory if row["src_subject_id"] == sid)
        rows.append({"src_subject_id": sid, "side": side, "role": role,
                     "block": block, "state": state,
                     "reasons": [] if state == "accepted" else ["unclassified_mark"]})
        age, sex, bmi = ("60", "F", "") if role == "routine" else ("70", "M", "30.5")
        base = "3" if side == "LEFT" else "2"
        baseline_rows.append({"SRC_SUBJECT_ID": sid, "SIDE": "1" if side == "RIGHT" else "2",
            "KL_BASE": base, "AGEYEARS": age, "SEX": sex, "BMI": bmi,
            "BARCODE_BASE": inv["barcode_base"], "IMAGE_FILE": inv["image_file"]})
        if state == "excluded":
            continue
        asset_rows.append({"src_subject_id": sid, "side": side, "role": role, "block": block,
            "acquisition_key": inv["acquisition_key"], "image_file": inv["image_file"],
            "barcode_base": inv["barcode_base"],
            "crop": {"root_key": "runtime", "relative_path": "accepted/" + sid + "_" + side + ".npy",
                     "sha256": str(index + 1) * 64, "pixel_sha256": str(index + 2) * 64,
                     "shape": [100, 120], "dtype": "uint16"},
            "photometric": "MONOCHROME2", "row_spacing_mm": .15, "column_spacing_mm": .15,
            "spacing_tag": "ImagerPixelSpacing", "box": [10, 130, 20, 120],
            "half_x0": 0, "half_x1": 150, "source_shape": [150, 300]})
        follow = "3" if index < 2 else "1"
        label_rows.append({"SRC_SUBJECT_ID": sid, "SIDE": "1" if side == "RIGHT" else "2",
            "KL_BASE": base, "KL_48": follow, "PROGRESION_48M": "1" if index == 0 else "0",
            "FECHA_V00": "20/01/2005", "FECHA_V06": "25/02/2009"})
    ledger = {"rows": rows, "inventory": inventory,
        "roles": {"development_ids": ["synthetic_dev_excluded", "synthetic_dev_used"],
                  "routine_ids": ["synthetic_routine"]},
        "summary": {"all_inventory": {"participants": 3, "eligible_knees": 4,
                                     "accepted": 3, "excluded": 1, "pending": 0}}}
    return ledger, baseline_rows, {"rows": asset_rows, "pins": {}, "summary": {}}, label_rows


def membership(data=None):
    ledger, baseline_rows, assets, _ = data or fixture()
    return subject.build_membership(ledger, baseline_rows, assets)


class CommonMembershipTests(unittest.TestCase):
    def test_common_membership_preserves_missing_bmi_and_all_reservations(self):
        data = fixture()
        before = deepcopy(data)
        result = membership(data)
        self.assertEqual(len(result["rows"]), 3)
        self.assertEqual(len({row["src_subject_id"] for row in result["rows"]}), 2)
        routine = [row for row in result["rows"] if row["role"] == "routine"]
        self.assertEqual(len(routine), 2)
        self.assertTrue(all(row["clinical"]["BMI"] is None for row in routine))
        self.assertTrue(all(row["clinical"]["AGEYEARS"] == 60 and row["clinical"]["SEX"] == "F" for row in routine))
        self.assertEqual(set(result["development_reservations"]), set(data[0]["roles"]["development_ids"]))
        self.assertEqual(result["exclusions"], [dict(data[0]["rows"][-1],
            acquisition_key="acq_synthetic_dev_excluded")])
        self.assertEqual(result["summary"], {"eligible_knees": 4, "accepted": 3, "excluded": 1,
            "pending": 0, "participants": 2, "permanent_development_participants": 2,
            "routine_knees": 2, "routine_participants": 1, "development_knees": 1,
            "development_participants": 1, "missing_bmi_knees": 2, "missing_bmi_participants": 1})
        self.assertEqual(result["imputation"], {"performed": False,
            "BMI": "median_fit_on_training_fold_only", "missingness_indicator_added": False})
        self.assertEqual(result["same_membership_for"], ["clinical", "radiographic", "multimodal"])
        self.assertEqual(data, before)
        self.assertNotIn("PROGRESION_48M", json.dumps(result))
        self.assertNotIn("KL_48", json.dumps(result))

    def test_duplicate_baseline_key_is_not_silently_overwritten(self):
        data = fixture()
        data[1].append(deepcopy(data[1][0]))
        self.assertRaises(ValueError, membership, data)

    def test_all_inventory_baseline_rows_are_required(self):
        for index in (0, 3):
            data = fixture()
            data[1].pop(index)
            self.assertRaises(ValueError, membership, data)

    def test_baseline_extra_and_unknown_side_fail(self):
        data = fixture()
        extra = dict(data[1][0], SRC_SUBJECT_ID="unknown_subject")
        data[1].append(extra)
        self.assertRaises(ValueError, membership, data)
        for side in ("", "RIGHT", "3"):
            data = fixture()
            data[1][0]["SIDE"] = side
            self.assertRaises(ValueError, membership, data)

    def test_acquisition_link_must_match_sealed_source(self):
        for field in ("BARCODE_BASE", "IMAGE_FILE"):
            data = fixture()
            data[1][0][field] = "different_acquisition"
            self.assertRaises(ValueError, membership, data)
        for field in ("acquisition_key", "image_file", "barcode_base", "side", "role", "block"):
            data = fixture()
            data[2]["rows"][0][field] = "different_acquisition_or_role"
            self.assertRaises(ValueError, membership, data)

    def test_invalid_clinical_source_values_fail_without_new_exclusions(self):
        cases = {"AGEYEARS": ("", "NaN", "inf", "0", "-1", "60.5"),
                 "SEX": ("", "1", "female"),
                 "BMI": ("NaN", "inf", "-inf", "0", "-2", "missing"),
                 "KL_BASE": ("", "0", "1", "4", "2.5")}
        for field, values in cases.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    data = fixture()
                    data[1][0][field] = value
                    self.assertRaises(ValueError, membership, data)

    def test_bilateral_participant_clinical_inconsistency_fails(self):
        for field, value in (("AGEYEARS", "61"), ("SEX", "M"), ("BMI", "26")):
            data = fixture()
            data[1][1][field] = value
            self.assertRaises(ValueError, membership, data)

    def test_assets_match_all_and_only_accepted_keys_once(self):
        for mutation in (lambda rows: rows.pop(), lambda rows: rows.append(deepcopy(rows[0]))):
            data = fixture()
            mutation(data[2]["rows"])
            self.assertRaises(ValueError, membership, data)
        data = fixture()
        extra = deepcopy(data[2]["rows"][0])
        extra.update(src_subject_id="synthetic_dev_excluded", side="LEFT", role="development", block="mcr012")
        data[2]["rows"].append(extra)
        self.assertRaises(ValueError, membership, data)

    def test_permanent_role_reservations_cannot_be_released(self):
        data = fixture()
        data[0]["roles"]["development_ids"].remove("synthetic_dev_excluded")
        self.assertRaises(ValueError, membership, data)
        data = fixture()
        data[0]["rows"][2]["role"] = "routine"
        self.assertRaises(ValueError, membership, data)


class CommonLabelTests(unittest.TestCase):
    def test_all_accepted_labels_and_only_labels_are_linked(self):
        data = fixture()
        fixed = membership(data)
        before = deepcopy((fixed, data[3]))
        result = subject.build_labels(fixed, data[3])
        self.assertEqual(len(result["rows"]), 3)
        self.assertEqual({(row["src_subject_id"], row["side"]) for row in result["rows"]},
                         {(row["src_subject_id"], row["side"]) for row in fixed["rows"]})
        self.assertEqual(sum(row["PROGRESION_48M"] for row in result["rows"]), 1)
        self.assertEqual(result["summary"], {"knees": 3, "positive_knees": 1, "negative_knees": 2,
            "participants_with_any_positive_knee": 1, "participants_without_positive_knee": 1,
            "role_class_counts": {"routine": {"positive": 1, "negative": 1},
                                  "development": {"positive": 0, "negative": 1}}})
        self.assertEqual((fixed, data[3]), before)
        # A lower follow-up grade remains a valid negative under the approved rule.
        dev = next(row for row in result["rows"] if row["src_subject_id"] == "synthetic_dev_used")
        self.assertEqual((dev["KL_BASE"], dev["KL_48"], dev["PROGRESION_48M"]), (2, 1, 0))

    def test_missing_extra_and_duplicate_label_keys_fail(self):
        for mutate in (lambda rows: rows.pop(), lambda rows: rows.append(deepcopy(rows[0])),
                       lambda rows: rows.append(dict(rows[0], SRC_SUBJECT_ID="synthetic_dev_excluded", SIDE="2"))):
            data = fixture()
            fixed = membership(data)
            mutate(data[3])
            self.assertRaises(ValueError, subject.build_labels, fixed, data[3])

    def test_label_mismatch_and_ineligible_or_noninteger_grades_fail(self):
        cases = {"KL_BASE": ("3", "1", "4", "2.5", "NaN"),
                 "KL_48": ("", "-1", "5", "3.5", "NaN"),
                 "PROGRESION_48M": ("", "0", "2", "1.0", "NaN")}
        for field, values in cases.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    data = fixture()
                    fixed = membership(data)
                    data[3][0][field] = value
                    self.assertRaises(ValueError, subject.build_labels, fixed, data[3])

    def test_chronology_is_required_but_no_new_exact_day_window_is_added(self):
        for base, follow in (("20/01/2005", "20/01/2005"), ("20/01/2010", "25/02/2009"),
                             ("not-a-date", "25/02/2009"), ("20/01/2005", "31/02/2009")):
            data = fixture()
            fixed = membership(data)
            data[3][0].update(FECHA_V00=base, FECHA_V06=follow)
            self.assertRaises(ValueError, subject.build_labels, fixed, data[3])
        data = fixture()
        result = subject.build_labels(membership(data), data[3])
        self.assertEqual(len(result["rows"]), 3)

    def test_label_side_change_fails_membership_match(self):
        data = fixture()
        fixed = membership(data)
        data[3][2]["SIDE"] = "2"
        self.assertRaises(ValueError, subject.build_labels, fixed, data[3])


class ProjectionTests(unittest.TestCase):
    def write_source(self, folder, *, columns=None, rows=None):
        path = Path(folder) / "synthetic_source.csv"
        header = columns or SOURCE_COLUMNS
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=header, delimiter=";")
            writer.writeheader()
            for row in rows or []:
                writer.writerow({key: row.get(key, "future_or_unneeded_value") for key in header})
        return path

    def test_projection_uses_only_requested_whitelist_and_keeps_blank_bmi(self):
        data = fixture()
        with tempfile.TemporaryDirectory() as temp:
            path = self.write_source(temp, rows=data[1])
            rows = subject.read_projection(path, subject.BASELINE_COLUMNS)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(set(row) == set(subject.BASELINE_COLUMNS) for row in rows))
        self.assertEqual(rows[0]["BMI"], "")
        self.assertNotIn("PROGRESION_48M", json.dumps(rows))

    def test_outcome_projection_materializes_only_fixed_accepted_keys(self):
        data = fixture()
        rows = [dict(base, **{key: value for key, value in label.items() if key not in base})
                for base, label in zip(data[1], data[3])]
        rows.append(data[1][-1])
        keys = {(row["src_subject_id"], row["side"]) for row in data[2]["rows"]}
        with tempfile.TemporaryDirectory() as temp:
            path = self.write_source(temp, rows=rows)
            result = subject.read_projection(path, subject.LABEL_COLUMNS, accepted_keys=keys)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(set(row) == set(subject.LABEL_COLUMNS) for row in result))
        self.assertNotIn("synthetic_dev_excluded", json.dumps(result))

    def test_missing_or_duplicate_source_header_fails(self):
        for header in (SOURCE_COLUMNS[:-1], SOURCE_COLUMNS + ["SIDE"]):
            with tempfile.TemporaryDirectory() as temp:
                path = self.write_source(temp, columns=header)
                self.assertRaises(ValueError, subject.read_projection, path, subject.BASELINE_COLUMNS)

    def test_outcome_projection_requires_fixed_keys_and_only_approved_whitelist(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.write_source(temp)
            self.assertRaises(ValueError, subject.read_projection, path, subject.LABEL_COLUMNS)
            self.assertRaises(ValueError, subject.read_projection, path, ["SRC_SUBJECT_ID", "WOMAC_TOTAL"])
            self.assertRaises(ValueError, subject.read_projection, path, subject.BASELINE_COLUMNS, set())

    def test_duplicate_records_and_missing_accepted_label_fail_during_projection(self):
        for rows in ([fixture()[1][0]] * 2, [fixture()[1][0]]):
            with tempfile.TemporaryDirectory() as temp:
                path = self.write_source(temp, rows=rows)
                self.assertRaises(ValueError, subject.read_projection, path, subject.LABEL_COLUMNS,
                                  {("synthetic_routine", "RIGHT"), ("synthetic_routine", "LEFT")})


class ImmutableOutputTests(unittest.TestCase):
    def test_first_write_and_exact_rerun_leave_content_and_timestamp_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "synthetic_membership.json"
            payload = {"version": "synthetic", "rows": [{"BMI": None}], "training_executed": False}
            self.assertIs(subject.write_once(path, payload), False)
            before, timestamp = path.read_bytes(), path.stat().st_mtime_ns
            self.assertIs(subject.write_once(path, deepcopy(payload)), True)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(path.stat().st_mtime_ns, timestamp)
            self.assertEqual(json.loads(before), payload)

    def test_changed_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "synthetic_membership.json"
            subject.write_once(path, {"accepted": 3})
            before = path.read_bytes()
            self.assertRaises(ValueError, subject.write_once, path, {"accepted": 4})
            self.assertEqual(path.read_bytes(), before)

    def test_nonfinite_payload_is_rejected_before_file_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "synthetic_membership.json"
            self.assertRaises(ValueError, subject.write_once, path, {"BMI": float("nan")})
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
