"""Native export reconciliation fails closed without loading image arrays."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from knee import common_cohort_assets as a


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return a.sha_file(path)


class NativeAssetsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = {k: str(self.root / k) for k in a.ROOT_KEYS}
        for root in self.config.values():
            Path(root).mkdir()
        self.ws = Path(self.config["campaign_root"]) / "groups/routine_batch_006/workspace"
        self.ws.mkdir(parents=True)
        self.alias = "case_001_RIGHT"
        self.left = "case_001_LEFT"
        self.native = np.arange(6, dtype=np.uint16).reshape(2, 3)
        path = self.ws / "accepted_native" / (self.alias + ".npy")
        path.parent.mkdir()
        np.save(path, self.native, allow_pickle=False)
        self.file_sha = a.sha_file(path)
        self.pixel_sha = hashlib.sha256(self.native.tobytes()).hexdigest()
        self.study = {"src_subject_id": "1", "case_alias": "case_001", "acquisition_key": "v00_1",
                      "image_file": "image1", "barcode_base": "100"}
        self.source = {"knee_alias": self.alias, "case_alias": "case_001", "side": "RIGHT",
            "source_shape": [4, 5], "source_dtype": "uint16", "source_half_sha256": "d" * 64,
            "half_x0": 0, "half_x1": 5, "photometric": "MONOCHROME2",
            "row_spacing_mm": 0.15, "column_spacing_mm": 0.15, "spacing_tag": "ImagerPixelSpacing"}
        self.decision = {"src_subject_id": "1", "side": "RIGHT", "knee_alias": self.alias,
            "state": "accepted", "box": [0, 3, 0, 2], "crop_sha256": self.pixel_sha,
            "source": {k: self.source[k] for k in
                       ("knee_alias", "source_shape", "source_dtype", "source_half_sha256")}}
        self.excluded = {"src_subject_id": "1", "side": "LEFT", "knee_alias": self.left,
                         "state": "excluded", "box": None, "crop_sha256": None, "source": None}
        self.plan = {"selected": [self.study]}
        self.checkpoint = {"contract_snapshot": {"rows": [self.decision, self.excluded]}}
        self.ledger = {"rows": [
            {"src_subject_id": "1", "side": "RIGHT", "role": "routine", "block": "routine_batch_006", "state": "accepted"},
            {"src_subject_id": "1", "side": "LEFT", "role": "routine", "block": "routine_batch_006", "state": "excluded"}],
            "inventory": [self.study], "pins": {}}
        self.refresh()

    def refresh(self):
        pins = self.ledger["pins"]
        for rel, value in (("plan.json", self.plan), ("checkpoints/terminal.json", self.checkpoint)):
            p = self.ws / rel
            pins[str(p)] = write_json(p, value)
        source_rel = "cases/case_001/sources.json"
        source_sha = write_json(self.ws / source_rel, {"rows": [self.source]})
        qc = {"candidate_evidence_files_sha256": {source_rel: source_sha,
              "candidates/" + self.alias + ".npy": self.file_sha}}
        p = self.ws / "qc_review_seal.json"
        pins[str(p)] = write_json(p, qc)

    def collect(self, **kwargs):
        return a.collect_assets(self.ledger, self.config, **kwargs)

    def test_accepted_only_native_with_dual_hashes_and_portable_path(self):
        value = self.collect()
        self.assertEqual(value["summary"]["accepted_native_assets"], 1)
        self.assertFalse(value["summary"]["native_arrays_loaded"])
        row = value["rows"][0]
        self.assertEqual(row["crop"]["sha256"], self.file_sha)
        self.assertEqual(row["crop"]["pixel_sha256"], self.pixel_sha)
        self.assertEqual(row["crop"]["shape"], [2, 3])
        self.assertEqual(row["crop"]["dtype"], "uint16")
        self.assertEqual(row["crop"]["root_key"], "campaign_root")
        self.assertFalse(Path(row["crop"]["relative_path"]).is_absolute())
        self.assertNotIn("outcomes", row)

    def test_excluded_export_forbidden(self):
        np.save(self.ws / "accepted_native" / (self.left + ".npy"), self.native)
        with self.assertRaisesRegex(ValueError, "excluded_export"):
            self.collect()

    def test_missing_accepted_export(self):
        (self.ws / "accepted_native" / (self.alias + ".npy")).unlink()
        with self.assertRaisesRegex(ValueError, "missing_accepted_export"):
            self.collect()

    def test_changed_npy_data_file_hash(self):
        path = self.ws / "accepted_native" / (self.alias + ".npy")
        with path.open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"x")
        with self.assertRaisesRegex(ValueError, "file_hash_mismatch"):
            self.collect()

    def test_pixel_hash_distinct_from_npy_file_hash(self):
        self.decision["crop_sha256"] = self.file_sha
        self.refresh()
        with self.assertRaisesRegex(ValueError, "pixel_hash_mismatch"):
            self.collect()

    def test_wrong_box_shape(self):
        self.decision["box"] = [0, 2, 0, 2]
        self.refresh()
        with self.assertRaisesRegex(ValueError, "shape_changed"):
            self.collect()

    def test_wrong_source_dtype(self):
        self.source["source_dtype"] = "int16"
        self.decision["source"]["source_dtype"] = "int16"
        self.refresh()
        with self.assertRaisesRegex(ValueError, "dtype_changed"):
            self.collect()

    def test_laterality_mismatch(self):
        self.source["side"] = "LEFT"
        self.refresh()
        with self.assertRaisesRegex(ValueError, "laterality_changed"):
            self.collect()

    def test_metadata_hash_tamper(self):
        p = self.ws / "cases/case_001/sources.json"
        p.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "metadata_changed"):
            self.collect()

    def test_nonpositive_spacing(self):
        self.source["row_spacing_mm"] = 0
        self.refresh()
        with self.assertRaisesRegex(ValueError, "pixel_spacing_required"):
            self.collect()

    def test_box_outside_half(self):
        self.decision["box"] = [0, 10, 0, 2]
        self.refresh()
        with self.assertRaisesRegex(ValueError, "box_outside_half"):
            self.collect()

    def test_half_width_metadata_mismatch(self):
        self.source["half_x1"] = 6
        self.refresh()
        with self.assertRaisesRegex(ValueError, "half_geometry_changed"):
            self.collect()

    def test_duplicate_ledger_knee(self):
        self.ledger["rows"].append(dict(self.ledger["rows"][0]))
        with self.assertRaisesRegex(ValueError, "unique_asset_ledger"):
            self.collect()

    def test_unverified_mode_never_claims_verified_payload(self):
        value = self.collect(verify_payloads=False)
        self.assertFalse(value["summary"]["native_file_and_pixel_hashes_verified"])

    def test_unsafe_metadata_path(self):
        with self.assertRaisesRegex(ValueError, "relative_metadata_path"):
            a.safe(self.ws, "../native.npy")

    def test_object_npy_rejected(self):
        p = self.ws / "object.npy"
        np.save(p, np.array([[object()]], dtype=object), allow_pickle=True)
        with self.assertRaisesRegex(ValueError, "integer_dtype"):
            a.inspect_npy(p, expected_file_sha=a.sha_file(p), expected_pixel_sha="a" * 64,
                          shape=[1, 1], dtype="object")

    def test_fortran_npy_rejected(self):
        p = self.ws / "fortran.npy"
        np.save(p, np.asfortranarray(self.native))
        with self.assertRaisesRegex(ValueError, "c_order"):
            a.inspect_npy(p, expected_file_sha=a.sha_file(p), expected_pixel_sha=self.pixel_sha,
                          shape=[2, 3], dtype="uint16")

    def test_trailing_native_bytes_rejected(self):
        p = self.ws / "accepted_native" / (self.alias + ".npy")
        with p.open("ab") as stream:
            stream.write(b"extra")
        with self.assertRaisesRegex(ValueError, "payload_length"):
            self.collect()

    def test_historical_ineligible_export_not_added_to_common_cohort(self):
        root = Path(self.config["runtime_root"])
        base = root / "outputs/mcr009_contact_confirmation"
        out = base / "application/accepted_native"
        out.mkdir(parents=True)
        np.save(out / (self.alias + ".npy"), self.native)
        np.save(out / (self.left + ".npy"), self.native)
        selected_path = base / "selection/selection.json"
        self.ledger["pins"][str(selected_path)] = write_json(selected_path, {"selected": [self.study]})
        task_path = base / "sources/task.json"
        task_sha = write_json(task_path, {"rows": [self.source]})
        prepared_path = base / "prepared/manifest.json"
        prepared_sha = write_json(prepared_path, {"task_sha256": task_sha})
        result = {"rows": [{**self.decision, "accepted": True},
                           {"knee_alias": self.left, "accepted": True}],
            "manifest_sha256": prepared_sha,
            "native_files_sha256": {"accepted_native/" + alias + ".npy": self.file_sha
                                     for alias in (self.alias, self.left)}}
        result_path = base / "application/results.json"
        self.ledger["pins"][str(result_path)] = write_json(result_path, result)
        manifest = {"workspace_files_sha256": {
            task_path.relative_to(root).as_posix(): task_sha,
            prepared_path.relative_to(root).as_posix(): prepared_sha}}
        p = root / "bounded_local_capsule_manifest.json"
        self.ledger["pins"][str(p)] = write_json(p, manifest)
        self.ledger["rows"] = [{**self.ledger["rows"][0], "block": "mcr009", "role": "development"}]
        value = self.collect()
        self.assertEqual(len(value["rows"]), 1)
        self.assertEqual(value["rows"][0]["side"], "RIGHT")
        self.assertEqual(value["rows"][0]["crop"]["root_key"], "runtime_root")

    def test_required_source_pixel_pin(self):
        self.source["source_half_sha256"] = "invalid"
        self.decision["source"]["source_half_sha256"] = "invalid"
        self.refresh()
        with self.assertRaisesRegex(ValueError, "native_half_pixel_pin"):
            self.collect()


if __name__ == "__main__":
    unittest.main()
