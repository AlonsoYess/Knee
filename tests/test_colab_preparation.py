"""End-to-end preparation tests using only temporary synthetic tensors/labels."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from knee import colab_preparation as subject
from knee.colab_data import ROW_FIELDS, verify_development_package
from test_participant_partitions import synthetic_inputs


class ColabPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="knee_colab_prep_synthetic_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.project = Path(subject.__file__).resolve().parents[2]
        self.incoming = self.root / "incoming"
        self.governance = self.root / "governance"
        self.output = self.root / "portable_outputs"
        self.incoming.mkdir()
        self.governance.mkdir()
        self.config_path = self.governance / "config.json"
        self.authority_path = self.governance / "authority.json"
        members, labels = synthetic_inputs()
        # These inherited source fields must never enter the portable model rows.
        for row in members["rows"]:
            row.update(KL_BASE=3, image_file="SYNTHETIC_NATIVE_SOURCE.dcm",
                       crop={"relative_path": "synthetic_private_native.npy",
                             "private_path": str(self.incoming / "never_read_native.npy")})
        for row in labels["rows"]:
            row.update(KL_BASE=3, KL_48=4, FECHA_V00="synthetic_visit_0",
                       FECHA_V06="synthetic_visit_48")
        self.payloads = {"membership": members, "labels": labels,
            "cohort_closure": {"status": "common_cohort_frozen_input_tensor_and_splits_pending",
                               "files_sha256": {}},
            "split_policy": deepcopy(subject.POLICY),
            "routine_verification": {"status": "passed",
                "development_and_routine_inputs_disjoint_and_complete": True}}
        tensor = np.full((1, 224, 224), 0.25, dtype=np.float32)
        buffer = io.BytesIO()
        np.save(buffer, tensor, allow_pickle=False)
        raw = buffer.getvalue()
        self.tensor_sha = hashlib.sha256(raw).hexdigest()
        self.pixel_sha = hashlib.sha256(tensor.tobytes()).hexdigest()
        self.source_paths = {}
        contract_sha = "a" * 64
        for prefix in ("development", "routine"):
            source_root = self.incoming / prefix
            source_root.mkdir()
            tensor_root = source_root / "tensors"
            tensor_root.mkdir()
            records = []
            for index, row in enumerate((r for r in members["rows"] if r["role"] == prefix), 1):
                relative = f"tensors/synthetic_{index:04d}.npy"
                path = source_root / relative
                path.write_bytes(raw)
                self.source_paths[path] = self.tensor_sha
                records.append({"src_subject_id": row["src_subject_id"], "side": row["side"],
                    "role": prefix, "native_crop": deepcopy(row["crop"]),
                    "tensor": {"relative_path": relative, "sha256": self.tensor_sha,
                        "pixel_sha256": self.pixel_sha, "shape": [1, 224, 224], "dtype": "float32"}})
            self.payloads[prefix + "_manifest"] = {
                "tensor_contract_version": "physical_letterbox_224_p01_p99_v1.0",
                "tensor_contract_sha256": contract_sha, "rows": records}
            self.payloads[prefix + "_closure"] = {"training_executed": False,
                                                   "partitions_created": False}
        self.authority = {"version": subject.VERSION, "approved_by": "researcher",
            "user_statement": "Synthetic authorized fixture; no private data or training.",
            "allowed_actions": deepcopy(subject.ALLOWED_ACTIONS),
            "forbidden_actions": deepcopy(subject.FORBIDDEN_ACTIONS)}
        paths = {name: (self.incoming / (name + ".json")) for name in subject.SOURCE_NAMES}
        for prefix in ("development", "routine"):
            paths[prefix + "_manifest"] = self.incoming / prefix / "manifest.json"
        self.config = {"version": subject.VERSION, "project_root": str(self.project),
            "sources": {name: {"path": str(path), "sha256": ""} for name, path in paths.items()},
            "authority": {"path": str(self.authority_path), "sha256": ""},
            "component_pins": {relative: subject.sha_file(self.project / relative)
                               for relative in subject.PINNED_COMPONENTS},
            "output_root": str(self.output)}
        self.refresh_sources()

    @staticmethod
    def write_json(path, value):
        path.write_bytes(subject.canonical(value) + b"\n")

    def write_payload(self, name):
        spec = self.config["sources"][name]
        self.write_json(Path(spec["path"]), self.payloads[name])
        spec["sha256"] = subject.sha_file(spec["path"])

    def refresh_sources(self):
        self.payloads["labels"]["membership_sha256"] = subject.digest(self.payloads["membership"])
        for name in ("membership", "labels", "split_policy", "development_manifest", "routine_manifest"):
            self.write_payload(name)
        for prefix in ("development", "routine"):
            self.payloads[prefix + "_closure"]["manifest_sha256"] = self.config["sources"][prefix + "_manifest"]["sha256"]
            self.write_payload(prefix + "_closure")
        self.payloads["cohort_closure"]["files_sha256"] = {
            "membership_without_outcomes.json": self.config["sources"]["membership"]["sha256"],
            "outcomes.json": self.config["sources"]["labels"]["sha256"]}
        self.write_payload("cohort_closure")
        self.payloads["routine_verification"]["closure_summary_sha256"] = self.config["sources"]["routine_closure"]["sha256"]
        self.write_payload("routine_verification")
        self.write_json(self.authority_path, self.authority)
        self.config["authority"]["sha256"] = subject.sha_file(self.authority_path)
        self.write_json(self.config_path, self.config)

    def run_preparation(self):
        return subject.run(self.config_path)

    def reject(self):
        with self.assertRaises((ValueError, TypeError, KeyError, FileNotFoundError)):
            self.run_preparation()
        self.assertFalse((Path(self.config["output_root"]) / "closure_summary.json").exists())

    def output_hashes(self):
        return {str(p.relative_to(self.output)): subject.sha_file(p)
                for p in self.output.rglob("*") if p.is_file()}

    def test_end_to_end_synthetic_packages_preserve_sources_and_grouping(self):
        before = {path: subject.sha_file(path) for path in self.source_paths}
        result = self.run_preparation()
        self.assertEqual(result["status"], "participant_partitions_and_private_colab_packages_verified")
        self.assertFalse(result["training_executed"])
        self.assertFalse(result["drive_upload_performed"])
        self.assertEqual(before, {path: subject.sha_file(path) for path in self.source_paths})
        closure = json.loads((self.output / "closure_summary.json").read_text(encoding="utf-8"))
        for flag in ("originals_modified", "imputation_performed", "training_executed",
                     "reserved_test_predictive_evaluation_opened", "reserved_test_used_for_model_development",
                     "cohort_membership_changed", "model_specific_normalization_applied"):
            self.assertIs(closure[flag], False)
        handoff = json.loads((self.output / "handoff.json").read_text(encoding="utf-8"))
        self.assertEqual(handoff["development"]["participants"], 56)
        self.assertEqual(handoff["reserved_test"]["participants"], 14)
        self.assertEqual(handoff["development"]["knees"] + handoff["reserved_test"]["knees"], 140)
        # Opening synthetic manifests is safe; real reserved-test data are never used by this test.
        manifests = {}
        for kind in ("development", "reserved_test"):
            with zipfile.ZipFile(self.output / handoff[kind]["file"]) as archive:
                manifests[kind] = json.loads(archive.read("manifest.json"))
                self.assertEqual(len(archive.namelist()), len(manifests[kind]["rows"]) + 1)
                for row in manifests[kind]["rows"]:
                    self.assertEqual(set(row), ROW_FIELDS)
                    self.assertEqual(row["tensor"]["shape"], [1, 224, 224])
                    self.assertEqual(row["tensor"]["dtype"], "float32")
                text = archive.read("manifest.json").decode("utf-8")
                for forbidden in ("KL_BASE", "KL_48", "FECHA_V00", "FECHA_V06", "native_crop",
                                  "SYNTHETIC_NATIVE_SOURCE", str(self.incoming), "never_read_native"):
                    self.assertNotIn(forbidden, text)
        development_ids = {r["src_subject_id"] for r in manifests["development"]["rows"]}
        reserved_ids = {r["src_subject_id"] for r in manifests["reserved_test"]["rows"]}
        self.assertFalse(development_ids & reserved_ids)
        self.assertFalse(reserved_ids & set(self.payloads["membership"]["development_reservations"]))
        self.assertTrue(all(r["outer_block"] != 0 for r in manifests["development"]["rows"]))
        self.assertTrue(all(r["outer_block"] == 0 and r["cv_fold"] is None
                            for r in manifests["reserved_test"]["rows"]))
        verified = verify_development_package(self.output / handoff["development"]["file"],
                                              handoff["development"]["sha256"])
        self.assertEqual(verified["verified_tensors"], 112)
        self.assertFalse(verified["reserved_test_opened"])

    def test_identical_rerun_reuses_every_byte(self):
        first = self.run_preparation()
        before = self.output_hashes()
        second = self.run_preparation()
        self.assertFalse(first["existing_partitions_reused"])
        self.assertFalse(first["existing_closure_reused"])
        self.assertTrue(second["existing_partitions_reused"])
        self.assertTrue(second["existing_closure_reused"])
        self.assertEqual(self.output_hashes(), before)

    def test_modified_existing_package_cannot_be_overwritten(self):
        result = self.run_preparation()
        path = self.output / result["packages"]["development"]["file"]
        path.write_bytes(b"synthetic changed package")
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "existing_package_changed_no_overwrite"):
            self.run_preparation()
        self.assertEqual(path.read_bytes(), before)

    def test_changed_source_without_new_pin_is_rejected(self):
        path = Path(self.config["sources"]["labels"]["path"])
        path.write_bytes(path.read_bytes() + b" ")
        self.reject()

    def test_changed_authority_without_new_pin_is_rejected(self):
        self.authority_path.write_bytes(self.authority_path.read_bytes() + b" ")
        self.reject()

    def test_semantically_changed_authority_is_rejected_even_when_pinned(self):
        self.authority["allowed_actions"].append("training")
        self.refresh_sources()
        self.reject()

    def test_missing_component_pin_is_rejected(self):
        del self.config["component_pins"][next(iter(self.config["component_pins"]))]
        self.write_json(self.config_path, self.config)
        self.reject()

    def test_mismatched_frozen_tensor_contract_is_rejected(self):
        self.payloads["routine_manifest"]["tensor_contract_sha256"] = "b" * 64
        self.refresh_sources()
        self.reject()

    def test_missing_common_tensor_is_rejected(self):
        self.payloads["routine_manifest"]["rows"].pop()
        self.refresh_sources()
        self.reject()

    def test_unapproved_preprocessing_role_is_rejected(self):
        self.payloads["routine_manifest"]["rows"][0]["role"] = "development"
        self.refresh_sources()
        self.reject()

    def test_duplicate_tensor_membership_is_rejected(self):
        self.payloads["routine_manifest"]["rows"].append(deepcopy(self.payloads["routine_manifest"]["rows"][0]))
        self.refresh_sources()
        self.reject()

    def test_changed_derived_tensor_source_is_rejected(self):
        path = next(iter(self.source_paths))
        path.write_bytes(b"synthetic changed tensor")
        self.reject()

    def test_unsafe_tensor_descriptor_cannot_escape_source_root(self):
        self.payloads["routine_manifest"]["rows"][0]["tensor"]["relative_path"] = "../outside.npy"
        self.refresh_sources()
        self.reject()

    def test_absolute_tensor_descriptor_is_rejected(self):
        self.payloads["routine_manifest"]["rows"][0]["tensor"]["relative_path"] = str(self.root / "outside.npy")
        self.refresh_sources()
        self.reject()

    def test_private_output_cannot_overlap_sealed_source_directory(self):
        self.config["output_root"] = str(self.incoming / "overlapping_output")
        self.write_json(self.config_path, self.config)
        self.reject()
        self.assertFalse(Path(self.config["output_root"]).exists())

    def test_private_output_cannot_be_written_inside_repository(self):
        self.config["output_root"] = str(self.project / ("_forbidden_synthetic_" + self.root.name))
        self.write_json(self.config_path, self.config)
        self.reject()
        self.assertFalse(Path(self.config["output_root"]).exists())

    def test_invalid_cohort_label_binding_is_rejected(self):
        self.payloads["labels"]["membership_sha256"] = "0" * 64
        self.write_payload("labels")
        self.payloads["cohort_closure"]["files_sha256"]["outcomes.json"] = self.config["sources"]["labels"]["sha256"]
        self.write_payload("cohort_closure")
        self.write_json(self.config_path, self.config)
        self.reject()

    def test_missing_independent_routine_verification_is_rejected(self):
        self.payloads["routine_verification"]["development_and_routine_inputs_disjoint_and_complete"] = False
        self.refresh_sources()
        self.reject()


if __name__ == "__main__":
    unittest.main()
