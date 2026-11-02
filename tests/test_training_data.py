"""Synthetic fold/reference checks; no predictive fitting or reserved data."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from knee import training_data as data
from knee import model_inputs_preflight as previous
from knee.clinical_fold_inputs import fit_clinical_fold, transform_clinical_rows
from knee.colab_data import canonical_json, sha_file
from knee.model_inputs_colab import SCIENTIFIC_FILES
from test_model_inputs_preflight import synthetic_development

ROOT = Path(__file__).resolve().parents[1]


def synthetic_inputs():
    """Thirty artificial knees, fifteen people, five folds with both classes."""
    return synthetic_development()


def reference_fixture(directory):
    root = Path(directory)
    package = root / "synthetic_development"
    package.mkdir()
    rows, tensors = synthetic_inputs()
    bindings = {key: hashlib.sha256(key.encode()).hexdigest() for key in (
        "partition_sha256", "split_policy_sha256", "cohort_closure_sha256", "tensor_contract_sha256")}
    manifest = {"version": "colab_data_package_v1.0", "package_kind": "development", "bindings": bindings, "rows": rows}
    archive = package / "development_package.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as stream:
        stream.writestr("manifest.json", canonical_json(manifest))
        for row in rows:
            buffer = io.BytesIO()
            np.save(buffer, tensors[(row["src_subject_id"], row["side"])], allow_pickle=False)
            stream.writestr(row["tensor"]["relative_path"], buffer.getvalue())
    package_sha = sha_file(archive)
    handoff = {"version": "participant_split_colab_preparation_v1.0",
        "development": {"file": archive.name, "sha256": package_sha, "bytes": archive.stat().st_size,
                        "participants": 15, "knees": 30},
        "partition_sha256": bindings["partition_sha256"], "split_policy_sha256": bindings["split_policy_sha256"],
        "training_executed": False, "reserved_test_predictive_evaluation_opened": False}
    handoff_path = package / "handoff.json"
    handoff_path.write_bytes(canonical_json(handoff))
    policy = json.loads((ROOT / "configs/model_inputs_preflight_v1.json").read_text(encoding="utf-8"))
    folds, artifacts = [], {}
    for fold, state in enumerate(previous.prepare_fold_states(rows, policy)):
        artifacts[f"clinical/fold_{fold}_state.json"] = hashlib.sha256(canonical_json(state)).hexdigest()
        record = {"validation_fold": fold, "clinical_state_sha256": state["record_sha256"], "splits": {}}
        for split in ("training", "validation"):
            selected = [r for r in rows if (r["cv_fold"] == fold) == (split == "validation")]
            clinical = transform_clinical_rows(selected, state)
            identities = [[r["src_subject_id"], r["side"]] for r in selected]
            buffer = io.BytesIO()
            np.save(buffer, clinical, allow_pickle=False)
            artifacts[f"clinical/fold_{fold}_{split}.npy"] = hashlib.sha256(buffer.getvalue()).hexdigest()
            artifacts[f"clinical/fold_{fold}_{split}_order.json"] = hashlib.sha256(canonical_json(identities)).hexdigest()
            record["splits"][split] = {"rows": len(selected), "participants": len({r["src_subject_id"] for r in selected}),
                "imputed_BMI_rows": sum(r["BMI"] is None for r in selected),
                "ordered_identity_sha256": previous.digest(identities),
                "clinical_sha256": hashlib.sha256(clinical.tobytes(order="C")).hexdigest()}
        folds.append(record)
    reference = {"version": previous.VERSION, "status": "development_inputs_and_cpu_loading_verified",
        "development_participants": 15, "development_knees": 30,
        "clinical_features": ["AGEYEARS", "SEX", "BMI"], "clinical_states_fitted": 5,
        "sources_sha256": {"development_package": package_sha, "handoff": sha_file(handoff_path),
            "policy": sha_file(ROOT / "configs/model_inputs_preflight_v1.json"), "registry": sha_file(ROOT / "configs/model_registry.json")},
        "implementation_sha256": {name: sha_file(ROOT / relative) for name, relative in SCIENTIFIC_FILES.items()},
        "folds": folds, "clinical_artifacts_sha256": artifacts,
        "image_loading": {encoder: [{"validation_fold": fold} for fold in range(5)] for encoder in policy["image_encoders"]},
        "validation_or_target_used_for_fitting": False, "predictive_model_training_executed": False,
        "optimizer_created": False, "backward_executed": False, "reserved_test_opened": False,
        "originals_modified": False, "partitions_modified": False, "pretrained_weights_loaded": False,
        "additional_resize_crop_flip_or_augmentation": False, "clinical_preprocessing_fitted": True,
        "aligned_multimodal_input_pairs_verified": True, "source_files_unchanged": True,
        "development_rows_and_pixels_unchanged": True}
    reference_path = root / "synthetic_notebook32_reference.json"
    reference_path.write_bytes(canonical_json(reference))
    return {"rows": rows, "tensors": tensors, "package": package, "archive": archive,
        "package_sha256": package_sha, "handoff": handoff, "handoff_path": handoff_path,
        "reference": reference, "reference_path": reference_path}


class TrainingDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.rows, self.tensors = synthetic_inputs()

    def test_all_five_folds_both_classes_and_grouped_participants(self):
        for fold in range(5):
            inputs = data.prepare_fold(self.rows, self.tensors, fold, package_sha256="a" * 64)
            self.assertEqual(set(inputs.training._targets), {0, 1})
            self.assertEqual(set(inputs.validation._targets), {0, 1})
            self.assertFalse({p for p, _ in inputs.training.identities} & {p for p, _ in inputs.validation.identities})
            self.assertEqual(inputs.training.train_identity_sha256, inputs.validation.train_identity_sha256)

    def test_validation_extremes_and_targets_do_not_change_training_preprocessing(self):
        original = data.prepare_fold(self.rows, self.tensors, 0)
        changed = deepcopy(self.rows)
        for row in changed:
            if row["cv_fold"] == 0:
                row["AGEYEARS"] = 1000
                row["BMI"] = 100000.0
                row["PROGRESION_48M"] = 1 - row["PROGRESION_48M"]
        extreme = data.prepare_fold(changed, self.tensors, 0)
        self.assertEqual(original.clinical_state, extreme.clinical_state)
        self.assertEqual(original.class_weight, extreme.class_weight)
        self.assertTrue(np.array_equal(original.training._clinical, extreme.training._clinical))

    def test_positive_weight_uses_training_knees_only(self):
        inputs = data.prepare_fold(self.rows, self.tensors, 1)
        positive = sum(r["PROGRESION_48M"] for r in self.rows if r["cv_fold"] != 1)
        count = sum(r["cv_fold"] != 1 for r in self.rows)
        self.assertEqual(inputs.class_weight["positive_weight"], (count - positive) / positive)

    def test_rejects_missing_training_or_validation_class(self):
        for training in (True, False):
            changed = deepcopy(self.rows)
            for row in changed:
                if (row["cv_fold"] != 0) == training:
                    row["PROGRESION_48M"] = 0
            with self.assertRaises(ValueError):
                data.prepare_fold(changed, self.tensors, 0)

    def test_frozen_arrays_not_mutated_and_clip_loader_normalization_is_specific(self):
        before = {pair: array.tobytes() for pair, array in self.tensors.items()}
        inputs = data.prepare_fold(self.rows, self.tensors, 0, encoder_id="image_biomedclip")
        first = inputs.validation[0]
        self.assertEqual(tuple(first["image"].shape), (3, 224, 224))
        self.assertEqual(tuple(first["clinical"].shape), (3,))
        self.assertTrue(all(array.tobytes() == before[pair] and not array.flags.writeable for pair, array in self.tensors.items()))

    def test_loaders_seeded_shuffle_no_drops_or_workers(self):
        fold = data.prepare_fold(self.rows, self.tensors, 0, encoder_id="image_densenet121")
        left = data.loaders(fold, seed=2026, micro_batch_size=4)
        right = data.loaders(fold, seed=2026, micro_batch_size=4)
        self.assertIsInstance(left[0].sampler, RandomSampler)
        self.assertFalse(left[0].sampler.replacement)
        self.assertIsInstance(left[1].sampler, SequentialSampler)
        self.assertEqual([b["src_subject_id"] for b in left[0]], [b["src_subject_id"] for b in right[0]])
        self.assertEqual(sum(len(b["target"]) for b in left[1]), len(fold.validation))
        self.assertTrue(all(loader.drop_last is False and loader.num_workers == 0 for loader in left))

    def test_reserved_roles_participant_overlap_or_mutable_pixels_rejected(self):
        changed = deepcopy(self.rows)
        changed[0]["outer_block"], changed[0]["cv_fold"] = 0, None
        with self.assertRaises(ValueError):
            data.prepare_fold(changed, self.tensors, 0)
        pair = next(iter(self.tensors))
        tensors = dict(self.tensors)
        tensors[pair] = tensors[pair].copy()
        with self.assertRaises(ValueError):
            data.prepare_fold(self.rows, tensors, 0)

    def test_approved_reference_exact_clinical_states_and_artifacts_reproduced(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = reference_fixture(directory)
            loaded = data.load_sealed_inputs(fixture["package"], fixture["reference_path"], ROOT, fixture["package_sha256"])
            self.assertEqual(loaded["manifest"]["rows"], fixture["rows"])
            self.assertEqual(len(loaded["tensors"]), 30)

    def test_handoff_corruption_or_reference_hash_rejected_before_zip_open(self):
        for mutation in ("version", "file", "bytes", "hash", "reference"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                fixture = reference_fixture(directory)
                handoff, reference = fixture["handoff"], fixture["reference"]
                if mutation == "version": handoff["version"] = "changed"
                elif mutation == "file": handoff["development"]["file"] = "reserved_test.zip"
                elif mutation == "bytes": handoff["development"]["bytes"] += 1
                elif mutation == "hash": handoff["development"]["sha256"] = "b" * 64
                else: reference["sources_sha256"]["development_package"] = "b" * 64
                fixture["handoff_path"].write_bytes(canonical_json(handoff))
                reference["sources_sha256"]["handoff"] = sha_file(fixture["handoff_path"])
                fixture["reference_path"].write_bytes(canonical_json(reference))
                with patch.object(data, "load_development_package") as loader, self.assertRaises(ValueError):
                    data.load_sealed_inputs(fixture["package"], fixture["reference_path"], ROOT, fixture["package_sha256"])
                loader.assert_not_called()

    def test_partition_handoff_binding_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = reference_fixture(directory)
            fixture["handoff"]["partition_sha256"] = "c" * 64
            fixture["handoff_path"].write_bytes(canonical_json(fixture["handoff"]))
            fixture["reference"]["sources_sha256"]["handoff"] = sha_file(fixture["handoff_path"])
            fixture["reference_path"].write_bytes(canonical_json(fixture["reference"]))
            with self.assertRaisesRegex(ValueError, "verified_development_membership_changed"):
                data.load_sealed_inputs(fixture["package"], fixture["reference_path"], ROOT, fixture["package_sha256"])

    def test_five_fold_clinical_state_and_value_reference_mutations_rejected(self):
        for mutation in ("state", "values", "artifact"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                fixture = reference_fixture(directory)
                reference = fixture["reference"]
                if mutation == "state": reference["folds"][0]["clinical_state_sha256"] = "d" * 64
                elif mutation == "values": reference["folds"][1]["splits"]["validation"]["clinical_sha256"] = "d" * 64
                else: reference["clinical_artifacts_sha256"]["clinical/fold_0_state.json"] = "d" * 64
                fixture["reference_path"].write_bytes(canonical_json(reference))
                with self.assertRaisesRegex(ValueError, "previously_verified_clinical"):
                    data.load_sealed_inputs(fixture["package"], fixture["reference_path"], ROOT, fixture["package_sha256"])


if __name__ == "__main__":
    unittest.main()
