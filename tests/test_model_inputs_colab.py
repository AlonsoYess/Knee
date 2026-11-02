"""Synthetic-only transport/orchestration checks for the development preflight."""
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from knee import model_inputs_colab as subject
from knee import model_inputs_preflight as core
from knee.clinical_fold_inputs import transform_clinical_rows
from knee.colab_data import canonical_json, sha_file
from knee.model_input_adapters import DevelopmentFoldDataset
from test_model_inputs_preflight import synthetic_development


class ModelInputsColabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project = Path(__file__).resolve().parents[1]
        cls.policy = json.loads((cls.project / "configs/model_inputs_preflight_v1.json").read_text(encoding="utf-8"))
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="knee_colab_inputs_synthetic_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.package_dir = self.root / "sealed_synthetic_development"
        self.output = self.root / "new_preflight_outputs"
        self.package_dir.mkdir()
        self.output.mkdir()
        reference_dir = self.root / "reference"
        reference_dir.mkdir()
        self.reference_path = reference_dir / "synthetic_previous_local_preflight.json"
        self.rows, self.tensors = synthetic_development()
        self.package_path = self.package_dir / "development_package.zip"
        bindings = {key: hashlib.sha256(key.encode()).hexdigest() for key in (
            "partition_sha256", "split_policy_sha256", "cohort_closure_sha256", "tensor_contract_sha256")}
        manifest = {"version": "colab_data_package_v1.0", "package_kind": "development",
                    "bindings": bindings, "rows": self.rows}
        with zipfile.ZipFile(self.package_path, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("manifest.json", canonical_json(manifest))
            for row in self.rows:
                tensor = self.tensors[(row["src_subject_id"], row["side"])]
                buffer = io.BytesIO()
                np.save(buffer, tensor, allow_pickle=False)
                archive.writestr(row["tensor"]["relative_path"], buffer.getvalue())
        self.handoff_path = self.package_dir / "handoff.json"
        self.handoff = {"version": "participant_split_colab_preparation_v1.0",
            "development": {"file": self.package_path.name, "sha256": sha_file(self.package_path),
                "bytes": self.package_path.stat().st_size, "participants": 15, "knees": 30},
            "reserved_test": {"file": "reserved_test_must_remain_absent.zip", "sha256": "0" * 64,
                              "bytes": 0, "participants": 0, "knees": 0},
            "partition_sha256": bindings["partition_sha256"],
            "split_policy_sha256": bindings["split_policy_sha256"], "training_executed": False,
            "reserved_test_predictive_evaluation_opened": False}
        self.write_json(self.handoff_path, self.handoff)
        states = core.prepare_fold_states(self.rows, self.policy)
        folds, artifacts = [], {}
        for fold, state in enumerate(states):
            artifacts[f"clinical/fold_{fold}_state.json"] = hashlib.sha256(canonical_json(state)).hexdigest()
            record = {"validation_fold": fold, "clinical_state_sha256": state["record_sha256"], "splits": {}}
            for split in ("training", "validation"):
                selected = [r for r in self.rows if (r["cv_fold"] == fold) == (split == "validation")]
                clinical = transform_clinical_rows(selected, state)
                identities = [[r["src_subject_id"], r["side"]] for r in selected]
                buffer = io.BytesIO()
                np.save(buffer, clinical, allow_pickle=False)
                artifacts[f"clinical/fold_{fold}_{split}.npy"] = hashlib.sha256(buffer.getvalue()).hexdigest()
                artifacts[f"clinical/fold_{fold}_{split}_order.json"] = hashlib.sha256(canonical_json(identities)).hexdigest()
                record["splits"][split] = {"rows": len(selected),
                    "participants": len({r["src_subject_id"] for r in selected}),
                    "imputed_BMI_rows": sum(r["BMI"] is None for r in selected),
                    "ordered_identity_sha256": core.digest(identities),
                    "clinical_sha256": hashlib.sha256(clinical.tobytes(order="C")).hexdigest()}
            folds.append(record)
        loaders = {}
        for encoder in self.policy["image_encoders"]:
            loaders[encoder] = []
            for fold, state in enumerate(states):
                record = {"validation_fold": fold, "splits": {}}
                for split in ("training", "validation"):
                    dataset = DevelopmentFoldDataset(self.rows, self.tensors, state, encoder, split)
                    record["splits"][split] = core.check_loader(dataset, batch_size=8)
                loaders[encoder].append(record)
        public_components = ("src/knee/clinical_fold_inputs.py", "src/knee/colab_data.py",
            "src/knee/model_input_adapters.py", "src/knee/model_inputs_preflight.py",
            "configs/model_inputs_preflight_v1.json", "configs/model_registry.json",
            "scripts/prepare_model_inputs.py", "requirements-model-input-loading.txt")
        self.reference = {"version": core.VERSION, "status": "development_inputs_and_cpu_loading_verified",
            "execution_environment": "local_frozen_cpu_not_colab",
            "development_participants": 15, "development_knees": 30,
            "clinical_features": ["AGEYEARS", "SEX", "BMI"], "clinical_states_fitted": 5,
            "sources_sha256": {"development_package": sha_file(self.package_path),
                "handoff": sha_file(self.handoff_path),
                "policy": sha_file(self.project / "configs/model_inputs_preflight_v1.json"),
                "registry": sha_file(self.project / "configs/model_registry.json")},
            "implementation_sha256": {Path(relative).name: sha_file(self.project / relative)
                                       for relative in public_components},
            "folds": folds, "image_loading": loaders, "clinical_artifacts_sha256": artifacts,
            "total_loader_row_visits": 300, "loader_batch_size": 8,
            "new_image_files_written": 0, "validation_or_target_used_for_fitting": False,
            "additional_resize_crop_flip_or_augmentation": False,
            "clinical_preprocessing_fitted": True, "aligned_multimodal_input_pairs_verified": True,
            "pretrained_weights_loaded": False, "predictive_model_training_executed": False,
            "optimizer_created": False, "backward_executed": False, "reserved_test_opened": False,
            "originals_modified": False, "partitions_modified": False, "source_files_unchanged": True,
            "development_rows_and_pixels_unchanged": True}
        self.write_reference()

    @staticmethod
    def write_json(path, value):
        path.write_bytes(canonical_json(value))

    def write_reference(self):
        self.write_json(self.reference_path, self.reference)

    @staticmethod
    def tiny_probe(encoder, samples):
        if encoder not in {"image_densenet121", "image_vit_b16"} or len(samples) != 5:
            raise AssertionError("unexpected architecture probe arguments")
        width = 1024 if encoder == "image_densenet121" else 768
        for sample in samples:
            if (sample.device.type != "cpu" or sample.dtype != torch.float32
                    or tuple(sample.shape[1:]) != (3, 224, 224) or not 1 <= sample.shape[0] <= 2):
                raise AssertionError("architecture probe exceeded bounded samples")
        return {"encoder_id": encoder, "initialization": "random_weights_none",
            "pretrained_checkpoint_loaded": False, "pretrained_download_attempted": False,
            "mode": "eval_torch_inference_mode", "parameters_and_buffers_unchanged": True,
            "all_parameter_gradients_none": True, "state_sha256_before": "f" * 64,
            "state_sha256_after": "f" * 64,
            "folds": [{"validation_fold": fold, "rows": len(sample), "feature_width": width,
                       "features_sha256": hashlib.sha256(sample.numpy().tobytes()).hexdigest()}
                      for fold, sample in enumerate(samples)]}

    def run_preflight(self, *, environment="local_rehearsal"):
        with patch.object(subject, "architecture_probe", side_effect=self.tiny_probe) as probe:
            result = subject.run(self.package_dir, self.reference_path, self.project,
                                 self.output, environment)
        return result, probe

    def reject(self, *, environment="local_rehearsal"):
        with self.assertRaises((ValueError, TypeError, KeyError, FileNotFoundError)):
            self.run_preflight(environment=environment)

    def test_real_synthetic_loading_matches_every_reference_fold_without_reserved_file(self):
        before = {path: sha_file(path) for path in (self.package_path, self.handoff_path, self.reference_path)}
        result, probe = self.run_preflight()
        self.assertEqual(probe.call_count, 2)
        self.assertEqual([c.args[0] for c in probe.call_args_list], self.policy["image_encoders"])
        self.assertEqual(result["development_participants"], 15)
        self.assertEqual(result["development_knees"], 30)
        self.assertEqual(result["folds"], self.reference["folds"])
        self.assertEqual(result["image_loading"], self.reference["image_loading"])
        self.assertEqual(result["total_loader_row_visits"], 300)
        self.assertFalse(result["reserved_test_opened"])
        self.assertFalse(result["predictive_model_training_executed"])
        self.assertFalse(result["pretrained_weights_loaded"])
        self.assertEqual(before, {path: sha_file(path) for path in before})
        self.assertFalse((self.package_dir / "reserved_test_must_remain_absent.zip").exists())
        self.assertEqual(result["clinical_artifacts_sha256"], self.reference["clinical_artifacts_sha256"])
        for relative, expected_sha in result["clinical_artifacts_sha256"].items():
            self.assertEqual(sha_file(self.output / relative), expected_sha)
        self.assertEqual(len(list((self.output / "clinical").iterdir())), 25)
        self.assertEqual(json.loads((self.output / "closure_summary.json").read_text(encoding="utf-8")), result)

    def test_local_rehearsal_cannot_claim_google_colab_execution(self):
        result, _ = self.run_preflight()
        self.assertFalse(result["colab_runtime_executed"])
        self.assertTrue(result["local_rehearsal"])

    def test_local_rehearsal_remains_local_even_with_colab_environment_tag(self):
        with patch.dict(os.environ, {"COLAB_RELEASE_TAG": "synthetic_tag_for_guard_test"}):
            result, _ = self.run_preflight()
        self.assertFalse(result["colab_runtime_executed"])
        self.assertTrue(result["local_rehearsal"])

    def test_google_colab_claim_requires_actual_runtime_tag(self):
        with patch.dict(os.environ, {"COLAB_RELEASE_TAG": ""}):
            self.reject(environment="google_colab_cpu")

    def test_gpu_or_tpu_environment_is_rejected(self):
        for key in ("COLAB_GPU", "COLAB_TPU_ADDR", "TPU_NAME", "COLAB_TPU_TASK_ID"):
            with self.subTest(environment_key=key):
                with patch.dict(os.environ, {key: "1"}):
                    self.reject()

    def test_existing_nonempty_output_is_rejected_without_overwrite(self):
        sentinel = self.output / "user_existing.txt"
        sentinel.write_text("synthetic existing user output", encoding="utf-8")
        self.reject()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "synthetic existing user output")

    def test_reference_package_sha_mismatch_is_rejected(self):
        self.reference["sources_sha256"]["development_package"] = "0" * 64
        self.write_reference()
        self.reject()

    def test_reference_handoff_sha_mismatch_is_rejected(self):
        self.reference["sources_sha256"]["handoff"] = "0" * 64
        self.write_reference()
        self.reject()

    def test_reference_policy_sha_mismatch_is_rejected(self):
        self.reference["sources_sha256"]["policy"] = "0" * 64
        self.write_reference()
        self.reject()

    def test_reference_source_component_sha_mismatch_is_rejected(self):
        self.reference["implementation_sha256"]["clinical_fold_inputs.py"] = "0" * 64
        self.write_reference()
        self.reject()

    def test_reference_clinical_state_mismatch_is_rejected(self):
        self.reference["folds"][2]["clinical_state_sha256"] = "0" * 64
        self.write_reference()
        self.reject()

    def test_reference_loader_image_hash_mismatch_is_rejected(self):
        self.reference["image_loading"]["image_densenet121"][3]["splits"]["validation"]["image_sha256"] = "0" * 64
        self.write_reference()
        self.reject()

    def test_reference_loader_target_hash_mismatch_is_rejected(self):
        self.reference["image_loading"]["image_vit_b16"][1]["splits"]["training"]["targets_sha256"] = "0" * 64
        self.write_reference()
        self.reject()

    def test_reference_artifact_hash_mismatch_is_rejected(self):
        self.reference["clinical_artifacts_sha256"]["clinical/fold_0_state.json"] = "0" * 64
        self.write_reference()
        self.reject()

    def test_unknown_execution_environment_is_rejected(self):
        self.reject(environment="pretend_google_colab")

    def test_handoff_cannot_redirect_to_reserved_test_filename(self):
        self.handoff["development"]["file"] = self.handoff["reserved_test"]["file"]
        self.write_json(self.handoff_path, self.handoff)
        self.reference["sources_sha256"]["handoff"] = sha_file(self.handoff_path)
        self.write_reference()
        self.reject()

    def test_package_corruption_is_rejected(self):
        self.package_path.write_bytes(self.package_path.read_bytes() + b"synthetic tampering")
        self.reject()

    def test_cli_passes_explicit_paths_and_environment_to_same_controller(self):
        script = self.project / "scripts/check_model_inputs_colab.py"
        spec = importlib.util.spec_from_file_location("synthetic_check_model_inputs_colab_cli", script)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        arguments = [str(script), "--package-dir", str(self.package_dir),
            "--reference-record", str(self.reference_path), "--repository-root", str(self.project),
            "--output-dir", str(self.output), "--execution-environment", "local_rehearsal"]
        with patch("sys.argv", arguments), patch.object(cli, "run") as run:
            cli.main()
        run.assert_called_once_with(str(self.package_dir), str(self.reference_path),
                                    str(self.project), str(self.output), "local_rehearsal")


if __name__ == "__main__":
    unittest.main()
