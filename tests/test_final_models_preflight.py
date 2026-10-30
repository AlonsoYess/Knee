"""No-data/no-training release preflight integration and failure guards."""
from copy import deepcopy
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from knee import final_models_preflight as subject
from knee.final_model_release import load_release

ROOT = Path(__file__).resolve().parents[1]


class SyntheticEncoder(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.register_buffer("offset", torch.tensor(.5))
    def forward(self, batch):
        return (batch.mean((1, 2, 3)) + self.offset)[:, None].repeat(1, self.width)


class FinalModelsPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.catalog, _, _ = load_release(ROOT)

    def test_api_has_no_dataset_or_reserved_package_argument(self):
        self.assertEqual(set(inspect.signature(subject.run).parameters), {"repository_root", "weights_root", "output_dir", "allow_download", "execution_environment"})

    def test_all_normalizers_only_change_channels_and_intensity(self):
        for model_id in self.catalog["models"]:
            batch = subject.synthetic_batch(model_id, self.catalog)
            self.assertEqual(batch.shape, (2, 3, 224, 224))
            self.assertEqual(batch.dtype, torch.float32)
            self.assertTrue(torch.isfinite(batch).all())

    def test_biomedclip_does_not_silently_use_imagenet_normalization(self):
        bio = subject.synthetic_batch("image_biomedclip", self.catalog)
        other = subject.synthetic_batch("image_mambaout_tiny", self.catalog)
        self.assertFalse(torch.equal(bio, other))

    def test_tampered_catalog_fails_before_any_download(self):
        changed = deepcopy(self.catalog)
        changed["models"]["image_biomedclip"]["files"][0]["url"] = "https://example.com/weights"
        with patch.object(subject, "acquire") as acquire, self.assertRaises(ValueError):
            subject.checkpoint_files("image_biomedclip", changed, "unused", True)
        acquire.assert_not_called()

    def test_gated_model_is_not_downloaded_even_with_download_flag(self):
        with patch.object(subject, "acquire") as acquire, self.assertRaisesRegex(ValueError, "checkpoint_access_or_review_pending"):
            subject.checkpoint_files("image_dinov3_vits16", self.catalog, "unused", True)
        acquire.assert_not_called()

    def test_missing_cache_requires_explicit_download_flag(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(subject, "acquire") as acquire, self.assertRaisesRegex(ValueError, "checkpoint_not_cached"):
            subject.checkpoint_files("image_biomedclip", self.catalog, temp)
        acquire.assert_not_called()

    def test_non_boolean_download_authority_rejected(self):
        with self.assertRaises(ValueError):
            subject.checkpoint_files("image_biomedclip", self.catalog, "unused", "yes")

    def test_unknown_model_rejected(self):
        with self.assertRaises(ValueError):
            subject.synthetic_batch("image_unknown", self.catalog)

    def test_local_run_cannot_claim_colab(self):
        with patch.dict(subject.os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, "actual_colab_runtime_required"):
            subject.verify_environment("google_colab_cpu")

    def test_environment_rejects_wrong_dependency_and_gpu_runtime(self):
        versions = deepcopy(subject.REQUIRED_PACKAGES)
        versions["timm"] = "0.0"
        with patch.object(subject.importlib.metadata, "version", side_effect=versions.__getitem__), self.assertRaises(ValueError):
            subject.verify_environment("local_cpu")
        with patch.object(torch.cuda, "is_available", return_value=True), self.assertRaises(ValueError):
            subject.verify_environment("local_cpu")

    def test_output_cannot_overwrite_previous_results_or_write_in_repo(self):
        with tempfile.TemporaryDirectory() as temp:
            for output in (temp, ROOT / "never-created-final-output"):
                with patch.object(subject, "verify_environment", return_value={}), self.assertRaises(ValueError):
                    subject.run(ROOT, Path(temp) / "weights", output)

    def test_end_to_end_mocked_checkpoints_probes_without_optimizer_or_backward(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            fake = base / "synthetic.bin"
            fake.write_bytes(b"synthetic")
            def loader(model_id, paths):
                return SyntheticEncoder(self.catalog["models"][model_id]["feature_dimension"]).eval().requires_grad_(False), {"synthetic_test_loader": True}
            with patch.object(subject, "verify_environment", return_value={"synthetic_test": True}), patch.object(subject, "checkpoint_files", return_value=[fake]), patch.object(subject, "load_encoder", side_effect=loader), patch("knee.pretrained_final_encoders.load_final_encoder", side_effect=loader), patch.object(torch.optim.Optimizer, "__init__", side_effect=AssertionError("optimizer forbidden")), patch.object(torch.Tensor, "backward", side_effect=AssertionError("backward forbidden")):
                result = subject.run(ROOT, base / "weights", base / "output")
            self.assertEqual(result["pretrained_models_loaded"], 6)
            self.assertEqual(result["registered_image_models"], 7)
            self.assertEqual(result["pending_model_ids"], ["image_dinov3_vits16"])
            self.assertEqual(result, json.loads((base / "output/checkpoint_verification.json").read_text(encoding="utf-8")))
            for field in ("training_executed", "participant_data_opened", "optimizer_created", "gradients_created", "originals_modified", "partitions_modified", "reserved_test_predictive_evaluation_opened", "training_runner_implemented"):
                self.assertIs(result[field], False)

    def test_failed_probe_cannot_leave_a_success_record(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "output"
            with patch.object(subject, "verify_environment", return_value={}), patch.object(subject, "checkpoint_files", side_effect=ValueError("wrong_hash")), self.assertRaisesRegex(ValueError, "wrong_hash"):
                subject.run(ROOT, Path(temp) / "weights", output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
