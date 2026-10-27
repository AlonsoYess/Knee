"""Real CPU gradient updates on artificial images prove exact epoch recovery."""
from pathlib import Path
import json
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from knee import methodology_release as release
from knee import training_epoch_resume as subject
from knee.aligned_training import authority, data, engine
from test_model_inputs_preflight import synthetic_development
from test_methodology_augmentation_integration import request_for, synthetic_authority

ROOT = Path(__file__).resolve().parents[1]


class SyntheticInterruption(RuntimeError):
    pass


class StopAfterEpoch(subject.EpochCheckpoint):
    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        if kwargs["epoch"] == 3:
            raise SyntheticInterruption("Synthetic disconnection after committed epoch three")


class TinyFrozenImage(torch.nn.Module):
    model_id = "image_vit_b16"
    image_encoder_id = model_id
    clinical_scenario = "clinical_main"

    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2026)
            self.encoder = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(1),
                torch.nn.Linear(3, 4))
            # Dropout deliberately exercises global RNG recovery on a tiny fixture.
            self.task_head = torch.nn.Sequential(torch.nn.Dropout(.2), torch.nn.Linear(4, 1))
        self.encoder.requires_grad_(False)
        self.eval()

    def forward(self, images):
        return self.task_head(self.encoder(images)).squeeze(-1)


def phase(model, value):
    if value != "head_only":
        raise AssertionError("Only frozen encoder fixtures are permitted")
    model.encoder.requires_grad_(False).eval()
    model.task_head.requires_grad_(True)


def groups(model, plan, value, *, stage):
    if value != "head_only" or stage != "stage_1":
        raise AssertionError("Only the registered frozen stage is exercised")
    spec = plan["radiographic_models"]["stage_1"]
    return [{"name": "task_head", "params": list(model.task_head.parameters()),
        "lr": spec["head_learning_rate"], "weight_decay": spec["head_weight_decay"]}]


class TrainingEpochResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        _, _, cls.plan, cls.registry = release.load_release(ROOT)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="er_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.request = request_for(model_id="image_vit_b16")
        self.permit = authority.validate_authority(synthetic_authority(self.request, self.registry),
            self.request, self.plan, self.registry)
        self.rows, self.images = synthetic_development()

    def checkpoint(self, base, checkpoint_class=subject.EpochCheckpoint):
        directory = self.root / base / "epoch_checkpoints" / "clinical_main__image_vit_b16__fold_0"
        return checkpoint_class(directory, self.request)

    def run_fixture(self, output_name, checkpoint=None, model=None):
        model = TinyFrozenImage() if model is None else model
        fold = data.prepare_fold(self.rows, self.images, 0, encoder_id="image_vit_b16",
            package_sha256="a" * 64, methodology=True, seed=2026,
            augmentation_policy=self.plan["neural_training"]["augmentation"])
        train, validation = data.loaders(fold, seed=2026, micro_batch_size=4)
        return engine.train_neural_fold(model, train, validation, self.plan, device="cpu", seed=2026,
            positive_weight=fold.class_weight["positive_weight"], output_dir=self.root / output_name,
            phase_callback=phase, groups_callback=groups, execution_authority=self.permit,
            epoch_checkpoint=checkpoint)

    def assert_tree_equal(self, left, right):
        self.assertEqual(type(left), type(right))
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_real_cpu_updates_resume_exactly_without_replaying_completed_epochs(self):
        original_pixels = {key: image.tobytes() for key, image in self.images.items()}
        baseline_checkpoint = self.checkpoint("baseline")
        baseline = self.run_fixture("full", baseline_checkpoint)
        interrupted = self.checkpoint("resumed", StopAfterEpoch)
        with self.assertRaises(SyntheticInterruption):
            self.run_fixture("interrupted", interrupted)
        abandoned = interrupted.directory / ".writing_abandoned"
        abandoned.mkdir()
        (abandoned / "state.pt").write_bytes(b"INCOMPLETE SYNTHETIC WRITE")
        # A different caller RNG state and newly constructed model/loaders must
        # not change the restored run's next shuffle, dropout or gamma epoch.
        random.random()
        np.random.random()
        torch.rand(7)
        resumed_checkpoint = self.checkpoint("resumed")
        visited_epochs = []
        execute_epoch = engine._execute_training_epoch
        def observed(*args, **kwargs):
            visited_epochs.append(args[1].dataset.epoch)
            return execute_epoch(*args, **kwargs)
        with patch.object(engine, "_execute_training_epoch", side_effect=observed):
            resumed = self.run_fixture("continued", resumed_checkpoint)
        self.assertEqual(visited_epochs, list(range(4, baseline["completed_epochs"] + 1)))
        self.assertEqual(resumed["resumed_from_epoch"], 3)
        self.assertEqual(resumed["history"], baseline["history"])
        for key in ("best_epoch", "best_validation_ap", "completed_epochs", "optimizer_updates", "checkpoint_sha256"):
            self.assertEqual(resumed[key], baseline[key], key)
        self.assertEqual(json.loads((self.root / "continued/best_validation_predictions.json").read_text()),
            json.loads((self.root / "full/best_validation_predictions.json").read_text()))
        full_state = torch.load(baseline_checkpoint._latest()[1] / "state.pt", weights_only=True)
        resumed_state = torch.load(resumed_checkpoint._latest()[1] / "state.pt", weights_only=True)
        for key in ("head", "best_head", "optimizer", "scaler", "rng", "monitor", "history", "total_updates"):
            self.assert_tree_equal(full_state[key], resumed_state[key])
        self.assertTrue(all(name.startswith("task_head.") for name in full_state["head"]))
        self.assertNotIn("encoder", full_state)
        self.assertLess((baseline_checkpoint._latest()[1] / "state.pt").stat().st_size, 1024**2)
        self.assertEqual(original_pixels, {key: image.tobytes() for key, image in self.images.items()})
        self.assertTrue(abandoned.is_dir())

    def test_default_execution_preserves_historical_nonresume_record(self):
        result = self.run_fixture("default")
        self.assertFalse(result["resume_supported"])
        self.assertFalse(result["optimizer_state_saved"])
        self.assertNotIn("epoch_checkpoint_version", result)
        self.assertFalse((self.root / "epoch_checkpoints").exists())

    def test_checkpoint_from_completed_early_stop_finalizes_without_more_updates(self):
        checkpoint = self.checkpoint("finished")
        first = self.run_fixture("first", checkpoint)
        with patch.object(engine, "_execute_training_epoch", side_effect=AssertionError("No extra epoch permitted")):
            restored = self.run_fixture("again", self.checkpoint("finished"))
        self.assertEqual(restored["checkpoint_sha256"], first["checkpoint_sha256"])
        self.assertEqual(restored["resumed_from_epoch"], first["completed_epochs"])

    def test_tampered_committed_state_is_rejected_before_any_new_update(self):
        checkpoint = self.checkpoint("tampered", StopAfterEpoch)
        with self.assertRaises(SyntheticInterruption):
            self.run_fixture("first", checkpoint)
        state = checkpoint._latest()[1] / "state.pt"
        state.write_bytes(state.read_bytes() + b"TAMPERED")
        with patch.object(engine, "_execute_training_epoch", side_effect=AssertionError("No epoch permitted")), \
             self.assertRaisesRegex(ValueError, "committed_epoch_state_hash_mismatch"):
            self.run_fixture("retry", self.checkpoint("tampered"))

    def test_changed_encoder_is_rejected_before_any_new_update(self):
        checkpoint = self.checkpoint("changed", StopAfterEpoch)
        with self.assertRaises(SyntheticInterruption):
            self.run_fixture("first", checkpoint)
        model = TinyFrozenImage()
        with torch.no_grad():
            next(model.encoder.parameters()).add_(1)
        with patch.object(engine, "_execute_training_epoch", side_effect=AssertionError("No epoch permitted")), \
             self.assertRaisesRegex(ValueError, "same_recipe_runtime_frozen_encoder"):
            self.run_fixture("retry", self.checkpoint("changed"), model)

    def test_changed_runtime_cannot_resume_an_exact_checkpoint(self):
        checkpoint = self.checkpoint("runtime", StopAfterEpoch)
        with self.assertRaises(SyntheticInterruption):
            self.run_fixture("first", checkpoint)
        original = subject._runtime
        with patch.object(subject, "_runtime", side_effect=lambda device: {**original(device), "torch": "different-runtime"}), \
             patch.object(engine, "_execute_training_epoch", side_effect=AssertionError("No epoch permitted")), \
             self.assertRaisesRegex(ValueError, "same_recipe_runtime_frozen_encoder"):
            self.run_fixture("retry", self.checkpoint("runtime"))

    def test_smoke_clinical_dense_unfreezing_and_wrong_job_paths_are_rejected(self):
        changes = ({"operation": "single_fold_smoke"}, {"model_id": "clinical_mlp"},
            {"model_id": "image_densenet121", "image_encoder_id": "image_densenet121"},
            {"stage": "stage_2"}, {"seed": 2027})
        for change in changes:
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "only_five_frozen"):
                subject.EpochCheckpoint(self.root / "invalid", {**self.request, **change})
        with self.assertRaisesRegex(ValueError, "private_job_specific"):
            subject.EpochCheckpoint(self.root / "wrong_job", self.request)
        checkpoint = self.checkpoint("wrong_authority")
        with self.assertRaisesRegex(ValueError, "exact_full_fold_authority"):
            checkpoint.validate(self.permit, smoke=True, stage="stage_1")


if __name__ == "__main__":
    unittest.main()
