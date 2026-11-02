"""Exact stage-2 interruption/recovery with real gradients on artificial pixels."""
from contextlib import redirect_stdout, redirect_stderr
import errno
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import uuid

import numpy as np
import torch

from knee import gradual_epoch_resume as subject
from knee import methodology_release as release
from knee.aligned_training import authority, data, engine
from test_model_inputs_preflight import synthetic_development
from test_methodology_augmentation_integration import request_for, synthetic_authority

ROOT = Path(__file__).resolve().parents[1]


class SyntheticInterruption(RuntimeError):
    pass


class StopAfterEpoch(subject.GradualEpochCheckpoint):
    stop_epoch = 3

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        if kwargs["epoch"] == self.stop_epoch:
            raise SyntheticInterruption("Artificial disconnection after durable epoch")


class TinyGradualImage(torch.nn.Module):
    model_id = "image_vit_b16"
    image_encoder_id = model_id
    clinical_scenario = "clinical_main"

    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2026)
            self.encoder = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(1),
                torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Dropout(.2)),
                torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4), torch.nn.Tanh()))
            self.task_head = torch.nn.Sequential(torch.nn.Dropout(.2), torch.nn.Linear(4, 1))
        self.eval()

    def forward(self, images):
        return self.task_head(self.encoder(images)).squeeze(-1)


def phase(model, value):
    if value not in {"head_only", "last_block", "full"}:
        raise AssertionError("Unknown fixture phase")
    model.encoder.requires_grad_(value == "full")
    if value == "last_block":
        model.encoder[3].requires_grad_(True)
    model.task_head.requires_grad_(True)
    if value != "full":
        model.encoder.eval()
    if value == "last_block":
        model.encoder[3].train(model.training)
    # The registered recipe keeps batch-normalization statistics frozen.
    for module in model.encoder.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def groups(model, plan, value, *, stage):
    if value != "full" or stage != "stage_2":
        raise AssertionError("All phase-2 optimizer groups must exist before freezing")
    spec = plan["radiographic_models"]["stage_2"]
    return [{"name": "task_head", "params": list(model.task_head.parameters()),
                "lr": spec["head_learning_rate"], "weight_decay": spec["weight_decay"]},
            {"name": "first_block", "params": list(model.encoder[2].parameters()),
                "lr": spec["encoder_learning_rate"] * spec["layerwise_learning_rate_decay"],
                "weight_decay": spec["weight_decay"]},
            {"name": "last_block", "params": list(model.encoder[3].parameters()),
                "lr": spec["encoder_learning_rate"], "weight_decay": spec["weight_decay"]}]


class GradualEpochResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        _, _, cls.plan, cls.registry = release.load_release(ROOT)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gr_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.request = {**request_for(model_id="image_vit_b16"), "stage": "stage_2", "selection_evidence_sha256": "c" * 64}
        approved = synthetic_authority(self.request, self.registry)
        approved["full_encoder_training_authorized"] = True
        self.permit = authority.validate_authority(approved, self.request, self.plan, self.registry)
        self.rows, self.images = synthetic_development()

    def checkpoint(self, base, *, stop=None):
        directory = self.root / base / "epoch_checkpoints" / "clinical_main__image_vit_b16__fold_0"
        value = (StopAfterEpoch if stop else subject.GradualEpochCheckpoint)(directory, self.request)
        if stop:
            value.stop_epoch = stop
        return value

    def run_fixture(self, output_name, checkpoint=None, model=None):
        fold = data.prepare_fold(self.rows, self.images, 0, encoder_id="image_vit_b16",
            package_sha256="a" * 64, methodology=True, seed=2026,
            augmentation_policy=self.plan["neural_training"]["augmentation"])
        training, validation = data.loaders(fold, seed=2026, micro_batch_size=4)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return engine.train_neural_fold(TinyGradualImage() if model is None else model,
                training, validation, self.plan, device="cpu", seed=2026, stage="stage_2",
                positive_weight=fold.class_weight["positive_weight"], output_dir=self.root / output_name,
                phase_callback=phase, groups_callback=groups, execution_authority=self.permit,
                epoch_checkpoint=checkpoint)

    def assert_tree_equal(self, first, second):
        self.assertEqual(type(first), type(second))
        if isinstance(first, torch.Tensor):
            torch.testing.assert_close(first, second, atol=0, rtol=0)
        elif isinstance(first, dict):
            self.assertEqual(set(first), set(second))
            for key in first:
                self.assert_tree_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(first, second)

    def latest_state(self, checkpoint):
        return torch.load(checkpoint._latest()[1] / "state.pt", weights_only=True)

    def interrupt(self, base, epoch=3):
        checkpoint = self.checkpoint(base, stop=epoch)
        with self.assertRaises(SyntheticInterruption):
            self.run_fixture(base + "_interrupted", checkpoint)
        return checkpoint

    def test_exact_real_updates_resume_after_epochs3_6_11_including_phase_transitions(self):
        pixels = {key: value.tobytes() for key, value in self.images.items()}
        full_checkpoint = self.checkpoint("full")
        baseline = self.run_fixture("full_output", full_checkpoint)
        baseline_state = self.latest_state(full_checkpoint)
        self.assertGreaterEqual(baseline["completed_epochs"], 19)
        initial = TinyGradualImage().state_dict()
        for at_epoch in (3, 6, 11):
            with self.subTest(at_epoch=at_epoch):
                base = f"resume_{at_epoch}"
                interrupted = self.interrupt(base, at_epoch)
                saved = self.latest_state(interrupted)
                self.assertEqual(saved["phase"], engine.epoch_phase("image_vit_b16", "stage_2", at_epoch))
                self.assertEqual(saved["scheduler_epoch"], at_epoch)
                self.assertEqual(saved["best_model"] is None, at_epoch <= 10)
                self.assertEqual(saved["best_predictions"] is None, at_epoch <= 10)
                if at_epoch == 3:
                    for name in initial:
                        if name.startswith("encoder."):
                            torch.testing.assert_close(saved["current_model"][name], initial[name], atol=0, rtol=0)
                if at_epoch >= 6:
                    self.assertFalse(torch.equal(saved["current_model"]["encoder.3.0.weight"], initial["encoder.3.0.weight"]))
                if at_epoch == 11:
                    self.assertFalse(torch.equal(saved["current_model"]["encoder.2.0.weight"], initial["encoder.2.0.weight"]))
                else:
                    self.assertTrue(torch.equal(saved["current_model"]["encoder.2.0.weight"], initial["encoder.2.0.weight"]))
                random.random(); np.random.random(); torch.rand(7)
                visited = []
                original = engine._execute_training_epoch
                def tracked(*args, **kwargs):
                    visited.append(args[1].dataset.epoch)
                    return original(*args, **kwargs)
                continued_checkpoint = self.checkpoint(base)
                with patch.object(engine, "_execute_training_epoch", side_effect=tracked):
                    continued = self.run_fixture(base + "_continued", continued_checkpoint)
                self.assertEqual(visited, list(range(at_epoch + 1, baseline["completed_epochs"] + 1)))
                self.assertEqual(continued["resumed_from_epoch"], at_epoch)
                for key in ("history", "best_epoch", "best_validation_ap", "completed_epochs", "optimizer_updates", "checkpoint_sha256"):
                    self.assertEqual(continued[key], baseline[key], key)
                resumed_state = self.latest_state(continued_checkpoint)
                for key in ("current_model", "best_model", "optimizer", "scaler", "rng", "monitor", "history", "total_updates", "trainable_parameters"):
                    self.assert_tree_equal(resumed_state[key], baseline_state[key])
                self.assertEqual(json.loads((self.root / (base + "_continued") / "best_validation_predictions.json").read_text()),
                    json.loads((self.root / "full_output/best_validation_predictions.json").read_text()))
                self.assertEqual(len(continued_checkpoint._committed()), 2)
        self.assertEqual(pixels, {key: value.tobytes() for key, value in self.images.items()})

    def test_default_and_checkpoint_runs_have_exact_same_training_results(self):
        default = self.run_fixture("default")
        checkpointed = self.run_fixture("checkpointed", self.checkpoint("checkpointed"))
        self.assertFalse(default["resume_supported"])
        self.assertFalse(default["optimizer_state_saved"])
        self.assertNotIn("epoch_checkpoint_version", default)
        self.assertEqual(default["history"], checkpointed["history"])
        self.assertEqual(default["checkpoint_sha256"], checkpointed["checkpoint_sha256"])
        self.assertEqual(checkpointed["epoch_checkpoint_version"], subject.VERSION)
        self.assertEqual(checkpointed["epoch_checkpoint_scope"], subject.SCOPE)

    def test_keep_only_two_committed_generations_and_ignore_incomplete_writes(self):
        checkpoint = self.interrupt("bounded", 6)
        self.assertEqual([item[0] for item in checkpoint._committed()], [5, 6])
        incomplete = checkpoint.directory / (".writing_" + uuid.uuid4().hex)
        incomplete.mkdir()
        (incomplete / "state.pt").write_bytes(b"ARTIFICIAL UNCOMMITTED PARTIAL FILE")
        retirement = checkpoint.directory / (".retiring_" + uuid.uuid4().hex)
        retirement.mkdir()
        self.assertEqual(checkpoint._latest()[0], 6)
        continued = self.run_fixture("bounded_continued", self.checkpoint("bounded"))
        self.assertEqual(continued["resumed_from_epoch"], 6)
        self.assertEqual(len(checkpoint._committed()), 2)
        self.assertTrue(incomplete.exists())

    def test_interrupted_write_keeps_last_committed_epoch_and_duplicate_epoch_rejected(self):
        checkpoint = self.interrupt("write_failure", 3)
        initial_commit = checkpoint._latest()
        with patch.object(subject, "_sync", side_effect=OSError("Artificial disconnect during new write")), \
             self.assertRaisesRegex(OSError, "Artificial disconnect"):
            self.run_fixture("failed_save", self.checkpoint("write_failure"))
        self.assertEqual(checkpoint._latest(), initial_commit)
        self.assertTrue(any(path.name.startswith(".writing_") for path in checkpoint.directory.iterdir()))
        continued = self.run_fixture("after_write_failure", self.checkpoint("write_failure"))
        self.assertEqual(continued["resumed_from_epoch"], 3)
        # Repeating an epoch number cannot replace a committed recovery point.
        with self.assertRaisesRegex(ValueError, "next_uncommitted"):
            checkpoint.save(None, None, None, None, None, None, None,
                epoch=continued["completed_epochs"], history=[], total_updates=0,
                best_state=None, best_predictions=None, stopped=False)

    def test_completed_checkpoint_finalizes_without_any_new_update(self):
        first = self.run_fixture("completed", self.checkpoint("completed"))
        with patch.object(engine, "_execute_training_epoch", side_effect=AssertionError("No new update permitted")):
            again = self.run_fixture("finalized", self.checkpoint("completed"))
        self.assertEqual(first["checkpoint_sha256"], again["checkpoint_sha256"])
        self.assertEqual(again["resumed_from_epoch"], first["completed_epochs"])

    def test_tampered_state_changed_runtime_or_changed_initial_weights_fail_before_updates(self):
        for issue in ("state", "runtime", "initial"):
            with self.subTest(issue=issue):
                checkpoint = self.interrupt(issue, 3)
                model = TinyGradualImage()
                if issue == "state":
                    path = checkpoint._latest()[1] / "state.pt"
                    path.write_bytes(path.read_bytes() + b"CHANGED")
                if issue == "initial":
                    with torch.no_grad():
                        next(model.encoder.parameters()).add_(1)
                original_runtime = subject._runtime
                runtime = (lambda device: {**original_runtime(device), "torch": "changed"}) if issue == "runtime" else original_runtime
                pattern = "committed_gradual_state_hash_mismatch" if issue == "state" else "same_gradual_recipe_runtime_initial"
                with patch.object(subject, "_runtime", side_effect=runtime), \
                     patch.object(engine, "_execute_training_epoch", side_effect=AssertionError("No update allowed")), \
                     self.assertRaisesRegex(ValueError, pattern):
                    self.run_fixture(issue + "_retry", self.checkpoint(issue), model)

    def test_exact_model_scope_and_authority_are_required(self):
        for model_id in subject.MODELS:
            request = {**self.request, "model_id": model_id, "image_encoder_id": model_id}
            directory = self.root / model_id / "epoch_checkpoints" / f"clinical_main__{model_id}__fold_0"
            subject.GradualEpochCheckpoint(directory, request)
        for change in ({"operation": "single_fold_smoke"}, {"stage": "stage_1"}, {"seed": 2027},
                {"model_id": "image_biomedclip", "image_encoder_id": "image_biomedclip"},
                {"model_id": "image_convnextv2_tiny", "image_encoder_id": "image_convnextv2_tiny"},
                {"clinical_scenario": "clinical_kl"}, {"selection_evidence_sha256": None}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "only_four_selected_stage2"):
                subject.GradualEpochCheckpoint(self.root / "wrong", {**self.request, **change})
        with self.assertRaisesRegex(ValueError, "private_gradual_job"):
            subject.GradualEpochCheckpoint(self.root / "wrong", self.request)
        with self.assertRaisesRegex(ValueError, "selected_full_stage2_authority"):
            self.checkpoint("smoke").validate(self.permit, smoke=True, stage="stage_2")

    def test_sync_tolerates_only_unsupported_fuse_operation(self):
        stream = unittest.mock.Mock()
        with patch.object(subject.os, "fsync", side_effect=OSError(errno.EINVAL, "Unsupported FUSE sync")):
            subject._sync(stream)
        stream.flush.assert_called_once()
        with patch.object(subject.os, "fsync", side_effect=OSError(errno.ENOSPC, "Full disk")), self.assertRaises(OSError):
            subject._sync(stream)

    def test_save_never_deserializes_full_state_and_reads_payload_only_twice_per_epoch(self):
        checkpoint = self.checkpoint("bounded_io")
        original_sha = subject.sha_file
        payload_reads = []
        def tracked(path):
            if Path(path).name == "state.pt":
                payload_reads.append(Path(path))
            return original_sha(path)
        with patch.object(subject.torch, "load", side_effect=AssertionError("No extra full CPU copy during save")), \
             patch.object(subject, "sha_file", side_effect=tracked):
            result = self.run_fixture("bounded_io_output", checkpoint)
        self.assertEqual(len(payload_reads), 2 * result["completed_epochs"])
        self.assertEqual(len(checkpoint._committed()), 2)

    def test_real_runner_binds_authenticated_stage2_selection_to_rows_and_checkpoint(self):
        from knee import final_models_preflight, gradual_selection
        from knee.aligned_training import models, runner
        from knee.colab_data import sha_file
        from test_methodology_campaign import synthetic_preflight
        from test_gradual_selection import screening_sources
        gradual_selection._VALIDATED.clear()
        sources = self.root / "artificial_sources"
        sources.mkdir()
        value = synthetic_preflight(sources)
        screening = self.root / "artificial_screening"
        screening.mkdir()
        screening_roots, _ = screening_sources(screening, value, self.plan, self.registry)
        evidence = self.root / "evidence"
        evidence.mkdir()
        selection_path = evidence / "selection.json"
        selection = gradual_selection.prepare_selection(screening_roots["40"], screening_roots["41"],
            selection_path, self.plan, self.registry, value["rows"],
            authority_path=screening / gradual_selection.AUTHORITY_FILENAME)
        request = {**self.request, "development_package_sha256": value["package_sha256"],
            "selection_evidence_sha256": sha_file(selection_path)}
        approved = synthetic_authority(request, self.registry)
        approved["full_encoder_training_authorized"] = True
        weights = self.root / "weights"
        weights.mkdir()
        output = self.root / "runner_output"
        checkpoint = subject.GradualEpochCheckpoint(self.root / "runner_recovery" / "epoch_checkpoints"
            / "clinical_main__image_vit_b16__fold_0", request)
        original_engine = engine.train_neural_fold
        def fixture_engine(*args, **kwargs):
            self.assertIs(kwargs["epoch_checkpoint"], checkpoint)
            self.assertEqual(kwargs["stage"], "stage_2")
            return original_engine(*args, **kwargs, phase_callback=phase, groups_callback=groups)
        with patch.object(final_models_preflight, "checkpoint_files", return_value=[]), \
             patch.object(models, "build_model", side_effect=lambda *args, **kwargs: (TinyGradualImage(), {"synthetic": True})), \
             patch.object(engine, "train_neural_fold", side_effect=fixture_engine), \
             patch.object(runner, "validate_selection", side_effect=AssertionError("Legacy/fusion selection must stay separate")), \
             redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            result = runner.execute(request, approved, repository_root=ROOT,
                package_dir=value["package"], reference_path=value["reference_path"], weights_root=weights,
                output_dir=output, selection_evidence=selection_path, epoch_checkpoint=checkpoint)
        self.assertEqual(result["selection_evidence"], gradual_selection.validate_selection_evidence(
            selection_path, request, self.plan, self.registry, value["rows"]))
        self.assertEqual(result["request_sha256"], authority.digest(request))
        self.assertEqual(runner.completed_job(output, value["package_sha256"], self.plan, self.registry), result)
        self.assertFalse(result["reserved_test_accessed"])
        self.assertEqual(runner.read_json(output / "training_record.json")["epoch_checkpoint_version"], subject.VERSION)

        bad_selection = {**selection, "promoted_modern_models": ["image_convnextv2_tiny", "image_biomedclip"]}
        bad_path = evidence / "bad_selection.json"
        runner.save_json(bad_path, bad_selection)
        bad_request = {**request, "selection_evidence_sha256": sha_file(bad_path)}
        bad_approved = synthetic_authority(bad_request, self.registry)
        bad_approved["full_encoder_training_authorized"] = True
        bad_checkpoint = subject.GradualEpochCheckpoint(self.root / "bad_recovery" / "epoch_checkpoints"
            / "clinical_main__image_vit_b16__fold_0", bad_request)
        with patch.object(models, "build_model", side_effect=AssertionError("No model before valid selection")), \
             self.assertRaisesRegex(ValueError, "artifact_must_equal"):
            runner.execute(bad_request, bad_approved, repository_root=ROOT,
                package_dir=value["package"], reference_path=value["reference_path"], weights_root=weights,
                output_dir=self.root / "rejected_output", selection_evidence=bad_path, epoch_checkpoint=bad_checkpoint)
        self.assertFalse((self.root / "rejected_output").exists())


if __name__ == "__main__":
    unittest.main()
