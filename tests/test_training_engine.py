"""Preparation tests: pure accounting/math and mocked training orchestration.

No test performs an optimizer update, backward pass, or participant-data fit.
"""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from knee import training_engine as e

ROOT = Path(__file__).resolve().parents[1]


def plan():
    return json.loads((ROOT / "configs/training_plan_v1_1.json").read_text(encoding="utf-8"))


class EngineMathTests(unittest.TestCase):
    def test_full_immutable_recipe_is_authenticated(self):
        self.assertTrue(e.validate_recipe(plan()))
        changed = plan()
        changed["neural_training"]["maximum_epochs"] = 2
        with self.assertRaisesRegex(ValueError, "approved_training_recipe_required"):
            e.validate_recipe(changed)

    def test_warmup_and_final_cosine_endpoints(self):
        self.assertEqual([e.learning_rate_factor(x) for x in range(1, 6)], [.2, .4, .6, .8, 1])
        self.assertAlmostEqual(e.learning_rate_factor(60), .01)
        values = [e.learning_rate_factor(x) for x in range(5, 61)]
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))
        self.assertAlmostEqual(e.learning_rate_factor(6), .01 + .99 * (1 + math.cos(math.pi / 55)) / 2)

    def test_invalid_schedule_indices_rejected(self):
        for bad in (0, 61, True, 2.0):
            with self.subTest(epoch=bad), self.assertRaises(ValueError):
                e.learning_rate_factor(bad)

    def test_frozen_image_probe_never_unfreezes(self):
        self.assertEqual({e.epoch_phase("image_densenet121", "stage_1", x) for x in range(1, 61)}, {"head_only"})

    def test_staged_unfreezing_exact_epoch_boundaries(self):
        expected = ["head_only"] * 5 + ["last_block"] * 5 + ["full"] * 50
        self.assertEqual([e.epoch_phase("image_vit_b16", "stage_2", x) for x in range(1, 61)], expected)
        self.assertEqual([e.epoch_phase("fusion_intermediate_concat", "fusion", x) for x in range(1, 61)], expected)
        self.assertEqual(e.epoch_phase("clinical_mlp", "stage_1", 60), "full")

    def test_wrong_model_stage_pair_rejected(self):
        for model, stage in (("clinical_mlp", "stage_2"), ("image_vit_b16", "fusion"),
                             ("fusion_intermediate_concat", "stage_1")):
            with self.subTest(model=model, stage=stage), self.assertRaises(ValueError):
                e.epoch_phase(model, stage, 1)

    def test_partial_accumulation_uses_actual_knees(self):
        self.assertEqual(e.accumulation_denominators([4, 4, 4, 4, 4, 3]),
                         [((4, 4, 4, 4), 16), ((4, 3), 7)])
        self.assertEqual(e.accumulation_denominators([2] * 9 + [1]),
                         [((2,) * 8, 16), ((2, 1), 3)])
        self.assertEqual(e.accumulation_denominators([1] * 17),
                         [((1,) * 16, 16), ((1,), 1)])

    def test_accumulated_weighted_loss_matches_knee_mean(self):
        # Pure BCE arithmetic: no tensors, gradients, optimizer, or fit.
        targets = np.array([0, 1, 0, 1, 1, 0, 1])
        logits = np.array([-.4, .2, -.3, 1.2, .3, -.7, -.2])
        weight = 2.5
        losses = (1 - targets) * np.logaddexp(0, logits) + weight * targets * np.logaddexp(0, -logits)
        actual_denominator = e.accumulation_denominators([4, 3])[0][1]
        accumulated = float(losses[:4].sum() / actual_denominator + losses[4:].sum() / actual_denominator)
        self.assertAlmostEqual(accumulated, float(losses.mean()))
        self.assertNotAlmostEqual(accumulated, float((losses[:4].mean() + losses[4:].mean()) / 4))

    def test_crossing_accumulation_boundary_is_not_silently_rebatched(self):
        for sizes in ([4, 4, 4, 3, 4], [0], [True], [17]):
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                e.accumulation_denominators(sizes)

    def test_exact_ap_tie_keeps_earliest_epoch_and_counts_patience(self):
        monitor = e.EarlyStopping()
        self.assertEqual(monitor.update(1, .7), (True, False))
        for epoch in range(2, 9):
            self.assertEqual(monitor.update(epoch, .7), (False, False))
        self.assertEqual(monitor.update(9, .7), (False, True))
        self.assertEqual((monitor.best_epoch, monitor.best_score), (1, .7))

    def test_strict_improvement_resets_patience(self):
        monitor = e.EarlyStopping()
        monitor.update(1, .7)
        monitor.update(2, .6)
        self.assertEqual(monitor.update(3, .70000000001), (True, False))
        self.assertEqual(monitor.bad_epochs, 0)
        self.assertEqual(monitor.best_epoch, 3)

    def test_stage_two_monitor_starts_after_epoch_ten(self):
        monitor = e.EarlyStopping(monitor_starts_after_epoch=10)
        for epoch in range(1, 11):
            self.assertEqual(monitor.update(epoch, 1.0), (False, False))
        self.assertIsNone(monitor.best_score)
        self.assertEqual(monitor.update(11, .5), (True, False))
        self.assertEqual(monitor.best_epoch, 11)

    def test_unweighted_five_fold_means_do_not_accept_pooled_scores(self):
        rows = [{"validation_fold": x, "average_precision_score": .1 * (x + 1),
                 "roc_auc": .6, "brier": .2} for x in range(5)]
        self.assertAlmostEqual(e.aggregate_fold_metrics(rows)["average_precision_score"], .3)
        for bad in (rows[:-1], rows + rows[:1], [dict(row, knees=100) for row in rows]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                e.aggregate_fold_metrics(bad)
        repeated = deepcopy(rows)
        repeated[-1]["validation_fold"] = 0
        with self.assertRaises(ValueError):
            e.aggregate_fold_metrics(repeated)

    def test_no_authority_cannot_construct_optimizer_or_change_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Mock(model_id="clinical_mlp")
            with patch.object(torch.optim, "AdamW", side_effect=AssertionError("optimizer forbidden")):
                with self.assertRaisesRegex(ValueError, "explicit_training_permit_required_before_execution"):
                    e.train_neural_fold(model, None, None, plan(), device="cpu", seed=2026,
                                        positive_weight=1, output_dir=Path(tmp) / "not_created")
            model.to.assert_not_called()
            self.assertFalse((Path(tmp) / "not_created").exists())

    def test_new_output_cannot_overwrite_or_enter_public_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                e._new_output(Path(tmp))
        with self.assertRaises(ValueError):
            e._new_output(ROOT / "new_training_output")
        with self.assertRaises(ValueError):
            e._new_output(Path("relative_output"))

    def test_deterministic_cublas_configuration_before_cuda_initialization(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(torch.cuda, "is_initialized", return_value=False):
            e._prepare_cuda_environment()
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        with patch.dict(os.environ, {}, clear=True), patch.object(torch.cuda, "is_initialized", return_value=True):
            with self.assertRaisesRegex(ValueError, "restart_cuda_runtime_with_cublas"):
                e._prepare_cuda_environment()
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}, clear=True), \
             patch.object(torch.cuda, "is_initialized", return_value=True):
            e._prepare_cuda_environment()

    def test_binary_targets_and_logit_alignment_reject_bad_batches(self):
        for target in (torch.tensor([0, 2]), torch.tensor([0., float("nan")]), torch.zeros(2, 1)):
            with self.subTest(target=target), self.assertRaises(ValueError):
                e._target_count({"target": target})
        self.assertEqual(e._target_count({"target": torch.tensor([0, 1])}), 2)

    def test_binary_metric_contract_rejects_one_class_before_sklearn_call(self):
        # The fake metrics prevent a dependency-only test from fitting anything.
        fake = SimpleNamespace(average_precision_score=Mock(), brier_score_loss=Mock(), roc_auc_score=Mock())
        with patch.dict("sys.modules", {"sklearn.metrics": fake}):
            for y, z in (([0, 0], [0, 1]), ([0, 1], [float("nan"), 1]), ([0, 1], [0])):
                with self.assertRaises(ValueError):
                    e.binary_metrics(y, z)
        fake.average_precision_score.assert_not_called()

    def test_stable_sigmoid_and_sklearn_ap_api_with_ties(self):
        # Metric calls are mocked, no estimator is created; inspect exact inputs.
        fake = SimpleNamespace(average_precision_score=Mock(return_value=.5),
                               brier_score_loss=Mock(return_value=.25), roc_auc_score=Mock(return_value=.5))
        with patch.dict("sys.modules", {"sklearn.metrics": fake}):
            result = e.binary_metrics([0, 1, 0, 1], [-1000, 1000, 0, 0])
        self.assertEqual(result, {"average_precision_score": .5, "roc_auc": .5, "brier": .25})
        np.testing.assert_array_equal(fake.average_precision_score.call_args.args[1], [0, 1, .5, .5])
        np.testing.assert_array_equal(fake.roc_auc_score.call_args.args[1], [0, 1, .5, .5])

    def test_actual_sklearn_metrics_for_ties_and_perfect_ranking_without_estimator(self):
        tied = e.binary_metrics([0, 1, 0, 1], [0, 0, 0, 0])
        self.assertEqual(tied, {"average_precision_score": .5, "roc_auc": .5, "brier": .25})
        perfect = e.binary_metrics([0, 0, 1, 1], [-1000, -1000, 1000, 1000])
        self.assertEqual(perfect, {"average_precision_score": 1., "roc_auc": 1., "brier": 0.})

    def test_forward_passes_only_modalities_required_by_image_model(self):
        model = Mock(model_id="image_densenet121", return_value=torch.tensor([.2, -.3]))
        images, clinical = torch.zeros(2, 3, 224, 224), torch.zeros(2, 3)
        e._forward(model, {"target": torch.tensor([0, 1]), "image": images, "clinical": clinical}, torch.device("cpu"))
        self.assertEqual(set(model.call_args.kwargs), {"images"})


class BindingTests(unittest.TestCase):
    def _loaders(self):
        from knee.training_data import TrainingFoldDataset
        from torch.utils.data import DataLoader
        identities = [["S001", "LEFT"], ["S002", "LEFT"], ["S003", "LEFT"]]
        state = {"validation_fold": 0, "train_identity_sha256": e._digest(identities)}
        train = [{"src_subject_id": subject, "side": side, "PROGRESION_48M": target}
                 for (subject, side), target in zip(identities, [0, 0, 1])]
        val = [{"src_subject_id": subject, "side": "LEFT", "PROGRESION_48M": target}
               for subject, target in (("S004", 0), ("S005", 1))]
        datasets = [TrainingFoldDataset(rows, {}, np.zeros((len(rows), 3), dtype=np.float32), state,
                                       split, None, "a" * 64) for rows, split in ((train, "training"), (val, "validation"))]
        permit = SimpleNamespace(package_sha256="a" * 64, validation_fold=0, image_encoder_id=None,
                                 seed=2026, request_sha256="b" * 64)
        generator = torch.Generator().manual_seed(2026)
        return (DataLoader(datasets[0], batch_size=64, shuffle=True, generator=generator),
                DataLoader(datasets[1], batch_size=64, shuffle=False), permit)

    def test_actual_dataset_bindings_checked_without_loading_rows(self):
        train, val, permit = self._loaders()
        record = e._validate_loader_bindings(train, val, permit, 2)
        self.assertEqual(record["training_knees"], 3)
        self.assertEqual(record["trained_participants"], ["S001", "S002", "S003"])

    def test_validation_labels_cannot_determine_positive_weight(self):
        train, val, permit = self._loaders()
        with self.assertRaisesRegex(ValueError, "class_weight_must_be_computed_from_training_targets_only"):
            e._validate_loader_bindings(train, val, permit, 1)

    def test_package_fold_identity_overlap_and_shuffle_mutations_rejected(self):
        for mutation in ("package", "fold", "identity", "overlap", "shuffle"):
            train, val, permit = self._loaders()
            if mutation == "package":
                val.dataset.package_sha256 = "c" * 64
            elif mutation == "fold":
                val.dataset.validation_fold = 1
            elif mutation == "identity":
                val.dataset.train_identity_sha256 = "c" * 64
            elif mutation == "overlap":
                val.dataset.identities = (("S001", "RIGHT"), ("S005", "LEFT"))
            else:
                train.generator.manual_seed(2027)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                e._validate_loader_bindings(train, val, permit, 2)


class MockedEngineLifecycleTests(unittest.TestCase):
    def _execute_mocked(self, tmp, *, stage="stage_1", smoke=False):
        model_id = "image_densenet121"
        model = torch.nn.Linear(1, 1)
        model.model_id = model_id
        model.image_encoder_id = model_id
        # No model.forward, loss.backward, optimizer construction/update is used.
        optimizer = SimpleNamespace(param_groups=[{"lr": .001, "initial_lr": .001}])
        loader = SimpleNamespace(batch_size=4, drop_last=False, num_workers=0,
                                 dataset=SimpleNamespace(identities=(("S001", "LEFT"), ("S002", "LEFT"))),
                                 generator=torch.Generator().manual_seed(2026))
        phase = Mock()
        groups = Mock(return_value=[{"params": list(model.parameters()), "lr": .001, "weight_decay": .0001}])
        run_epoch = Mock(return_value={"weighted_loss_per_knee": .5, "training_knees_seen": 16, "optimizer_updates": 1})
        evaluate = Mock(return_value=({"average_precision_score": .7, "roc_auc": .8, "brier": .2},
                                      {"targets": [0, 1], "logits": [-1, 1]}))
        with patch("knee.training_authority.require_permit"), \
             patch.object(e, "_new_optimizer", return_value=optimizer), \
             patch.object(e, "_execute_training_epoch", run_epoch), \
             patch.object(e, "_evaluate", evaluate), \
             patch.object(e, "_validate_loader_bindings", return_value={"synthetic_mock": True}), \
             patch.object(e, "_write_checkpoint", return_value="a" * 64), \
             patch.object(torch.optim, "AdamW", side_effect=AssertionError("real optimizer forbidden")), \
             patch.object(torch.Tensor, "backward", side_effect=AssertionError("backward forbidden")):
            result = e.train_neural_fold(model, loader, loader, plan(), device="cpu", seed=2026,
                positive_weight=2, stage=stage, output_dir=Path(tmp) / "synthetic_mock",
                phase_callback=phase, groups_callback=groups, maximum_steps=1 if smoke else None,
                smoke=smoke, execution_authority=SimpleNamespace(seed=2026,
                    image_micro_batch_size=4, image_encoder_id=model_id))
        return result, phase, groups, run_epoch

    def test_mocked_loop_restores_earliest_best_and_explicitly_no_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _, _, execute = self._execute_mocked(tmp)
            self.assertEqual((result["best_epoch"], result["completed_epochs"]), (1, 9))
            self.assertEqual(execute.call_count, 9)
            self.assertTrue(result["restored_best_checkpoint"])
            self.assertFalse(result["resume_supported"])
            self.assertFalse(result["optimizer_state_saved"])
            self.assertFalse(result["gpu_resource_profile_recorded"])
            self.assertFalse(result["resource_profile"]["cuda_peak_measured"])
            self.assertEqual(json.loads((Path(tmp) / "synthetic_mock/training_record.json").read_text())["best_epoch"], 1)

    def test_mocked_stage_two_does_not_stop_before_unfreezing_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, phase, groups, _ = self._execute_mocked(tmp, stage="stage_2")
            self.assertEqual((result["best_epoch"], result["completed_epochs"]), (11, 19))
            self.assertEqual(groups.call_args.kwargs["stage"], "stage_2")
            self.assertIn("full", [call.args[1] for call in phase.call_args_list])

    def test_mocked_smoke_is_bounded_and_ineligible_for_model_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _, _, execute = self._execute_mocked(tmp, smoke=True)
            self.assertEqual(result["optimizer_updates"], 1)
            self.assertFalse(result["eligible_for_model_selection"])
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(result["status"], "bounded_training_smoke_complete")

    def test_authorized_seed_cannot_be_changed_inside_engine(self):
        with tempfile.TemporaryDirectory() as tmp, patch("knee.training_authority.require_permit"):
            with self.assertRaisesRegex(ValueError, "training_seed_must_match_execution_authority"):
                e.train_neural_fold(Mock(model_id="clinical_mlp"), None, None, plan(), device="cpu",
                    seed=2027, positive_weight=1, output_dir=Path(tmp) / "new",
                    execution_authority=SimpleNamespace(seed=2026))


if __name__ == "__main__":
    unittest.main()
