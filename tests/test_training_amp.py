"""Deterministic AMP overflow regressions using CPU tensors and mocked updates.

No test creates an optimizer, invokes Tensor.backward, updates a parameter, or
opens participant data. CUDA, backward, and optimizer steps are test doubles.
"""
from contextlib import ExitStack, nullcontext
import random
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from knee import training_engine as engine


class _BufferModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.25], dtype=torch.float32))
        self.register_buffer("integer_counter", torch.zeros((), dtype=torch.int64))
        self.register_buffer("floating_counter", torch.zeros((), dtype=torch.float32))
        self.model_id = "image_densenet121"


class _ScalerDouble:
    """Public GradScaler protocol, without automatic differentiation or CUDA."""
    def __init__(self, fixture, *, enabled=True, initial_scale=65536., bad_backoff=None):
        self.fixture, self.enabled = fixture, enabled
        self.value, self.bad_backoff = float(initial_scale), bad_backoff
        self.found_inf = False
        self.scale = Mock(side_effect=self._scale)
        self.unscale_ = Mock(side_effect=self._unscale)
        self.step = Mock(side_effect=self._step)
        self.update = Mock(side_effect=self._update)
        self.get_scale = Mock(side_effect=lambda: self.value)
        self.is_enabled = Mock(return_value=enabled)

    def _scale(self, loss):
        return SimpleNamespace(backward=Mock(side_effect=self.fixture.fake_backward))

    def _unscale(self, optimizer):
        self.found_inf = any(p.grad is not None and not torch.isfinite(p.grad).all().item()
                             for group in optimizer.param_groups for p in group["params"])

    def _step(self, optimizer):
        # Matching GradScaler: an overflow never calls optimizer.step().
        if not self.found_inf:
            optimizer.step()

    def _update(self, new_scale=None):
        if new_scale is not None:
            self.value = float(new_scale)
        elif self.found_inf:
            if self.bad_backoff == "unchanged":
                pass
            elif self.bad_backoff == "increase":
                self.value *= 2
            elif self.bad_backoff == "zero":
                self.value = 0.
            elif self.bad_backoff == "nan":
                self.value = float("nan")
            else:
                self.value *= .5
        self.found_inf = False


class _ReplayFixture:
    """Inject finite/overflow gradients and observe all replayable state."""
    def __init__(self, outcomes, *, enabled=True, initial_scale=65536., bad_backoff=None):
        self.outcomes = list(outcomes)
        self.model = _BufferModel()
        self.initial_weight = self.model.weight.detach().clone()
        self.attempt = -1
        self.micro_index = 0
        self.observations = []
        self.cuda_state = torch.tensor([17, 23, 42], dtype=torch.uint8)
        self.optimizer = SimpleNamespace(param_groups=[{"params": [self.model.weight]}])
        self.optimizer.step = Mock()  # Deliberately does not update any tensor.
        self.optimizer.zero_grad = Mock(side_effect=self._zero_grad)
        self.scaler = _ScalerDouble(self, enabled=enabled, initial_scale=initial_scale,
                                    bad_backoff=bad_backoff)
        self.clip = Mock(return_value=torch.tensor(.25))
        self.cpu_device = torch.device("cpu")
        self.cuda_device = torch.device("cuda:0")

    def _zero_grad(self, *, set_to_none=True):
        self.model.weight.grad = None

    def _forward(self, model, batch, device):
        if batch["micro"] == 0:
            self.attempt += 1
        self.micro_index = batch["micro"]
        self.observations.append({"attempt": self.attempt, "identity": batch["identity"],
            "torch_mask": torch.rand(5).clone(), "python_draw": random.random(),
            "numpy_draw": float(np.random.random()), "cuda_state": self.cuda_state.clone(),
            "integer_buffer": int(model.integer_counter),
            "floating_buffer": float(model.floating_counter)})
        self.cuda_state.add_(1)
        model.integer_counter.add_(1)
        model.floating_counter.add_(.5)
        return torch.zeros(2), batch["target"]

    def fake_backward(self):
        overflow = self.outcomes[min(self.attempt, len(self.outcomes) - 1)]
        value = float("inf") if overflow and self.micro_index == 7 else .125
        gradient = torch.full_like(self.model.weight, value)
        if self.model.weight.grad is None:
            self.model.weight.grad = gradient
        else:
            self.model.weight.grad.add_(gradient)

    def _restore_cuda(self, states, *args, **kwargs):
        self.cuda_state = (states[0] if isinstance(states, list) else states).clone()

    def batches(self, group=0):
        return [{"target": torch.tensor([0, 1]), "micro": i,
                 "identity": (group, i)} for i in range(8)]

    def patches(self):
        stack = ExitStack()
        stack.enter_context(patch.object(engine, "_forward", side_effect=self._forward))
        stack.enter_context(patch.object(torch, "autocast", side_effect=lambda **kw: nullcontext()))
        stack.enter_context(patch.object(torch.nn.functional, "binary_cross_entropy_with_logits",
                                        side_effect=lambda *args, **kw: torch.tensor(2.)))
        stack.enter_context(patch.object(torch.nn.utils, "clip_grad_norm_", self.clip))
        stack.enter_context(patch.object(torch.cuda, "get_rng_state_all",
                                        side_effect=lambda: [self.cuda_state.clone()]))
        stack.enter_context(patch.object(torch.cuda, "set_rng_state_all", side_effect=self._restore_cuda))
        stack.enter_context(patch.object(torch.cuda, "get_rng_state",
                                        side_effect=lambda *args, **kw: self.cuda_state.clone()))
        stack.enter_context(patch.object(torch.cuda, "set_rng_state", side_effect=self._restore_cuda))
        stack.enter_context(patch.object(torch.Tensor, "backward",
                                        side_effect=AssertionError("real backward forbidden")))
        stack.enter_context(patch.object(torch.optim, "AdamW",
                                        side_effect=AssertionError("real optimizer forbidden")))
        return stack

    def group(self, *, group=0, device=None):
        return engine._execute_effective_group(self.model, self.batches(group), 16,
            self.optimizer, self.scaler, device or self.cuda_device, torch.tensor(2.))


class AmpEffectiveGroupReplayTests(unittest.TestCase):
    def setUp(self):
        self.cpu_state = torch.get_rng_state()
        self.python_state = random.getstate()
        self.numpy_state = np.random.get_state()

    def tearDown(self):
        torch.set_rng_state(self.cpu_state)
        random.setstate(self.python_state)
        np.random.set_state(self.numpy_state)

    def test_finite_group_keeps_nonfinite_clipping_protection_and_one_update(self):
        fixture = _ReplayFixture([False])
        with fixture.patches():
            result = fixture.group()
        self.assertEqual(result["summed_loss"], 16.)
        self.assertEqual(result["amp_overflow_replays"], 0)
        fixture.optimizer.step.assert_called_once_with()
        self.assertEqual(fixture.clip.call_args.kwargs,
                         {"max_norm": 1.0, "error_if_nonfinite": True})
        torch.testing.assert_close(fixture.model.weight, fixture.initial_weight)

    def test_overflow_replays_identical_group_without_counting_failed_update(self):
        fixture = _ReplayFixture([True, False])
        with fixture.patches():
            result = fixture.group()
        self.assertEqual(result["summed_loss"], 16.)
        self.assertEqual(result["amp_overflow_replays"], 1)
        self.assertEqual(result["amp_scale_audit"], {"start_scale": 65536., "successful_scale": 32768.,
                         "scale_after_update": 32768., "backoffs": 1, "attempts": 2,
                         "gradient_norm_before_clip": .25})
        fixture.optimizer.step.assert_called_once_with()
        self.assertEqual(fixture.scaler.step.call_count, 2)
        self.assertEqual(fixture.clip.call_count, 1)
        self.assertEqual([row["identity"] for row in fixture.observations[:8]],
                         [row["identity"] for row in fixture.observations[8:]])
        torch.testing.assert_close(fixture.model.weight, fixture.initial_weight)

    def test_replay_restores_dropout_rng_and_mutable_integer_and_float_buffers(self):
        fixture = _ReplayFixture([True, False])
        with fixture.patches():
            fixture.group()
        for original, replay in zip(fixture.observations[:8], fixture.observations[8:]):
            self.assertEqual(original["python_draw"], replay["python_draw"])
            self.assertEqual(original["numpy_draw"], replay["numpy_draw"])
            torch.testing.assert_close(original["torch_mask"], replay["torch_mask"], rtol=0, atol=0)
            torch.testing.assert_close(original["cuda_state"], replay["cuda_state"], rtol=0, atol=0)
            self.assertEqual(original["integer_buffer"], replay["integer_buffer"])
            self.assertEqual(original["floating_buffer"], replay["floating_buffer"])
        self.assertEqual(int(fixture.model.integer_counter), 8)
        self.assertEqual(float(fixture.model.floating_counter), 4.)

    def test_sixteen_backoffs_can_recover_at_scale_one(self):
        fixture = _ReplayFixture([True] * 16 + [False])
        with fixture.patches():
            result = fixture.group()
        self.assertEqual(engine.AMP_MAX_BACKOFFS, 16)
        self.assertEqual(result["amp_overflow_replays"], 16)
        self.assertEqual(result["amp_scale_audit"]["successful_scale"], 1.)
        self.assertEqual(result["amp_scale_audit"]["attempts"], 17)
        fixture.optimizer.step.assert_called_once_with()
        self.assertEqual(int(fixture.model.integer_counter), 8)

    def test_persistent_nonfinite_gradients_stop_bounded_without_parameter_updates(self):
        fixture = _ReplayFixture([True])
        cpu_before = torch.get_rng_state().clone()
        python_before, numpy_before = random.getstate(), np.random.get_state()
        cuda_before = fixture.cuda_state.clone()
        with fixture.patches(), self.assertRaisesRegex(ValueError, "amp_nonfinite_gradients_persist_after_bounded_replay"):
            fixture.group()
        self.assertEqual(fixture.attempt + 1, 17)
        fixture.optimizer.step.assert_not_called()
        fixture.clip.assert_not_called()
        self.assertIsNone(fixture.model.weight.grad)
        self.assertEqual(int(fixture.model.integer_counter), 0)
        self.assertEqual(float(fixture.model.floating_counter), 0.)
        torch.testing.assert_close(torch.get_rng_state(), cpu_before, rtol=0, atol=0)
        self.assertEqual(random.getstate(), python_before)
        self.assertEqual(np.random.get_state()[0], numpy_before[0])
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_before[1])
        self.assertEqual(np.random.get_state()[2:], numpy_before[2:])
        torch.testing.assert_close(fixture.cuda_state, cuda_before, rtol=0, atol=0)
        torch.testing.assert_close(fixture.model.weight, fixture.initial_weight)

    def test_invalid_scaler_backoff_never_allows_optimizer_update(self):
        for mode in ("unchanged", "increase", "zero", "nan"):
            with self.subTest(mode=mode):
                fixture = _ReplayFixture([True], bad_backoff=mode)
                with fixture.patches(), self.assertRaises(ValueError):
                    fixture.group()
                fixture.optimizer.step.assert_not_called()
                fixture.clip.assert_not_called()
                self.assertEqual(fixture.attempt + 1, 1)
                torch.testing.assert_close(fixture.model.weight, fixture.initial_weight)

    def test_cpu_nonfinite_gradients_do_not_enter_amp_recovery(self):
        fixture = _ReplayFixture([True], enabled=False, initial_scale=1.)
        with fixture.patches(), self.assertRaisesRegex(ValueError, "nonfinite_unscaled_gradients_without_amp_recovery"):
            fixture.group(device=fixture.cpu_device)
        fixture.optimizer.step.assert_not_called()
        fixture.scaler.step.assert_not_called()
        fixture.scaler.update.assert_not_called()
        self.assertEqual(fixture.attempt + 1, 1)

    def test_disabled_cuda_scaler_cannot_retry_bad_gradients(self):
        fixture = _ReplayFixture([True], enabled=False)
        with fixture.patches(), self.assertRaisesRegex(ValueError, "nonfinite_unscaled_gradients_without_amp_recovery"):
            fixture.group()
        fixture.optimizer.step.assert_not_called()
        fixture.scaler.update.assert_not_called()

    def test_nonfinite_training_loss_is_fatal_without_scaler_recovery(self):
        fixture = _ReplayFixture([False])
        with fixture.patches(), patch.object(torch.nn.functional, "binary_cross_entropy_with_logits",
                return_value=torch.tensor(float("nan"))), \
                self.assertRaisesRegex(ValueError, "finite_weighted_training_loss_required"):
            fixture.group()
        fixture.scaler.scale.assert_not_called()
        fixture.scaler.step.assert_not_called()
        fixture.scaler.update.assert_not_called()
        fixture.optimizer.step.assert_not_called()

    def test_finite_element_gradients_but_nonfinite_total_norm_remain_fatal(self):
        fixture = _ReplayFixture([False])
        fixture.clip.return_value = torch.tensor(float("inf"))
        with fixture.patches(), self.assertRaises(ValueError):
            fixture.group()
        fixture.clip.assert_called_once()
        self.assertTrue(fixture.clip.call_args.kwargs["error_if_nonfinite"])
        fixture.scaler.step.assert_not_called()
        fixture.optimizer.step.assert_not_called()

    def test_missing_gradients_cannot_count_as_a_successful_update(self):
        fixture = _ReplayFixture([False])
        fixture.fake_backward = Mock()
        with fixture.patches(), self.assertRaises(ValueError):
            fixture.group()
        fixture.optimizer.step.assert_not_called()
        fixture.clip.assert_not_called()

    def test_later_overflow_does_not_repeat_or_discard_prior_successful_group(self):
        fixture = _ReplayFixture([False, True, False])
        with fixture.patches():
            first = fixture.group(group=0)
            second = fixture.group(group=1)
        self.assertEqual((first["amp_overflow_replays"], second["amp_overflow_replays"]), (0, 1))
        self.assertEqual(fixture.optimizer.step.call_count, 2)
        self.assertEqual(int(fixture.model.integer_counter), 16)
        self.assertEqual([row["identity"][0] for row in fixture.observations], [0] * 8 + [1] * 16)
        torch.testing.assert_close(fixture.model.weight, fixture.initial_weight)

    def test_epoch_counts_two_successful_updates_and_32_knees_not_failed_attempts(self):
        fixture = _ReplayFixture([True, False, False])
        loader = fixture.batches(0) + fixture.batches(1) + fixture.batches(2)
        real_tensor = torch.tensor

        def cpu_tensor(*args, **kwargs):
            if kwargs.get("device") == fixture.cuda_device:
                kwargs["device"] = fixture.cpu_device
            return real_tensor(*args, **kwargs)

        with fixture.patches(), patch.object(torch, "tensor", side_effect=cpu_tensor):
            result = engine._execute_training_epoch(fixture.model, loader, fixture.optimizer,
                fixture.scaler, fixture.cuda_device, 2., 16, maximum_updates=2)
        self.assertEqual(result["optimizer_updates"], 2)
        self.assertEqual(result["training_knees_seen"], 32)
        self.assertEqual(result["weighted_loss_per_knee"], 1.)
        self.assertEqual(result["amp_overflow_replays"], 1)
        self.assertEqual(len(result["amp_scale_audit"]), 2)
        self.assertEqual(fixture.optimizer.step.call_count, 2)
        self.assertEqual(int(fixture.model.integer_counter), 16)
        self.assertEqual({row["identity"][0] for row in fixture.observations}, {0, 1})
        torch.testing.assert_close(fixture.model.weight, fixture.initial_weight)


if __name__ == "__main__":
    unittest.main()
