"""Synthetic recovery and fail-closed checks for the external runtime launcher."""
from contextlib import nullcontext, redirect_stderr
from copy import deepcopy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(os.environ.get("KNEE_FROZEN_REPOSITORY", Path(__file__).resolve().parents[1]))
SUPPORT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("gradual_saved_runtime_launcher_test",
    SUPPORT_ROOT / "scripts/launch_gradual_with_saved_runtime.py")
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def saved_runtime(**changes):
    return {"python": "3.12.14", "torch": "2.6.0+cu124", "cuda": "12.4", "device": "cuda:0",
        "cpu_threads": 2, "interop_threads": 2, "deterministic": True,
        "deterministic_warn_only": False, "cublas_workspace_config": ":4096:8",
        "matmul_allow_tf32": False, "cudnn_allow_tf32": True, "cudnn_benchmark": False,
        "cudnn_deterministic": False, "device_name": "Tesla T4", "device_capability": [7, 5], **changes}


class FakeTorch:
    def __init__(self, cpu=8, interop=8):
        self.cpu, self.interop = cpu, interop
        self.calls = []
    def get_num_threads(self):
        return self.cpu
    def get_num_interop_threads(self):
        return self.interop
    def set_num_threads(self, value):
        self.calls.append(("cpu", value))
        self.cpu = value
    def set_num_interop_threads(self, value):
        self.calls.append(("interop", value))
        self.interop = value
    def device(self, value):
        return value


class SavedRuntimeLauncherTests(unittest.TestCase):
    def test_saved_runtime_is_copied_as_primitive_metadata(self):
        original = saved_runtime()
        checked = subject.validate_runtime(original)
        checked["device_capability"][0] = 1
        self.assertEqual(original["device_capability"], [7, 5])

    def test_runtime_missing_extra_and_tensor_like_fields_rejected(self):
        malformed = [None, {}, {**saved_runtime(), "extra": True},
            {key: value for key, value in saved_runtime().items() if key != "torch"},
            saved_runtime(cpu_threads=True), saved_runtime(interop_threads=0),
            saved_runtime(cpu_threads=1025), saved_runtime(device="cpu"),
            saved_runtime(device_capability=[7, True]), saved_runtime(device_capability=[7])]
        for value in malformed:
            with self.subTest(value=value), self.assertRaises(ValueError):
                subject.validate_runtime(value)

    def test_threads_restored_before_other_operations_and_interop_first(self):
        torch = FakeTorch()
        before = subject.configure_threads(torch, saved_runtime())
        self.assertEqual(before, {"cpu_threads": 8, "interop_threads": 8})
        self.assertEqual(torch.calls, [("interop", 2), ("cpu", 2)])
        self.assertEqual((torch.cpu, torch.interop), (2, 2))

    def test_equal_thread_settings_do_not_repeat_one_shot_interop_setter(self):
        torch = FakeTorch(2, 2)
        subject.configure_threads(torch, saved_runtime())
        self.assertEqual(torch.calls, [])

    def test_interop_initialization_error_requires_new_process(self):
        torch = FakeTorch()
        torch.set_num_interop_threads = Mock(side_effect=RuntimeError("already initialized"))
        with self.assertRaisesRegex(RuntimeError, "fresh_process_required"):
            subject.configure_threads(torch, saved_runtime())
        self.assertEqual(torch.calls, [])

    def test_fresh_real_cpu_process_sets_both_thread_pools_before_tensor_work(self):
        code = """import importlib.util,json,torch
spec=importlib.util.spec_from_file_location('external', %r)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
before=m.configure_threads(torch, {'cpu_threads':1,'interop_threads':1})
x=torch.tensor([2.0],requires_grad=True);(x*x).sum().backward()
print(json.dumps({'threads':torch.get_num_threads(),'interop':torch.get_num_interop_threads(),'gradient':x.grad.item()}))
""" % str(SUPPORT_ROOT / "scripts/launch_gradual_with_saved_runtime.py")
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", COLAB_GPU="0", PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-B", "-c", code], text=True,
            capture_output=True, env=environment, check=True, timeout=60)
        self.assertEqual(json.loads(result.stdout), {"threads": 1, "interop": 1, "gradient": 4.0})

    def _compatibility(self, current):
        torch = FakeTorch()
        with patch("knee.training_smoke.verify_gpu_environment", return_value={}) as verify, \
             patch("knee.aligned_training.engine._seeded_execution", return_value=nullcontext()) as seeded, \
             patch("knee.training_epoch_resume._runtime", return_value=current), \
             redirect_stderr(io.StringIO()) as stream:
            inspection = {"status": "authenticated_saved_runtime", "runtime": saved_runtime(),
                "checkpoints": [{"key": "clinical_main__image_skelex__fold_0", "epoch": 18}]}
            result = subject.compatibility_before_training(torch, inspection)
            verify.assert_called_once_with("google_colab_gpu")
            seeded.assert_called_once_with(2026)
            return result, stream.getvalue()

    def test_only_cpu_concurrency_difference_is_repaired_and_reported(self):
        result, output = self._compatibility(saved_runtime())
        self.assertEqual(result["differences_after"], {})
        self.assertEqual(set(result["differences_before"]), set(subject.THREAD_KEYS))
        self.assertFalse(result["training_started"])
        self.assertIn('"epoch": 18', output)

    def test_gpu_versions_and_other_runtime_differences_are_never_normalized(self):
        for changes in ({"device_name": "NVIDIA A100"}, {"torch": "new"}, {"python": "3.13"},
                        {"deterministic": False}, {"cudnn_allow_tf32": False}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "incompatible_details_recorded"):
                self._compatibility(saved_runtime(**changes))

    def test_no_checkpoint_leaves_runtime_untouched(self):
        torch = FakeTorch()
        report = subject.compatibility_before_training(torch,
            {"status": "no_saved_checkpoint", "checkpoints": []})
        self.assertEqual(torch.calls, [])
        self.assertFalse(report["training_started"])

    def test_checkpoint_hash_validation_precedes_deserialization(self):
        checkpoint = SimpleNamespace(identity={"synthetic": True},
            _checked_state_path=Mock(side_effect=ValueError("committed_gradual_state_hash_mismatch")))
        with patch("torch.load") as load, redirect_stderr(io.StringIO()), \
             self.assertRaisesRegex(ValueError, "hash_mismatch"):
            subject.checkpoint_metadata(checkpoint, (18, Path("generation"), {}))
        load.assert_not_called()

    def test_checkpoint_reader_uses_weights_only_cpu_mmap_and_returns_no_tensors(self):
        checkpoint = SimpleNamespace(identity={"synthetic": True},
            _checked_state_path=Mock(return_value=Path("safe_state.pt")))
        payload = {"identity": checkpoint.identity, "epoch": 18, "runtime": saved_runtime(),
            "current_model": {"opaque": object()}}
        with patch("torch.load", return_value=payload) as load, redirect_stderr(io.StringIO()):
            result = subject.checkpoint_metadata(checkpoint, (18, Path("generation"), {}))
        load.assert_called_once_with(Path("safe_state.pt"), map_location="cpu", weights_only=True, mmap=True)
        self.assertEqual(set(result), {"epoch", "runtime"})
        json.dumps(result, allow_nan=False)

    def test_checkpoint_payload_identity_and_epoch_mismatch_are_rejected(self):
        checkpoint = SimpleNamespace(identity={"synthetic": True},
            _checked_state_path=Mock(return_value=Path("safe_state.pt")))
        for change in ({"identity": {}}, {"epoch": 17}):
            payload = {"identity": checkpoint.identity, "epoch": 18, "runtime": saved_runtime(), **change}
            with self.subTest(change=change), patch("torch.load", return_value=payload), \
                 redirect_stderr(io.StringIO()), self.assertRaisesRegex(ValueError, "identity_and_epoch"):
                subject.checkpoint_metadata(checkpoint, (18, Path("generation"), {}))

    def test_lookup_never_creates_a_campaign_when_no_match_exists(self):
        core = SimpleNamespace(VERSION="synthetic", transport=SimpleNamespace(MAX_DIRECT_CHILDREN=10))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIsNone(subject.find_campaign(root, {"development_package_sha256": "a"}, core))
            self.assertEqual(list(root.iterdir()), [])

    def test_lookup_rejects_ambiguous_and_incompatible_existing_campaigns(self):
        core = SimpleNamespace(VERSION="synthetic", transport=SimpleNamespace(MAX_DIRECT_CHILDREN=10))
        identity = {"development_package_sha256": "a", "code_revision": "same"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in (1, 2):
                child = root / ("gradual_image_campaign_" + str(index))
                child.mkdir()
                (child / "campaign_identity.json").write_text(json.dumps({"version": "synthetic", "identity": identity}))
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                subject.find_campaign(root, identity, core)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                subject.find_campaign(root, {**identity, "code_revision": "changed"}, core)

    def _inspection_fixture(self):
        from knee import gradual_campaign as core, gradual_campaign_colab as orchestration
        from knee.training_authority import digest
        from knee.colab_data import sha_file
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        records, proof, campaign = [root / name for name in ("records", "proof", "campaign")]
        for path in (records, proof, campaign):
            path.mkdir()
        reference = root / "reference.json"
        reference.write_text("{}")
        (proof / core.SELECTION_FILENAME).write_text("{}")
        prepared = {"development_package_sha256": "a" * 64, "source_files_sha256": {"synthetic": "b" * 64},
            "model_input_reference_sha256": sha_file(reference)}
        prepared["record_sha256"] = digest(prepared)
        (proof / core.PREPARATION_FILENAME).write_text(json.dumps(prepared))
        (campaign / (core.MODELS[0] + "__batch.json")).write_text('{"micro_batch_size":1}')
        for fold in (0, 1):
            (campaign / "epoch_checkpoints" / f"clinical_main__{core.MODELS[0]}__fold_{fold}").mkdir(parents=True)
        args = SimpleNamespace(repository_root=ROOT, records_root=records, preparation_dir=proof,
            code_revision="f" * 40, reference_path=reference)
        return core, orchestration, args, prepared, campaign

    def test_inspection_rejects_conflicting_pending_runtime_profiles(self):
        core, orchestration, args, prepared, campaign = self._inspection_fixture()
        checkpoint = SimpleNamespace(_latest=lambda: (18, Path("generation"), {}))
        with patch.object(orchestration, "validate_preparation_receipt"), \
             patch.object(core, "_sources", return_value=prepared["source_files_sha256"]), \
             patch.object(subject, "find_campaign", return_value=campaign), \
             patch("knee.gradual_epoch_resume.GradualEpochCheckpoint", return_value=checkpoint), \
             patch.object(subject, "checkpoint_metadata", side_effect=[
                 {"epoch": 18, "runtime": saved_runtime()}, {"epoch": 18, "runtime": saved_runtime(cpu_threads=4)}]), \
             self.assertRaisesRegex(ValueError, "conflicting_saved"):
            subject.inspect_saved_runtime(args)

    def test_inspection_returns_small_metadata_only_and_preserves_checkpoint_inventory(self):
        core, orchestration, args, prepared, campaign = self._inspection_fixture()
        before = sorted(str(path.relative_to(campaign)) for path in campaign.rglob("*"))
        checkpoint = SimpleNamespace(_latest=lambda: (18, Path("generation"), {}))
        with patch.object(orchestration, "validate_preparation_receipt"), \
             patch.object(core, "_sources", return_value=prepared["source_files_sha256"]), \
             patch.object(subject, "find_campaign", return_value=campaign), \
             patch("knee.gradual_epoch_resume.GradualEpochCheckpoint", return_value=checkpoint), \
             patch.object(subject, "checkpoint_metadata", return_value={"epoch": 18, "runtime": saved_runtime()}):
            result = subject.inspect_saved_runtime(args)
        self.assertEqual(len(result["checkpoints"]), 2)
        self.assertNotIn("current_model", json.dumps(result))
        self.assertEqual(before, sorted(str(path.relative_to(campaign)) for path in campaign.rglob("*")))

    def test_metadata_reader_is_a_separate_gpu_hidden_process(self):
        output = {"version": subject.VERSION, "status": "authenticated_saved_runtime",
            "runtime": saved_runtime(), "checkpoints": []}
        with patch.object(subject.subprocess, "run", return_value=SimpleNamespace(returncode=0,
                stdout=json.dumps(output))) as run:
            result = subject.read_in_separate_process(["--operation", "train"])
        self.assertEqual(result, output)
        command, = run.call_args.args
        self.assertEqual(command[0], sys.executable)
        self.assertIn("--runtime-inspection-only", command)
        environment = run.call_args.kwargs["env"]
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(environment["COLAB_GPU"], "0")

    def test_failed_inspection_stops_before_training(self):
        with patch.object(subject.subprocess, "run", return_value=SimpleNamespace(returncode=1, stdout="")), \
             self.assertRaisesRegex(ValueError, "inspection_failed_before_training"):
            subject.read_in_separate_process([])

    def _argv(self):
        args = ["--operation", "train", "--campaign", "gradual_four_models", "--code-revision", "f" * 40]
        for name in ("repository-root", "records-root", "preparation-dir", "package-dir", "reference-path", "weights-root", "output-dir"):
            args += ["--" + name, str(ROOT if name == "repository-root" else ROOT / "synthetic" / name)]
        return args

    def test_frozen_cli_receives_original_arguments_after_preflight(self):
        argv = self._argv()
        events = []
        def preflight(*args):
            events.append("preflight")
        def frozen(path, **kwargs):
            events.append("frozen")
            self.assertEqual(Path(path), ROOT / "scripts/train_gradual_campaign_colab.py")
            self.assertEqual(sys.argv[1:], argv)
            self.assertEqual(kwargs, {"run_name": "__main__"})
        previous = list(sys.argv)
        with patch.object(subject, "read_in_separate_process", return_value={"status": "no_saved_checkpoint", "checkpoints": []}), \
             patch.object(subject, "compatibility_before_training", side_effect=preflight), \
             patch.object(subject.runpy, "run_path", side_effect=frozen):
            subject.main(argv)
        self.assertEqual(events, ["preflight", "frozen"])
        self.assertEqual(sys.argv, previous)

    def test_incompatible_runtime_prevents_frozen_cli_dispatch(self):
        with patch.object(subject, "read_in_separate_process", return_value={}), \
             patch.object(subject, "compatibility_before_training", side_effect=ValueError("incompatible")), \
             patch.object(subject.runpy, "run_path") as frozen, self.assertRaisesRegex(ValueError, "incompatible"):
            subject.main(self._argv())
        frozen.assert_not_called()

    def test_diagnostic_failure_preserves_original_training_error_and_arguments(self):
        original = ValueError("same_gradual_recipe_runtime_initial_weights_and_epoch_state_required")
        previous = list(sys.argv)
        with patch.object(subject, "read_in_separate_process", return_value={}), \
             patch.object(subject, "compatibility_before_training"), \
             patch.object(subject.runpy, "run_path", side_effect=original), \
             patch.object(subject, "report_restore_failure", side_effect=TypeError("malformed history")), \
             redirect_stderr(io.StringIO()) as stream:
            with self.assertRaises(ValueError) as caught:
                subject.main(self._argv())
        self.assertIs(caught.exception, original)
        self.assertIn("TypeError", stream.getvalue())
        self.assertEqual(sys.argv, previous)

    def test_guard_failure_reports_initial_state_problem_without_participant_values(self):
        from dataclasses import asdict
        from knee.aligned_training.engine import EarlyStopping
        monitor = EarlyStopping(patience=8, monitor_starts_after_epoch=10)
        state = {"identity": {"same": True}, "epoch": 1, "binding": {"private": "participant-do-not-print"},
            "initial_state_sha256": "wrong", "scheduler_epoch": 1, "phase": "head_only",
            "history": [{"epoch": 1, "phase": "head_only", "optimizer_updates": 1}],
            "total_updates": 1, "monitor": asdict(monitor), "stopped": False, "runtime": saved_runtime()}
        checkpoint = SimpleNamespace(identity={"same": True}, initial_state_digest="expected",
            request={"model_id": "image_vit_b16"})
        namespace = {"__name__": "knee.gradual_epoch_resume"}
        exec("def restore(state,self,epoch,required,binding,monitor,device):\n    raise ValueError('same_gradual_recipe_runtime_initial_weights_and_epoch_state_required')", namespace)
        try:
            namespace["restore"](state, checkpoint, 1, set(state), state["binding"], monitor, "cuda:0")
        except ValueError as error:
            with patch("knee.training_epoch_resume._runtime", return_value=saved_runtime()), \
                 patch("knee.aligned_training.engine._seeded_execution", return_value=nullcontext()), \
                 redirect_stderr(io.StringIO()) as stream:
                subject.report_restore_failure(error)
        output = stream.getvalue()
        self.assertIn('"initial_weights": false', output)
        self.assertIn('"runtime_differences": {}', output)
        self.assertNotIn("participant-do-not-print", output)

    def test_existing_frozen_engine_recovers_exactly_after_cpu_profile_restoration(self):
        import torch
        sys.path.insert(0, str(ROOT / "tests"))
        from test_gradual_epoch_resume import GradualEpochResumeTests
        from knee.aligned_training import engine
        case = GradualEpochResumeTests(methodName="test_completed_checkpoint_finalizes_without_any_new_update")
        case.setUpClass()
        try:
            case.setUp()
            full = case.run_fixture("launcher_baseline", case.checkpoint("baseline"))
            interrupted = case.interrupt("launcher_interrupted", 11)
            saved = case.latest_state(interrupted)
            torch.set_num_threads(2)
            with patch.object(engine, "_execute_training_epoch", side_effect=AssertionError("No update before compatibility")), \
                 self.assertRaisesRegex(ValueError, "same_gradual_recipe_runtime_initial"):
                case.run_fixture("wrong_profile", case.checkpoint("launcher_interrupted"))
            before = subject.configure_threads(torch, saved["runtime"])
            self.assertEqual(before["cpu_threads"], 2)
            self.assertEqual(torch.get_num_threads(), 1)
            continued = case.run_fixture("launcher_continued", case.checkpoint("launcher_interrupted"))
            self.assertEqual(continued["resumed_from_epoch"], 11)
            for key in ("history", "checkpoint_sha256", "best_epoch", "best_validation_ap", "optimizer_updates"):
                self.assertEqual(full[key], continued[key], key)
        finally:
            case.doCleanups()
            case.tearDownClass()


if __name__ == "__main__":
    unittest.main()
