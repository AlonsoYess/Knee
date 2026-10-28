"""Exercise notebook supervision with tiny CPU children, never model training."""
from __future__ import annotations

from contextlib import redirect_stdout
import errno
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROOT = Path(os.environ.get("KNEE_FROZEN_REPOSITORY", ROOT))
MODULE_PATH = ROOT / "scripts/colab_training_supervisor.py"
SPEC = importlib.util.spec_from_file_location("verified_colab_training_supervisor", MODULE_PATH)
supervisor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = supervisor
SPEC.loader.exec_module(supervisor)

REVISION = "8f9832fac7e53862f0c16020c82898e5cb6a74e6"
HEALTHY = {
    "mem_available_bytes": 32 * 1024**3,
    "disk_free_bytes": 64 * 1024**3,
    "cgroup_oom_kill": 0,
    "cgroup_scope": "/synthetic_test_cgroup",
}


def healthy(*args, **kwargs):
    return dict(HEALTHY)


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="supervisor_cpu_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.session = self.root / "session"
        self.records = self.root / "records"
        self.session.mkdir()
        self.records.mkdir()
        self.stdout = io.StringIO()

    def arguments(self, source):
        return [sys.executable, "-u", "-c", source,
                "--operation", "train", "--campaign", "gradual_four_models",
                "--output-dir", str(self.session / "original_output")]

    def invoke(self, source, **overrides):
        settings = dict(cwd=self.root, env=dict(os.environ), records_root=self.records,
                        session_root=self.session, code_revision=REVISION,
                        sample_interval=0.02, heartbeat_interval=60,
                        max_retries=1, retry_delay=0, sample_provider=healthy)
        settings.update(overrides)
        with redirect_stdout(self.stdout):
            return supervisor.run(self.arguments(source), **settings)

    def status_files(self):
        return sorted((self.records / "gradual_runtime_logs").glob("run_*/attempt_*/status.json"))

    def read(self, path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def test_redaction_removes_urls_and_credentials(self):
        raw = ("Authorization: Bearer privateBearerValue\n"
               "password=privatePassword token:privateToken secret=privateSecret\n"
               "https://example.test/download?token=privateUrlValue")
        value = supervisor.sanitize(raw)
        for private in ("privateBearerValue", "privatePassword", "privateToken",
                        "privateSecret", "privateUrlValue", "https://"):
            self.assertNotIn(private, value)

    def test_sanitizer_preserves_normal_progress(self):
        value = "Epoca 15/60 guardada: image_skelex, pliegue 0, fase full."
        self.assertEqual(supervisor.sanitize(value), value)

    def test_success_preserves_stdout_and_persists_terminal_evidence(self):
        source = "import sys; print('synthetic-result'); print('epoch saved', file=sys.stderr)"
        outcome = self.invoke(source)
        self.assertIn("synthetic-result", outcome.stdout)
        self.assertEqual(outcome.attempts, 1)
        self.assertTrue(Path(outcome.run_dir).is_dir())
        status = self.read(self.status_files()[0])
        self.assertEqual(status["state"], "succeeded")
        self.assertEqual(status["returncode"], 0)
        self.assertEqual(self.read(Path(outcome.run_dir) / "run.json")["state"], "succeeded")
        attempt = self.status_files()[0].parent
        self.assertIn("epoch saved", (attempt / "stderr.log").read_text(encoding="utf-8"))
        self.assertTrue((attempt / "telemetry.jsonl").is_file())

    def test_failure_persists_code_and_tail_without_traceback_retention(self):
        source = "import sys; print('synthetic-error password=doNotPersist', file=sys.stderr); sys.exit(7)"
        with self.assertRaises(supervisor.TrainingStopped) as caught:
            self.invoke(source)
        self.assertIsInstance(caught.exception.report, dict)
        statuses = self.status_files()
        self.assertEqual(len(statuses), 1)
        status = self.read(statuses[0])
        self.assertEqual(status["state"], "failed")
        self.assertEqual(status["returncode"], 7)
        self.assertNotIn("doNotPersist", statuses[0].read_text(encoding="utf-8"))
        self.assertNotIn("doNotPersist", (statuses[0].parent / "stderr.log").read_text(encoding="utf-8"))

    def test_unknown_kill_is_not_a_transient_category(self):
        kind = supervisor.failure_kind(-9, "", [healthy(), healthy()])
        self.assertIn("unknown", kind.lower())
        self.assertNotIn("transient", kind.lower())

    def test_cumulative_oom_counter_alone_does_not_prove_a_new_kill(self):
        samples = [{**HEALTHY, "cgroup_oom_kill": 3}] * 2
        kind = supervisor.failure_kind(-9, "", samples)
        self.assertIn("unknown", kind.lower())

    def test_increased_oom_counter_is_distinguished_from_unknown_kill(self):
        samples = [{**HEALTHY, "cgroup_oom_kill": 3},
                   {**HEALTHY, "cgroup_oom_kill": 4}]
        kind = supervisor.failure_kind(-9, "", samples)
        self.assertIn("oom", kind.lower())
        self.assertNotIn("transient", kind.lower())

    def test_counters_from_different_cgroups_do_not_prove_an_oom_delta(self):
        samples = [{**HEALTHY, "cgroup_oom_kill": 0},
                   {**HEALTHY, "cgroup_scope": "/another_scope", "cgroup_oom_kill": 9}]
        kind = supervisor.failure_kind(-9, "", samples)
        self.assertIn("unknown", kind.lower())

    def test_handled_http_error_does_not_hide_final_unknown_exception(self):
        trace = ("Traceback (most recent call last):\n"
                 "urllib.error.HTTPError: HTTP Error 503: Service Unavailable\n\n"
                 "During handling of the above exception, another exception occurred:\n\n"
                 "Traceback (most recent call last):\n"
                 "RuntimeError: unexpected terminal failure\n")
        kind = supervisor.failure_kind(1, trace, [healthy(), healthy()])
        self.assertNotIn("transient", kind.lower())

    def test_final_http_exception_can_be_classified_as_transient(self):
        trace = ("Traceback (most recent call last):\n"
                 "RuntimeError: previous handled synthetic failure\n\n"
                 "During handling of the above exception, another exception occurred:\n\n"
                 "Traceback (most recent call last):\n"
                 "urllib.error.HTTPError: HTTP Error 503: Service Unavailable\n")
        kind = supervisor.failure_kind(1, trace, [healthy(), healthy()])
        self.assertEqual(kind, "transient_http_server_error")

    def test_nonfinite_gradients_are_not_retried(self):
        with self.assertRaises(supervisor.TrainingStopped):
            self.invoke("import sys; print('RuntimeError: gradient norm is non-finite', file=sys.stderr); sys.exit(1)")
        self.assertEqual(len(self.status_files()), 1)

    def test_integrity_error_is_not_retried(self):
        with self.assertRaises(supervisor.TrainingStopped):
            self.invoke("import sys; print('ValueError: source_sha256_mismatch', file=sys.stderr); sys.exit(1)")
        self.assertEqual(len(self.status_files()), 1)

    def test_explicit_cuda_out_of_memory_is_not_retried(self):
        with self.assertRaises(supervisor.TrainingStopped):
            self.invoke("import sys; print('torch.OutOfMemoryError: CUDA out of memory.', file=sys.stderr); sys.exit(1)")
        self.assertEqual(len(self.status_files()), 1)

    def test_transient_http_failure_retries_once_with_new_output(self):
        marker = self.root / "seen.json"
        source = (
            "import json, pathlib, sys\n"
            f"marker = pathlib.Path({str(marker)!r})\n"
            "output = sys.argv[sys.argv.index('--output-dir') + 1]\n"
            "seen = json.loads(marker.read_text()) if marker.exists() else []\n"
            "seen.append(output); marker.write_text(json.dumps(seen))\n"
            "if len(seen) == 1:\n"
            " print('urllib.error.HTTPError: HTTP Error 503: Service Unavailable', file=sys.stderr)\n"
            " sys.exit(1)\n"
            "print('recovered-result')\n"
        )
        outcome = self.invoke(source)
        seen = self.read(marker)
        self.assertEqual(len(seen), 2)
        self.assertNotEqual(seen[0], seen[1])
        self.assertEqual(Path(seen[-1]), Path(outcome.output_dir))
        self.assertEqual(outcome.attempts, 2)
        self.assertIn("recovered-result", outcome.stdout)
        self.assertEqual(len(self.status_files()), 2)

    def test_persistent_transient_failure_obeys_retry_limit(self):
        with self.assertRaises(supervisor.TrainingStopped):
            self.invoke("import sys; print('urllib.error.HTTPError: HTTP Error 503: Service Unavailable', file=sys.stderr); sys.exit(1)")
        self.assertEqual(len(self.status_files()), 2)

    def test_unknown_terminal_exception_after_http_error_is_not_retried(self):
        source = (
            "import sys\n"
            "print('urllib.error.HTTPError: HTTP Error 503: Service Unavailable', file=sys.stderr)\n"
            "print('During handling of the above exception, another exception occurred:', file=sys.stderr)\n"
            "print('RuntimeError: unknown terminal error', file=sys.stderr)\n"
            "sys.exit(1)\n"
        )
        with self.assertRaises(supervisor.TrainingStopped):
            self.invoke(source)
        self.assertEqual(len(self.status_files()), 1)

    def test_retry_disabled_never_launches_second_attempt(self):
        with self.assertRaises(supervisor.TrainingStopped):
            self.invoke("import sys; print('urllib.error.HTTPError: HTTP Error 503: Service Unavailable', file=sys.stderr); sys.exit(1)", max_retries=0)
        self.assertEqual(len(self.status_files()), 1)

    def test_non_training_operation_cannot_launch(self):
        arguments = self.arguments("raise AssertionError('must not launch')")
        arguments[arguments.index("--operation") + 1] = "evaluate_reserved"
        with patch.object(supervisor.subprocess, "Popen", side_effect=AssertionError("must not launch")):
            with self.assertRaises((ValueError, RuntimeError)):
                supervisor.run(arguments, cwd=self.root, env=dict(os.environ),
                    records_root=self.records, session_root=self.session, code_revision=REVISION,
                    sample_provider=healthy)

    def test_wrong_campaign_cannot_launch(self):
        arguments = self.arguments("raise AssertionError('must not launch')")
        arguments[arguments.index("--campaign") + 1] = "unapproved_campaign"
        with patch.object(supervisor.subprocess, "Popen", side_effect=AssertionError("must not launch")):
            with self.assertRaises((ValueError, RuntimeError)):
                supervisor.run(arguments, cwd=self.root, env=dict(os.environ),
                    records_root=self.records, session_root=self.session, code_revision=REVISION,
                    sample_provider=healthy)

    def test_unavailable_persistent_log_destination_cannot_silently_start(self):
        invalid_records = self.root / "not_a_directory"
        invalid_records.write_text("fixture", encoding="utf-8")
        with patch.object(supervisor.subprocess, "Popen", side_effect=AssertionError("must not launch")):
            with self.assertRaises((OSError, ValueError, RuntimeError)):
                self.invoke("raise AssertionError('must not launch')", records_root=invalid_records)

    def test_drivefs_unsupported_fsync_preserves_successful_logging(self):
        for code in (errno.EINVAL, errno.ENOTSUP):
            with self.subTest(errno=code):
                with patch.object(supervisor.os, "fsync", side_effect=OSError(code, "unsupported synthetic fsync")):
                    outcome = self.invoke("print('synthetic')")
                self.assertEqual(self.read(Path(outcome.run_dir) / "run.json")["state"], "succeeded")

    def test_disk_full_fsync_is_not_silently_ignored(self):
        target = self.root / "disk_full.json"
        with patch.object(supervisor.os, "fsync", side_effect=OSError(errno.ENOSPC, "synthetic disk full")):
            with self.assertRaises(OSError) as caught:
                supervisor._save(target, {"synthetic": True})
        self.assertEqual(caught.exception.errno, errno.ENOSPC)
        self.assertFalse(target.exists())

    def test_stdout_limit_fails_explicitly_instead_of_retaining_unbounded_data(self):
        with patch.object(supervisor, "MAX_STREAM_BYTES", 64):
            with self.assertRaises(supervisor.TrainingStopped) as caught:
                self.invoke("print('x' * 256)")
        self.assertIn("limit", caught.exception.report.get("supervisor_error", ""))
        self.assertEqual(len(self.status_files()), 1)

    def test_single_overlong_stderr_line_fails_explicitly(self):
        with self.assertRaises(supervisor.TrainingStopped) as caught:
            self.invoke("import sys; sys.stderr.write('x' * (256 * 1024) + '\\n'); sys.stderr.flush()")
        self.assertIn("line", caught.exception.report.get("supervisor_error", ""))
        self.assertEqual(len(self.status_files()), 1)

    def test_low_available_ram_stops_with_explicit_resource_record(self):
        def pressure(*args, **kwargs):
            return {**HEALTHY, "mem_available_bytes": 8 * 1024**2}
        started = time.monotonic()
        with self.assertRaises(supervisor.TrainingStopped):
            self.invoke("import time; time.sleep(5)", sample_provider=pressure)
        self.assertLess(time.monotonic() - started, 10)
        statuses = self.status_files()
        self.assertTrue(statuses)
        self.assertEqual(self.read(statuses[-1])["state"], "resource_guard")
        self.assertEqual(len(statuses), 1)

    def test_sampler_can_report_unknown_resources_without_crashing(self):
        outcome = self.invoke("print('synthetic')", sample_provider=lambda *a, **kw: {})
        self.assertEqual(outcome.attempts, 1)

    def test_keyboard_interrupt_is_recorded_and_not_retried(self):
        calls = []
        def cancel(*args, **kwargs):
            calls.append((args, kwargs))
            child = args[0] if args else kwargs.get("child_pid")
            if child is not None:
                raise KeyboardInterrupt()
            return healthy()
        with self.assertRaises(KeyboardInterrupt):
            self.invoke("import time; time.sleep(5)", sample_provider=cancel)
        self.assertTrue(calls)
        statuses = self.status_files()
        self.assertEqual(len(statuses), 1)
        self.assertEqual(self.read(statuses[0])["state"], "cancelled")

    def test_sample_resources_requires_no_gpu_tool_or_linux_proc(self):
        with patch.object(supervisor.subprocess, "run", side_effect=FileNotFoundError):
            value = supervisor.sample_resources(child_pid=99999999)
        self.assertIsInstance(value, dict)
        self.assertIsNone(value.get("child_rss_bytes"))

    def test_run_does_not_modify_the_frozen_training_sources(self):
        names = ("src/knee/gradual_campaign.py", "src/knee/gradual_epoch_resume.py",
                 "src/knee/aligned_training/engine.py", "configs/training_plan_v1_2.json")
        before = {name: hashlib.sha256((FROZEN_ROOT / name).read_bytes()).hexdigest() for name in names}
        self.invoke("print('synthetic')")
        after = {name: hashlib.sha256((FROZEN_ROOT / name).read_bytes()).hexdigest() for name in names}
        self.assertEqual(before, after)

    def test_previous_status_with_no_logs_is_empty(self):
        result = supervisor.load_previous_status(self.records)
        self.assertFalse(result)

    def test_unfinished_previous_receipt_reports_unknown_cause(self):
        outcome = self.invoke("print('synthetic')")
        path = Path(outcome.run_dir) / "run.json"
        value = self.read(path)
        value.update(state="running", termination_cause_confirmed=False)
        value.pop("returncode", None)
        value.pop("failure_kind", None)
        path.write_text(json.dumps(value), encoding="utf-8")
        previous = supervisor.load_previous_status(self.records)
        self.assertEqual(previous["state"], "interrupted_unknown")
        self.assertFalse(previous["termination_cause_confirmed"])
        self.assertIsNone(previous.get("returncode"))

    def test_active_controller_prevents_a_second_child(self):
        logs = self.records / "gradual_runtime_logs"
        logs.mkdir()
        (logs / "active.json").write_text(json.dumps({
            "state": "running", "controller": {"pid": 123},
            "child": {"pid": 456}, "heartbeat_unix": time.time()
        }), encoding="utf-8")
        with patch.object(supervisor, "_alive", return_value=True), \
             patch.object(supervisor.subprocess, "Popen", side_effect=AssertionError("must not launch")):
            with self.assertRaises(RuntimeError):
                self.invoke("raise AssertionError('must not launch')")

    def test_retry_cooldown_keeps_active_lease_and_blocks_duplicate(self):
        observed = []
        def inspect_cooldown(delay):
            logs = self.records / "gradual_runtime_logs"
            active = self.read(logs / "active.json")
            current = self.read(logs / active["run_name"] / "run.json")
            observed.append((active["state"], current["state"]))
            self.assertEqual(active["state"], "retrying")
            self.assertEqual(current["state"], "retrying")
            with self.assertRaises(RuntimeError):
                self.invoke("raise AssertionError('duplicate must not launch')")
        with patch.object(supervisor.time, "sleep", side_effect=inspect_cooldown):
            with self.assertRaises(supervisor.TrainingStopped):
                self.invoke("import sys; print('urllib.error.HTTPError: HTTP Error 503: Service Unavailable', file=sys.stderr); sys.exit(1)")
        self.assertEqual(observed, [("retrying", "retrying")])
        self.assertEqual(len(self.status_files()), 2)

    def test_lost_active_claim_does_not_replace_latest_pointer(self):
        outcome = self.invoke("print('previous result')")
        logs = self.records / "gradual_runtime_logs"
        latest = self.read(logs / "latest.json")
        active_path = logs / "active.json"
        active_path.unlink()
        competitor = {"run_name": "run_" + "f" * 32, "state": "running", "heartbeat_unix": time.time()}
        real_open = Path.open
        def competing_open(path, *args, **kwargs):
            mode = args[0] if args else kwargs.get("mode", "r")
            if Path(path) == active_path and mode == "x":
                with real_open(path, "w", encoding="utf-8") as stream:
                    json.dump(competitor, stream)
                raise FileExistsError("synthetic competing active claim")
            return real_open(path, *args, **kwargs)
        with patch.object(Path, "open", competing_open), \
             patch.object(supervisor.subprocess, "Popen", side_effect=AssertionError("must not launch")):
            with self.assertRaises(supervisor.TrainingStopped):
                self.invoke("raise AssertionError('must not launch')")
        self.assertEqual(self.read(logs / "latest.json"), latest)
        self.assertEqual(self.read(active_path), competitor)
        self.assertEqual(latest["run_name"], Path(outcome.run_dir).name)

    def test_midrun_persistence_failure_stops_and_reaps_owned_child(self):
        real_save = supervisor._save
        def fail_heartbeat(path, payload):
            if (Path(path).name == "status.json" and payload.get("child")
                    and "last_sample" in payload):
                raise OSError("synthetic persistent storage unavailable")
            return real_save(path, payload)
        started = time.monotonic()
        with patch.object(supervisor, "_save", side_effect=fail_heartbeat):
            with self.assertRaises(supervisor.TrainingStopped) as caught:
                self.invoke("import time; time.sleep(5)")
        self.assertLess(time.monotonic() - started, 4)
        self.assertEqual(caught.exception.report["failure_kind"], "supervisor_or_persistence_error")
        self.assertIsNotNone(caught.exception.report["returncode"])
        reports = list((self.records / "gradual_runtime_logs").glob("run_*/run.json"))
        self.assertEqual(len(reports), 1)
        self.assertEqual(self.read(reports[0])["state"], "failed")

    def test_child_exit_with_inherited_pipe_has_bounded_drain(self):
        # Grandchild is a short-lived synthetic sleeper, never a training process.
        source = (
            "import subprocess, sys\n"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(5)'], "
            "stdout=sys.stdout, stderr=sys.stderr, cwd=sys.prefix)\n"
            "print('parent done')\n"
        )
        started = time.monotonic()
        outcome = self.invoke(source)
        self.assertLess(time.monotonic() - started, 4.5)
        self.assertIn("parent done", outcome.stdout)
        self.assertEqual(outcome.attempts, 1)

    def test_no_memory_policy_preserves_original_environment(self):
        original = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "2"}
        result, record = supervisor.host_memory_environment(original)
        self.assertEqual(result, original)
        self.assertIsNot(result, original)
        self.assertFalse(record["enabled"])

    def test_host_memory_policy_only_adds_registered_cpu_allocator_controls(self):
        original = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "2", "CUDA_VISIBLE_DEVICES": "0"}
        with patch.object(supervisor.sys, "platform", "linux"), \
             patch.object(supervisor.platform, "libc_ver", return_value=("glibc", "2.35")):
            result, record = supervisor.host_memory_environment(original, supervisor.HOST_MEMORY_POLICY)
        self.assertEqual({k: result[k] for k in original}, original)
        self.assertEqual({k: result[k] for k in result if k not in original}, supervisor.HOST_MEMORY_ENV)
        self.assertTrue(record["enabled"])
        self.assertFalse(record["frozen_core_modified"])
        self.assertFalse(record["checkpoint_format_changed"])

    def test_host_memory_policy_rejects_unknown_backends_or_policy(self):
        with self.assertRaises(ValueError):
            supervisor.host_memory_environment({}, "unregistered")
        with patch.object(supervisor.sys, "platform", "win32"):
            with self.assertRaisesRegex(ValueError, "linux_glibc"):
                supervisor.host_memory_environment({}, supervisor.HOST_MEMORY_POLICY)
        with patch.object(supervisor.sys, "platform", "linux"), \
             patch.object(supervisor.platform, "libc_ver", return_value=("glibc", "2.35")):
            for environment in ({"LD_PRELOAD": "/lib/libtcmalloc.so"},
                                {"LD_PRELOAD": "/lib/libjemalloc.so"},
                                {"GLIBC_TUNABLES": "glibc.malloc.arena_max=8"}):
                with self.subTest(environment=environment), self.assertRaises(ValueError):
                    supervisor.host_memory_environment(environment, supervisor.HOST_MEMORY_POLICY)

    def test_memory_math_probe_hides_gpu_and_does_not_mutate_training_environment(self):
        proof = {"synthetic_cpu_only": True, "participant_data_opened": False,
                 "weights_sha256": "synthetic", "rng_sha256": "synthetic", "losses": [0.1]}
        original = {"CUDA_VISIBLE_DEVICES": "0", "COLAB_GPU": "1"}
        proposed = {**original, **supervisor.HOST_MEMORY_ENV}
        response = type("Response", (), {"returncode": 0, "stdout": json.dumps(proof), "stderr": ""})()
        with patch.object(supervisor.subprocess, "run", return_value=response) as process:
            result = supervisor.verify_host_memory_math(sys.executable, original, proposed)
        self.assertEqual(process.call_count, 2)
        for call in process.call_args_list:
            self.assertEqual(call.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
            self.assertEqual(call.kwargs["env"]["COLAB_GPU"], "0")
        self.assertEqual(original, {"CUDA_VISIBLE_DEVICES": "0", "COLAB_GPU": "1"})
        self.assertTrue(result["synthetic_cpu_updates_and_rng_identical"])
        self.assertFalse(result["capacity_sufficiency_proven"])

    def test_memory_math_probe_rejects_different_numerical_updates(self):
        base = {"synthetic_cpu_only": True, "participant_data_opened": False, "weights_sha256": "first"}
        responses = [type("R", (), {"returncode": 0, "stdout": json.dumps(base), "stderr": ""})(),
                     type("R", (), {"returncode": 0, "stdout": json.dumps({**base, "weights_sha256": "other"}), "stderr": ""})()]
        with patch.object(supervisor.subprocess, "run", side_effect=responses):
            with self.assertRaisesRegex(ValueError, "changed_synthetic"):
                supervisor.verify_host_memory_math(sys.executable, {}, {})

    def test_training_child_receives_cpu_memory_policy_after_math_verification(self):
        source = "import json,os;print(json.dumps({k:os.environ.get(k) for k in ('MALLOC_ARENA_MAX','MALLOC_MMAP_THRESHOLD_','MALLOC_TRIM_THRESHOLD_')}))"
        proof = {"synthetic_cpu_updates_and_rng_identical": True, "participant_data_opened": False}
        with patch.object(supervisor.sys, "platform", "linux"), \
             patch.object(supervisor.platform, "libc_ver", return_value=("glibc", "2.35")), \
             patch.object(supervisor, "verify_host_memory_math", return_value=proof) as verifier:
            outcome = self.invoke(source, host_memory_policy=supervisor.HOST_MEMORY_POLICY,
                                  env={k:v for k,v in os.environ.items() if k not in ("LD_PRELOAD","GLIBC_TUNABLES")})
        verifier.assert_called_once()
        self.assertEqual(json.loads(outcome.stdout), supervisor.HOST_MEMORY_ENV)
        record = self.read(Path(outcome.run_dir) / "run.json")
        self.assertEqual(record["host_memory_policy"]["verification"], proof)
        self.assertEqual(record["code_revision"], REVISION)

    def test_failed_memory_policy_verification_never_starts_training(self):
        with patch.object(supervisor.sys, "platform", "linux"), \
             patch.object(supervisor.platform, "libc_ver", return_value=("glibc", "2.35")), \
             patch.object(supervisor, "verify_host_memory_math", side_effect=ValueError("synthetic_math_mismatch")), \
             patch.object(supervisor.subprocess, "Popen", side_effect=AssertionError("must not launch")):
            with self.assertRaises(supervisor.TrainingStopped):
                self.invoke("raise AssertionError('must not launch')", host_memory_policy=supervisor.HOST_MEMORY_POLICY,
                            env={k:v for k,v in os.environ.items() if k not in ("LD_PRELOAD","GLIBC_TUNABLES")})


if __name__ == "__main__":
    unittest.main()
