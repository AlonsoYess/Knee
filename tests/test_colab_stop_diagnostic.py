"""Synthetic parent-kernel exceptions only; never create a training process."""
from contextlib import ExitStack
import importlib.util
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("stop_diagnostic",ROOT/"scripts/diagnose_colab_training_stop.py")
subject=importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


def simulated_traceback(code, with_status=True):
    def training_run():
        process=SimpleNamespace(returncode=code)
        if with_status:
            status=code
        raise RuntimeError("fabricated child termination")
    try:
        training_run()
    except RuntimeError as exc:
        return exc.__traceback__


class ColabStopDiagnosticTests(unittest.TestCase):
    def fixture(self, traceback=None, gpu=True):
        def text(path,limit=16384):
            if str(path)=="/proc/meminfo":return "MemTotal: 12000000 kB\nMemAvailable: 9000000 kB\nSwapTotal: 0 kB\n"
            if str(path).endswith("memory.events"):return "oom 1\noom_kill 1"
            return "max"
        with ExitStack() as stack:
            stack.enter_context(patch.object(subject,"_text",side_effect=text))
            stack.enter_context(patch.object(subject.shutil,"disk_usage",return_value=shutil._ntuple_diskusage(100,50,50)))
            command=stack.enter_context(patch.object(subject.subprocess,"run",return_value=SimpleNamespace(returncode=0,stdout="Tesla T4, 15360 MiB, 0 MiB"),
                side_effect=None if gpu else FileNotFoundError()))
            result=subject.collect(traceback)
            self.assertEqual(command.call_args.args[0][0],"nvidia-smi")
            self.assertIn("--query-gpu=name,memory.total,memory.used",command.call_args.args[0])
            for name in ("training_executed","training_resumed","files_modified","participant_data_opened","reserved_test_opened","termination_cause_confirmed"):
                self.assertFalse(result[name])
            return result

    def test_negative_returncode_records_signal_without_claiming_memory_cause(self):
        # This suite also runs on Windows, whose signal enum lacks SIGKILL.
        # Simulate only the Linux name lookup, not the exception-frame evidence.
        with patch.object(subject.signal,"Signals",return_value=SimpleNamespace(name="SIGKILL")):
            result=self.fixture(simulated_traceback(-9))
        self.assertEqual(result["training_subprocess_returncode"],-9)
        self.assertEqual(result["termination_signal"],"SIGKILL")
        self.assertFalse(result["termination_cause_confirmed"])

    def test_falls_back_to_process_returncode_when_status_is_absent(self):
        result=self.fixture(simulated_traceback(137,with_status=False))
        self.assertEqual(result["training_subprocess_returncode"],137)
        self.assertIsNone(result["termination_signal"])

    def test_missing_saved_exception_does_not_invent_cause(self):
        result=self.fixture()
        self.assertFalse(result["traceback_available"])
        self.assertIsNone(result["training_subprocess_returncode"])

    def test_unavailable_gpu_query_is_nonfatal(self):
        result=self.fixture(simulated_traceback(1),gpu=False)
        self.assertEqual(result["training_subprocess_returncode"],1)
        self.assertIsNone(result["gpu_memory_readout"])

    def test_system_file_reads_are_bounded_and_unavailable_files_are_nonfatal(self):
        from io import StringIO
        with patch.object(Path,"open",return_value=StringIO("x"*20)):
            self.assertIsNone(subject._text("unused",limit=10))
        with patch.object(Path,"open",side_effect=FileNotFoundError()):
            self.assertIsNone(subject._text("unused"))

    def test_saved_exception_fallback_when_sys_last_traceback_is_absent(self):
        traceback=simulated_traceback(-9)
        with patch.object(subject.sys,"last_traceback",None,create=True), \
             patch.object(subject.sys,"last_exc",SimpleNamespace(__traceback__=traceback),create=True):
            self.assertIs(subject.last_traceback(),traceback)


if __name__=="__main__":
    unittest.main()
