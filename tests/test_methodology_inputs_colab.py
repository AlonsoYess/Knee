"""Notebook38 staging, actual synthetic preparation, and private readback."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from knee import methodology_inputs_colab as subject
from knee import methodology_preparation
from knee.colab_data import sha_file
from knee.training_plan import _digest
from test_training_data import reference_fixture

ROOT = Path(__file__).resolve().parents[1]


def fixture(directory):
    base = Path(directory)
    sources = base / "sources"
    sources.mkdir()
    value = reference_fixture(sources)
    value["projection_path"] = sources / subject.PROJECTION_FILENAME
    projection = [{**{name: row[name] for name in ("src_subject_id", "side", "AGEYEARS", "SEX", "BMI")},
        "KL_BASE": 2 + index % 2, "CIRUGIA_PREVIA": None if index % 7 == 0 else index % 2,
        "WOMAC_TOTAL": None if index % 6 == 0 else float(index % 30)} for index, row in enumerate(value["rows"])]
    subject._save(value["projection_path"], projection)
    descriptor = subject.make_baseline_descriptor(value["projection_path"], package_sha256=value["package_sha256"],
        participants=15, knees=30, source_sha256="b" * 64)
    value["descriptor_path"] = sources / subject.DESCRIPTOR_FILENAME
    subject._save(value["descriptor_path"], descriptor)
    value["output"] = base / "verified"
    value["source_dir"] = sources
    value["descriptor"] = descriptor
    return value


class MethodologyColabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def verify(self, value):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "", "COLAB_GPU": "0"}), \
             patch("sklearn.linear_model.LogisticRegression.fit", side_effect=AssertionError("Estimator forbidden")), \
             patch("xgboost.XGBClassifier.fit", side_effect=AssertionError("Estimator forbidden")), \
             patch("torch.optim.AdamW", side_effect=AssertionError("Optimizer forbidden")), \
             patch("torch.Tensor.backward", side_effect=AssertionError("Backward forbidden")), \
             patch("knee.aligned_training.models.build_model", side_effect=AssertionError("Model creation forbidden")), \
             patch("knee.aligned_training.runner.execute", side_effect=AssertionError("Training forbidden")):
            return subject.verify(ROOT, value["package"], value["reference_path"], value["projection_path"],
                value["descriptor_path"], value["output"], execution_environment="local_cpu")

    def receipts(self):
        tests = {"passed": True, "tests_run": 100, "skipped": 0, "scope": list(subject.TEST_MODULES)}
        runtime = {"version": subject.runtime_base.VERSION, "cpu_only": True,
            "training_executed": False, "host_python_modified": False, "pretrained_weights_downloaded": False,
            "python_version": "3.12.14", "source_sha256": {name: sha_file(ROOT / name) for name in
                ("src/knee/training_preparation_colab.py", "requirements-training-runner.txt")}}
        return tests, runtime

    def test_full_preflight_and_persistence_are_development_only_without_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            value = fixture(directory)
            result = self.verify(value)
            self.assertTrue(subject.validate_result(result, "local_cpu"))
            self.assertFalse(result["colab_runtime_executed"])
            self.assertTrue(result["local_rehearsal"])
            self.assertEqual(result["development_knees"], 30)
            self.assertEqual(len(result["artifacts_sha256"]), 125)
            self.assertEqual(result["augmentation"]["normalization_pairs_verified"], 120)
            for name in subject.FALSE_FLAGS:
                self.assertIs(result[name], False)
            records = Path(directory) / "private_records"
            records.mkdir()
            tests, runtime = self.receipts()
            summary = subject.save_result(ROOT, value["output"], records, code_revision="a" * 40,
                cpu_tests=tests, runtime=runtime, execution_environment="local_cpu")
            self.assertTrue(subject.validate_summary(summary, "local_cpu"))
            self.assertEqual(summary["result_files_verified"], 126)
            self.assertTrue(summary["result_files_saved_and_readback_verified"])
            saved = list(records.iterdir())
            self.assertEqual(len(saved), 1)
            self.assertEqual(len(list(saved[0].rglob("*.*"))), 127)

    def test_stage_copies_only_fixed_files_and_preserves_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            value = fixture(directory)
            extra = value["source_dir"] / "reserved_test_package.zip"
            extra.write_bytes(b"FORBIDDEN UNRELATED DATA; MUST NOT BE READ")
            local = Path(directory) / "baseline_stage"
            with patch("knee.methodology_inputs_colab.sha_file", wraps=sha_file) as hash_call:
                result = subject.stage_baseline(value["source_dir"], local)
            self.assertEqual({p.name for p in local.iterdir()}, {subject.PROJECTION_FILENAME, subject.DESCRIPTOR_FILENAME})
            self.assertTrue(result["development_only"])
            self.assertNotIn(extra, [call.args[0] for call in hash_call.call_args_list])

    def test_descriptor_rejects_future_or_unknown_fields_and_changed_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            value = fixture(directory)
            for field, changed_value in (("scope", "reserved_test"), ("baseline_visit", "V06"),
                                          ("unexpected", True), ("development_knees", True)):
                descriptor = deepcopy(value["descriptor"])
                descriptor[field] = changed_value
                descriptor["record_sha256"] = _digest({k: v for k, v in descriptor.items() if k != "record_sha256"})
                with self.subTest(field=field), self.assertRaises(ValueError):
                    subject._descriptor(descriptor)

    def test_changed_projection_fails_before_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            value = fixture(directory)
            value["projection_path"].write_bytes(b"[]")
            with patch.object(methodology_preparation, "prepare", side_effect=AssertionError("Must reject before preparation")), \
                 self.assertRaisesRegex(ValueError, "pinned_baseline_projection_required"):
                self.verify(value)

    def test_cpu_guard_rejects_gpu_before_input_access(self):
        with patch.dict(os.environ, {"COLAB_GPU": "1"}), \
             patch.object(subject, "read_json", side_effect=AssertionError("Input read forbidden")), \
             self.assertRaisesRegex(RuntimeError, "cpu_only_no_accelerator_required"):
            subject.verify("missing", "missing", "missing", "missing", "missing", "missing", execution_environment="local_cpu")

    def test_colab_claim_rejected_on_non_colab_host(self):
        with patch.dict(os.environ, {"COLAB_GPU": "0"}), \
             patch.object(subject.runtime_base.previous.platform, "system", return_value="Windows"), \
             self.assertRaisesRegex(RuntimeError, "actual_linux_colab_environment_required"):
            subject._cpu_environment("google_colab_cpu")

    def test_result_rejects_incomplete_geometry_or_any_training_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.verify(fixture(directory))
            for field in subject.FALSE_FLAGS:
                changed = deepcopy(result)
                changed[field] = True
                with self.subTest(field=field), self.assertRaises(ValueError):
                    subject.validate_result(changed, "local_cpu")
            changed = deepcopy(result)
            changed["augmentation"]["normalization_pairs_verified"] = 119
            with self.assertRaises(ValueError):
                subject.validate_result(changed, "local_cpu")

    def test_save_rejects_unregistered_file_or_changed_runtime_before_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            value = fixture(directory)
            self.verify(value)
            records = Path(directory) / "private_records"
            records.mkdir()
            tests, runtime = self.receipts()
            bad_runtime = deepcopy(runtime)
            bad_runtime["source_sha256"] = {}
            with self.assertRaisesRegex(ValueError, "isolated_methodology_cpu_runtime_receipt_required"):
                subject.save_result(ROOT, value["output"], records, code_revision="a" * 40,
                    cpu_tests=tests, runtime=bad_runtime, execution_environment="local_cpu")
            (value["output"] / "unexpected_private_file.json").write_bytes(b"{}")
            with self.assertRaisesRegex(ValueError, "only_registered_methodology_outputs_may_be_saved"):
                subject.save_result(ROOT, value["output"], records, code_revision="a" * 40,
                    cpu_tests=tests, runtime=runtime, execution_environment="local_cpu")
            self.assertEqual(list(records.iterdir()), [])

    def test_summary_rejects_fake_training_or_colab_claim(self):
        with self.assertRaises(ValueError):
            subject.validate_summary({"version": subject.CORE_VERSION, "status": subject.CORE_STATUS,
                "training_executed": True}, "local_cpu")


if __name__ == "__main__":
    unittest.main()
