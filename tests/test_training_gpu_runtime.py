"""Mocked CUDA bootstrap checks; never allocate a GPU or run training locally."""
from __future__ import annotations

import ast
from contextlib import nullcontext
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

from knee import training_gpu_runtime as subject

PROJECT = Path(__file__).resolve().parents[1]


class TrainingGpuRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="knee_gpu_runtime_mock_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_cpu_numerical_encoder_and_clinical_pins_are_preserved(self):
        observed = subject.expected_packages(PROJECT)
        bootstrap = subject.helpers._bootstrap(PROJECT)
        for name, version in bootstrap.EXPECTED_PACKAGES.items():
            if name not in subject.GPU_WHEELS:
                self.assertEqual(observed[name], version)
        self.assertEqual(observed["torch"], "2.6.0+cu124")
        self.assertEqual(observed["torchvision"], "0.21.0+cu124")
        self.assertEqual(observed["transformers"], "4.50.2")
        self.assertEqual(observed["timm"], "1.0.15")
        self.assertEqual(observed["scikit-learn"], "1.6.1")
        self.assertEqual(observed["xgboost-cpu"], "3.0.5")
        self.assertEqual(observed["triton"], "3.2.0")

    def test_gpu_requirement_inventory_is_exact_and_has_no_unpinned_entries(self):
        self.assertEqual(subject._pins(PROJECT / "requirements-training-gpu.txt"), subject.CUDA_PACKAGES)
        for line in ("torch>=2.6", "--extra-index-url https://invalid", "torch==2.6\ntorch==2.6"):
            path = self.root / "pins.txt"
            path.write_text(line, encoding="utf-8")
            with self.subTest(line=line), self.assertRaises(RuntimeError):
                subject._pins(path)

    def test_official_cuda_wheels_have_fixed_exact_hashes_and_host(self):
        from urllib.parse import urlsplit
        for name, url in subject.GPU_WHEELS.items():
            parsed = urlsplit(url)
            self.assertEqual(parsed.scheme, "https")
            self.assertEqual(parsed.hostname, "download-r2.pytorch.org")
            self.assertTrue(parsed.path.startswith("/whl/cu124/"))
            self.assertIn("cp312-cp312-linux_x86_64.whl", parsed.path)
            self.assertRegex(parsed.fragment, r"^sha256=[0-9a-f]{64}$")
            self.assertIn(name + "-", parsed.path)

    def test_runtime_environment_preserves_gpu_assignment_and_removes_host_and_secrets(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0", "PYTHONHOME": "untrusted",
                "PYTHONPATH": "untrusted", "PIP_INDEX_URL": "https://secret@example.invalid",
                "HF_TOKEN": "secret", "GOOGLE_APPLICATION_CREDENTIALS": "secret"}, clear=False):
            env = subject.runtime_environment(PROJECT)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(env["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["PYTHONPATH"], os.pathsep.join(str(PROJECT / child) for child in ("src", "tests")))
        for name in ("PYTHONHOME", "PIP_INDEX_URL", "HF_TOKEN", "GOOGLE_APPLICATION_CREDENTIALS"):
            self.assertNotIn(name, env)

    def test_setup_host_requires_actual_linux_colab_gpu_and_rejects_cpu_tpu(self):
        with patch.object(subject.platform, "system", return_value="Linux"), \
                patch.object(subject.platform, "machine", return_value="x86_64"), \
                patch.dict(os.environ, {"COLAB_RELEASE_TAG": "synthetic", "COLAB_GPU": "1",
                    "COLAB_TPU_ADDR": "", "TPU_NAME": "", "COLAB_TPU_TASK_ID": ""}, clear=True):
            subject._host()
            with self.assertRaisesRegex(RuntimeError, "explicit_google_colab_gpu"):
                subject._host("local_cpu")
            with patch.dict(os.environ, {"COLAB_GPU": "0"}), self.assertRaisesRegex(RuntimeError, "select_gpu"):
                subject._host()
            with patch.dict(os.environ, {"COLAB_TPU_ADDR": "tpu"}), self.assertRaisesRegex(RuntimeError, "select_gpu"):
                subject._host()
            with patch.dict(os.environ, {"COLAB_RELEASE_TAG": ""}), self.assertRaisesRegex(RuntimeError, "actual_linux"):
                subject._host()

    def fake_runtime(self, available=True):
        tensor = MagicMock()
        product = tensor.__matmul__.return_value
        product.__eq__.return_value.all.return_value = True
        torch = Mock(__version__="2.6.0+cu124")
        torch.version.cuda = "12.4"
        torch.cuda.is_available.return_value = available
        torch.cuda.device_count.return_value = 1
        torch.cuda.get_device_properties.return_value = SimpleNamespace(major=7, minor=5,
            name="synthetic gpu", total_memory=16 * 1024**3)
        torch.inference_mode.side_effect = nullcontext
        torch.ones.return_value = tensor
        torch.isfinite.return_value.all.return_value = True
        modules = {"torch": torch, "torchvision": SimpleNamespace(__version__="0.21.0+cu124"),
            "numpy": Mock(), "sklearn": Mock(), "xgboost": Mock()}
        return torch, modules

    def verify_mock(self, *, available=True, configuration=":4096:8", packages=None):
        torch, modules = self.fake_runtime(available)
        self.enterContext(patch.object(subject, "_host"))
        self.enterContext(patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": configuration}))
        self.enterContext(patch.dict("sys.modules", modules))
        self.enterContext(patch.object(subject, "expected_packages", return_value=packages or {"torch": "2.6.0+cu124"}))
        self.enterContext(patch.object(subject.platform, "python_version", return_value="3.12.14"))
        self.enterContext(patch.object(subject.importlib.metadata, "version", return_value="2.6.0+cu124"))
        return torch

    def test_gpu_verification_only_runs_small_inference_probe_with_no_backward_or_optimizer(self):
        torch = self.verify_mock()
        result = subject.verify_current_runtime()
        self.assertTrue(result["gpu_only"] and result["eval_only_cuda_matmul_verified"])
        self.assertFalse(result["training_executed"] or result["optimizer_created"] or result["backward_executed"])
        self.assertEqual(result["device"], "cuda:0")
        torch.ones.assert_called_once_with((2, 2), dtype=torch.float32, device="cuda:0")
        torch.optim.assert_not_called()
        torch.cuda.synchronize.assert_called_once()

    def test_gpu_verification_rejects_no_gpu_and_missing_deterministic_configuration(self):
        torch = self.verify_mock(available=False)
        with self.assertRaisesRegex(RuntimeError, "no_cpu_fallback"):
            subject.verify_current_runtime()
        torch.ones.assert_not_called()

    def test_gpu_verification_rejects_environment_before_cuda_initialization(self):
        torch = self.verify_mock(configuration="")
        with self.assertRaisesRegex(RuntimeError, "before_cuda"):
            subject.verify_current_runtime()
        torch.cuda.is_available.assert_not_called()

    def test_gpu_verification_rejects_different_packages_or_cuda_build(self):
        torch = self.verify_mock(packages={"torch": "invalid"})
        with self.assertRaisesRegex(RuntimeError, "frozen_training_gpu_packages"):
            subject.verify_current_runtime()
        torch.ones.assert_not_called()

    def setup_fixture(self):
        repository = self.root / "repository"
        repository.mkdir()
        for name in subject.SOURCE_FILES:
            path = repository / name
            path.parent.mkdir(exist_ok=True, parents=True)
            path.write_text("synthetic pinned source", encoding="utf-8")
        config = {"python_archive": {"file": "python.tar.gz"}}
        bootstrap = Mock(PYTHON_SHA="a" * 64)
        bootstrap._configuration.return_value = (config, None, None)
        bootstrap.download.side_effect = lambda descriptor, target: target
        def extract(archive, destination, config):
            python = destination / "python/bin/python3.12"
            python.parent.mkdir(parents=True)
            python.write_bytes(b"synthetic isolated interpreter")
            return python
        bootstrap.extract_python.side_effect = extract
        self.enterContext(patch.object(subject, "_host"))
        self.enterContext(patch.object(subject.helpers, "_bootstrap", return_value=bootstrap))
        self.enterContext(patch.object(subject, "expected_packages", return_value={"numpy": "1.26.4", **subject.CUDA_PACKAGES}))
        self.enterContext(patch.object(subject, "runtime_environment", return_value={"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}))
        run = self.enterContext(patch.object(subject.helpers, "_run", return_value="3.12.14"))
        verify = self.enterContext(patch.object(subject, "_verify_installed", return_value={"gpu_only": True}))
        return repository, bootstrap, run, verify

    def test_setup_uses_safe_frozen_python_only_then_pinned_cuda_not_cpu_wheels(self):
        repository, bootstrap, run, verify = self.setup_fixture()
        target = self.root / "gpu_runtime"
        result = subject.setup_runtime(repository, target)
        self.assertEqual(bootstrap.download.call_count, 1)
        bootstrap.setup.assert_not_called()
        self.assertTrue(result["gpu_only"])
        self.assertFalse(result["cpu_only"] or result["host_python_modified"] or result["training_executed"])
        self.assertEqual(result["runtime_archive_sha256"], "a" * 64)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(len(commands), 4)
        self.assertTrue(all(Path(command[0]).is_relative_to(target) for command in commands))
        cuda_command = commands[-1]
        for value in subject.GPU_WHEELS.values():
            self.assertIn(value, cuda_command)
        self.assertIn("--no-deps", cuda_command)
        self.assertIn("--only-binary=:all:", cuda_command)
        self.assertEqual(subject.helpers._json(target / subject.RECEIPT_FILENAME), result)
        verify.assert_called_once()

    def test_repeated_setup_reverifies_receipt_and_imports_without_reinstalling(self):
        repository, bootstrap, run, verify = self.setup_fixture()
        target = self.root / "gpu_runtime"
        subject.setup_runtime(repository, target)
        result = subject.setup_runtime(repository, target)
        self.assertTrue(result["existing_runtime_reverified"])
        self.assertEqual(run.call_count, 4)
        self.assertEqual(verify.call_count, 2)
        self.assertEqual(bootstrap.download.call_count, 1)

    def test_setup_rejects_incomplete_directory_changed_source_and_external_interpreter(self):
        repository, _, _, _ = self.setup_fixture()
        target = self.root / "incomplete"
        target.mkdir()
        with self.assertRaisesRegex(RuntimeError, "incomplete_gpu_runtime"):
            subject.setup_runtime(repository, target)
        target = self.root / "complete"
        result = subject.setup_runtime(repository, target)
        path = target / subject.RECEIPT_FILENAME
        external = self.root / "external_python"
        external.write_bytes(b"fake")
        path.write_text(json.dumps({**result, "python_path": str(external)}), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "isolated_gpu_python"):
            subject.setup_runtime(repository, target)
        path.write_text(json.dumps(result), encoding="utf-8")
        (repository / subject.SOURCE_FILES[0]).write_text("changed source", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "receipt_mismatch"):
            subject.setup_runtime(repository, target)

    def test_setup_rejects_drive_and_repository_destinations(self):
        repository, _, _, _ = self.setup_fixture()
        drive = self.root / "drive"
        drive.mkdir()
        for target in (repository / "runtime", drive / "runtime"):
            with self.subTest(target=target), self.assertRaisesRegex(RuntimeError, "temporary_gpu_runtime"):
                subject.setup_runtime(repository, target)

    def test_verification_subprocess_runs_pip_check_then_checkout_probe(self):
        expected = {"torch": "2.6.0+cu124"}
        info = {"packages": expected, "gpu_only": True, "training_executed": False}
        with patch.object(subject, "runtime_environment", return_value={"CUDA_VISIBLE_DEVICES": "0"}), \
                patch.object(subject.helpers, "_run", side_effect=["", json.dumps(info)]) as run:
            result = subject._verify_installed(Path("isolated-python"), expected, PROJECT)
        self.assertEqual(result, info)
        self.assertIn("check", run.call_args_list[0].args[0])
        self.assertIn("verify_current_runtime", run.call_args_list[1].args[0][-1])
        self.assertNotIn("-I", run.call_args_list[1].args[0])

    def test_gpu_module_has_no_eager_tensor_import_or_training_operations(self):
        path = PROJECT / "src/knee/training_gpu_runtime.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module)
        self.assertFalse(imported & {"torch", "torchvision", "numpy", "sklearn", "xgboost"})
        source = path.read_text(encoding="utf-8")
        for operation in (".backward(", "torch.optim.", ".fit(", "load_sealed_inputs(", "training_runner.execute("):
            self.assertNotIn(operation, source)


if __name__ == "__main__":
    unittest.main()
