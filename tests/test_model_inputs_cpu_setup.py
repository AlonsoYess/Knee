"""Synthetic bootstrap safety checks: no network, interpreter or data execution."""
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cpu_setup_under_test", PROJECT / "scripts/setup_model_inputs_cpu.py")
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


class Response(io.BytesIO):
    def __init__(self, raw, url, length=None):
        super().__init__(raw)
        self.url = url
        self.headers = {} if length is None else {"Content-Length": str(length)}

    def geturl(self):
        return self.url


class CPUSetupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / "repository"
        (self.repo / "configs").mkdir(parents=True)
        for relative in ("configs/model_inputs_colab_runtime_v1.json", "requirements-colab-model-inputs.txt"):
            (self.repo / relative).write_bytes((PROJECT / relative).read_bytes())
        self.config, _, _ = subject._configuration(self.repo)

    def tar(self, entries):
        path = self.root / "python.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            for name, kind, value in entries:
                info = tarfile.TarInfo(name)
                if kind == "file":
                    info.size = len(value)
                    archive.addfile(info, io.BytesIO(value))
                else:
                    info.type = tarfile.SYMTYPE if kind == "link" else tarfile.FIFOTYPE
                    info.linkname = value
                    archive.addfile(info)
        return path

    def test_historical_runtime_and_official_cpu_wheel_pins_exact(self):
        # Freeze the previously verified public pins here; this suite must not
        # depend on publishing older operational notebooks or their resources.
        spec = self.config["python_archive"]
        self.assertEqual(spec["sha256"], "ef605200f8174e87ecfc308e52a88127543f85dd5c940dc5e92cab244b98a003")
        self.assertEqual(spec["bytes"], 34277962)
        self.assertTrue(spec["url"].startswith("https://github.com/astral-sh/python-build-standalone/releases/download/20260929/"))
        self.assertEqual([wheel["sha256"] for wheel in self.config["cpu_wheels"]], [
            "59e78aa0c690f70734e42670036d6b541930b8eabbaa18d94e090abf14cc4d91",
            "d6874431e678ba107b60a83f255c33f3755f06bad587b1b919aa514ec325dcd8"])
        for wheel in self.config["cpu_wheels"]:
            self.assertIn("cp312-cp312-linux_x86_64.whl", wheel["url"])
        self.assertFalse({"scipy", "pydicom", "openpyxl", "opencv-python-headless", "solt"} & set(self.config["packages"]))

    def test_unregistered_requirement_or_runtime_configuration_rejected(self):
        req = self.repo / "requirements-colab-model-inputs.txt"
        req.write_text(req.read_text() + "openpyxl==3.1.5\n")
        with self.assertRaisesRegex(RuntimeError, "minimal_pypi_requirements"):
            subject._configuration(self.repo)

    def test_environment_sanitized_without_changing_parent(self):
        source = {"HOME": "private", "PYTHONHOME": "wrong", "PYTHONPATH": "wrong", "PIP_INDEX_URL": "secret",
                  "PIP_EXTRA_INDEX_URL": "secret", "HTTP_PROXY": "secret", "VIRTUAL_ENV": "wrong", "PATH": "expected"}
        with patch.dict(subject.os.environ, source, clear=True):
            result = subject.isolated_environment()
            self.assertEqual(dict(subject.os.environ), source)
        self.assertFalse(set(source).intersection(result) - {"PATH"})
        self.assertEqual(result["PIP_CONFIG_FILE"], subject.os.devnull)
        self.assertEqual(result["HF_HUB_OFFLINE"], "1")

    def test_download_stream_verifies_hash_and_refuses_overwrite(self):
        raw = b"synthetic interpreter bytes"
        expected = hashlib.sha256(raw).hexdigest()
        spec = {"url": subject.PYTHON_URL, "sha256": expected, "max_bytes": len(raw), "bytes": len(raw)}
        response = Response(raw, subject.PYTHON_URL, len(raw))
        with patch.object(subject, "PYTHON_SHA", expected), patch.object(subject.urllib.request, "build_opener", return_value=Mock(open=Mock(return_value=response))):
            target = subject.download(spec, self.root / "download")
            self.assertEqual(target.read_bytes(), raw)
            with self.assertRaisesRegex(RuntimeError, "already_exists"):
                subject.download(spec, target)

    def test_download_wrong_hash_oversize_and_unregistered_url_rejected(self):
        raw = b"synthetic runtime"
        for label, maximum, expected in (("size", 2, hashlib.sha256(raw).hexdigest()), ("hash", 100, "0" * 64)):
            spec = {"url": subject.PYTHON_URL, "sha256": expected, "max_bytes": maximum}
            with self.subTest(label=label), patch.object(subject, "PYTHON_SHA", expected), \
                 patch.object(subject.urllib.request, "build_opener", return_value=Mock(open=Mock(return_value=Response(raw, subject.PYTHON_URL)))):
                with self.assertRaises(RuntimeError):
                    subject.download(spec, self.root / label)
        with self.assertRaisesRegex(RuntimeError, "unregistered"):
            subject.download({"url": "https://untrusted.example/python", "sha256": "0" * 64}, self.root / "bad")

    def test_only_https_allowlisted_redirects(self):
        for url in ("http://github.com/archive", "https://untrusted.example/archive", "https://user:secret@github.com/archive", "https://github.com:444/archive"):
            with self.subTest(url=url), self.assertRaises(RuntimeError):
                subject.HTTPSRedirects().redirect_request(None, None, 302, "redirect", {}, url)

    def test_safe_python_tar_extracts_regular_interpreter_and_never_overwrites(self):
        path = self.tar([("python/bin/python3.12", "file", b"synthetic")])
        target = self.root / "runtime"
        executable = subject.extract_python(path, target, self.config)
        self.assertEqual(executable.read_bytes(), b"synthetic")
        with self.assertRaisesRegex(RuntimeError, "fresh_python"):
            subject.extract_python(path, target, self.config)

    def test_tar_rejects_escape_duplicates_devices_and_external_links(self):
        cases = [[("../outside", "file", b"x")], [("/absolute", "file", b"x")],
                 [("other/bin/python3.12", "file", b"x")], [("python/x", "fifo", "")],
                 [("python/x", "file", b"x"), ("python/x", "file", b"x")],
                 [("python/bin/python3.12", "link", "../../../outside")],
                 [("python/bin/python3.12", "link", "/absolute")]]
        for i, entries in enumerate(cases):
            with self.subTest(entries=entries), self.assertRaises(RuntimeError):
                subject.extract_python(self.tar(entries), self.root / ("bad" + str(i)), self.config)

    def test_tar_accepts_legitimate_internal_link_without_platform_symlink_creation(self):
        path = self.tar([("python/bin/python3.12", "file", b"x"), ("python/bin/python3", "link", "python3.12")])
        def extract(archive, destination, **kwargs):
            self.assertEqual(kwargs["filter"], "data")
            exe = Path(destination) / "python/bin/python3.12"
            exe.parent.mkdir(parents=True)
            exe.write_bytes(b"x")
        with patch.object(tarfile.TarFile, "extractall", extract):
            self.assertTrue(subject.extract_python(path, self.root / "links", self.config).is_file())

    def test_tar_member_count_and_expansion_bounded(self):
        path = self.tar([("python/bin/python3.12", "file", b"synthetic")])
        for key, value in (("maximum_tar_members", 0), ("maximum_tar_expanded_bytes", 1)):
            config = dict(self.config, **{key: value})
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "bounded_python_tar"):
                subject.extract_python(path, self.root / key, config)

    def test_full_setup_mocked_downloads_and_subprocess_only_use_isolated_python(self):
        target = self.root / "cpu"
        commands = []
        def fake_download(spec, path):
            path.write_bytes(b"fake pinned artifact")
            return path
        def fake_extract(path, destination, config):
            exe = destination / "python/bin/python3.12"
            exe.parent.mkdir(parents=True)
            exe.write_bytes(b"fake interpreter")
            return exe
        def fake_run(command, env, cwd, label):
            commands.append((command, env, label))
            if label == "isolated_python_version":
                return "3.12.14"
            if label == "cpu_runtime_import_versions":
                return json.dumps({"python": "3.12.14", "packages": subject.EXPECTED_PACKAGES,
                                   "cuda_available": False, "cuda_build": None})
            return "ok"
        with patch.object(subject.platform, "system", return_value="Linux"), patch.object(subject.platform, "machine", return_value="x86_64"), \
             patch.object(subject, "download", side_effect=fake_download), patch.object(subject, "extract_python", side_effect=fake_extract), \
             patch.object(subject, "_run", side_effect=fake_run), patch.dict(subject.os.environ, {}, clear=True):
            result = subject.setup(target, self.repo)
        self.assertTrue(result["cpu_only"])
        self.assertFalse(result["host_python_modified"])
        self.assertTrue(all(Path(cmd[0]).is_relative_to(target) and cmd[1] == "-I" for cmd, _, _ in commands))
        pip_commands = [cmd for cmd, _, _ in commands if "install" in cmd]
        self.assertEqual(len(pip_commands), 2)
        self.assertTrue(all("--no-deps" in cmd for cmd in pip_commands))
        self.assertFalse(result["data_opened"] or result["pretrained_weights_downloaded"] or result["training_executed"])

    def test_host_platform_or_existing_destination_fails_before_download(self):
        with patch.object(subject, "download", side_effect=AssertionError("must not download")):
            with patch.object(subject.platform, "system", return_value="Windows"), self.assertRaisesRegex(RuntimeError, "linux_x86_64"):
                subject.setup(self.root / "cpu", self.repo)
            with patch.object(subject.platform, "system", return_value="Linux"), patch.object(subject.platform, "machine", return_value="x86_64"), \
                 patch.dict(subject.os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, "fresh_bootstrap"):
                subject.setup(self.repo, self.repo)

    def test_cli_prints_one_json_and_no_logs_to_stdout(self):
        output = io.StringIO()
        with patch.object(subject.sys, "argv", ["bootstrap", "--destination", str(self.root / "cpu"), "--repository-root", str(self.repo)]), \
             patch.object(subject, "setup", return_value={"python_path": "synthetic-python", "cpu_only": True}), redirect_stdout(output):
            subject.main()
        self.assertEqual(json.loads(output.getvalue()), {"python_path": "synthetic-python", "cpu_only": True})


if __name__ == "__main__":
    unittest.main()
