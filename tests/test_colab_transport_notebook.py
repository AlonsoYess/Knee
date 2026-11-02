"""Fragment bootstrap remains CPU-only and preserves the sealed verification."""
import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "notebooks/31_verificar_paquete_desarrollo_sin_entrenar.ipynb"
UPDATED = ROOT / "notebooks/31_verificar_paquete_desarrollo_sin_entrenar_v1_1.ipynb"


def cells(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in value["cells"] if c["cell_type"] == "code"]


class TransportNotebookTests(unittest.TestCase):
    def test_exact_base_cells_preserved_with_only_input_and_mount_change(self):
        self.assertEqual(hashlib.sha256(BASE.read_bytes()).hexdigest(), "231ca2151400028c19cf57c46419c7a3ef57ff9f0fb2c399ac2e04c3c13e8018")
        original, changed = cells(BASE), cells(UPDATED)
        self.assertEqual(len(changed), 5)
        transformed = "\n".join("PACKAGE_DIR = LOCAL_PACKAGE_DIR" if line.startswith("PACKAGE_DIR = Path(input(") else line
            for line in original[0].split("\n") if line != "drive.mount('/content/drive')")
        self.assertEqual(changed[1], transformed)
        self.assertEqual(changed[2:4], original[1:])
        for source in changed:
            compile(source, str(UPDATED), "exec")

    def test_bootstrap_does_not_import_knee_and_enforces_cpu_before_mount(self):
        code = cells(UPDATED)[0]
        imports = [n for n in ast.walk(ast.parse(code)) if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any(isinstance(n, ast.ImportFrom) and (n.module or "").startswith("knee") for n in imports))
        google, colab = types.ModuleType("google"), types.ModuleType("google.colab")
        colab.drive = types.SimpleNamespace(mount=lambda _: self.fail("GPU must fail before mount"))
        google.colab = colab
        for variable in ("COLAB_GPU", "COLAB_TPU_ADDR", "TPU_NAME", "COLAB_TPU_TASK_ID"):
            with self.subTest(variable=variable), patch.dict(sys.modules, {"google": google, "google.colab": colab}), \
                    patch.dict(os.environ, {variable: "1"}, clear=True), self.assertRaises(RuntimeError):
                exec(compile(code, str(UPDATED), "exec"), {})

    def test_local_rehearsal_never_claims_colab_or_writes_remote_report(self):
        final = cells(UPDATED)[-1]
        with tempfile.TemporaryDirectory() as base:
            local, remote = Path(base) / "local", Path(base) / "remote"
            local.mkdir()
            remote.mkdir()
            namespace = {"os": os, "json": json, "uuid": __import__("uuid"), "LOCAL_PACKAGE_DIR": local,
                "REMOTE_PACKAGE_DIR": remote, "_LOCAL_REHEARSAL": True,
                "transport_result": {"sha256": "a" * 64, "bytes": 100, "parts": 4},
                "summary": {"participants": 2, "verified_tensors": 3, "all_tensor_file_and_pixel_hashes_verified": True}}
            with patch.dict(os.environ, {"COLAB_RELEASE_TAG": "simulated"}):
                exec(compile(final, str(UPDATED), "exec"), namespace)
            self.assertFalse(namespace["FINAL_PREFLIGHT"]["colab_runtime_executed"])
            self.assertEqual(namespace["FINAL_REPORT_PATH"].parent, local)
            self.assertEqual(list(remote.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
