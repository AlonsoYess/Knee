"""Execute generated notebook41 launch preparation with an inert launcher."""
import ast
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("notebook41_builder", ROOT / "scripts/build_image_screening_notebook.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
REVISION = "1234567890abcdef1234567890abcdef12345678"


class ImageScreeningNotebookTests(unittest.TestCase):
    def test_clean_pinned_gpu_notebook_and_exact_approved_scope(self):
        notebook = builder.build(REVISION)
        self.assertEqual(notebook["metadata"]["accelerator"], "GPU")
        intro = notebook["cells"][0]["source"]
        self.assertIn("25 trabajos", intro)
        self.assertIn("Revisión 1.0", intro)
        for name in builder.MODEL_NAMES:
            self.assertIn(name, intro)
        self.assertEqual([intro.index(name) for name in builder.MODEL_NAMES],
                         sorted(intro.index(name) for name in builder.MODEL_NAMES))
        self.assertIn("no se vuelve a entrenar", intro)
        self.assertIn("DINOv3 permanece", intro)
        bootstrap = notebook["cells"][1]["source"]
        self.assertIn(repr(REVISION), bootstrap)
        self.assertIn("knee_image_screening41_", bootstrap)
        self.assertIn("CAMPAIGN = 'remaining_encoders_frozen'", bootstrap)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse(cell["source"])
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
                if "scripts/train_" in cell["source"]:
                    self.assertIn("scripts/train_image_screening_colab.py", cell["source"])
                self.assertNotIn("/content/drive/MyDrive/", cell["source"])

    def test_no_unpublished_revision_without_explicit_test_option(self):
        for revision in ("", "main", "0" * 40, "https://example.invalid"):
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                builder.build(revision)

    def test_gpu_hidden_only_for_synthetic_tests(self):
        notebook = builder.build(REVISION)
        tests = notebook["cells"][5]["source"]
        self.assertIn("TEST_ENV = dict(RUN_ENV)", tests)
        self.assertIn("TEST_ENV.update(CUDA_VISIBLE_DEVICES='', COLAB_GPU='0')", tests)
        self.assertIn("env=TEST_ENV", tests)
        train = notebook["cells"][9]["source"]
        self.assertIn("env=RUN_ENV", train)
        self.assertNotIn("CUDA_VISIBLE_DEVICES='',", train)

    def _launch(self, *, approved=True, repeat=False, changed_receipt=False):
        notebook = builder.build(REVISION)
        tree = ast.parse(notebook["cells"][9]["source"])
        launchers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "training_run"]
        self.assertEqual(len(launchers), 1)
        tree.body.remove(launchers[0])
        with tempfile.TemporaryDirectory(prefix="n41_cell_") as directory:
            root = Path(directory)
            campaign = builder.CAMPAIGN
            confirmation = "AUTORIZO CODIFICADORES V1.2 5 PLIEGUES"
            summary = {"fixture": "synthetic_no_training"}
            records, repository = root / "records", root / "repository"
            package, proof = root / "development", root / "proof"
            reference = root / "reference" / "inputs.json"
            session = root / "session"
            for path in (records, repository, package, proof, reference.parent, session):
                path.mkdir()
            reference.write_text("{}", encoding="utf-8")
            cache = session / "verified_public_weights"
            environment = {"CUDA_VISIBLE_DEVICES": "0"}

            def save(path, value):
                with Path(path).open("x", encoding="utf-8") as stream:
                    json.dump(value, stream)

            def launch(arguments, *, cwd, env):
                self.assertEqual(cwd, repository)
                self.assertIs(env, environment)
                arg = lambda name: arguments[arguments.index(name) + 1]
                self.assertEqual(arg("--campaign"), campaign)
                self.assertEqual(arg("--code-revision"), REVISION)
                self.assertEqual(arg("--operation"), "train")
                self.assertEqual(arg("--approval-statement"), "SYNTHETIC SCOPE")
                self.assertEqual(Path(arg("--weights-root")), cache)
                self.assertTrue(cache.is_dir())
                self.assertEqual(Path(arguments[3]).name, "train_image_screening_colab.py")
                for protected in (repository, proof, package, reference.parent, records):
                    self.assertFalse(cache.is_relative_to(protected))
                    self.assertFalse(protected.is_relative_to(cache))
                output = Path(arg("--output-dir"))
                output.mkdir()
                save(output / "summary.json", summary)
                return "synthetic launcher; no optimizer or participant training"

            orchestration = SimpleNamespace(CONFIRMATIONS={campaign: confirmation},
                APPROVAL_STATEMENTS={campaign: "SYNTHETIC SCOPE"}, SUMMARY_FILENAME="summary.json",
                _save=Mock(side_effect=save), _json=lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
                validate_result=Mock())
            launch_mock = Mock(side_effect=launch)
            namespace = {"input": Mock(return_value=confirmation if approved else "INVALID"),
                "orchestration": orchestration, "CAMPAIGN": campaign, "SESSION_ROOT": session,
                "CPU_TESTS": {"synthetic": "tests"}, "runtime": {"synthetic": "runtime"},
                "NEURAL_PYTHON": Path("synthetic_python"), "REPO_DIR": repository,
                "PREPARATION_DIR": proof, "LOCAL_PACKAGE_DIR": package,
                "LOCAL_REFERENCE": reference, "PERSISTENT_ROOT": records,
                "CODE_REVISION": REVISION, "RUN_ENV": environment, "json": json,
                "training_run": launch_mock}
            before = set(session.iterdir())
            with redirect_stdout(io.StringIO()):
                if not approved:
                    with self.assertRaisesRegex(RuntimeError, "No se confirmó"):
                        exec(compile(tree, "notebook41_cell", "exec"), namespace)
                    self.assertEqual(set(session.iterdir()), before)
                    launch_mock.assert_not_called()
                    orchestration._save.assert_not_called()
                    return
                exec(compile(tree, "notebook41_cell", "exec"), namespace)
                first_output = namespace["LOCAL_OUTPUT"]
                saved = {p.name: p.read_bytes() for p in (session / "synthetic_test_receipt.json", session / "setup_receipt.json")}
                if repeat:
                    if changed_receipt:
                        namespace["CPU_TESTS"] = {"synthetic": "changed"}
                        with self.assertRaisesRegex(RuntimeError, "otra preparación"):
                            exec(compile(tree, "notebook41_cell", "exec"), namespace)
                    else:
                        exec(compile(tree, "notebook41_cell", "exec"), namespace)
                        self.assertNotEqual(namespace["LOCAL_OUTPUT"], first_output)
                    for name, raw in saved.items():
                        self.assertEqual((session / name).read_bytes(), raw)
                expected_calls = 2 if repeat and not changed_receipt else 1
                self.assertEqual(launch_mock.call_count, expected_calls)
                self.assertEqual(orchestration._save.call_count, 2)
                self.assertEqual(namespace["FINAL_CHECK"], summary)

    def test_generated_cell_creates_cache_before_launch(self):
        self._launch()

    def test_repeated_cell_reuses_identical_receipts_and_existing_cache(self):
        self._launch(repeat=True)

    def test_changed_receipt_stops_before_relaunch(self):
        self._launch(repeat=True, changed_receipt=True)

    def test_bad_confirmation_launches_nothing(self):
        self._launch(approved=False)

    def test_cli_exposes_only_one_registered_campaign_and_no_test_operation(self):
        source = (ROOT / "scripts/train_image_screening_colab.py").read_text(encoding="utf-8")
        ast.parse(source)
        self.assertIn('choices=("remaining_encoders_frozen",)', source)
        self.assertIn('choices=("setup", "tests", "locate", "train")', source)
        self.assertIn("find_image_preflight(args.records_root, args.repository_root)", source)
        self.assertIn("campaign=args.campaign", source)
        self.assertNotIn("reserved_test", source)


if __name__ == "__main__":
    unittest.main()
