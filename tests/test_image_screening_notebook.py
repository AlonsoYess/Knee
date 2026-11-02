"""Execute generated notebook41 launch preparation with an inert launcher."""
import ast
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
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
        self.assertIn("Revisión 2.0", intro)
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
        self.assertIn("image_campaign_recovery as orchestration", bootstrap)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse(cell["source"])
                self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "input" for node in ast.walk(ast.parse(cell["source"]))),
                    "The authorized recovery notebook must not ask for an interactive input")
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
            input_mock = Mock(side_effect=AssertionError("Interactive input is forbidden in recovery"))
            namespace = {"input": input_mock,
                "orchestration": orchestration, "CAMPAIGN": campaign if approved else "reserved_test", "SESSION_ROOT": session,
                "CPU_TESTS": {"synthetic": "tests"}, "runtime": {"synthetic": "runtime"},
                "NEURAL_PYTHON": Path("synthetic_python"), "REPO_DIR": repository,
                "PREPARATION_DIR": proof, "LOCAL_PACKAGE_DIR": package,
                "LOCAL_REFERENCE": reference, "PERSISTENT_ROOT": records,
                "CODE_REVISION": REVISION, "RUN_ENV": environment, "json": json,
                "training_run": launch_mock}
            before = set(session.iterdir())
            with redirect_stdout(io.StringIO()):
                if not approved:
                    with self.assertRaises(RuntimeError):
                        exec(compile(tree, "notebook41_cell", "exec"), namespace)
                    self.assertEqual(set(session.iterdir()), before)
                    launch_mock.assert_not_called()
                    orchestration._save.assert_not_called()
                    input_mock.assert_not_called()
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
                input_mock.assert_not_called()

    def test_generated_cell_creates_cache_before_launch(self):
        self._launch()

    def test_repeated_cell_reuses_identical_receipts_and_existing_cache(self):
        self._launch(repeat=True)

    def test_changed_receipt_stops_before_relaunch(self):
        self._launch(repeat=True, changed_receipt=True)

    def test_unregistered_campaign_launches_nothing(self):
        self._launch(approved=False)

    def _configured_paths(self, *, updates=None, missing_transport=False,
            raw_settings=None, external_project=False, lexical_escape=False, project_override=False):
        """Run the real generated locator cell; replace only Drive mounting and subprocess I/O."""
        from knee.colab_data import checked_path
        source = builder.build(REVISION)["cells"][7]["source"]
        tree = ast.parse(source)
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.ImportFrom) and node.module == "google.colab"
            or isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "mount"
                and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == "drive"
            or isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                and target.id == "MY_DRIVE_ROOT" for target in node.targets))]
        with tempfile.TemporaryDirectory(prefix="n41_paths_") as directory:
            root = Path(directory)
            drive_root = root / "mounted_drive"
            project = (root / "outside") if external_project or lexical_escape else (
                drive_root / ("custom_project" if project_override else "KneeOAI"))
            records, development = project / "private_records", project / "private_development"
            session, repository, preparation = root / "session", root / "repository", root / "proof"
            for path in (drive_root, project, records, development, session, repository, preparation):
                path.mkdir(parents=True, exist_ok=True)
            # These exist so dotted/repeated-separator paths cannot pass merely
            # because their normalized destination happens to be absent.
            (records / "child").mkdir()
            if not missing_transport:
                (development / "transport.json").write_text("{}", encoding="utf-8")
            settings = {"version": "knee_colab_runtime_paths_v1.0", "records_relative": "private_records",
                "development_relative": "private_development"}
            settings.update(updates or {})
            (project / "colab_runtime_paths.json").write_text(
                raw_settings if raw_settings is not None else json.dumps(settings), encoding="utf-8")
            environ = {}
            if external_project or project_override:
                environ["KNEE_DATA_ROOT"] = str(project)
            if lexical_escape:
                environ["KNEE_DATA_ROOT"] = str(drive_root / ".." / "outside")
            stage = Mock(return_value={"synthetic": True})
            locate = Mock(return_value=json.dumps({"preparation_dir": str(preparation)}))
            input_mock = Mock(side_effect=AssertionError("No path questions are allowed"))
            namespace = {"Path": Path, "PurePosixPath": PurePosixPath, "json": json,
                "os": SimpleNamespace(environ=environ), "MY_DRIVE_ROOT": drive_root,
                "orchestration": SimpleNamespace(checked_path=checked_path,
                    _json=lambda p: json.loads(Path(p).read_text(encoding="utf-8")), stage_development=stage),
                "checked_run": locate, "NEURAL_PYTHON": Path("synthetic_python"),
                "REPO_DIR": repository, "RUN_ENV": {"synthetic": "no_training"},
                "CAMPAIGN": builder.CAMPAIGN, "SESSION_ROOT": session, "input": input_mock}
            invalid = bool(updates or missing_transport or raw_settings is not None or external_project or lexical_escape)
            with redirect_stdout(io.StringIO()):
                if invalid:
                    with self.assertRaises((RuntimeError, ValueError)):
                        exec(compile(tree, "notebook41_paths", "exec"), namespace)
                    locate.assert_not_called()
                    stage.assert_not_called()
                    self.assertEqual(list(session.iterdir()), [])
                else:
                    exec(compile(tree, "notebook41_paths", "exec"), namespace)
                    self.assertEqual(namespace["PROJECT_ROOT"], project)
                    self.assertEqual(namespace["PERSISTENT_ROOT"], records)
                    self.assertEqual(namespace["REMOTE_PACKAGE_DIR"], development)
                    locate.assert_called_once()
                    stage.assert_called_once_with(development, session / "development",
                        session / "reference" / "model_inputs_loading_reference_v1.json")
                input_mock.assert_not_called()

    def test_private_path_configuration_locates_and_stages_without_input(self):
        self._configured_paths()

    def test_project_override_must_still_be_within_mounted_drive(self):
        self._configured_paths(project_override=True)
        self._configured_paths(external_project=True)
        self._configured_paths(lexical_escape=True)

    def test_wrong_path_configuration_version_fails_before_staging(self):
        self._configured_paths(updates={"version": "unknown"})

    def test_path_configuration_rejects_escape_absolute_and_ambiguous_paths(self):
        for relative in ("../outside", "/outside", "C:/outside", "private_records/../outside",
                "private_records\\child", "private_records/./child", "private_records//child", ""):
            with self.subTest(relative=relative):
                self._configured_paths(updates={"records_relative": relative})

    def test_path_configuration_requires_bounded_object_without_extra_fields(self):
        self._configured_paths(raw_settings=" " * 16385)
        self._configured_paths(raw_settings="[]")
        self._configured_paths(updates={"extra": "not allowed"})

    def test_path_configuration_rejects_missing_transport_and_overlap(self):
        self._configured_paths(missing_transport=True)
        self._configured_paths(updates={"records_relative": "private_development"})

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
