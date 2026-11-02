"""The new notebooks must invoke the current runner only after explicit consent."""
import ast
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("campaign_notebook_builder",
    ROOT / "scripts/build_methodology_campaign_notebooks.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
REVISION = "1234567890abcdef1234567890abcdef12345678"


class MethodologyCampaignNotebooksTests(unittest.TestCase):
    def _execute_generated_training_cell(self, campaign, *, authorized=True):
        cells = builder.build(REVISION, campaign)["cells"]
        source = next(cell["source"] for cell in cells
            if cell["cell_type"] == "code" and "'--operation', 'train'" in cell["source"])
        tree = ast.parse(source)
        launchers = [node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "training_run"]
        self.assertEqual(len(launchers), 1)
        # Replace only the subprocess launcher; execute the real cell's consent,
        # directory preparation, argument construction and result readback.
        tree.body.remove(launchers[0])
        with TemporaryDirectory(prefix="synthetic_campaign_cell_") as temporary:
            root = Path(temporary)
            session = root / "session"
            package = session / "development"
            reference = session / "reference" / "synthetic_reference.json"
            repository = root / "repository"
            preparation = root / "preparation"
            records = root / "records"
            for directory in (session, package, reference.parent, repository, preparation, records):
                directory.mkdir(parents=True, exist_ok=True)
            reference.write_text("{}", encoding="utf-8")
            cache = session / "verified_public_weights"
            environment = {"CUDA_VISIBLE_DEVICES": "0", "SYNTHETIC_FIXTURE": "1"}
            summary = {"synthetic_cell_fixture": True, "campaign": campaign}
            summary_name = "synthetic_campaign_summary.json"
            approval = "SYNTHETIC APPROVAL " + campaign
            confirmation = "SYNTHETIC CONFIRMATION " + campaign

            def save(path, value):
                Path(path).write_text(json.dumps(value), encoding="utf-8")

            def training_run(arguments, **keywords):
                self.assertEqual(keywords["cwd"], repository)
                self.assertIs(keywords["env"], environment)
                argument = lambda name: arguments[arguments.index(name) + 1]
                self.assertEqual(argument("--operation"), "train")
                self.assertEqual(argument("--campaign"), campaign)
                self.assertEqual(argument("--approval-statement"), approval)
                output = Path(argument("--output-dir"))
                if campaign == "densenet_frozen":
                    weights = Path(argument("--weights-root"))
                    self.assertEqual(weights, cache)
                    self.assertTrue(weights.is_dir(),
                        "separate_existing_external_weights_cache_required")
                    for other in (package, reference.parent, preparation, records, repository, output):
                        self.assertFalse(weights.resolve().is_relative_to(other.resolve()))
                        self.assertFalse(other.resolve().is_relative_to(weights.resolve()))
                else:
                    self.assertNotIn("--weights-root", arguments)
                    self.assertFalse(cache.exists())
                output.mkdir()
                save(output / summary_name, summary)
                return "synthetic launcher; no model execution"

            orchestration = SimpleNamespace(CONFIRMATIONS={campaign: confirmation},
                APPROVAL_STATEMENTS={campaign: approval}, SUMMARY_FILENAME=summary_name,
                _save=Mock(side_effect=save), validate_result=Mock())
            launcher = Mock(side_effect=training_run)
            namespace = {"input": Mock(return_value=confirmation if authorized else "WRONG CONFIRMATION"),
                "orchestration": orchestration, "CAMPAIGN": campaign, "SESSION_ROOT": session,
                "CPU_TESTS": {"synthetic_test_receipt": True}, "runtime": {"synthetic_runtime_receipt": True},
                "NEURAL_PYTHON": root / "synthetic_python", "REPO_DIR": repository,
                "PREPARATION_DIR": preparation, "LOCAL_PACKAGE_DIR": package,
                "LOCAL_REFERENCE": reference, "PERSISTENT_ROOT": records,
                "CODE_REVISION": REVISION, "RUN_ENV": environment, "json": json,
                "training_run": launcher}
            before = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
            with redirect_stdout(io.StringIO()):
                if authorized:
                    exec(compile(tree, "generated_training_cell", "exec"), namespace)
                else:
                    with self.assertRaisesRegex(RuntimeError, "No se confirmó la campaña"):
                        exec(compile(tree, "generated_training_cell", "exec"), namespace)
            namespace["input"].assert_called_once()
            if authorized:
                launcher.assert_called_once()
                self.assertEqual(orchestration._save.call_count, 2)
                self.assertEqual(namespace["FINAL_CHECK"], summary)
                orchestration.validate_result.assert_called_once_with(summary)
                for name, value in (("synthetic_test_receipt.json", namespace["CPU_TESTS"]),
                                    ("setup_receipt.json", namespace["runtime"])):
                    self.assertEqual(json.loads((session / name).read_text(encoding="utf-8")), value)
            else:
                launcher.assert_not_called()
                orchestration._save.assert_not_called()
                orchestration.validate_result.assert_not_called()
                self.assertFalse(cache.exists())
                self.assertEqual(sorted(str(path.relative_to(root)) for path in root.rglob("*")), before)

    def test_densenet_training_cell_prepares_separate_weights_cache_before_launch(self):
        self._execute_generated_training_cell("densenet_frozen")

    def test_clinical_training_cell_neither_creates_nor_passes_weights_cache(self):
        self._execute_generated_training_cell("clinical")

    def test_unconfirmed_training_cell_creates_nothing_and_does_not_launch(self):
        for campaign in builder.NOTEBOOKS:
            with self.subTest(campaign=campaign):
                self._execute_generated_training_cell(campaign, authorized=False)

    def test_each_notebook_is_clean_and_has_exact_revision(self):
        for campaign in builder.NOTEBOOKS:
            with self.subTest(campaign=campaign):
                value = builder.build(REVISION, campaign)
                self.assertEqual(value["nbformat"], 4)
                bootstrap = value["cells"][1]["source"]
                self.assertIn(repr(REVISION), bootstrap)
                self.assertIn("from knee import methodology_campaign_colab as orchestration", bootstrap)
                self.assertIn("CAMPAIGN = " + repr(campaign), bootstrap)
                for cell in value["cells"]:
                    if cell["cell_type"] == "code":
                        ast.parse(cell["source"])
                        self.assertIsNone(cell["execution_count"])
                        self.assertEqual(cell["outputs"], [])

    def test_resource_guards_are_preserved(self):
        clinical = builder.build(REVISION, "clinical")
        image = builder.build(REVISION, "densenet_frozen")
        self.assertIn("os.environ.get('COLAB_GPU', '0') == '1'", clinical["cells"][1]["source"])
        self.assertIn("os.environ.get('COLAB_GPU', '0') != '1'", image["cells"][1]["source"])
        self.assertNotIn("accelerator", clinical["metadata"])
        self.assertEqual(image["metadata"]["accelerator"], "GPU")

    def test_synthetic_checks_hide_gpu_without_changing_the_training_environment(self):
        for campaign in builder.NOTEBOOKS:
            with self.subTest(campaign=campaign):
                cells = builder.build(REVISION, campaign)["cells"]
                sources = [cell["source"] for cell in cells if cell["cell_type"] == "code"]
                test_source = next(source for source in sources if "'--operation', 'tests'" in source)
                training_source = next(source for source in sources if "'--operation', 'train'" in source)
                original = {"CUDA_VISIBLE_DEVICES": "0", "COLAB_GPU": "1", "SENTINEL": "preserved"}
                run_environment = dict(original)
                calls = []

                def checked_run(arguments, **kwargs):
                    calls.append((arguments, kwargs))
                    return json.dumps({"tests_run": 1, "skipped": 0})

                namespace = {"RUN_ENV": run_environment, "NEURAL_PYTHON": Path("isolated/python"),
                    "REPO_DIR": ROOT, "CAMPAIGN": campaign, "checked_run": checked_run, "json": json}
                with redirect_stdout(io.StringIO()):
                    exec(compile(ast.parse(test_source), "synthetic_test_cell", "exec"), namespace)
                self.assertEqual(len(calls), 1)
                arguments, keywords = calls[0]
                self.assertEqual(arguments[arguments.index("--operation") + 1], "tests")
                self.assertEqual(arguments[arguments.index("--campaign") + 1], campaign)
                self.assertEqual(keywords["env"], {**original, "CUDA_VISIBLE_DEVICES": "", "COLAB_GPU": "0"})
                self.assertIsNot(keywords["env"], run_environment)
                self.assertEqual(run_environment, original)
                training_call = next(node for node in ast.walk(ast.parse(training_source))
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "training_run")
                environment = next(keyword.value for keyword in training_call.keywords if keyword.arg == "env")
                self.assertIsInstance(environment, ast.Name)
                self.assertEqual(environment.id, "RUN_ENV")

    def test_notebook40_revision_describes_gpu_build_with_cpu_test_computation(self):
        clinical = builder.build(REVISION, "clinical")
        image = builder.build(REVISION, "densenet_frozen")
        self.assertIn("Revisión 1.0", clinical["cells"][0]["source"])
        self.assertIn("Revisión 1.2", image["cells"][0]["source"])
        clinical_tests = next(cell["source"] for cell in clinical["cells"]
            if cell["cell_type"] == "markdown" and cell["source"].startswith("## 2."))
        image_tests = next(cell["source"] for cell in image["cells"]
            if cell["cell_type"] == "markdown" and cell["source"].startswith("## 2."))
        self.assertIn("calculan en CPU con la compilación de PyTorch que incluye soporte para GPU", image_tests)
        self.assertIn("La GPU permanece oculta", image_tests)
        self.assertIn("no se entrena con datos de participantes", image_tests)
        self.assertNotIn("soporte para GPU", clinical_tests)

    def test_notebook40_setup_failure_describes_pretraining_campaign_checks(self):
        bootstrap = builder.build(REVISION, "densenet_frozen")["cells"][1]["source"]
        messages = {node.value for node in ast.walk(ast.parse(bootstrap))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)}
        self.assertIn("La comprobación previa de la campaña se detuvo. Conserva el error; no inicies entrenamiento.",
            messages)
        self.assertNotIn("La prueba breve se detuvo. Conserva el error; no intentes entrenamiento completo.", messages)

    def test_only_one_training_cell_and_confirmation_precedes_execution(self):
        for campaign in builder.NOTEBOOKS:
            value = builder.build(REVISION, campaign)
            cells = [c["source"] for c in value["cells"] if c["cell_type"] == "code"]
            training = [c for c in cells if "'--operation', 'train'" in c]
            self.assertEqual(len(training), 1)
            self.assertLess(training[0].index("if confirmation !="), training[0].index("training_log = training_run("))
            self.assertIn("raise RuntimeError('No se confirmó la campaña.", training[0])
            for source in cells:
                self.assertNotIn("train_clinical_fivefold_colab.py", source)
                self.assertNotIn("train_densenet_fivefold_colab.py", source)
                self.assertNotIn("run_development_training.py", source)
                self.assertNotIn(".backward(", source)

    def test_completed_preflight_is_located_before_training(self):
        for campaign in builder.NOTEBOOKS:
            cells = builder.build(REVISION, campaign)["cells"]
            sources = [c["source"] for c in cells if c["cell_type"] == "code"]
            locate = next(i for i, c in enumerate(sources) if "'--operation', 'locate'" in c)
            train = next(i for i, c in enumerate(sources) if "'--operation', 'train'" in c)
            self.assertLess(locate, train)
            self.assertIn("PREPARATION_DIR = Path(proof['preparation_dir'])", sources[locate])
            self.assertIn("'--preparation-dir', str(PREPARATION_DIR)", sources[train])
            self.assertIn("'--test-receipt'", sources[train])
            self.assertIn("'--runtime-receipt'", sources[train])

    def test_scope_and_privacy_are_explicit(self):
        clinical = builder.build(REVISION, "clinical")
        image = builder.build(REVISION, "densenet_frozen")
        self.assertIn("45 trabajos", clinical["cells"][0]["source"])
        self.assertIn("WOMAC", clinical["cells"][0]["source"])
        self.assertIn("gamma", image["cells"][0]["source"])
        self.assertIn("codificador congelado", image["cells"][0]["source"])
        for value in (clinical, image):
            text = "\n".join(c["source"] for c in value["cells"])
            self.assertIn("prueba reservada", text)
            self.assertNotIn("/content/drive/MyDrive/", text)
            self.assertNotIn("D:\\", text)
            self.assertNotIn("C:\\Users", text)
            self.assertIn("No es un porcentaje de avance", text)

    def test_floating_invalid_revision_or_unregistered_campaign_rejected(self):
        for revision in ("main", "", "0" * 40):
            with self.assertRaises(ValueError):
                builder.build(revision, "clinical")
        with self.assertRaises(ValueError):
            builder.build(REVISION, "all_models")


if __name__ == "__main__":
    unittest.main()
