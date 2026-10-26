"""The new notebooks must invoke the current runner only after explicit consent."""
import ast
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("campaign_notebook_builder",
    ROOT / "scripts/build_methodology_campaign_notebooks.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
REVISION = "1234567890abcdef1234567890abcdef12345678"


class MethodologyCampaignNotebooksTests(unittest.TestCase):
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
