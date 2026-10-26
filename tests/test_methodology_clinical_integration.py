"""Actual runner routing with synthetic artifacts and mocked fits only.

No estimator fit, optimizer, backward pass, private cohort or reserved package
is used. Neural construction/forward uses seeded random weights in eval mode.
"""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from knee import clinical_scenarios as clinical
from knee import clinical_fold_inputs as historical
from knee import methodology_release as release
from knee.aligned_training import authority, data, engine, models
from knee.aligned_training import runner
from knee.colab_data import sha_file
from test_training_data import reference_fixture
from test_training_runner import request as historical_request
from test_training_runner import authority as historical_authority
from test_training_runner import fake_conventional_fit

ROOT = Path(__file__).resolve().parents[1]


def request(package_sha256, scenario, model_id="clinical_logistic", sidecar_sha256=None):
    result = historical_request(package_sha256, model_id=model_id)
    result.update(version=release.REQUEST_VERSION, clinical_scenario=scenario,
        baseline_sidecar_sha256=sidecar_sha256, augmentation_policy="identity",
        modeling_release_sha256=release.PINS[release.RELEASE_PATH])
    return result


def permit_record(req, registry):
    record = historical_authority(req, registry)
    record["modeling_release_sha256"] = release.PINS[release.RELEASE_PATH]
    return record


class MethodologyClinicalIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.catalog, cls.plan, cls.registry = release.load_release(ROOT)
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fixture(self, directory):
        base = Path(directory)
        sources = base / "sources"
        sources.mkdir()
        fixture = reference_fixture(sources)
        projection = []
        for i, row in enumerate(fixture["rows"]):
            projection.append({**{key: row[key] for key in ("src_subject_id", "side", "AGEYEARS", "SEX", "BMI")},
                "KL_BASE": 2 + i % 2, "CIRUGIA_PREVIA": None if i % 7 == 0 else i % 2,
                "WOMAC_TOTAL": None if i % 6 == 0 else float(i % 20)})
        fixture["sidecar"] = clinical.make_baseline_sidecar(projection, fixture["rows"],
            fixture["package_sha256"], "d" * 64)
        fixture["sidecar_path"] = sources / "synthetic_baseline_only.json"
        runner.save_json(fixture["sidecar_path"], fixture["sidecar"])
        fixture["outputs"] = base / "outputs"
        fixture["outputs"].mkdir()
        return fixture

    def run_mocked(self, fixture, scenario, model_id, *, request_override=None, authority_override=None,
                   conventional_effect=None, neural_effect=None, sidecar_path_override=None):
        sidecar_hash = None if scenario == "clinical_main" else fixture["sidecar"]["record_sha256"]
        req = request(fixture["package_sha256"], scenario, model_id, sidecar_hash)
        if request_override:
            req.update(request_override)
        authorization = permit_record(req, self.registry)
        if authority_override:
            authorization.update(authority_override)
        output = fixture["outputs"] / (model_id + "_" + scenario)
        sidecar_path = None if scenario == "clinical_main" else fixture["sidecar_path"]
        if sidecar_path_override:
            sidecar_path = sidecar_path_override
        with patch.object(runner, "_fit_conventional", side_effect=conventional_effect or fake_conventional_fit) as fit, \
             patch.object(engine, "train_neural_fold", side_effect=neural_effect or self.fake_neural_fit) as neural, \
             patch("sklearn.linear_model.LogisticRegression.fit", side_effect=AssertionError("Actual estimator fit forbidden")), \
             patch("xgboost.XGBClassifier.fit", side_effect=AssertionError("Actual estimator fit forbidden")), \
             patch("torch.optim.AdamW", side_effect=AssertionError("Actual optimizer forbidden")), \
             patch("torch.Tensor.backward", side_effect=AssertionError("Actual backward forbidden")):
            result = runner.execute(req, authorization, repository_root=ROOT,
                package_dir=fixture["package"], reference_path=fixture["reference_path"], weights_root=None,
                output_dir=output, baseline_sidecar_path=sidecar_path)
        return result, fit, neural, output

    @staticmethod
    def fake_neural_fit(model, training, validation, plan, **kwargs):
        # Reuse the real binding validator, then perform inference only.
        engine._validate_loader_bindings(training, validation, kwargs["execution_authority"], kwargs["positive_weight"])
        model.eval()
        batch = next(iter(validation))
        with torch.inference_mode():
            logits = model(clinical=batch["clinical"])
        output = kwargs["output_dir"]
        output.mkdir()
        runner.save_json(output / "best_validation_predictions.json", {
            "targets": list(validation.dataset._targets), "logits": logits.tolist(),
            "identities": [list(pair) for pair in validation.dataset.identities]})
        runner.save_json(output / "training_record.json", {"synthetic_mock": True,
            "optimizer_created": False, "training_executed": False})
        for name in ("best_model.safetensors", "last_model.safetensors"):
            (output / name).write_bytes(b"SYNTHETIC MOCK ARTIFACT; NO TRAINED MODEL")
        return {"status": "training_fold_complete", "optimizer_created": False}

    def test_runner_routes_all_nine_clinical_scenario_algorithm_combinations(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            original_archive = sha_file(fixture["archive"])
            for scenario, features in clinical.SCENARIOS.items():
                for model_id in ("clinical_logistic", "clinical_xgboost", "clinical_mlp"):
                    with self.subTest(scenario=scenario, model=model_id):
                        record, conventional, neural, output = self.run_mocked(fixture, scenario, model_id)
                        self.assertEqual(record["request"]["clinical_scenario"], scenario)
                        self.assertEqual(record["plan_sha256"], release.PINS[release.PLAN_PATH])
                        self.assertFalse(record["reserved_test_accessed"])
                        state = runner.read_json(output / "clinical_state.json")
                        self.assertEqual(state["feature_order"], list(features))
                        self.assertEqual(runner.completed_job(output, fixture["package_sha256"], self.plan, self.registry), record)
                        if model_id == "clinical_mlp":
                            self.assertEqual(conventional.call_count, 0)
                            self.assertEqual(neural.call_count, 1)
                            model, train, validation, _ = neural.call_args.args
                            self.assertEqual(model.clinical_dimensions, len(features))
                            self.assertEqual(train.dataset._clinical.shape[1], len(features))
                            self.assertEqual(validation.dataset._clinical.shape[1], len(features))
                        else:
                            self.assertEqual(neural.call_count, 0)
                            self.assertEqual(conventional.call_count, 1)
                            fold = conventional.call_args.args[1]
                            self.assertEqual(fold.training._clinical.shape[1], len(features))
                            self.assertEqual(fold.validation._clinical.shape[1], len(features))
            self.assertEqual(sha_file(fixture["archive"]), original_archive)

    def test_aligned_main_preserves_historical_main_feature_values(self):
        from knee import training_data as historical_data
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            historical_fold = historical_data.prepare_fold(fixture["rows"], fixture["tensors"], 0,
                                                package_sha256=fixture["package_sha256"])
            record, fit, _, output = self.run_mocked(fixture, "clinical_main", "clinical_logistic")
            aligned_fold = fit.call_args.args[1]
            np.testing.assert_array_equal(aligned_fold.training._clinical, historical_fold.training._clinical)
            np.testing.assert_array_equal(aligned_fold.validation._clinical, historical_fold.validation._clinical)
            self.assertEqual(aligned_fold.class_weight, historical_fold.class_weight)
            self.assertEqual(aligned_fold.clinical_state["main_state"], historical_fold.clinical_state)

    def test_three_mlp_dimensions_forward_without_fit_or_optimizer(self):
        before = torch.get_rng_state().clone()
        with patch("torch.optim.AdamW", side_effect=AssertionError("Actual optimizer forbidden")):
            for scenario, features in clinical.SCENARIOS.items():
                model, audit = models.build_model("clinical_mlp", self.plan, self.catalog, clinical_scenario=scenario)
                self.assertEqual(model.clinical_encoder[0].in_features, len(features))
                with torch.inference_mode():
                    output = model(clinical=torch.ones(3, len(features)))
                self.assertEqual(tuple(output.shape), (3,))
                self.assertFalse(audit["training_executed"])
                self.assertFalse(audit["optimizer_created"])
                with torch.inference_mode(), self.assertRaises(ValueError):
                    model(clinical=torch.ones(3, len(features) + 1))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_wrong_sidecar_binding_fails_before_package_is_opened(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            with patch.object(data, "load_sealed_inputs", side_effect=AssertionError("Package access forbidden")):
                with self.assertRaisesRegex(ValueError, "baseline_sidecar_does_not_match_execution_authority"):
                    self.run_mocked(fixture, "clinical_plus_baseline_kl", "clinical_logistic",
                                    request_override={"baseline_sidecar_sha256": "c" * 64})

    def test_tampered_baseline_payload_fails_before_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            value = deepcopy(fixture["sidecar"])
            value["rows"][0]["KL_BASE"] = 5 - value["rows"][0]["KL_BASE"]
            # A different temporary file keeps the original fixture immutable.
            tampered = fixture["sidecar_path"].parent / "tampered.json"
            runner.save_json(tampered, value)
            with self.assertRaisesRegex(ValueError, "sidecar_seal_changed"):
                self.run_mocked(fixture, "clinical_plus_baseline_kl", "clinical_logistic",
                    sidecar_path_override=tampered,
                    conventional_effect=lambda *args: self.fail("Fit reached with altered sidecar"))

    def test_wrong_recipe_or_authority_release_rejected_before_data_access(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            for field, override in (("request", {"modeling_release_sha256": "e" * 64}),
                                    ("authority", {"modeling_release_sha256": "e" * 64})):
                with self.subTest(field=field), patch.object(data, "load_sealed_inputs",
                            side_effect=AssertionError("Package access forbidden")):
                    with self.assertRaises(ValueError):
                        self.run_mocked(fixture, "clinical_main", "clinical_logistic",
                            request_override=override if field == "request" else None,
                            authority_override=override if field == "authority" else None)

    def test_missing_training_authority_blocks_even_known_synthetic_request(self):
        req = request("a" * 64, "clinical_main")
        with patch.object(release, "load_release", side_effect=AssertionError("No repository read allowed")):
            with self.assertRaisesRegex(ValueError, "separate_training_authority"):
                runner.execute(req, None, repository_root=ROOT, package_dir="absent", reference_path="absent",
                               weights_root=None, output_dir="absent")

    def test_legacy_request_cannot_enable_complementary_features(self):
        from knee.final_model_release import load_release
        _, _, legacy_plan, registry = load_release(ROOT)
        req = request("a" * 64, "clinical_plus_baseline_kl", sidecar_sha256="b" * 64)
        with self.assertRaisesRegex(ValueError, "methodology_training_plan_changed"):
            authority.validate_request(req, legacy_plan, registry)


if __name__ == "__main__":
    unittest.main()
