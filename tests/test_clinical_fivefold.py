"""Artificial clinical artifacts and mocked execution; never fit a predictor."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import save_file

from knee import clinical_fivefold as subject
from knee import training_runner as runner
from knee.training_authority import digest, validate_authority
from knee.training_engine import binary_metrics, learning_rate_factor, VERSION as ENGINE_VERSION
from knee.final_model_release import load_release, PINS
from knee.colab_data import sha_file
from knee.clinical_fold_inputs import fit_clinical_fold, transform_clinical_rows
from test_training_data import synthetic_inputs, reference_fixture
from test_training_smoke import preparation_fixture

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def fake_job(directory, request, rows, registry):
    """Serialized initialized/perturbed weights stand in for training artifacts."""
    root = Path(directory)
    root.mkdir()
    _, catalog, plan, _ = load_release(ROOT)
    fold, model_id = request["validation_fold"], request["model_id"]
    training = [row for row in rows if row["cv_fold"] != fold]
    validation = [row for row in rows if row["cv_fold"] == fold]
    ids = sorted([[row["src_subject_id"], row["side"]] for row in training])
    participants = sorted({row[0] for row in ids})
    state = fit_clinical_fold(rows, fold)
    clinical = transform_clinical_rows(validation, state)
    if model_id == "clinical_logistic":
        model = {"family": "logistic_regression", "coef": [[.2, -.3, .1]],
            "intercept": [.05], "classes": [0, 1], "n_features_in": 3}
        logits = (clinical.astype(np.float64) @ np.asarray(model["coef"]).T).reshape(-1) + .05
        write_json(root / "clinical_model.json", model)
    elif model_id == "clinical_xgboost":
        logits = clinical.astype(np.float64).sum(axis=1)
        (root / "clinical_model.ubj").write_bytes(b"MOCKED_SYNTHETIC_XGBOOST_NO_TRAINING")
    else:
        from knee.training_models import build_model
        model, _ = build_model(model_id, plan, catalog, seed=subject.SEED)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(.01)
        model.eval()
        with torch.inference_mode():
            logits = model(clinical=torch.from_numpy(clinical.copy())).numpy()
        adopted = {key: value.detach().clone().contiguous() for key, value in model.state_dict().items()}
        for name in ("best_model.safetensors", "last_model.safetensors"):
            save_file(adopted, str(root / name))
    predictions = {"targets": [row["PROGRESION_48M"] for row in validation], "logits": logits.tolist(),
        "identities": [[row["src_subject_id"], row["side"]] for row in validation]}
    metrics = binary_metrics(predictions["targets"], predictions["logits"])
    write_json(root / "best_validation_predictions.json", predictions)
    write_json(root / "clinical_state.json", state)
    if model_id == "clinical_mlp":
        updates = math.ceil(len(training) / 64)
        history = []
        for epoch in range(1, 10):
            audit = [{"update_index_within_epoch": index, "start_scale": 1., "successful_scale": 1.,
                "scale_after_update": 1., "backoffs": 0, "attempts": 1,
                "gradient_norm_before_clip": .5} for index in range(1, updates + 1)]
            history.append({"epoch": epoch, "phase": "full", "training_knees_seen": len(training),
                "optimizer_updates": updates, "learning_rate_factor": learning_rate_factor(epoch),
                "weighted_loss_per_knee": .4, "validation": metrics, "selected_best": epoch == 1,
                "amp_overflow_replays": 0, "amp_scale_audit": audit})
        positives = sum(row["PROGRESION_48M"] for row in training)
        engine = {"version": ENGINE_VERSION, "model_id": model_id, "stage": "stage_1", "seed": 2026,
            "smoke": False, "scope": "development_only", "status": "training_fold_complete",
            "eligible_for_model_selection": True, "reserved_test_accessed": False,
            "optimizer_created": True, "training_executed": True,
            "micro_batch_size": 64, "effective_batch_size": 64, "device": "cpu", "precision": "float32",
            "scheduler_horizon_epochs": 60, "restored_best_checkpoint": True, "resume_supported": False,
            "optimizer_state_saved": False, "gpu_resource_profile_recorded": False,
            "provenance": {"validation_fold": fold,
                "development_package_sha256": request["development_package_sha256"],
                "train_identity_sha256": digest(ids), "request_sha256": digest(request),
                "training_knees": len(training), "validation_knees": len(validation),
                "trained_participants": participants},
            "positive_weight": (len(training) - positives) / positives,
            "history": history, "completed_epochs": 9, "optimizer_updates": 9 * updates,
            "best_epoch": 1, "best_validation_ap": metrics["average_precision_score"],
            "amp_overflow_replays": 0, "amp_scale_audit": [{"epoch": epoch["epoch"], **audit}
                for epoch in history for audit in epoch["amp_scale_audit"]],
            "checkpoint_sha256": {name: sha_file(root / name) for name in
                ("best_model.safetensors", "last_model.safetensors")},
            "resource_profile": {"device_type": "cpu", "cuda_version": None, "cuda_peak_measured": False,
                "torch_version": "2.6.0+cpu", "elapsed_seconds": 1.}}
        write_json(root / "training_record.json", engine)
    record = {"version": runner.VERSION, "scope": "development_only", "request": request,
        "request_sha256": digest(request), "plan_sha256": PINS["configs/training_plan_v1_1.json"],
        "eligible_for_model_selection": True, "reserved_test_accessed": False,
        "weights_provenance": {"fit_scope": "development_training_fold", "validation_fold": fold,
            "train_identity_sha256": digest(ids), "trained_participants": participants,
            "reserved_test_accessed": False}, "metrics": metrics,
        "eligibility_gates": {key: False for key in registry["eligibility_gates"]},
        "selection_evidence": None, "engine_status": "training_fold_complete", "training_executed": True,
        "files_sha256": {path.name: sha_file(path) for path in root.iterdir()}}
    record["record_sha256"] = digest(record)
    write_json(root / runner.JOB_FILENAME, record)
    return record


def reseal(directory):
    root = Path(directory)
    job = json.loads((root / runner.JOB_FILENAME).read_text())
    job["files_sha256"] = {name: sha_file(root / name) for name in job["files_sha256"]}
    job["record_sha256"] = digest({key: value for key, value in job.items() if key != "record_sha256"})
    write_json(root / runner.JOB_FILENAME, job)


class FakeXGB:
    def load_model(self, path):
        if Path(path).read_bytes() != b"MOCKED_SYNTHETIC_XGBOOST_NO_TRAINING":
            raise ValueError("synthetic artifact changed")

    def get_booster(self):
        return SimpleNamespace(num_features=lambda: 3, num_boosted_rounds=lambda: 200)

    def predict(self, values, output_margin):
        assert output_margin is True
        return values.astype(np.float64).sum(axis=1)


class ClinicalCampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="clinical37_synthetic_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        _, self.catalog, self.plan, self.registry = load_release(ROOT)
        self.rows, _ = synthetic_inputs()
        self.xgb = patch("xgboost.XGBClassifier", FakeXGB)
        self.xgb.start()
        self.addCleanup(self.xgb.stop)

    def job(self, model="clinical_logistic", fold=0, name="job"):
        request = subject.make_request("a" * 64, model, fold)
        root = self.root / name
        fake_job(root, request, self.rows, self.registry)
        return root, request

    def validate(self, root, request):
        return subject.validate_completed_job(root, request, self.plan, self.registry, self.rows)

    def test_exact_fifteen_requests_without_image_or_selection_authority(self):
        for model in subject.MODEL_IDS:
            for fold in range(5):
                request = subject.make_request("a" * 64, model, fold)
                self.assertEqual(request["operation"], "development_fold")
                self.assertEqual(request["stage"], "stage_1")
                self.assertIsNone(request["image_encoder_id"])
                self.assertIsNone(request["selection_evidence_sha256"])
                authority = subject._authority(request, self.registry, subject.APPROVAL_STATEMENT)
                permit = validate_authority(authority, request, self.plan, self.registry)
                self.assertFalse(permit.full_encoder_training_authorized)
                self.assertFalse(any(authority["eligibility_gates"].values()))
        for model, fold, checksum in [("image_densenet121", 0, "a" * 64),
                ("clinical_logistic", True, "a" * 64), ("clinical_logistic", 5, "a" * 64),
                ("clinical_mlp", "0", "a" * 64), ("clinical_mlp", 0, "bad")]:
            with self.subTest(model=model, fold=fold), self.assertRaises(ValueError):
                subject.make_request(checksum, model, fold)
        request = subject.make_request("a" * 64, "clinical_mlp", 0)
        for change in ({"seed": 2027}, {"operation": "single_fold_smoke"}, {"image_encoder_id": "image_densenet121"}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "clinical_only"):
                subject._authority({**request, **change}, self.registry, subject.APPROVAL_STATEMENT)

    def test_no_approval_stops_before_runtime_or_data(self):
        with patch.object(subject, "verify_cpu_environment") as runtime:
            with self.assertRaisesRegex(ValueError, "approval"):
                subject.run("missing", "missing", "missing", "missing", "missing", approval_statement="")
            runtime.assert_not_called()

    def test_all_three_saved_model_artifacts_reproduce_validation_logits(self):
        for model in subject.MODEL_IDS:
            root, request = self.job(model, name=model)
            actual, engine = self.validate(root, request)
            self.assertEqual(actual["request"], request)
            self.assertEqual(engine is None, model != "clinical_mlp")

    def test_resealed_modified_logits_fail_model_replay(self):
        root, request = self.job()
        file = root / "best_validation_predictions.json"
        predictions = json.loads(file.read_text())
        predictions["logits"][0] += .1
        write_json(file, predictions)
        record = json.loads((root / runner.JOB_FILENAME).read_text())
        record["metrics"] = binary_metrics(predictions["targets"], predictions["logits"])
        write_json(root / runner.JOB_FILENAME, record)
        reseal(root)
        with self.assertRaisesRegex(ValueError, "reproduce"):
            self.validate(root, request)

    def test_resealed_global_fit_or_wrong_training_identity_fails(self):
        for index, name in enumerate(("state", "provenance")):
            root, request = self.job(name=name)
            if name == "state":
                file = root / "clinical_state.json"
                value = json.loads(file.read_text())
                value["BMI_median"] += .1
            else:
                file = root / runner.JOB_FILENAME
                value = json.loads(file.read_text())
                value["weights_provenance"]["trained_participants"].append("NOT_TRAINED")
            write_json(file, value)
            reseal(root)
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.validate(root, request)

    def test_wrong_validation_order_and_unlisted_artifacts_fail(self):
        root, request = self.job()
        file = root / "best_validation_predictions.json"
        value = json.loads(file.read_text())
        value["identities"] = value["identities"][::-1]
        write_json(file, value)
        reseal(root)
        with self.assertRaisesRegex(ValueError, "identity"):
            self.validate(root, request)
        root2, request2 = self.job(name="extra")
        (root2 / "unexpected.txt").write_text("synthetic")
        with self.assertRaisesRegex(ValueError, "artifacts"):
            self.validate(root2, request2)

    def test_mlp_partial_gpu_wrong_weight_epoch_and_audit_rejected(self):
        changes = [lambda e: e.update(device="cuda:0"), lambda e: e.update(smoke=True),
            lambda e: e.update(positive_weight=99.), lambda e: e.update(best_epoch=9),
            lambda e: e.update(completed_epochs=8), lambda e: e.update(optimizer_updates=1),
            lambda e: e["history"][0].update(phase="head_only"),
            lambda e: e["history"][0]["amp_scale_audit"][0].update(successful_scale=2.),
            lambda e: e["history"][0].update(training_knees_seen=2),
            lambda e: e.update(amp_scale_audit=[])]
        for index, change in enumerate(changes):
            root, request = self.job("clinical_mlp", name=str(index))
            file = root / "training_record.json"
            engine = json.loads(file.read_text())
            change(engine)
            write_json(file, engine)
            reseal(root)
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.validate(root, request)

    def test_saved_logistic_parameter_shape_and_changed_prediction_fail(self):
        for index, change in enumerate((lambda v: v.update(n_features_in=4),
                lambda v: v.update(coef=[[1., 2.]]), lambda v: v.update(intercept=[.7]))):
            root, request = self.job(name=str(index))
            file = root / "clinical_model.json"
            model = json.loads(file.read_text())
            change(model)
            write_json(file, model)
            reseal(root)
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.validate(root, request)

    def test_constant_reference_uses_training_prevalence_only(self):
        result = subject.constant_reference(self.rows)
        self.assertEqual(result["fit_source"], "training_fold_prevalence_only")
        self.assertFalse(result["calibration_performed"])
        for report in result["fold_results"]:
            training = [r for r in self.rows if r["cv_fold"] != report["validation_fold"]]
            expected = sum(r["PROGRESION_48M"] for r in training) / len(training)
            self.assertEqual(report["training_prevalence"], expected)
            self.assertEqual(report["roc_auc"], .5)
        self.assertEqual(result["fivefold_selection_metrics"]["roc_auc"], .5)

    def test_complete_oof_has_every_development_knee_once(self):
        jobs = {}
        for fold in range(5):
            root, request = self.job(fold=fold, name=str(fold))
            jobs[fold] = (root, self.validate(root, request)[0])
        means, pooled, rows = subject.aggregate_predictions(jobs, self.rows)
        self.assertEqual(len(rows), len(self.rows))
        self.assertEqual(len({(r["src_subject_id"], r["side"]) for r in rows}), len(self.rows))
        self.assertEqual(means["average_precision_score"], sum(j[1]["metrics"]["average_precision_score"] for j in jobs.values()) / 5)
        jobs.pop(4)
        with self.assertRaisesRegex(ValueError, "all_five"):
            subject.aggregate_predictions(jobs, self.rows)

    def test_aggregation_rejects_swapped_fold_and_duplicate_oof(self):
        jobs = {}
        for fold in range(5):
            root, request = self.job(fold=fold, name=str(fold))
            jobs[fold] = (root, self.validate(root, request)[0])
        jobs[0], jobs[1] = jobs[1], jobs[0]
        with self.assertRaisesRegex(ValueError, "fold_identity"):
            subject.aggregate_predictions(jobs, self.rows)

    def test_aggregation_rejects_model_mixing_or_changed_completed_predictions(self):
        jobs = {}
        for fold in range(5):
            root, request = self.job(fold=fold, name=str(fold))
            jobs[fold] = (root, self.validate(root, request)[0])
        different, request = self.job("clinical_mlp", fold=2, name="different")
        mixed = {**jobs, 2: (different, self.validate(different, request)[0])}
        with self.assertRaisesRegex(ValueError, "fold_identity"):
            subject.aggregate_predictions(mixed, self.rows)
        file = jobs[0][0] / "best_validation_predictions.json"
        value = json.loads(file.read_text())
        value["logits"][0] += .5
        write_json(file, value)
        with self.assertRaisesRegex(ValueError, "changed_before_aggregation"):
            subject.aggregate_predictions(jobs, self.rows)

    def test_complete_fifteen_job_orchestration_and_recovery_never_fit(self):
        stage = self.root / "stage"
        stage.mkdir()
        fixture = reference_fixture(stage)
        proof = preparation_fixture(self.root / "proof")
        observed = []
        def execute(request, authority, **kwargs):
            self.assertEqual(kwargs["device"], "cpu")
            self.assertFalse(kwargs["allow_download"])
            self.assertIsNone(kwargs["weights_root"])
            validate_authority(authority, request, self.plan, self.registry)
            observed.append((request["model_id"], request["validation_fold"]))
            return fake_job(kwargs["output_dir"], request, self.rows, self.registry)
        preserved = {}
        def retain(model, fold, directory, job):
            preserved[(model, fold)] = directory
        with patch.object(subject, "verify_cpu_environment", return_value={"device": "cpu"}), \
             patch.object(runner, "execute", side_effect=execute), \
             patch("torch.optim.AdamW", side_effect=AssertionError("optimizer forbidden")), \
             patch("torch.Tensor.backward", side_effect=AssertionError("backward forbidden")), \
             patch("sklearn.linear_model.LogisticRegression.fit", side_effect=AssertionError("fit forbidden")):
            first = subject.run(ROOT, proof, fixture["package"], fixture["reference_path"],
                self.root / "output", approval_statement=subject.APPROVAL_STATEMENT, job_completed=retain)
            second = subject.run(ROOT, proof, fixture["package"], fixture["reference_path"],
                self.root / "recovered", approval_statement=subject.APPROVAL_STATEMENT, completed_jobs=preserved)
        self.assertEqual(len(observed), 15)
        self.assertEqual(len(first["artifact_files_sha256"]), 73)
        self.assertEqual(first["job_count"], 15)
        self.assertEqual(first["development_knees"], len(self.rows))
        self.assertEqual(set(first["model_results"]), set(subject.MODEL_IDS))
        self.assertTrue(all(row["reused_authenticated_complete_job"] for item in second["model_results"].values()
            for row in item["fold_results"]))
        for field in ("reserved_test_accessed", "image_models_trained", "calibration_performed", "threshold_selected"):
            self.assertFalse(first[field])


if __name__ == "__main__":
    unittest.main()
