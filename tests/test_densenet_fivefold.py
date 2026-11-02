"""Artificial five-fold records and mocked execution; no predictor is fitted."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from knee import densenet_fivefold as subject
from knee import training_runner as runner
from knee.training_authority import digest, validate_authority
from knee.training_engine import binary_metrics, learning_rate_factor, VERSION as ENGINE_VERSION
from knee.final_model_release import load_release, PINS
from knee.colab_data import sha_file
from knee.clinical_fold_inputs import fit_clinical_fold
from test_training_data import synthetic_inputs, reference_fixture

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def initial_state():
    return {"encoder.weight": torch.tensor([1., 2.]), "encoder.running_mean": torch.tensor([0., 0.]),
        "task_head.weight": torch.tensor([[.5, -.5]]), "task_head.bias": torch.tensor([0.])}


def fake_fold(directory, request, rows, registry, *, change=None):
    """Nine wholly synthetic epoch records: strict AP tie early-stops at nine."""
    root = Path(directory)
    root.mkdir()
    training = [row for row in rows if row["cv_fold"] != request["validation_fold"]]
    validation = [row for row in rows if row["cv_fold"] == request["validation_fold"]]
    ids = sorted([[row["src_subject_id"], row["side"]] for row in training])
    participants = sorted({identity[0] for identity in ids})
    predictions = {"targets": [row["PROGRESION_48M"] for row in validation],
        "logits": [2. if row["PROGRESION_48M"] else -2. for row in validation],
        "identities": [[row["src_subject_id"], row["side"]] for row in validation]}
    metrics = binary_metrics(predictions["targets"], predictions["logits"])
    write_json(root / "best_validation_predictions.json", predictions)
    write_json(root / "clinical_state.json", fit_clinical_fold(rows, request["validation_fold"]))
    state = initial_state()
    state["task_head.bias"] = torch.tensor([.01])
    for name in ("best_model.safetensors", "last_model.safetensors"):
        save_file(state, str(root / name))
    updates = math.ceil(len(training) / 16)
    history = []
    for epoch in range(1, 10):
        audit = [{"update_index_within_epoch": index, "start_scale": 65536.,
            "successful_scale": 65536., "scale_after_update": 65536., "backoffs": 0,
            "attempts": 1, "gradient_norm_before_clip": 1.} for index in range(1, updates + 1)]
        history.append({"epoch": epoch, "phase": "head_only", "training_knees_seen": len(training),
            "optimizer_updates": updates, "learning_rate_factor": learning_rate_factor(epoch),
            "weighted_loss_per_knee": .4, "validation": metrics, "selected_best": epoch == 1,
            "amp_scale_audit": audit, "amp_overflow_replays": 0})
    positives = sum(row["PROGRESION_48M"] for row in training)
    engine = {"version": ENGINE_VERSION, "model_id": subject.MODEL_ID, "stage": "stage_1", "seed": 2026,
        "smoke": False, "scope": "development_only", "status": "training_fold_complete",
        "eligible_for_model_selection": True, "reserved_test_accessed": False,
        "optimizer_created": True, "training_executed": True, "micro_batch_size": 2,
        "effective_batch_size": 16, "device": "cuda:0", "precision": "float16_autocast_grad_scaler",
        "scheduler_horizon_epochs": 60, "gpu_resource_profile_recorded": True,
        "restored_best_checkpoint": True, "resume_supported": False, "optimizer_state_saved": False,
        "provenance": {"validation_fold": request["validation_fold"],
            "development_package_sha256": request["development_package_sha256"],
            "train_identity_sha256": digest(ids), "request_sha256": digest(request),
            "training_knees": len(training), "validation_knees": len(validation),
            "trained_participants": participants},
        "positive_weight": (len(training) - positives) / positives,
        "history": history, "completed_epochs": 9, "optimizer_updates": 9 * updates,
        "best_epoch": 1, "best_validation_ap": metrics["average_precision_score"],
        "checkpoint_sha256": {name: sha_file(root / name) for name in
            ("best_model.safetensors", "last_model.safetensors")},
        "numerical_repair": {"version": subject.smoke.AMP_REPAIR_VERSION,
            "maximum_backoffs_per_effective_group": 16, "same_effective_group_replayed": True,
            "nonfinite_gradients_never_clipped": True},
        "amp_scale_audit": [{"epoch": epoch["epoch"], **audit}
            for epoch in history for audit in epoch["amp_scale_audit"]], "amp_overflow_replays": 0,
        "resource_profile": {"device_type": "cuda", "cuda_peak_measured": True,
            "torch_version": "2.6.0+cu124", "cuda_version": "12.4", "device_name": "SYNTHETIC GPU",
            "total_memory_bytes": 10000, "peak_memory_allocated_bytes": 500,
            "peak_memory_reserved_bytes": 1000, "elapsed_seconds": 1.}}
    if change:
        change(engine)
    write_json(root / "training_record.json", engine)
    record = {"version": runner.VERSION, "scope": "development_only", "request": request,
        "request_sha256": digest(request), "plan_sha256": PINS["configs/training_plan_v1_1.json"],
        "eligible_for_model_selection": True, "reserved_test_accessed": False,
        "weights_provenance": {"fit_scope": "development_training_fold",
            "validation_fold": request["validation_fold"], "train_identity_sha256": digest(ids),
            "trained_participants": participants, "reserved_test_accessed": False},
        "metrics": metrics, "eligibility_gates": {name: True for name in registry["eligibility_gates"]},
        "selection_evidence": None, "engine_status": "training_fold_complete", "training_executed": True,
        "files_sha256": {name: sha_file(root / name) for name in subject.smoke.JOB_FILES}}
    record["record_sha256"] = digest(record)
    write_json(root / runner.JOB_FILENAME, record)
    return record


def reseal(directory):
    root = Path(directory)
    record = json.loads((root / runner.JOB_FILENAME).read_text())
    record["files_sha256"] = {name: sha_file(root / name) for name in subject.smoke.JOB_FILES}
    record["record_sha256"] = digest({key: value for key, value in record.items() if key != "record_sha256"})
    write_json(root / runner.JOB_FILENAME, record)


class FixedCampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="fivefold36_synthetic_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        _, self.catalog, self.plan, self.registry = load_release(ROOT)
        self.rows, _ = synthetic_inputs()
        self.request = subject.make_request("a" * 64, 0)

    def validate(self, directory):
        return subject.validate_completed_fold(directory, self.request, self.plan,
            self.registry, initial_state(), self.rows)

    def test_only_five_registered_requests_and_no_other_operations(self):
        for fold in range(5):
            request = subject.make_request("a" * 64, fold)
            self.assertEqual(request["operation"], "development_fold")
            self.assertEqual(request["stage"], "stage_1")
            self.assertEqual(request["model_id"], "image_densenet121")
            self.assertEqual(request["seed"], 2026)
            self.assertEqual(request["image_micro_batch_size"], 2)
        for fold in (True, -1, 5, "0"):
            with self.subTest(fold=fold), self.assertRaises(ValueError):
                subject.make_request("a" * 64, fold)
        with self.assertRaises(ValueError):
            subject.make_request("INVALID", 0)

    def test_authority_requires_same_completed_smoke_not_unfreezing(self):
        proof = {"model_id": subject.MODEL_ID, "training_smoke_test_passed": True,
            "gpu_resource_profile_recorded": True}
        authority = subject._authority(self.request, self.registry, self.catalog,
            subject.APPROVAL_STATEMENT, proof)
        permit = validate_authority(authority, self.request, self.plan, self.registry)
        self.assertFalse(permit.full_encoder_training_authorized)
        self.assertTrue(all(authority["eligibility_gates"].values()))
        for change in ({"model_id": "image_vit_b16"}, {"training_smoke_test_passed": False},
                {"gpu_resource_profile_recorded": False}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                subject._authority(self.request, self.registry, self.catalog,
                    subject.APPROVAL_STATEMENT, {**proof, **change})

    def test_complete_frozen_fold_is_authenticated_without_fitting(self):
        root = self.root / "fold"
        record = fake_fold(root, self.request, self.rows, self.registry)
        actual, engine, frozen = self.validate(root)
        self.assertEqual(actual, record)
        self.assertEqual(engine["completed_epochs"], 9)
        self.assertTrue(frozen["frozen_encoder_state_unchanged"])

    def test_partial_smoke_cpu_wrong_recipe_or_unfreezing_cannot_be_reused(self):
        changes = (lambda row: row.update(smoke=True), lambda row: row.update(device="cpu"),
            lambda row: row.update(stage="stage_2"), lambda row: row.update(completed_epochs=8),
            lambda row: row.update(optimizer_updates=1), lambda row: row.update(best_epoch=9),
            lambda row: row.update(positive_weight=99.), lambda row: row.update(resume_supported=True),
            lambda row: row["history"][0].update(phase="full"),
            lambda row: row["history"][0].update(training_knees_seen=2),
            lambda row: row["history"][0].update(weighted_loss_per_knee=-1),
            lambda row: row["history"][0].update(learning_rate_factor=1.),
            lambda row: row["history"][-1].update(selected_best=True),
            lambda row: row["resource_profile"].update(peak_memory_allocated_bytes=20000))
        for index, change in enumerate(changes):
            directory = self.root / str(index)
            fake_fold(directory, self.request, self.rows, self.registry, change=change)
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.validate(directory)

    def test_premature_stop_before_patience_is_rejected(self):
        root = self.root / "short"
        def partial(engine):
            engine["history"] = engine["history"][:8]
            engine["completed_epochs"] = 8
            engine["optimizer_updates"] = sum(row["optimizer_updates"] for row in engine["history"])
        fake_fold(root, self.request, self.rows, self.registry, change=partial)
        with self.assertRaisesRegex(ValueError, "early_stop"):
            self.validate(root)

    def test_sixty_complete_improving_epochs_allowed_without_training_execution(self):
        root = self.root / "sixty"
        def sixty(engine):
            first = deepcopy(engine["history"][0])
            final_metrics = deepcopy(first["validation"])
            engine["history"] = []
            for epoch in range(1, 61):
                row = deepcopy(first)
                row.update(epoch=epoch, learning_rate_factor=learning_rate_factor(epoch), selected_best=True)
                row["validation"] = {**final_metrics, "average_precision_score": epoch / 60}
                engine["history"].append(row)
            engine.update(completed_epochs=60, best_epoch=60, best_validation_ap=1.,
                optimizer_updates=60 * first["optimizer_updates"])
            engine["amp_scale_audit"] = [{"epoch": epoch["epoch"], **audit}
                for epoch in engine["history"] for audit in epoch["amp_scale_audit"]]
        fake_fold(root, self.request, self.rows, self.registry, change=sixty)
        _, engine, _ = self.validate(root)
        self.assertEqual(engine["completed_epochs"], 60)

    def test_completed_fold_rejects_unlisted_artifact(self):
        root = self.root / "unexpected"
        fake_fold(root, self.request, self.rows, self.registry)
        (root / "unexpected.txt").write_text("synthetic extra")
        with self.assertRaisesRegex(ValueError, "exact_frozen"):
            self.validate(root)

    def test_encoder_or_bn_change_and_unchanged_task_head_fail(self):
        for index, key in enumerate(("encoder.weight", "encoder.running_mean", "task_head.bias")):
            root = self.root / str(index)
            fake_fold(root, self.request, self.rows, self.registry)
            state = initial_state()
            if key.startswith("encoder"):
                state[key] = state[key] + .01
                state["task_head.bias"] = state["task_head.bias"] + .01
            save_file(state, str(root / "best_model.safetensors"))
            reseal(root)
            engine = json.loads((root / "training_record.json").read_text())
            engine["checkpoint_sha256"]["best_model.safetensors"] = sha_file(root / "best_model.safetensors")
            write_json(root / "training_record.json", engine)
            reseal(root)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(root)

    def test_training_identity_or_clinical_state_tampering_rejected_even_resealed(self):
        root = self.root / "clinical"
        fake_fold(root, self.request, self.rows, self.registry)
        state = json.loads((root / "clinical_state.json").read_text())
        state["BMI_median"] += .1
        write_json(root / "clinical_state.json", state)
        reseal(root)
        with self.assertRaisesRegex(ValueError, "clinical_state"):
            self.validate(root)
        root2 = self.root / "identity"
        fake_fold(root2, self.request, self.rows, self.registry)
        job = json.loads((root2 / runner.JOB_FILENAME).read_text())
        job["weights_provenance"]["trained_participants"].append("NOT_TRAINED")
        write_json(root2 / runner.JOB_FILENAME, job)
        reseal(root2)
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.validate(root2)

    def test_all_development_rows_are_covered_once_and_pooled_is_descriptive_only(self):
        jobs = {}
        for fold in range(5):
            root = self.root / str(fold)
            request = subject.make_request("a" * 64, fold)
            jobs[fold] = (root, fake_fold(root, request, self.rows, self.registry))
        means, pooled, rows = subject.aggregate_predictions(jobs, self.rows)
        self.assertEqual(len(rows), len(self.rows))
        self.assertEqual(len({(row["src_subject_id"], row["side"]) for row in rows}), len(self.rows))
        self.assertEqual(means["average_precision_score"], 1.)
        self.assertEqual(pooled["average_precision_score"], 1.)
        jobs.pop(4)
        with self.assertRaisesRegex(ValueError, "all_five"):
            subject.aggregate_predictions(jobs, self.rows)

    def test_oof_wrong_fold_or_targets_are_rejected(self):
        jobs = {}
        for fold in range(5):
            root = self.root / str(fold)
            req = subject.make_request("a" * 64, fold)
            jobs[fold] = (root, fake_fold(root, req, self.rows, self.registry))
        file = jobs[1][0] / "best_validation_predictions.json"
        predictions = json.loads(file.read_text())
        predictions["identities"][0] = list(jobs[0][1]["weights_provenance"]["trained_participants"][:1]) + ["LEFT"]
        write_json(file, predictions)
        with self.assertRaises(ValueError):
            subject.aggregate_predictions(jobs, self.rows)

    def test_no_approval_fails_before_any_gpu_or_data_access(self):
        with patch.object(subject.smoke, "verify_gpu_environment") as runtime:
            with self.assertRaisesRegex(ValueError, "approval"):
                subject.run("missing", "missing", "missing", "missing", "missing", "missing",
                    approval_statement="")
            runtime.assert_not_called()

    def test_large_engine_audit_allowed_but_duplicate_and_nonfinite_json_forbidden(self):
        path = self.root / "big.json"
        path.write_bytes(b'{"large":"' + b"a" * (4 * 1024**2) + b'"}')
        self.assertGreater(len(subject.read_engine_record(path)["large"]), 4 * 1024**2 - 1)
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}'):
            path.write_bytes(raw)
            with self.assertRaises(ValueError):
                subject.read_engine_record(path)
        with patch.object(subject, "MAX_ENGINE_RECORD_BYTES", 2), self.assertRaisesRegex(ValueError, "bounded"):
            subject.read_engine_record(path)

    def test_amp_scale_growth_is_legal_for_longer_runs_without_skipping_groups(self):
        directory = self.root / "amp"
        fake_fold(directory, self.request, self.rows, self.registry)
        engine = subject.read_engine_record(directory / "training_record.json")
        previous = 65536.
        for epoch_index, epoch in enumerate(engine["history"]):
            for index, row in enumerate(epoch["amp_scale_audit"]):
                row.update(start_scale=previous, successful_scale=previous,
                    scale_after_update=previous * (2 if epoch_index == 0 and index == 0 else 1))
                previous = row["scale_after_update"]
        engine["amp_scale_audit"] = [{"epoch": epoch["epoch"], **audit}
            for epoch in engine["history"] for audit in epoch["amp_scale_audit"]]
        subject._validate_amp(engine)
        engine["amp_scale_audit"][0]["attempts"] = 99
        with self.assertRaises(ValueError):
            subject._validate_amp(engine)


class MockedCampaignExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="fivefold36_mock_execute_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        inputs = self.root / "inputs"
        inputs.mkdir()
        fixture = reference_fixture(inputs)
        self.package, self.reference = fixture["package"], fixture["reference_path"]
        self.rows, self.package_sha = fixture["rows"], fixture["package_sha256"]
        self.proof = self.root / "proof"
        self.proof.mkdir()
        for name in ("training_smoke.json", "colab_training_smoke_check.json"):
            write_json(self.proof / name, {"synthetic_only": True})
        self.weights = self.root / "weights"
        self.weights.mkdir()
        self.checkpoint = self.weights / "synthetic.pth"
        self.checkpoint.write_bytes(b"SYNTHETIC ONLY; NO PRETRAINED MODEL LOADED")
        _, _, _, self.registry = load_release(ROOT)
        self.proof_record = {"development_package_sha256": self.package_sha,
            "model_input_reference_sha256": sha_file(self.reference), "model_id": subject.MODEL_ID,
            "training_smoke_test_passed": True, "gpu_resource_profile_recorded": True,
            "eligibility_evidence": {"synthetic_fixture": True}}

    def execute(self, directory, **kwargs):
        def fake_execute(request, authority, **arguments):
            self.assertEqual(arguments["device"], "cuda:0")
            self.assertFalse(arguments["allow_download"])
            self.assertFalse(authority["full_encoder_training_authorized"])
            return fake_fold(arguments["output_dir"], request, self.rows, self.registry)
        with patch.object(subject.smoke, "verify_gpu_environment", return_value={"synthetic_runtime": True}), \
                patch("knee.training_smoke_evidence.validate_smoke", return_value=self.proof_record), \
                patch.object(subject, "sources", return_value={"synthetic_source": "a" * 64}), \
                patch("knee.final_models_preflight.checkpoint_files", return_value=[self.checkpoint]), \
                patch("knee.training_models.build_model", return_value=(
                    SimpleNamespace(state_dict=initial_state), {"synthetic": True})), \
                patch("knee.training_runner.execute", side_effect=fake_execute) as executed:
            result = subject.run(ROOT, self.proof, self.package, self.reference, self.weights, directory,
                approval_statement=subject.APPROVAL_STATEMENT, **kwargs)
            return result, executed.call_count

    def test_five_fold_workflow_only_calls_mocked_executor_and_saves_private_oof(self):
        calls = []
        output = self.root / "complete"
        result, count = self.execute(output, fold_completed=lambda fold, root, job: calls.append(fold))
        self.assertEqual(count, 5)
        self.assertEqual(calls, list(range(5)))
        self.assertEqual(result["folds_completed"], list(range(5)))
        self.assertEqual(result["out_of_fold_predictions"], len(self.rows))
        self.assertEqual(result["development_participants"], 15)
        self.assertEqual(result["development_knees"], 30)
        self.assertFalse(result["reserved_test_accessed"])
        self.assertFalse(result["encoder_unfreezing_executed"])
        self.assertEqual(json.loads((output / subject.FILENAME).read_text()), result)

    def test_verified_complete_fold_reuse_does_not_retrain_that_fold(self):
        prior = self.root / "prior0"
        fake_fold(prior, subject.make_request(self.package_sha, 0), self.rows, self.registry)
        result, count = self.execute(self.root / "resumed", completed_folds={0: prior})
        self.assertEqual(count, 4)
        self.assertTrue(result["fold_results"][0]["reused_authenticated_complete_fold"])

    def test_partial_fold_is_preserved_but_never_promoted_or_retrained_silently(self):
        prior = self.root / "partial"
        prior.mkdir()
        write_json(prior / "training_record.json", {"not_complete": True})
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.execute(self.root / "failed", completed_folds={0: prior})
        self.assertTrue((prior / "training_record.json").is_file())

    def test_durable_callback_failure_stops_before_next_fold(self):
        completed = []
        def stop(fold, root, job):
            completed.append(fold)
            raise ValueError("synthetic_durable_copy_failure")
        with self.assertRaisesRegex(ValueError, "durable_copy"):
            self.execute(self.root / "stopped", fold_completed=stop)
        self.assertEqual(completed, [0])
        self.assertFalse((self.root / "stopped" / "fold_1").exists())

    def test_wrong_reference_or_package_does_not_invoke_executor(self):
        self.proof_record["development_package_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "same_verified_smoke"):
            self.execute(self.root / "wrong")


if __name__ == "__main__":
    unittest.main()
