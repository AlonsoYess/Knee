"""Synthetic campaign contracts only; actual fit/optimizer/backward are blocked."""
from copy import deepcopy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from knee import methodology_campaign as subject
from knee import methodology_release as release
from knee import clinical_scenarios
from knee.aligned_training import runner
from knee.colab_data import sha_file
from test_training_runner import fake_conventional_fit

ROOT = Path(__file__).resolve().parents[1]


def synthetic_cpu_environment(execution_environment):
    """Describe the artificial CPU receipt; never attest to the host runtime."""
    if execution_environment != "local_cpu":
        raise AssertionError("Only the synthetic local-CPU fixture may use this stub")
    return {"execution_environment": "local_cpu", "colab_runtime_executed": False,
        "local_rehearsal": True, "cpu_only_guard_passed": True,
        "environment": {"python": "3.12.14", "numpy": "1.26.4",
            "torch": "2.6.0+cpu", "torchvision": "0.21.0+cpu"}}


def synthetic_preflight(directory):
    """Artificial receipt fixture, never a claim of real Colab execution."""
    from test_methodology_inputs_colab import fixture, MethodologyColabTests
    value = fixture(directory)
    helper = MethodologyColabTests()
    # This test fixture is also constructed by the GPU notebook's CPU-compute
    # tests. Only its invented runtime receipt is simulated; all data checks,
    # artifact hashes, persistence and no-training guards remain exercised.
    with patch.object(subject.inputs, "_cpu_environment", side_effect=synthetic_cpu_environment):
        core = helper.verify(value)
    # Explicit test-only simulation permits exercise of the real-record gate.
    core.update(execution_environment="google_colab_cpu", colab_runtime_executed=True, local_rehearsal=False)
    target = value["output"] / subject.inputs.CORE_FILENAME
    target.unlink()
    subject.inputs._save(target, core)
    tests, runtime = helper.receipts()
    records = Path(directory) / "records"
    records.mkdir()
    subject.inputs.save_result(ROOT, value["output"], records, code_revision="a" * 40,
        cpu_tests=tests, runtime=runtime)
    value["preflight"] = next(records.iterdir())
    value["core"] = core
    value["records"] = records
    return value


def synthetic_job(directory, *, scenario="clinical_main", model="clinical_logistic", fold=0):
    from test_methodology_clinical_integration import MethodologyClinicalIntegrationTests
    fixture = MethodologyClinicalIntegrationTests().fixture(directory)
    _, _, plan, registry = release.load_release(ROOT)
    req = next(request for request in subject.requests("clinical", fixture["package_sha256"],
        fixture["sidecar"]["record_sha256"], plan, registry)
        if request["clinical_scenario"] == scenario and request["model_id"] == model
        and request["validation_fold"] == fold)
    authorization = subject.authority(req, "clinical", subject.APPROVAL_STATEMENTS["clinical"], plan, registry)
    output = fixture["outputs"] / "job"
    with patch.object(runner, "_fit_conventional", side_effect=fake_conventional_fit), \
         patch("sklearn.linear_model.LogisticRegression.fit", side_effect=AssertionError("Real fit forbidden")), \
         patch("xgboost.XGBClassifier.fit", side_effect=AssertionError("Real fit forbidden")), \
         patch("torch.optim.AdamW", side_effect=AssertionError("Optimizer forbidden")), \
         patch("torch.Tensor.backward", side_effect=AssertionError("Backward forbidden")):
        record = runner.execute(req, authorization, repository_root=ROOT, package_dir=fixture["package"],
            reference_path=fixture["reference_path"], weights_root=None, output_dir=output,
            baseline_sidecar_path=fixture["sidecar_path"] if scenario != "clinical_main" else None)
    return fixture, req, record, output


def synthetic_neural_job(output, request, rows, registry, initial_state, device):
    """Fabricated full-epoch record and tiny tensors; never an actual GPU job."""
    import math
    import torch
    from safetensors.torch import save_file
    from knee.aligned_training import engine
    fold = request["validation_fold"]
    train = [r for r in rows if r["cv_fold"] != fold]
    validation = [r for r in rows if r["cv_fold"] == fold]
    predictions = {"identities": [[r["src_subject_id"], r["side"]] for r in validation],
        "targets": [r["PROGRESION_48M"] for r in validation],
        "logits": [1. if r["PROGRESION_48M"] else -1. for r in validation]}
    metrics = engine.binary_metrics(predictions["targets"], predictions["logits"])
    state = clinical_scenarios.fit_scenario_fold(rows, fold)
    output.mkdir()
    runner.save_json(output / "clinical_state.json", state)
    runner.save_json(output / "best_validation_predictions.json", predictions)
    saved = {key: value.clone() + (1 if key.startswith("task_head.") else 0) for key, value in initial_state.items()}
    for name in ("best_model.safetensors", "last_model.safetensors"):
        save_file(saved, str(output / name))
    updates = math.ceil(len(train) / 16)
    history = []
    for epoch in range(1, 10):
        audit = [{"update_index_within_epoch": index, "start_scale": 128., "successful_scale": 128.,
            "scale_after_update": 128., "backoffs": 0, "attempts": 1, "gradient_norm_before_clip": .5}
            for index in range(1, updates + 1)]
        history.append({"epoch": epoch, "phase": "head_only", "training_knees_seen": len(train),
            "optimizer_updates": updates, "weighted_loss_per_knee": .5,
            "learning_rate_factor": engine.learning_rate_factor(epoch), "validation": metrics,
            "selected_best": epoch == 1, "amp_overflow_replays": 0, "amp_scale_audit": audit})
    train_ids = sorted([[r["src_subject_id"], r["side"]] for r in train])
    participants = sorted({identity[0] for identity in train_ids})
    positive = sum(r["PROGRESION_48M"] for r in train)
    training = {"version": engine.VERSION, "status": "training_fold_complete", "scope": "development_only",
        "smoke": False, "model_id": "image_densenet121", "stage": "stage_1", "seed": 2026,
        "training_executed": True, "optimizer_created": True, "eligible_for_model_selection": True,
        "reserved_test_accessed": False, "restored_best_checkpoint": True, "resume_supported": False,
        "optimizer_state_saved": False, "scheduler_horizon_epochs": 60, "effective_batch_size": 16,
        "micro_batch_size": request["image_micro_batch_size"], "device": str(torch.device(device)),
        "precision": "float16_autocast_grad_scaler", "gpu_resource_profile_recorded": True,
        "provenance": {"validation_fold": fold, "development_package_sha256": request["development_package_sha256"],
            "train_identity_sha256": subject.digest(train_ids), "request_sha256": subject.digest(request),
            "training_knees": len(train), "validation_knees": len(validation), "trained_participants": participants},
        "positive_weight": (len(train) - positive) / positive, "history": history, "completed_epochs": 9,
        "best_epoch": 1, "best_validation_ap": metrics["average_precision_score"], "optimizer_updates": updates * 9,
        "checkpoint_sha256": {name: sha_file(output / name) for name in ("best_model.safetensors", "last_model.safetensors")},
        "numerical_repair": {"version": "same_effective_group_amp_replay_v1.0", "maximum_backoffs_per_effective_group": 16,
            "same_effective_group_replayed": True, "nonfinite_gradients_never_clipped": True},
        "amp_scale_audit": [{"epoch": item["epoch"], **entry} for item in history for entry in item["amp_scale_audit"]],
        "amp_overflow_replays": 0}
    runner.save_json(output / "training_record.json", training)
    record = {"version": runner.VERSION, "scope": "development_only", "request": request,
        "request_sha256": subject.digest(request), "plan_sha256": release.PINS[release.PLAN_PATH],
        "eligible_for_model_selection": True, "reserved_test_accessed": False,
        "weights_provenance": {"fit_scope": "development_training_fold", "validation_fold": fold,
            "train_identity_sha256": state["train_identity_sha256"], "trained_participants": participants,
            "reserved_test_accessed": False}, "metrics": metrics,
        "eligibility_gates": {key: True for key in registry["eligibility_gates"]}, "selection_evidence": None,
        "engine_status": "training_fold_complete", "training_executed": True,
        "files_sha256": {path.name: sha_file(path) for path in output.iterdir()}}
    record["record_sha256"] = subject.digest(record)
    runner.save_json(output / runner.JOB_FILENAME, record)
    return record


class MethodologyCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        _, _, cls.plan, cls.registry = release.load_release(ROOT)

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.threads)

    def test_synthetic_preflight_supports_hidden_cuda_build_and_restores_real_guard(self):
        import torch
        original_guard = subject.inputs._cpu_environment
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "", "COLAB_GPU": "0"}), \
             patch.object(torch.cuda, "is_available", return_value=False), \
             patch.object(torch.version, "cuda", "12.4"):
            value = synthetic_preflight(directory)
            self.assertEqual(value["core"]["development_knees"], 30)
            self.assertIs(subject.inputs._cpu_environment, original_guard)
            with self.assertRaisesRegex(ValueError, "cpu_only_pytorch_build_required"):
                original_guard("local_cpu")

    def test_real_cpu_preflight_still_rejects_a_hidden_cuda_build(self):
        import torch
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "", "COLAB_GPU": "0"}), \
             patch.object(torch.cuda, "is_available", return_value=False), \
             patch.object(torch.version, "cuda", "12.4"):
            with self.assertRaisesRegex(ValueError, "cpu_only_pytorch_build_required"):
                subject.inputs._cpu_environment("local_cpu")

    def test_synthetic_runtime_stub_cannot_claim_real_colab_execution(self):
        with self.assertRaisesRegex(AssertionError, "synthetic local-CPU"):
            synthetic_cpu_environment("google_colab_cpu")

    def test_clinical_campaign_contains_every_scenario_algorithm_fold_once(self):
        requests = subject.requests("clinical", "a" * 64, "b" * 64, self.plan, self.registry)
        self.assertEqual(len(requests), 45)
        self.assertEqual(len({subject.job_key(req) for req in requests}), 45)
        self.assertEqual({(req["clinical_scenario"], req["model_id"], req["validation_fold"]) for req in requests},
            {(scenario, model, fold) for scenario in clinical_scenarios.SCENARIOS
                for model in subject.CLINICAL_IDS for fold in subject.FOLDS})
        for req in requests:
            self.assertEqual(req["version"], release.REQUEST_VERSION)
            self.assertEqual(req["augmentation_policy"], "identity")
            self.assertEqual(req["baseline_sidecar_sha256"], None if req["clinical_scenario"] == "clinical_main" else "b" * 64)
            self.assertNotIn("training_authorized", req)

    def test_densenet_requests_are_frozen_and_use_approved_gamma(self):
        requests = subject.requests("densenet_frozen", "a" * 64, "b" * 64, self.plan, self.registry)
        self.assertEqual(len(requests), 5)
        for fold, req in enumerate(requests):
            self.assertEqual(req["validation_fold"], fold)
            self.assertEqual(req["model_id"], "image_densenet121")
            self.assertEqual(req["image_encoder_id"], "image_densenet121")
            self.assertEqual(req["clinical_scenario"], "clinical_main")
            self.assertEqual(req["stage"], "stage_1")
            self.assertEqual(req["augmentation_policy"], "training_intensity_gamma_v1.0")
            self.assertIsNone(req["baseline_sidecar_sha256"])

    def test_unknown_campaign_hashes_and_model_changes_are_rejected(self):
        for campaign, package, sidecar in (("all_models", "a" * 64, "b" * 64),
            ("clinical", "invalid", "b" * 64), ("clinical", "a" * 64, "invalid")):
            with self.subTest(campaign=campaign), self.assertRaises(ValueError):
                subject.requests(campaign, package, sidecar, self.plan, self.registry)

    def test_authority_requires_exact_confirmation_and_registered_request(self):
        req = subject.requests("clinical", "a" * 64, "b" * 64, self.plan, self.registry)[0]
        with self.assertRaises(ValueError):
            subject.authority(req, "clinical", "yes", self.plan, self.registry)
        changed = {**req, "seed": 2027}
        with self.assertRaises(ValueError):
            subject.authority(changed, "clinical", subject.APPROVAL_STATEMENTS["clinical"], self.plan, self.registry)
        auth = subject.authority(req, "clinical", subject.APPROVAL_STATEMENTS["clinical"], self.plan, self.registry)
        self.assertFalse(auth["reserved_test_access_authorized"])
        self.assertFalse(auth["full_encoder_training_authorized"])
        self.assertFalse(any(auth["eligibility_gates"].values()))

    def test_densenet_authority_requires_verified_prior_gpu_evidence(self):
        req = subject.requests("densenet_frozen", "a" * 64, "b" * 64, self.plan, self.registry)[0]
        with self.assertRaises(ValueError):
            subject.authority(req, "densenet_frozen", subject.APPROVAL_STATEMENTS["densenet_frozen"], self.plan, self.registry)
        auth = subject.authority(req, "densenet_frozen", subject.APPROVAL_STATEMENTS["densenet_frozen"],
            self.plan, self.registry, image_eligibility_verified=True)
        self.assertTrue(all(auth["eligibility_gates"].values()))
        self.assertFalse(auth["full_encoder_training_authorized"])

    def test_missing_confirmation_fails_before_reading_any_data_or_fitting(self):
        with patch.object(subject, "validate_preparation", side_effect=AssertionError("Must reject before reads")), \
             patch.object(runner, "execute", side_effect=AssertionError("Training forbidden")), \
             self.assertRaisesRegex(ValueError, "explicit_current_methodology_campaign_approval_required"):
            subject.run("missing", "missing", "missing", "missing", "missing", campaign="clinical", approval_statement="")

    def test_extra_clinical_scenario_job_is_authenticated_against_exact_fold(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture, req, record, output = synthetic_job(directory, scenario="clinical_plus_baseline_surgery_and_womac")
            result, engine = subject.validate_completed_job(output, req, self.plan, self.registry,
                fixture["rows"], fixture["sidecar"])
            self.assertEqual(result, record)
            self.assertIsNone(engine)
            self.assertEqual(runner.read_json(output / "clinical_state.json")["feature_order"],
                ["AGEYEARS", "SEX", "BMI", "CIRUGIA_PREVIA", "WOMAC_TOTAL"])
            wrong = {**req, "clinical_scenario": "clinical_plus_baseline_kl"}
            with self.assertRaises(ValueError):
                subject.validate_completed_job(output, wrong, self.plan, self.registry, fixture["rows"], fixture["sidecar"])

    def test_mutated_held_out_target_is_rejected_even_when_hashes_resealed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture, req, record, output = synthetic_job(directory)
            predictions = runner.read_json(output / "best_validation_predictions.json")
            predictions["targets"][0] = 1 - predictions["targets"][0]
            path = output / "best_validation_predictions.json"
            path.unlink()
            runner.save_json(path, predictions)
            record["files_sha256"][path.name] = sha_file(path)
            from knee.aligned_training.engine import binary_metrics
            record["metrics"] = binary_metrics(predictions["targets"], predictions["logits"])
            record["record_sha256"] = subject.digest({k: v for k, v in record.items() if k != "record_sha256"})
            (output / runner.JOB_FILENAME).unlink()
            runner.save_json(output / runner.JOB_FILENAME, record)
            with self.assertRaisesRegex(ValueError, "exact_methodology_held_out_predictions_required"):
                subject.validate_completed_job(output, req, self.plan, self.registry, fixture["rows"], fixture["sidecar"])

    def test_historical_recipe_job_is_rejected_for_current_campaign(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture, req, record, output = synthetic_job(directory)
            record["plan_sha256"] = "f" * 64
            record["record_sha256"] = subject.digest({k: v for k, v in record.items() if k != "record_sha256"})
            (output / runner.JOB_FILENAME).unlink()
            runner.save_json(output / runner.JOB_FILENAME, record)
            with self.assertRaises(ValueError):
                subject.validate_completed_job(output, req, self.plan, self.registry, fixture["rows"], fixture["sidecar"])

    def test_oof_cannot_aggregate_incomplete_folds(self):
        with self.assertRaisesRegex(ValueError, "all_five_current_methodology_folds_required"):
            subject.aggregate({}, [])

    def test_saved_colab38_full_inventory_is_authenticated_and_detects_change(self):
        with tempfile.TemporaryDirectory() as directory:
            value = synthetic_preflight(directory)
            summary, core, sidecar = subject.validate_preparation(value["preflight"], ROOT)
            self.assertEqual(core["development_knees"], 30)
            self.assertEqual(summary["prepared_clinical_requests"], 45)
            self.assertEqual(len(sidecar["rows"]), 30)
            (value["preflight"] / "baseline_sidecar.json").write_bytes(b"{}")
            with self.assertRaisesRegex(ValueError, "complete_unchanged_real_colab38_artifacts_required"):
                subject.validate_preparation(value["preflight"])

    def test_local_rehearsal_receipt_cannot_authorize_real_campaign(self):
        with tempfile.TemporaryDirectory() as directory:
            value = synthetic_preflight(directory)
            path = value["preflight"] / subject.inputs.SUMMARY_FILENAME
            summary = runner.read_json(path)
            summary.update(local_rehearsal=True, colab_runtime_executed=False, execution_environment="local_cpu")
            path.unlink()
            runner.save_json(path, summary)
            with self.assertRaisesRegex(ValueError, "saved_methodology_preflight_summary_required"):
                subject.validate_preparation(value["preflight"])

    def test_all_45_jobs_route_through_v1_2_and_recover_without_any_new_fit(self):
        from test_methodology_clinical_integration import MethodologyClinicalIntegrationTests
        from knee.aligned_training import engine
        with tempfile.TemporaryDirectory() as directory:
            value = synthetic_preflight(directory)
            outputs = Path(directory) / "outputs"
            outputs.mkdir()
            output = outputs / "campaign"
            callbacks = []
            with patch("knee.clinical_fivefold.verify_cpu_environment", return_value={"synthetic_no_fit": True}), \
                 patch.object(runner, "_fit_conventional", side_effect=fake_conventional_fit) as fit, \
                 patch.object(engine, "train_neural_fold", side_effect=MethodologyClinicalIntegrationTests.fake_neural_fit) as neural, \
                 patch.object(subject, "_validate_neural", return_value={"synthetic_mock": True}), \
                 patch("sklearn.linear_model.LogisticRegression.fit", side_effect=AssertionError("Real estimator forbidden")), \
                 patch("xgboost.XGBClassifier.fit", side_effect=AssertionError("Real estimator forbidden")), \
                 patch("torch.optim.AdamW", side_effect=AssertionError("Real optimizer forbidden")), \
                 patch("torch.Tensor.backward", side_effect=AssertionError("Real backward forbidden")):
                result = subject.run(ROOT, value["preflight"], value["package"], value["reference_path"], output,
                    campaign="clinical", approval_statement=subject.APPROVAL_STATEMENTS["clinical"],
                    job_completed=lambda key, path, record: callbacks.append((key, path, record)))
            self.assertEqual(fit.call_count, 30)
            self.assertEqual(neural.call_count, 15)
            self.assertEqual(len(callbacks), 45)
            self.assertEqual(result["job_count"], 45)
            self.assertEqual(len(result["model_results"]), 9)
            self.assertTrue(all(item["out_of_fold_predictions"] == 30 for item in result["model_results"].values()))
            previous = {key: path for key, path, _ in callbacks}
            with patch("knee.clinical_fivefold.verify_cpu_environment", return_value={"synthetic_no_fit": True}), \
                 patch.object(runner, "execute", side_effect=AssertionError("Recovered jobs must not fit")), \
                 patch.object(subject, "_validate_neural", return_value={"synthetic_mock": True}):
                recovered = subject.run(ROOT, value["preflight"], value["package"], value["reference_path"], outputs / "recovered",
                    campaign="clinical", approval_statement=subject.APPROVAL_STATEMENTS["clinical"], completed_jobs=previous)
            self.assertEqual(result["model_results"], recovered["model_results"])
            self.assertTrue(all(row["reused_authenticated_complete_job"] for row in recovered["job_results"]))

    def test_five_image_jobs_use_explicit_cuda_zero_and_preserve_frozen_encoder(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            value = synthetic_preflight(directory)
            output = Path(directory) / "campaign"
            weights = Path(directory) / "weights"
            weights.mkdir()
            model = torch.nn.Module()
            model.add_module("encoder", torch.nn.Linear(2, 2))
            model.add_module("task_head", torch.nn.Linear(2, 1))
            initial = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
            def artificial_execute(request, authorization, **kwargs):
                self.assertEqual(kwargs["device"], "cuda:0")
                self.assertEqual(request["augmentation_policy"], "training_intensity_gamma_v1.0")
                return synthetic_neural_job(kwargs["output_dir"], request, value["rows"], self.registry,
                    initial, kwargs["device"])
            with patch("knee.training_smoke.verify_gpu_environment", return_value={"synthetic_no_gpu": True}), \
                 patch("knee.final_models_preflight.checkpoint_files", return_value=[]), \
                 patch("knee.aligned_training.models.build_model", return_value=(model, {})), \
                 patch.object(runner, "execute", side_effect=artificial_execute) as execute, \
                 patch("torch.optim.AdamW", side_effect=AssertionError("Real optimizer forbidden")), \
                 patch("torch.Tensor.backward", side_effect=AssertionError("Real backward forbidden")):
                result = subject.run(ROOT, value["preflight"], value["package"], value["reference_path"], output,
                    campaign="densenet_frozen", approval_statement=subject.APPROVAL_STATEMENTS["densenet_frozen"],
                    weights_root=weights, gpu_evidence={"image_eligibility_verified": True,
                        "development_package_sha256": value["package_sha256"],
                        "model_input_reference_sha256": sha_file(value["reference_path"])})
            self.assertEqual(execute.call_count, 5)
            self.assertEqual(result["job_count"], 5)
            self.assertFalse(result["encoder_unfreezing_executed"])
            request = subject.requests("densenet_frozen", value["package_sha256"], "b" * 64, self.plan, self.registry)[0]
            jobdir = output / subject.job_key(request)
            job, engine = subject.validate_completed_job(jobdir, request, self.plan, self.registry, value["rows"], {}, initial_state=initial)
            bad = {key: tensor.clone() + (1 if key.startswith("encoder.") else 0) for key, tensor in initial.items()}
            with self.assertRaisesRegex(ValueError, "frozen_encoder_must_remain_exact"):
                subject.validate_completed_job(jobdir, request, self.plan, self.registry, value["rows"], {}, initial_state=bad)
