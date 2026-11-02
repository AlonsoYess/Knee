"""Five-encoder campaign wiring with artificial images and no fitting/network."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

import torch

from knee import methodology_campaign as subject
from knee import methodology_release as release
from knee.aligned_training import runner, engine, models
from knee.aligned_training.authority import require_permit
from knee.colab_data import sha_file
from test_methodology_campaign import synthetic_preflight, synthetic_neural_job

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = "remaining_encoders_frozen"


def synthetic_smoke_receipt(model_id, package_sha256):
    """Invented receipt metadata for artificial campaign tests, not GPU evidence."""
    return {"model_id": model_id, "training_smoke_test_passed": True, "gpu_resource_profile_recorded": True,
        "optimizer_updates": 2, "validation_fold": 0, "seed": 2026, "stage": "stage_1",
        "development_package_sha256": package_sha256, "training_plan_sha256": release.PINS[release.PLAN_PATH],
        "frozen_encoder_state_unchanged": True, "task_head_update_verified": True,
        "eligible_for_model_selection": False, "reserved_test_accessed": False,
        "smoke_request_sha256": "1" * 64, "smoke_job_record_sha256": "2" * 64,
        "smoke_training_record_sha256": "3" * 64}


def synthetic_evidence(value, catalog):
    """Artificial eligibility for artificial rows; not real GPU/Colab evidence."""
    records = {}
    for model_id in subject.REMAINING_IMAGE_IDS:
        spec = catalog["models"][model_id]
        records[model_id] = {"model_id": model_id, "image_eligibility_verified": True,
            "synthetic_gpu_forward_verified": True,
            "precision_fp16_verified": True,
            "synthetic_gpu_output_shape": [2, spec["feature_dimension"]],
            "synthetic_gpu_logits_shape": [2], "authenticated_colab33_record_sha256": "c" * 64,
            "synthetic_gpu_fp16_output_shape": [2, spec["feature_dimension"]],
            "synthetic_gpu_fp16_logits_shape": [2],
            "verified_checkpoint_files_sha256": {item["name"]: item["sha256"] for item in spec["files"]},
            "license_review": deepcopy(spec["license_review"]), "overlap_review": spec["overlap_review"]}
        records[model_id].update(synthetic_smoke_receipt(model_id, value["package_sha256"]))
    return {"image_eligibility_verified": True, "model_eligibility": records,
        "development_package_sha256": value["package_sha256"],
        "model_input_reference_sha256": sha_file(value["reference_path"])}


def tiny_image_model(model_id, *args, **kwargs):
    """Distinct tiny states detect accidental cross-encoder reference reuse."""
    if model_id not in subject.REMAINING_IMAGE_IDS:
        raise AssertionError("No DenseNet, gated encoder or extra model in this campaign")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(100 + subject.REMAINING_IMAGE_IDS.index(model_id))
        model = torch.nn.Module()
        model.add_module("encoder", torch.nn.Linear(2, 2))
        model.add_module("task_head", torch.nn.Linear(2, 1))
    model.model_id = model.image_encoder_id = model_id
    model.phase = "head_only"
    model.clinical_scenario = "clinical_main"
    model.encoder.requires_grad_(False)
    return model, {"synthetic_factory": True}


class MethodologyImageCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        _, cls.catalog, cls.plan, cls.registry = release.load_release(ROOT)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_exact_model_major_order_and_frozen_gamma_scope(self):
        requests = subject.requests(CAMPAIGN, "a" * 64, "b" * 64, self.plan, self.registry)
        expected_models = ("image_vit_b16", "image_convnextv2_tiny", "image_skelex",
            "image_biomedclip", "image_mambaout_tiny")
        self.assertEqual(subject.image_models(CAMPAIGN), expected_models)
        self.assertEqual([(r["model_id"], r["validation_fold"]) for r in requests],
            [(model, fold) for model in expected_models for fold in range(5)])
        self.assertEqual(len({subject.job_key(r) for r in requests}), 25)
        for request in requests:
            self.assertEqual(request["image_encoder_id"], request["model_id"])
            self.assertEqual(request["clinical_scenario"], "clinical_main")
            self.assertEqual(request["augmentation_policy"], "training_intensity_gamma_v1.0")
            self.assertEqual(request["stage"], "stage_1")
            self.assertEqual(request["seed"], 2026)
            self.assertIsNone(request["baseline_sidecar_sha256"])
            self.assertIsNone(request["selection_evidence_sha256"])
            self.assertIsNone(request["fold_image_job_sha256"])

    def test_authority_requires_own_campaign_and_verified_eligibility(self):
        request = subject.requests(CAMPAIGN, "a" * 64, "b" * 64, self.plan, self.registry)[0]
        with self.assertRaisesRegex(ValueError, "verified_image_eligibility"):
            subject.authority(request, CAMPAIGN, subject.APPROVAL_STATEMENTS[CAMPAIGN], self.plan, self.registry)
        with self.assertRaisesRegex(ValueError, "campaign_training_confirmation"):
            subject.authority(request, CAMPAIGN, subject.APPROVAL_STATEMENTS["densenet_frozen"],
                self.plan, self.registry, image_eligibility_verified=True)
        authority = subject.authority(request, CAMPAIGN, subject.APPROVAL_STATEMENTS[CAMPAIGN],
            self.plan, self.registry, image_eligibility_verified=True,
            encoder_smoke_evidence=synthetic_smoke_receipt(request["model_id"], "a" * 64))
        self.assertTrue(all(authority["eligibility_gates"].values()))
        self.assertFalse(authority["full_encoder_training_authorized"])
        self.assertFalse(authority["reserved_test_access_authorized"])
        for model_id in ("image_densenet121", "image_dinov3_vits16", "fusion_intermediate_concat"):
            changed = {**request, "model_id": model_id, "image_encoder_id": model_id}
            with self.subTest(model_id=model_id), self.assertRaises(ValueError):
                subject.authority(changed, CAMPAIGN, subject.APPROVAL_STATEMENTS[CAMPAIGN],
                    self.plan, self.registry, image_eligibility_verified=True)

    def test_own_weight_license_and_gpu_shape_evidence_cannot_be_replaced(self):
        # Metadata-only evidence fixture; no source file needs to be opened.
        records = {}
        for model_id in subject.REMAINING_IMAGE_IDS:
            spec = self.catalog["models"][model_id]
            records[model_id] = {"model_id": model_id, "image_eligibility_verified": True,
                "synthetic_gpu_forward_verified": True, "precision_fp16_verified": True,
                "synthetic_gpu_output_shape": [2, spec["feature_dimension"]],
                "synthetic_gpu_logits_shape": [2], "authenticated_colab33_record_sha256": "c" * 64,
                "synthetic_gpu_fp16_output_shape": [2, spec["feature_dimension"]],
                "synthetic_gpu_fp16_logits_shape": [2],
                "verified_checkpoint_files_sha256": {p["name"]: p["sha256"] for p in spec["files"]},
                "license_review": deepcopy(spec["license_review"]), "overlap_review": spec["overlap_review"]}
            records[model_id].update(synthetic_smoke_receipt(model_id, "a" * 64))
        evidence = {"model_eligibility": records, "development_package_sha256": "a" * 64}
        self.assertEqual(subject._validate_remaining_eligibility(evidence, self.catalog), records)
        model_id = subject.REMAINING_IMAGE_IDS[0]
        for key, bad in (("image_eligibility_verified", False), ("synthetic_gpu_forward_verified", False),
                ("precision_fp16_verified", False),
                ("model_id", "image_densenet121"), ("synthetic_gpu_output_shape", [2, 1]),
                ("synthetic_gpu_logits_shape", [2, 2]), ("authenticated_colab33_record_sha256", "bad"),
                ("synthetic_gpu_fp16_output_shape", [2, 1]), ("synthetic_gpu_fp16_logits_shape", [2, 2]),
                ("verified_checkpoint_files_sha256", {}), ("license_review", {}), ("overlap_review", "unknown")):
            changed = deepcopy(evidence)
            changed["model_eligibility"][model_id][key] = bad
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "model_specific_verified"):
                subject._validate_remaining_eligibility(changed, self.catalog)
        for field, bad in (("training_smoke_test_passed", False), ("optimizer_updates", 1),
                ("gpu_resource_profile_recorded", False), ("smoke_request_sha256", "invalid"),
                ("development_package_sha256", "b" * 64), ("task_head_update_verified", False)):
            changed = deepcopy(evidence)
            changed["model_eligibility"][model_id][field] = bad
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "own_two_update_smoke"):
                subject._validate_remaining_eligibility(changed, self.catalog)
        for replacement in ({}, {**records, "image_densenet121": records[model_id]}):
            with self.assertRaisesRegex(ValueError, "exact_remaining_encoder"):
                subject._validate_remaining_eligibility({"model_eligibility": replacement}, self.catalog)

    def test_25_jobs_use_real_runner_bindings_validators_and_recover_without_fit(self):
        from knee import final_models_preflight
        from knee import methodology_campaign_colab as orchestration
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            value = synthetic_preflight(root)
            weights = root / "public_weights"
            weights.mkdir()
            evidence = synthetic_evidence(value, self.catalog)
            persistent = root / "persistent"
            persistent.mkdir()
            identity = orchestration.campaign_identity(ROOT, value["preflight"],
                value["reference_path"], "a" * 40, CAMPAIGN)
            durable = orchestration.open_campaign(persistent, identity)
            requests = subject.requests(CAMPAIGN, value["package_sha256"], "b" * 64, self.plan, self.registry)
            request_map = {(r["model_id"], r["validation_fold"]): r for r in requests}
            binding_calls, callbacks, initial_loads, checkpoint_ids = [], [], [], []
            last_references = []
            actual_initial_state = subject._initial_image_state

            def persist_job(key, path, record):
                stored = orchestration.persist_completed_job(durable, key, path, record)
                callbacks.append((key, stored, record))

            def tracked_initial(model_id, *args):
                self.assertTrue(all(reference() is None for reference in last_references),
                    "The previous encoder reference must be released before loading another")
                state = actual_initial_state(model_id, *args)
                last_references[:] = [weakref.ref(tensor) for tensor in state.values()]
                initial_loads.append(model_id)
                return state

            def checkpoint_stub(model_id, catalog, destination, allow_download):
                self.assertEqual(Path(destination), weights)
                self.assertTrue(allow_download)
                self.assertIn(model_id, subject.REMAINING_IMAGE_IDS)
                checkpoint_ids.append(model_id)
                return []

            def synthetic_engine(model, training, validation, plan, *, device, seed, positive_weight,
                    stage, output_dir, maximum_steps, smoke, execution_authority):
                permit = execution_authority
                require_permit(permit, model_id=model.model_id, stage=stage, smoke=smoke)
                binding = engine._validate_loader_bindings(training, validation, permit, positive_weight)
                self.assertEqual(device, "cuda:0")
                self.assertEqual(stage, "stage_1")
                self.assertEqual(seed, 2026)
                self.assertFalse(smoke)
                self.assertIsNone(maximum_steps)
                self.assertTrue(all(not p.requires_grad for p in model.encoder.parameters()))
                self.assertEqual(training.dataset.augmentation_policy, plan["neural_training"]["augmentation"])
                self.assertIsNone(validation.dataset.augmentation_policy)
                binding_calls.append((model.model_id, permit.validation_fold, binding))
                request = request_map[(model.model_id, permit.validation_fold)]
                self.assertEqual(subject.digest(request), permit.request_sha256)
                initial = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
                output = Path(output_dir)
                synthetic_neural_job(output, request, value["rows"], self.registry, initial, device)
                # Only the helper's artificial engine model label is generalized.
                # The real runner writes and seals its own job/state files below.
                record_path = output / "training_record.json"
                record = runner.read_json(record_path)
                record["model_id"] = model.model_id
                record_path.unlink()
                runner.save_json(record_path, record)
                for name in (runner.JOB_FILENAME, "clinical_state.json"):
                    (output / name).unlink()
                return record

            forbidden = AssertionError("Real fitting, optimizer, backward or network forbidden")
            for target in ("torch.optim.AdamW", "torch.Tensor.backward", "torch.autograd.backward",
                    "sklearn.linear_model.LogisticRegression.fit", "xgboost.XGBClassifier.fit",
                    "urllib.request.OpenerDirector.open", "requests.sessions.Session.send"):
                stack.enter_context(patch(target, side_effect=forbidden))
            stack.enter_context(patch("knee.training_smoke.verify_gpu_environment", return_value={"synthetic_no_gpu": True}))
            stack.enter_context(patch.object(final_models_preflight, "checkpoint_files", side_effect=checkpoint_stub))
            stack.enter_context(patch.object(models, "build_model", side_effect=tiny_image_model))
            stack.enter_context(patch.object(subject, "_initial_image_state", side_effect=tracked_initial))
            neural = stack.enter_context(patch.object(engine, "train_neural_fold", side_effect=synthetic_engine))
            result = subject.run(ROOT, value["preflight"], value["package"], value["reference_path"], root / "campaign",
                campaign=CAMPAIGN, approval_statement=subject.APPROVAL_STATEMENTS[CAMPAIGN],
                weights_root=weights, gpu_evidence=evidence,
                job_completed=persist_job)
            self.assertEqual(len(binding_calls), 25)
            self.assertEqual(neural.call_count, 25)
            self.assertEqual(initial_loads, list(subject.REMAINING_IMAGE_IDS))
            self.assertEqual(checkpoint_ids, [model for model in subject.REMAINING_IMAGE_IDS for _ in range(6)])
            self.assertTrue(all(reference() is None for reference in last_references))
            self.assertEqual(result["job_count"], 25)
            self.assertEqual(len(result["model_results"]), 5)
            self.assertEqual(len(result["artifact_files_sha256"]), 155)
            self.assertEqual([r["key"] for r in result["job_results"]], [subject.job_key(r) for r in requests])
            self.assertEqual(result["image_augmentation_policy"], self.plan["neural_training"]["augmentation"])
            self.assertTrue(all(item["out_of_fold_predictions"] == 30 for item in result["model_results"].values()))
            for flag in ("encoder_unfreezing_executed", "historical_v1_1_jobs_reused", "calibration_performed",
                    "threshold_selected", "final_refit_executed", "reserved_test_accessed"):
                self.assertFalse(result[flag])
            transports = orchestration.completed_job_transports(durable)
            self.assertEqual(len(transports), 25)
            previous = {key: item["directory"] for key, item in transports.items()}
            self.assertEqual(set(previous), {key for key, _, _ in callbacks})
            with patch.object(runner, "execute", side_effect=AssertionError("Recovered jobs must not run")):
                recovered = subject.run(ROOT, value["preflight"], value["package"], value["reference_path"], root / "recovered",
                    campaign=CAMPAIGN, approval_statement=subject.APPROVAL_STATEMENTS[CAMPAIGN],
                    weights_root=weights, gpu_evidence=evidence, completed_jobs=previous,
                    job_completed=persist_job)
            self.assertEqual(neural.call_count, 25)
            self.assertEqual(recovered["model_results"], result["model_results"])
            self.assertTrue(all(j["reused_authenticated_complete_job"] for j in recovered["job_results"]))
            self.assertEqual(initial_loads, list(subject.REMAINING_IMAGE_IDS) * 2)
            self.assertEqual(len(orchestration.completed_job_transports(durable)), 25)
            self.assertEqual(len(callbacks), 50)

    def test_missing_encoder_evidence_fails_before_checkpoint_or_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = synthetic_preflight(root)
            weights = root / "weights"
            weights.mkdir()
            evidence = synthetic_evidence(value, self.catalog)
            evidence["model_eligibility"].pop(subject.REMAINING_IMAGE_IDS[-1])
            with patch("knee.training_smoke.verify_gpu_environment", return_value={"synthetic_no_gpu": True}), \
                    patch("knee.final_models_preflight.checkpoint_files", side_effect=AssertionError("No weights before eligibility")), \
                    patch.object(runner, "execute", side_effect=AssertionError("No jobs before eligibility")), \
                    self.assertRaisesRegex(ValueError, "exact_remaining_encoder_eligibility"):
                subject.run(ROOT, value["preflight"], value["package"], value["reference_path"], root / "campaign",
                    campaign=CAMPAIGN, approval_statement=subject.APPROVAL_STATEMENTS[CAMPAIGN],
                    weights_root=weights, gpu_evidence=evidence)
            self.assertFalse((root / "campaign").exists())

    def test_two_update_prerequisite_uses_real_runner_and_frozen_state_checks(self):
        from knee import final_models_preflight
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            value = synthetic_preflight(root)
            weights = root / "weights"
            weights.mkdir()
            model_id = subject.REMAINING_IMAGE_IDS[0]
            evidence = synthetic_evidence(value, self.catalog)["model_eligibility"][model_id]
            req = subject.requests(CAMPAIGN, value["package_sha256"], "b" * 64, self.plan, self.registry)[0]
            req["operation"] = "single_fold_smoke"
            calls = []

            def artificial_smoke(model, training, validation, plan, *, device, seed, positive_weight,
                    stage, output_dir, maximum_steps, smoke, execution_authority):
                self.assertTrue(smoke)
                self.assertEqual(maximum_steps, 2)
                require_permit(execution_authority, model_id=model_id, stage="stage_1", smoke=True)
                binding = engine._validate_loader_bindings(training, validation, execution_authority, positive_weight)
                self.assertEqual(subject.digest(req), execution_authority.request_sha256)
                initial = {k:t.detach().clone() for k,t in model.state_dict().items()}
                output = Path(output_dir)
                synthetic_neural_job(output, req, value["rows"], self.registry, initial, device)
                path = output / "training_record.json"
                result = runner.read_json(path)
                history = result["history"][:1]
                result.update(model_id=model_id, smoke=True, eligible_for_model_selection=False,
                    status="bounded_training_smoke_complete", completed_epochs=1, optimizer_updates=2, history=history,
                    amp_scale_audit=[{"epoch":1, **row} for row in history[0]["amp_scale_audit"]],
                    resource_profile={"device_type":"cuda","cuda_peak_measured":True,"torch_version":"2.6.0+cu124",
                        "cuda_version":"12.4","device_name":"synthetic-no-GPU","total_memory_bytes":100,
                        "peak_memory_allocated_bytes":20,"peak_memory_reserved_bytes":30,"elapsed_seconds":1.})
                path.unlink()
                runner.save_json(path, result)
                for name in (runner.JOB_FILENAME, "clinical_state.json"):
                    (output / name).unlink()
                calls.append(binding)
                return result

            for target in ("torch.optim.AdamW", "torch.Tensor.backward", "torch.autograd.backward",
                    "urllib.request.OpenerDirector.open", "requests.sessions.Session.send"):
                stack.enter_context(patch(target, side_effect=AssertionError("No real fitting/network")))
            stack.enter_context(patch("knee.training_smoke.verify_gpu_environment", return_value={"synthetic_no_gpu":True}))
            stack.enter_context(patch.object(final_models_preflight,"checkpoint_files",return_value=[]))
            stack.enter_context(patch.object(models,"build_model",side_effect=tiny_image_model))
            stack.enter_context(patch.object(engine,"train_neural_fold",side_effect=artificial_smoke))
            output = root / "smoke"
            result = subject.run_remaining_encoder_smoke(ROOT,value["preflight"],value["package"],value["reference_path"],
                weights, output, model_id=model_id, approval_statement=subject.APPROVAL_STATEMENTS[CAMPAIGN], gpu_probe=evidence)
            self.assertEqual(len(calls),1)
            self.assertTrue(result["training_smoke_test_passed"])
            self.assertFalse(result["eligible_for_model_selection"])
            job = runner.read_json(output / "job" / runner.JOB_FILENAME)
            self.assertEqual(list(job["eligibility_gates"].values()).count(False),2)
            record = runner.read_json(output / "job" / "training_record.json")
            self.assertEqual(subject.validate_remaining_smoke_record(model_id,result,record,job,self.registry),result)
            changed = deepcopy(record)
            changed["optimizer_updates"] = 1
            with self.assertRaisesRegex(ValueError,"exact_current_two_update"):
                subject.validate_remaining_smoke_record(model_id,result,changed,job,self.registry)
            bad = {**evidence,"training_smoke_test_passed":False}
            full = subject.requests(CAMPAIGN,value["package_sha256"],"b"*64,self.plan,self.registry)[0]
            with self.assertRaisesRegex(ValueError,"own_two_update_smoke"):
                subject.authority(full,CAMPAIGN,subject.APPROVAL_STATEMENTS[CAMPAIGN],self.plan,self.registry,
                    image_eligibility_verified=True,encoder_smoke_evidence=bad)


if __name__ == "__main__":
    unittest.main()
