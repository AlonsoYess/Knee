"""Synthetic records, tensors and mocked execution; never train a predictor."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from knee import training_smoke as subject
from knee import training_runner as runner
from knee.colab_data import sha_file
from knee.final_model_release import PINS, load_release
from knee.training_authority import digest, validate_authority

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, record):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(record), encoding="utf-8")


def preparation_fixture(directory):
    """Synthetic completed34 schema; no actual scientific inputs are read."""
    core = {"version": subject.preparation.VERSION, "status": subject.preparation.STATUS,
        "execution_environment": "google_colab_cpu", "colab_runtime_executed": True,
        **{name: False for name in subject.previous.FALSE_FLAGS},
        "training_runner_implemented": True, "synthetic_clinical_eval_forward_verified": True,
        "image_normalization_only_verified": True, "shortlist_search_closed": True,
        "pending_authorized_access": ["image_dinov3_vits16"],
        "actual_training_loop_executed": False, "training_smoke_test_passed": False,
        "gpu_resource_profile_recorded": False,
        "source_files_sha256": dict(subject.SOURCE_34), "release_component_canonical_sha256": dict(PINS),
        "previous_colab33_checkpoint_record_sha256": subject.PREVIOUS_CHECKPOINT_SHA256,
        "clinical_models_implemented": ["clinical_logistic", "clinical_xgboost", "clinical_mlp"],
        "accessible_image_models_implemented": ["image_densenet121", "image_vit_b16", "image_convnextv2_tiny",
            "image_skelex", "image_biomedclip", "image_mambaout_tiny"],
        "conditional_fusion_models_implemented": ["fusion_intermediate_concat", "fusion_intermediate_gated_film"],
        "executed_at_utc": "2026-10-22T16:00:00+00:00"}
    root = Path(directory)
    write_json(root / subject.previous.CORE_FILENAME, core)
    summary = {**core, "code_revision": subject.PREPARATION_REVISION,
        "local_rehearsal": False, "cpu_only_guard_passed": True,
        "result_files_saved_and_readback_verified": True,
        "preparation_record_sha256": sha_file(root / subject.previous.CORE_FILENAME),
        "runtime_requirements_sha256": subject.SOURCE_34["requirements-training-runner.txt"],
        "cpu_tests": {"passed": True, "tests_run": 313, "skipped": 0,
            "scope": list(subject.previous.TEST_MODULES)},
        "executed_at_utc": "2026-10-22T16:01:00+00:00"}
    write_json(root / subject.previous.SUMMARY_FILENAME, summary)
    return root


def initial_state():
    return {"encoder.weight": torch.tensor([1., 2.]), "encoder.running_mean": torch.tensor([0., 0.]),
        "task_head.weight": torch.tensor([[.5, -.5]]), "task_head.bias": torch.tensor([0.])}


def fake_engine_record():
    return {"version": "prepared_neural_fold_engine_v1.0", "model_id": subject.MODEL_ID,
        "stage": "stage_1", "seed": 2026, "smoke": True, "scope": "development_only",
        "status": "bounded_training_smoke_complete", "eligible_for_model_selection": False,
        "reserved_test_accessed": False, "optimizer_created": True, "training_executed": True,
        "optimizer_updates": 2, "completed_epochs": 1, "micro_batch_size": 2,
        "effective_batch_size": 16, "device": "cuda:0", "precision": "float16_autocast_grad_scaler",
        "gpu_resource_profile_recorded": True, "restored_best_checkpoint": True,
        "resume_supported": False, "optimizer_state_saved": False,
        "history": [{"epoch": 1, "phase": "head_only", "optimizer_updates": 2, "training_knees_seen": 32}],
        "resource_profile": {"device_type": "cuda", "cuda_peak_measured": True,
            "torch_version": "2.6.0+cu124", "cuda_version": "12.4", "device_name": "SYNTHETIC GPU",
            "total_memory_bytes": 10000, "peak_memory_allocated_bytes": 500,
            "peak_memory_reserved_bytes": 1000, "elapsed_seconds": 1.}}


def fake_job(directory, request, *, engine_change=None, encoder_change=False, unchanged_head=False):
    """Write wholly artificial checkpoint copies; no model forward/backward."""
    root = Path(directory)
    root.mkdir()
    engine = fake_engine_record()
    if engine_change:
        engine_change(engine)
    write_json(root / "training_record.json", engine)
    for name in ("clinical_state.json", "best_validation_predictions.json"):
        write_json(root / name, {"synthetic_fixture": True})
    state = initial_state()
    if not unchanged_head:
        state["task_head.bias"] = torch.tensor([.01])
    if encoder_change:
        state["encoder.running_mean"] = torch.tensor([.01, 0.])
    for name in ("best_model.safetensors", "last_model.safetensors"):
        save_file(state, str(root / name))
    job = {"version": runner.VERSION, "scope": "development_only", "request": request,
        "request_sha256": digest(request), "eligible_for_model_selection": False,
        "reserved_test_accessed": False, "training_executed": True,
        "engine_status": "bounded_training_smoke_complete",
        "files_sha256": {name: sha_file(root / name) for name in subject.JOB_FILES}}
    job["record_sha256"] = digest(job)
    write_json(root / runner.JOB_FILENAME, job)
    return job


class PreparationEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gpu_smoke35_synthetic_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.proof = preparation_fixture(self.root / "proof")

    def edit_summary(self, function):
        path = self.proof / subject.previous.SUMMARY_FILENAME
        value = json.loads(path.read_text(encoding="utf-8"))
        function(value)
        write_json(path, value)

    def test_exact_completed_colab34_proof_passes_without_training_authority(self):
        proof = subject.validate_preparation(self.proof)
        self.assertFalse(proof["training_authorized"])
        self.assertEqual(proof["cpu_tests"]["tests_run"], 313)

    def test_changed_raw_preparation_bytes_are_rejected(self):
        path = self.proof / subject.previous.CORE_FILENAME
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "exact_preparation_record"):
            subject.validate_preparation(self.proof)

    def test_mismatched_revision_test_count_skips_and_prior_training_rejected(self):
        path = self.proof / subject.previous.SUMMARY_FILENAME
        original = path.read_bytes()
        edits = [lambda v: v.update(code_revision="a"*40), lambda v:v.update(training_executed=True),
            lambda v:v.update(local_rehearsal=True), lambda v:v["cpu_tests"].update(tests_run=312),
            lambda v:v["cpu_tests"].update(skipped=1), lambda v:v["cpu_tests"].update(scope=[])]
        for edit in edits:
            path.write_bytes(original)
            self.edit_summary(edit)
            with self.subTest(edit=edit), self.assertRaises((ValueError, RuntimeError)):
                subject.validate_preparation(self.proof)

    def test_resealed_changed_source_pin_and33_chain_cannot_pass(self):
        for field, changed in (("source_files_sha256", {}),
            ("previous_colab33_checkpoint_record_sha256", "0"*64),
            ("release_component_canonical_sha256", {})):
            root = preparation_fixture(self.root / field)
            path = root / subject.previous.CORE_FILENAME
            value = json.loads(path.read_text(encoding="utf-8"))
            value[field] = changed
            write_json(path, value)
            summary_path = root / subject.previous.SUMMARY_FILENAME
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary.update(value, preparation_record_sha256=sha_file(path))
            write_json(summary_path, summary)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "code_release"):
                subject.validate_preparation(root)

    def test_missing_pair_and_naive_or_earlier_publication_fail(self):
        with self.assertRaises(ValueError):
            subject.validate_preparation(self.root)
        for value in (None, "invalid", "2026-10-22T16:01:00", "2026-10-22T15:59:59+00:00"):
            self.edit_summary(lambda v: v.update(executed_at_utc=value))
            with self.subTest(value=value), self.assertRaises(ValueError):
                subject.validate_preparation(self.proof)

    def test_preserved_source_pins_match_repository_without_any_model_call(self):
        sources = subject._sources(ROOT)
        self.assertEqual({name:sources[name] for name in subject.SOURCE_34}, subject.SOURCE_34)

    def test_request_has_only_fixed_densenet_fold_zero_seed_and_two_microbatch(self):
        request = subject.make_request("a"*64)
        _, catalog, plan, registry = load_release(ROOT)
        authority = subject._authority(request, registry, catalog, subject.APPROVAL_STATEMENT)
        permit = validate_authority(authority, request, plan, registry)
        self.assertTrue(permit.smoke)
        self.assertFalse(permit.full_encoder_training_authorized)
        self.assertEqual((permit.model_id, permit.validation_fold, permit.seed, permit.image_micro_batch_size),
            ("image_densenet121", 0, 2026, 2))
        self.assertEqual(list(authority["eligibility_gates"].values()), [True]*4+[False]*2)

    def test_invalid_hash_and_other_authority_statement_fail_closed(self):
        for value in ("", "G"*64, "a"*63, None, 123):
            with self.subTest(value=value), self.assertRaises(ValueError):
                subject.make_request(value)
        _, catalog, _, registry = load_release(ROOT)
        for statement in ("", "authorize all training", None):
            with self.subTest(statement=statement), self.assertRaisesRegex(ValueError, "exact_separate"):
                subject._authority(subject.make_request("a"*64), registry, catalog, statement)

    def test_scope_license_and_unresolved_overlap_are_not_silently_redefined(self):
        _, catalog, _, registry = load_release(ROOT)
        for mutate in (lambda v:v["models"][subject.MODEL_ID]["license_review"].update(use_scope="commercial"),
            lambda v:v["shared_limits"].update(pretraining_subject_disjointness_proven=True)):
            changed = deepcopy(catalog)
            mutate(changed)
            with self.assertRaisesRegex(ValueError, "eligibility_evidence"):
                subject._authority(subject.make_request("a"*64), registry, changed, subject.APPROVAL_STATEMENT)


class SmokeResultTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="smoke_job35_mock_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.request = subject.make_request("a"*64)

    def test_mocked_exact_two_step_job_and_frozen_state_pass(self):
        root = self.root / "job"
        job = fake_job(root, self.request)
        self.assertEqual(subject._validate_job(root, job, self.request)["optimizer_updates"], 2)
        state = subject._verify_frozen_encoder(root, initial_state())
        self.assertTrue(state["frozen_encoder_state_unchanged"] and state["task_head_update_verified"])

    def test_encoder_buffers_or_head_not_updated_fail(self):
        for label, kwargs, message in (("encoder", {"encoder_change":True}, "running_statistics_changed"),
            ("head", {"unchanged_head":True}, "must_actually_update")):
            root = self.root / label
            fake_job(root, self.request, **kwargs)
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, message):
                subject._verify_frozen_encoder(root, initial_state())

    def test_one_or_three_updates_unfreezing_and_cpu_fallback_fail(self):
        for index, change in enumerate((lambda v:v.update(optimizer_updates=1), lambda v:v.update(optimizer_updates=3),
            lambda v:v.update(device="cpu"), lambda v:v.update(stage="stage_2"),
            lambda v:v["history"][0].update(phase="full"),
            lambda v:v["history"][0].update(training_knees_seen=16))):
            root = self.root / str(index)
            job = fake_job(root, self.request, engine_change=change)
            with self.subTest(index=index), self.assertRaises(ValueError):
                subject._validate_job(root, job, self.request)

    def test_missing_gpu_profile_or_nonfinite_resource_measurements_fail(self):
        for index, change in enumerate((lambda v:v.update(gpu_resource_profile_recorded=False),
            lambda v:v["resource_profile"].update(cuda_peak_measured=False),
            lambda v:v["resource_profile"].update(elapsed_seconds=float("nan")),
            lambda v:v["resource_profile"].update(peak_memory_allocated_bytes=20000))):
            root = self.root / str(index)
            job = fake_job(root, self.request, engine_change=change)
            with self.subTest(index=index), self.assertRaises(ValueError):
                subject._validate_job(root, job, self.request)

    def test_changed_artifact_additional_file_or_resealed_model_selection_are_rejected(self):
        root = self.root / "job"
        job = fake_job(root, self.request)
        path = root / "clinical_state.json"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "artifacts"):
            subject._validate_job(root, job, self.request)
        root2 = self.root / "job2"
        job2 = fake_job(root2, self.request)
        (root2 / "unexpected.json").write_bytes(b"{}")
        with self.assertRaisesRegex(ValueError, "artifacts"):
            subject._validate_job(root2, job2, self.request)
        root3 = self.root / "job3"
        job3 = fake_job(root3, self.request)
        job3["eligible_for_model_selection"] = True
        job3["record_sha256"] = digest({key:value for key,value in job3.items() if key != "record_sha256"})
        write_json(root3 / runner.JOB_FILENAME, job3)
        with self.assertRaisesRegex(ValueError, "authenticated_development"):
            subject._validate_job(root3, job3, self.request)


class RuntimeAndInvocationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="smoke_run35_mock_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.proof = preparation_fixture(self.root / "proof")
        self.package = self.root / "development"
        self.package.mkdir()
        (self.package / "development_package.zip").write_bytes(b"SYNTHETIC; NO ZIP OR PARTICIPANTS")
        write_json(self.package / "handoff.json", {"synthetic_only": True})
        self.reference = self.root / "reference" / "reference.json"
        write_json(self.reference, {"sources_sha256": {"development_package": "a"*64}})
        self.weights = self.root / "weights"
        self.weights.mkdir()
        self.checkpoint = self.weights / "synthetic_checkpoint.pth"
        self.checkpoint.write_bytes(b"SYNTHETIC MOCK; NEVER LOADED")

    def run_mock(self, *, execute_side_effect=None, output=None, approval=subject.APPROVAL_STATEMENT):
        output = output or self.root / "output"
        def execute(request, authority, **kwargs):
            self.assertEqual(kwargs["device"], "cuda:0")
            self.assertFalse(kwargs["allow_download"])
            _, _, plan, registry = load_release(ROOT)
            permit = validate_authority(authority, request, plan, registry)
            self.assertTrue(permit.smoke)
            self.assertFalse(permit.full_encoder_training_authorized)
            job = fake_job(kwargs["output_dir"], request)
            if execute_side_effect:
                execute_side_effect(request, authority, kwargs)
            return job
        with patch.object(subject, "verify_gpu_environment", return_value={"synthetic_gpu_mock":True}), \
            patch("knee.model_inputs_colab._reference"), \
            patch("knee.final_models_preflight.checkpoint_files", return_value=[self.checkpoint]) as acquire, \
            patch("knee.training_models.build_model", return_value=(SimpleNamespace(state_dict=initial_state), {})), \
            patch.object(runner, "execute", side_effect=execute) as train, \
            patch("torch.optim.AdamW", side_effect=AssertionError("actual optimizer forbidden")), \
            patch("torch.Tensor.backward", side_effect=AssertionError("actual backward forbidden")), \
            patch("knee.training_engine.train_neural_fold", side_effect=AssertionError("actual training forbidden")):
            result = subject.run(ROOT, self.proof, self.package, self.reference, self.weights, output,
                approval_statement=approval)
        return result, acquire, train

    def test_mocked_lifecycle_seals_private_artifacts_and_never_runs_real_training(self):
        result, acquire, train = self.run_mock()
        self.assertEqual(train.call_count, 1)
        self.assertEqual(acquire.call_args.args[0], subject.MODEL_ID)
        self.assertEqual(acquire.call_args.kwargs, {"allow_download":True})
        self.assertEqual(result["status"], subject.STATUS)
        self.assertTrue(result["training_smoke_test_passed"])
        self.assertFalse(result["full_training_authorized"] or result["eligible_for_model_selection"]
            or result["reserved_test_accessed"] or result["encoder_unfreezing_executed"])
        self.assertEqual(len(result["artifact_files_sha256"]), 6)
        self.assertEqual(subject.preparation.read_json(self.root / "output" / subject.FILENAME), result)
        self.assertNotIn("metrics", result)
        self.assertNotIn("trained_participants", result)

    def test_no_approval_fails_before_environment_or_any_source_access(self):
        with patch.object(subject, "verify_gpu_environment", side_effect=AssertionError("no runtime call")):
            with self.assertRaisesRegex(ValueError, "exact_separate"):
                subject.run(None, None, None, None, None, None, approval_statement="")

    def test_existing_or_source_overlapping_output_rejected_before_runner(self):
        for output in (self.proof, self.root / "reference" / "output", self.weights / "output"):
            with self.subTest(output=output), self.assertRaises(ValueError):
                self.run_mock(output=output)

    def test_changed_reference_during_mocked_execution_fails_before_success_record(self):
        def mutate(request, authority, kwargs):
            self.reference.write_bytes(self.reference.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "sources_changed"):
            self.run_mock(execute_side_effect=mutate)
        self.assertFalse((self.root / "output" / subject.FILENAME).exists())

    def test_gpu_environment_rejects_local_cpu_and_unpinned_torch(self):
        for env in ("local_cpu", "google_colab_cpu", "development_gpu"):
            with self.subTest(env=env), self.assertRaisesRegex(ValueError, "actual_isolated"):
                subject.verify_gpu_environment(env)
        with patch.object(subject.platform, "system", return_value="Linux"), \
            patch.object(subject.platform, "python_version", return_value="3.12.14"), \
            patch.dict(os.environ, {"COLAB_RELEASE_TAG":"synthetic"}), \
            patch("knee.training_engine._prepare_cuda_environment"), \
            patch("knee.training_gpu_runtime.verify_current_runtime", return_value={"synthetic_full_runtime": True}), \
            patch.object(subject.importlib.metadata, "version", side_effect=subject.GPU_PACKAGES.__getitem__), \
            patch.object(torch, "__version__", "2.6.0+cpu"):
            with self.assertRaisesRegex(ValueError, "no_cpu_fallback"):
                subject.verify_gpu_environment("google_colab_gpu")

    def test_gpu_environment_requires_cuda_available_and_exact_cu124(self):
        with patch.object(subject.platform, "system", return_value="Linux"), \
            patch.object(subject.platform, "python_version", return_value="3.12.14"), \
            patch.dict(os.environ, {"COLAB_RELEASE_TAG":"synthetic"}), \
            patch("knee.training_engine._prepare_cuda_environment"), \
            patch("knee.training_gpu_runtime.verify_current_runtime", return_value={"synthetic_full_runtime": True}) as full, \
            patch.object(subject.importlib.metadata, "version", side_effect=subject.GPU_PACKAGES.__getitem__), \
            patch.object(torch, "__version__", "2.6.0+cu124"), \
            patch.object(torch.version, "cuda", "12.4"), patch.object(torch.cuda, "is_available", return_value=True), \
            patch.object(torch.cuda, "device_count", return_value=1):
            result = subject.verify_gpu_environment("google_colab_gpu")
        self.assertEqual(result["device"], "cuda:0")
        full.assert_called_once_with("google_colab_gpu")
        self.assertEqual(result["runtime_verification"], {"synthetic_full_runtime": True})

    def test_full_runtime_dependency_drift_prevents_any_data_or_training_access(self):
        with patch.object(subject.platform, "system", return_value="Linux"), \
            patch.object(subject.platform, "python_version", return_value="3.12.14"), \
            patch.dict(os.environ, {"COLAB_RELEASE_TAG":"synthetic"}), \
            patch("knee.training_engine._prepare_cuda_environment"), \
            patch("knee.training_gpu_runtime.verify_current_runtime", side_effect=RuntimeError("package drift")) as full, \
            patch.object(subject, "checked_path", side_effect=AssertionError("data access forbidden")), \
            patch.object(runner, "execute", side_effect=AssertionError("training forbidden")):
            with self.assertRaisesRegex(RuntimeError, "package drift"):
                subject.run(None, None, None, None, None, None, approval_statement=subject.APPROVAL_STATEMENT)
        full.assert_called_once_with("google_colab_gpu")


if __name__ == "__main__":
    unittest.main()
