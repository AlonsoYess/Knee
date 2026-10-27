"""Synthetic gates and bounded transport for the five remaining encoders.

No real checkpoint, participant, optimizer, network or GPU is used by this suite.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from knee import methodology_campaign as core
from knee import methodology_campaign_colab as subject
from knee import methodology_release as release
from knee.colab_data import sha_file
from test_training_preparation import preflight_fixture

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = "remaining_encoders_frozen"


def image_preflight_fixture(root):
    proof = preflight_fixture(root)
    _, catalog, _, _ = release.load_release(ROOT)
    path = proof / "checkpoint_verification.json"
    checkpoint = subject._json(path)
    for model_id in core.REMAINING_IMAGE_IDS:
        specification = catalog["models"][model_id]
        checkpoint["model_records"][model_id].update(
            feature_shape=[2, specification["feature_dimension"]],
            files=[{key: item[key] for key in ("name", "bytes", "sha256")}
                for item in specification["files"]])
    path.write_text(json.dumps(checkpoint), encoding="utf-8")
    summary_path = proof / "colab_final_models_check.json"
    summary = subject._json(summary_path)
    summary["checkpoint_record_sha256"] = sha_file(path)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return proof


def tiny_image_model():
    import torch
    from torch import nn
    class TinyImage(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(3, 4))
            self.encoder.requires_grad_(False)
            self.head = nn.Linear(4, 1)
        def forward(self, images):
            return self.head(self.encoder(images)).squeeze(-1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        return TinyImage().eval()


def small_checkpoint_job(root, model_id, *, stage="stage_1"):
    root.mkdir()
    for name in ("best_model.safetensors", "last_model.safetensors"):
        (root / name).write_bytes(b"SYNTHETIC TENSOR PLACEHOLDER; NOT LOADABLE")
    request = {"version": release.REQUEST_VERSION, "model_id": model_id,
        "image_encoder_id": model_id, "stage": stage, "clinical_scenario": "clinical_main"}
    record = {"request": request, "files_sha256": {name: sha_file(root / name)
        for name in ("best_model.safetensors", "last_model.safetensors")}}
    (root / "training_job.json").write_text(json.dumps(record), encoding="utf-8")
    return root


class MethodologyImageCampaignColabTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ic_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_scope_is_expanded_only_for_notebook41(self):
        self.assertEqual(subject.test_modules("clinical"), subject.TEST_MODULES)
        self.assertEqual(subject.test_modules("densenet_frozen"), subject.TEST_MODULES)
        self.assertEqual(subject.test_modules(CAMPAIGN), subject.TEST_MODULES + (
            "test_methodology_image_campaign", "test_methodology_image_campaign_colab", "test_image_screening_notebook"))
        for campaign in ("all_models", "image_dinov3_vits16", "stage_2"):
            with self.assertRaises(ValueError):
                subject.test_modules(campaign)

    def test_new_campaign_reuses_exact_existing_gpu_setup_and_environment(self):
        with patch("knee.training_gpu_runtime.setup_runtime", return_value={"synthetic": True}) as setup, \
             patch("knee.training_gpu_runtime.runtime_environment", return_value={"synthetic": "gpu"}) as environment:
            self.assertEqual(subject.setup_runtime(ROOT, self.root / "runtime", CAMPAIGN), {"synthetic": True})
            self.assertEqual(subject.runtime_environment(ROOT, CAMPAIGN), {"synthetic": "gpu"})
            setup.assert_called_once_with(ROOT, self.root / "runtime")
            environment.assert_called_once_with(ROOT)

    def test_find33_requires_actual_authenticated_complete_checkpoint_records(self):
        proof = image_preflight_fixture(self.root)
        sentinel = self.root / "reserved_test_package.zip"
        sentinel.write_bytes(b"FORBIDDEN TEST SENTINEL")
        with patch("knee.aligned_training.models.build_model", side_effect=AssertionError("No models")), \
             patch("knee.final_models_preflight.checkpoint_files", side_effect=AssertionError("No downloads")), \
             patch("knee.aligned_training.runner.execute", side_effect=AssertionError("No training")):
            self.assertEqual(subject.find_image_preflight(self.root, ROOT), proof)
        self.assertEqual(sentinel.read_bytes(), b"FORBIDDEN TEST SENTINEL")

    def test_resealed_wrong_model_checkpoint_or_feature_shape_is_rejected(self):
        proof = image_preflight_fixture(self.root)
        checkpoint_path = proof / "checkpoint_verification.json"
        original = subject._json(checkpoint_path)
        summary_path = proof / "colab_final_models_check.json"
        for change in (lambda item: item["files"][0].update(sha256="0" * 64),
                       lambda item: item.update(feature_shape=[2, 1])):
            value = deepcopy(original)
            change(value["model_records"][core.REMAINING_IMAGE_IDS[0]])
            checkpoint_path.write_text(json.dumps(value), encoding="utf-8")
            summary = subject._json(summary_path)
            summary["checkpoint_record_sha256"] = sha_file(checkpoint_path)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact_five_registered_colab33"):
                subject.find_image_preflight(self.root, ROOT)

    def test_old_or_wrong_test_count_cannot_authorize_image_checkpoint_evidence(self):
        proof = image_preflight_fixture(self.root)
        path = proof / "colab_final_models_check.json"
        value = subject._json(path)
        value["cpu_tests"]["tests_run"] = 194
        path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "colab33_test_scope"):
            subject.find_image_preflight(self.root, ROOT)

    def test_eval_probe_checks_features_logits_and_does_not_fit_or_mutate(self):
        import torch
        model = tiny_image_model()
        batch = torch.linspace(0., 1., 2 * 3 * 4 * 4).reshape(2, 3, 4, 4)
        with patch("torch.optim.AdamW", side_effect=AssertionError("No optimizer")), \
             patch("torch.Tensor.backward", side_effect=AssertionError("No backward")):
            result = subject._eval_image_probe(model, batch, 4)
        self.assertEqual(result["synthetic_gpu_output_shape"], [2, 4])
        self.assertEqual(result["synthetic_gpu_logits_shape"], [2])
        self.assertTrue(result["parameters_and_buffers_unchanged"])
        self.assertTrue(result["synthetic_input_unchanged"])
        self.assertFalse(result["optimizer_created"])
        self.assertFalse(result["backward_executed"])
        self.assertFalse(result["gradients_created"])

    def test_probe_rejects_nonfinite_output_wrong_shape_and_unfrozen_encoder(self):
        import torch
        batch = torch.zeros(2, 3, 4, 4)
        model = tiny_image_model()
        with self.assertRaisesRegex(ValueError, "finite_registered"):
            subject._eval_image_probe(model, batch, 5)
        with torch.no_grad():
            model.head.bias.fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "finite_registered"):
            subject._eval_image_probe(model, batch, 4)
        model.encoder.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "frozen_eval_encoder"):
            subject._eval_image_probe(model, batch, 4)

    def test_eval_probe_exercises_mixed_precision_context_without_claiming_cuda_on_cpu(self):
        import torch
        model = tiny_image_model()
        with patch("torch.optim.AdamW", side_effect=AssertionError("No optimizer")), \
             patch("torch.Tensor.backward", side_effect=AssertionError("No backward")):
            result = subject._eval_image_probe(model, torch.ones(2, 3, 4, 4), 4,
                autocast_context=torch.autocast("cpu", dtype=torch.bfloat16))
        self.assertEqual(result["synthetic_gpu_output_shape"], [2, 4])
        self.assertEqual(result["synthetic_gpu_logits_shape"], [2])
        self.assertNotIn("precision_fp16_verified", result)
        self.assertNotIn("synthetic_gpu_forward_verified", result)

    def test_probe_detects_changed_buffers(self):
        import torch
        model = tiny_image_model()
        model.register_buffer("probe_counter", torch.zeros(()))
        original = model.forward
        def mutate(images):
            model.probe_counter.add_(1)
            return original(images)
        model.forward = mutate
        with self.assertRaisesRegex(ValueError, "must_not_change_weights"):
            subject._eval_image_probe(model, torch.zeros(2, 3, 4, 4), 4)

    def test_gpu_probe_cannot_silently_use_cpu(self):
        import torch
        model = tiny_image_model()
        model.to = lambda *args, **kwargs: model
        original_to = torch.Tensor.to
        def cpu_only(tensor, *args, **kwargs):
            if args and args[0] == "cuda:0":
                return tensor
            return original_to(tensor, *args, **kwargs)
        with patch("knee.aligned_training.models.build_model", return_value=(model, {})), \
             patch("knee.final_models_preflight.synthetic_batch", return_value=torch.zeros(2, 3, 4, 4)), \
             patch.object(torch.Tensor, "to", cpu_only), patch("torch.cuda.empty_cache"), \
             self.assertRaisesRegex(ValueError, "actual_cuda_synthetic_image_probe_required"):
            subject._probe_image_gpu(core.REMAINING_IMAGE_IDS[0], {}, {}, [])

    def test_five_distinct_probes_and_smokes_then_recovery_do_not_use_dense35(self):
        proof = image_preflight_fixture(self.root)
        preparation = self.root / "preparation"
        preparation.mkdir()
        reference = self.root / "reference" / "synthetic.json"
        reference.parent.mkdir()
        reference.write_text("{}", encoding="utf-8")
        weights = self.root / "weights"
        weights.mkdir()
        records = self.root / "records"
        records.mkdir()
        package = self.root / "package"
        package.mkdir()
        _, catalog, _, _ = release.load_release(ROOT)
        record38 = {"development_package_sha256": "a" * 64,
            "sources_sha256": {"notebook32_reference": sha_file(reference)}}
        saved_smokes = {model_id: {"model_id": model_id, "training_smoke_test_passed": True,
            "gpu_resource_profile_recorded": True, "optimizer_updates": 2, "validation_fold": 0,
            "seed": 2026, "stage": "stage_1", "development_package_sha256": "a" * 64,
            "training_plan_sha256": release.PINS[release.PLAN_PATH],
            "frozen_encoder_state_unchanged": True, "task_head_update_verified": True,
            "eligible_for_model_selection": False, "reserved_test_accessed": False,
            "smoke_request_sha256": "b" * 64, "smoke_job_record_sha256": "c" * 64,
            "smoke_training_record_sha256": "d" * 64} for model_id in core.REMAINING_IMAGE_IDS}
        calls = []
        def probe(model_id, catalog, plan, paths):
            calls.append(model_id)
            return {"synthetic_gpu_forward_verified": True, "precision_fp16_verified": True,
                "synthetic_gpu_output_shape": [2, catalog["models"][model_id]["feature_dimension"]],
                "synthetic_gpu_logits_shape": [2],
                "synthetic_gpu_fp16_output_shape": [2, catalog["models"][model_id]["feature_dimension"]],
                "synthetic_gpu_fp16_logits_shape": [2]}
        with patch("knee.training_smoke.verify_gpu_environment", return_value={"synthetic": True}), \
             patch.object(subject, "find_image_preflight", return_value=proof), \
             patch.object(core, "validate_preparation", return_value=({}, record38, {})), \
             patch.object(subject, "_stored_image_smokes", side_effect=[{}, saved_smokes]), \
             patch.object(core, "run_remaining_encoder_smoke") as smokes, \
             patch.object(subject, "_persist_image_smoke", side_effect=lambda directory, model_id, source:
                saved_smokes[model_id]) as publish, \
             patch("knee.final_models_preflight.checkpoint_files", return_value=[]) as checkpoints, \
             patch.object(subject, "_probe_image_gpu", side_effect=probe), \
             patch.object(subject, "_gpu_evidence", side_effect=AssertionError("DenseNet35 forbidden")):
            evidence = subject._image_gpu_evidence(records, ROOT, preparation, reference,
                weights, package, self.root / "output", records / "campaign")
            recovered = subject._image_gpu_evidence(records, ROOT, preparation, reference,
                weights, package, self.root / "next_output", records / "campaign")
        self.assertEqual(evidence, recovered)
        self.assertEqual(calls, list(core.REMAINING_IMAGE_IDS) * 2)
        self.assertEqual(smokes.call_count, 5)
        self.assertEqual(publish.call_count, 5)
        self.assertEqual([call.kwargs["model_id"] for call in smokes.call_args_list], list(core.REMAINING_IMAGE_IDS))
        for call in smokes.call_args_list:
            self.assertEqual(call.args[:5], (ROOT, preparation, package, reference, weights))
            self.assertEqual(call.kwargs["approval_statement"], subject.APPROVAL_STATEMENTS[CAMPAIGN])
            self.assertNotEqual(call.args[5], self.root / "output")
        self.assertEqual(checkpoints.call_count, 20)
        self.assertEqual([call.args[3] for call in checkpoints.call_args_list], [True, False] * 10)
        self.assertEqual(list(evidence["model_eligibility"]), list(core.REMAINING_IMAGE_IDS))
        for model_id, item in evidence["model_eligibility"].items():
            self.assertEqual(item["verified_checkpoint_files_sha256"],
                {entry["name"]: entry["sha256"] for entry in catalog["models"][model_id]["files"]})
            self.assertFalse(item["pretraining_subject_disjointness_proven"])
            self.assertTrue(item["preprocessing_localizer_oai_overlap_limitation_remains"])
        core._validate_remaining_eligibility(evidence, catalog)

    def test_authority_and_new_scope_receipt_are_checked_before_any_gpu_probe(self):
        with patch.object(subject, "_image_gpu_evidence", side_effect=AssertionError("Probe forbidden")), \
             patch.object(core, "run", side_effect=AssertionError("Training forbidden")):
            with self.assertRaisesRegex(ValueError, "future_investigator_confirmation"):
                subject.train(None, None, None, None, None, None, campaign=CAMPAIGN,
                    approval_statement="", code_revision="a" * 40, cpu_tests={}, runtime={})
            receipt = {"passed": True, "tests_run": 105, "skipped": 0,
                "scope": list(subject.TEST_MODULES), "test_runtime": "gpu_build_cpu_compute_no_training"}
            with self.assertRaisesRegex(ValueError, "synthetic_test_receipt"):
                subject.train(None, None, None, None, None, None, campaign=CAMPAIGN,
                    approval_statement=subject.APPROVAL_STATEMENTS[CAMPAIGN],
                    code_revision="a" * 40, cpu_tests=receipt, runtime={})

    def test_smoke_transport_keeps_small_records_and_hashes_but_not_checkpoints(self):
        records = self.root / "records"
        records.mkdir()
        campaign = subject.open_campaign(records, {"campaign": CAMPAIGN, "code_revision": "a" * 40})
        source = self.root / "synthetic_smoke"
        job = source / "job"
        job.mkdir(parents=True)
        (job / "training_record.json").write_text('{"synthetic_training_record":true}', encoding="utf-8")
        (job / "training_job.json").write_text('{"synthetic_training_job":true}', encoding="utf-8")
        (job / "best_model.safetensors").write_bytes(b"SYNTHETIC CHECKPOINT MUST REMAIN LOCAL")
        model_id = core.REMAINING_IMAGE_IDS[0]
        smoke = {"model_id": model_id, "smoke_training_record_sha256": sha_file(job / "training_record.json")}
        (source / "encoder_smoke.json").write_text(json.dumps(smoke), encoding="utf-8")
        # Semantic smoke validation is covered by the core's real-runner fixture;
        # this fixture isolates atomic publication, identity and byte authentication.
        with patch.object(core, "validate_remaining_smoke_record", return_value=smoke) as validate:
            self.assertEqual(subject._persist_image_smoke(campaign, model_id, source), smoke)
            self.assertEqual(subject._stored_image_smokes(campaign), {model_id: smoke})
            self.assertEqual(subject._persist_image_smoke(campaign, model_id, source), smoke)
            self.assertGreater(validate.call_count, 0)
        attempts = [path for path in campaign.iterdir() if path.name.startswith(subject.IMAGE_SMOKE_PREFIX)]
        self.assertEqual(len(attempts), 1)
        self.assertEqual({path.name for path in attempts[0].iterdir()}, {
            "encoder_smoke.json", "training_record.json", "training_job.json", subject.IMAGE_SMOKE_RECEIPT})
        self.assertEqual(subject.completed_job_transports(campaign), {})
        (attempts[0] / "training_record.json").write_bytes(b"TAMPERED")
        with self.assertRaisesRegex(ValueError, "exact_completed_current_image_smoke"):
            subject._stored_image_smokes(campaign)

    def test_interrupted_smoke_publication_is_preserved_but_never_reused(self):
        records = self.root / "records"
        records.mkdir()
        campaign = subject.open_campaign(records, {"campaign": CAMPAIGN, "code_revision": "a" * 40})
        interrupted = campaign / (subject.IMAGE_SMOKE_PREFIX + "b" * 32)
        interrupted.mkdir()
        sentinel = interrupted / "training_record.json"
        sentinel.write_bytes(b"INCOMPLETE SYNTHETIC UPLOAD")
        self.assertEqual(subject._stored_image_smokes(campaign), {})
        self.assertEqual(subject.completed_job_transports(campaign), {})
        self.assertEqual(sentinel.read_bytes(), b"INCOMPLETE SYNTHETIC UPLOAD")

    def _simulated_file_sizes(self, sizes):
        original = Path.stat
        def stat(path, *args, **kwargs):
            actual = original(path, *args, **kwargs)
            if path.name not in sizes:
                return actual
            values = list(actual)
            values[6] = sizes[path.name]
            return os.stat_result(values)
        return patch.object(Path, "stat", stat)

    def test_large_vit_and_skelex_transport_uses_model_specific_bounds_without_gb_allocation(self):
        for model, size in (("image_vit_b16", 350 * 1024**2), ("image_skelex", 1318 * 1024**2)):
            with self.subTest(model=model):
                source = small_checkpoint_job(self.root / model, model)
                with self._simulated_file_sizes({name: size for name in
                        ("best_model.safetensors", "last_model.safetensors")}):
                    inventory = subject._inventory(source)
                    target = self.root / (model + "_copy")
                    subject._copy_job(source, target, inventory)
                    self.assertEqual(subject._inventory(target), inventory)
                self.assertLess(sum(path.stat().st_size for path in target.iterdir()), 4096)

    def test_larger_limits_cannot_be_used_for_dino_dense_clinical_or_unfreezing(self):
        for model, stage in (("image_dinov3_vits16", "stage_1"), ("image_densenet121", "stage_1"),
                             ("clinical_mlp", "stage_1"), ("image_vit_b16", "stage_2")):
            with self.subTest(model=model, stage=stage):
                job = small_checkpoint_job(self.root / model, model, stage=stage)
                with self._simulated_file_sizes({"best_model.safetensors": 350 * 1024**2}), \
                     self.assertRaises(ValueError):
                    subject._inventory(job)

    def test_large_checkpoint_bounds_remain_finite_and_metadata_stays_small(self):
        job = small_checkpoint_job(self.root / "job", "image_skelex")
        with self._simulated_file_sizes({"best_model.safetensors": subject.IMAGE_CHECKPOINT_LIMITS["image_skelex"] + 1}), \
             self.assertRaisesRegex(ValueError, "bounded_plain_methodology"):
            subject._inventory(job)
        with self._simulated_file_sizes({"training_job.json": 1024**2 + 1}), \
             self.assertRaisesRegex(ValueError, "bounded_methodology_job_record"):
            subject._inventory(job)

    def test_recovery_oof_names_allow_only_the_five_registered_additions(self):
        for model in core.REMAINING_IMAGE_IDS:
            (self.root / f"clinical_main__{model}__oof.json").write_text("{}", encoding="utf-8")
        self.assertEqual(subject.completed_job_transports(self.root), {})
        for model in ("image_dinov3_vits16", "image_unregistered"):
            path = self.root / f"clinical_main__{model}__oof.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "only_registered_methodology"):
                subject.completed_job_transports(self.root)
            path.unlink()

    def test_result_requires_exact_five_models_and_twenty_five_jobs(self):
        value = {"version": core.VERSION, "status": core.STATUS, "campaign": CAMPAIGN,
            "job_count": 25, "folds_completed": list(core.FOLDS), "seed": core.SEED,
            "training_plan_sha256": release.PINS[release.PLAN_PATH],
            "modeling_release_sha256": release.PINS[release.RELEASE_PATH],
            "training_executed": True, "colab_runtime_executed": True, "local_rehearsal": False,
            "job_results": [{} for _ in range(25)],
            "model_results": {f"clinical_main__{model}": {} for model in core.REMAINING_IMAGE_IDS}}
        for field in ("reserved_test_accessed", "reserved_test_predictive_evaluation_opened", "originals_modified",
            "partitions_modified", "historical_v1_1_jobs_reused", "calibration_performed", "threshold_selected",
            "final_refit_executed", "full_encoder_training_authorized", "encoder_unfreezing_executed",
            "predictive_performance_claimed", "unbiased_final_performance_claimed"):
            value[field] = False
        self.assertTrue(subject.validate_result(value, CAMPAIGN))
        broken = deepcopy(value)
        broken["job_count"] = 24
        with self.assertRaises(ValueError):
            subject.validate_result(broken, CAMPAIGN)
        broken = deepcopy(value)
        broken["model_results"]["clinical_main__image_dinov3_vits16"] = broken["model_results"].pop(
            "clinical_main__" + core.REMAINING_IMAGE_IDS[0])
        with self.assertRaisesRegex(ValueError, "exact_five_registered_image"):
            subject.validate_result(broken, CAMPAIGN)


if __name__ == "__main__":
    unittest.main()
