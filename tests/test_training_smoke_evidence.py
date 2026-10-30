"""Artificial completed35 receipts and checkpoint bytes; never train a model."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from knee import training_smoke_evidence as subject
from knee import training_smoke as historical
from knee import training_smoke_colab as colab
from knee.colab_data import sha_file
from knee.final_model_release import PINS, load_release
from knee.training_authority import digest
from test_training_smoke import fake_job, write_json


ROOT = Path(__file__).resolve().parents[1]


def evidence_fixture(directory):
    """Build a wholly synthetic completed proof; no predictor is fitted."""
    root = Path(directory)
    root.mkdir()
    request = historical.make_request("a" * 64)
    job = fake_job(root / "job", request)
    job.update(plan_sha256=PINS["configs/training_plan_v1_1.json"], selection_evidence=None,
        executed_at_utc="2026-10-23T01:00:00+00:00")
    _, catalog, _, registry = load_release(ROOT)
    gates = {key: index < 4 for index, key in enumerate(registry["eligibility_gates"])}
    job["eligibility_gates"] = gates
    engine_path = root / "job/training_record.json"
    engine = json.loads(engine_path.read_bytes())
    engine["checkpoint_sha256"] = {name: sha_file(root / "job" / name)
        for name in ("best_model.safetensors", "last_model.safetensors")}
    write_json(engine_path, engine)
    job["files_sha256"]["training_record.json"] = sha_file(engine_path)
    job["record_sha256"] = digest({key:value for key,value in job.items() if key != "record_sha256"})
    write_json(root / "job/training_job.json", job)
    spec = catalog["models"][historical.MODEL_ID]
    core = {"version": historical.VERSION, "status": historical.STATUS,
        "execution_environment": "google_colab_gpu", "colab_runtime_executed": True,
        "scope": "development_only", "stage": "stage_1", "smoke": True,
        "local_rehearsal": False, "model_id": historical.MODEL_ID,
        "validation_fold": 0, "seed": 2026, "optimizer_updates": 2, "training_knees_seen": 32,
        "micro_batch_size": 2, "effective_batch_size": 16,
        "request": request, "request_sha256": digest(request),
        "development_package_sha256": request["development_package_sha256"],
        "amp_numerical_repair_version": historical.AMP_REPAIR_VERSION,
        "amp_overflow_replays": engine["amp_overflow_replays"], "amp_scale_audit": engine["amp_scale_audit"],
        "resource_profile": engine["resource_profile"],
        **{name:True for name in ("training_authorized", "training_executed", "optimizer_created",
            "backward_executed", "participant_data_opened", "training_smoke_test_passed",
            "gpu_resource_profile_recorded", "clinical_preprocessing_fitted_inside_training_folds_only",
            "frozen_encoder_state_unchanged", "task_head_update_verified")},
        **{name:False for name in ("full_training_authorized", "full_training_executed",
            "full_encoder_training_authorized", "encoder_unfreezing_executed", "eligible_for_model_selection",
            "predictive_performance_claimed", "reserved_test_predictive_evaluation_opened",
            "reserved_test_accessed", "originals_modified", "partitions_modified")},
        "encoder_tensors_compared": 725, "model_input_reference_sha256": "b" * 64,
        "previous_colab34_preparation_record_sha256": "c" * 64,
        "previous_colab33_checkpoint_record_sha256": historical.PREVIOUS_CHECKPOINT_SHA256,
        "historical_training_source_files_sha256": historical.SOURCE_34,
        "release_component_canonical_sha256": PINS, "source_files_sha256": subject.SOURCE_FILES,
        "numerical_repair_source_lineage": {name:{"historical_sha256": historical.SOURCE_34[name],
            "runtime_sha256": value, "repair_version": historical.AMP_REPAIR_VERSION}
            for name,value in historical.RUNTIME_SOURCE_PATCHES.items()},
        "training_job_record_sha256": job["record_sha256"],
        "artifact_files_sha256": {"job/" + path.name: sha_file(path) for path in (root / "job").iterdir()},
        "eligibility_gates_before_smoke": gates,
        "public_checkpoint_files_sha256": {item["name"]:item["sha256"] for item in spec["files"]},
        "eligibility_evidence": {"model_id": historical.MODEL_ID, "scope": "private_noncommercial_academic_research",
            "catalog_sha256": PINS["configs/pretrained_weights_v1_1.json"],
            "license_review": spec["license_review"], "overlap_review": spec["overlap_review"],
            "pretraining_subject_disjointness_proven": False,
            "preprocessing_localizer_oai_overlap_limitation_remains": True,
            "normalization_and_loader_evidence": "authenticated_colab32_reference_and_colab34_frozen_sources"},
        "executed_at_utc": "2026-10-23T01:01:00+00:00"}
    write_json(root / colab.CORE_FILENAME, core)
    summary = {**core, "code_revision": subject.SMOKE_REVISION,
        "result_files_saved_and_readback_verified": True, "historical_cpu_preparation_authenticated": True,
        "historical_cpu_preparation_suite_reexecuted": False,
        "smoke_record_sha256": sha_file(root / colab.CORE_FILENAME),
        "cpu_tests": {"passed": True, "tests_run": 352, "skipped": 0,
            "scope": list(colab.TEST_MODULES), "test_runtime": "gpu_build_cpu_compute_no_training",
            "historical_cpu_preparation_suite_reexecuted": False},
        "executed_at_utc": "2026-10-23T01:02:00+00:00"}
    write_json(root / colab.SUMMARY_FILENAME, summary)
    return root


class CompletedSmokeEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="completed35_evidence_synthetic_")
        self.addCleanup(self.temp.cleanup)
        self.root = evidence_fixture(Path(self.temp.name) / "proof")

    def edit_summary(self, mutate):
        path = self.root / colab.SUMMARY_FILENAME
        value = json.loads(path.read_bytes())
        mutate(value)
        write_json(path, value)

    def edit_core(self, mutate):
        path = self.root / colab.CORE_FILENAME
        value = json.loads(path.read_bytes())
        mutate(value)
        write_json(path, value)
        self.edit_summary(lambda summary: summary.update({
            **{key:item for key,item in value.items() if key != "executed_at_utc"},
            "smoke_record_sha256": sha_file(path)}))

    def edit_job(self, *, engine_change=None, job_change=None):
        root = self.root / "job"
        engine = json.loads((root / "training_record.json").read_bytes())
        job = json.loads((root / "training_job.json").read_bytes())
        if engine_change:
            engine_change(engine)
            write_json(root / "training_record.json", engine)
        job["files_sha256"] = {name:sha_file(root / name) for name in historical.JOB_FILES}
        if job_change:
            job_change(job)
        job["record_sha256"] = digest({key:value for key,value in job.items() if key != "record_sha256"})
        write_json(root / "training_job.json", job)
        self.edit_core(lambda core: core.update(training_job_record_sha256=job["record_sha256"],
            artifact_files_sha256={"job/" + name:sha_file(root / name)
                for name in historical.JOB_FILES | {"training_job.json"}}))

    def test_intact_pair_and_six_hashed_job_files_authenticate_without_training(self):
        with patch("knee.training_runner.execute", side_effect=AssertionError("training forbidden")), \
                patch("knee.training_smoke._verify_frozen_encoder", side_effect=AssertionError("checkpoint loading forbidden")):
            value = subject.validate_smoke(self.root, ROOT)
        self.assertEqual(value["cpu_tests"]["tests_run"], 352)
        self.assertEqual(value["executed_at_utc"], "2026-10-23T01:02:00+00:00")
        self.assertFalse(value["full_training_authorized"] or value["eligible_for_model_selection"])

    def test_repository_argument_is_optional_and_does_not_change_proof(self):
        self.assertEqual(subject.validate_smoke(self.root), subject.validate_smoke(self.root, ROOT))

    def test_raw_core_whitespace_tampering_is_rejected(self):
        path = self.root / colab.CORE_FILENAME
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "exact_core_receipt"):
            subject.validate_smoke(self.root)

    def test_missing_and_extra_root_or_job_files_are_rejected(self):
        for index, target in enumerate(("unexpected.json", "job/unexpected.json")):
            extra = self.root / target
            extra.write_bytes(b"{}")
            with self.subTest(target=target), self.assertRaises(ValueError):
                subject.validate_smoke(self.root)
            extra.unlink()
        (self.root / "job/last_model.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "six_smoke_job"):
            subject.validate_smoke(self.root)

    def test_checkpoint_is_stream_hashed_but_never_deserialized(self):
        path = self.root / "job/best_model.safetensors"
        path.write_bytes(path.read_bytes() + b"CHANGED")
        with self.assertRaisesRegex(ValueError, "artifact_bytes_changed"):
            subject.validate_smoke(self.root)

    def test_resealed_changed_test_count_skips_revision_and_scope_fail(self):
        original = (self.root / colab.SUMMARY_FILENAME).read_bytes()
        for edit in (lambda v:v.update(code_revision="f"*40),
            lambda v:v["cpu_tests"].update(tests_run=351), lambda v:v["cpu_tests"].update(tests_run=True),
            lambda v:v["cpu_tests"].update(skipped=1), lambda v:v["cpu_tests"].update(scope=[]),
            lambda v:v.update(historical_cpu_preparation_authenticated=False)):
            (self.root / colab.SUMMARY_FILENAME).write_bytes(original)
            self.edit_summary(edit)
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                subject.validate_smoke(self.root)

    def test_scope_and_bool_substitution_cannot_be_resealed_as_passed(self):
        changes = ({"full_training_authorized": True}, {"encoder_unfreezing_executed": True},
            {"reserved_test_accessed": True}, {"scope": "reserved_test"}, {"local_rehearsal": True},
            {"training_authorized": False}, {"encoder_tensors_compared": 724},
            {"eligible_for_model_selection": True}, {"optimizer_updates": True})
        for index, changed in enumerate(changes):
            self.root = evidence_fixture(Path(self.temp.name) / str(index))
            self.edit_core(lambda value: value.update(changed))
            with self.subTest(changed=changed), self.assertRaises((ValueError, RuntimeError)):
                subject.validate_smoke(self.root)

    def test_source_release_chain_and_license_not_silently_redefined(self):
        edits = (lambda v:v.update(source_files_sha256={}),
            lambda v:v.update(historical_training_source_files_sha256={}),
            lambda v:v.update(previous_colab33_checkpoint_record_sha256="f"*64),
            lambda v:v.update(release_component_canonical_sha256={}),
            lambda v:v.update(numerical_repair_source_lineage={}),
            lambda v:v.update(public_checkpoint_files_sha256={}),
            lambda v:v["eligibility_evidence"].update(pretraining_subject_disjointness_proven=True),
            lambda v:v["eligibility_evidence"].update(scope="commercial"))
        for index, edit in enumerate(edits):
            self.root = evidence_fixture(Path(self.temp.name) / str(index))
            self.edit_core(edit)
            with self.subTest(index=index), self.assertRaises(ValueError):
                subject.validate_smoke(self.root)

    def test_summary_core_disagreement_and_invalid_timestamps_fail(self):
        original = (self.root / colab.SUMMARY_FILENAME).read_bytes()
        for value in (None, "invalid", "2026-10-23T01:03:00", "2026-10-23T00:00:00+00:00"):
            (self.root / colab.SUMMARY_FILENAME).write_bytes(original)
            self.edit_summary(lambda v:v.update(executed_at_utc=value))
            with self.subTest(value=value), self.assertRaises(ValueError):
                subject.validate_smoke(self.root)
        (self.root / colab.SUMMARY_FILENAME).write_bytes(original)
        self.edit_summary(lambda v:v.update(model_input_reference_sha256="f"*64))
        with self.assertRaisesRegex(ValueError, "content_mismatch"):
            subject.validate_smoke(self.root)

    def test_current_scientific_source_drift_rejected_without_rewriting_history(self):
        real = subject.sha_file
        def changed(path):
            return "0"*64 if Path(path) == ROOT / "src/knee/training_engine.py" else real(path)
        with patch.object(subject, "sha_file", side_effect=changed), \
                self.assertRaisesRegex(ValueError, "runtime_sources_changed"):
            subject.validate_smoke(self.root, ROOT)

    def test_resealed_engine_cannot_unfreeze_change_amp_or_claim_cpu_fallback(self):
        changes = (lambda v:v["history"][0].update(phase="full"),
            lambda v:v.update(device="cpu"),
            lambda v:v["amp_scale_audit"][0].update(backoffs=17),
            lambda v:v.update(checkpoint_sha256={}),
            lambda v:v.update(resource_profile={}))
        for index, mutate in enumerate(changes):
            self.root = evidence_fixture(Path(self.temp.name) / str(index))
            self.edit_job(engine_change=mutate)
            with self.subTest(index=index), self.assertRaises((ValueError, RuntimeError)):
                subject.validate_smoke(self.root)

    def test_resealed_job_cannot_promote_smoke_change_plan_or_reverse_execution_order(self):
        changes = (lambda v:v.update(eligible_for_model_selection=True),
            lambda v:v.update(plan_sha256="f"*64),
            lambda v:v.update(selection_evidence={"promote":True}),
            lambda v:v.update(executed_at_utc="2026-10-23T02:00:00+00:00"))
        for index, mutate in enumerate(changes):
            self.root = evidence_fixture(Path(self.temp.name) / str(index))
            self.edit_job(job_change=mutate)
            with self.subTest(index=index), self.assertRaises((ValueError, RuntimeError)):
                subject.validate_smoke(self.root)

    def test_request_bool_fold_and_recomputed_digest_still_rejected(self):
        def mutate(value):
            value["request"]["validation_fold"] = False
            value["request_sha256"] = digest(value["request"])
        self.edit_core(mutate)
        with self.assertRaisesRegex(ValueError, "fixed_development_smoke_request"):
            subject.validate_smoke(self.root)

    def test_duplicate_json_key_is_rejected_before_authentication(self):
        path = self.root / colab.CORE_FILENAME
        original = path.read_bytes()
        path.write_bytes(b'{"scope":"development_only",' + original.lstrip()[1:])
        with self.assertRaisesRegex(ValueError, "duplicate_preparation_record_key"):
            subject.validate_smoke(self.root)


if __name__ == "__main__":
    unittest.main()
