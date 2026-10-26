"""Synthetic persistence/recovery and scope gates; never train participant data."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from knee import methodology_campaign as core
from knee import methodology_campaign_colab as subject
from knee import methodology_release as release
from knee.aligned_training import runner
from test_methodology_campaign import synthetic_job, synthetic_preflight

ROOT = Path(__file__).resolve().parents[1]


class MethodologyCampaignColabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.threads)

    def test_setup_and_environment_dispatch_to_correct_existing_runtime(self):
        with patch.object(subject.inputs, "setup_runtime", return_value={"synthetic": "cpu"}) as cpu, \
             patch("knee.training_gpu_runtime.setup_runtime", return_value={"synthetic": "gpu"}) as gpu:
            self.assertEqual(subject.setup_runtime(ROOT, "destination", "clinical"), {"synthetic": "cpu"})
            self.assertEqual(subject.setup_runtime(ROOT, "destination", "densenet_frozen"), {"synthetic": "gpu"})
            self.assertEqual(cpu.call_count, 1)
            self.assertEqual(gpu.call_count, 1)
        with self.assertRaises(ValueError):
            subject.runtime_environment(ROOT, "reserved_test")

    def test_new_campaign_manifest_reuses_only_exact_same_recipe_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = {"campaign": "clinical", "code_revision": "a" * 40,
                "training_plan_sha256": release.PINS[release.PLAN_PATH]}
            first = subject.open_campaign(root, identity)
            self.assertEqual(subject.open_campaign(root, identity), first)
            changed = {**identity, "code_revision": "b" * 40}
            self.assertNotEqual(subject.open_campaign(root, changed), first)

    def test_completed_job_transport_is_atomic_immutable_and_recoverable(self):
        with tempfile.TemporaryDirectory() as directory:
            value, request, record, job = synthetic_job(directory, scenario="clinical_plus_baseline_kl")
            persistent = Path(directory) / "persistent"
            persistent.mkdir()
            key = core.job_key(request)
            target = subject.persist_completed_job(persistent, key, job, record)
            self.assertEqual(subject.completed_job_transports(persistent)[key]["directory"], target)
            self.assertEqual(subject.persist_completed_job(persistent, key, job, record), target)
            self.assertEqual(subject._inventory(target), subject._inventory(job))
            (target / "clinical_model.json").write_bytes(b"CORRUPTED")
            with self.assertRaises(ValueError):
                subject.completed_job_transports(persistent)

    def test_incomplete_attempt_is_not_reused_or_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = root / ("job_attempt_" + "a" * 32)
            attempt.mkdir()
            partial = attempt / "unverified.part"
            partial.write_bytes(b"SYNTHETIC INTERRUPTED UPLOAD")
            self.assertEqual(subject.completed_job_transports(root), {})
            self.assertTrue(partial.is_file())

    def test_legacy_job_is_rejected_even_with_resealed_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            _, request, record, job = synthetic_job(directory)
            root = Path(directory) / "persistent"
            root.mkdir()
            key = core.job_key(request)
            target = subject.persist_completed_job(root, key, job, record)
            value = runner.read_json(target / runner.JOB_FILENAME)
            value["plan_sha256"] = "f" * 64
            (target / runner.JOB_FILENAME).unlink()
            runner.save_json(target / runner.JOB_FILENAME, value)
            receipt_path = target.parent / subject.JOB_RECEIPT
            receipt = runner.read_json(receipt_path)
            receipt["files_sha256"] = subject._inventory(target)
            receipt_path.unlink()
            runner.save_json(receipt_path, receipt)
            with self.assertRaisesRegex(ValueError, "only_exact_v1_2_jobs_may_be_recovered"):
                subject.completed_job_transports(root)

    def test_confirmation_is_checked_before_any_campaign_created(self):
        with patch.object(subject, "open_campaign", side_effect=AssertionError("Campaign forbidden")), \
             patch.object(core, "run", side_effect=AssertionError("Training forbidden")), \
             self.assertRaisesRegex(ValueError, "future_investigator_confirmation_required"):
            subject.train("missing", "missing", "missing", "missing", "missing", "missing",
                campaign="clinical", approval_statement="", code_revision="a" * 40, cpu_tests={}, runtime={})

    def test_incomplete_test_or_runtime_receipts_stop_before_reading_data(self):
        with patch.object(subject, "campaign_identity", side_effect=AssertionError("Data forbidden")), \
             self.assertRaisesRegex(ValueError, "synthetic_test_receipt_required"):
            subject.train("missing", "missing", "missing", "missing", "missing", "missing",
                campaign="clinical", approval_statement=subject.APPROVAL_STATEMENTS["clinical"],
                code_revision="a" * 40, cpu_tests={}, runtime={})

    def test_locate_checks_only_completed38_and_not_reserved_archives(self):
        with tempfile.TemporaryDirectory() as directory:
            value = synthetic_preflight(directory)
            unrelated = value["records"] / "reserved_test_package.zip"
            unrelated.write_bytes(b"FORBIDDEN TEST SENTINEL; MUST NOT BE READ")
            with patch("builtins.open", wraps=open) as opened:
                self.assertEqual(subject.find_preparation(value["records"]), value["preflight"])
            self.assertNotIn(unrelated, [call.args[0] for call in opened.call_args_list if call.args])

    def test_locate_rejects_old34_36_37_only_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("training_preparation_check_old", "densenet_fivefold_campaign_old", "clinical_fivefold_campaign_old"):
                (root / name).mkdir()
            with self.assertRaisesRegex(ValueError, "completed_real_colab38_preflight_not_found"):
                subject.find_preparation(root)

    def test_historical_gpu_smoke_is_mandatory_before_image_training(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(core, "run", side_effect=AssertionError("Training forbidden")), \
             self.assertRaisesRegex(ValueError, "completed_actual_densenet_gpu35_smoke_required"):
            subject._gpu_evidence(Path(directory), ROOT)

    def test_original_gpu_smoke_can_be_adopted_without_loading_weights_or_training(self):
        import test_training_smoke_evidence as fixture_module
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # The fixture only needs catalog/registry; use the same unchanged
            # release with verified LF/CRLF portability, not the old byte-only reader.
            with patch.object(fixture_module, "load_release", side_effect=release.load_release):
                fixture_module.evidence_fixture(root / ("training_smoke_check_" + "a" * 32))
            with patch("knee.aligned_training.runner.execute", side_effect=AssertionError("Training forbidden")), \
                 patch("knee.aligned_training.models.build_model", side_effect=AssertionError("Model forbidden")), \
                 patch("torch.optim.AdamW", side_effect=AssertionError("Optimizer forbidden")), \
                 patch("torch.Tensor.backward", side_effect=AssertionError("Backward forbidden")):
                result = subject._gpu_evidence(root, ROOT)
            self.assertTrue(result["image_eligibility_verified"])
            self.assertEqual(result["development_package_sha256"], "a" * 64)
            self.assertEqual(result["historical_smoke_scope"], "same_encoder_runtime_only_not_v1_2_augmentation_efficacy")

    def test_publication_uses_staging_and_keeps_existing_artifact_immutable(self):
        from knee.colab_data import sha_file
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_bytes(b'{"synthetic":true}\n')
            target = root / "target.json"
            checksum = sha_file(source)
            subject._publish_copy(source, target, checksum)
            self.assertEqual(sha_file(target), checksum)
            subject._publish_copy(source, target, checksum)
            target.write_bytes(b"DIFFERENT")
            with self.assertRaisesRegex(ValueError, "cannot_be_overwritten"):
                subject._publish_copy(source, target, checksum)

    def test_historical_source_portability_accepts_only_registered_lf_crlf_bytes(self):
        from knee.training_smoke_evidence import SOURCE_FILES
        name = "src/knee/training_authority.py"
        pinned = SOURCE_FILES[name]
        original = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / name
            path.parent.mkdir(parents=True)
            for raw in (original, original.replace(b"\n", b"\r\n")):
                path.write_bytes(raw)
                self.assertTrue(subject._historical_source_matches(root, name, pinned))
            for raw in (original + b"# Changed body\n", original.replace(b"\n", b"\r", 1), b"\xff"):
                path.write_bytes(raw)
                self.assertFalse(subject._historical_source_matches(root, name, pinned))
            with self.assertRaises(ValueError):
                subject._historical_source_matches(root, "private.csv", pinned)
