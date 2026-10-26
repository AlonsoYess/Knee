"""Synthetic private preparation artifacts only; never fit a predictor."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from knee import clinical_scenarios as clinical
from knee import methodology_preparation as subject
from knee import methodology_release as release
from knee.aligned_training import data
from knee.aligned_training import runner
from knee.colab_data import sha_file
from knee.aligned_training.authority import validate_request
from test_training_data import reference_fixture

ROOT = Path(__file__).resolve().parents[1]


class MethodologyPreparationTests(unittest.TestCase):
    def fixture(self, directory):
        base = Path(directory)
        sources = base / "sources"
        sources.mkdir()
        fixture = reference_fixture(sources)
        fixture["projection"] = [{**{key: row[key] for key in ("src_subject_id", "side", "AGEYEARS", "SEX", "BMI")},
            "KL_BASE": 2 + i % 2, "CIRUGIA_PREVIA": None if i % 7 == 0 else i % 2,
            "WOMAC_TOTAL": None if i % 6 == 0 else float(i % 30)} for i, row in enumerate(fixture["rows"])]
        fixture["projection_path"] = sources / "synthetic_development_baseline_projection.json"
        runner.save_json(fixture["projection_path"], fixture["projection"])
        fixture["projection_sha256"] = sha_file(fixture["projection_path"])
        fixture["output"] = base / "prepared"
        return fixture

    def prepare(self, fixture, **kwargs):
        with patch("sklearn.linear_model.LogisticRegression.fit", side_effect=AssertionError("Estimator fitting forbidden")), \
             patch("xgboost.XGBClassifier.fit", side_effect=AssertionError("Estimator fitting forbidden")), \
             patch("torch.optim.AdamW", side_effect=AssertionError("Optimizer construction forbidden")), \
             patch("torch.Tensor.backward", side_effect=AssertionError("Backpropagation forbidden")), \
             patch.object(runner, "execute", side_effect=AssertionError("Training runner invocation forbidden")):
            return subject.prepare(ROOT, fixture["package"], fixture["reference_path"],
                kwargs.get("projection_path", fixture["projection_path"]),
                kwargs.get("projection_sha256", fixture["projection_sha256"]),
                kwargs.get("output", fixture["output"]),
                expected_package_sha256=kwargs.get("package_sha256", fixture["package_sha256"]))

    def test_complete_three_scenario_five_fold_preparation_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            original_archive = sha_file(fixture["archive"])
            result = self.prepare(fixture)
            self.assertEqual(result["status"], subject.STATUS)
            self.assertEqual(result["development_participants"], 15)
            self.assertEqual(result["development_knees"], 30)
            self.assertEqual(result["clinical_preprocessing_states_verified"], 15)
            self.assertEqual(result["prepared_clinical_requests"], 45)
            self.assertEqual(result["execution_authorities_created"], 0)
            for name in ("training_authorized", "training_executed", "optimizer_created", "backward_executed",
                         "reserved_test_predictive_evaluation_opened", "originals_modified", "partitions_modified",
                         "new_studies_selected", "image_augmentation_applied"):
                self.assertIs(result[name], False)
            self.assertEqual(runner.read_json(fixture["output"] / subject.FILENAME), result)
            self.assertEqual(sha_file(fixture["archive"]), original_archive)
            self.assertEqual(sha_file(fixture["projection_path"]), fixture["projection_sha256"])
            self.assertEqual(len(result["artifacts_sha256"]), 121)
            for name, checksum in result["artifacts_sha256"].items():
                self.assertEqual(sha_file(fixture["output"] / name), checksum)

    def test_prepared_states_matrices_and_requests_match_actual_core_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            self.prepare(fixture)
            _, _, plan, registry = release.load_release(ROOT)
            sidecar = runner.read_json(fixture["output"] / "baseline_sidecar.json")
            enriched = clinical.join_baseline_sidecar(fixture["rows"], sidecar, fixture["package_sha256"])
            for scenario, features in clinical.SCENARIOS.items():
                rows = fixture["rows"] if scenario == "clinical_main" else enriched
                for fold in range(5):
                    prefix = fixture["output"] / "clinical" / scenario
                    state = runner.read_json(prefix / f"fold_{fold}_state.json")
                    self.assertTrue(clinical.verify_scenario_fold_state(rows, state))
                    for split in ("training", "validation"):
                        selected = [row for row in rows if (row["cv_fold"] == fold) == (split == "validation")]
                        matrix = np.load(prefix / f"fold_{fold}_{split}.npy", allow_pickle=False)
                        np.testing.assert_array_equal(matrix, clinical.transform_scenario_rows(selected, state))
                        self.assertEqual(matrix.shape[1], len(features))
                        self.assertEqual(runner.read_json(prefix / f"fold_{fold}_{split}_order.json"),
                                         [[row["src_subject_id"], row["side"]] for row in selected])
                    for model_id in subject.CLINICAL_IDS:
                        job = runner.read_json(fixture["output"] / "requests" / scenario / model_id / f"fold_{fold}.json")
                        self.assertEqual(validate_request(job, plan, registry), job)
                        self.assertEqual(job["clinical_scenario"], scenario)
                        self.assertEqual(job["validation_fold"], fold)
                        self.assertNotIn("training_authorized", job)
                        self.assertEqual(job["baseline_sidecar_sha256"],
                            None if scenario == "clinical_main" else sidecar["record_sha256"])

    def test_external_projection_hash_rejected_before_development_package_access(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            with patch.object(data, "load_sealed_inputs", side_effect=AssertionError("No data access before source pin")):
                with self.assertRaisesRegex(ValueError, "external_baseline_projection_hash_mismatch"):
                    self.prepare(fixture, projection_sha256="0" * 64)
            self.assertFalse(fixture["output"].exists())

    def test_wrong_package_hash_is_rejected_and_no_preparation_written(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            with self.assertRaisesRegex(ValueError, "package_does_not_match"):
                self.prepare(fixture, package_sha256="0" * 64)
            self.assertFalse(fixture["output"].exists())

    def test_baseline_projection_cannot_add_unknown_reserved_or_future_fields(self):
        for mutation in ("unknown", "outcome", "future", "missing", "duplicate"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                fixture = self.fixture(directory)
                rows = deepcopy(fixture["projection"])
                if mutation == "unknown":
                    rows[0]["src_subject_id"] = "reserved_or_unknown"
                elif mutation == "outcome":
                    rows[0]["PROGRESION_48M"] = 1
                elif mutation == "future":
                    rows[0]["KL_48"] = 4
                elif mutation == "missing":
                    rows.pop()
                else:
                    rows[0] = rows[1]
                alternate = fixture["projection_path"].parent / "invalid_projection.json"
                runner.save_json(alternate, rows)
                with self.assertRaises(ValueError):
                    self.prepare(fixture, projection_path=alternate, projection_sha256=sha_file(alternate))
                self.assertFalse(fixture["output"].exists())

    def test_output_cannot_overlap_sources_or_overwrite_previous_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            for output in (fixture["package"] / "output", fixture["projection_path"].parent / "output",
                           ROOT / "output", Path(directory)):
                with self.assertRaisesRegex(ValueError, "new_separate"):
                    self.prepare(fixture, output=output)
            self.prepare(fixture)
            with self.assertRaisesRegex(ValueError, "new_separate"):
                self.prepare(fixture)

    def test_source_change_during_preparation_prevents_completed_record(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            real_write = subject._write_matrix
            changed = False
            def mutate_source(path, matrix):
                nonlocal changed
                real_write(path, matrix)
                if not changed:
                    with fixture["projection_path"].open("ab") as stream:
                        stream.write(b" ")
                    changed = True
            with patch.object(subject, "_write_matrix", side_effect=mutate_source):
                with self.assertRaisesRegex(ValueError, "changed_during_preparation"):
                    self.prepare(fixture)
            self.assertFalse((fixture["output"] / subject.FILENAME).exists())

    def test_nonfinite_or_duplicate_json_projection_fails_closed(self):
        for raw in (b'[{"BMI":NaN}]', b'[{"BMI":25,"BMI":26}]'):
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as directory:
                fixture = self.fixture(directory)
                alternate = fixture["projection_path"].parent / "invalid_json.json"
                alternate.write_bytes(raw)
                with self.assertRaises(ValueError):
                    self.prepare(fixture, projection_path=alternate, projection_sha256=sha_file(alternate))
                self.assertFalse(fixture["output"].exists())


if __name__ == "__main__":
    unittest.main()
