"""Closed historical diagnostic boundaries, using synthetic data only."""

import importlib.util
import json
import os
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

DEPENDENCIES = all(importlib.util.find_spec(name) for name in ("cv2", "scipy"))
if DEPENDENCIES:
    import test_roi_mcr004_review as fixtures
    from knee import roi_mcr004_diagnostic as diagnostic
    from knee.roi_mcr004 import candidate_for_half
    from knee.dicom_audit import sha256_file
    from knee.roi_mcr004_review import close_review, _source_studies
    from test_roi_mcr004_diagnostic_trace import synthetic_knee


@unittest.skipUnless(DEPENDENCIES, "MCR004 isolated environment required")
class Mcr004DiagnosticTests(unittest.TestCase):
    def setUp(self):
        # Existing closure fixture helper; no TestCase class is imported into this
        # module's discovery namespace. The closure itself is executed once.
        self.fixture = fixtures.Mcr004ReviewTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.config = self.fixture.config
        self.root, self.historical = self.fixture.root, self.fixture.output
        self.config.update(source_dir=self.root/"source", bilateral_output_dir=self.root/"bilateral",
            pilot_audit_json=self.root/"pilot"/"audit.json",
            v04_review_summary_json=self.root/"v04"/"summary.json",
            v04_closure_record_json=self.root/"v04"/"record.json",
            v04_review_csv=self.root/"v04"/"review.csv",
            _diagnostic_config_file=self.root/"config.json")
        self.relative_output = "outputs/diagnostic_v1"
        self.output = self.root/self.relative_output
        self.env = patch.dict(os.environ, {"KNEE_DATA_ROOT": str(self.root)})
        self.env.start()
        self.addCleanup(self.env.stop)
        half = synthetic_knee()
        pixels = np.concatenate((half, half), axis=1)
        studies = []
        audit_entries = []
        for study, record, dataset, _ in self.fixture.studies:
            study = {**study, "split_column": "600"}
            studies.append((study, record, dataset, pixels))
            package = self.config["source_dir"]/record["package_relative_path"]
            package.parent.mkdir(parents=True, exist_ok=True)
            package.write_bytes(b"synthetic archived input; not DICOM")
            audit_entries.append({**record, "manifest_key": study["manifest_key"],
                "package_sha256": sha256_file(package)})
        self.studies = studies
        self.references = {}
        for row in self.fixture.results:
            offset = 0 if row["patient_side"] == "RIGHT" else 600
            candidate = candidate_for_half(half, row["patient_side"], "MONOCHROME2", offset, .15, .2)
            self.assertEqual(candidate["status"], "candidate")
            self.references[row["knee_alias"]] = candidate
            row.update(half_x0=str(offset), half_x1=str(offset+600),
                native_dicom_box=json.dumps(candidate["native_dicom_box"]),
                requested_working_box=json.dumps(candidate["requested_working_box"]),
                trace_json=json.dumps(candidate["trace"]),
                native_crop_sha256=diagnostic.sha256_bytes(candidate["crop"].tobytes()),
                crop_rows=str(candidate["crop_rows"]), crop_columns=str(candidate["crop_columns"]),
                crop_height_mm=str(candidate["crop_height_mm"]), crop_width_mm=str(candidate["crop_width_mm"]))
            np.save(self.historical/row["native_crop_file"], candidate["crop"], allow_pickle=False)
        self.fixture.studies = studies
        self.fixture.review = self.fixture._all_accepted()
        for index in (1, 6, 15):
            self.fixture.review[index].update(decision="aceptable_con_advertencia_periferica",
                peripheral_warning_yes_no="SI")
        self.fixture.review[10].update(decision="rechazado", critical_contamination_yes_no="SI")
        self.fixture.review[7].update(decision="no_evaluable", critical_contamination_yes_no="")
        self.fixture.review[-1] = dict(self.fixture.template[-1])
        self.fixture._save_fixture()
        response = self.fixture._response()
        response["updates"][self.fixture.pending].update(decision="no_evaluable",
            critical_contamination_yes_no="", reviewer_notes="Synthetic: mark origin unresolved.")
        with patch("knee.roi_mcr004_review._source_studies", return_value=iter(studies)):
            self.closed = close_review(self.config, response, "a"*40)
        self.assertEqual(self.closed["acceptable_crops"], 17)
        for key in ("v04_review_summary_json", "v04_closure_record_json", "v04_review_csv", "_diagnostic_config_file"):
            path = self.config[key]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
        self.config["bilateral_output_dir"].mkdir()
        for name in ("cierre_paso_3_publico.json", "parametros_congelados.json", "resultados_separacion_privados.csv"):
            (self.config["bilateral_output_dir"]/name).write_text("synthetic", encoding="utf-8")
        self.config["pilot_audit_json"].parent.mkdir(parents=True)
        fixtures.write_json(self.config["pilot_audit_json"], {"public_summary": {"status": "ok"},
            "private_reconciliation": {"selected_packages": audit_entries}})

    def run_diagnostic(self, generator=None):
        with patch.object(diagnostic, "_source_studies", side_effect=(
                generator if generator else lambda config: iter(self.studies))):
            return diagnostic.run_diagnostic(self.config, self.relative_output, "b"*40)

    def test_closed_context_preserves_rejection_and_saved_draft_anchored_window(self):
        context = diagnostic.read_closed_context(self.config)
        self.assertEqual(context["metrics"]["acceptable_crops"], 17)
        self.assertEqual(context["metrics"]["incorrect_candidates"], 3)
        self.assertEqual(context["metrics"]["acceptable_with_peripheral_warning"], 3)
        self.assertEqual(len(context["results"]), 20)
        view = context["views"][self.fixture.pending]
        self.assertEqual(view["review_csv_sha256"], self.closed["assisted_draft_csv_sha256"])
        self.assertNotEqual(view["review_csv_sha256"], self.closed["review_csv_sha256"])
        self.assertEqual(diagnostic._snapshot([Path(p) for p in context["input_snapshot"]]), context["input_snapshot"])
        self.assertFalse(self.output.exists())

    def test_tampered_closed_artifacts_each_stop_without_reopening_review(self):
        for name in (diagnostic.SUMMARY_FILE, "revision_tecnica_ciega.csv",
                "resultados_roi_privados.csv", "revision_tecnica_ciega_borrador_asistido.csv",
                "procedencia_revision_asistida.json", "respuestas_investigador.json"):
            file = self.historical/name
            original = file.read_bytes()
            with self.subTest(name=name):
                file.write_bytes(original.replace(b"true", b"false", 1) if name == "respuestas_investigador.json" else original+b"\n")
                with self.assertRaises(ValueError):
                    diagnostic.read_closed_context(self.config)
                file.write_bytes(original)

    def test_partial_closure_stops_before_creating_diagnostic_output(self):
        file = self.historical/diagnostic.RECORD_FILE
        file.unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_diagnostic()
        self.assertFalse(self.output.exists())

    def _prepared_mode_fixture(self):
        """Closed, content-addressed assistance like notebook14; synthetic only."""
        summary = diagnostic._read_json(self.historical/diagnostic.SUMMARY_FILE)
        record = diagnostic._read_json(self.historical/diagnostic.RECORD_FILE)
        response = record["response"]
        csv_file, prov_file = "revision_tecnica_asistida_completada.csv", "procedencia_revision_asistida_completada.json"
        (self.historical/csv_file).write_bytes((self.historical/"revision_tecnica_ciega.csv").read_bytes())
        provenance = {"draft_csv_sha256": summary["assisted_draft_csv_sha256"],
            "completed_csv_sha256": summary["review_csv_sha256"],
            "original_assisted_provenance_sha256": record["assisted_provenance_sha256"],
            "window_evidence": {alias: update["view"] for alias, update in response["updates"].items()}}
        fixtures.write_json(self.historical/prov_file, provenance)
        fingerprint = {"csv_file": csv_file, "csv_sha256": sha256_file(self.historical/csv_file),
            "provenance_file": prov_file, "provenance_sha256": sha256_file(self.historical/prov_file)}
        response["assisted_completion"] = fingerprint
        summary["review_entry_mode"] = "codex_prepared"
        fixtures.write_json(self.historical/diagnostic.SUMMARY_FILE, summary)
        record.update(summary)
        record.update(summary_sha256=sha256_file(self.historical/diagnostic.SUMMARY_FILE),
            completed_assistance=fingerprint, response=response)
        fixtures.write_json(self.historical/"respuestas_investigador.json", response)
        fixtures.write_json(self.historical/diagnostic.RECORD_FILE, record)
        return provenance, record

    def test_completed_assistance_closed_csv_and_windows_are_content_anchored(self):
        self._prepared_mode_fixture()
        context = diagnostic.read_closed_context(self.config)
        self.assertEqual(context["summary"]["review_entry_mode"], "codex_prepared")
        self.assertIn(self.fixture.pending, context["views"])
        path = self.historical/"revision_tecnica_asistida_completada.csv"
        path.write_bytes(path.read_bytes()+b"\n")
        with self.assertRaisesRegex(ValueError, "completed_assistance_hash_mismatch"):
            diagnostic.read_closed_context(self.config)

    def test_lost_window_even_with_rehashed_completed_manifest_blocks(self):
        provenance, record = self._prepared_mode_fixture()
        provenance["window_evidence"] = {}
        path = self.historical/record["completed_assistance"]["provenance_file"]
        fixtures.write_json(path, provenance)
        fingerprint = {**record["completed_assistance"], "provenance_sha256": sha256_file(path)}
        record["completed_assistance"] = fingerprint
        record["response"]["assisted_completion"] = fingerprint
        fixtures.write_json(self.historical/diagnostic.RECORD_FILE, record)
        fixtures.write_json(self.historical/"respuestas_investigador.json", record["response"])
        with self.assertRaisesRegex(ValueError, "window_evidence_missing_or_changed"):
            diagnostic.read_closed_context(self.config)

    def test_existing_and_overlapping_or_escaping_outputs_are_never_used(self):
        for relative in ("outputs", "../escape", "output", "source/diagnostic"):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                diagnostic.resolve_diagnostic_output(self.config, relative)
        self.output.mkdir(parents=True)
        (self.output/"preserved.txt").write_text("original", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.run_diagnostic()
        self.assertEqual((self.output/"preserved.txt").read_text(encoding="utf-8"), "original")

    def test_exact_synthetic_replay_checks_all20_and_does_not_regrade_or_rewrite(self):
        before = diagnostic._snapshot([file for file in self.historical.rglob("*") if file.is_file()])
        summary = self.run_diagnostic()
        self.assertEqual(summary["exact_replayed_candidates"], 20)
        self.assertEqual(summary["instrumented_equivalence_checked"], 20)
        self.assertEqual(summary["incidence_views_prepared"], 6)
        self.assertEqual(summary["status"], "diagnostic_evidence_ready_pending_interpretation")
        for key in (*diagnostic.BLOCKED, "outcome_data_loaded", "acceptability_regraded",
                    "production_crops_written", "feasibility_certified", "new_crop_algorithm_applied"):
            self.assertIs(summary[key], False)
        after = diagnostic._snapshot([Path(path) for path in before])
        self.assertEqual(before, after)
        report = diagnostic._read_json(self.output/"diagnostico_privado.json")
        self.assertEqual(len(report["rows"]), 20)
        self.assertTrue(all(row["new_acceptability_decision"] is None for row in report["rows"]))
        record = diagnostic._read_json(self.output/"registro_diagnostico.json")
        self.assertEqual(record["gallery_sha256"], sha256_file(self.output/"galeria_diagnostica.html"))
        self.assertIn(str(self.config["pilot_audit_json"].resolve()), record["input_snapshot"])
        self.assertIn(str(self.config["_diagnostic_config_file"].resolve()), record["input_snapshot"])
        self.assertEqual(len([p for p in record["input_snapshot"] if p.endswith(".tar")]), 10)
        public_text = (self.output/"resumen_diagnostico_publico.json").read_text(encoding="utf-8")
        self.assertNotIn("case_", public_text)
        self.assertNotIn(str(self.root), public_text)

    def test_replay_difference_stops_without_success_record(self):
        original = diagnostic.candidate_for_half
        def changed_candidate(*args):
            result = original(*args)
            result["native_dicom_box"][0] += 1
            return result
        with patch.object(diagnostic, "candidate_for_half", side_effect=changed_candidate):
            with self.assertRaisesRegex(ValueError, "replay_not_exact"):
                self.run_diagnostic()
        self.assertFalse((self.output/"registro_diagnostico.json").exists())
        self.assertFalse((self.output/"resumen_diagnostico_publico.json").exists())

    def test_instrumented_trace_difference_stops_without_success_record(self):
        original = diagnostic.trace_for_half
        def changed_trace(*args):
            result = original(*args)
            result["native_crop_sha256"] = "not the historical pixels"
            return result
        with patch.object(diagnostic, "trace_for_half", side_effect=changed_trace):
            with self.assertRaisesRegex(ValueError, "instrumented_replay_not_exact"):
                self.run_diagnostic()
        self.assertFalse((self.output/"registro_diagnostico.json").exists())

    def test_source_anchor_changed_during_replay_stops_final_success(self):
        def changed_anchor(config):
            config["_diagnostic_config_file"].write_text('{"changed":true}', encoding="utf-8")
            yield from self.studies
        with self.assertRaisesRegex(ValueError, "inputs_changed_during_diagnostic"):
            self.run_diagnostic(changed_anchor)
        self.assertFalse((self.output/"registro_diagnostico.json").exists())

    def test_native_display_has_exact_spatial_size_and_expected_linear_intensities(self):
        raw = np.arange(120, dtype=np.uint16).reshape(12, 10)
        copy = raw.copy()
        direct = diagnostic.native_window(raw, "MONOCHROME2", 20, 100)
        inverted = diagnostic.native_window(raw, "MONOCHROME1", 20, 100)
        expected = np.rint(255*np.clip((raw.astype(float)-20)/80, 0, 1)).astype(np.uint8)
        self.assertEqual(direct.size, (10, 12))
        np.testing.assert_array_equal(np.asarray(direct), expected)
        np.testing.assert_array_equal(np.asarray(inverted), np.rint(255*(1-np.clip((raw.astype(float)-20)/80, 0, 1))).astype(np.uint8))
        np.testing.assert_array_equal(raw, copy)
        for low, high in ((1, 1), (float("nan"), 20)):
            with self.assertRaises(ValueError):
                diagnostic.native_window(raw, "MONOCHROME2", low, high)

    def test_native_png_is_not_resized_and_saved_window_is_reused(self):
        target = self.root/"views"
        target.mkdir()
        raw = np.arange(120, dtype=np.uint16).reshape(12, 10)
        assets = diagnostic.write_native_views(raw, [2, 8, 3, 10], "MONOCHROME2", target,
            "synthetic", {"display_low": 20, "display_high": 100})
        self.assertEqual(len(assets), 4)
        for asset in assets:
            path = target/Path(asset["file"]).name
            with Image.open(path) as rendered:
                self.assertEqual(rendered.size, (asset["width"], asset["height"]))
                if asset["kind"] == "roi_sin_superposiciones":
                    self.assertEqual(rendered.size, (6, 7))
                    low, high = asset["window"]
                    source = diagnostic.native_window(raw, "MONOCHROME2", low, high)
                    np.testing.assert_array_equal(np.asarray(rendered), np.asarray(source)[3:10, 2:8])

    def test_html_escapes_notes_and_forces_scroll_natural_size(self):
        row = {"knee_alias": "synthetic", "closed_decision": "rechazado",
            "trace": {"mask_route": "none_detected"}, "saved_window": None,
            "incidence": True, "closed_notes": "<script>bad()</script>", "native_views": []}
        gallery = diagnostic.gallery_html([row], {}, self.root)
        self.assertNotIn("<script>", gallery)
        self.assertIn("&lt;script&gt;", gallery)
        self.assertIn("max-width:none!important", gallery)
        self.assertIn("overflow:auto", gallery)
        self.assertIn("pendiente de interpretación", gallery)


if __name__ == "__main__":
    unittest.main()
