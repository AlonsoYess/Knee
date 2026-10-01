"""Synthetic review, integrity, human-input and threshold tests; no OAI images."""

import csv
import importlib.util
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from knee.dicom_audit import sha256_bytes, sha256_file

DEPENDENCIES = all(importlib.util.find_spec(name) for name in ("cv2", "scipy"))
if DEPENDENCIES:
    from knee.roi_mcr004_review import (
        ALGORITHM_VERSION, UPSTREAM_COMMIT, BLOCKED, FLAGS, RESULT_FIELDS,
        REVIEW_FIELDS, SUMMARY_FILE, RECORD_FILE, CropConfig, Trace,
        close_review, make_view, read_context, validate_decisions,
        verify_native_integrity, window_image,
    )


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def acceptable(row):
    row.update(decision="aceptable", **{key:"SI" for key in FLAGS[:4]},
               critical_contamination_yes_no="NO", peripheral_warning_yes_no="NO",
               outcome_blinded_yes_no="SI", reviewer_notes="Synthetic technical observation")
    return row


@unittest.skipUnless(DEPENDENCIES, "MCR004 isolated environment required")
class Mcr004ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.output = self.root/"output"
        self.output.mkdir()
        (self.output/"recortes_nativos").mkdir()
        (self.output/"previews_ciegas").mkdir()
        self.config = {"output_dir":self.output, "source_dir":self.root,
                       "algorithm_version":ALGORITHM_VERSION,
                       "expected_unique_studies":10, "expected_knees":20}
        self.results, self.template, self.review, self.studies = [], [], [], []
        pixels = np.arange(100*200, dtype=np.uint16).reshape(100, 200)
        dataset = SimpleNamespace(PhotometricInterpretation="MONOCHROME2",
                                  ImagerPixelSpacing=[.15,.2])
        for index in range(1,11):
            case = f"case_{index:03d}"
            study = {"case_alias":case, "manifest_key":f"key_{index}", "split_column":"100"}
            record = {"package_relative_path":f"synthetic_{index}.tar",
                      "dicom_sha256":"digest", "pixel_sha256":"digest"}
            self.studies.append((study,record,dataset,pixels))
            for side, offset in (("RIGHT",0),("LEFT",100)):
                alias = case+"_"+side
                crop = pixels[30:70, offset+10:offset+90].copy()
                np.save(self.output/"recortes_nativos"/f"{alias}_crop.npy", crop, allow_pickle=False)
                (self.output/"previews_ciegas"/f"{alias}_localizacion.png").write_bytes(b"synthetic")
                trace = Trace(100,100,0,100,0,100,100,100,0,0,100,0,100)
                result = {key:"" for key in RESULT_FIELDS}
                result.update(case_alias=case, knee_alias=alias, patient_side=side,
                    manifest_key=study["manifest_key"], **record, status="candidate",
                    half_x0=str(offset), half_x1=str(offset+100),
                    native_dicom_box=json.dumps([offset+10,offset+90,30,70]),
                    requested_working_box=json.dumps([10,90,30,70]),
                    crop_rows="40", crop_columns="80", row_spacing_mm=".15",
                    column_spacing_mm=".2", crop_height_mm="6", crop_width_mm="16",
                    native_crop_sha256=sha256_bytes(crop.tobytes()),
                    trace_json=json.dumps(asdict(trace)),
                    preview_file=f"previews_ciegas/{alias}_localizacion.png",
                    native_crop_file=f"recortes_nativos/{alias}_crop.npy")
                self.results.append(result)
                row = {key:"" for key in REVIEW_FIELDS}
                row.update(knee_alias=alias,case_alias=case,patient_side=side,
                           candidate_status="candidate",preview_file=result["preview_file"])
                self.template.append(dict(row))
                self.review.append(acceptable(dict(row)))
        self.review[-1] = dict(self.template[-1])
        self.pending = self.review[-1]["knee_alias"]
        self.summary = {"status":"ready_for_technical_review",
            "algorithm_version":ALGORITHM_VERSION,"upstream_commit":UPSTREAM_COMMIT,
            "expected_unique_studies":10,"expected_knees":20,
            "candidate_knees":20,"abstained_knees":0,
            "core_config":json.loads(json.dumps(asdict(CropConfig()))),
            "outcome_data_loaded":False, **{key:False for key in BLOCKED}}
        self._save_fixture()

    def _save_fixture(self):
        write_csv(self.output/"resultados_roi_privados.csv",RESULT_FIELDS,self.results)
        write_csv(self.output/"revision_tecnica_ciega.csv",REVIEW_FIELDS,self.review)
        write_csv(self.output/"revision_tecnica_ciega_plantilla_original.csv",REVIEW_FIELDS,self.template)
        write_json(self.output/"resumen_regresion_publico.json",self.summary)
        self.prep = {"operation":"prepare_mcr004_historical_regression",
                     "git_commit":"e"*40,"summary_sha256":sha256_file(self.output/"resumen_regresion_publico.json"),
                     "review_template_sha256":sha256_file(self.output/"revision_tecnica_ciega_plantilla_original.csv"),
                     "candidate_knees":self.summary["candidate_knees"],
                     "abstained_knees":self.summary["abstained_knees"],
                     **{key:False for key in BLOCKED}}
        write_json(self.output/"registro_preparacion.json",self.prep)
        provenance = {"status":"assisted_draft_pending_investigator_confirmation",
            "outcome_data_consulted":False,"independent_second_reader":False,
            "investigator_confirmation":False,"closure_executed":False,
            "version_known_to_assistant":True,"pilot_git_commit":"e"*40,
            "original_template_sha256":self.prep["review_template_sha256"],
            "preparation_summary_sha256":self.prep["summary_sha256"],
            "prepared_review_csv_sha256":sha256_file(self.output/"revision_tecnica_ciega.csv"),
            "results_csv_sha256":sha256_file(self.output/"resultados_roi_privados.csv"),
            "pending_knee_aliases":[self.pending],"proposed_decisions":19}
        write_json(self.output/"procedencia_revision_asistida.json",provenance)

    def _response(self):
        with patch("knee.roi_mcr004_review._source_studies",return_value=iter(self.studies)):
            view = make_view(self.config,self.pending,None,None)
        update = {key:self.review[0][key] for key in ("decision",*FLAGS)}
        update.update(reviewer_notes="Investigator compared original and ROI; synthetic",
                      window_reviewed=True,view=view)
        return {"confirm_entire_assisted_review":True,"outcome_blinded":True,"technical_tests_passed":True,
                "draft_csv_sha256":sha256_file(self.output/"revision_tecnica_ciega.csv"),
                "updates":{self.pending:update}}

    def _all_accepted(self):
        return [acceptable(dict(row)) for row in self.template]

    def test_context_returns_only_existing_pending_and_anchors_original_template(self):
        context = read_context(self.config)
        self.assertEqual(context["pending_knee_aliases"],[self.pending])
        self.assertEqual(len(context["review"]),20)
        (self.output/"revision_tecnica_ciega_plantilla_original.csv").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError,"hash_mismatch"):
            read_context(self.config)

    def test_modified_draft_results_and_preparation_summary_each_block(self):
        for filename in ("revision_tecnica_ciega.csv","resultados_roi_privados.csv",
                         "resumen_regresion_publico.json"):
            with self.subTest(filename=filename):
                file = self.output/filename
                original = file.read_bytes()
                file.write_bytes(original+b"\n")
                with self.assertRaisesRegex(ValueError,"hash_mismatch"):
                    read_context(self.config)
                file.write_bytes(original)

    def test_changed_algorithm_parameters_or_downstream_flag_block(self):
        self.summary["core_config"]["intensity_offset"] = 99
        self._save_fixture()
        with self.assertRaisesRegex(ValueError,"parameters_changed"):
            read_context(self.config)
        self.summary["core_config"] = json.loads(json.dumps(asdict(CropConfig())))
        self.summary["training_executed"] = True
        self._save_fixture()
        with self.assertRaisesRegex(ValueError,"blocked_stage"):
            read_context(self.config)

    def test_identity_side_order_and_duplicates_are_not_repaired(self):
        self.review[0]["patient_side"] = "LEFT"
        self._save_fixture()
        with self.assertRaisesRegex(ValueError,"identity_changed"):
            read_context(self.config)
        self.review[0]["patient_side"] = "RIGHT"
        self.results[0]["knee_alias"] = self.results[1]["knee_alias"]
        self._save_fixture()
        with self.assertRaisesRegex(ValueError,"aliases_changed"):
            read_context(self.config)

    def test_threshold_20_accepted_up_to_two_warnings_passes(self):
        rows = self._all_accepted()
        for row in rows[:2]:
            row.update(decision="aceptable_con_advertencia_periferica",peripheral_warning_yes_no="SI")
        metrics = validate_decisions(rows)
        self.assertTrue(metrics["historical_gate_passed"])
        self.assertEqual(metrics["acceptable_crops"],20)
        self.assertEqual(metrics["both_knees_acceptable_studies"],10)

    def test_three_warnings_fail_even_with_20_accepted(self):
        rows = self._all_accepted()
        for row in rows[:3]:
            row.update(decision="aceptable_con_advertencia_periferica",peripheral_warning_yes_no="SI")
        self.assertFalse(validate_decisions(rows)["historical_gate_passed"])

    def test_one_incorrect_candidate_fails_despite_19_of_20(self):
        rows = self._all_accepted()
        rows[0].update(decision="rechazado",critical_contamination_yes_no="SI")
        metrics = validate_decisions(rows)
        self.assertEqual(metrics["acceptable_crops"],19)
        self.assertEqual(metrics["incorrect_candidates"],1)
        self.assertFalse(metrics["historical_gate_passed"])

    def test_one_abstention_not_one_rejection_can_pass_and_denominator_stays_20(self):
        rows = self._all_accepted()
        rows[0] = dict(self.template[0])
        rows[0]["candidate_status"] = "abstain"
        metrics = validate_decisions(rows)
        self.assertTrue(metrics["historical_gate_passed"])
        self.assertEqual(metrics["acceptable_fraction"],.95)
        self.assertEqual(metrics["incorrect_candidates"],0)
        self.assertEqual(metrics["both_knees_acceptable_studies"],9)
        acceptable(rows[0])
        with self.assertRaisesRegex(ValueError,"abstention"):
            validate_decisions(rows)

    def test_unknown_criteria_pending_or_non_blinded_cannot_be_accepted(self):
        for key,value in (("decision",""),("coverage_ok_yes_no",""),
                          ("critical_contamination_yes_no","SI"),("outcome_blinded_yes_no","NO")):
            rows = self._all_accepted()
            rows[0][key] = value
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_decisions(rows)

    def test_non_evaluable_requires_visualization_failure_and_counts_as_incorrect(self):
        rows = self._all_accepted()
        rows[0].update(decision="no_evaluable",visualizable_yes_no="NO",
                       coverage_ok_yes_no="",frame_ok_yes_no="")
        metrics = validate_decisions(rows)
        self.assertEqual(metrics["non_evaluable_crops"],1)
        self.assertEqual(metrics["incorrect_candidates"],1)
        self.assertFalse(metrics["historical_gate_passed"])
        rows[0]["visualizable_yes_no"] = "SI"
        with self.assertRaisesRegex(ValueError,"failed_visualization"):
            validate_decisions(rows)

    def test_reject_requires_failed_criterion_not_only_unpermitted_warning(self):
        rows = self._all_accepted()
        rows[0]["decision"] = "rechazado"
        with self.assertRaisesRegex(ValueError,"failed_criterion"):
            validate_decisions(rows)

    def test_display_windows_change_only_visualization_and_handle_both_polarities(self):
        half = np.arange(10000,dtype=np.uint16).reshape(100,100)
        original = half.copy()
        image = window_image(half,[10,90,30,70],"MONOCHROME2",0,9999)
        inverse = window_image(half,[10,90,30,70],"MONOCHROME1",0,9999)
        self.assertTrue(np.array_equal(half,original))
        self.assertEqual(image.size,inverse.size)
        self.assertEqual(image.getpixel((0,0))[0]+inverse.getpixel((0,0))[0],255)
        with self.assertRaisesRegex(ValueError,"invalid_display"):
            window_image(half,[10,90,30,70],"MONOCHROME2",1,1)

    def test_integrity_verifies_all_saved_native_crops_then_detects_one_changed_pixel(self):
        with patch("knee.roi_mcr004_review._source_studies",return_value=iter(self.studies)):
            self.assertEqual(verify_native_integrity(self.config,self.results),20)
        file = self.output/self.results[0]["native_crop_file"]
        crop = np.load(file,allow_pickle=False)
        crop[0,0] += 1
        np.save(file,crop,allow_pickle=False)
        with patch("knee.roi_mcr004_review._source_studies",return_value=iter(self.studies)):
            with self.assertRaisesRegex(ValueError,"pixel_or_shape"):
                verify_native_integrity(self.config,self.results)

    def test_integrity_blocks_native_bounds_trace_half_and_spacing_changes(self):
        for key,value in (("half_x0","1"),("requested_working_box","[11,90,30,70]"),
                          ("native_dicom_box","[0,201,30,70]"),("row_spacing_mm",".2")):
            rows = [dict(row) for row in self.results]
            rows[0][key] = value
            with self.subTest(key=key),patch("knee.roi_mcr004_review._source_studies",return_value=iter(self.studies)):
                with self.assertRaises(ValueError):
                    verify_native_integrity(self.config,rows)

    def test_human_confirmations_pending_rows_and_window_record_are_mandatory(self):
        response = self._response()
        for key in ("confirm_entire_assisted_review","outcome_blinded","technical_tests_passed"):
            modified = {**response,key:False}
            with self.assertRaisesRegex(ValueError,"confirmation_required"):
                close_review(self.config,modified,"a"*40)
        with self.assertRaisesRegex(ValueError,"unresolved"):
            close_review(self.config,{**response,"updates":{}},"a"*40)
        response["updates"][self.pending]["view"]["display_low"] += 1
        with self.assertRaisesRegex(ValueError,"view_integrity"):
            close_review(self.config,response,"a"*40)
        self.assertFalse((self.output/SUMMARY_FILE).exists())

    def test_stale_input_and_invalid_integrity_do_not_write_a_closure(self):
        response = self._response()
        with self.assertRaisesRegex(ValueError,"stale"):
            close_review(self.config,{**response,"draft_csv_sha256":"wrong"},"a"*40)
        with patch("knee.roi_mcr004_review.verify_native_integrity",side_effect=ValueError("changed")):
            with self.assertRaisesRegex(ValueError,"changed"):
                close_review(self.config,response,"a"*40)
        self.assertFalse((self.output/RECORD_FILE).exists())
        self.assertFalse((self.output/"revision_tecnica_ciega_borrador_asistido.csv").exists())

    def test_successful_closure_preserves_draft_native_arrays_and_blocks_downstream(self):
        response = self._response()
        original_draft = (self.output/"revision_tecnica_ciega.csv").read_bytes()
        native_hashes = [sha256_file(self.output/row["native_crop_file"]) for row in self.results]
        with patch("knee.roi_mcr004_review._source_studies",return_value=iter(self.studies)):
            summary = close_review(self.config,response,"a"*40)
        self.assertTrue(summary["historical_gate_passed"])
        self.assertEqual(summary["step_4_status"],"open")
        self.assertFalse(summary["parameters_frozen"])
        self.assertTrue(all(summary[key] is False for key in BLOCKED))
        self.assertEqual((self.output/"revision_tecnica_ciega_borrador_asistido.csv").read_bytes(),original_draft)
        self.assertEqual(sha256_file(self.output/"revision_tecnica_ciega.csv"),summary["review_csv_sha256"])
        self.assertEqual([sha256_file(self.output/row["native_crop_file"]) for row in self.results],native_hashes)
        record = json.loads((self.output/RECORD_FILE).read_text())
        self.assertEqual(record["summary_sha256"],sha256_file(self.output/SUMMARY_FILE))
        with self.assertRaises(FileExistsError):
            close_review(self.config,response,"a"*40)

    def test_rejected_closure_keeps_known_failure_not_retroactive_abstention(self):
        response = self._response()
        response["updates"][self.pending].update(decision="rechazado",critical_contamination_yes_no="SI")
        with patch("knee.roi_mcr004_review._source_studies",return_value=iter(self.studies)):
            summary = close_review(self.config,response,"a"*40)
        self.assertEqual(summary["status"],"rejected_after_technical_review")
        self.assertEqual(summary["candidate_knees"],20)
        self.assertEqual(summary["abstained_knees"],0)
        self.assertEqual(summary["incorrect_candidates"],1)


if __name__ == "__main__":
    unittest.main()
