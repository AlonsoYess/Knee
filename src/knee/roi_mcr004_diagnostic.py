"""Read-only historical MCR004 replay and diagnostic views, not a new cropper.

The rejected closure is an input, never an output. This module does not assign
new acceptability decisions, identify physical marks or tune any parameter.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from knee.dicom_audit import sha256_bytes, sha256_file
from knee.joint_localization import (
    _extract_spacing, _load_verified_pilot, _resolve_package, _resolve_under_root,
)
from knee.roi_mcr004 import ALGORITHM_VERSION, UPSTREAM_COMMIT, candidate_for_half
from knee.roi_mcr004_diagnostic_trace import trace_for_half
from knee.roi_mcr004_pilot import RESULT_FIELDS, REVIEW_FIELDS, _read_json, load_config
from knee.roi_mcr004_review import (
    BLOCKED, FLAGS, IMMUTABLE, RECORD_FILE, SUMMARY_FILE, _require_false, _rows,
    _source_studies, _verify_crop, _verify_saved_view, validate_decisions,
)
from knee.third_party.emory_hiti.config import CropConfig


DIAGNOSTIC_VERSION = "mcr004_equivalent_diagnostic_v1.0"
EXPECTED_ENVIRONMENT = {"numpy": "1.26.4", "scipy": "1.14.1",
    "opencv-python": "4.10.0.84", "pydicom": "3.0.1", "Pillow": "11.3.0"}
INPUT_FILES = (SUMMARY_FILE, RECORD_FILE, "respuestas_investigador.json",
    "registro_preparacion.json", "resumen_regresion_publico.json",
    "resultados_roi_privados.csv", "revision_tecnica_ciega.csv",
    "revision_tecnica_ciega_borrador_asistido.csv", "procedencia_revision_asistida.json",
    "revision_tecnica_ciega_plantilla_original.csv")
LIMITS = (
    "La equivalencia no acredita calidad anatómica ni ausencia de contaminación.",
    "El detector de contornos oscuros existente no identifica semánticamente una máscara.",
    "La presencia de una marca en píxeles originales no determina su origen físico.",
    "Las vistas son 1:1 espacialmente; su intensidad se transforma solo para visualización.",
    "No se calcula una caja corregida ni se cambian las decisiones cerradas.",
    "La factibilidad geométrica/anatómica requiere interpretar la evidencia; no se certifica automáticamente.",
)


def _environment() -> dict[str, str]:
    environment = {"python": sys.version.split()[0],
                   **{name: version(name) for name in EXPECTED_ENVIRONMENT}}
    if sys.version_info[:2] != (3, 12) or any(
            environment[name] != expected for name, expected in EXPECTED_ENVIRONMENT.items()):
        raise ValueError("diagnostic_requires_fixed_python312_mcr004_environment")
    return environment


def _snapshot(paths: list[Path]) -> dict[str, str]:
    return {str(path.resolve()): sha256_file(path) for path in paths}


def read_closed_context(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the *closed* record; pending-review APIs deliberately aren't used."""
    if config["algorithm_version"] != ALGORITHM_VERSION or (
            config["expected_unique_studies"], config["expected_knees"]) != (10, 20):
        raise ValueError("historical_contract_changed")
    output = config["output_dir"]
    summary, record = _read_json(output/SUMMARY_FILE), _read_json(output/RECORD_FILE)
    if sha256_file(output/SUMMARY_FILE) != record.get("summary_sha256"):
        raise ValueError("closed_summary_hash_mismatch")
    if any(record.get(key) != value for key, value in summary.items()):
        raise ValueError("closed_summary_record_mismatch")
    expected = {"operation": "close_mcr004_historical_technical_review", "phase": 1,
        "step": 4, "status": "rejected_after_technical_review",
        "algorithm_version": ALGORITHM_VERSION, "upstream_commit": UPSTREAM_COMMIT,
        "visual_review_status": "complete", "step_4_status": "open",
        "parameters_frozen": False, "native_candidate_integrity_checked": 20,
        "closure_execution_environment": "Google Colab, libreta 14"}
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError("expected_rejected_colab14_closure_required")
    _require_false(summary, BLOCKED+("outcome_data_loaded",))
    for key in ("pilot_git_commit", "closure_git_commit"):
        if not re.fullmatch(r"[0-9a-f]{40}", str(summary.get(key, ""))):
            raise ValueError("full_historical_commit_required")
    hash_checks = (
        ("revision_tecnica_ciega.csv", summary.get("review_csv_sha256")),
        ("resultados_roi_privados.csv", summary.get("results_csv_sha256")),
        ("revision_tecnica_ciega_borrador_asistido.csv", summary.get("assisted_draft_csv_sha256")),
        ("procedencia_revision_asistida.json", record.get("assisted_provenance_sha256")),
    )
    for name, digest in hash_checks:
        if not digest or sha256_file(output/name) != digest:
            raise ValueError(f"closed_input_hash_mismatch:{name}")
    response = _read_json(output/"respuestas_investigador.json")
    if response != record.get("response") or any(response.get(key) is not True for key in (
            "confirm_entire_assisted_review", "outcome_blinded", "technical_tests_passed")):
        raise ValueError("recorded_investigator_response_mismatch")
    if response.get("draft_csv_sha256") != summary["assisted_draft_csv_sha256"]:
        raise ValueError("response_draft_hash_mismatch")
    prep = _read_json(output/"registro_preparacion.json")
    generation = _read_json(output/"resumen_regresion_publico.json")
    if (prep.get("operation") != "prepare_mcr004_historical_regression" or
            prep.get("git_commit") != summary["pilot_git_commit"] or
            sha256_file(output/"resumen_regresion_publico.json") != prep.get("summary_sha256") or
            sha256_file(output/"revision_tecnica_ciega_plantilla_original.csv") != prep.get("review_template_sha256")):
        raise ValueError("historical_preparation_anchor_mismatch")
    if (generation.get("algorithm_version") != ALGORITHM_VERSION or
            generation.get("upstream_commit") != UPSTREAM_COMMIT or
            generation.get("status") != "ready_for_technical_review" or
            generation.get("core_config") != json.loads(json.dumps(asdict(CropConfig())))):
        raise ValueError("fixed_historical_generation_mismatch")
    _require_false(generation, BLOCKED+("outcome_data_loaded",))
    _require_false(prep, BLOCKED)
    results = _rows(output/"resultados_roi_privados.csv", RESULT_FIELDS)
    review = _rows(output/"revision_tecnica_ciega.csv", REVIEW_FIELDS)
    template = _rows(output/"revision_tecnica_ciega_plantilla_original.csv", REVIEW_FIELDS)
    aliases = [row["knee_alias"] for row in results]
    expected_aliases = {f"case_{i:03d}_{side}" for i in range(1, 11) for side in ("RIGHT", "LEFT")}
    if len(results) != 20 or len(set(aliases)) != 20 or set(aliases) != expected_aliases or (
            [r["knee_alias"] for r in review] != aliases or [r["knee_alias"] for r in template] != aliases):
        raise ValueError("closed_historical_identity_or_order_changed")
    for result, row, original in zip(results, review, template):
        identity = dict(zip(IMMUTABLE, (result["knee_alias"], result["case_alias"],
            result["patient_side"], result["status"], result["preview_file"])))
        if (result["status"] != "candidate" or result["knee_alias"] !=
                result["case_alias"]+"_"+result["patient_side"] or
                any(row[key] != identity[key] or original[key] != identity[key] for key in IMMUTABLE) or
                any(original[key] for key in REVIEW_FIELDS[5:])):
            raise ValueError("closed_candidate_identity_changed")
    metrics = validate_decisions(review)
    if any(summary.get(key) != value for key, value in metrics.items()):
        raise ValueError("closed_metrics_do_not_match_review")
    if (metrics["acceptable_crops"], metrics["incorrect_candidates"],
            metrics["acceptable_with_peripheral_warning"]) != (17, 3, 3):
        raise ValueError("expected_closed_17_3_3_historical_result_required")
    if any(generation.get(key) != metrics[key] or prep.get(key) != metrics[key] for key in (
            "candidate_knees", "abstained_knees")):
        raise ValueError("historical_candidate_counts_changed")
    provenance = _read_json(output/"procedencia_revision_asistida.json")
    for key, expected in (
        ("prepared_review_csv_sha256", summary["assisted_draft_csv_sha256"]),
        ("results_csv_sha256", summary["results_csv_sha256"]),
        ("pilot_git_commit", summary["pilot_git_commit"]),
        ("preparation_summary_sha256", prep["summary_sha256"]),
        ("original_template_sha256", prep["review_template_sha256"]),
    ):
        if provenance.get(key) != expected:
            raise ValueError("closed_original_assistance_anchor_mismatch")
    paths = [output/name for name in INPUT_FILES]
    completed = record.get("completed_assistance")
    completed_provenance = None
    if summary.get("review_entry_mode") == "codex_prepared":
        if not isinstance(completed, dict) or response.get("assisted_completion") != completed:
            raise ValueError("closed_completed_assistance_fingerprint_missing")
        for file_key, hash_key in (("csv_file", "csv_sha256"), ("provenance_file", "provenance_sha256")):
            path = _resolve_under_root(output, completed[file_key], file_key)
            if sha256_file(path) != completed.get(hash_key):
                raise ValueError("closed_completed_assistance_hash_mismatch")
            paths.append(path)
        completed_rows = _rows(_resolve_under_root(output, completed["csv_file"], "csv_file"), REVIEW_FIELDS)
        completed_provenance = _read_json(_resolve_under_root(output, completed["provenance_file"], "provenance_file"))
        if (completed_rows != review or
                completed_provenance.get("draft_csv_sha256") != summary["assisted_draft_csv_sha256"] or
                completed_provenance.get("completed_csv_sha256") != summary["review_csv_sha256"] or
                completed_provenance.get("original_assisted_provenance_sha256") != record["assisted_provenance_sha256"]):
            raise ValueError("closed_completed_assistance_content_mismatch")
    views = {}
    updates = response.get("updates")
    if not isinstance(updates, dict) or not set(updates) <= set(aliases):
        raise ValueError("closed_updates_identity_changed")
    if completed_provenance is not None:
        saved_evidence = completed_provenance.get("window_evidence")
        if not isinstance(saved_evidence, dict) or set(saved_evidence) != set(updates) or any(
                update.get("view") != saved_evidence[alias] for alias, update in updates.items()):
            raise ValueError("closed_window_evidence_missing_or_changed")
    for alias, update in updates.items():
        row = next(r for r in review if r["knee_alias"] == alias)
        if update.get("window_reviewed") is not True or any(
                update.get(key) != row[key] for key in ("decision", *FLAGS)):
            raise ValueError("closed_update_does_not_match_final_decision")
        view = update.get("view")
        _verify_saved_view(config, {"results": results,
            "review_csv_sha256": summary["assisted_draft_csv_sha256"]}, row, view)
        views[alias] = view
        path = _resolve_under_root(output, view["display_file"], "display_file")
        paths.extend((path, path.with_suffix(".json")))
    for row in results:
        paths.append(_resolve_under_root(output, row["native_crop_file"], "native_crop_file"))
    return {"summary": summary, "record": record, "metrics": metrics,
        "results": results, "review": review, "views": views, "input_snapshot": _snapshot(paths)}


def resolve_diagnostic_output(config: dict[str, Any], relative: str) -> Path:
    root_value = os.environ.get("KNEE_DATA_ROOT")
    if not root_value or Path(relative).is_absolute() or not relative.strip():
        raise ValueError("relative_diagnostic_output_and_private_root_required")
    root = Path(root_value).expanduser().resolve()
    target = _resolve_under_root(root, relative, "diagnostic_output_dir")
    outputs = (root/"outputs").resolve()
    if target == outputs or outputs not in target.parents:
        raise ValueError("diagnostic_output_must_be_below_private_outputs")
    protected = [config[key].resolve() for key in ("output_dir", "source_dir", "bilateral_output_dir")]
    protected += [config[key].resolve().parent for key in (
        "pilot_audit_json", "v04_review_summary_json", "v04_closure_record_json", "v04_review_csv")]
    if any(target == p or target in p.parents or p in target.parents for p in protected):
        raise ValueError("diagnostic_output_overlaps_historical_inputs")
    if target.exists():
        raise FileExistsError("diagnostic_output_already_exists_do_not_overwrite_use_existing_report")
    return target


def native_window(half: np.ndarray, photometric: str, low: float, high: float) -> Image.Image:
    """One display pixel per native pixel. No resizing, interpolation or marks."""
    if (half.ndim != 2 or not math.isfinite(low) or not math.isfinite(high) or high <= low or
            photometric not in {"MONOCHROME1", "MONOCHROME2"}):
        raise ValueError("invalid_native_display_window")
    gray = np.clip((half.astype(np.float64)-low)/(high-low), 0, 1)
    if photometric == "MONOCHROME1":
        gray = 1-gray
    return Image.fromarray(np.rint(gray*255).astype(np.uint8))


def write_native_views(half: np.ndarray, box: list[int], photometric: str,
                       target: Path, alias: str, saved_view: dict | None) -> list[dict]:
    """Display-only PNGs for flagged cases; original arrays are never written."""
    x0, x1, y0, y1 = box
    baseline = np.percentile(half, [1, 99]).tolist()
    windows = [("ventana_base", baseline)]
    if saved_view is not None:
        saved = [saved_view["display_low"], saved_view["display_high"]]
        if saved != baseline:
            windows.append(("ventana_conservada", saved))
    assets = []
    for label, (low, high) in windows:
        image = native_window(half, photometric, float(low), float(high))
        crop = image.crop((x0, y0, x1, y1))
        # Context keeps natural size; cyan box is only on this separate copy.
        context = image.convert("RGB")
        ImageDraw.Draw(context).rectangle((x0, y0, x1-1, y1-1), outline=(0, 255, 255), width=2)
        for kind, rendered in (("contexto", context), ("roi_sin_superposiciones", crop)):
            name = f"{alias}_{label}_{kind}.png"
            path = target/name
            with path.open("xb") as stream:
                rendered.save(stream, format="PNG")
            with Image.open(path) as reread:
                if reread.size != rendered.size or not np.array_equal(np.asarray(reread), np.asarray(rendered)):
                    raise ValueError("native_display_roundtrip_mismatch")
            assets.append({"file": "vistas_nativas/"+name, "kind": kind,
                "window_source": label, "window": [float(low), float(high)],
                "width": rendered.width, "height": rendered.height, "spatial_scale": "1:1",
                "sha256": sha256_file(path), "visualization_only": True})
    return assets


def gallery_html(rows: list[dict], report: dict, output: Path) -> str:
    escape = lambda value: html.escape(str(value), quote=True)
    pieces = ["<!doctype html><html lang='es'><meta charset='utf-8'><title>Diagnóstico MCR004</title>",
        "<style>body{font:16px system-ui;margin:20px}td,th{border:1px solid #aaa;padding:6px;vertical-align:top}",
        "table{border-collapse:collapse}.native{overflow:auto;max-height:800px;border:1px solid #aaa}",
        ".native img{max-width:none!important;width:auto!important;height:auto!important;display:block}</style>",
        "<h1>Diagnóstico equivalente MCR004 — privado</h1><p>Resultado histórico conservado: Q=17/20, F=3, W=3; rechazo. Paso 4 abierto.</p>",
        "<p>Las imágenes se muestran a tamaño natural con desplazamiento. No uses ‘ajustar al ancho’ para inspeccionar los trazos finos. Una ventana cambia intensidades de visualización, no coordenadas ni píxeles de la ROI original.</p>",
        "<h2>Límites</h2><ul>", *["<li>"+escape(limit)+"</li>" for limit in LIMITS], "</ul>",
        "<h2>Denominador completo: 20 rodillas</h2><table><tr><th>Rodilla</th><th>Decisión cerrada (no nueva)</th><th>Equivalencia</th><th>Ruta del detector oscuro</th><th>Ventana conservada disponible</th></tr>"]
    for row in rows:
        pieces.append("<tr>"+"".join("<td>"+escape(value)+"</td>" for value in (
            row["knee_alias"], row["closed_decision"], "Exacta: caja y píxeles",
            row["trace"]["mask_route"], "Sí" if row["saved_window"] else "No"))+"</tr>")
    pieces.append("</table><h2>Seis incidencias de la revisión cerrada</h2><p>Las notas son evidencia histórica, no identificación clínica de marcas. La relación con la anatomía permanece pendiente de interpretación técnica de estas vistas, sin pedir al estudiante diagnosticarla.</p>")
    for row in rows:
        if not row["incidence"]:
            continue
        pieces.extend(("<h3>"+escape(row["knee_alias"])+" — "+escape(row["closed_decision"])+"</h3>",
            "<p>Nota cerrada: "+escape(row["closed_notes"])+"</p>",
            "<p>Relación elemento/anatomía: indeterminada, pendiente de interpretación de evidencia; no se afirma que exista un borde seguro.</p>",
            "<details><summary>Traza de cálculo equivalente (coordenadas de trabajo explícitas)</summary><pre>"+
                escape(json.dumps(row["trace"], ensure_ascii=False, indent=2))+"</pre></details>"))
        for asset in row["native_views"]:
            path = _resolve_under_root(output, asset["file"], "diagnostic_view")
            if sha256_file(path) != asset["sha256"]:
                raise ValueError("diagnostic_view_hash_mismatch")
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            pieces.append("<p>"+escape(asset["kind"])+"; "+escape(asset["window_source"])+
                "; límites "+escape(asset["window"])+"; "+escape(asset["width"])+"×"+
                escape(asset["height"])+" píxeles, 1:1.</p><div class='native'><img alt='Vista diagnóstica privada' src='data:image/png;base64,"+
                encoded+"'></div>")
    pieces.append("<h2>Conclusión de esta ejecución</h2><p>Se reconstruyó el comportamiento, no se validó una corrección. La interpretación de las seis incidencias es el siguiente trabajo. No se modifica el cierre ni se promueve el candidato. No se ejecutan etapas posteriores.</p></html>")
    return "\n".join(pieces)


def run_diagnostic(config: dict[str, Any], relative_output: str, git_commit: str) -> dict[str, Any]:
    started = time.perf_counter()
    if not re.fullmatch(r"[0-9a-f]{40}", git_commit):
        raise ValueError("full_diagnostic_git_commit_required")
    environment = _environment()
    output = resolve_diagnostic_output(config, relative_output)
    context = read_closed_context(config)
    anchor_paths = [config[key] for key in ("pilot_audit_json", "v04_review_summary_json",
        "v04_closure_record_json", "v04_review_csv")]
    anchor_paths += [config["bilateral_output_dir"]/name for name in (
        "cierre_paso_3_publico.json", "parametros_congelados.json", "resultados_separacion_privados.csv")]
    anchor_paths.append(config["_diagnostic_config_file"])
    audit = _load_verified_pilot(config["pilot_audit_json"], 10)
    anchor_paths += [_resolve_package(config["source_dir"], row["package_relative_path"]) for row in audit.values()]
    context["input_snapshot"].update(_snapshot(anchor_paths))
    diagnostic_rows, seen = [], set()
    # No output is created before closure validation; no success marker before all20 pass.
    output.mkdir(parents=True, exist_ok=False)
    (output/"vistas_nativas").mkdir()
    for study, source_record, dataset, pixels in _source_studies(config):
        matching = [row for row in context["results"] if row["case_alias"] == study["case_alias"]]
        if len(matching) != 2 or {r["patient_side"] for r in matching} != {"RIGHT", "LEFT"}:
            raise ValueError("historical_study_mapping_mismatch")
        for row in matching:
            alias = row["knee_alias"]
            if alias in seen:
                raise ValueError("duplicate_diagnostic_candidate")
            seen.add(alias)
            half, native_box = _verify_crop(config, row, study, source_record, dataset, pixels)
            photometric = str(dataset.PhotometricInterpretation).upper().strip()
            row_mm, col_mm, _ = _extract_spacing(dataset, ["ImagerPixelSpacing", "PixelSpacing"])
            candidate = candidate_for_half(half, row["patient_side"], photometric,
                int(row["half_x0"]), row_mm, col_mm, CropConfig())
            if candidate["status"] != "candidate":
                raise ValueError("closed_candidate_replay_abstained")
            if (candidate["native_half_box"] != native_box or
                    candidate["native_dicom_box"] != json.loads(row["native_dicom_box"]) or
                    candidate["requested_working_box"] != json.loads(row["requested_working_box"]) or
                    candidate["trace"] != json.loads(row["trace_json"]) or
                    sha256_bytes(candidate["crop"].tobytes()) != row["native_crop_sha256"] or
                    not np.array_equal(candidate["crop"], half[native_box[2]:native_box[3], native_box[0]:native_box[1]])):
                raise ValueError("closed_candidate_replay_not_exact")
            trace = trace_for_half(half, row["patient_side"], photometric)
            if (trace["native_half_box"] != native_box or
                    trace["requested_working_box"] != candidate["requested_working_box"] or
                    trace["native_crop_sha256"] != row["native_crop_sha256"]):
                raise ValueError("instrumented_replay_not_exact")
            closed_row = next(r for r in context["review"] if r["knee_alias"] == alias)
            incidence = closed_row["decision"] != "aceptable"
            view = context["views"].get(alias)
            assets = write_native_views(half, native_box, photometric, output/"vistas_nativas",
                alias, view) if incidence else []
            diagnostic_rows.append({"knee_alias": alias, "case_alias": row["case_alias"],
                "patient_side": row["patient_side"], "closed_decision": closed_row["decision"],
                "closed_notes": closed_row["reviewer_notes"], "incidence": incidence,
                "saved_window": view, "trace": trace, "native_views": assets,
                "spatial_relation": "indeterminate_pending_technical_interpretation",
                "physical_origin": "not_determined", "new_acceptability_decision": None})
    if (seen != {row["knee_alias"] for row in context["results"]} or len(diagnostic_rows) != 20 or
            sum(row["incidence"] for row in diagnostic_rows) != 6):
        raise ValueError("diagnostic_denominator_or_incidence_count_changed")
    order = {row["knee_alias"]: index for index, row in enumerate(context["results"])}
    diagnostic_rows.sort(key=lambda row: order[row["knee_alias"]])
    report = {"diagnostic_version": DIAGNOSTIC_VERSION, "algorithm_version": ALGORITHM_VERSION,
        "diagnostic_git_commit": git_commit, "closure": context["summary"],
        "input_snapshot": context["input_snapshot"], "rows": diagnostic_rows, "limitations": LIMITS}
    gallery = gallery_html(diagnostic_rows, report, output)
    if _snapshot([Path(path) for path in context["input_snapshot"]]) != context["input_snapshot"]:
        raise ValueError("historical_inputs_changed_during_diagnostic")
    summary = {"phase": 1, "step": 4, "operation": "diagnose_mcr004_equivalent_historical_replay",
        "status": "diagnostic_evidence_ready_pending_interpretation",
        "diagnostic_version": DIAGNOSTIC_VERSION, "algorithm_version": ALGORITHM_VERSION,
        "diagnostic_git_commit": git_commit, "upstream_commit": UPSTREAM_COMMIT,
        "expected_unique_studies": 10, "expected_knees": 20,
        "exact_replayed_candidates": 20, "native_candidate_integrity_checked": 20,
        "instrumented_equivalence_checked": 20, "incidence_views_prepared": 6,
        "historical_status_unchanged": "rejected_after_technical_review",
        "historical_acceptable_crops_unchanged": 17, "historical_incorrect_candidates_unchanged": 3,
        "historical_peripheral_warnings_unchanged": 3, "historical_inputs_unchanged": True,
        "new_crop_algorithm_applied": False, "production_crops_written": False,
        "acceptability_regraded": False, "feasibility_certified": False,
        "step_4_status": "open", "parameters_frozen": False,
        "new_studies_selected": False, "outcome_data_loaded": False,
        **{flag: False for flag in BLOCKED}, "environment": environment,
        "elapsed_seconds": round(time.perf_counter()-started, 3),
        "diagnostic_execution_environment": os.environ.get("KNEE_DIAGNOSTIC_EXECUTION_ENVIRONMENT", "synthetic_test"),
        "limitations": LIMITS}
    for name, payload in (("diagnostico_privado.json", report),
                          ("resumen_diagnostico_publico.json", summary)):
        with (output/name).open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, indent=2)+"\n")
    with (output/"galeria_diagnostica.html").open("x", encoding="utf-8") as stream:
        stream.write(gallery)
    record = {"diagnostic_version": DIAGNOSTIC_VERSION,
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_git_commit": git_commit, "summary_sha256": sha256_file(output/"resumen_diagnostico_publico.json"),
        "private_report_sha256": sha256_file(output/"diagnostico_privado.json"),
        "gallery_sha256": sha256_file(output/"galeria_diagnostica.html"),
        "input_snapshot": context["input_snapshot"],
        "view_hashes": {asset["file"]: asset["sha256"] for row in diagnostic_rows for asset in row["native_views"]}}
    with (output/"registro_diagnostico.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, indent=2)+"\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()
    if os.environ.get("KNEE_DIAGNOSTIC_EXECUTION_ENVIRONMENT") != "Google Colab, libreta 15":
        raise RuntimeError("Diagnostic CLI runs only in investigator Google Colab, libreta 15.")
    config = load_config(args.config)
    config["_diagnostic_config_file"] = args.config.resolve()
    print(json.dumps(run_diagnostic(config, args.output_dir, args.git_commit),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
