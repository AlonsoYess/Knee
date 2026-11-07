"""Protocol double for API/browser tests only; never enabled by application configuration.

These geometries and decisions are deliberately invented. Scientific parity is
verified separately against the frozen worker, not by this double.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image

from knee_app.processor import ProcessorError


class SyntheticProcessor:
    pipeline_version = "synthetic-tests-only"
    qc_policy_version = "synthetic-qc-tests-only"

    def __init__(self):
        self.calls = []
        self.enabled = True
        self.fail_source = False
        self.fail_crop = False

    def is_available(self):
        return self.enabled

    @staticmethod
    def _json(path, value):
        Path(path).write_text(json.dumps(value, sort_keys=True), encoding="utf-8")

    @staticmethod
    def _read(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def _views(array, out, prefix):
        values = np.clip(array.astype(float) / 4095 * 255, 0, 255).astype(np.uint8)
        result = {}
        for suffix in ("a", "b"):
            name = f"{prefix}_{suffix}.png"
            Image.fromarray(values).save(out / name)
            result[f"{prefix}_{suffix}"] = name
        return result

    def source(self, sourcePath, side, outDir):
        self.calls.append("source")
        if self.fail_source:
            raise ProcessorError("IMAGE_REJECTED", "La imagen sintética no pudo prepararse.")
        out = Path(outDir)
        try:
            dataset = pydicom.dcmread(sourcePath)
            full = dataset.pixel_array
            if full.ndim != 2 or full.size > 2_000_000:
                raise ValueError("unsupported synthetic fixture")
        except Exception as error:
            raise ProcessorError("IMAGE_REJECTED", "El archivo no es un DICOM de prueba válido.") from error
        midpoint = full.shape[1] // 2
        half = full[:, :midpoint] if side == "right" else full[:, midpoint:]
        np.save(out / "original.npy", full, allow_pickle=False)
        np.save(out / "half.npy", half, allow_pickle=False)
        result = {"version": "synthetic-tests-only", "side": side,
                  "fullNativeShape": list(full.shape), "nativeHalfShape": list(half.shape),
                  "artifacts": {**self._views(full, out, "full"), **self._views(half, out, "half")},
                  "sourceMeta": {"knee_alias": "synthetic_" + side.upper()},
                  "dicomSha256": hashlib.sha256(Path(sourcePath).read_bytes()).hexdigest()}
        self._json(out / "source.json", result)
        return result

    def validate_reference(self, sourceDir, reference):
        self.calls.append("reference")
        # API/service normalizes the public camelCase fields before the worker boundary.
        source = self._read(Path(sourceDir) / "source.json")
        resolved = reference.get("center_resolved", reference.get("centerResolved"))
        xy = reference.get("center_xy", reference.get("centerXY"))
        inspected = reference.get("original_and_half_inspected", reference.get("originalAndHalfInspected"))
        blinded = reference.get("outcome_blinded", reference.get("outcomeBlinded"))
        rows, columns = source["nativeHalfShape"]
        if inspected is not True or blinded is not True or len(reference.get("note", "").strip()) < 35:
            raise ProcessorError("SOURCE_REFERENCE_INVALID", "Revisa la referencia de prueba.")
        if resolved and (not isinstance(xy, list) or len(xy) != 2 or not all(type(v) is int for v in xy) or not 0 <= xy[0] < columns or not 0 <= xy[1] < rows):
            raise ProcessorError("SOURCE_REFERENCE_INVALID", "Referencia fuera de la imagen.")
        if not resolved and xy is not None:
            raise ProcessorError("SOURCE_REFERENCE_INVALID", "Referencia incoherente.")
        return {**reference, "knee_alias": source["sourceMeta"]["knee_alias"]}

    def crop(self, sourceDir, outDir):
        self.calls.append("crop")
        if self.fail_crop:
            raise ProcessorError("PROCESSING_FAILED", "Fallo de procesamiento sintético.")
        out = Path(outDir)
        half = np.load(Path(sourceDir) / "half.npy", allow_pickle=False)
        box = [100, 300, 70, 170]
        native = half[70:170, 100:300].copy()
        np.save(out / "crop.npy", native, allow_pickle=False)
        result = {"version": "synthetic-tests-only", "status": "candidate", "reason": None,
                  "box": box, "nativeHalfBox": box, "cropNativeShape": list(native.shape),
                  "artifacts": self._views(native, out, "crop")}
        self._json(out / "candidate.json", result)
        return result

    def assess(self, sourceDir, candidateDir, reference, observation):
        self.calls.append("assess")
        self.validate_reference(sourceDir, reference)
        if not isinstance(observation, dict) or len(observation.get("note", "").strip()) < 35:
            raise ProcessorError("QC_INVALID", "Observación de prueba incompleta.")
        blocked = observation.get("unclassified_mark_present") is not False or observation.get("side_supported") is not True
        blocked = blocked or observation.get("coverage_complete") is not True or observation.get("lateral_margins_visible") is not True
        blocked = blocked or any(value != "visible" for value in observation.get("visibility", {}).values())
        blocked = blocked or observation.get("hardware") != "absent" or observation.get("text_or_mask") != "absent"
        blocked = blocked or any(observation.get(key) is not True for key in ("original_half_inspected", "both_windows_inspected", "outcome_blinded"))
        return {"accepted": not blocked, "reasons": ["synthetic_blocker"] if blocked else [],
                "geometryAdequate": True,
                "observation": {**observation, "knee_alias": reference['knee_alias']},
                "cropSha256": hashlib.sha256((Path(candidateDir) / 'crop.npy').read_bytes()).hexdigest(),
                "frozenCandidate": self.pipeline_version, "qcVersion": self.qc_policy_version}
