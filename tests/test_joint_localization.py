"""Tests for the blinded tibiofemoral localization pilot."""

import csv
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from knee.bilateral_separation import normalize_working_copy
from knee.dicom_audit import sha256_bytes, sha256_file
from knee.joint_localization import (
    _extract_spacing,
    _validate_parameters,
    locate_tibiofemoral_joint,
    prepare_joint_localization_pilot,
)


PARAMETERS = {
    "expected_unique_studies": 2,
    "expected_knees": 4,
    "expected_bilateral_algorithm_version": "bilateral_split_v0.2_pilot",
    "algorithm_version": "test_joint_crop",
    "joint_line_search_band": [0.32, 0.70],
    "anatomy_x_band": [0.12, 0.88],
    "compartment_inner_offset_fraction": 0.055,
    "compartment_outer_offset_fraction": 0.23,
    "minimum_compartment_width_pixels": 18,
    "maximum_compartment_peak_distance_fraction": 0.04,
    "compartment_consensus_sigma_fraction": 0.018,
    "compartment_consensus_weight": 0.35,
    "profile_smoothing_fraction": 0.012,
    "bone_offset_fraction": 0.035,
    "expected_joint_line_fraction": 0.52,
    "darkness_weight": 0.45,
    "bone_contrast_weight": 0.40,
    "gradient_weight": 0.15,
    "vertical_center_penalty": 0.08,
    "crop_height_mm": 140.0,
    "crop_width_mm": 140.0,
    "inner_edge_margin_mm": 12.0,
    "accepted_spacing_tags": ["ImagerPixelSpacing", "PixelSpacing"],
    "minimum_score_prominence": 0.10,
    "maximum_boundary_shift_fraction": 0.03,
    "maximum_background_fraction": 0.30,
    "high_confidence_threshold": 0.70,
    "max_preview_width": 800,
}

V03_PARAMETERS = {
    **PARAMETERS,
    "algorithm_version": "tibiofemoral_crop_v0.3_pilot",
    "localization_strategy": "directed_edge_pairs_v0.3",
    "compartment_consensus_weight": 0.25,
    "maximum_compartment_width_difference_mm": 8.0,
    "compartment_width_consensus_weight": 0.10,
    "joint_gap_min_mm": 1.5,
    "joint_gap_max_mm": 18.0,
    "bone_context_mm": 6.0,
    "edge_pair_weight": 0.45,
    "gap_darkness_pair_weight": 0.25,
    "outside_bone_pair_weight": 0.20,
    "pair_center_prior_weight": 0.10,
    "top_edge_pairs_per_compartment": 12,
    "minimum_directed_edge_strength": 0.12,
    "minimum_pair_score": 0.35,
    "artifact_scan_fraction": 0.35,
    "artifact_bright_threshold": 0.90,
    "artifact_component_min_area_pixels": 12,
    "artifact_component_max_area_fraction": 0.015,
    "artifact_component_max_height_fraction": 0.10,
    "artifact_component_max_width_fraction": 0.10,
    "artifact_component_min_fill_ratio": 0.25,
    "artifact_safety_buffer_mm": 4.0,
    "maximum_saturation_fraction": 0.40,
}


def synthetic_half(seed: int = 0) -> np.ndarray:
    rows, columns = 600, 500
    yy, xx = np.mgrid[:rows, :columns]
    image = np.full((rows, columns), 120, dtype=np.float64)
    femur = ((xx - 250) / 170) ** 2 + ((yy - 150) / 150) ** 2 <= 1
    tibia = ((xx - 250) / 180) ** 2 + ((yy - 455) / 150) ** 2 <= 1
    image[femur] = 3100
    image[tibia] = 2800
    image[300:322, 90:410] = 180
    image += ((xx + 2 * yy + seed) % 53).astype(np.float64)
    return image.astype(np.uint16)


def synthetic_bilateral(seed: int = 0) -> np.ndarray:
    left = synthetic_half(seed)
    right = synthetic_half(seed + 7)
    return np.concatenate([left, right], axis=1)


def dicom_bytes(pixels: np.ndarray, include_spacing: bool = True) -> bytes:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = generate_uid()
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.PixelRepresentation = 0
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.NumberOfFrames = 1
    dataset.Modality = "CR"
    if include_spacing:
        dataset.ImagerPixelSpacing = [0.5, 0.5]
    dataset.PixelData = pixels.tobytes()
    stream = io.BytesIO()
    dataset.save_as(stream, enforce_file_format=True)
    return stream.getvalue()


def write_archive(path: Path, payload: bytes) -> None:
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("image.dcm")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


class JointLocalizationTest(unittest.TestCase):
    def test_public_v03_configuration_is_complete_and_valid(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "joint_localization.v0.3.example.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))

        _validate_parameters(config)

        self.assertEqual(config["algorithm_version"], "tibiofemoral_crop_v0.3_pilot")
        self.assertEqual(
            config["localization_strategy"], "directed_edge_pairs_v0.3"
        )
        self.assertEqual(config["expected_knees"], 20)
        self.assertIn("v0_3_piloto", config["output_dir"])

    def test_localization_finds_joint_gap_and_physical_crop(self):
        normalized = normalize_working_copy(synthetic_half(), "MONOCHROME2")
        result = locate_tibiofemoral_joint(normalized, 0.5, 0.5, PARAMETERS)

        self.assertLessEqual(abs(result["joint_center_y"] - 311), 24)
        self.assertLess(abs(result["joint_center_x_half"] - 250), 36)
        self.assertEqual(result["crop_rows"], 280)
        self.assertEqual(result["crop_columns"], 280)
        self.assertAlmostEqual(result["crop_height_mm"], 140.0)
        self.assertAlmostEqual(result["crop_width_mm"], 140.0)
        self.assertLessEqual(result["compartment_peak_distance_fraction"], 0.04)

    def test_compartment_consensus_ignores_a_central_false_gap(self):
        pixels = synthetic_half().astype(np.float64)
        pixels[385:410, 205:295] = 120
        normalized = normalize_working_copy(pixels.astype(np.uint16), "MONOCHROME2")

        result = locate_tibiofemoral_joint(normalized, 0.5, 0.5, PARAMETERS)

        self.assertLessEqual(abs(result["joint_center_y"] - 311), 24)

    def test_inner_edge_guard_keeps_crop_away_from_central_ruler(self):
        normalized = normalize_working_copy(synthetic_half(), "MONOCHROME2")

        right_half = locate_tibiofemoral_joint(
            normalized, 0.5, 0.5, PARAMETERS, inner_edge="RIGHT"
        )
        left_half = locate_tibiofemoral_joint(
            normalized, 0.5, 0.5, PARAMETERS, inner_edge="LEFT"
        )

        expected_margin = round(PARAMETERS["inner_edge_margin_mm"] / 0.5)
        self.assertGreaterEqual(
            normalized.shape[1] - right_half["crop_x1_half"], expected_margin
        )
        self.assertGreaterEqual(left_half["crop_x0_half"], expected_margin)

    def test_v03_uses_directed_edges_around_the_true_joint_gap(self):
        normalized = normalize_working_copy(synthetic_half(), "MONOCHROME2")

        result = locate_tibiofemoral_joint(
            normalized, 0.5, 0.5, V03_PARAMETERS
        )

        self.assertLessEqual(abs(result["joint_center_y"] - 311), 12)
        self.assertLess(result["left_femoral_edge_y"], result["left_tibial_edge_y"])
        self.assertLess(result["right_femoral_edge_y"], result["right_tibial_edge_y"])
        self.assertTrue(result["vertical_edge_gate_passed"])
        self.assertTrue(result["compartment_consensus_gate_passed"])

    def test_v03_rejects_a_false_dark_band_below_the_joint(self):
        pixels = synthetic_half().copy()
        pixels[382:402, 70:430] = 120
        normalized = normalize_working_copy(pixels, "MONOCHROME2")

        result = locate_tibiofemoral_joint(
            normalized, 0.5, 0.5, V03_PARAMETERS
        )

        self.assertLessEqual(abs(result["joint_center_y"] - 311), 16)

    def test_v03_rejects_a_false_dark_band_above_the_joint(self):
        pixels = synthetic_half().copy()
        pixels[228:248, 70:430] = 120
        normalized = normalize_working_copy(pixels, "MONOCHROME2")

        result = locate_tibiofemoral_joint(
            normalized, 0.5, 0.5, V03_PARAMETERS
        )

        self.assertLessEqual(abs(result["joint_center_y"] - 311), 16)

    def test_v03_marks_incompatible_compartments_for_review(self):
        pixels = synthetic_half().copy()
        pixels[300:322, 80:240] = 2800
        pixels[370:392, 80:240] = 120
        normalized = normalize_working_copy(pixels, "MONOCHROME2")
        parameters = {
            **V03_PARAMETERS,
            "maximum_compartment_peak_distance_fraction": 0.02,
            "top_edge_pairs_per_compartment": 1,
        }

        result = locate_tibiofemoral_joint(normalized, 0.5, 0.5, parameters)

        self.assertFalse(result["compartment_consensus_gate_passed"])
        self.assertEqual(
            result["technical_status"],
            "REVIEW_REQUIRED_VERTICAL_DISAGREEMENT",
        )
        self.assertEqual(result["confidence_level"], "LOW")

    def test_v03_expands_the_inner_guard_for_bright_ruler_components(self):
        pixels = synthetic_half().copy()
        for y0 in (220, 270, 330, 380):
            pixels[y0 : y0 + 5, 458:464] = 4095
        normalized = normalize_working_copy(pixels, "MONOCHROME2")

        result = locate_tibiofemoral_joint(
            normalized,
            0.5,
            0.5,
            V03_PARAMETERS,
            inner_edge="RIGHT",
        )

        self.assertGreater(result["detected_artifact_components"], 0)
        self.assertGreater(
            result["effective_inner_margin_mm"],
            V03_PARAMETERS["inner_edge_margin_mm"],
        )
        self.assertGreaterEqual(
            normalized.shape[1] - result["crop_x1_half"],
            round(result["effective_inner_margin_mm"] / 0.5),
        )

    def test_v03_never_declares_high_when_a_mandatory_gate_fails(self):
        normalized = normalize_working_copy(synthetic_half(), "MONOCHROME2")
        parameters = {
            **V03_PARAMETERS,
            "minimum_pair_score": 1.0,
            "high_confidence_threshold": 0.01,
        }

        result = locate_tibiofemoral_joint(
            normalized, 0.5, 0.5, parameters
        )

        self.assertFalse(result["vertical_edge_gate_passed"])
        self.assertFalse(result["mandatory_gates_passed"])
        self.assertEqual(result["confidence_level"], "LOW")

    def test_spacing_is_required_and_explicit(self):
        payload = dicom_bytes(synthetic_bilateral(), include_spacing=False)
        from pydicom import dcmread

        dataset = dcmread(io.BytesIO(payload))
        with self.assertRaisesRegex(ValueError, "valid_pixel_spacing_not_available"):
            _extract_spacing(dataset, ["ImagerPixelSpacing", "PixelSpacing"])

    def test_pilot_uses_frozen_step3_and_writes_blinded_review(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            selected = []
            bilateral_rows = []
            for index in range(2):
                pixels = synthetic_bilateral(index)
                payload = dicom_bytes(pixels)
                package = source / f"private_{index}.tar.gz"
                write_archive(package, payload)
                manifest_key = f"sensitive_{index}"
                selected.append(
                    {
                        "manifest_key": manifest_key,
                        "package_relative_path": package.name,
                        "package_sha256": sha256_file(package),
                        "dicom_sha256": sha256_bytes(payload),
                        "pixel_sha256": sha256_bytes(
                            np.ascontiguousarray(pixels).tobytes()
                        ),
                    }
                )
                bilateral_rows.append(
                    {
                        "case_alias": f"case_{index + 1:03d}",
                        "manifest_key": manifest_key,
                        "split_column": "500",
                        "technical_status": "CANDIDATE_OK",
                        "error_code": "",
                    }
                )

            audit = root / "audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "public_summary": {"status": "ok"},
                        "private_reconciliation": {"selected_packages": selected},
                    }
                ),
                encoding="utf-8",
            )
            bilateral = root / "bilateral"
            bilateral.mkdir()
            results_path = bilateral / "resultados_separacion_privados.csv"
            write_csv(results_path, bilateral_rows)
            (bilateral / "cierre_paso_3_publico.json").write_text(
                json.dumps(
                    {
                        "status": "closed",
                        "parameters_frozen": True,
                        "algorithm_version": "bilateral_split_v0.2_pilot",
                    }
                ),
                encoding="utf-8",
            )
            (bilateral / "parametros_congelados.json").write_text(
                json.dumps(
                    {
                        "status": "frozen_after_blinded_visual_review",
                        "algorithm_version": "bilateral_split_v0.2_pilot",
                        "proposed_mapping": {
                            "image_left": "RIGHT",
                            "image_right": "LEFT",
                        },
                        "results_csv_sha256": sha256_file(results_path),
                    }
                ),
                encoding="utf-8",
            )

            output = root / "output"
            summary = prepare_joint_localization_pilot(
                source, audit, bilateral, output, PARAMETERS
            )

            self.assertEqual(summary["status"], "ready_for_blinded_review")
            self.assertEqual(summary["processed_knees"], 4)
            self.assertFalse(summary["outcome_data_loaded"])
            public_text = (output / "resumen_localizacion_publico.json").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("sensitive_", public_text)
            with (output / "revision_visual_ciega.csv").open(
                newline="", encoding="utf-8-sig"
            ) as stream:
                review = list(csv.DictReader(stream, delimiter=";"))
            self.assertEqual(len(review), 4)
            self.assertEqual(review[0]["knee_alias"], "case_001_RIGHT")
            self.assertTrue((output / review[0]["preview_file"]).is_file())
            self.assertTrue((output / "recortes_nativos/case_001_RIGHT_crop.npy").is_file())

            review[0]["crop_acceptable_yes_no"] = "SI"
            write_csv(output / "revision_visual_ciega.csv", review)
            with self.assertRaisesRegex(RuntimeError, "refusing_to_overwrite"):
                prepare_joint_localization_pilot(
                    source, audit, bilateral, output, PARAMETERS
                )


if __name__ == "__main__":
    unittest.main()
