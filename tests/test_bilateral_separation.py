"""Tests for blinded bilateral separation pilot preparation."""

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

from knee.bilateral_separation import (
    estimate_bilateral_split,
    normalize_working_copy,
    prepare_bilateral_pilot,
)
from knee.dicom_audit import sha256_bytes, sha256_file


PARAMETERS = {
    "expected_unique_studies": 2,
    "algorithm_version": "test_algorithm",
    "search_band": [0.40, 0.60],
    "vertical_analysis_band": [0.15, 0.85],
    "smoothing_fraction": 0.015,
    "intensity_weight": 0.70,
    "gradient_weight": 0.30,
    "center_penalty": 0.12,
    "minimum_half_width_ratio": 0.66,
    "minimum_valley_prominence": 0.05,
    "high_confidence_threshold": 0.75,
    "proposed_mapping": {"image_left": "RIGHT", "image_right": "LEFT"},
    "max_preview_width": 800,
}


def synthetic_bilateral(seed: int = 0) -> np.ndarray:
    rows, columns = 400, 600
    yy, xx = np.mgrid[:rows, :columns]
    image = np.full((rows, columns), 100, dtype=np.float64)
    left = ((xx - 170) / 105) ** 2 + ((yy - 220) / 145) ** 2 <= 1
    right = ((xx - 430) / 105) ** 2 + ((yy - 220) / 145) ** 2 <= 1
    image[left | right] = 2300
    image += ((yy % 31) * 2 + seed).astype(np.float64)
    image[:, 292:309] = 25
    return image.astype(np.uint16)


def dicom_bytes(pixels: np.ndarray, photometric: str = "MONOCHROME2") -> bytes:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = generate_uid()
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = photometric
    dataset.PixelRepresentation = 0
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.NumberOfFrames = 1
    dataset.Modality = "CR"
    dataset.Laterality = "L"
    dataset.PixelData = pixels.tobytes()
    stream = io.BytesIO()
    dataset.save_as(stream, enforce_file_format=True)
    return stream.getvalue()


def write_archive(path: Path, payload: bytes) -> None:
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("image.dcm")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))


class BilateralSeparationTest(unittest.TestCase):
    def test_adaptive_split_finds_the_central_valley(self):
        normalized = normalize_working_copy(synthetic_bilateral(), "MONOCHROME2")
        result = estimate_bilateral_split(normalized, PARAMETERS)
        self.assertLess(abs(result["split_fraction"] - 0.5), 0.025)
        self.assertGreater(result["half_width_ratio"], 0.9)

    def test_monochrome_one_is_inverted_only_in_the_working_copy(self):
        original = synthetic_bilateral()
        inverted = np.iinfo(np.uint16).max - original
        normalized_two = normalize_working_copy(original, "MONOCHROME2")
        normalized_one = normalize_working_copy(inverted, "MONOCHROME1")
        np.testing.assert_allclose(normalized_two, normalized_one, atol=1e-6)
        self.assertEqual(inverted.dtype, np.uint16)

    def test_preparation_uses_closed_audit_and_writes_pseudonymous_review(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            selected = []
            for index in range(2):
                pixels = synthetic_bilateral(index)
                payload = dicom_bytes(pixels)
                package = source / f"private_{index}.tar.gz"
                write_archive(package, payload)
                selected.append(
                    {
                        "manifest_key": f"sensitive_{index}",
                        "package_relative_path": package.name,
                        "package_sha256": sha256_file(package),
                        "dicom_sha256": sha256_bytes(payload),
                        "pixel_sha256": sha256_bytes(
                            np.ascontiguousarray(pixels).tobytes()
                        ),
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
            output = root / "output"
            summary = prepare_bilateral_pilot(source, audit, output, PARAMETERS)

            self.assertEqual(summary["status"], "ready_for_blinded_review")
            self.assertFalse(summary["outcome_data_loaded"])
            public_text = (output / "resumen_separacion_publico.json").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("sensitive_", public_text)
            with (output / "revision_visual_ciega.csv").open(
                newline="", encoding="utf-8-sig"
            ) as stream:
                review = list(csv.DictReader(stream, delimiter=";"))
            self.assertEqual([row["case_alias"] for row in review], ["case_001", "case_002"])
            self.assertTrue((output / review[0]["preview_file"]).is_file())
            self.assertEqual(review[0]["proposed_image_left_side"], "RIGHT")
            self.assertEqual(summary["laterality_mapping"]["dicom_laterality_used_as_decision"], False)

            review[0]["split_acceptable_yes_no"] = "SI"
            with (output / "revision_visual_ciega.csv").open(
                "w", newline="", encoding="utf-8-sig"
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=review[0].keys(), delimiter=";")
                writer.writeheader()
                writer.writerows(review)
            with self.assertRaisesRegex(RuntimeError, "refusing_to_overwrite"):
                prepare_bilateral_pilot(source, audit, output, PARAMETERS)


if __name__ == "__main__":
    unittest.main()
