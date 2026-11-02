"""Pinned header audit cannot cross roles, read pixels, or hide intensity changes."""
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch, PropertyMock

import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage

from knee import development_dicom_headers as h


class DevelopmentHeadersTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.roots = {"project_root": str(self.root / "project"),
                      "development_root": str(self.root / "development")}
        self.path = self.root / "development/sources/raw/cohorte_v00/folder/image.tar.gz"
        self.path.parent.mkdir(parents=True)
        self.study = {"src_subject_id": "synthetic", "acquisition_key": "a", "barcode_base": "b",
            "image_file": "private-image", "canonical_source": "cohorte_v00",
            "canonical_package_relative_path": "folder/image.tar.gz"}
        self.row = {"src_subject_id": "synthetic", "acquisition_key": "a", "barcode_base": "b",
            "image_file": "private-image", "role": "development", "block": "development_completion_001",
            "side": "RIGHT", "knee_alias": "synthetic_RIGHT", "photometric": "MONOCHROME2",
            "source_dtype": "uint16", "source_shape": [4, 3], "half_x0": 0, "half_x1": 3}
        self.dataset = FileDataset(None, {}, file_meta=FileMetaDataset(), preamble=b"\0" * 128)
        self.dataset.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        self.dataset.file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        self.dataset.file_meta.MediaStorageSOPInstanceUID = "1.2.826.0.1.3680043.10.543.1"
        self.dataset.Rows, self.dataset.Columns = 4, 6
        self.dataset.SamplesPerPixel = 1
        self.dataset.PhotometricInterpretation = "MONOCHROME2"
        self.dataset.BitsAllocated, self.dataset.BitsStored, self.dataset.HighBit = 16, 12, 11
        self.dataset.PixelRepresentation = 0
        self.dataset.RescaleSlope, self.dataset.RescaleIntercept = 1, 0
        self.dataset.PatientName = "MUST_NOT_BE_RETURNED"
        self.dataset.PatientID = "MUST_NOT_BE_RETURNED"
        self.dataset.PixelData = b"\0" * 48
        self.save()

    def save(self):
        buffer = io.BytesIO()
        pydicom.dcmwrite(buffer, self.dataset, enforce_file_format=True)
        data = buffer.getvalue()
        with tarfile.open(self.path, "w:gz") as archive:
            member = tarfile.TarInfo("image.dcm")
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        self.study.update(package_sha256=h.sha_file(self.path), dicom_sha256=hashlib.sha256(data).hexdigest())

    def audit(self, rows=None):
        return h.audit_development_headers(rows or [self.row], self.roots, [self.study])

    def test_hash_verified_allowlist_without_decode_or_patient_tags(self):
        before = h.sha_file(self.path)
        with patch.object(Dataset, "pixel_array", new_callable=PropertyMock,
                          side_effect=AssertionError("pixel decoding forbidden")):
            result = self.audit()
        self.assertTrue(result["summary"]["compatible_with_fixed_quantile_native_recipe"])
        self.assertFalse(result["summary"]["pixel_data_decoded"])
        self.assertEqual(set(result["rows"][0]["header"]), set(h.HEADER_TAGS))
        self.assertNotIn("MUST_NOT_BE_RETURNED", str(result))
        self.assertEqual(before, h.sha_file(self.path))

    def test_routine_source_rejected_before_any_archive_read(self):
        row = dict(self.row, role="routine")
        with patch.object(h, "_archive_header", side_effect=AssertionError("must not read")):
            with self.assertRaisesRegex(ValueError, "only_development"):
                self.audit([row])

    def test_duplicate_knee_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique_development_knees"):
            self.audit([self.row, dict(self.row)])

    def test_two_knees_share_one_pinned_header(self):
        left = dict(self.row, side="LEFT", knee_alias="synthetic_LEFT", half_x0=3, half_x1=6)
        result = self.audit([self.row, left])
        self.assertEqual(result["summary"]["development_knees"], 2)
        self.assertEqual(result["summary"]["original_dicom_headers_audited"], 1)

    def test_changed_package_fails_before_header_read(self):
        self.study["package_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "archive_hash_mismatch"):
            self.audit()

    def test_changed_dicom_pin_fails(self):
        self.study["dicom_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "dicom_hash_mismatch"):
            self.audit()

    def test_identity_and_photometric_checked(self):
        for key, value in (("acquisition_key", "other"), ("photometric", "MONOCHROME1")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.audit([dict(self.row, **{key: value})])

    def test_nonlinear_modality_lut_is_incompatible(self):
        self.dataset.ModalityLUTSequence = [Dataset()]
        self.save()
        result = self.audit()
        self.assertFalse(result["summary"]["compatible_with_fixed_quantile_native_recipe"])
        self.assertEqual(result["rows"][0]["compatibility_reasons"],
                         ["modality_lut_requires_separate_resolution"])

    def test_missing_positive_affine_and_negative_rescale(self):
        header = self.audit()["rows"][0]["header"]
        for slope, intercept, okay in ((None, None, True), (2.0, -4.0, True), (-1.0, 0.0, False),
                                       (0.0, 0.0, False), (None, 0.0, False), (float("nan"), 0.0, False)):
            with self.subTest(slope=slope, intercept=intercept):
                amended = dict(header, RescaleSlope=slope, RescaleIntercept=intercept)
                self.assertEqual(not h.compatibility_reasons(amended), okay)

    def test_voi_and_padding_are_recorded_without_application(self):
        self.dataset.VOILUTSequence = [Dataset()]
        self.dataset.WindowCenter, self.dataset.WindowWidth = 50, 100
        self.dataset.PixelPaddingValue = 0
        self.save()
        result = self.audit()
        self.assertTrue(result["summary"]["compatible_with_fixed_quantile_native_recipe"])
        self.assertEqual(result["rows"][0]["header"]["VOILUTSequence"], {"items": 1})
        self.assertFalse(result["summary"]["dicom_window_and_voi_applied"])
        self.assertFalse(result["summary"]["padding_pixels_masked_or_repaired"])

    def test_path_escape_is_rejected(self):
        self.study["canonical_package_relative_path"] = "../image.tar.gz"
        with self.assertRaisesRegex(ValueError, "relative_metadata_path"):
            self.audit()

    def test_present_inversion_and_multiframe_incompatible(self):
        header = self.audit()["rows"][0]["header"]
        self.assertIn("presentation_lut_requires_separate_resolution",
            h.compatibility_reasons(dict(header, PresentationLUTShape="INVERSE")))
        self.assertIn("single_plane_grayscale_required",
            h.compatibility_reasons(dict(header, NumberOfFrames=2)))


if __name__ == "__main__":
    unittest.main()
