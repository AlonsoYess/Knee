"""Accepted routine headers retain the frozen development intensity audit."""
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import PropertyMock, patch

import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage

from knee import development_dicom_headers as development
from knee import routine_dicom_headers as h


class RoutineHeadersTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.roots = {"private_sources_root": self.root / "private_sources",
            "project_root": self.root / "project", "runtime_root": self.root / "runtime"}
        self.path = self.roots["private_sources_root"] / "dicom/originales/cohorte_v00/lote/image.tar.gz"
        self.path.parent.mkdir(parents=True)
        self.study = {"src_subject_id": "synthetic", "acquisition_key": "a", "barcode_base": "b",
            "image_file": "private-image", "canonical_source": "cohorte_v00",
            "canonical_package_relative_path": "lote/image.tar.gz"}
        self.row = {"src_subject_id": "synthetic", "acquisition_key": "a", "barcode_base": "b",
            "image_file": "private-image", "role": "routine", "block": "routine_batch_092",
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

    def audit(self, rows=None, inventory=None, **kwargs):
        return h.audit_routine_headers(rows or [self.row], self.roots, inventory or [self.study], **kwargs)

    def test_reuses_exact_development_helpers(self):
        self.assertIs(h._archive_header, development._archive_header)
        self.assertIs(h.compatibility_reasons, development.compatibility_reasons)

    def test_header_allowlist_hashes_and_no_pixel_or_patient_access(self):
        before = h.sha_file(self.path)
        with patch.object(Dataset, "pixel_array", new_callable=PropertyMock,
                          side_effect=AssertionError("pixel decode forbidden")):
            result = self.audit()
        self.assertTrue(result["summary"]["compatible_with_fixed_quantile_native_recipe"])
        self.assertFalse(result["summary"]["pixel_data_decoded"])
        self.assertEqual(set(result["rows"][0]["header"]), set(development.HEADER_TAGS))
        self.assertNotIn("MUST_NOT_BE_RETURNED", str(result))
        self.assertEqual(before, h.sha_file(self.path))

    def test_development_cannot_enter_archive_read(self):
        with patch.object(h, "_archive_header", side_effect=AssertionError("must not read")):
            with self.assertRaisesRegex(ValueError, "only_routine"):
                self.audit([dict(self.row, role="development")])

    def test_duplicate_knee_and_cross_group_subject_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique_routine_knees"):
            self.audit([self.row, dict(self.row)])
        with self.assertRaisesRegex(ValueError, "one_source_block"):
            self.audit([self.row, dict(self.row, side="LEFT", block="routine_batch_091")])

    def test_two_knees_audit_archive_once_and_report_final_progress(self):
        left = dict(self.row, side="LEFT", knee_alias="synthetic_LEFT", half_x0=3, half_x1=6)
        progress = []
        with patch.object(h, "_archive_header", wraps=h._archive_header) as audit:
            result = self.audit([self.row, left], progress=lambda n, total: progress.append((n, total)))
        self.assertEqual(audit.call_count, 1)
        self.assertEqual(result["summary"]["routine_knees"], 2)
        self.assertEqual(result["summary"]["original_dicom_headers_audited"], 1)
        self.assertEqual(progress, [(1, 1)])

    def test_missing_archive_stops_before_any_archive_read_and_aggregates(self):
        missing = dict(self.row, src_subject_id="second-synthetic", knee_alias="second_RIGHT")
        study = dict(self.study, src_subject_id="second-synthetic", canonical_package_relative_path="missing.tar.gz")
        with patch.object(h, "_archive_header", side_effect=AssertionError("availability first")):
            with self.assertRaisesRegex(ValueError, "participants=1;knees=1;blocks=routine_batch_092:1"):
                self.audit([self.row, missing], [self.study, study])

    def test_prefix_source_cache_is_supported_without_central_copy(self):
        self.row["block"] = "routine_batch_003"
        destination = self.roots["project_root"] / (
            "outputs/roi_third_batch_20261004/private_sources/dicom/originales/cohorte_v00/lote/image.tar.gz")
        destination.parent.mkdir(parents=True)
        self.path.rename(destination)
        self.assertEqual(self.audit()["rows"][0]["source"]["root_key"], "project_root")

    def test_first_available_bad_copy_never_falls_back_to_hide_change(self):
        self.row["block"] = "routine_batch_003"
        backup = self.roots["project_root"] / (
            "outputs/roi_third_batch_20261004/private_sources/dicom/originales/cohorte_v00/lote/image.tar.gz")
        backup.parent.mkdir(parents=True)
        backup.write_bytes(self.path.read_bytes())
        self.path.write_bytes(b"changed archive")
        with self.assertRaisesRegex(ValueError, "routine_archive_hash_mismatch"):
            self.audit()

    def test_dicom_pin_photometric_geometry_and_identity_fail(self):
        with self.assertRaisesRegex(ValueError, "dicom_hash_mismatch"):
            self.audit(inventory=[dict(self.study, dicom_sha256="a" * 64)])
        for key, value, error in (("photometric", "MONOCHROME1", "photometric_changed"),
                                 ("half_x1", 7, "native_geometry_changed"),
                                 ("acquisition_key", "other", "membership_identity_changed")):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, error):
                self.audit([dict(self.row, **{key: value})])

    def test_incompatible_presentation_is_reported_and_never_applied(self):
        self.dataset.PresentationLUTShape = "INVERSE"
        self.save()
        result = self.audit()
        self.assertFalse(result["summary"]["compatible_with_fixed_quantile_native_recipe"])
        self.assertEqual(result["summary"]["incompatible_participants"], 1)
        self.assertFalse(result["summary"]["dicom_window_and_voi_applied"])

    def test_path_escape_and_unapproved_block_rejected(self):
        with self.assertRaisesRegex(ValueError, "relative_metadata_path"):
            self.audit(inventory=[dict(self.study, canonical_package_relative_path="../source.tar.gz")])
        for block in ("routine_batch_000", "routine_batch_093", "development_completion_001"):
            with self.subTest(block=block), self.assertRaisesRegex(ValueError, "sealed_routine_block"):
                self.audit([dict(self.row, block=block)])

    def test_input_membership_not_mutated(self):
        rows, inventory = deepcopy([self.row]), deepcopy([self.study])
        self.audit()
        self.assertEqual(rows, [self.row])
        self.assertEqual(inventory, [self.study])


if __name__ == "__main__":
    unittest.main()
