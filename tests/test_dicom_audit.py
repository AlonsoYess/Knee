"""Pilot DICOM reconciliation and privacy-safe reporting tests."""

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

from knee.dicom_audit import audit_pilot, canonical_package_key, write_audit


def dicom_bytes(seed: int) -> bytes:
    pixels = (np.arange(24, dtype=np.uint16).reshape(4, 6) + seed)
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
    dataset.ViewPosition = "PA"
    dataset.PixelSpacing = [0.17, 0.17]
    dataset.PixelData = pixels.tobytes()
    stream = io.BytesIO()
    dataset.save_as(stream, enforce_file_format=True)
    return stream.getvalue()


def write_archive(path: Path, payload: bytes) -> None:
    mode = "w:gz" if path.name.endswith(".gz") else "w"
    with tarfile.open(path, mode) as archive:
        info = tarfile.TarInfo("001")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))


class PilotDicomAuditTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "dicom"
        self.source.mkdir()
        self.manifest = self.root / "pilot.csv"

    def write_manifest(self, names: list[str]) -> None:
        with self.manifest.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["N_MUESTRA", "NOMBRE_ARCHIVO"], delimiter=";"
            )
            writer.writeheader()
            for number, name in enumerate(names, start=1):
                writer.writerow({"N_MUESTRA": number, "NOMBRE_ARCHIVO": name})

    def test_encoded_paths_and_archive_extensions_share_a_key(self):
        self.assertEqual(canonical_package_key("path%2Fstudy_a.tar"), "study_a")
        self.assertEqual(canonical_package_key("study_a.tar.gz"), "study_a")

    def test_duplicate_packaging_and_omission_are_detected_by_content(self):
        self.write_manifest(["study_a.tar.gz", "study_b.tar.gz", "study_c.tar.gz"])
        first = dicom_bytes(1)
        write_archive(self.source / "study_a.tar", first)
        write_archive(self.source / "study_a.tar.gz", first)
        write_archive(self.source / "study_b.tar", dicom_bytes(2))

        result = audit_pilot(self.source, self.manifest, expected_unique_studies=3)
        summary = result["public_summary"]

        self.assertEqual(summary["status"], "fail")
        self.assertEqual(summary["selected_unique_acquisitions"], 2)
        self.assertEqual(summary["duplicate_package_copies"], 1)
        self.assertEqual(summary["checks"]["missing_expected_studies"], 1)
        self.assertEqual(summary["checks"]["duplicate_package_copies"], 1)

    def test_clean_manifest_defined_pilot_passes_and_writes_public_summary(self):
        names = ["study_a.tar.gz", "study_b.tar.gz", "study_c.tar.gz"]
        self.write_manifest(names)
        for index, name in enumerate(names, start=1):
            write_archive(self.source / name, dicom_bytes(index))

        result = audit_pilot(self.source, self.manifest, expected_unique_studies=3)
        output = self.root / "output"
        write_audit(result, output)
        public = json.loads(
            (output / "auditoria_piloto_publica.json").read_text(encoding="utf-8")
        )

        self.assertEqual(public["status"], "ok")
        self.assertEqual(public["selected_unique_acquisitions"], 3)
        self.assertEqual(public["technical_profile"]["modality"], {"CR": 3})
        self.assertNotIn("study_a", json.dumps(public))

    def test_same_dicom_under_two_manifest_keys_is_not_counted_twice(self):
        self.write_manifest(["study_a.tar", "study_b.tar"])
        payload = dicom_bytes(7)
        write_archive(self.source / "study_a.tar", payload)
        write_archive(self.source / "study_b.tar", payload)

        summary = audit_pilot(
            self.source, self.manifest, expected_unique_studies=2
        )["public_summary"]

        self.assertEqual(summary["status"], "fail")
        self.assertEqual(summary["selected_unique_acquisitions"], 1)
        self.assertEqual(summary["checks"]["cross_key_duplicate_acquisitions"], 1)


if __name__ == "__main__":
    unittest.main()
