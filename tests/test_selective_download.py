"""Tests for exact-list NDA download preparation and promotion."""

import csv
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
import tarfile

from knee.selective_download import (
    audit_downloaded_batch,
    prepare_batch,
    promote_downloaded_batch,
    write_batch_evidence,
)


def dicom_bytes(seed: int) -> bytes:
    pixels = np.arange(24, dtype=np.uint16).reshape(4, 6) + seed
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
    dataset.PixelData = pixels.tobytes()
    stream = io.BytesIO()
    dataset.save_as(stream, enforce_file_format=True)
    return stream.getvalue()


def write_archive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo("image.dcm")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))


class SelectiveDownloadTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.queue = self.root / "queue.csv"
        with self.queue.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(
                stream,
                delimiter=";",
                fieldnames=[
                    "download_batch",
                    "inventory_index",
                    "acquisition_key",
                    "image_file",
                    "local_status",
                    "action_required",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "download_batch": 1,
                    "inventory_index": 1,
                    "acquisition_key": "image_10000001",
                    "image_file": "s3://bucket/path/image_10000001.tar.gz",
                    "local_status": "PENDIENTE_DESCARGA",
                    "action_required": "DESCARGAR",
                }
            )
            writer.writerow(
                {
                    "download_batch": 1,
                    "inventory_index": 2,
                    "acquisition_key": "image_10000002",
                    "image_file": "s3://bucket/path/image_10000002.tar.gz",
                    "local_status": "PENDIENTE_DESCARGA",
                    "action_required": "DESCARGAR",
                }
            )
            writer.writerow(
                {
                    "download_batch": 2,
                    "inventory_index": 3,
                    "acquisition_key": "image_10000003",
                    "image_file": "s3://bucket/path/image_10000003.tar.gz",
                    "local_status": "PENDIENTE_DESCARGA",
                    "action_required": "DESCARGAR",
                }
            )

    def test_prepare_batch_writes_only_exact_s3_urls(self):
        batch = self.root / "batch.txt"
        summary = prepare_batch(self.queue, 1, batch)

        self.assertEqual(summary["status"], "ready")
        self.assertEqual(summary["expected_files"], 2)
        self.assertEqual(
            batch.read_text(encoding="utf-8").splitlines(),
            [
                "s3://bucket/path/image_10000001.tar.gz",
                "s3://bucket/path/image_10000002.tar.gz",
            ],
        )

    def test_complete_batch_is_audited_then_promoted(self):
        batch = self.root / "batch.txt"
        prepare_batch(self.queue, 1, batch)
        downloaded = self.root / "downloaded"
        write_archive(downloaded / "nested" / "image_10000001.tar.gz", dicom_bytes(1))
        write_archive(downloaded / "image_10000002.tar.gz", dicom_bytes(2))

        destination = self.root / "drive"
        result = promote_downloaded_batch(downloaded, batch, destination, 1)
        evidence = self.root / "evidence"
        write_batch_evidence(result, evidence)

        self.assertEqual(result["public_summary"]["status"], "ok")
        self.assertEqual(result["public_summary"]["copied_to_drive"], 2)
        self.assertEqual(len(list((destination / "lote_001").glob("*.tar.gz"))), 2)
        self.assertTrue((evidence / "lote_001_resumen_publico.json").is_file())
        self.assertTrue((evidence / "lote_001_evidencia_privada.json").is_file())

    def test_incomplete_batch_is_not_promoted(self):
        batch = self.root / "batch.txt"
        prepare_batch(self.queue, 1, batch)
        downloaded = self.root / "downloaded"
        write_archive(downloaded / "image_10000001.tar.gz", dicom_bytes(1))
        destination = self.root / "drive"

        audit = audit_downloaded_batch(downloaded, batch)
        self.assertEqual(audit["public_summary"]["status"], "fail")
        self.assertEqual(audit["public_summary"]["checks"], {"missing": 1})
        with self.assertRaisesRegex(ValueError, "no package was promoted"):
            promote_downloaded_batch(downloaded, batch, destination, 1)
        self.assertFalse(destination.exists())

    def test_existing_identical_package_is_not_overwritten(self):
        batch = self.root / "batch.txt"
        prepare_batch(self.queue, 1, batch)
        downloaded = self.root / "downloaded"
        first = downloaded / "image_10000001.tar.gz"
        second = downloaded / "image_10000002.tar.gz"
        write_archive(first, dicom_bytes(1))
        write_archive(second, dicom_bytes(2))
        destination = self.root / "drive"

        first_result = promote_downloaded_batch(downloaded, batch, destination, 1)
        second_result = promote_downloaded_batch(downloaded, batch, destination, 1)

        self.assertEqual(first_result["public_summary"]["copied_to_drive"], 2)
        self.assertEqual(second_result["public_summary"]["copied_to_drive"], 0)
        self.assertEqual(
            second_result["public_summary"]["already_present_identical"], 2
        )

    def test_incomplete_existing_lot_is_left_unchanged(self):
        batch = self.root / "batch.txt"
        prepare_batch(self.queue, 1, batch)
        downloaded = self.root / "downloaded"
        write_archive(downloaded / "image_10000001.tar.gz", dicom_bytes(1))
        write_archive(downloaded / "image_10000002.tar.gz", dicom_bytes(2))
        destination = self.root / "drive"
        lot = destination / "lote_001"
        lot.mkdir(parents=True)
        sentinel = lot / "existing.txt"
        sentinel.write_text("preserve", encoding="utf-8")

        with self.assertRaisesRegex(FileExistsError, "incomplete or contains extra"):
            promote_downloaded_batch(downloaded, batch, destination, 1)

        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")
        self.assertEqual([path.name for path in lot.iterdir()], ["existing.txt"])


if __name__ == "__main__":
    unittest.main()
