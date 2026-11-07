"""Wholly invented DICOM fixtures. No patient/cohort image or model is read."""
import io

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ComputedRadiographyImageStorage, ExplicitVRLittleEndian, generate_uid


def synthetic_pixels():
    y, x = np.meshgrid(np.arange(241) * .5, np.arange(401) * .5, indexing="ij")
    sigmoid = lambda value: 1 / (1 + np.exp(np.clip(-value, -700, 700)))
    lateral = sigmoid((x - 60) / .6) * sigmoid((140 - x) / .6)
    half = np.rint(100 + 1800 * lateral * (sigmoid((55 - y) / .6) + sigmoid((y - 65) / .6))).astype(np.uint16)
    return np.concatenate((half, half), axis=1)


def dicom_bytes(*, photometric="MONOCHROME2", spacing=(.5, .5), pixels=None, patient_name="SYNTHETIC^ONLY"):
    pixels = synthetic_pixels() if pixels is None else pixels
    metadata = FileMetaDataset()
    metadata.MediaStorageSOPClassUID = ComputedRadiographyImageStorage
    metadata.MediaStorageSOPInstanceUID = generate_uid()
    metadata.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(None, {}, file_meta=metadata, preamble=b"\0" * 128)
    dataset.SOPClassUID = metadata.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = metadata.MediaStorageSOPInstanceUID
    dataset.PatientName = patient_name
    dataset.PatientID = "NOT-A-PATIENT"
    dataset.StudyDate = "20260101"
    dataset.Modality = "CR"
    dataset.ViewPosition = "PA"
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = photometric
    dataset.PixelRepresentation = 0
    dataset.BitsAllocated = dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.NumberOfFrames = 1
    dataset.ImagerPixelSpacing = list(spacing)
    dataset.PixelData = pixels.tobytes()
    target = io.BytesIO()
    dataset.save_as(target, enforce_file_format=True)
    return target.getvalue()


def source_reference(*, center=(200, 120), resolved=True, side="right"):
    return {"confirmedSide": side, "centerResolved": resolved,
            "centerXY": list(center) if resolved else None,
            "originalAndHalfInspected": True, "outcomeBlinded": True,
            "note": "Synthetic original and native half inspected before any automatic ROI was displayed."}


def qc_observation(**changes):
    value = {"side_supported": True, "coverage_complete": True,
             "visibility": {"medial_joint_borders": "visible", "lateral_joint_borders": "visible", "subchondral_bone": "visible"},
             "hardware": "absent", "text_or_mask": "absent", "marks": [], "mark_effect": "none",
             "physical_origin_known": False,
             "note": "Synthetic original and both native ROI windows inspected for this software test only.",
             "original_half_inspected": True, "both_windows_inspected": True, "outcome_blinded": True,
             "lateral_margins_visible": True, "unclassified_mark_present": False,
             "ruler_outside_bone": None, "ruler_locations_xy": []}
    return {**value, **changes}
