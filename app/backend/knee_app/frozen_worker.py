"""Small JSON worker around unchanged scientific functions, never a model trainer.

Run only with the configured frozen Python and checkout. Inputs and outputs are
private application files; no cohort inventory, labels or reserved partition is
opened. A synthetic test may inject a predictor into ``crop`` in process; that
injection is not exposed by the JSON protocol or production processor.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import tarfile
import tempfile
import uuid

sys.dont_write_bytecode = True

COMPONENTS = {
    'src/knee/roi_contact_geometry.py': 'bbf42ba23acc38fa4b7157d1898bcc75bd392a6650bcea19fdb839deeae897ed',
    'src/knee/roi_contour_geometry.py': '39e488d79ae25253925c4b9f1a3e7998a15fa39b2af8739c49b5e5059b752607',
    'src/knee/roi_landmark_geometry.py': 'd613f8816d6d110d32d1c5923ffe1048a43e2115ddd2737906c4eb12352bdf16',
    'src/knee/roi_task_qc.py': 'e143855d41ef997964b4fe6fc8a33f700ebee1c7b54b6e7f34f7db90acf9ef94',
    'src/knee/roi_confirmation_v2.py': '4e8532cebdd54b1999dc104592d7f5091a5aa21ea57584259475107e78d6c547',
}
CANDIDATE = 'kneel_guided_gradient_contours_contact_anchor_v1'
MAX_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_PIXELS = 32_000_000  # Resource bound, not a clinical eligibility criterion.
VERSION = 'application_frozen_preparation_v1'


class WorkerError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def sha_file(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_json(path):
    path = Path(path)
    if path.stat().st_size > 1_048_576:
        raise WorkerError('PROCESSING_FAILED')
    return json.loads(path.read_text(encoding='utf-8'))


def safe(root, path):
    root, original = Path(root).resolve(), Path(path).absolute()
    target = original.resolve()
    if target == root or not target.is_relative_to(root):
        raise WorkerError('PROCESSING_FAILED')
    for item in (original, *original.parents):
        if item == root:
            break
        if item.is_symlink() or getattr(item, 'is_junction', lambda: False)():
            raise WorkerError('PROCESSING_FAILED')
    return target


def output_directory(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise WorkerError('PROCESSING_FAILED')
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':'))
    if os.name != 'nt':
        os.chmod(path, 0o600)


def save_array(path, array):
    import numpy as np
    with Path(path).open('xb') as handle:
        np.save(handle, array, allow_pickle=False)
    saved = np.load(path, allow_pickle=False)
    if saved.dtype != array.dtype or not np.array_equal(saved, array):
        raise WorkerError('PROCESSING_FAILED')
    if os.name != 'nt':
        os.chmod(path, 0o600)


def verify_components(checkout):
    checkout = Path(checkout).resolve()
    if platform.python_version() != '3.12.14':
        raise WorkerError('PROCESSOR_UNAVAILABLE')
    for relative, expected in COMPONENTS.items():
        path = checkout / relative
        if not path.is_file() or sha_file(path) != expected:
            raise WorkerError('PROCESSOR_UNAVAILABLE')
    policy = read_json(checkout / 'configs/roi_mcr009_contact_confirmation.json')
    if policy.get('algorithm_version') != CANDIDATE or policy.get('scientific_freeze', {}).get('component_sha256') != COMPONENTS:
        raise WorkerError('PROCESSOR_UNAVAILABLE')
    if str(checkout / 'src') not in sys.path:
        sys.path.insert(0, str(checkout / 'src'))
    import knee
    if not Path(knee.__file__).resolve().is_relative_to(checkout / 'src'):
        raise WorkerError('PROCESSOR_UNAVAILABLE')
    # Source, display and QC use the same frozen non-neural dependencies.
    import importlib.metadata
    from knee.roi_second_batch_local_execution import PACKAGES
    for name in ('numpy', 'scipy', 'opencv-python-headless', 'pydicom', 'Pillow'):
        if importlib.metadata.version(name) != PACKAGES[name]:
            raise WorkerError('PROCESSOR_UNAVAILABLE')
    return checkout


def render_windows(array, photometric, output, prefix):
    import numpy as np
    from PIL import Image
    from knee.roi_refinement_pilot import display_pixels
    artifacts = {}
    for suffix, percentiles in (('a', (1, 99)), ('b', (5, 95))):
        low, high = (float(value) for value in np.percentile(array, percentiles))
        if high <= low:
            high = low + 1
        name = f'{prefix}_{suffix}.png'
        # Native dimensions are retained. These eight-bit views are never
        # scientific inputs or replacements for the integer native arrays.
        Image.fromarray(display_pixels(array, photometric, [low, high])).save(output / name)
        if os.name != 'nt':
            os.chmod(output / name, 0o600)
        artifacts[f'{prefix}_{suffix}'] = name
    return artifacts


def source(source_path, side, out_dir):
    import numpy as np
    import pydicom
    from knee.bilateral_separation import _read_single_image, normalize_working_copy, estimate_bilateral_split
    from knee.development_dicom_headers import HEADER_TAGS, _header, compatibility_reasons
    from knee.joint_localization import _extract_spacing
    from knee.roi_confirmation import BILATERAL
    from knee.roi_supervised_qc import sha_pixels

    source_path = Path(source_path)
    if side not in {'right', 'left', 'RIGHT', 'LEFT'} or not source_path.is_file() or not 0 < source_path.stat().st_size <= MAX_UPLOAD_BYTES:
        raise WorkerError('IMAGE_REJECTED')
    side = side.upper()
    try:
        header_dataset = pydicom.dcmread(source_path, force=True, stop_before_pixels=True,
            specific_tags=list(HEADER_TAGS) + ['ImagerPixelSpacing', 'PixelSpacing'])
        header = _header(header_dataset)
        if compatibility_reasons(header):
            raise WorkerError('IMAGE_REJECTED')
        if header['Rows'] * header['Columns'] > MAX_PIXELS or min(header['Rows'], header['Columns']) < 32:
            raise WorkerError('IMAGE_REJECTED')
        # Only uncompressed transfer syntaxes are enabled for this bounded
        # upload adapter. Adding codecs requires separate engineering checks.
        transfer = str(getattr(header_dataset.file_meta, 'TransferSyntaxUID', ''))
        if transfer not in {'1.2.840.10008.1.2', '1.2.840.10008.1.2.1', '1.2.840.10008.1.2.2'}:
            raise WorkerError('IMAGE_REJECTED')
        out = output_directory(out_dir)
        # The unchanged reader consumes a TAR. This one-member temporary
        # envelope preserves uploaded DICOM bytes and accepts no user archive.
        with tempfile.TemporaryDirectory(prefix='dicom-bridge-', dir=out) as temporary:
            archive = Path(temporary) / 'single.tar'
            with tarfile.open(archive, 'w') as handle:
                info = tarfile.TarInfo('image.dcm')
                info.size = source_path.stat().st_size
                with source_path.open('rb') as incoming:
                    handle.addfile(info, incoming)
            payload, dataset, pixels = _read_single_image(archive)
        if hashlib.sha256(payload).hexdigest() != sha_file(source_path):
            raise WorkerError('PROCESSING_FAILED')
        if pixels.ndim != 2 or pixels.dtype != np.uint16 or list(pixels.shape) != [header['Rows'], header['Columns']]:
            raise WorkerError('IMAGE_REJECTED')
        dr, dc, spacing = _extract_spacing(dataset, ['ImagerPixelSpacing', 'PixelSpacing'])
        photo = str(dataset.PhotometricInterpretation)
        original_hash = sha_pixels(pixels)
        split = estimate_bilateral_split(normalize_working_copy(pixels, photo), BILATERAL)
        column = int(split['split_column'])
        if not 0 < column < pixels.shape[1]:
            raise WorkerError('IMAGE_REJECTED')
        start, end = (0, column) if side == 'RIGHT' else (column, pixels.shape[1])
        half = np.ascontiguousarray(pixels[:, start:end])
        case = 'app_' + uuid.uuid4().hex
        meta = {'knee_alias': case + '_' + side, 'case_alias': case, 'side': side,
            'source_status': 'ready', 'automatic_status': 'not_run',
            'source_shape': list(half.shape), 'source_dtype': str(half.dtype),
            'source_half_sha256': sha_pixels(half), 'half_x0': start, 'half_x1': end,
            'photometric': photo, 'row_spacing_mm': dr, 'column_spacing_mm': dc,
            'spacing_tag': spacing, 'bilateral_split': split}
        save_array(out / 'original.npy', pixels)
        save_array(out / 'half.npy', half)
        artifacts = {**render_windows(pixels, photo, out, 'full'), **render_windows(half, photo, out, 'half')}
        if sha_pixels(pixels) != original_hash:
            raise WorkerError('PROCESSING_FAILED')
        result = {'version': VERSION, 'side': side.lower(), 'fullNativeShape': list(pixels.shape),
            'nativeHalfShape': list(half.shape), 'sourceMeta': meta, 'artifacts': artifacts,
            'fullPixelSha256': original_hash, 'dicomSha256': hashlib.sha256(payload).hexdigest(),
            'header': header, 'frozenCandidate': CANDIDATE}
        write_json(out / 'source.json', result)
        return {**result, 'sourceManifestSha256': sha_file(out / 'source.json')}
    except WorkerError:
        raise
    except OSError as exc:
        raise WorkerError('PROCESSING_FAILED') from exc
    except (ValueError, AttributeError, KeyError, TypeError) as exc:
        raise WorkerError('IMAGE_REJECTED') from exc


def load_source(source_dir):
    import numpy as np
    from knee.roi_supervised_qc import sha_pixels
    directory = Path(source_dir)
    record = read_json(directory / 'source.json')
    if record.get('version') != VERSION or record.get('frozenCandidate') != CANDIDATE:
        raise WorkerError('PROCESSING_FAILED')
    full = np.load(directory / 'original.npy', allow_pickle=False)
    half = np.load(directory / 'half.npy', allow_pickle=False)
    meta = record['sourceMeta']
    if (full.ndim != 2 or half.ndim != 2 or full.dtype != np.uint16 or half.dtype != np.uint16
            or full.size > MAX_PIXELS or list(full.shape) != record['fullNativeShape']
            or list(half.shape) != record['nativeHalfShape'] or list(half.shape) != meta['source_shape']
            or sha_pixels(full) != record['fullPixelSha256'] or sha_pixels(half) != meta['source_half_sha256']
            or meta['side'] not in {'RIGHT', 'LEFT'} or record['side'] != meta['side'].lower()
            or not (0 <= meta['half_x0'] < meta['half_x1'] <= full.shape[1])
            or not np.array_equal(half, full[:, meta['half_x0']:meta['half_x1']])):
        raise WorkerError('PROCESSING_FAILED')
    return record, full, half


def validate_reference(source_dir, reference):
    from knee.roi_routine_contract import _center, SOURCE_FIELDS
    record, _, _ = load_source(source_dir)
    meta = record['sourceMeta']
    if not isinstance(reference, dict):
        raise WorkerError('SOURCE_REFERENCE_INVALID')
    value = dict(reference)
    if 'knee_alias' in value and value['knee_alias'] != meta['knee_alias']:
        raise WorkerError('SOURCE_REFERENCE_INVALID')
    value['knee_alias'] = meta['knee_alias']
    try:
        _center({key: meta[key] for key in SOURCE_FIELDS}, value)
    except (ValueError, TypeError, KeyError) as exc:
        raise WorkerError('SOURCE_REFERENCE_INVALID') from exc
    return value


def create_predictor(checkout):
    from knee.roi_routine_assets import FrozenKneelPredictor
    return FrozenKneelPredictor(checkout)


def crop(source_dir, out_dir, checkout, predictor=None):
    import numpy as np
    from knee.roi_contact_confirmation import native_candidate
    from knee.roi_supervised_qc import sha_pixels
    record, full, half = load_source(source_dir)
    meta = record['sourceMeta']
    output = output_directory(out_dir)
    # Reference centers and QC observations never enter this function.
    if meta['row_spacing_mm'] != meta['column_spacing_mm']:
        row, native = {'status': 'abstain', 'error_code': 'frozen_localizer_requires_isotropic_spacing'}, None
    else:
        try:
            localizer = predictor if predictor is not None else create_predictor(checkout)
        except (ValueError, RuntimeError, OSError, ImportError) as exc:
            raise WorkerError('PROCESSOR_UNAVAILABLE') from exc
        try:
            points = np.asarray(localizer(full.copy(), dict(meta)), dtype=float)
            if points.shape != (2, 16, 2) or not np.isfinite(points).all():
                raise ValueError('unexpected_frozen_landmark_shape')
        except ValueError as exc:
            reason = str(exc)
            row, native = {'status': 'abstain', 'error_code': reason if re.fullmatch('[a-z][a-z0-9_]{0,119}', reason) else 'frozen_localizer_abstained'}, None
        else:
            local = points[0 if meta['side'] == 'RIGHT' else 1].copy()
            local[:, 0] -= meta['half_x0']
            # Match Runner.prepare: native_candidate already reports its
            # ordinary abstentions. Unexpected integrity/geometry exceptions
            # must stop processing, not be relabeled as routine exclusions.
            row, native = native_candidate(meta, half, local)
    if sha_pixels(full) != record['fullPixelSha256'] or sha_pixels(half) != meta['source_half_sha256']:
        raise WorkerError('PROCESSING_FAILED')
    result = {'version': VERSION, 'status': 'candidate' if native is not None else 'abstain',
        'reason': None if native is not None else row.get('error_code', 'automatic_abstention'),
        'box': row.get('box') if native is not None else None,
        'nativeHalfBox': row.get('box') if native is not None else None,
        'cropNativeShape': list(native.shape) if native is not None else None,
        'sourceManifestSha256': sha_file(Path(source_dir) / 'source.json'),
        'candidateMeta': row, 'artifacts': {}}
    if native is not None:
        save_array(output / 'crop.npy', native)
        result['artifacts'] = render_windows(native, meta['photometric'], output, 'crop')
    write_json(output / 'candidate.json', result)
    return result


def assess(source_dir, candidate_dir, reference, observation):
    import numpy as np
    from knee.roi_confirmation_v2 import assess as frozen_assess
    from knee.roi_routine_contract import SOURCE_FIELDS
    from knee.roi_supervised_qc import sha_pixels, validate_box
    record, _, half = load_source(source_dir)
    center = validate_reference(source_dir, reference)
    candidate = read_json(Path(candidate_dir) / 'candidate.json')
    if (candidate.get('version') != VERSION or candidate.get('status') != 'candidate'
            or candidate.get('sourceManifestSha256') != sha_file(Path(source_dir) / 'source.json')):
        raise WorkerError('QC_INVALID')
    if not center['center_resolved']:
        raise WorkerError('QC_INVALID')
    meta = record['sourceMeta']
    x0, x1, y0, y1 = validate_box(candidate['box'], half.shape)
    native = np.load(Path(candidate_dir) / 'crop.npy', allow_pickle=False)
    if (native.dtype != half.dtype or not np.array_equal(native, half[y0:y1, x0:x1])
            or sha_pixels(native) != candidate['candidateMeta']['crop_sha256']):
        raise WorkerError('PROCESSING_FAILED')
    if not isinstance(observation, dict):
        raise WorkerError('QC_INVALID')
    value = dict(observation)
    if 'knee_alias' in value and value['knee_alias'] != meta['knee_alias']:
        raise WorkerError('QC_INVALID')
    value['knee_alias'] = meta['knee_alias']
    try:
        result = frozen_assess({key: meta[key] for key in SOURCE_FIELDS}, center, candidate['box'], value)
    except (ValueError, TypeError, KeyError) as exc:
        raise WorkerError('QC_INVALID') from exc
    return {'accepted': result['accepted'], 'reasons': result['reasons'],
            'geometryAdequate': result['geometry_adequate'], 'observation': value,
            'cropSha256': candidate['candidateMeta']['crop_sha256'], 'frozenCandidate': CANDIDATE,
            'qcVersion': 'mcr008_confirmation_v1.0'}


def install_guard(private_root):
    private_root = Path(private_root).resolve()
    temporary = Path(tempfile.gettempdir()).resolve()
    def audit(event, args):
        if event in {'socket.connect', 'socket.getaddrinfo', 'socket.bind'}:
            raise RuntimeError('network_forbidden_in_image_worker')
        if event != 'open':
            return
        path, mode, flags = args
        if not isinstance(path, (str, bytes, os.PathLike)):
            return
        target = Path(os.fsdecode(path)).resolve()
        write = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
            isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
        if write and not target.is_relative_to(private_root) and not target.is_relative_to(temporary):
            raise RuntimeError('scientific_checkout_is_readonly')
    sys.addaudithook(audit)


def run(payload):
    if not isinstance(payload, dict) or payload.get('operation') not in {'source', 'validate_reference', 'crop', 'assess'}:
        raise WorkerError('PROCESSING_FAILED')
    operation = payload['operation']
    allowed = {
        'source': {'sourcePath', 'side', 'outDir'},
        'validate_reference': {'sourceDir', 'reference'},
        'crop': {'sourceDir', 'outDir'},
        'assess': {'sourceDir', 'candidateDir', 'reference', 'observation'},
    }[operation] | {'operation', 'checkout', 'privateRoot'}
    if set(payload) != allowed:
        raise WorkerError('PROCESSING_FAILED')
    checkout = verify_components(payload['checkout'])
    root = Path(payload['privateRoot']).resolve()
    paths = {key: safe(root, payload[key]) for key in ('sourcePath', 'sourceDir', 'candidateDir', 'outDir') if key in payload}
    if operation == 'source':
        return source(paths['sourcePath'], payload['side'], paths['outDir'])
    if operation == 'validate_reference':
        return validate_reference(paths['sourceDir'], payload['reference'])
    if operation == 'crop':
        return crop(paths['sourceDir'], paths['outDir'], checkout)
    return assess(paths['sourceDir'], paths['candidateDir'], payload['reference'], payload['observation'])


def main():
    try:
        raw = sys.stdin.buffer.read(131073)
        if len(raw) > 131072:
            raise WorkerError('PROCESSING_FAILED')
        payload = json.loads(raw)
        install_guard(payload['privateRoot'])
        with redirect_stdout(sys.stderr):
            result = run(payload)
        response = {'ok': True, 'data': result}
    except WorkerError as exc:
        response = {'ok': False, 'error': {'code': exc.code}}
    except (ImportError, FileNotFoundError) as exc:
        response = {'ok': False, 'error': {'code': 'PROCESSOR_UNAVAILABLE'}}
    except Exception:
        response = {'ok': False, 'error': {'code': 'PROCESSING_FAILED'}}
    print(json.dumps(response, allow_nan=False, ensure_ascii=False, separators=(',', ':')))
    return 0 if response['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
