"""One authenticated neural development job, using only fixed invented inputs.

No patient, cohort, outcome, training operation, model selection, calibration or
classification is accepted. Original construction/normalization/inference are
reused. Python audit checks prevent accidental I/O; they do not sandbox native
code. Only a new caller-owned scratch directory may receive temporary files.
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import sys
import tempfile
import time

if __package__:
    from . import predictor_worker as common
else:
    import predictor_worker as common


FORMAT = 'kneeoai-neural-development-probe-v1'
PURPOSE = 'technical_development_check_only'
MAX_REQUEST = 131072
MAX_RESPONSE = 65536
MAX_METADATA = 2 * 1024 * 1024
MAX_SOURCE = 4 * 1024 * 1024
MAX_CHECKPOINT = 2_000_000_000
RUNTIME = {'python': '3.12.14', 'numpy': '1.26.4', 'torch': '2.6.0+cpu',
           'torchvision': '0.21.0+cpu', 'transformers': '4.50.2',
           'safetensors': '0.5.3', 'timm': '1.0.15'}
SLOTS = {'radiographic': ('image_skelex', 'stage_2'),
         'multimodal': ('fusion_intermediate_gated_film', 'fusion')}
EXTERNAL_NAMES = ('skelex_model.safetensors', 'skelex_config.json',
                  'skelex_preprocessor_config.json')
MINIMUM_SOURCES = frozenset({'src/knee/sensitivity_real_inference.py',
    'src/knee/aligned_training/models.py', 'src/knee/aligned_training/engine.py',
    'src/knee/methodology_release.py', 'src/knee/clinical_scenarios.py',
    'src/knee/clinical_fold_inputs.py'})
MINIMUM_RECIPES = frozenset({'configs/training_plan_v1_2.json',
    'configs/pretrained_weights_v1_1.json', 'configs/model_registry_v1_1.json'})
FIXTURES = (
    {'AGEYEARS': 62, 'SEX': 'F', 'BMI': 29, 'KL_BASE': 2,
     'CIRUGIA_PREVIA': None, 'WOMAC_TOTAL': None},
    {'AGEYEARS': 70, 'SEX': 'M', 'BMI': 31.4, 'KL_BASE': 3,
     'CIRUGIA_PREVIA': 0, 'WOMAC_TOTAL': 40},
)


class WorkerError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def require(condition, code='NEURAL_PROBE_BUNDLE_INVALID'):
    if not condition:
        raise WorkerError(code)


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def strict_json(raw, code='NEURAL_PROBE_BUNDLE_INVALID'):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, code)
            result[key] = value
        return result
    def nonfinite(_):
        raise WorkerError(code)
    try:
        return json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=nonfinite)
    except (UnicodeError, ValueError, TypeError, RecursionError) as error:
        raise WorkerError(code) from error


def plain(value, *, directory=False, allow_missing=False):
    require(isinstance(value, (str, os.PathLike)))
    original = Path(value)
    require(original.is_absolute())
    require(all(not part.is_symlink() and not getattr(part, 'is_junction', lambda: False)()
                for part in (original, *original.parents)))
    try:
        target = original.resolve(strict=not allow_missing)
        if allow_missing and not target.exists():
            return target
        mode = target.stat().st_mode
    except FileNotFoundError as error:
        raise WorkerError('ASSET_MISSING') from error
    except OSError as error:
        raise WorkerError('NEURAL_PROBE_BUNDLE_INVALID') from error
    require(stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode))
    return target


def relative_path(root, value, *, allow_missing=False):
    require(isinstance(value, str) and 0 < len(value) <= 512
            and not any(ord(character) < 32 for character in value)
            and '\\' not in value and ':' not in value)
    pieces = value.split('/')
    require(not value.startswith('/') and all(piece not in {'', '.', '..'} for piece in pieces))
    require(PurePosixPath(value).as_posix() == value)
    target = plain(root / value, allow_missing=allow_missing)
    require(target != root and target.is_relative_to(root))
    return target


@dataclass(frozen=True)
class Pin:
    path: Path
    checksum: str
    maximum: int
    expected_bytes: int | None = None
    code: str = 'NEURAL_PROBE_ARTIFACT_MISMATCH'

    def verify(self):
        target = plain(self.path)
        before = target.stat()
        require(0 < before.st_size <= self.maximum, self.code)
        require(self.expected_bytes is None or before.st_size == self.expected_bytes, self.code)
        digest, count = hashlib.sha256(), 0
        with target.open('rb') as stream:
            while chunk := stream.read(4 * 1024 * 1024):
                count += len(chunk)
                require(count <= self.maximum, self.code)
                digest.update(chunk)
        after = target.stat()
        require(count == before.st_size and digest.hexdigest() == self.checksum
                and (before.st_size, before.st_mtime_ns, before.st_ino)
                    == (after.st_size, after.st_mtime_ns, after.st_ino), self.code)

    def read(self):
        # SourceLoader needs bytes; checkpoint verification is streaming only.
        require(self.maximum <= MAX_SOURCE, self.code)
        target = plain(self.path)
        require(0 < target.stat().st_size <= self.maximum, self.code)
        with target.open('rb') as stream:
            raw = stream.read(self.maximum + 1)
        require(len(raw) <= self.maximum and hashlib.sha256(raw).hexdigest() == self.checksum, self.code)
        return raw


@dataclass(frozen=True)
class Job:
    slot: str
    fold: int
    request: dict
    source_record_sha256: str
    weights: Pin
    clinical_state: Pin


@dataclass(frozen=True)
class Bundle:
    operation: str
    checkout: Path
    root: Path
    scratch: Path
    manifest: Pin
    sources: dict
    recipes: dict
    external: tuple
    job: Job
    implementation_pins: tuple = ()

    def pins(self):
        return (self.manifest, *self.sources.values(), *self.recipes.values(),
                *self.external, self.job.weights, self.job.clinical_state,
                *self.implementation_pins)

    def verify(self):
        for pin in self.pins():
            pin.verify()


def trusted_implementation_pins():
    """Allow introspection of these two trusted worker files, never a caller path."""
    result = []
    for name in (__file__, common.__file__):
        path = plain(name)
        require(0 < path.stat().st_size <= MAX_SOURCE, 'NEURAL_PROBE_SOURCE_MISMATCH')
        with path.open('rb') as stream:
            raw = stream.read(MAX_SOURCE + 1)
        require(len(raw) <= MAX_SOURCE, 'NEURAL_PROBE_SOURCE_MISMATCH')
        result.append(Pin(path, hashlib.sha256(raw).hexdigest(), MAX_SOURCE, len(raw),
                          'NEURAL_PROBE_SOURCE_MISMATCH'))
    return tuple(result)


def validate_job_request(request, slot, fold):
    require(isinstance(request, dict))
    model, stage = SLOTS[slot]
    require(request.get('model_id') == model and request.get('stage') == stage
            and request.get('image_encoder_id') == 'image_skelex'
            and request.get('clinical_scenario') == 'clinical_main'
            and type(request.get('validation_fold')) is int and request['validation_fold'] == fold
            and type(request.get('seed')) is int and request['seed'] == 2026
            and request.get('operation') == 'development_fold'
            and type(request.get('image_micro_batch_size')) is int
            and request['image_micro_batch_size'] in {1, 2, 4})


def load_bundle(payload):
    fields = {'operation', 'checkout', 'bundle', 'manifestSha256', 'slot', 'fold', 'scratch'}
    require(isinstance(payload, dict) and set(payload) == fields, 'NEURAL_PROBE_REQUEST_INVALID')
    require(isinstance(payload['operation'], str) and payload['operation'] in {'preflight', 'diagnostic'}
            and isinstance(payload['slot'], str) and payload['slot'] in SLOTS
            and type(payload['fold']) is int and payload['fold'] in range(5)
            and valid_sha(payload['manifestSha256']), 'NEURAL_PROBE_REQUEST_INVALID')
    require(all(isinstance(payload[key], str) and 0 < len(payload[key]) <= 4096
                for key in ('checkout', 'bundle', 'scratch')), 'NEURAL_PROBE_REQUEST_INVALID')
    checkout, root, scratch = [plain(payload[key], directory=True) for key in ('checkout', 'bundle', 'scratch')]
    require(not root.is_relative_to(checkout) and not checkout.is_relative_to(root))
    require(scratch != root.parent and scratch.is_relative_to(root.parent)
            and not scratch.is_relative_to(root) and not root.is_relative_to(scratch)
            and not scratch.is_relative_to(checkout) and not checkout.is_relative_to(scratch)
            and not any(scratch.iterdir()), 'NEURAL_PROBE_REQUEST_INVALID')
    require(not any(scratch.is_relative_to(Path(base).resolve()) for base in (sys.prefix, sys.base_prefix)),
            'NEURAL_PROBE_REQUEST_INVALID')
    manifest_pin = Pin(relative_path(root, 'manifest.json'), payload['manifestSha256'], MAX_METADATA)
    manifest = strict_json(manifest_pin.read())
    require(isinstance(manifest, dict) and set(manifest) == {'format', 'purpose', 'clinical_release_ready',
        'runtime', 'scientific_sources', 'recipe_files', 'external', 'jobs'})
    require(manifest['format'] == FORMAT and manifest['purpose'] == PURPOSE
            and manifest['clinical_release_ready'] is False and manifest['runtime'] == RUNTIME)
    source_records, recipe_records = manifest['scientific_sources'], manifest['recipe_files']
    require(isinstance(source_records, dict) and MINIMUM_SOURCES <= set(source_records)
            and len(source_records) <= 256)
    require(isinstance(recipe_records, dict) and MINIMUM_RECIPES <= set(recipe_records)
            and len(recipe_records) <= 256)
    sources, recipes = {}, {}
    for name, checksum in source_records.items():
        require(isinstance(name, str) and re.fullmatch(
            r'src/knee/(?:[A-Za-z_][A-Za-z0-9_]*/)*[A-Za-z_][A-Za-z0-9_]*\.py', name)
            and valid_sha(checksum))
        sources[name] = Pin(relative_path(checkout, name), checksum, MAX_SOURCE,
                            code='NEURAL_PROBE_SOURCE_MISMATCH')
    for name, checksum in recipe_records.items():
        require(isinstance(name, str) and (re.fullmatch(r'configs/(?:[A-Za-z0-9_-]+/)*[A-Za-z0-9_.-]+\.json', name)
            or re.fullmatch(r'requirements-[A-Za-z0-9_-]+\.txt', name)
            or re.fullmatch(r'src/knee/licenses/[A-Za-z0-9_.-]+\.txt', name)
            or name == 'scripts/prepare_model_inputs.py') and valid_sha(checksum))
        recipes[name] = Pin(relative_path(checkout, name), checksum, MAX_SOURCE,
                            code='NEURAL_PROBE_SOURCE_MISMATCH')
    require(type(manifest['external']) is list and len(manifest['external']) == 3)
    external, seen_external = {}, set()
    for item in manifest['external']:
        require(isinstance(item, dict) and set(item) == {'path', 'name', 'bytes', 'sha256'}
                and item['name'] in EXTERNAL_NAMES and item['name'] not in seen_external
                and type(item['bytes']) is int and 0 < item['bytes'] <= MAX_CHECKPOINT
                and valid_sha(item['sha256']))
        target = relative_path(root, item['path'], allow_missing=True)
        require(target.name == item['name'])
        seen_external.add(item['name'])
        external[item['name']] = Pin(target, item['sha256'], MAX_CHECKPOINT, item['bytes'])
    require(type(manifest['jobs']) is list and len(manifest['jobs']) == 10)
    jobs, weight_paths, state_pins = {}, set(), {}
    for record in manifest['jobs']:
        require(isinstance(record, dict) and set(record) == {'slot', 'fold', 'request',
            'source_record_sha256', 'weights', 'clinical_state'})
        slot, fold = record['slot'], record['fold']
        require(isinstance(slot, str) and slot in SLOTS and type(fold) is int and fold in range(5)
                and (slot, fold) not in jobs and valid_sha(record['source_record_sha256']))
        validate_job_request(record['request'], slot, fold)
        weights, state = record['weights'], record['clinical_state']
        require(isinstance(weights, dict) and set(weights) == {'path', 'sha256', 'bytes'}
                and valid_sha(weights['sha256']) and type(weights['bytes']) is int
                and 0 < weights['bytes'] <= MAX_CHECKPOINT)
        weight_path = relative_path(root, weights['path'], allow_missing=True)
        require(weight_path.suffix == '.safetensors' and weight_path not in weight_paths)
        weight_paths.add(weight_path)
        require(isinstance(state, dict) and set(state) == {'path', 'sha256'} and valid_sha(state['sha256']))
        state_path = relative_path(root, state['path'], allow_missing=True)
        require(state_path.suffix == '.json' and (state_path not in state_pins or state_pins[state_path] == state['sha256']))
        state_pins[state_path] = state['sha256']
        jobs[slot, fold] = Job(slot, fold, record['request'], record['source_record_sha256'],
            Pin(weight_path, weights['sha256'], MAX_CHECKPOINT, weights['bytes']),
            Pin(state_path, state['sha256'], MAX_METADATA))
    require(set(jobs) == {(slot, fold) for slot in SLOTS for fold in range(5)})
    require(not (weight_paths & set(state_pins)))
    bundle = Bundle(payload['operation'], checkout, root, scratch, manifest_pin, sources, recipes,
        tuple(external[name] for name in EXTERNAL_NAMES), jobs[payload['slot'], payload['fold']],
        trusted_implementation_pins())
    # Only the chosen job is opened. Other jobs may still await download.
    bundle.verify()
    return bundle


class ScratchPolicy:
    """Allow exact immutable inputs and mutations only in new private scratch."""
    PROHIBITED = {'os.system', 'os.fork', 'os.forkpty', 'pty.spawn', 'os.chdir',
                  'os.link', 'os.symlink', 'os.chown'}

    def __init__(self, pins, runtime_roots, scratch):
        self.files = frozenset(pin.path for pin in pins)
        self.directories = frozenset(path.parent for path in self.files)
        self.runtime_roots = tuple(Path(root).resolve() for root in runtime_roots)
        self.scratch = Path(scratch).resolve()

    def path(self, value, dir_fd=None):
        if type(value) is int:
            if os.name != 'posix':
                return None
            try:
                return Path(os.readlink(f'/proc/self/fd/{value}')).resolve()
            except OSError:
                return None
        if not isinstance(value, (str, bytes, os.PathLike)):
            return None
        target = Path(os.fsdecode(value))
        if dir_fd not in {None, -1}:
            base = self.path(dir_fd)
            if base is None:
                return None
            target = base / target
        return target.absolute().resolve()

    def writable(self, value, dir_fd=None):
        target = self.path(value, dir_fd)
        return target is not None and target != self.scratch and target.is_relative_to(self.scratch)

    def readable(self, value, *, listing=False):
        target = self.path(value)
        if target is None:
            return False
        return (target.is_relative_to(self.scratch)
            or any(target.is_relative_to(root) for root in self.runtime_roots)
            or target in (self.directories if listing else self.files))

    def __call__(self, event, arguments):
        if (event.startswith(('socket.', 'urllib.', 'subprocess.', 'os.exec', 'os.spawn', 'os.posix_spawn'))
                and event != 'socket.gethostname') or event in self.PROHIBITED:
            raise WorkerError('NEURAL_PROBE_SAFETY_VIOLATION')
        allowed = True
        if event == 'open':
            path, mode, flags = arguments
            write = isinstance(mode, str) and any(character in mode for character in 'wax+')
            write = write or isinstance(flags, int) and bool(flags & (
                os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            allowed = self.writable(path) if write else self.readable(path)
        elif event in {'os.listdir', 'os.scandir'}:
            allowed = self.readable(arguments[0], listing=True)
        elif event in {'os.remove', 'os.rmdir'}:
            allowed = self.writable(arguments[0], arguments[1] if len(arguments) > 1 else None)
        elif event == 'os.mkdir':
            allowed = self.writable(arguments[0], arguments[2] if len(arguments) > 2 else None)
        elif event == 'os.rename':
            allowed = self.writable(arguments[0], arguments[2]) and self.writable(arguments[1], arguments[3])
        elif event in {'os.chmod', 'os.utime', 'os.truncate'}:
            allowed = self.writable(arguments[0])
        elif event == 'shutil.rmtree':
            allowed = self.writable(arguments[0], arguments[1] if len(arguments) > 1 else None)
        if not allowed:
            raise WorkerError('NEURAL_PROBE_SAFETY_VIOLATION')


class ScientificFinder(common.ScientificFinder):
    def find_spec(self, fullname, path=None, target=None):
        try:
            return super().find_spec(fullname, path, target)
        except common.WorkerError as error:
            raise WorkerError('NEURAL_PROBE_SOURCE_MISMATCH') from error


def scientific_runtime(bundle):
    require(platform.python_version() == RUNTIME['python'], 'NEURAL_PROBE_RUNTIME_UNAVAILABLE')
    require(not any(name == 'knee' or name.startswith('knee.') for name in sys.modules),
            'NEURAL_PROBE_RUNTIME_UNAVAILABLE')
    tempfile.tempdir = str(bundle.scratch)
    os.environ.update(TMP=str(bundle.scratch), TEMP=str(bundle.scratch), TMPDIR=str(bundle.scratch),
                      CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_HUB_DISABLE_TELEMETRY='1', PYTHONDONTWRITEBYTECODE='1')
    sys.dont_write_bytecode = True
    runtime_roots = {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}
    # Source-only finder resolves knee.*; remove checkout/cwd search entries.
    sys.path[:] = [item for item in sys.path if item and any(
        Path(item).resolve().is_relative_to(root) for root in runtime_roots)]
    sys.meta_path.insert(0, ScientificFinder(bundle.sources))
    sys.addaudithook(ScratchPolicy(bundle.pins(), runtime_roots, bundle.scratch))
    try:
        actual = {name: importlib.metadata.version(name) for name in RUNTIME if name != 'python'}
        require(actual == {name: value for name, value in RUNTIME.items() if name != 'python'},
                'NEURAL_PROBE_RUNTIME_UNAVAILABLE')
        import numpy as np
        import torch
        require(np.__version__ == RUNTIME['numpy'] and str(torch.__version__) == RUNTIME['torch']
                and torch.version.cuda is None and not torch.cuda.is_available(),
                'NEURAL_PROBE_RUNTIME_UNAVAILABLE')
        torch.set_num_threads(2)
        from knee.methodology_release import load_release
        from knee.aligned_training.authority import validate_request
        from knee.clinical_scenarios import validate_scenario_state
        from knee.sensitivity_real_inference import _neural_logits, _transform_baseline
    except WorkerError:
        raise
    except (ImportError, OSError, AttributeError, importlib.metadata.PackageNotFoundError) as error:
        raise WorkerError('NEURAL_PROBE_RUNTIME_UNAVAILABLE') from error
    return np, load_release, validate_request, validate_scenario_state, _neural_logits, _transform_baseline


def fixed_images(np):
    first = np.linspace(0, 1, 224 * 224, dtype=np.float32).reshape(1, 224, 224)
    second = np.ascontiguousarray(1 - first)
    for image in (first, second):
        image.setflags(write=False)
    return [first, second]


def run(payload):
    started = time.monotonic()
    bundle = load_bundle(payload)
    with redirect_stdout(common.QuietOutput()), redirect_stderr(common.QuietOutput()):
        np, load_release, validate_request, validate_state, neural_logits, transform = scientific_runtime(bundle)
        try:
            _, catalog, plan, registry = load_release(bundle.checkout)
            validate_request(bundle.job.request, plan, registry)
            specs = catalog['models']['image_skelex']['files']
            require(len(specs) == len(bundle.external) and all(
                (pin.path.name, pin.checksum, pin.expected_bytes) == (spec['name'], spec['sha256'], spec['bytes'])
                for pin, spec in zip(bundle.external, specs)), 'NEURAL_PROBE_ARTIFACT_MISMATCH')
            state = strict_json(bundle.job.clinical_state.read())
            validate_state(state)
            require(state['scenario'] == 'clinical_main' and state['validation_fold'] == bundle.job.fold)
            if bundle.operation == 'preflight':
                clinical, images = np.empty((0, 3), dtype=np.float32), []
            else:
                clinical = transform([dict(row) for row in FIXTURES], state)
                images = fixed_images(np)
            job = {'receipt': {'request': bundle.job.request}, 'weight': bundle.job.weights.path,
                   'weight_sha256': bundle.job.weights.checksum}
            logits = neural_logits(job, clinical, images, plan, catalog,
                [pin.path for pin in bundle.external], device='cpu')
            expected = 0 if bundle.operation == 'preflight' else 2
            require(type(logits) is list and len(logits) == expected and all(
                type(value) in {int, float} and math.isfinite(value) for value in logits),
                'NEURAL_PROBE_CHECK_FAILED')
        finally:
            bundle.verify()
    return {'scope': PURPOSE, 'clinicalReleaseReady': False, 'slot': bundle.job.slot,
            'fold': bundle.job.fold, 'operation': bundle.operation, 'weightsVerified': True,
            'logits': [float(value) for value in logits], 'runtime': dict(RUNTIME),
            'elapsedSeconds': round(time.monotonic() - started, 6)}


def main():
    try:
        raw = sys.stdin.buffer.read(MAX_REQUEST + 1)
        require(len(raw) <= MAX_REQUEST, 'NEURAL_PROBE_REQUEST_INVALID')
        response = {'ok': True, 'data': run(strict_json(raw, 'NEURAL_PROBE_REQUEST_INVALID'))}
    except WorkerError as error:
        response = {'ok': False, 'error': {'code': error.code}}
    except Exception:
        response = {'ok': False, 'error': {'code': 'NEURAL_PROBE_CHECK_FAILED'}}
    encoded = json.dumps(response, ensure_ascii=True, allow_nan=False, separators=(',', ':')).encode('utf-8')
    if len(encoded) > MAX_RESPONSE:
        response = {'ok': False, 'error': {'code': 'NEURAL_PROBE_CHECK_FAILED'}}
        encoded = json.dumps(response, separators=(',', ':')).encode('utf-8')
    sys.stdout.buffer.write(encoded + b'\n')
    return 0 if response['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
