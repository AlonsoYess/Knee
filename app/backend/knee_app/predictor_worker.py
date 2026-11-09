"""Read-only technical probe of authenticated development models.

This worker cannot predict for a doctor: it accepts no patient data and runs
only three fixed invented fixtures. It never chooses/averages folds, calibrates,
fits, applies a threshold, or opens a cohort. The audit guard prevents accidental
I/O in authenticated Python code; it is not a native-code security boundary.
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import hashlib
import importlib.abc
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import sys


MAX_REQUEST = 131072
MAX_RESPONSE = 65536
MAX_JSON = 262144
MAX_SOURCE = 4 * 1024 * 1024
PURPOSE = 'technical_development_check_only'
FORMAT = 'kneeoai-development-model-probe-v1'
PYTHON_VERSION = '3.12.14'
NUMPY_VERSION = '2.5.3'
MINIMUM_SOURCES = frozenset({
    'src/knee/sensitivity_real_inference.py', 'src/knee/clinical_scenarios.py',
    'src/knee/clinical_fold_inputs.py', 'src/knee/aligned_training/runner.py',
})
FIXTURES = (
    {'AGEYEARS': 62, 'SEX': 'F', 'BMI': 29, 'KL_BASE': 2,
     'CIRUGIA_PREVIA': None, 'WOMAC_TOTAL': None},
    {'AGEYEARS': 70, 'SEX': 'M', 'BMI': 31.4, 'KL_BASE': 3,
     'CIRUGIA_PREVIA': 0, 'WOMAC_TOTAL': 40},
    {'AGEYEARS': 55, 'SEX': 'F', 'BMI': None, 'KL_BASE': 2,
     'CIRUGIA_PREVIA': None, 'WOMAC_TOTAL': None},
)


class WorkerError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def require(condition, code='PREDICTOR_BUNDLE_INVALID'):
    if not condition:
        raise WorkerError(code)


def sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def strict_json(raw, code='PREDICTOR_BUNDLE_INVALID'):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, code)
            result[key] = value
        return result

    def constant(_):
        raise WorkerError(code)

    try:
        return json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeError, ValueError, TypeError, RecursionError) as error:
        raise WorkerError(code) from error


def plain(path, *, directory=False):
    require(isinstance(path, (str, os.PathLike)))
    original = Path(path).absolute()
    require(all(not part.is_symlink() and not getattr(part, 'is_junction', lambda: False)()
                for part in (original, *original.parents)))
    try:
        resolved = original.resolve(strict=True)
        mode = resolved.stat().st_mode
    except OSError as error:
        raise WorkerError('PREDICTOR_BUNDLE_INVALID') from error
    require(stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode))
    return resolved


def relative_file(root, value):
    require(isinstance(value, str) and 0 < len(value) <= 256
            and not any(ord(character) < 32 for character in value)
            and '\\' not in value and ':' not in value)
    parts = PurePosixPath(value).parts
    require(not value.startswith('/') and parts and all(part not in {'', '.', '..'} for part in value.split('/')))
    target = plain(root / value)
    require(target.is_relative_to(root) and target != root)
    return target


@dataclass(frozen=True)
class Pin:
    path: Path
    checksum: str
    maximum: int
    code: str = 'PREDICTOR_ARTIFACT_MISMATCH'

    def read(self):
        path = plain(self.path)
        require(0 < path.stat().st_size <= self.maximum, self.code)
        with path.open('rb') as source:
            raw = source.read(self.maximum + 1)
        require(len(raw) <= self.maximum and hashlib.sha256(raw).hexdigest() == self.checksum, self.code)
        return raw


@dataclass(frozen=True)
class Bundle:
    operation: str
    checkout: Path
    root: Path
    manifest: Pin
    sources: dict
    models: tuple

    def pins(self):
        return (self.manifest, *self.sources.values(),
                *(pin for _, weight, state in self.models for pin in (weight, state)))

    def verify(self):
        for pin in self.pins():
            pin.read()


def load_bundle(payload):
    require(isinstance(payload, dict) and set(payload) == {'operation', 'checkout', 'bundle', 'manifestSha256'},
            'PREDICTOR_REQUEST_INVALID')
    require(isinstance(payload['operation'], str) and payload['operation'] in {'preflight', 'diagnostic'}
            and sha(payload['manifestSha256']),
            'PREDICTOR_REQUEST_INVALID')
    require(all(isinstance(payload[name], str) and 0 < len(payload[name]) <= 4096
                for name in ('checkout', 'bundle')), 'PREDICTOR_REQUEST_INVALID')
    checkout, root = (plain(payload[key], directory=True) for key in ('checkout', 'bundle'))
    manifest_pin = Pin(relative_file(root, 'manifest.json'), payload['manifestSha256'], MAX_JSON)
    manifest = strict_json(manifest_pin.read())
    fields = {'format', 'purpose', 'clinical_release_ready', 'python_version', 'numpy_version',
              'selection_record_sha256', 'scientific_sources', 'models'}
    require(isinstance(manifest, dict) and set(manifest) == fields)
    require(manifest['format'] == FORMAT and manifest['purpose'] == PURPOSE
            and manifest['clinical_release_ready'] is False
            and manifest['python_version'] == PYTHON_VERSION and manifest['numpy_version'] == NUMPY_VERSION
            and sha(manifest['selection_record_sha256']))
    source_records = manifest['scientific_sources']
    require(isinstance(source_records, dict) and MINIMUM_SOURCES <= set(source_records)
            and len(source_records) <= 128)
    sources = {}
    for name, checksum in source_records.items():
        require(re.fullmatch(r'src/knee/(?:[A-Za-z_][A-Za-z0-9_]*/)*[A-Za-z_][A-Za-z0-9_]*\.py', name)
                and sha(checksum))
        sources[name] = Pin(relative_file(checkout, name), checksum, MAX_SOURCE, 'PREDICTOR_SOURCE_MISMATCH')
    require(isinstance(manifest['models'], list) and len(manifest['models']) == 5)
    models, used, folds = [], {manifest_pin.path}, set()
    for record in manifest['models']:
        require(isinstance(record, dict) and set(record) == {'fold', 'weights', 'clinical_state'})
        fold = record['fold']
        require(type(fold) is int and fold in range(5) and fold not in folds)
        folds.add(fold)
        pins = []
        for key in ('weights', 'clinical_state'):
            item = record[key]
            require(isinstance(item, dict) and set(item) == {'path', 'sha256'} and sha(item['sha256']))
            path = relative_file(root, item['path'])
            require(path.suffix == '.json' and path not in used)
            used.add(path)
            pins.append(Pin(path, item['sha256'], MAX_JSON))
        models.append((fold, *pins))
    bundle = Bundle(payload['operation'], checkout, root, manifest_pin, sources, tuple(sorted(models)))
    bundle.verify()
    return bundle


class ReadOnlyPolicy:
    """Pure audit policy; registration happens only in the worker process."""
    MUTATIONS = frozenset({'os.remove', 'os.rename', 'os.mkdir', 'os.rmdir', 'os.chmod', 'os.chown',
                          'os.link', 'os.symlink', 'os.truncate', 'os.utime', 'os.chdir'})

    def __init__(self, pins, runtime_roots):
        self.files = frozenset(pin.path for pin in pins)
        self.directories = frozenset(path.parent for path in self.files)
        self.runtime_roots = tuple(Path(root).resolve() for root in runtime_roots)

    def permitted(self, value, *, listing=False):
        if not isinstance(value, (str, bytes, os.PathLike)):
            return False
        path = Path(os.fsdecode(value)).absolute().resolve()
        if any(path.is_relative_to(root) for root in self.runtime_roots):
            return True
        return path in (self.directories if listing else self.files)

    def __call__(self, event, arguments):
        if (event.startswith(('socket.', 'urllib.', 'subprocess.', 'os.exec', 'os.spawn', 'os.posix_spawn'))
                or event in self.MUTATIONS or event in {'os.system', 'os.fork', 'os.forkpty', 'pty.spawn'}):
            raise WorkerError('PREDICTOR_SAFETY_VIOLATION')
        if event == 'open':
            path, mode, flags = arguments
            write = isinstance(mode, str) and any(character in mode for character in 'wax+')
            write = write or isinstance(flags, int) and bool(flags & (
                os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if write or not self.permitted(path):
                raise WorkerError('PREDICTOR_SAFETY_VIOLATION')
        if event in {'os.listdir', 'os.scandir'} and not self.permitted(arguments[0], listing=True):
            raise WorkerError('PREDICTOR_SAFETY_VIOLATION')


class SourceLoader(importlib.abc.Loader):
    def __init__(self, pin):
        self.pin = pin

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        # Compile authenticated source bytes directly; never trust a stale .pyc.
        code = compile(self.pin.read(), str(self.pin.path), 'exec', dont_inherit=True)
        exec(code, module.__dict__)


class ScientificFinder(importlib.abc.MetaPathFinder):
    def __init__(self, sources):
        self.sources = sources

    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'knee' and not fullname.startswith('knee.'):
            return None
        stem = 'src/' + fullname.replace('.', '/')
        for name, package in ((stem + '.py', False), (stem + '/__init__.py', True)):
            pin = self.sources.get(name)
            if pin is not None:
                return importlib.util.spec_from_file_location(fullname, pin.path,
                    loader=SourceLoader(pin), submodule_search_locations=[str(pin.path.parent)] if package else None)
        raise WorkerError('PREDICTOR_SOURCE_MISMATCH')


class QuietOutput:
    def write(self, value):
        return len(value)

    def flush(self):
        pass


def scientific_runtime(bundle):
    require(platform.python_version() == PYTHON_VERSION, 'PREDICTOR_RUNTIME_UNAVAILABLE')
    require(not any(name == 'knee' or name.startswith('knee.') for name in sys.modules),
            'PREDICTOR_RUNTIME_UNAVAILABLE')
    sys.dont_write_bytecode = True
    sys.addaudithook(ReadOnlyPolicy(bundle.pins(), {sys.prefix, sys.base_prefix}))
    sys.meta_path.insert(0, ScientificFinder(bundle.sources))
    try:
        import numpy as np
        require(np.__version__ == NUMPY_VERSION, 'PREDICTOR_RUNTIME_UNAVAILABLE')
        from knee.clinical_scenarios import validate_scenario_state
        from knee.sensitivity_real_inference import _logistic_logits, _transform_baseline
    except WorkerError:
        raise
    except (ImportError, OSError, AttributeError) as error:
        raise WorkerError('PREDICTOR_RUNTIME_UNAVAILABLE') from error
    return np, validate_scenario_state, _logistic_logits, _transform_baseline


def run(payload):
    bundle = load_bundle(payload)
    with redirect_stdout(QuietOutput()), redirect_stderr(QuietOutput()):
        np, validate_state, logits, transform = scientific_runtime(bundle)
        diagnostics = []
        try:
            for fold, weights, state_pin in bundle.models:
                state = strict_json(state_pin.read())
                validate_state(state)
                require(state['scenario'] == 'clinical_main' and state['validation_fold'] == fold)
                # Loading is exercised even in preflight, with no patient/fixture row.
                require(logits(weights.path, weights.checksum, np.empty((0, 3), dtype=np.float32)) == [])
                if bundle.operation == 'diagnostic':
                    values = logits(weights.path, weights.checksum, transform([dict(row) for row in FIXTURES], state))
                    require(isinstance(values, list) and len(values) == 3
                            and all(type(value) in {float, int} and math.isfinite(value) for value in values))
                    diagnostics.append({'fold': fold, 'logits': [float(value) for value in values]})
        finally:
            bundle.verify()
    return {'scope': PURPOSE, 'clinicalReleaseReady': False, 'modelsVerified': 5,
            'operation': bundle.operation, 'runtime': {'python': PYTHON_VERSION, 'numpy': NUMPY_VERSION},
            'foldDiagnostics': diagnostics}


def main():
    try:
        raw = sys.stdin.buffer.read(MAX_REQUEST + 1)
        require(len(raw) <= MAX_REQUEST, 'PREDICTOR_REQUEST_INVALID')
        payload = strict_json(raw, 'PREDICTOR_REQUEST_INVALID')
        response = {'ok': True, 'data': run(payload)}
    except WorkerError as error:
        response = {'ok': False, 'error': {'code': error.code}}
    except Exception:
        response = {'ok': False, 'error': {'code': 'PREDICTOR_CHECK_FAILED'}}
    encoded = json.dumps(response, ensure_ascii=True, allow_nan=False, separators=(',', ':')).encode('utf-8')
    if len(encoded) > MAX_RESPONSE:
        encoded = b'{"ok":false,"error":{"code":"PREDICTOR_CHECK_FAILED"}}'
    sys.stdout.buffer.write(encoded + b'\n')
    return 0 if response['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
