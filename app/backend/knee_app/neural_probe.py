"""Private, one-job IPC checks for saved neural development models.

The API interpreter never imports the scientific package or its dependencies.
This client cannot select a clinical release or accept a patient/image payload.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from .predictor import _json


CONFIG_FIELDS = {'python', 'checkout', 'bundle', 'manifestSha256', 'scratchRoot'}
RUNTIME = {
    'python': '3.12.14', 'numpy': '1.26.4', 'torch': '2.6.0+cpu',
    'torchvision': '0.21.0+cpu', 'transformers': '4.50.2',
    'safetensors': '0.5.3', 'timm': '1.0.15',
}
SLOTS = {'radiographic', 'multimodal'}
ERROR_CODES = {
    'NEURAL_PROBE_REQUEST_INVALID', 'NEURAL_PROBE_BUNDLE_INVALID', 'ASSET_MISSING',
    'NEURAL_PROBE_ARTIFACT_MISMATCH', 'NEURAL_PROBE_SOURCE_MISMATCH',
    'NEURAL_PROBE_RUNTIME_UNAVAILABLE', 'NEURAL_PROBE_SAFETY_VIOLATION',
    'NEURAL_PROBE_CHECK_FAILED',
}
MAX_RESPONSE = 65536
SCRATCH_PREFIX = 'neural-probe-'
UNAVAILABLE = 'No se pudo completar la comprobación técnica del modelo neuronal.'


class NeuralProbeError(RuntimeError):
    def __init__(self, code='NEURAL_PROBE_CHECK_FAILED'):
        self.code = code if code in ERROR_CODES else 'NEURAL_PROBE_CHECK_FAILED'
        super().__init__(UNAVAILABLE)


def plain(value):
    if not isinstance(value, (str, os.PathLike)):
        raise NeuralProbeError('NEURAL_PROBE_BUNDLE_INVALID')
    original = Path(value)
    if not original.is_absolute():
        raise NeuralProbeError('NEURAL_PROBE_BUNDLE_INVALID')
    if any(item.is_symlink() or getattr(item, 'is_junction', lambda: False)()
           for item in (original, *original.parents)):
        raise NeuralProbeError('NEURAL_PROBE_SAFETY_VIOLATION')
    return original.resolve(strict=True)


def runtime_root(python):
    return python.parent.parent if python.parent.name.casefold() in {'scripts', 'bin'} else python.parent


class NeuralProbe:
    """Run only preflight or two worker-owned fictional fixtures, one fold at a time."""

    def __init__(self, config, *, timeout=300):
        try:
            if not isinstance(config, dict) or set(config) != CONFIG_FIELDS:
                raise NeuralProbeError('NEURAL_PROBE_BUNDLE_INVALID')
            if (not isinstance(config['manifestSha256'], str)
                    or re.fullmatch(r'[0-9a-f]{64}', config['manifestSha256']) is None):
                raise NeuralProbeError('NEURAL_PROBE_BUNDLE_INVALID')
            if type(timeout) not in {int, float} or not math.isfinite(timeout) or not 1 <= timeout <= 600:
                raise NeuralProbeError('NEURAL_PROBE_REQUEST_INVALID')
            self.python, self.checkout, self.bundle, self.scratch_root = (
                plain(config[key]) for key in ('python', 'checkout', 'bundle', 'scratchRoot'))
            if (not self.python.is_file() or not self.checkout.is_dir() or not self.bundle.is_dir()
                    or not self.scratch_root.is_dir()
                    or not plain(self.bundle / 'manifest.json').is_file()
                    or not plain(self.checkout / 'src/knee/sensitivity_real_inference.py').is_file()):
                raise NeuralProbeError('NEURAL_PROBE_BUNDLE_INVALID')
            self.runtime_root = runtime_root(self.python)
            protected = (self.bundle, self.checkout, self.runtime_root)
            if (not self.scratch_root.is_relative_to(self.bundle.parent)
                    or any(self.scratch_root == root or self.scratch_root.is_relative_to(root)
                           or root.is_relative_to(self.scratch_root) for root in protected)):
                raise NeuralProbeError('NEURAL_PROBE_SAFETY_VIOLATION')
            self.manifest_sha256 = config['manifestSha256']
            self.timeout = timeout
            self.worker = Path(__file__).with_name('neural_probe_worker.py')
        except NeuralProbeError:
            raise
        except (OSError, ValueError, TypeError, KeyError, OverflowError):
            raise NeuralProbeError('NEURAL_PROBE_BUNDLE_INVALID') from None

    def _cleanup(self, scratch, identity):
        """Remove only this call's verified, still-owned scratch directory."""
        try:
            target = plain(scratch)
            current = target.stat()
            if (target.parent != self.scratch_root or target == self.scratch_root
                    or not target.is_relative_to(self.scratch_root)
                    or not target.name.startswith(SCRATCH_PREFIX)
                    or (current.st_dev, current.st_ino) != identity):
                raise NeuralProbeError('NEURAL_PROBE_SAFETY_VIOLATION')
            # Check every directory before recursing. Never traverse a link or
            # junction that a failed worker might have left inside its scratch.
            pending = [target]
            while pending:
                directory = pending.pop()
                if plain(directory) != directory or not directory.is_relative_to(target):
                    raise NeuralProbeError('NEURAL_PROBE_SAFETY_VIOLATION')
                with os.scandir(directory) as entries:
                    for entry in entries:
                        path = Path(entry.path)
                        if entry.is_symlink() or getattr(path, 'is_junction', lambda: False)():
                            raise NeuralProbeError('NEURAL_PROBE_SAFETY_VIOLATION')
                        resolved = path.resolve(strict=True)
                        if not resolved.is_relative_to(target):
                            raise NeuralProbeError('NEURAL_PROBE_SAFETY_VIOLATION')
                        if entry.is_dir(follow_symlinks=False):
                            pending.append(resolved)
            # The final absolute target has been checked to remain within the
            # dedicated scratch root, with the identity created by this call.
            shutil.rmtree(target)
        except NeuralProbeError:
            raise
        except (OSError, ValueError, TypeError):
            raise NeuralProbeError('NEURAL_PROBE_SAFETY_VIOLATION') from None

    def _call(self, operation, slot, fold):
        if (operation not in {'preflight', 'diagnostic'} or not isinstance(slot, str) or slot not in SLOTS
                or type(fold) is not int or fold not in range(5)):
            raise NeuralProbeError('NEURAL_PROBE_REQUEST_INVALID')
        scratch = None
        identity = None
        try:
            # Recheck the parent before creating a new directory. No existing
            # scratch is adopted, and no model/case path is writable here.
            if plain(self.scratch_root) != self.scratch_root:
                raise NeuralProbeError('NEURAL_PROBE_SAFETY_VIOLATION')
            scratch = Path(tempfile.mkdtemp(prefix=SCRATCH_PREFIX, dir=self.scratch_root)).resolve()
            created = scratch.stat()
            identity = (created.st_dev, created.st_ino)
            payload = {'operation': operation, 'checkout': str(self.checkout), 'bundle': str(self.bundle),
                       'manifestSha256': self.manifest_sha256, 'slot': slot, 'fold': fold, 'scratch': str(scratch)}
            environment = {key: value for key, value in os.environ.items()
                           if not key.startswith(('KNEEOAI_', 'PYTHON', 'HF_', 'TRANSFORMERS_'))
                           and key not in {'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY',
                                           'SSL_CERT_FILE', 'REQUESTS_CA_BUNDLE'}}
            environment.update(PYTHONPATH=str(self.checkout / 'src'), PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1',
                HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', TRANSFORMERS_OFFLINE='1',
                HF_HOME=str(scratch / 'hf'), TORCH_HOME=str(scratch / 'torch'),
                OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                TEMP=str(scratch), TMP=str(scratch), TMPDIR=str(scratch))
            result = subprocess.run([str(self.python), '-B', str(self.worker)],
                input=json.dumps(payload, allow_nan=False), capture_output=True, text=True,
                encoding='utf-8', errors='strict', cwd=self.checkout, env=environment, timeout=self.timeout,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0) if os.name == 'nt' else 0)
            if len(result.stdout.encode('utf-8')) > MAX_RESPONSE:
                raise ValueError('oversized_response')
            response = _json(result.stdout)
            if not isinstance(response, dict) or type(response.get('ok')) is not bool:
                raise ValueError('invalid_envelope')
            if response['ok'] is False:
                if (not result.returncode or set(response) != {'ok', 'error'}
                        or not isinstance(response['error'], dict) or set(response['error']) != {'code'}
                        or response['error']['code'] not in ERROR_CODES):
                    raise ValueError('invalid_error')
                raise NeuralProbeError(response['error']['code'])
            if result.returncode or set(response) != {'ok', 'data'}:
                raise ValueError('worker_failed')
            data = response['data']
            if (not isinstance(data, dict) or set(data) != {'scope', 'clinicalReleaseReady', 'slot', 'fold',
                    'operation', 'weightsVerified', 'logits', 'runtime', 'elapsedSeconds'}
                    or data['scope'] != 'technical_development_check_only' or data['clinicalReleaseReady'] is not False
                    or data['weightsVerified'] is not True or data['operation'] != operation or data['slot'] != slot
                    or type(data['fold']) is not int or data['fold'] != fold or data['runtime'] != RUNTIME
                    or type(data['elapsedSeconds']) not in {int, float} or not math.isfinite(data['elapsedSeconds'])
                    or data['elapsedSeconds'] < 0 or type(data['logits']) is not list):
                raise ValueError('invalid_development_response')
            if operation == 'preflight' and data['logits'] != []:
                raise ValueError('unexpected_preflight_output')
            if operation == 'diagnostic' and (len(data['logits']) != 2
                    or not all(type(value) is float and math.isfinite(value) for value in data['logits'])):
                raise ValueError('invalid_diagnostic_output')
            return data
        except NeuralProbeError:
            raise
        except (OSError, ValueError, TypeError, KeyError, AttributeError, UnicodeError,
                RecursionError, OverflowError, subprocess.TimeoutExpired):
            raise NeuralProbeError() from None
        finally:
            if scratch is not None and identity is not None:
                self._cleanup(scratch, identity)

    def preflight(self, slot, fold):
        return self._call('preflight', slot, fold)

    def diagnostic(self, slot, fold):
        return self._call('diagnostic', slot, fold)
