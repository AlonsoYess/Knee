"""Local development-model IPC checks; never a clinical prediction endpoint."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import subprocess


PENDING = 'Motor predictivo pendiente de formalización.'
CONNECTED = 'Conexión local comprobada. Falta formalizar el paquete predictivo final.'
UNAVAILABLE = 'No se pudo verificar la conexión local con los modelos guardados.'
MESSAGES = {PENDING, CONNECTED, UNAVAILABLE}


class PredictorError(RuntimeError):
    def __init__(self, code='PREDICTOR_UNAVAILABLE'):
        self.code = code
        super().__init__(UNAVAILABLE)


def _plain(value):
    original = Path(value).absolute()
    if any(p.is_symlink() or getattr(p, 'is_junction', lambda: False)() for p in (original, *original.parents)):
        raise PredictorError()
    return original.resolve()


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate_key')
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError('nonfinite_json')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)


class FrozenPredictor:
    """Verify real stored development models in a separate Python process.

    The only operations are preflight and a fixed fictional diagnostic. Neither
    operation accepts a patient, selects a final model or returns clinical risk.
    The private bundle and its expected hash are configured by the operator.
    """

    def __init__(self, settings):
        self.python = getattr(settings, 'predictor_python', None)
        self.checkout = getattr(settings, 'predictor_checkout', None)
        self.bundle = getattr(settings, 'predictor_bundle', None)
        self.manifest_sha256 = getattr(settings, 'predictor_manifest_sha256', None)
        self.timeout = getattr(settings, 'predictor_timeout_seconds', 30)
        self.worker = Path(__file__).with_name('predictor_worker.py')

    def configured(self):
        return any(value is not None for value in (self.python, self.checkout, self.bundle, self.manifest_sha256))

    def _call(self, operation):
        if operation not in {'preflight', 'diagnostic'}:
            raise PredictorError('PREDICTOR_PROTOCOL_INVALID')
        if not all((self.python, self.checkout, self.bundle, self.manifest_sha256)):
            raise PredictorError()
        if not re.fullmatch(r'[0-9a-f]{64}', self.manifest_sha256):
            raise PredictorError()
        python, checkout, bundle = map(_plain, (self.python, self.checkout, self.bundle))
        if not python.is_file() or not checkout.is_dir() or not bundle.is_dir():
            raise PredictorError()
        if not (checkout / 'src/knee/sensitivity_real_inference.py').is_file() or not _plain(bundle / 'manifest.json').is_file():
            raise PredictorError()
        payload = {'operation': operation, 'checkout': str(checkout), 'bundle': str(bundle),
                   'manifestSha256': self.manifest_sha256}
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith(('KNEEOAI_', 'PYTHON', 'HF_', 'TRANSFORMERS_'))}
        environment.update(PYTHONPATH=str(checkout / 'src'), PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1',
                           HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', TRANSFORMERS_OFFLINE='1',
                           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        try:
            result = subprocess.run([str(python), '-B', str(self.worker)],
                input=json.dumps(payload, allow_nan=False), capture_output=True, text=True,
                encoding='utf-8', errors='strict', cwd=checkout, env=environment, timeout=self.timeout,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0) if os.name == 'nt' else 0)
            if result.returncode or len(result.stdout.encode('utf-8')) > 65536:
                raise ValueError('worker_failed')
            response = _json(result.stdout)
            if not isinstance(response, dict) or set(response) != {'ok', 'data'} or response['ok'] is not True:
                raise ValueError('invalid_envelope')
            data = response['data']
            if (not isinstance(data, dict) or set(data) != {'scope', 'clinicalReleaseReady', 'modelsVerified',
                    'operation', 'runtime', 'foldDiagnostics'}
                    or data['scope'] != 'technical_development_check_only' or data['clinicalReleaseReady'] is not False
                    or type(data['modelsVerified']) is not int or data['modelsVerified'] != 5 or data['operation'] != operation
                    or data['runtime'] != {'python': '3.12.14', 'numpy': '2.5.3'}):
                raise ValueError('invalid_development_response')
            rows = data['foldDiagnostics']
            if type(rows) is not list or (operation == 'preflight' and rows != []):
                raise ValueError('invalid_diagnostics')
            if operation == 'diagnostic':
                if len(rows) != 5:
                    raise ValueError('five_diagnostic_folds_required')
                for fold, row in enumerate(rows):
                    if (not isinstance(row, dict) or set(row) != {'fold', 'logits'}
                            or type(row['fold']) is not int or row['fold'] != fold
                            or type(row['logits']) is not list or len(row['logits']) != 3
                            or not all(type(v) in {float, int} and math.isfinite(v) for v in row['logits'])):
                        raise ValueError('invalid_fold_diagnostic')
            return data
        except (OSError, ValueError, TypeError, KeyError, AttributeError, UnicodeError, RecursionError, OverflowError, subprocess.TimeoutExpired):
            raise PredictorError() from None

    def preflight(self):
        return self._call('preflight')

    def diagnostic(self):
        return self._call('diagnostic')

    def engine_status(self):
        if not self.configured():
            return {'status': 'not_configured', 'package': None, 'message': PENDING}
        try:
            self.preflight()
        except PredictorError:
            return {'status': 'unavailable', 'package': None, 'message': UNAVAILABLE}
        return {'status': 'not_configured', 'package': None, 'message': CONNECTED}
