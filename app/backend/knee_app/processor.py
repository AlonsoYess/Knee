"""Application IPC boundary; scientific libraries live only in a separate process."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess


class ProcessorError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code, self.message = code, message
        super().__init__(message)


class FrozenProcessor:
    """Call the explicitly configured, existing frozen scientific environment.

    Availability means configuration exists, not scientific acceptance of an
    image. Every worker call independently verifies the frozen components.
    Neither the backend interpreter nor its dependencies import ``knee``.
    """

    pipeline_version = 'kneel_guided_gradient_contours_contact_anchor_v1'
    qc_policy_version = 'mcr008_confirmation_v1.0'

    def __init__(self, settings):
        python = getattr(settings, 'frozen_python', None)
        checkout = getattr(settings, 'frozen_checkout', None)
        self.python = Path(python).resolve() if python else None
        self.checkout = Path(checkout).resolve() if checkout else None
        storage = getattr(settings, 'private_files_path', None)
        self.storage = Path(storage or Path(settings.database_path).parent / 'files').resolve()
        self.timeout = float(getattr(settings, 'frozen_timeout_seconds', 180))
        self.worker = Path(__file__).with_name('frozen_worker.py').resolve()

    def is_available(self) -> bool:
        return bool(self.python and self.python.is_file() and self.checkout and
                    (self.checkout / 'src/knee/roi_contact_confirmation.py').is_file() and
                    (self.checkout / 'configs/roi_mcr009_contact_confirmation.json').is_file() and
                    (self.checkout / 'outputs/mcr009_contact_confirmation/execution_lock.json').is_file())

    @property
    def available(self) -> bool:
        return self.is_available()

    def _path(self, path) -> str:
        target = Path(path).resolve()
        if target == self.storage or not target.is_relative_to(self.storage):
            raise ProcessorError('PROCESSING_FAILED', 'La ubicación del archivo no está autorizada.')
        # Reject existing links, including Windows junctions, before handing
        # a server-owned path to the separate scientific process.
        for candidate in (Path(path).absolute(), *Path(path).absolute().parents):
            if candidate == self.storage:
                break
            if candidate.is_symlink() or getattr(candidate, 'is_junction', lambda: False)():
                raise ProcessorError('PROCESSING_FAILED', 'La ubicación del archivo no está autorizada.')
        return str(target)

    def _call(self, operation: str, **arguments):
        if not self.is_available():
            raise ProcessorError('PROCESSOR_UNAVAILABLE', 'La preparación de imágenes no está configurada.')
        payload = {'operation': operation, 'checkout': str(self.checkout),
                   'privateRoot': str(self.storage), **arguments}
        encoded = json.dumps(payload, allow_nan=False, ensure_ascii=False)
        if len(encoded.encode('utf-8')) > 131072:
            raise ProcessorError('PROCESSING_FAILED', 'La solicitud de preparación supera el límite permitido.')
        environment = os.environ.copy()
        environment.pop('PYTHONHOME', None)
        environment.update(PYTHONPATH=str(self.checkout / 'src'), PYTHONDONTWRITEBYTECODE='1',
                           PYTHONUTF8='1', HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                           TRANSFORMERS_OFFLINE='1')
        try:
            result = subprocess.run([str(self.python), '-B', str(self.worker)], input=encoded,
                capture_output=True, text=True, encoding='utf-8', errors='strict',
                cwd=self.checkout, env=environment, timeout=self.timeout,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0) if os.name == 'nt' else 0)
        except subprocess.TimeoutExpired as exc:
            raise ProcessorError('PROCESSOR_UNAVAILABLE', 'La preparación excedió el tiempo disponible. El caso no fue aprobado.') from exc
        except (OSError, UnicodeError) as exc:
            raise ProcessorError('PROCESSOR_UNAVAILABLE', 'No se pudo iniciar la preparación de imágenes.') from exc
        try:
            if len(result.stdout.encode('utf-8')) > 1048576:
                raise ValueError('oversized_worker_response')
            response = json.loads(result.stdout)
            if not isinstance(response, dict) or type(response.get('ok')) is not bool:
                raise ValueError('invalid_worker_response')
        except (ValueError, UnicodeError) as exc:
            raise ProcessorError('PROCESSOR_UNAVAILABLE', 'No se pudo verificar la respuesta del preparador de imágenes.') from exc
        if result.returncode or not response['ok']:
            error = response.get('error', {})
            code = error.get('code')
            messages = {
                'PROCESSOR_UNAVAILABLE': 'El preparador de imágenes no está disponible o sus componentes no coinciden con la versión congelada.',
                'IMAGE_REJECTED': 'La imagen no es compatible con la preparación disponible. Revisa el archivo y sus características.',
                'SOURCE_REFERENCE_INVALID': 'Revisa la referencia indicada sobre la imagen original.',
                'QC_INVALID': 'Revisa los controles de calidad y las observaciones indicadas.',
                'PROCESSING_FAILED': 'No se pudo completar la preparación. El caso no fue aprobado.',
            }
            if code not in messages:
                code = 'PROCESSING_FAILED'
            raise ProcessorError(code, messages[code])
        data = response.get('data')
        if not isinstance(data, dict):
            raise ProcessorError('PROCESSING_FAILED', 'La respuesta de preparación no es válida.')
        return data

    def source(self, sourcePath, side, outDir):
        return self._call('source', sourcePath=self._path(sourcePath), side=side, outDir=self._path(outDir))

    def validate_reference(self, sourceDir, reference):
        return self._call('validate_reference', sourceDir=self._path(sourceDir), reference=reference)

    def crop(self, sourceDir, outDir):
        # A reference center is deliberately absent from this interface.
        return self._call('crop', sourceDir=self._path(sourceDir), outDir=self._path(outDir))

    def assess(self, sourceDir, candidateDir, reference, observation):
        return self._call('assess', sourceDir=self._path(sourceDir), candidateDir=self._path(candidateDir),
                          reference=reference, observation=observation)
