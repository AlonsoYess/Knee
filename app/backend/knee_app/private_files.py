from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import uuid

from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from .errors import ApiError


class ApiBodyLimit:
    """Count received bytes before body parsers, including absent/understated lengths."""
    def __init__(self, app, max_bytes):
        self.app,self.max_bytes=app,max_bytes

    async def __call__(self, scope, receive, send):
        if scope['type']!='http':
            return await self.app(scope,receive,send)
        total=0
        async def limited_receive():
            nonlocal total
            message=await receive()
            if message['type']=='http.request':
                total+=len(message.get('body',b''))
                if total>self.max_bytes:
                    raise ApiError(413,'UPLOAD_TOO_LARGE','La solicitud supera el límite técnico permitido.')
            return message
        await self.app(scope,limited_receive,send)


def digest_file(path):
    digest, size = hashlib.sha256(), 0
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


class PrivateFiles:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.staging = self.root / '.staging'
        self.objects = self.root / 'objects'
        for directory in (self.staging, self.objects):
            if directory.is_symlink() or getattr(directory,'is_junction',lambda:False)():
                raise ValueError('Private storage must not use symbolic links')
            directory.mkdir(exist_ok=True, mode=0o700)

    def path(self, key):
        if not isinstance(key, str) or '\\' in key:
            raise ValueError('Invalid private object key')
        relative = PurePosixPath(key)
        parts=relative.parts
        if relative.is_absolute() or '..' in parts or len(parts) not in {2,3} or parts[0]!='objects' or not re.fullmatch(r'[a-f0-9]{32}',parts[1]) or (len(parts)==3 and not re.fullmatch(r'[a-z_]+\.(dcm|png|npy|json)',parts[2])):
            raise ValueError('Invalid private object key')
        current = self.root
        for part in relative.parts:
            current = current / part
            if current.is_symlink() or getattr(current,'is_junction',lambda:False)():
                raise ValueError('Symbolic links are not private objects')
        resolved = current.resolve()
        if not resolved.is_relative_to(self.root) or resolved == self.root:
            raise ValueError('Private object is outside storage')
        return resolved

    def stage(self):
        directory = self.staging / uuid.uuid4().hex
        directory.mkdir(mode=0o700)
        return directory

    def discard(self, directory):
        resolved = Path(directory).resolve()
        if not resolved.is_relative_to(self.staging.resolve()) or resolved == self.staging.resolve():
            raise ValueError('Refusing to remove a non-staging directory')
        if resolved.exists():
            shutil.rmtree(resolved)

    def publish(self, stage):
        stage = Path(stage).resolve()
        if stage.parent != self.staging.resolve():
            raise ValueError('Only private staging directories may be published')
        for child in stage.iterdir():
            if child.is_symlink() or not child.is_file():
                raise ValueError('Worker outputs must be regular files')
            if os.name != 'nt':
                child.chmod(0o600)
        target = self.objects / uuid.uuid4().hex
        os.rename(stage, target)
        return target.relative_to(self.root).as_posix()

    def inventory(self, directory_key, roles):
        directory = self.path(directory_key)
        result = []
        for path in sorted(directory.iterdir()):
            if path.is_symlink() or not path.is_file():
                raise ValueError('Worker outputs must be regular files')
            digest, size = digest_file(path)
            result.append({'role': roles.get(path.name, 'internal'), 'object_key': path.relative_to(self.root).as_posix(), 'sha256': digest, 'byte_length': size})
        return result


def png_shape(path):
    with Path(path).open('rb') as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR':
        raise ValueError('Invalid worker PNG')
    width, height = struct.unpack('>II', header[16:24])
    if not 0 < width <= 65535 or not 0 < height <= 65535:
        raise ValueError('Invalid worker PNG dimensions')
    return [height, width]


async def receive_dicom(request, files, *, max_body, max_file):
    if request.headers.get('content-type', '').split(';', 1)[0].strip().lower() != 'multipart/form-data':
        raise ApiError(415, 'UNSUPPORTED_MEDIA_TYPE', 'Carga un único archivo DICOM mediante el formulario de imagen.')
    length = request.headers.get('content-length')
    if length is not None:
        try:
            if int(length) > max_body:
                raise ApiError(413, 'UPLOAD_TOO_LARGE', 'La carga supera el límite técnico permitido.')
        except ValueError:
            raise ApiError(422, 'INVALID_UPLOAD', 'El tamaño declarado de la carga no es válido.')
    count = 0

    async def bounded_stream():
        nonlocal count
        async for chunk in request.stream():
            count += len(chunk)
            if count > max_body:
                raise ApiError(413, 'UPLOAD_TOO_LARGE', 'La carga supera el límite técnico permitido.')
            yield chunk

    parser = MultiPartParser(request.headers, bounded_stream(), max_files=1, max_fields=0)
    # Body size is bounded independently; keep the parser's spool in memory rather than OS temp.
    parser.spool_max_size = max_body + 1
    form, stage = None, None
    try:
        form = await parser.parse()
        items = list(form.multi_items())
        if len(items) != 1 or items[0][0] != 'file' or not isinstance(items[0][1], UploadFile):
            raise ApiError(422, 'INVALID_UPLOAD', 'Selecciona un único archivo DICOM.')
        upload = items[0][1]
        if upload.content_type not in {'application/dicom', 'application/octet-stream'}:
            raise ApiError(415, 'UNSUPPORTED_MEDIA_TYPE', 'El archivo debe ser un DICOM original.')
        stage = files.stage()
        descriptor = os.open(stage / 'source.dcm', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        digest, size = hashlib.sha256(), 0
        with os.fdopen(descriptor, 'wb') as destination:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > max_file:
                    raise ApiError(413, 'UPLOAD_TOO_LARGE', 'El DICOM supera el límite técnico permitido.')
                digest.update(chunk)
                destination.write(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        if not size:
            raise ApiError(422, 'INVALID_UPLOAD', 'El archivo DICOM está vacío.')
        return stage, digest.hexdigest(), size
    except MultiPartException:
        if stage is not None:
            files.discard(stage)
        raise ApiError(422, 'INVALID_UPLOAD', 'La carga debe contener un único archivo DICOM y ningún campo adicional.')
    except BaseException:
        if stage is not None:
            files.discard(stage)
        raise
    finally:
        if form is not None:
            await form.close()
        for stream in getattr(parser, '_files_to_close_on_error', []):
            stream.close()
