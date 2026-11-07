"""Create an application-only Docker context without visiting private directories.

Usage: python app/deploy/build_context.py --output /explicit/path/context.tar
The report lists public source paths, sizes and digests, never private inventories.
The output is created exclusively; existing archives are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile


STATIC_FILES = (
    'Dockerfile', '.dockerignore', 'backend/requirements.lock',
    'frontend/package.json', 'frontend/package-lock.json', 'frontend/index.html',
    'frontend/tsconfig.json', 'frontend/vite.config.ts',
    'frontend/public/manifest.webmanifest',
    'frontend/public/icons/kneeoai.svg', 'frontend/public/icons/kneeoai-maskable.svg',
    'frontend/public/icons/kneeoai-192.png', 'frontend/public/icons/kneeoai-512.png',
    'frontend/scripts/pwa-build.mjs', 'frontend/scripts/pwa-build.d.mts',
    'frontend/scripts/pwa-worker.js',
)
BLOCKED_DIRS = frozenset({
    'runtime', 'node_modules', '__pycache__', '.git', 'secrets', 'private',
    'data', 'datasets', 'models', 'weights', 'notebooks',
})
SOURCE_EXTENSIONS = frozenset({'.ts', '.vue', '.css'})
MAX_PUBLIC_FILE_BYTES = 8 * 1024 * 1024


def _blocked_directory(name: str) -> bool:
    return name.casefold() in BLOCKED_DIRS or name.casefold().startswith('.venv')


def _is_link(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction())


def _checked_file(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative).parts
    if not parts or relative.startswith('/') or any(part in {'.', '..'} for part in parts):
        raise ValueError('Invalid public source path')
    current = root
    for part in parts:
        current /= part
        if _is_link(current):
            raise ValueError(f'Links/junctions are not allowed in the build context: {relative}')
    if not current.is_file() or not current.resolve().is_relative_to(root):
        raise ValueError(f'Expected a regular public source file: {relative}')
    mode = current.stat().st_mode
    if not stat.S_ISREG(mode) or current.stat().st_size > MAX_PUBLIC_FILE_BYTES:
        raise ValueError(f'Invalid or unexpectedly large public build file: {relative}')
    return current


def collect_files(app: Path) -> list[tuple[str, Path]]:
    if _is_link(app):
        raise ValueError('The application root cannot be a link or junction')
    root = app.resolve(strict=True)
    if app.absolute() != root:
        raise ValueError('Application root must not resolve through links or junctions')
    if not root.is_dir():
        raise ValueError('Application root must be a directory')
    selected = {relative: _checked_file(root, relative) for relative in STATIC_FILES}
    backend = root / 'backend' / 'knee_app'
    if _is_link(backend) or not backend.is_dir():
        raise ValueError('Missing regular backend source directory')
    for path in backend.iterdir():
        if path.name.endswith('.py') and not path.name.startswith('.env'):
            relative = path.relative_to(root).as_posix()
            selected[relative] = _checked_file(root, relative)
    for required in ('__init__.py', 'api.py', 'deploy.py'):
        if f'backend/knee_app/{required}' not in selected:
            raise ValueError(f'Missing application module: {required}')
    source = root / 'frontend' / 'src'
    if _is_link(source) or not source.is_dir():
        raise ValueError('Missing regular frontend source directory')
    # Only traverse this source tree; runtime, data and dependency trees are not opened.
    for directory, dirs, files in os.walk(source, followlinks=False):
        parent = Path(directory)
        retained = []
        for name in dirs:
            if _blocked_directory(name):
                continue
            if _is_link(parent / name):
                raise ValueError('Links/junctions are not allowed under frontend/src')
            retained.append(name)
        dirs[:] = retained
        for name in files:
            if name.startswith('.env') or Path(name).suffix not in SOURCE_EXTENSIONS:
                continue
            relative = (parent / name).relative_to(root).as_posix()
            selected[relative] = _checked_file(root, relative)
    if 'frontend/src/main.ts' not in selected or 'frontend/src/App.vue' not in selected:
        raise ValueError('Missing frontend entry points')
    return sorted(selected.items())


def create_archive(app: Path, output: Path) -> dict:
    entries = collect_files(app)
    target = output.absolute()
    if _is_link(target.parent) or target.parent.resolve() != target.parent:
        raise ValueError('Output directory must not resolve through links or junctions')
    records = []
    # Fail closed before opening an existing archive; all content is source code/assets.
    with target.open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w', format=tarfile.USTAR_FORMAT) as archive:
            for relative, path in entries:
                payload = path.read_bytes()
                info = tarfile.TarInfo(relative)
                info.size, info.mode, info.mtime = len(payload), 0o644, 0
                info.uid = info.gid = 0
                info.uname = info.gname = ''
                archive.addfile(info, io.BytesIO(payload))
                records.append({'path': relative, 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()})
    return {
        'application_only': True,
        'file_count': len(records),
        'files': records,
        'archive_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'archive_bytes': target.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    arguments = parser.parse_args()
    report = create_archive(arguments.app, arguments.output)
    if arguments.report:
        with arguments.report.open('x', encoding='utf-8') as output:
            output.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'files'}))


if __name__ == '__main__':
    main()
