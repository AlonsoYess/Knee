"""Fingerprint public application sources and publish an immutable local build copy.

No database, credentials, images of research subjects, or scientific assets are read.
"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import os
import shutil
import tempfile


def regular(path):
    if path.is_symlink() or getattr(path, 'is_junction', lambda: False)():
        raise ValueError('No se admiten enlaces en la distribución de la aplicación.')
    return path.is_file()


def source_fingerprint(frontend):
    frontend = Path(frontend).resolve(strict=True)
    names = ['src', 'public', 'pwa', 'scripts', 'index.html', 'vite.config.ts', 'package.json', 'package-lock.json']
    paths = []
    for name in names:
        entry = frontend / name
        if entry.is_symlink() or getattr(entry, 'is_junction', lambda: False)():
            raise ValueError('No se admiten enlaces en las fuentes de la interfaz.')
        if entry.is_file():
            paths.append(entry)
        elif entry.is_dir():
            for candidate in entry.rglob('*'):
                if candidate.is_symlink() or getattr(candidate, 'is_junction', lambda: False)():
                    raise ValueError('No se admiten enlaces en las fuentes de la interfaz.')
                if candidate.is_file():
                    paths.append(candidate)
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(frontend).as_posix().encode())
        digest.update(b'\0')
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def inventory(directory):
    if directory.is_symlink() or getattr(directory, 'is_junction', lambda: False)():
        raise ValueError('No se admiten enlaces en la distribución.')
    result, size = {}, 0
    for path in sorted(directory.rglob('*')):
        if path.is_symlink() or getattr(path, 'is_junction', lambda: False)():
            raise ValueError('No se admiten enlaces en la distribución.')
        if not regular(path):
            continue
        name = path.relative_to(directory).as_posix()
        top_files = {'index.html', 'sw.js', 'manifest.webmanifest', 'pwa-version.json'}
        public_asset = PurePosixPath(name).parts[0] in {'assets', 'icons', 'demo'}
        extensions = {'.js', '.css', '.png', '.svg', '.ico', '.webp', '.woff', '.woff2', '.ttf'}
        if name not in top_files and not (public_asset and path.suffix.lower() in extensions):
            raise ValueError('La distribución contiene un archivo ajeno a la interfaz pública.')
        data = path.read_bytes()
        size += len(data)
        if size > 32 * 1024 * 1024 or len(result) >= 500:
            raise ValueError('La distribución supera el límite técnico de publicación local.')
        result[name] = hashlib.sha256(data).hexdigest()
    if not {'index.html', 'sw.js', 'manifest.webmanifest', 'pwa-version.json'} <= result.keys():
        raise ValueError('Falta compilar la distribución PWA completa.')
    return result


def snapshot(dist, root):
    original = Path(dist).absolute()
    regular(original)  # Reject an explicitly supplied symlink before resolution.
    dist = original.resolve(strict=True)
    root = Path(root).absolute()
    if root.is_symlink() or getattr(root, 'is_junction', lambda: False)():
        raise ValueError('El destino de distribución no admite enlaces.')
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    entries = inventory(dist)
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    destination = root / digest
    if destination.exists():
        if inventory(destination) != entries:
            raise ValueError('La copia existente cambió; se conservó para revisar su integridad.')
        return {'snapshotPath': str(destination), 'digest': digest}
    stage = Path(tempfile.mkdtemp(prefix='.build-', dir=root))
    try:
        for name in entries:
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(dist / name, target)
        if inventory(stage) != entries:
            raise ValueError('No se pudo comprobar la copia de distribución.')
        os.rename(stage, destination)
        stage = None
    finally:
        if stage is not None:
            checked = stage.resolve()
            if checked.parent != root or not checked.name.startswith('.build-'):
                raise ValueError('No se pudo verificar la ubicación temporal de compilación.')
            shutil.rmtree(checked)
    return {'snapshotPath': str(destination), 'digest': digest}


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    fingerprint = commands.add_parser('fingerprint')
    fingerprint.add_argument('--frontend', required=True, type=Path)
    publish = commands.add_parser('snapshot')
    publish.add_argument('--dist', required=True, type=Path)
    publish.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    try:
        result = source_fingerprint(args.frontend) if args.command == 'fingerprint' else snapshot(args.dist, args.root)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (ValueError, OSError):
        print('No se pudo verificar la distribución pública de KneeOAI. Consulta la guía local.', file=__import__('sys').stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
