"""Synthetic application-context tests; no real datasets or model files are read.

Run: python app/deploy/test_packaging.py
Optional real Docker allowlist/export check:
  KNEEOAI_TEST_DOCKER_CONTEXT=1 python app/deploy/test_packaging.py
The optional check uses only a generated fake tree and FROM scratch.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import build_context

APP = Path(__file__).resolve().parents[1]
FORBIDDEN = (
    '.env', 'backend/.env', 'backend/.env.production', 'backend/runtime/session.key',
    'backend/runtime/application.sqlite3', 'backend/runtime/private_objects/fictional.dcm',
    'backend/.venv/lib/site-packages/unrelated.py', 'backend/knee_app/fictional.dcm',
    'frontend/node_modules/unrelated.js', 'frontend/dist/index.html',
    'frontend/public/demo/radiografia-demo-a.svg', 'frontend/public/patient.dcm',
    'frontend/src/private/private.ts', 'frontend/src/secrets/credential.ts',
    'frontend/src/data/labels.ts', 'frontend/src/runtime/session.ts',
    'frontend/src/models/weights.ts', 'frontend/src/node_modules/package.ts',
    'frontend/src/.env.production', 'frontend/src/raw.csv',
    'frontend/src/weights.pth', 'frontend/src/case.npy',
    'notebooks/training.ipynb', 'weights/model.pt', 'models/snapshot.safetensors',
    'src/knee/scientific.py', 'data/reserved.csv', 'deploy/compose.production.yml',
)
PRIVATE_MARKER = b'SYNTHETIC_FORBIDDEN_CONTENT_NEVER_PACKAGE_80637'


def write(root: Path, relative: str, payload: bytes = b'public synthetic source\n') -> Path:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return destination


class PackagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp(prefix='knee-packaging-')).resolve()
        self.app = self.temp / 'app'
        self.app.mkdir()
        for name in build_context.STATIC_FILES:
            write(self.app, name)
        write(self.app, 'Dockerfile', (APP / 'Dockerfile').read_bytes())
        write(self.app, '.dockerignore', (APP / '.dockerignore').read_bytes())
        for name in ('__init__.py', 'api.py', 'deploy.py', 'security.py'):
            write(self.app, f'backend/knee_app/{name}')
        for name in ('main.ts', 'App.vue', 'styles.css', 'components/Public.vue', 'state/connection.ts'):
            write(self.app, f'frontend/src/{name}')
        for name in FORBIDDEN:
            write(self.app, name, PRIVATE_MARKER)

    def tearDown(self):
        target = self.temp.resolve()
        self.assertTrue(target.is_relative_to(Path(tempfile.gettempdir()).resolve()))
        self.assertTrue(target.name.startswith('knee-packaging-'))
        shutil.rmtree(target)

    def test_context_contains_only_public_sources_and_never_reads_private_files(self):
        output = self.temp / 'context.tar'
        real_read = Path.read_bytes
        visited = []
        def checked_read(path):
            visited.append(path)
            self.assertNotIn(path, [self.app / name for name in FORBIDDEN])
            return real_read(path)
        with patch.object(Path, 'read_bytes', checked_read):
            report = build_context.create_archive(self.app, output)
        with tarfile.open(output) as archive:
            names = archive.getnames()
            self.assertIn('backend/knee_app/deploy.py', names)
            self.assertIn('frontend/src/components/Public.vue', names)
            self.assertIn('frontend/public/icons/kneeoai-512.png', names)
            for name in FORBIDDEN:
                self.assertNotIn(name, names)
            for member in archive:
                self.assertTrue(member.isfile())
                self.assertEqual(member.mode, 0o644)
                self.assertEqual((member.uid, member.gid, member.mtime), (0, 0, 0))
                self.assertNotIn(PRIVATE_MARKER, archive.extractfile(member).read())
        self.assertTrue(report['application_only'])
        self.assertEqual(report['file_count'], len(names))
        self.assertEqual(report['archive_sha256'], hashlib.sha256(output.read_bytes()).hexdigest())

    def test_context_is_byte_reproducible_when_host_timestamps_change(self):
        first, second = self.temp / 'first.tar', self.temp / 'second.tar'
        a = build_context.create_archive(self.app, first)
        for _, path in build_context.collect_files(self.app):
            os.utime(path, (12345, 67890))
        b = build_context.create_archive(self.app, second)
        self.assertEqual(a['archive_sha256'], b['archive_sha256'])
        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_existing_archive_is_not_overwritten(self):
        destination = self.temp / 'keep.tar'
        destination.write_bytes(b'previous archive remains')
        with self.assertRaises(FileExistsError):
            build_context.create_archive(self.app, destination)
        self.assertEqual(destination.read_bytes(), b'previous archive remains')

    def test_missing_deployment_entrypoint_fails_before_creating_archive(self):
        (self.app / 'backend/knee_app/deploy.py').unlink()
        destination = self.temp / 'context.tar'
        with self.assertRaisesRegex(ValueError, 'deploy.py'):
            build_context.create_archive(self.app, destination)
        self.assertFalse(destination.exists())

    def test_junction_or_symlink_under_permitted_source_tree_is_rejected(self):
        elsewhere = self.temp / 'elsewhere'
        elsewhere.mkdir()
        write(elsewhere, 'hidden.ts', PRIVATE_MARKER)
        link = self.app / 'frontend/src/linked'
        if os.name == 'nt':
            import _winapi
            _winapi.CreateJunction(str(elsewhere), str(link))
        else:
            link.symlink_to(elsewhere, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Links/junctions'):
            build_context.collect_files(self.app)

    def test_source_file_link_is_rejected_before_read(self):
        candidate = self.app / 'frontend/src/main.ts'
        real_is_link = build_context._is_link
        with patch.object(build_context, '_is_link', lambda path: path == candidate or real_is_link(path)):
            with self.assertRaisesRegex(ValueError, 'Links/junctions'):
                build_context.collect_files(self.app)

    def test_large_unexpected_source_file_fails_without_reading_payload(self):
        candidate = self.app / 'frontend/src/oversized.ts'
        with candidate.open('wb') as stream:
            stream.seek(build_context.MAX_PUBLIC_FILE_BYTES)
            stream.write(b'0')
        with self.assertRaisesRegex(ValueError, 'unexpectedly large'):
            build_context.collect_files(self.app)

    def test_dockerfile_has_pinned_multistage_nonroot_runtime_and_no_broad_copy(self):
        text = (APP / 'Dockerfile').read_text(encoding='utf-8')
        bases = re.findall(r'^FROM (\S+)', text, re.MULTILINE)
        self.assertEqual(len(bases), 3)
        for base in bases:
            self.assertRegex(base, r'^(node:22\.17\.0-bookworm-slim|python:3\.12\.14-slim-bookworm)@sha256:[a-f0-9]{64}$')
        self.assertIn('npm ci --ignore-scripts --no-audit --no-fund', text)
        self.assertIn('--only-binary=:all: --no-deps -r requirements.lock', text)
        self.assertRegex(text, r'(?m)^USER 10001:10001$')
        self.assertIn('ENTRYPOINT ["python", "-m", "knee_app.deploy"]', text)
        self.assertIn('KNEEOAI_ENV=production', text)
        self.assertIn('KNEEOAI_DATA_DIR=/var/lib/kneeoai', text)
        self.assertNotRegex(text, r'(?m)^(ADD|COPY)\s+(\.\s|/|\*|backend/\s|frontend/\s)')
        self.assertNotRegex(text, r'(?im)^\s*(ARG|ENV).*?(SECRET|PASSWORD|TOKEN|FROZEN)')
        final = text.split(' AS application\n', 1)[1]
        self.assertNotIn('npm', final)
        self.assertNotIn('COPY frontend/src', final)

    @unittest.skipUnless(os.environ.get('KNEEOAI_TEST_DOCKER_CONTEXT') == '1', 'opt-in Docker context export check')
    def test_real_docker_context_export_excludes_synthetic_forbidden_files(self):
        # No base image download, no network, no real application data: just COPY into scratch.
        write(self.app, 'Dockerfile', b'FROM scratch\nCOPY . /context/\n')
        export = self.temp / 'export'
        # Use the caller's selected Docker engine on Desktop or a Linux CI runner.
        command = ['docker', 'build', '--progress=plain',
                   '--network=none', '--output', f'type=local,dest={export}', str(self.app)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        exported = export / 'context'
        names = {path.relative_to(exported).as_posix() for path in exported.rglob('*') if path.is_file()}
        expected = {name for name, _ in build_context.collect_files(self.app)}
        self.assertEqual(names, expected)
        for path in exported.rglob('*'):
            if path.is_file():
                self.assertNotIn(PRIVATE_MARKER, path.read_bytes())


if __name__ == '__main__':
    unittest.main(verbosity=2)
