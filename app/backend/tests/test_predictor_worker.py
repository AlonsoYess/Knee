"""Closed technical probe boundaries, using invented files only."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from knee_app import predictor_worker as worker


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def fixture_bundle(tmp_path):
    checkout, root = tmp_path / 'source', tmp_path / 'bundle'
    checkout.mkdir()
    root.mkdir()
    sources = {}
    for name in worker.MINIMUM_SOURCES:
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# Invented source for boundary checks only.\n', encoding='utf-8')
        sources[name] = checksum(path)
    models = []
    for fold in range(5):
        pair = {'fold': fold}
        for role in ('weights', 'clinical_state'):
            path = root / f'fold_{fold}_{role}.json'
            path.write_text(json.dumps({'synthetic_fold': fold, 'synthetic_role': role}), encoding='utf-8')
            pair[role] = {'path': path.name, 'sha256': checksum(path)}
        models.append(pair)
    manifest = {'format': worker.FORMAT, 'purpose': worker.PURPOSE,
                'clinical_release_ready': False, 'python_version': worker.PYTHON_VERSION,
                'numpy_version': worker.NUMPY_VERSION, 'selection_record_sha256': 'e' * 64,
                'scientific_sources': sources, 'models': models}
    payload = {'operation': 'preflight', 'checkout': str(checkout), 'bundle': str(root)}

    def save():
        path = root / 'manifest.json'
        path.write_text(json.dumps(manifest), encoding='utf-8')
        payload['manifestSha256'] = checksum(path)
        return payload

    save()
    return root, checkout, manifest, payload, save


def test_authenticates_exact_five_fold_bundle_without_loading_scientific_runtime(fixture_bundle):
    root, checkout, _, payload, _ = fixture_bundle
    bundle = worker.load_bundle(payload)
    assert bundle.root == root and bundle.checkout == checkout
    assert [row[0] for row in bundle.models] == list(range(5))
    assert len(bundle.pins()) == 15


@pytest.mark.parametrize('extra', [{'patient': {'age': 62}}, {'fold': 0}, {'operation': 'fit'},
                                  {'operation': []}, {'operation': True}, {'manifestSha256': 'no'}])
def test_rejects_patient_input_fold_choice_and_nonregistered_request(fixture_bundle, extra):
    payload = {**fixture_bundle[3], **extra}
    with pytest.raises(worker.WorkerError, match='PREDICTOR_REQUEST_INVALID'):
        worker.load_bundle(payload)


@pytest.mark.parametrize('change', ['clinical', 'purpose', 'version', 'extra', 'bool_fold', 'duplicate_fold', 'missing_source'])
def test_rejects_release_promotion_unpinned_schema_or_ambiguous_fold(fixture_bundle, change):
    _, _, manifest, payload, save = fixture_bundle
    if change == 'clinical':
        manifest['clinical_release_ready'] = True
    elif change == 'purpose':
        manifest['purpose'] = 'clinical_prediction'
    elif change == 'version':
        manifest['numpy_version'] = 'unverified'
    elif change == 'extra':
        manifest['threshold'] = 0.5
    elif change == 'bool_fold':
        manifest['models'][0]['fold'] = False
    elif change == 'duplicate_fold':
        manifest['models'][4]['fold'] = 0
    else:
        manifest['scientific_sources'].pop(next(iter(worker.MINIMUM_SOURCES)))
    save()
    with pytest.raises(worker.WorkerError, match='PREDICTOR_BUNDLE_INVALID'):
        worker.load_bundle(payload)


@pytest.mark.parametrize('role', ['manifest', 'source', 'weights', 'state'])
def test_fails_closed_on_any_pinned_file_change(fixture_bundle, role):
    root, checkout, manifest, payload, _ = fixture_bundle
    path = {'manifest': root / 'manifest.json',
            'source': checkout / next(iter(manifest['scientific_sources'])),
            'weights': root / manifest['models'][0]['weights']['path'],
            'state': root / manifest['models'][0]['clinical_state']['path']}[role]
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(worker.WorkerError, match='PREDICTOR_(ARTIFACT|SOURCE)_MISMATCH'):
        worker.load_bundle(payload)


@pytest.mark.parametrize('name', ['../outside.json', '/outside.json', 'C:/outside.json',
                                 'folder\\outside.json', 'fold_0_weights.json/../fold_0_weights.json'])
def test_denies_escape_and_nonportable_paths(fixture_bundle, name):
    _, _, manifest, payload, save = fixture_bundle
    manifest['models'][0]['weights']['path'] = name
    save()
    with pytest.raises(worker.WorkerError, match='PREDICTOR_BUNDLE_INVALID'):
        worker.load_bundle(payload)


def test_denies_link_to_external_artifact(fixture_bundle, tmp_path):
    root, _, manifest, payload, save = fixture_bundle
    outside = tmp_path / 'outside.json'
    outside.write_text('{}', encoding='utf-8')
    link = root / 'linked.json'
    junction = False
    try:
        link.symlink_to(outside)
    except OSError:
        if os.name != 'nt':
            pytest.skip('File links are unavailable to this account')
        # A directory junction exercises the same ancestor-link rejection on
        # Windows accounts that cannot create file symlinks.
        link = root / 'linked-directory'
        result = subprocess.run(['cmd.exe', '/c', 'mklink', '/J', str(link), str(tmp_path)],
                                capture_output=True, text=True, timeout=10)
        assert result.returncode == 0 and link.is_junction(), result.stderr
        junction = True
    try:
        name = link.name + '/' + outside.name if junction else link.name
        manifest['models'][0]['weights'] = {'path': name, 'sha256': checksum(outside)}
        save()
        with pytest.raises(worker.WorkerError, match='PREDICTOR_BUNDLE_INVALID'):
            worker.load_bundle(payload)
    finally:
        if junction:
            assert link.parent == root and link.resolve() == tmp_path
            link.rmdir()


@pytest.mark.parametrize('raw', [b'{"operation":"preflight","operation":"diagnostic"}', b'{"x":NaN}', b'{"x":Infinity}'])
def test_rejects_duplicate_fields_and_nonfinite_json(raw):
    with pytest.raises(worker.WorkerError, match='PREDICTOR_REQUEST_INVALID'):
        worker.strict_json(raw, 'PREDICTOR_REQUEST_INVALID')


def test_importer_compiles_pinned_source_instead_of_stale_bytecode(tmp_path):
    source = tmp_path / 'frozen.py'
    source.write_text('VALUE = "authenticated_source"\n', encoding='utf-8')
    source.with_suffix('.pyc').write_bytes(b'not_authenticated_bytecode')
    pin = worker.Pin(source, checksum(source), 1024, 'PREDICTOR_SOURCE_MISMATCH')
    finder = worker.ScientificFinder({'src/knee/frozen.py': pin})
    spec = finder.find_spec('knee.frozen')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.VALUE == 'authenticated_source'
    source.write_text('VALUE = "modified"\n', encoding='utf-8')
    with pytest.raises(worker.WorkerError, match='PREDICTOR_SOURCE_MISMATCH'):
        spec.loader.exec_module(module)
    with pytest.raises(worker.WorkerError, match='PREDICTOR_SOURCE_MISMATCH'):
        finder.find_spec('knee.unlisted')


def test_read_only_policy_denies_other_files_writes_network_and_processes(tmp_path):
    allowed = tmp_path / 'allowed.json'
    allowed.write_bytes(b'{}')
    pin = worker.Pin(allowed, checksum(allowed), 100)
    policy = worker.ReadOnlyPolicy([pin], [tmp_path / 'python-runtime'])
    policy('open', (str(allowed), 'r', os.O_RDONLY))
    for event, arguments in [('open', (str(tmp_path / 'cohort.csv'), 'r', os.O_RDONLY)),
                             ('open', (str(allowed), 'w', os.O_WRONLY)),
                             ('open', (str(allowed), None, os.O_CREAT)),
                             ('socket.connect', (None, ('example.invalid', 443))),
                             ('subprocess.Popen', ()), ('os.mkdir', (str(tmp_path / 'new'),)),
                             ('os.rename', (str(allowed), str(tmp_path / 'other'))),
                             ('os.scandir', (str(tmp_path / 'private-cohort'),))]:
        with pytest.raises(worker.WorkerError, match='PREDICTOR_SAFETY_VIOLATION'):
            policy(event, arguments)


def test_audit_hook_in_child_really_blocks_file_write_and_socket_without_contaminating_pytest(tmp_path):
    target = tmp_path / 'must_not_exist'
    code = '''
import json,socket,sys
from pathlib import Path
from knee_app.predictor_worker import ReadOnlyPolicy,WorkerError
sys.addaudithook(ReadOnlyPolicy([], {sys.prefix,sys.base_prefix}))
out=[]
for operation in (lambda:Path(sys.argv[1]).write_text('forbidden'),lambda:socket.socket()):
 try: operation()
 except WorkerError as error:out.append(error.code)
print(json.dumps(out))
'''
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(Path(worker.__file__).resolve().parents[1])
    result = subprocess.run([sys.executable, '-B', '-c', code, str(target)], env=environment,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ['PREDICTOR_SAFETY_VIOLATION'] * 2
    assert not target.exists()


@pytest.mark.parametrize('raw', [b'{"private":"do_not_echo_this"}', b'x' * (worker.MAX_REQUEST + 1)])
def test_cli_outputs_only_bounded_safe_json_on_bad_requests(raw):
    result = subprocess.run([sys.executable, '-B', worker.__file__], input=raw,
                            capture_output=True, timeout=10)
    assert result.returncode == 1
    assert result.stderr == b'' and len(result.stdout) <= worker.MAX_RESPONSE
    assert json.loads(result.stdout) == {'ok': False, 'error': {'code': 'PREDICTOR_REQUEST_INVALID'}}
    assert b'do_not_echo_this' not in result.stdout
