"""Technical predictor bridge checks; no real model, cohort or clinical result."""
from contextlib import closing
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from knee_app.config import Settings
from knee_app.predictor import FrozenPredictor, PredictorError


PRIVATE_DETAIL = '/private/fictional-weights/never-publish-this-path'


def technical_response(operation='preflight'):
    return {
        'scope': 'technical_development_check_only',
        'clinicalReleaseReady': False,
        'modelsVerified': 5,
        'operation': operation,
        'runtime': {'python': '3.12.14', 'numpy': '2.5.3'},
        'foldDiagnostics': [] if operation == 'preflight' else [
            {'fold': fold, 'logits': [-0.25, 0.0, 0.25]} for fold in range(5)
        ],
    }


VALID_WIRE = json.dumps({'ok': True, 'data': technical_response()})


def completed(data=None, *, raw=None, returncode=0):
    if raw is None:
        raw = json.dumps({'ok': True, 'data': data or technical_response()})
    return SimpleNamespace(returncode=returncode, stdout=raw, stderr=PRIVATE_DETAIL)


@pytest.fixture
def predictor_settings(tmp_path):
    python = tmp_path / 'isolated-python.exe'
    python.write_bytes(b'not-an-executable: subprocess is replaced in these tests')
    checkout = tmp_path / 'frozen-checkout'
    (checkout / 'src' / 'knee').mkdir(parents=True)
    (checkout / 'src' / 'knee' / '__init__.py').write_text('', encoding='utf-8')
    (checkout / 'src' / 'knee' / 'sensitivity_real_inference.py').write_text('# Synthetic public source marker\n', encoding='utf-8')
    bundle = tmp_path / 'private-technical-bundle'
    bundle.mkdir()
    manifest = bundle / 'manifest.json'
    manifest.write_text('{}', encoding='utf-8')
    return Settings(
        database_path=tmp_path / 'application.sqlite3', environment='test',
        secret_key=b'fictional-predictor-connection-key',
        allowed_origins=('http://testserver',),
        predictor_python=python, predictor_checkout=checkout, predictor_bundle=bundle,
        predictor_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        predictor_timeout_seconds=4,
    )


def fake_run(monkeypatch, value):
    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr('knee_app.predictor.subprocess.run', run)
    return calls


def test_absent_predictor_is_not_configured_and_does_not_spawn(tmp_path, monkeypatch):
    calls = fake_run(monkeypatch, AssertionError('An absent configuration must not spawn'))
    predictor = FrozenPredictor(Settings(database_path=tmp_path / 'unused.sqlite3'))
    status = predictor.engine_status()
    assert status['status'] == 'not_configured' and status['package'] is None
    assert set(status) == {'status', 'package', 'message'}
    assert not calls and not (tmp_path / 'unused.sqlite3').exists()


def test_partial_predictor_configuration_is_unavailable_without_fallback(tmp_path, monkeypatch):
    python = tmp_path / 'fake-python.exe'
    python.write_bytes(b'fake')
    calls = fake_run(monkeypatch, AssertionError('A partial configuration must not spawn'))
    predictor = FrozenPredictor(Settings(database_path=tmp_path / 'unused.sqlite3', predictor_python=python))
    status = predictor.engine_status()
    assert status['status'] == 'unavailable' and status['package'] is None
    assert not calls and str(python) not in status['message']


@pytest.mark.parametrize('operation', ['preflight', 'diagnostic'])
def test_worker_protocol_is_separate_and_never_promotes_technical_check_to_release(predictor_settings, monkeypatch, operation):
    expected = technical_response(operation)
    calls = fake_run(monkeypatch, completed(expected))
    monkeypatch.setenv('PYTHONHOME', PRIVATE_DETAIL)
    monkeypatch.setenv('PYTHONPATH', PRIVATE_DETAIL)
    predictor = FrozenPredictor(predictor_settings)
    assert getattr(predictor, operation)() == expected
    assert len(calls) == 1
    arguments, kwargs = calls[0]
    assert Path(arguments[0][0]) == predictor_settings.predictor_python
    assert not kwargs.get('shell', False)
    request = json.loads(kwargs['input'])
    assert request == {
        'operation': operation, 'checkout': str(predictor_settings.predictor_checkout),
        'bundle': str(predictor_settings.predictor_bundle),
        'manifestSha256': predictor_settings.predictor_manifest_sha256,
    }
    assert kwargs['timeout'] == 4
    assert 'PYTHONHOME' not in kwargs['env']
    assert kwargs['env']['PYTHONPATH'] != PRIVATE_DETAIL
    assert not predictor_settings.database_path.exists()


def test_successful_connection_status_still_has_no_clinical_package(predictor_settings, monkeypatch):
    fake_run(monkeypatch, completed())
    status = FrozenPredictor(predictor_settings).engine_status()
    assert status['status'] == 'not_configured' and status['package'] is None
    assert set(status) == {'status', 'package', 'message'}
    assert str(predictor_settings.predictor_bundle) not in status['message']


@pytest.mark.parametrize('raw', [
    'not-json-' + PRIVATE_DETAIL,
    VALID_WIRE.replace('"ok": true', '"ok": true, "ok": true', 1),
    VALID_WIRE.replace('"modelsVerified": 5', '"modelsVerified": NaN', 1),
    VALID_WIRE.replace('"python": "3.12.14"', '"python": "3.12.14", "python": "3.12.14"', 1),
    VALID_WIRE + ' trailing',
    VALID_WIRE + ' ' * 2_097_153,
])
def test_malformed_worker_output_never_becomes_ready_or_leaks_details(predictor_settings, monkeypatch, raw):
    fake_run(monkeypatch, completed(raw=raw))
    predictor = FrozenPredictor(predictor_settings)
    with pytest.raises(PredictorError) as caught:
        predictor.preflight()
    assert isinstance(caught.value.code, str)
    assert PRIVATE_DETAIL not in str(caught.value)
    status = predictor.engine_status()
    assert status['status'] == 'unavailable' and status['package'] is None
    assert PRIVATE_DETAIL not in status['message']


@pytest.mark.parametrize('change', [
    {'clinicalReleaseReady': True}, {'modelsVerified': True}, {'modelsVerified': 4},
    {'scope': 'clinical_release'}, {'operation': 'diagnostic'},
    {'unexpected': PRIVATE_DETAIL}, {'runtime': {'python': PRIVATE_DETAIL, 'numpy': '2.5.3'}},
    {'foldDiagnostics': [{'fold': 0, 'logits': [0.0, 0.0, 0.0]}]},
])
def test_invalid_preflight_semantics_cannot_enable_prediction(predictor_settings, monkeypatch, change):
    data = technical_response()
    data.update(change)
    fake_run(monkeypatch, completed(data))
    with pytest.raises(PredictorError):
        FrozenPredictor(predictor_settings).preflight()


@pytest.mark.parametrize('bad_logit', [float('nan'), float('inf'), True, '0.2'])
def test_diagnostic_rejects_nonfinite_or_non_numeric_logits(predictor_settings, monkeypatch, bad_logit):
    data = technical_response('diagnostic')
    data['foldDiagnostics'][0]['logits'][0] = bad_logit
    fake_run(monkeypatch, completed(data))
    with pytest.raises(PredictorError):
        FrozenPredictor(predictor_settings).diagnostic()


def test_diagnostic_requires_five_distinct_folds(predictor_settings, monkeypatch):
    data = technical_response('diagnostic')
    data['foldDiagnostics'][-1]['fold'] = 0
    fake_run(monkeypatch, completed(data))
    with pytest.raises(PredictorError):
        FrozenPredictor(predictor_settings).diagnostic()


@pytest.mark.parametrize('failure', [
    subprocess.TimeoutExpired(cmd=['fictional-python'], timeout=4, stderr=PRIVATE_DETAIL),
    OSError(PRIVATE_DETAIL),
])
def test_process_failure_is_unavailable_without_private_error_details(predictor_settings, monkeypatch, failure):
    fake_run(monkeypatch, failure)
    predictor = FrozenPredictor(predictor_settings)
    with pytest.raises(PredictorError) as caught:
        predictor.preflight()
    assert PRIVATE_DETAIL not in str(caught.value)
    status = predictor.engine_status()
    assert status['status'] == 'unavailable' and status['package'] is None
    assert PRIVATE_DETAIL not in status['message']


@pytest.mark.parametrize('raw', [
    json.dumps({'ok': False, 'error': {'code': 'UNKNOWN_' + PRIVATE_DETAIL}}),
    json.dumps({'ok': True, 'data': technical_response()}),
])
def test_failed_worker_exit_cannot_be_accepted_as_success(predictor_settings, monkeypatch, raw):
    fake_run(monkeypatch, completed(raw=raw, returncode=1))
    with pytest.raises(PredictorError):
        FrozenPredictor(predictor_settings).preflight()


@pytest.mark.parametrize('value', ['', 'not-a-sha256', 'a' * 63, 'g' * 64, True])
def test_invalid_manifest_digest_is_rejected_by_configuration(tmp_path, value):
    with pytest.raises((ValueError, TypeError)):
        Settings(database_path=tmp_path / 'unused.sqlite3', predictor_manifest_sha256=value)


def test_nonpositive_predictor_timeout_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        Settings(database_path=tmp_path / 'unused.sqlite3', predictor_timeout_seconds=0)


def test_predictor_settings_use_only_explicit_environment_and_keep_other_engines_separate(predictor_settings):
    source = {
        'KNEEOAI_ENV': 'test',
        'KNEEOAI_PREDICTOR_PYTHON': str(predictor_settings.predictor_python),
        'KNEEOAI_PREDICTOR_CHECKOUT': str(predictor_settings.predictor_checkout),
        'KNEEOAI_PREDICTOR_BUNDLE': str(predictor_settings.predictor_bundle),
        'KNEEOAI_PREDICTOR_MANIFEST_SHA256': predictor_settings.predictor_manifest_sha256,
        'KNEEOAI_PREDICTOR_TIMEOUT_SECONDS': '4',
    }
    configured = Settings.from_env(source)
    for name in ('predictor_python', 'predictor_checkout', 'predictor_bundle', 'predictor_manifest_sha256'):
        assert getattr(configured, name) == getattr(predictor_settings, name)
    assert configured.predictor_timeout_seconds == 4
    assert configured.frozen_python is None and configured.frozen_checkout is None


def test_linked_predictor_path_is_unavailable_before_starting_a_worker(predictor_settings, monkeypatch):
    original = Path.is_symlink

    def linked(path):
        return path == predictor_settings.predictor_bundle or original(path)

    monkeypatch.setattr(Path, 'is_symlink', linked)
    calls = fake_run(monkeypatch, AssertionError('A linked package must not spawn'))
    status = FrozenPredictor(predictor_settings).engine_status()
    assert status['status'] == 'unavailable' and status['package'] is None
    assert not calls and str(predictor_settings.predictor_bundle) not in status['message']


class StatusDouble:
    def __init__(self, response):
        self.response, self.calls = response, 0

    def engine_status(self):
        self.calls += 1
        if isinstance(self.response, BaseException):
            raise self.response
        return deepcopy(self.response)


def test_extreme_integer_in_diagnostic_is_rejected_without_traceback(predictor_settings, monkeypatch):
    response = technical_response('diagnostic')
    response['foldDiagnostics'][0]['logits'][0] = 10 ** 500
    fake_run(monkeypatch, completed(response))
    with pytest.raises(PredictorError):
        FrozenPredictor(predictor_settings).diagnostic()


def test_engine_status_requires_session_before_calling_predictor(api):
    predictor = StatusDouble({'status': 'not_configured', 'package': None, 'message': 'No configurado'})
    api.app.state.predictor = predictor
    client = api.client()
    api.check(client.get('/api/v1/engine-status'), 'get', '/engine-status', 401)
    assert predictor.calls == 0


@pytest.mark.parametrize('payload', [
    {'status': 'ready', 'package': {'path': PRIVATE_DETAIL}, 'message': PRIVATE_DETAIL},
    {'status': 'not_configured', 'package': None, 'message': PRIVATE_DETAIL},
    {'status': 'unavailable', 'package': None, 'message': PRIVATE_DETAIL, 'path': PRIVATE_DETAIL},
    {'status': 'future_status', 'package': None, 'message': PRIVATE_DETAIL},
    None,
    RuntimeError(PRIVATE_DETAIL),
])
def test_api_sanitizes_predictor_status_and_never_publishes_clinical_readiness(api, payload):
    predictor = StatusDouble(payload)
    api.app.state.predictor = predictor
    client = api.client()
    api.login(client)
    body = api.check(client.get('/api/v1/engine-status'), 'get', '/engine-status', 200)
    assert body['status'] in {'not_configured', 'unavailable'} and body['package'] is None
    assert PRIVATE_DETAIL not in json.dumps(body)
    assert set(body) == {'status', 'package', 'message'}
    assert predictor.calls == 1


def test_configured_technical_predictor_still_cannot_create_an_execution(api):
    predictor = StatusDouble({'status': 'ready', 'package': {'path': PRIVATE_DETAIL}, 'message': PRIVATE_DETAIL})
    api.app.state.predictor = predictor
    client = api.client()
    api.login(client)
    created = api.create(client)
    identifier = created.json()['id']
    route = f'/api/v1/evaluations/{identifier}/executions'
    api.check(client.post(route, headers=api.headers(client), json={}), 'post', '/evaluations/{evaluationId}/executions', 503)
    after = client.get(f'/api/v1/evaluations/{identifier}')
    assert after.json() == created.json() and after.headers['etag'] == created.headers['etag']
    assert predictor.calls == 0
    assert api.revision_count(identifier) == 1
    with closing(api.app.state.db.connect()) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert 'executions' not in tables and 'prediction_results' not in tables
