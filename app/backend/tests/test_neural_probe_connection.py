"""One-job IPC tests with fake processes; no neural dependencies or real weights."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from knee_app.neural_probe import NeuralProbe, NeuralProbeError, RUNTIME
from knee_app import neural_probe_check


PRIVATE = '/private/fictional-neural-weights/never-publish'


@pytest.fixture
def config(tmp_path):
    checkout = tmp_path / 'checkout'
    (checkout / 'src/knee').mkdir(parents=True)
    (checkout / 'src/knee/sensitivity_real_inference.py').write_text('# synthetic source\n')
    python = tmp_path / 'runtime/Scripts/python.exe'
    python.parent.mkdir(parents=True)
    python.write_bytes(b'not executable; subprocess is replaced')
    bundle = tmp_path / 'private/neural_bundle'
    bundle.mkdir(parents=True)
    manifest = bundle / 'manifest.json'
    manifest.write_text('{}')
    scratch = bundle.parent / 'probe_scratch'
    scratch.mkdir()
    return {'python': str(python), 'checkout': str(checkout), 'bundle': str(bundle),
            'manifestSha256': hashlib.sha256(manifest.read_bytes()).hexdigest(), 'scratchRoot': str(scratch)}


def data(operation='preflight', slot='radiographic', fold=0):
    return {'scope': 'technical_development_check_only', 'clinicalReleaseReady': False,
            'slot': slot, 'fold': fold, 'operation': operation, 'weightsVerified': True,
            'logits': [] if operation == 'preflight' else [-0.25, 0.75],
            'runtime': dict(RUNTIME), 'elapsedSeconds': 0.5}


def response(value=None, *, raw=None, returncode=0):
    return SimpleNamespace(stdout=raw if raw is not None else json.dumps({'ok': True, 'data': value or data()}),
                           stderr=PRIVATE, returncode=returncode)


def fake_run(monkeypatch, result, callback=None):
    calls = []
    def run(*args, **kwargs):
        calls.append((args, kwargs))
        if callback:
            callback(json.loads(kwargs['input']), kwargs)
        if isinstance(result, BaseException):
            raise result
        return result
    monkeypatch.setattr('knee_app.neural_probe.subprocess.run', run)
    return calls


@pytest.mark.parametrize('operation,slot,fold', [
    ('preflight', 'radiographic', 0), ('preflight', 'multimodal', 4),
    ('diagnostic', 'radiographic', 4), ('diagnostic', 'multimodal', 0),
])
def test_one_job_uses_exact_protocol_and_cleans_only_its_scratch(config, monkeypatch, operation, slot, fold):
    root = Path(config['scratchRoot'])
    marker = root / 'unrelated-evidence.txt'
    marker.write_text('keep')
    observed = []
    def inspect(payload, kwargs):
        scratch = Path(payload['scratch'])
        assert scratch.is_dir() and list(scratch.iterdir()) == []
        observed.append(scratch)
        (scratch / 'worker-temporary.txt').write_text('fictional temporary output')
        assert set(payload) == {'operation', 'checkout', 'bundle', 'manifestSha256', 'slot', 'fold', 'scratch'}
        assert (payload['operation'], payload['slot'], payload['fold']) == (operation, slot, fold)
        assert kwargs['env']['TEMP'] == kwargs['env']['TMP'] == kwargs['env']['TMPDIR'] == str(scratch)
        assert kwargs['env']['HF_HUB_OFFLINE'] == '1'
        assert kwargs['env']['OMP_NUM_THREADS'] == '1'
        assert 'PYTHONHOME' not in kwargs['env']
        assert 'KNEEOAI_SECRET_KEY' not in kwargs['env']
        assert kwargs['env']['PYTHONPATH'] == str(Path(config['checkout']) / 'src')
        assert not kwargs.get('shell', False) and kwargs['timeout'] == 30
    monkeypatch.setenv('PYTHONHOME', PRIVATE)
    monkeypatch.setenv('PYTHONPATH', PRIVATE)
    monkeypatch.setenv('KNEEOAI_SECRET_KEY', PRIVATE)
    calls = fake_run(monkeypatch, response(data(operation, slot, fold)), inspect)
    assert getattr(NeuralProbe(config, timeout=30), operation)(slot, fold) == data(operation, slot, fold)
    assert len(calls) == 1 and len(observed) == 1
    assert not observed[0].exists() and marker.read_text() == 'keep'


@pytest.mark.parametrize('slot,fold', [('other', 0), ('radiographic', True), ('multimodal', 5), ('radiographic', -1)])
def test_invalid_job_does_not_start_a_process(config, monkeypatch, slot, fold):
    calls = fake_run(monkeypatch, AssertionError('must not start'))
    with pytest.raises(NeuralProbeError) as caught:
        NeuralProbe(config).preflight(slot, fold)
    assert caught.value.code == 'NEURAL_PROBE_REQUEST_INVALID'
    assert calls == [] and list(Path(config['scratchRoot']).iterdir()) == []


@pytest.mark.parametrize('timeout', [0, -1, 601, float('nan'), float('inf'), True, '300'])
def test_timeout_is_explicit_and_bounded(config, timeout):
    with pytest.raises(NeuralProbeError):
        NeuralProbe(config, timeout=timeout)


def test_missing_configuration_file_is_not_a_model_fallback(config):
    Path(config['bundle'], 'manifest.json').unlink()
    with pytest.raises(NeuralProbeError):
        NeuralProbe(config)


@pytest.mark.parametrize('mutate', [
    lambda value: value.update(patient={'age': 62}),
    lambda value: value.update(manifestSha256='INVALID'),
    lambda value: value.update(python='relative.exe'),
    lambda value: value.update(scratchRoot=value['bundle']),
    lambda value: value.update(scratchRoot=value['checkout']),
    lambda value: value.update(scratchRoot=str(Path(value['python']).parent.parent)),
])
def test_invalid_config_or_scratch_overlap_is_rejected(config, mutate):
    mutate(config)
    with pytest.raises(NeuralProbeError):
        NeuralProbe(config)


@pytest.mark.parametrize('raw', [
    'invalid-' + PRIVATE,
    json.dumps({'ok': True, 'data': data()}).replace('"ok": true', '"ok": true, "ok": true'),
    json.dumps({'ok': True, 'data': data()}).replace('0.5', 'NaN'),
    json.dumps({'ok': True, 'data': data()}) + ' ' * 65536,
    '{"ok":false,"error":{"code":"ASSET_MISSING","message":"' + PRIVATE + '"}}',
])
def test_malformed_output_is_safe_and_scratch_is_cleaned(config, monkeypatch, raw):
    fake_run(monkeypatch, response(raw=raw))
    with pytest.raises(NeuralProbeError) as caught:
        NeuralProbe(config).preflight('radiographic', 0)
    assert PRIVATE not in str(caught.value)
    assert list(Path(config['scratchRoot']).iterdir()) == []


@pytest.mark.parametrize('change', [
    {'clinicalReleaseReady': True}, {'weightsVerified': False}, {'slot': 'multimodal'},
    {'fold': True}, {'fold': 1}, {'patient': {'id': 'fictional'}},
    {'runtime': {**RUNTIME, 'torch': 'other'}}, {'logits': [0.0, 1.0]},
    {'elapsedSeconds': -1}, {'elapsedSeconds': True}, {'elapsedSeconds': 10 ** 500},
])
def test_invalid_preflight_cannot_promote_a_model(config, monkeypatch, change):
    value = data()
    value.update(change)
    fake_run(monkeypatch, response(value))
    with pytest.raises(NeuralProbeError):
        NeuralProbe(config).preflight('radiographic', 0)
    assert list(Path(config['scratchRoot']).iterdir()) == []


@pytest.mark.parametrize('values', [[0.0], [0.0, True], [0, 1], [0.0, 10 ** 500], [0.0, float('inf')]])
def test_diagnostic_requires_two_finite_float_logits(config, monkeypatch, values):
    value = data('diagnostic')
    value['logits'] = values
    fake_run(monkeypatch, response(value))
    with pytest.raises(NeuralProbeError):
        NeuralProbe(config).diagnostic('radiographic', 0)


@pytest.mark.parametrize('error', [OSError(PRIVATE), subprocess.TimeoutExpired('fictional-worker', 30, stderr=PRIVATE)])
def test_process_failure_is_sanitized_and_owned_scratch_removed(config, monkeypatch, error):
    fake_run(monkeypatch, error)
    with pytest.raises(NeuralProbeError) as caught:
        NeuralProbe(config).preflight('radiographic', 0)
    assert PRIVATE not in str(caught.value)
    assert list(Path(config['scratchRoot']).iterdir()) == []


def test_missing_asset_error_is_preserved_without_leaking_details(config, monkeypatch):
    fake_run(monkeypatch, response(raw='{"ok":false,"error":{"code":"ASSET_MISSING"}}', returncode=1))
    with pytest.raises(NeuralProbeError) as caught:
        NeuralProbe(config).preflight('radiographic', 0)
    assert caught.value.code == 'ASSET_MISSING' and PRIVATE not in str(caught.value)


def test_cleanup_refuses_a_replaced_scratch_directory(config, monkeypatch):
    moved = []
    def replace(payload, kwargs):
        scratch = Path(payload['scratch'])
        previous = scratch.with_name(scratch.name + '-moved')
        scratch.rename(previous)
        scratch.mkdir()
        (scratch / 'unowned-marker').write_text('do not remove')
        moved.extend([previous, scratch])
    fake_run(monkeypatch, response(), replace)
    with pytest.raises(NeuralProbeError) as caught:
        NeuralProbe(config).preflight('radiographic', 0)
    assert caught.value.code == 'NEURAL_PROBE_SAFETY_VIOLATION'
    assert all(path.is_dir() for path in moved)
    assert (moved[1] / 'unowned-marker').read_text() == 'do not remove'


def test_cli_keeps_logits_and_paths_in_a_new_private_report(config, monkeypatch, tmp_path, capsys):
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps(config))
    report = tmp_path / 'report.json'
    fake_run(monkeypatch, response(data('diagnostic', 'multimodal', 4)))
    assert neural_probe_check.main(['--config', str(profile), '--report', str(report),
        '--slot', 'multimodal', '--fold', '4', '--diagnostic']) == 0
    printed = capsys.readouterr().out
    summary = json.loads(printed)
    assert summary['success'] is True and summary['clinicalReleaseReady'] is False
    assert 'logits' not in summary and config['bundle'] not in printed and PRIVATE not in printed
    saved = json.loads(report.read_text())
    assert saved['data']['logits'] == [-0.25, 0.75]
    assert saved['clinicalResultsEnabled'] is False


@pytest.mark.parametrize('protected', ['checkout', 'bundle', 'scratchRoot'])
def test_cli_rejects_reports_inside_protected_directories(config, monkeypatch, tmp_path, capsys, protected):
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps(config))
    target = Path(config[protected]) / 'must-not-write.json'
    calls = fake_run(monkeypatch, AssertionError('must not start'))
    assert neural_probe_check.main(['--config', str(profile), '--report', str(target),
        '--slot', 'radiographic', '--fold', '0']) == 1
    assert not target.exists() and not calls
    assert config[protected] not in capsys.readouterr().out


def test_cli_preserves_an_existing_report(config, monkeypatch, tmp_path, capsys):
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps(config))
    report = tmp_path / 'report.json'
    report.write_text('existing evidence')
    calls = fake_run(monkeypatch, AssertionError('must not start'))
    assert neural_probe_check.main(['--config', str(profile), '--report', str(report),
        '--slot', 'radiographic', '--fold', '0']) == 1
    assert report.read_text() == 'existing evidence' and not calls
    assert str(report) not in capsys.readouterr().out


def test_cli_records_expected_missing_asset_without_claiming_load(config, monkeypatch, tmp_path, capsys):
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps(config))
    report = tmp_path / 'missing.json'
    fake_run(monkeypatch, response(raw='{"ok":false,"error":{"code":"ASSET_MISSING"}}', returncode=1))
    assert neural_probe_check.main(['--config', str(profile), '--report', str(report),
        '--slot', 'radiographic', '--fold', '0']) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary['code'] == 'ASSET_MISSING' and summary['weightsVerified'] is False
    assert json.loads(report.read_text())['status'] == 'error'
