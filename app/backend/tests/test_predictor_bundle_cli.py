"""Bundle-preparation checks using invented JSON, receipts and source markers.

No scientific module, real model, cohort or reserved partition is used here.
"""
from argparse import Namespace
import builtins
import hashlib
import json
from pathlib import Path

import pytest

from knee_app import predictor_bundle as tool


def raw_json(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode('utf-8')


def checksum(raw):
    return hashlib.sha256(raw).hexdigest()


def sealed_receipt(value):
    body = {key: item for key, item in value.items() if key != 'record_sha256'}
    encoded = json.dumps(body, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode('utf-8')
    return {**body, 'record_sha256': checksum(encoded)}


@pytest.fixture
def invented(tmp_path):
    checkout = tmp_path / 'fictional-checkout'
    checkout.mkdir()
    for relative in tool.SOURCES:
        path = checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'# Invented source marker: this file is never imported.\n')
    python = tmp_path / 'fictional-environment' / 'python.exe'
    python.parent.mkdir()
    python.write_bytes(b'Invented executable marker: never executed.\n')
    index_path = tmp_path / 'private-index' / 'index.json'
    index_path.parent.mkdir()
    artifacts = []
    for fold in range(5):
        directory = tmp_path / 'invented-saved-artifacts' / f'fold_{fold}'
        directory.mkdir(parents=True)
        weight = directory / 'clinical_model.json'
        state = directory / 'clinical_state.json'
        receipt_path = directory / 'source_training_job.json'
        weight.write_bytes(raw_json({'invented_coefficients': [0.1, 0.2, 0.3], 'fold': fold}))
        state.write_bytes(raw_json({'invented_state': True, 'fold': fold}))
        receipt = sealed_receipt({
            'scope': 'development_only', 'reserved_test_accessed': False,
            'training_executed': True, 'eligible_for_model_selection': True,
            'engine_status': 'training_fold_complete',
            'request': {'model_id': 'clinical_logistic', 'clinical_scenario': 'clinical_main',
                        'validation_fold': fold, 'stage': 'stage_1', 'seed': 2026,
                        'operation': 'development_fold'},
            'files_sha256': {'clinical_model.json': checksum(weight.read_bytes()),
                             'clinical_state.json': checksum(state.read_bytes())},
        })
        receipt_path.write_bytes(raw_json(receipt))
        artifacts.append({
            'selection_group': 'clinical_main', 'model_id': 'clinical_logistic',
            'validation_fold': fold, 'stage': 'stage_1', 'seed': 2026,
            'weights_local_path': str(weight), 'clinical_state_path': str(state),
            'source_receipt': str(receipt_path),
            'weights_expected_sha256': checksum(weight.read_bytes()),
            'clinical_state_sha256': checksum(state.read_bytes()),
            # This field deliberately uses the receipt's seal, not a hash of
            # the full JSON object that contains its own record_sha256 field.
            'source_receipt_canonical_sha256': receipt['record_sha256'],
        })
    index = {'version': 'selected_fold_artifact_references_v1',
             'selection_record_sha256': 'a' * 64, 'artifacts': artifacts}
    args = Namespace(artifact_index=index_path, artifact_index_sha256=None,
                     checkout=checkout, python=python, output=tmp_path / 'new-private-output')

    def refresh_index():
        index_path.write_bytes(raw_json(index))
        args.artifact_index_sha256 = checksum(index_path.read_bytes())

    refresh_index()
    return Namespace(args=args, index=index, refresh=refresh_index)


def test_bundle_copies_only_authenticated_artifacts_without_importing_science(invented, monkeypatch):
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name in {'knee', 'numpy', 'torch', 'sklearn'} or name.startswith('knee.'):
            raise AssertionError('The bundle builder must remain standard-library only')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    originals = [Path(item[key]) for item in invented.index['artifacts']
                 for key in ('weights_local_path', 'clinical_state_path', 'source_receipt')]
    before = {path: path.read_bytes() for path in originals}
    result = tool.prepare(invented.args)
    assert result == {'success': True, 'modelsCopied': 5, 'sourceModulesPinned': 12,
                      'clinicalReleaseReady': False, 'trainingExecuted': False, 'cohortOpened': False}
    output = invented.args.output
    bundle = output / 'bundle'
    files = {path.relative_to(bundle).as_posix() for path in bundle.rglob('*') if path.is_file()}
    assert files == {'manifest.json'} | {
        f'models/fold_{fold}/{name}' for fold in range(5)
        for name in ('clinical_model.json', 'clinical_state.json')
    }
    manifest = json.loads((bundle / 'manifest.json').read_bytes())
    assert manifest['clinical_release_ready'] is False
    assert manifest['purpose'] == 'technical_development_check_only'
    assert set(manifest['scientific_sources']) == set(tool.SOURCES)
    for item in manifest['models']:
        original = invented.index['artifacts'][item['fold']]
        for role, source_key in (('weights', 'weights_local_path'), ('clinical_state', 'clinical_state_path')):
            copied = bundle / item[role]['path']
            assert copied.read_bytes() == Path(original[source_key]).read_bytes()
            assert checksum(copied.read_bytes()) == item[role]['sha256']
    config = json.loads((output / 'config_private.json').read_bytes())
    assert set(config) == {'python', 'checkout', 'bundle', 'manifestSha256'}
    assert config['manifestSha256'] == checksum((bundle / 'manifest.json').read_bytes())
    assert config['bundle'] == str(bundle)
    assert all(path.read_bytes() == raw for path, raw in before.items())


@pytest.mark.parametrize('kind', ['weight', 'state', 'receipt'])
def test_changed_or_ineligible_artifact_is_rejected_before_output(invented, kind):
    item = invented.index['artifacts'][2]
    if kind == 'receipt':
        path = Path(item['source_receipt'])
        receipt = json.loads(path.read_bytes())
        receipt['reserved_test_accessed'] = True
        receipt = sealed_receipt(receipt)
        path.write_bytes(raw_json(receipt))
        item['source_receipt_canonical_sha256'] = receipt['record_sha256']
        invented.refresh()
        expected = 'RECEIPT_INVALID'
    else:
        path = Path(item['weights_local_path' if kind == 'weight' else 'clinical_state_path'])
        path.write_bytes(raw_json({'changed_after_registration': True}))
        expected = 'ARTIFACT_INVALID'
    with pytest.raises(tool.BundleError, match=expected):
        tool.prepare(invented.args)
    assert not invented.args.output.exists()


def test_wrong_raw_index_hash_is_rejected_before_output(invented):
    invented.args.artifact_index_sha256 = '0' * 64
    with pytest.raises(tool.BundleError, match='INDEX_HASH_INVALID'):
        tool.prepare(invented.args)
    assert not invented.args.output.exists()


@pytest.mark.parametrize('change', ['duplicate', 'missing'])
def test_exactly_five_distinct_registered_folds_are_required(invented, change):
    if change == 'duplicate':
        invented.index['artifacts'][-1]['validation_fold'] = 3
    else:
        invented.index['artifacts'].pop()
    invented.refresh()
    with pytest.raises(tool.BundleError, match='FIVE_REGISTERED_FOLDS_REQUIRED'):
        tool.prepare(invented.args)
    assert not invented.args.output.exists()


def test_existing_destination_is_never_overwritten(invented):
    invented.args.output.mkdir()
    sentinel = invented.args.output / 'existing.json'
    sentinel.write_bytes(b'Preserve this existing file unchanged.\n')
    with pytest.raises(tool.BundleError, match='OUTPUT_EXISTS'):
        tool.prepare(invented.args)
    assert list(invented.args.output.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b'Preserve this existing file unchanged.\n'


def test_destination_inside_checkout_is_rejected_before_writing(invented):
    invented.args.output = invented.args.checkout / 'new-output-must-not-be-created'
    with pytest.raises(tool.BundleError, match='OUTPUT_OVERLAPS_SOURCES'):
        tool.prepare(invented.args)
    assert not invented.args.output.exists()


def test_cli_error_summary_contains_no_private_paths(invented, capsys):
    args = invented.args
    result = tool.main([
        '--artifact-index', str(args.artifact_index), '--artifact-index-sha256', '0' * 64,
        '--checkout', str(args.checkout), '--python', str(args.python), '--output', str(args.output),
    ])
    captured = capsys.readouterr()
    assert result == 1 and not captured.err
    assert json.loads(captured.out) == {
        'success': False, 'code': 'INDEX_HASH_INVALID', 'clinicalReleaseReady': False,
    }
    assert str(args.artifact_index) not in captured.out
    assert str(args.checkout) not in captured.out
    assert not args.output.exists()
