"""Copy authenticated development-model artifacts for a local technical check.

This standard-library tool never imports scientific modules, starts Python,
selects a model by performance, fits statistics, or enables clinical results.
The artifact index and the expected raw SHA256 are supplied by the operator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat


MAX_JSON = 1024 * 1024
MAX_RECEIPT = 2 * 1024 * 1024
MAX_SOURCE = 4 * 1024 * 1024
SOURCES = (
    'src/knee/__init__.py',
    'src/knee/aligned_training/__init__.py',
    'src/knee/aligned_training/authority.py',
    'src/knee/aligned_training/runner.py',
    'src/knee/clinical_fold_inputs.py',
    'src/knee/clinical_scenarios.py',
    'src/knee/colab_data.py',
    'src/knee/methodology_release.py',
    'src/knee/sensitivity_evaluation_metrics.py',
    'src/knee/sensitivity_oof_scores.py',
    'src/knee/sensitivity_real_inference.py',
    'src/knee/training_plan.py',
)
FRONTEND = Path(__file__).resolve().parents[2] / 'frontend'


class BundleError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def require(condition, code):
    if not condition:
        raise BundleError(code)


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def plain(value, *, directory=False, new=False):
    require(isinstance(value, (str, os.PathLike)), 'PATH_INVALID')
    original = Path(value)
    require(original.is_absolute(), 'PATH_INVALID')
    require(all(not item.is_symlink() and not getattr(item, 'is_junction', lambda: False)()
                for item in (original, *original.parents)), 'PATH_INVALID')
    if new:
        require(not original.exists(), 'OUTPUT_EXISTS')
        parent = original.parent.resolve(strict=True)
        require(parent.is_dir(), 'PATH_INVALID')
        return parent / original.name
    path = original.resolve(strict=True)
    mode = path.stat().st_mode
    require(stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode), 'PATH_INVALID')
    return path


def read_file(path, maximum, *, expected=None, code='ARTIFACT_INVALID'):
    path = plain(path)
    require(0 < path.stat().st_size <= maximum, code)
    with path.open('rb') as stream:
        raw = stream.read(maximum + 1)
    require(0 < len(raw) <= maximum, code)
    if expected is not None:
        require(valid_sha(expected) and hashlib.sha256(raw).hexdigest() == expected, code)
    return raw


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'JSON_INVALID')
            result[key] = value
        return result
    def constant(_):
        raise BundleError('JSON_INVALID')
    return json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=constant)


def canonical_sha(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                     allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def write_new(path, raw):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(raw)


def write_json(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
           + '\n').encode('utf-8')
    write_new(path, raw)
    return hashlib.sha256(raw).hexdigest()


def prepare(args):
    require(valid_sha(args.artifact_index_sha256), 'INDEX_HASH_INVALID')
    checkout = plain(args.checkout, directory=True)
    python = plain(args.python)
    index_path = plain(args.artifact_index)
    output = plain(args.output, new=True)
    index = strict_json(read_file(index_path, MAX_RECEIPT,
                                 expected=args.artifact_index_sha256, code='INDEX_HASH_INVALID'))
    require(isinstance(index, dict) and index.get('version') == 'selected_fold_artifact_references_v1'
            and valid_sha(index.get('selection_record_sha256'))
            and type(index.get('artifacts')) is list, 'INDEX_INVALID')
    selected = [item for item in index['artifacts'] if isinstance(item, dict)
                and item.get('selection_group') == 'clinical_main'
                and item.get('model_id') == 'clinical_logistic']
    require(len(selected) == 5 and all(type(item.get('validation_fold')) is int for item in selected),
            'FIVE_REGISTERED_FOLDS_REQUIRED')
    selected.sort(key=lambda item: item['validation_fold'])
    require([item['validation_fold'] for item in selected] == list(range(5)),
            'FIVE_REGISTERED_FOLDS_REQUIRED')

    records = []
    originals = []
    protected = [checkout, FRONTEND.resolve(), index_path.parent]
    for item in selected:
        fold = item['validation_fold']
        require(item.get('stage') == 'stage_1' and type(item.get('seed')) is int
                and item['seed'] == 2026, 'REGISTERED_MODEL_REQUIRED')
        weight = plain(item['weights_local_path'])
        state = plain(item['clinical_state_path'])
        receipt_path = plain(item['source_receipt'])
        protected.extend((weight.parent, state.parent, receipt_path.parent))
        weight_raw = read_file(weight, MAX_JSON, expected=item['weights_expected_sha256'])
        state_raw = read_file(state, MAX_JSON, expected=item['clinical_state_sha256'])
        # Validate JSON without interpreting or fitting the scientific parameters.
        require(isinstance(strict_json(weight_raw), dict) and isinstance(strict_json(state_raw), dict),
                'ARTIFACT_INVALID')
        receipt_raw = read_file(receipt_path, MAX_RECEIPT, code='RECEIPT_INVALID')
        receipt = strict_json(receipt_raw)
        require(isinstance(receipt, dict)
                and valid_sha(item.get('source_receipt_canonical_sha256'))
                and receipt.get('record_sha256') == item['source_receipt_canonical_sha256']
                and canonical_sha({key: value for key, value in receipt.items() if key != 'record_sha256'})
                    == receipt['record_sha256'], 'RECEIPT_INVALID')
        request = receipt.get('request')
        files = receipt.get('files_sha256')
        require(receipt.get('scope') == 'development_only'
                and receipt.get('reserved_test_accessed') is False
                and receipt.get('training_executed') is True
                and receipt.get('eligible_for_model_selection') is True
                and receipt.get('engine_status') == 'training_fold_complete'
                and isinstance(request, dict) and isinstance(files, dict), 'RECEIPT_INVALID')
        require(request.get('model_id') == 'clinical_logistic'
                and request.get('clinical_scenario') == 'clinical_main'
                and type(request.get('validation_fold')) is int and request['validation_fold'] == fold
                and request.get('stage') == 'stage_1'
                and type(request.get('seed')) is int and request['seed'] == 2026
                and request.get('operation') == 'development_fold'
                and files.get('clinical_model.json') == item['weights_expected_sha256']
                and files.get('clinical_state.json') == item['clinical_state_sha256'], 'RECEIPT_INVALID')
        records.append((fold, weight_raw, state_raw))
        for role, path, raw in (('weights', weight, weight_raw), ('clinical_state', state, state_raw),
                                ('receipt', receipt_path, receipt_raw)):
            originals.append({'fold': fold, 'role': role, 'path': str(path),
                              'sha256': hashlib.sha256(raw).hexdigest()})
    require(all(output != root and not output.is_relative_to(root) for root in protected),
            'OUTPUT_OVERLAPS_SOURCES')
    require(not output.is_relative_to(python.parent), 'OUTPUT_OVERLAPS_SOURCES')
    source_pins = {}
    for relative in SOURCES:
        path = plain(checkout / relative)
        require(path.is_relative_to(checkout), 'SOURCE_INVALID')
        source_pins[relative] = hashlib.sha256(read_file(path, MAX_SOURCE, code='SOURCE_INVALID')).hexdigest()

    # All source validation happens before the first write. Every target is new.
    output.mkdir(mode=0o700)
    bundle = output / 'bundle'
    bundle.mkdir(mode=0o700)
    models = []
    for fold, weight_raw, state_raw in records:
        target = bundle / 'models' / f'fold_{fold}'
        target.mkdir(parents=True, mode=0o700)
        record = {'fold': fold}
        for role, filename, raw in (('weights', 'clinical_model.json', weight_raw),
                                     ('clinical_state', 'clinical_state.json', state_raw)):
            destination = target / filename
            write_new(destination, raw)
            record[role] = {'path': destination.relative_to(bundle).as_posix(),
                            'sha256': hashlib.sha256(raw).hexdigest()}
        models.append(record)
    manifest = {
        'format': 'kneeoai-development-model-probe-v1',
        'purpose': 'technical_development_check_only', 'clinical_release_ready': False,
        'python_version': '3.12.14', 'numpy_version': '2.5.3',
        'selection_record_sha256': index['selection_record_sha256'],
        'scientific_sources': source_pins, 'models': models,
    }
    manifest_sha = write_json(bundle / 'manifest.json', manifest)
    write_json(output / 'config_private.json', {
        'python': str(python), 'checkout': str(checkout), 'bundle': str(bundle),
        'manifestSha256': manifest_sha,
    })
    write_json(output / 'original_artifacts_private.json', {
        'artifactIndex': str(index_path), 'artifactIndexSha256': args.artifact_index_sha256,
        'selectionRecordSha256': index['selection_record_sha256'], 'artifacts': originals,
    })
    # Detect concurrent modification; never repair or overwrite an original.
    read_file(index_path, MAX_RECEIPT, expected=args.artifact_index_sha256, code='SOURCE_CHANGED')
    for item in originals:
        read_file(Path(item['path']), MAX_RECEIPT, expected=item['sha256'], code='SOURCE_CHANGED')
    for relative, checksum in source_pins.items():
        read_file(checkout / relative, MAX_SOURCE, expected=checksum, code='SOURCE_CHANGED')
    return {'success': True, 'modelsCopied': 5, 'sourceModulesPinned': len(source_pins),
            'clinicalReleaseReady': False, 'trainingExecuted': False, 'cohortOpened': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-index', required=True, type=Path)
    parser.add_argument('--artifact-index-sha256', required=True)
    parser.add_argument('--checkout', required=True, type=Path)
    parser.add_argument('--python', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = prepare(args)
    except BundleError as error:
        result = {'success': False, 'code': error.code, 'clinicalReleaseReady': False}
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, RecursionError):
        result = {'success': False, 'code': 'BUNDLE_PREPARATION_FAILED', 'clinicalReleaseReady': False}
    print(json.dumps(result, allow_nan=False, ensure_ascii=True))
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
