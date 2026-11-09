"""Check the local development-model bridge without enabling clinical results.

Only the worker can load scientific modules. The optional diagnostic uses its
fixed fictional inputs; this CLI never accepts a case or writes an application
database. Detailed diagnostics stay in a new private report chosen by the user.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

from .config import Settings
from .predictor import FrozenPredictor, PredictorError, _json, _plain


CONFIG_FIELDS = {'python', 'checkout', 'bundle', 'manifestSha256'}
FRONTEND = Path(__file__).resolve().parents[2] / 'frontend'


class CheckError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def read_config(path):
    original = Path(path)
    if not original.is_absolute():
        raise CheckError('CONFIG_INVALID')
    target = _plain(original)
    if not target.is_file():
        raise CheckError('CONFIG_INVALID')
    with target.open('rb') as stream:
        raw = stream.read(8193)
    if len(raw) > 8192:
        raise CheckError('CONFIG_INVALID')
    value = _json(raw)
    if not isinstance(value, dict) or set(value) != CONFIG_FIELDS:
        raise CheckError('CONFIG_INVALID')
    if not isinstance(value['manifestSha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', value['manifestSha256']):
        raise CheckError('CONFIG_INVALID')
    for key in ('python', 'checkout', 'bundle'):
        if not isinstance(value[key], str) or not value[key] or not Path(value[key]).is_absolute():
            raise CheckError('CONFIG_INVALID')
        value[key] = _plain(value[key])
    return target, value


def report_target(path, config_path, config):
    original = Path(path)
    if not original.is_absolute():
        raise CheckError('REPORT_INVALID')
    target = _plain(original)
    if not target.parent.is_dir() or target.exists() or target == config_path:
        raise CheckError('REPORT_INVALID')
    protected = (config['checkout'], config['bundle'], FRONTEND.resolve())
    if any(target == root or target.is_relative_to(root) for root in protected):
        raise CheckError('REPORT_INVALID')
    return target


def write_report(path, report):
    raw = (json.dumps(report, allow_nan=False, ensure_ascii=True, indent=2) + '\n').encode('utf-8')
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path, help='Private, explicit local bridge configuration')
    parser.add_argument('--report', required=True, type=Path, help='New private JSON report outside checkout and bundle')
    parser.add_argument('--diagnostic', action='store_true', help='Use only the worker\'s fixed fictional diagnostic inputs')
    args = parser.parse_args(argv)
    operation = 'diagnostic' if args.diagnostic else 'preflight'
    report = {
        'status': 'error', 'operation': operation,
        'clinicalReleaseReady': False, 'clinicalResultsEnabled': False,
    }
    target = None
    success = False
    models = 0
    try:
        config_path, config = read_config(args.config)
        target = report_target(args.report, config_path, config)
        settings = Settings(
            predictor_python=config['python'], predictor_checkout=config['checkout'],
            predictor_bundle=config['bundle'], predictor_manifest_sha256=config['manifestSha256'],
        )
        predictor = FrozenPredictor(settings)
        data = predictor.diagnostic() if args.diagnostic else predictor.preflight()
        report.update(status='technical_connection_verified', data=data)
        models = data['modelsVerified']
        success = True
    except CheckError as error:
        report['code'] = error.code
    except PredictorError:
        report['code'] = 'PREDICTOR_UNAVAILABLE'
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, RecursionError, OverflowError):
        report['code'] = 'LOCAL_CHECK_FAILED'
    if target is not None:
        try:
            write_report(target, report)
        except (OSError, ValueError, TypeError, UnicodeError):
            success = False
            models = 0
            report['code'] = 'REPORT_WRITE_FAILED'
    summary = {
        'operation': operation, 'success': success,
        'modelsVerified': models, 'clinicalReleaseReady': False,
        'clinicalResultsEnabled': False,
    }
    if not success:
        summary['code'] = report.get('code', 'LOCAL_CHECK_FAILED')
    print(json.dumps(summary, allow_nan=False, ensure_ascii=True))
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
