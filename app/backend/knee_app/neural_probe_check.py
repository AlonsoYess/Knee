"""Run one private neural development-model check; never a clinical prediction."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .predictor import _json
from .neural_probe import CONFIG_FIELDS, NeuralProbe, NeuralProbeError, plain


FRONTEND = Path(__file__).resolve().parents[2] / 'frontend'


def read_config(path):
    target = plain(path)
    if not target.is_file():
        raise NeuralProbeError('NEURAL_PROBE_BUNDLE_INVALID')
    with target.open('rb') as stream:
        raw = stream.read(8193)
    if len(raw) > 8192:
        raise NeuralProbeError('NEURAL_PROBE_BUNDLE_INVALID')
    value = _json(raw)
    if not isinstance(value, dict) or set(value) != CONFIG_FIELDS:
        raise NeuralProbeError('NEURAL_PROBE_BUNDLE_INVALID')
    return target, value


def report_target(path, config_path, probe):
    original = Path(path)
    if not original.is_absolute() or original.exists():
        raise NeuralProbeError('NEURAL_PROBE_SAFETY_VIOLATION')
    parent = plain(original.parent)
    if not parent.is_dir():
        raise NeuralProbeError('NEURAL_PROBE_SAFETY_VIOLATION')
    # Inspect the destination itself too, including a dangling link.
    if original.is_symlink() or getattr(original, 'is_junction', lambda: False)():
        raise NeuralProbeError('NEURAL_PROBE_SAFETY_VIOLATION')
    target = parent / original.name
    protected = (probe.checkout, probe.bundle, probe.scratch_root, probe.runtime_root, FRONTEND.resolve())
    if target == config_path or any(target == root or target.is_relative_to(root) for root in protected):
        raise NeuralProbeError('NEURAL_PROBE_SAFETY_VIOLATION')
    return target


def write_report(path, report):
    raw = (json.dumps(report, ensure_ascii=True, allow_nan=False, indent=2) + '\n').encode('utf-8')
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--slot', required=True, choices=('radiographic', 'multimodal'))
    parser.add_argument('--fold', required=True, type=int, choices=range(5))
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--timeout', type=float, default=300)
    args = parser.parse_args(argv)
    operation = 'diagnostic' if args.diagnostic else 'preflight'
    report = {'status': 'error', 'operation': operation, 'slot': args.slot, 'fold': args.fold,
              'clinicalReleaseReady': False, 'clinicalResultsEnabled': False}
    success, target = False, None
    try:
        config_path, config = read_config(args.config)
        probe = NeuralProbe(config, timeout=args.timeout)
        target = report_target(args.report, config_path, probe)
        data = getattr(probe, operation)(args.slot, args.fold)
        report.update(status='technical_neural_connection_verified', data=data)
        success = True
    except NeuralProbeError as error:
        report['code'] = error.code
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, OverflowError, RecursionError):
        report['code'] = 'NEURAL_PROBE_CHECK_FAILED'
    if target is not None:
        try:
            write_report(target, report)
        except (OSError, ValueError, TypeError, UnicodeError):
            success = False
            report['code'] = 'NEURAL_PROBE_CHECK_FAILED'
    summary = {'operation': operation, 'slot': args.slot, 'fold': args.fold, 'success': success,
               'weightsVerified': success, 'clinicalReleaseReady': False, 'clinicalResultsEnabled': False}
    if not success:
        summary['code'] = report.get('code', 'NEURAL_PROBE_CHECK_FAILED')
    print(json.dumps(summary, ensure_ascii=True, allow_nan=False))
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
