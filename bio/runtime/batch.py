#!/usr/bin/env python3
"""Run the Cartesian product of a local benchmark parameter file."""
import argparse
import itertools
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

from bio.io.paths import PROJECT_ROOT as ROOT


def commands(config, python):
    parameters = config['parameters']
    choices = []
    for name, spec in parameters.items():
        values = spec.get('values', [spec['value']] if 'value' in spec else None)
        if not isinstance(values, list) or not values:
            raise ValueError(f'{name} must specify a nonempty values list or value')
        choices.append(values)
    for values in itertools.product(*choices):
        command = [python, str(ROOT / config.get('program', 'main.py'))]
        for name, value in zip(parameters, values):
            command.extend(['--' + name, str(value)])
        command.extend(config.get('extra_args', []))
        yield command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--limit', type=int, help='Run/print only the first N combinations')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    config = yaml.safe_load(args.config.read_text())
    failed = 0
    selected = itertools.islice(commands(config, args.python), args.limit)
    for command in selected:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            failed += subprocess.run(command, cwd=ROOT).returncode != 0
    return int(failed > 0)


if __name__ == '__main__':
    raise SystemExit(main())
