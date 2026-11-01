"""Run each existing CPU test module in its own Python process.

The validators explicitly assert that they never import torch, so modules that
exercise tensor code must run separately from those validator contracts.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SUITES = [ROOT, ROOT / 'long_output', ROOT / 'long_output/single_pass',
          *sorted((ROOT / 'long_output').glob('bepc_*'))]


def main():
    failures = []
    for suite in SUITES:
        if not (suite / 'tests').is_dir():
            continue
        label = str(suite.relative_to(ROOT))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        paths = [str(suite / 'harness'), str(ROOT / 'harness'),
                 str(ROOT / 'long_output/harness'), str(ROOT / 'long_output/tests')]
        if env.get('PYTHONPATH'):
            paths.append(env['PYTHONPATH'])
        env['PYTHONPATH'] = os.pathsep.join(paths)
        for test_file in sorted((suite / 'tests').glob('test*.py')):
            module_label = f'{label}/{test_file.name}'
            print(f'Running {module_label}', flush=True)
            result = subprocess.run([sys.executable, '-m', 'unittest', 'discover',
                                     '-s', 'tests', '-p', test_file.name],
                                    cwd=suite, env=env)
            if result.returncode:
                failures.append(module_label)
    if failures:
        print(f'Failed suites: {", ".join(failures)}', file=sys.stderr)
        return 1
    print('All CPU test suites passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
