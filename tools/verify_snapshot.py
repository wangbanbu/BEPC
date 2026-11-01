"""Verify the uploaded source snapshot using only the Python standard library."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'PUBLICATION_MANIFEST.json').read_text())
    checked = 0
    for name, entry in manifest['source_files'].items():
        relative = manifest['readme_original_copy'] if name == 'README.md' else name
        path = ROOT / relative
        payload = path.read_bytes()
        assert len(payload) == entry['size_bytes'], f'Size mismatch: {relative}'
        assert hashlib.sha256(payload).hexdigest() == entry['sha256'], f'SHA mismatch: {relative}'
        assert not path.is_symlink(), f'Absolute link was not materialized: {relative}'
        checked += 1
    python_files = [p for directory in ('harness', 'tests', 'long_output', 'tools')
                    for p in (ROOT / directory).rglob('*.py')]
    for path in python_files:
        ast.parse(path.read_bytes(), filename=str(path))
    for directory, key in [('upstream/ViSpec', 'upstream'), ('reference/LVSpec', 'reference')]:
        path = ROOT / directory
        assert (path / '.git').exists(), f'Initialize submodules: {directory}'
        commit = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        assert commit == manifest[key]['commit'], f'Wrong upstream commit: {directory}'
        status = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True).strip()
        assert not status, f'Upstream checkout changed: {directory}'
    print(json.dumps(dict(status='pass', source_files=checked,
                          python_files=len(python_files), pinned_submodules=2)))


if __name__ == '__main__':
    main()
