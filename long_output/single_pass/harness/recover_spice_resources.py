"""Recover one failed official SPICE CPU score by increasing JVM heap only.

Default is a read-only plan. ``--execute`` archives the entire incomplete
LVSpec-paper metric attempt, supplies the exact same input bytes to the exact
same official Java scorer with -Xmx64G, and seeds the existing evaluator's cache
only after strict output/log validation. No GPU inference or frozen edit occurs.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
METHOD = ROOT/'metrics/main/coco/lvspec_paper'
AUDIT = ROOT/'audit/SPICE_RESOURCE_RECOVERY.json'
OFFICIAL_SOURCE = ROOT.parents[1]/'harness/evaluate_tasks.py'
OFFICIAL_SHA256 = '1f491b0f1dba0a148b5a65ddc574c73c76878c098c967fc1f45fc08ba2a83fc3'
BAD_LOG = re.compile(r'OutOfMemoryError|out\s+of\s+memory|\bOOM\b|\bignore(?:d)?\b|'
                     r'Could\s+not\s+score\s+batched|NullPointerException|'
                     r'Exception\s+in\s+thread|Error:\s+Could\s+not\b(?!\s+cache\s+item\b)', re.IGNORECASE)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(payload, encoding='utf-8')
    os.replace(temporary, path)


def inventory(folder):
    return {str(p.relative_to(folder)): dict(bytes=p.stat().st_size, sha256=sha(p))
            for p in sorted(Path(folder).rglob('*')) if p.is_file()}


def protected_inventory():
    """Seal GPU evidence, configuration/runtime and every already successful score."""
    files = set((ROOT/'results/main').glob('*.jsonl'))
    files.update((ROOT/'results/main').glob('*.complete.json'))
    files.update((ROOT/'results/development').glob('*.jsonl'))
    files.update((ROOT/'configs').glob('*.json'))
    files.update((ROOT/'harness').glob('*.py'))
    for phase in ('main', 'development'):
        for seal in (ROOT/f'metrics/{phase}').glob('*/*/SCORING_SEAL.json'):
            files.add(seal)
            files.update(p for p in seal.parent.rglob('*') if p.is_file())
    return {str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)}


def commands(folder, package, models):
    common = ['java', '-XX:ActiveProcessorCount=8', '-Xmx16G', '-cp',
        os.pathsep.join([str(models/'*'), str(package/'spice-1.0.jar'), str(package/'lib/*')]),
        'edu.anu.spice.SpiceScorer', str(folder/'input.json'), '-cache', str(folder/'cache'),
        '-out', str(folder/'output.json'), '-subset', '-silent', '-threads', '1']
    recovered = list(common)
    recovered[2] = '-Xmx64G'
    assert [(a,b) for a,b in zip(common,recovered) if a != b] == [('-Xmx16G','-Xmx64G')]
    return common, recovered


def input_ids(payload):
    assert isinstance(payload, list) and len(payload) == 100, 'Expected all 100 frozen captions'
    ids = [int(row['image_id']) for row in payload]
    assert len(set(ids)) == len(ids), 'Duplicate input IDs'
    assert all(isinstance(row['test'], str) and isinstance(row['refs'], list)
               and all(isinstance(v,str) for v in row['refs']) for row in payload)
    return set(ids)


def validate_result(output, expected, log):
    hits = [m.group(0) for m in BAD_LOG.finditer(log)]
    assert not hits, f'Fatal/skipped-parse indicators in recovery log: {hits}'
    assert isinstance(output, list) and len(output) == len(expected), 'Incomplete score batch'
    ids = [int(row['image_id']) for row in output]
    assert len(set(ids)) == len(ids) and set(ids) == expected, 'Output IDs missing/duplicated'
    for row in output:
        assert 'f' in row['scores']['All'], 'Official All.f missing'
    # Preserve the original scorer's undefined-value convention; do not fill
    # undefined facts with zero or modify any metric value here.
    return dict(status='pass', input_ids=len(expected), output_ids=len(ids),
                fatal_or_ignored_parse_log_matches=hits, score_values_unmodified=True)


def dependencies():
    assert sha(OFFICIAL_SOURCE) == OFFICIAL_SHA256, 'Frozen official wrapper changed'
    spec = importlib.util.spec_from_file_location('_spice_resource_recovery_official', OFFICIAL_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.COCO_DEPS/'pycocoevalcap/spice', module.SPICE_MODELS


def plan():
    assert METHOD.exists(), 'Original incomplete method directory unavailable'
    assert not (METHOD/'SCORING_SEAL.json').exists(), 'Refuse to replace successful scoring'
    folder = METHOD/'official_scoring/spice'
    assert (folder/'input.json').is_file() and (folder/'scorer.log').is_file()
    assert not (folder/'output.json').exists(), 'Unexpected existing scorer output; review first'
    assert not (folder/'input.sha256').exists(), 'Unexpected success signature; review first'
    payload = read(folder/'input.json')
    expected = input_ids(payload)
    original_log = (folder/'scorer.log').read_text(encoding='utf-8', errors='replace')
    assert BAD_LOG.search(original_log), 'Original attempt has no documented parse/resource failure'
    package, models = dependencies()
    original, recovered = commands(folder, package, models)
    return dict(status='planned_not_executed', method_path=str(METHOD), input_sha256=sha(folder/'input.json'),
        input_ids=len(expected), original_failure_inventory=inventory(METHOD),
        original_command=original, recovery_command=recovered,
        command_difference=[dict(index=2, original='-Xmx16G', recovery='-Xmx64G')],
        resource_only_change='JVM maximum heap 16GiB -> 64GiB; fresh writable parser cache to discard failed partial state',
        official_wrapper=dict(path=str(OFFICIAL_SOURCE), sha256=sha(OFFICIAL_SOURCE)),
        recovery_source=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
        scorer_dependencies={str(p): sha(p) for p in [package/'spice-1.0.jar', package/'spice.py',
            models/'stanford-corenlp-3.6.0-models.jar']},
        scorer_library_sha256={str(p): sha(p) for p in sorted(
            set((package/'lib').glob('*.jar')) | set(models.glob('*.jar')) | {package/'spice-1.0.jar'})},
        java_executable=shutil.which('java'), no_gpu_inference=True,
        caption_references_ptb_and_metric_rules_changed=False,
        future_frozen_wrapper_provenance_caveat='The frozen wrapper records its planned -Xmx16G command '
            'even when exact_same_input_result_reused=True. This audit identifies the actual -Xmx64G '
            'producer and output SHA; no execution of that planned 16G command is claimed for the cache hit.')


def execute():
    assert not AUDIT.exists(), 'Recovery audit already exists; preserve and review its attempt'
    proposal = plan()
    lock_path = ROOT/'metrics/.spice.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a', encoding='utf-8') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # Preserve every failure artifact, including outer metric_inputs and
        # metric_failure seals, not just the parser cache/log.
        assert inventory(METHOD) == proposal['original_failure_inventory']
        archive = ROOT/'audit/spice_recovery/failed_attempt_16g'
        archive.mkdir(parents=True, exist_ok=False)
        old = archive/'lvspec_paper'
        other_methods = {p.name: inventory(p) for p in sorted(METHOD.parent.iterdir())
                         if p.is_dir() and p != METHOD}
        protected = protected_inventory()
        main_files = sorted((ROOT/'results/main').glob('*.jsonl'))
        assert len(main_files) == 10
        main_records = sum(len(p.read_text(encoding='utf-8').splitlines()) for p in main_files)
        assert main_records == 4200, 'All 4,200 frozen GPU records must exist before CPU recovery'
        shutil.move(str(METHOD), str(old))
        assert inventory(old) == proposal['original_failure_inventory']
        folder = METHOD/'official_scoring/spice'
        folder.mkdir(parents=True, exist_ok=False)
        (folder/'cache').mkdir()
        shutil.copyfile(old/'official_scoring/spice/input.json', folder/'input.json')
        assert sha(folder/'input.json') == proposal['input_sha256']
        state = dict(proposal, status='running', preserved_failure_directory=str(old),
            preserved_failure_inventory_sha256=None, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            fresh_cache_initially_empty=not any((folder/'cache').iterdir()),
            fresh_cache_path=str(folder/'cache'), failed_attempt_had_no_output=True,
            other_method_directories_and_sha256_before=other_methods,
            protected_evidence_before=protected, main_gpu_records=main_records)
        write(archive/'FAILURE_INVENTORY.json', proposal['original_failure_inventory'])
        state['preserved_failure_inventory_sha256'] = sha(archive/'FAILURE_INVENTORY.json')
        write(AUDIT, state)
        started = time.perf_counter()
        try:
            with (folder/'scorer.log').open('w', encoding='utf-8') as log:
                result = subprocess.run(proposal['recovery_command'], cwd=folder,
                    stdout=log, stderr=subprocess.STDOUT, check=False)
            state['java_exit_code'] = result.returncode
            state['elapsed_seconds'] = time.perf_counter()-started
            assert result.returncode == 0, f'Official scorer exited {result.returncode}'
            assert sha(folder/'input.json') == proposal['input_sha256'], 'Scorer input changed'
            validation = validate_result(read(folder/'output.json'), input_ids(read(folder/'input.json')),
                (folder/'scorer.log').read_text(encoding='utf-8', errors='replace'))
            assert inventory(old) == proposal['original_failure_inventory'], 'Archived failure evidence changed'
            after = {p.name: inventory(p) for p in sorted(METHOD.parent.iterdir())
                     if p.is_dir() and p != METHOD}
            assert after == other_methods, 'Another method changed during isolated recovery'
            assert protected_inventory() == protected, 'Frozen GPU/config/source/successful score evidence changed'
            # This is exactly the existing wrapper's input signature format.
            # It is created only after a successful full, warning-audited batch.
            signature = folder/'input.sha256'
            signature.write_text(proposal['input_sha256']+'\n', encoding='utf-8')
            state.update(status='pass', validation=validation, output_sha256=sha(folder/'output.json'),
                recovery_log_sha256=sha(folder/'scorer.log'), signature_sha256=sha(signature),
                other_methods_unchanged=True, archived_failure_unchanged=True,
                protected_gpu_config_runtime_and_successful_metrics_unchanged=True,
                protected_evidence_after=protected_inventory(),
                input_bytes_unchanged=True, actual_java_producer_heap='64G',
                cache_ready_for_frozen_analyzer=True,
                completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            write(AUDIT, state)
        except BaseException as error:
            state.update(status='failed', exception=repr(error), elapsed_seconds=time.perf_counter()-started,
                recovery_log_sha256=sha(folder/'scorer.log') if (folder/'scorer.log').exists() else None,
                cache_ready_for_frozen_analyzer=False)
            write(AUDIT, state)
            raise
    return state


class RecoveryTests(unittest.TestCase):
    def test_only_heap_command_difference(self):
        first, second = commands(Path('/metric'), Path('/package'), Path('/models'))
        self.assertEqual([(a,b) for a,b in zip(first,second) if a != b], [('-Xmx16G','-Xmx64G')])
        self.assertEqual(second[-2:], ['-threads','1'])
        self.assertIn('-XX:ActiveProcessorCount=8', second)

    def test_valid_full_output_and_bad_logs(self):
        output = [dict(image_id=i,scores={'All':{'f':0.5}}) for i in range(100)]
        self.assertEqual(validate_result(output,set(range(100)),'Parsing test captions')['status'], 'pass')
        for text in ['Error: Could not cache item', 'Error: Could not  cache item #2']:
            self.assertEqual(validate_result(output,set(range(100)),text)['status'], 'pass')
        for text in ['java.lang.OutOfMemoryError', 'Will ignore and continue.',
                     'Ignored sentence', 'Error: Could not score batched file input:', 'NullPointerException',
                     'Error: Could not open parser model']:
            with self.assertRaises(AssertionError):
                validate_result(output,set(range(100)),text)

    def test_missing_and_duplicate_ids_fail(self):
        output = [dict(image_id=i,scores={'All':{'f':0.5}}) for i in range(100)]
        with self.assertRaises(AssertionError):
            validate_result(output[:-1],set(range(100)),'')
        output[-1]['image_id'] = 0
        with self.assertRaises(AssertionError):
            validate_result(output,set(range(100)),'')

    def test_exact_input_and_inventory_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            old = folder/'old'; old.mkdir()
            source = old/'input.json'; source.write_bytes(b'[{"a": 1}]\n')
            before = inventory(old)
            archive = folder/'archive'
            shutil.move(str(old), str(archive))
            self.assertEqual(before, inventory(archive))
            restored = folder/'restored.json'
            shutil.copyfile(archive/'input.json',restored)
            self.assertEqual(sha(restored),sha(archive/'input.json'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true', help='Run CPU-only recovery after review')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(RecoveryTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    outcome = execute() if args.execute else plan()
    print(json.dumps(outcome, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
