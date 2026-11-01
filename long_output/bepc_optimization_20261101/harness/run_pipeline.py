"""Run main only after development parity, then official scoring and reporting."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/data/wbb/miniconda3/envs/spec/bin/python'
TASKS = ('scienceqa', 'coco', 'textvqa')


def status(stage, **extra):
    value = dict(status=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
    (ROOT/'RUN_STATUS.json').write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(value), flush=True)


def command(script, args, logname):
    with (ROOT/'logs'/logname).open('w') as log:
        subprocess.run([PYTHON, '-u', str(ROOT/'harness'/script), *args], cwd=ROOT,
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def gpu_pids():
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    return [int(line.strip()) for line in text.splitlines() if line.strip().isdigit()]


def counts(phase):
    return {path.name: sum(1 for _ in path.open()) for path in sorted((ROOT/'results'/phase).glob('*.jsonl'))}


def main():
    started = time.monotonic()
    status('waiting_for_development_validation')
    while not all((ROOT/f'results/development/development_{t}_shard{s}.complete.json').exists()
                  for t in TASKS for s in (0, 1)):
        failures = list((ROOT/'audit').glob('development*.failure.json'))
        if failures:
            raise RuntimeError('Development failure: '+str(failures))
        if time.monotonic()-started > 1200:
            raise RuntimeError('Development did not complete; retained logs need inspection')
        time.sleep(2)
    command('validate_experiment.py', ['--phase', 'development'], 'development_validation.log')
    while gpu_pids():
        time.sleep(1)
    status('main_running', development_validation='pass',
           expected_development_requests=240, expected_main_requests=1200)
    processes, handles = [], []
    for shard in (0, 1):
        handle = (ROOT/f'logs/main_shard{shard}.log').open('w')
        handles.append(handle)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard), PYTHONDONTWRITEBYTECODE='1',
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
        processes.append(subprocess.Popen([PYTHON, '-u', str(ROOT/'harness/run_experiment.py'),
            '--phase', 'main', '--shard', str(shard)], cwd=ROOT, env=env,
            stdout=handle, stderr=subprocess.STDOUT))
    # Fixed development gamma was frozen before these scores exist. CPU scoring
    # may overlap generation; it never imports gold into a GPU process.
    devlog = (ROOT/'logs/development_scoring.log').open('w')
    devscore = subprocess.Popen([PYTHON, '-u', str(ROOT/'harness/evaluate_experiment.py'),
        '--phase', 'development'], cwd=ROOT, stdout=devlog, stderr=subprocess.STDOUT)
    last = 0.
    while any(process.poll() is None for process in processes):
        if any(process.poll() not in (None, 0) for process in processes):
            raise RuntimeError('Main GPU process failed; all evidence retained')
        now = time.monotonic()
        if now-last >= 30:
            status('main_running', main_counts=counts('main'),
                   pipeline_elapsed_seconds=now-started, pids=[p.pid for p in processes],
                   development_scoring_exit=devscore.poll())
            last = now
        time.sleep(2)
    assert all(process.returncode == 0 for process in processes)
    for handle in handles:
        handle.close()
    command('validate_experiment.py', ['--phase', 'main'], 'main_validation.log')
    status('scoring_and_analysis', main_counts=counts('main'))
    devcode = devscore.wait()
    devlog.close()
    assert devcode == 0, 'Development official scoring failed; retained log'
    command('analyze_experiment.py', [], 'analysis.log')
    # Verify existing code/results again after all evaluator imports and caches.
    import hashlib
    prior = json.loads((ROOT/'audit/PRIOR_PRESERVATION.json').read_text())
    for entry in prior['files']:
        assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() == entry['sha256'], entry['path']
    status('analysis_complete_pending_final_review', main_counts=counts('main'),
           total_seconds=time.monotonic()-started, old_files_unchanged=len(prior['files']),
           report=str(ROOT/'reports/FULL_EXPERIMENT_REPORT.md'))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', exception=repr(error), all_existing_evidence_preserved=True)
        raise
