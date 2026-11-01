"""Gate development, run paired GPU ablation, then CPU-only scoring."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
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


def counts():
    return {p.name: sum(1 for _ in p.open()) for p in sorted((ROOT/'results/main').glob('*.jsonl'))}


def main():
    started = time.monotonic()
    status('waiting_for_development_validation', main_expected=900, development_expected=180)
    while not all((ROOT/f'results/development/development_{t}_shard{s}.complete.json').exists()
                  for t in TASKS for s in (0, 1)):
        failures = list((ROOT/'audit').glob('development*.failure.json'))
        if failures:
            raise RuntimeError('Development failure, evidence retained: '+str(failures))
        if time.monotonic()-started > 1200:
            raise RuntimeError('Development did not finish; inspect retained logs')
        time.sleep(2)
    while gpu_pids():
        time.sleep(1)
    command('validate_uncapped.py', ['--phase', 'development'], 'development_validation.log')
    status('main_running', development_validation='pass', main_expected=900)
    processes, handles = [], []
    for shard in (0, 1):
        handle = (ROOT/f'logs/main_shard{shard}.log').open('w')
        handles.append(handle)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard), PYTHONDONTWRITEBYTECODE='1',
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
        processes.append(subprocess.Popen([PYTHON, '-u', str(ROOT/'harness/run_uncapped.py'),
            '--phase', 'main', '--shard', str(shard)], cwd=ROOT, env=env,
            stdout=handle, stderr=subprocess.STDOUT))
    last = 0.
    while any(p.poll() is None for p in processes):
        if any(p.poll() not in (None, 0) for p in processes):
            raise RuntimeError('Main GPU process failed; evidence retained')
        now = time.monotonic()
        if now-last >= 30:
            status('main_running', main_counts=counts(), main_expected=900,
                pipeline_elapsed_seconds=now-started, pids=[p.pid for p in processes],
                official_CPU_scoring_overlaps_GPU=False)
            last = now
        time.sleep(2)
    assert all(p.returncode == 0 for p in processes)
    for handle in handles:
        handle.close()
    command('validate_uncapped.py', ['--phase', 'main'], 'main_validation.log')
    assert not gpu_pids(), 'CPU scoring starts after GPU generation releases both GPUs'
    status('scoring_and_analysis', main_counts=counts(), official_CPU_scoring_overlaps_GPU=False)
    command('evaluate_uncapped.py', ['--phase', 'development'], 'development_scoring.log')
    command('analyze_uncapped.py', [], 'analysis.log')
    prior = json.loads((ROOT/'audit/PRIOR_PRESERVATION.json').read_text())
    for entry in prior['files']:
        assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() == entry['sha256'], entry['path']
    status('analysis_complete_pending_final_review', main_counts=counts(),
        total_seconds=time.monotonic()-started, old_files_unchanged=len(prior['files']),
        report=str(ROOT/'reports/FULL_EXPERIMENT_REPORT.md'))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', exception=repr(error), all_existing_evidence_preserved=True)
        raise
