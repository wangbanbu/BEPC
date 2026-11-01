"""Development check -> concurrent paired test -> CPU official metrics."""
from __future__ import annotations
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
PYTHON='/data/wbb/miniconda3/envs/spec/bin/python'


def status(stage,**extra):
    v=dict(status=stage,updated_utc=datetime.now(timezone.utc).isoformat(),**extra)
    (ROOT/'RUN_STATUS.json').write_text(json.dumps(v,indent=2)+'\n')
    print(json.dumps(v),flush=True)


def command(script,args,logname):
    with (ROOT/'logs'/logname).open('w') as log:
        subprocess.run([PYTHON,'-u',str(ROOT/'harness'/script),*args],cwd=ROOT,
                       stdout=log,stderr=subprocess.STDOUT,check=True)


def gpu_pids():
    t=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    return [int(s.strip()) for s in t.splitlines() if s.strip().isdigit()]


def gpu_phase(phase,expected):
    processes,handles=[],[]
    for shard in (0,1):
        h=(ROOT/f'logs/{phase}_shard{shard}.log').open('w');handles.append(h)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),PYTHONDONTWRITEBYTECODE='1',
            OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
        processes.append(subprocess.Popen([PYTHON,'-u',str(ROOT/'harness/run_comparison.py'),
            '--phase',phase,'--shard',str(shard)],cwd=ROOT,env=env,stdout=h,stderr=subprocess.STDOUT))
    last=0
    while any(p.poll() is None for p in processes):
        if any(p.poll() not in (None,0) for p in processes):
            raise RuntimeError('GPU phase failed; logs and all measurements retained')
        now=time.monotonic()
        if now-last>=30:
            counts={p.name:sum(1 for _ in p.open()) for p in (ROOT/f'results/{phase}').glob('*.jsonl')}
            status(phase+'_running',counts=counts,expected=expected,pids=[p.pid for p in processes])
            last=now
        time.sleep(2)
    assert all(p.returncode==0 for p in processes)
    for h in handles:h.close()
    assert not gpu_pids(),'Scoring/validation only after generation processes release GPUs'
    command('validate_comparison.py',['--phase',phase],phase+'_validation.log')


def main():
    started=time.monotonic()
    assert not gpu_pids()
    gpu_phase('development',120)
    gpu_phase('main',600)
    (ROOT/'audit/GPU_COMPLETION.json').write_text(json.dumps(dict(
        all_generation_exited=True,no_GPU_pids=True,main_records=600,development_records=120,
        official_CPU_scoring_overlaps_GPU=False,created_utc=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
    status('scoring_and_analysis',official_CPU_scoring_overlaps_GPU=False)
    while not (ROOT/'audit/METRIC_IMPLEMENTATION_FREEZE.json').exists():
        # Freeze is performed by the integrating agent once the evaluator has
        # passed CPU checks; elapsed time never substitutes for this condition.
        time.sleep(2)
    command('evaluate_comparison.py',['--phase','development'],'development_scoring.log')
    command('analyze_comparison.py',[],'analysis.log')
    status('analysis_complete_pending_final_review',total_seconds=time.monotonic()-started,
        main_records=600,development_records=120,report=str(ROOT/'reports/FULL_EXPERIMENT_REPORT.md'))


if __name__=='__main__':
    try:main()
    except Exception as e:
        status('failed',exception=repr(e),all_evidence_preserved=True)
        raise
