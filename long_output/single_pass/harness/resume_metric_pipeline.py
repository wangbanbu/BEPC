"""Resume CPU scoring after the retained official SPICE 16G heap failure.

This entry point never invokes the GPU inference runner. Frozen experiment,
generation, metric definition, statistical and report compiler sources remain
unchanged. The separate recovery and independent audit record the resource fix.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / 'RUN_STATUS.json'
LOGS = ROOT / 'logs'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def main():
    previous = STATUS.read_bytes()
    state = json.loads(previous)
    assert state['status'] == 'failed' and state['stage'] == 'final_analysis'
    assert state['repetitions'] == 1
    assert not (ROOT / 'reports/FULL_EXPERIMENT_REPORT.md').exists()
    preserved = ROOT / 'audit/SPICE_RESOURCE_PIPELINE_FAILED_STATUS.json'
    assert not preserved.exists()
    preserved.write_bytes(previous)
    state['resource_recovery'] = dict(
        reason='Official SPICE LVSpec paper Java heap OOM; incomplete result rejected',
        previous_status_sha256=sha(preserved),
        inference_repeated=False,
        controller_source_sha256=sha(__file__),
        started_utc=datetime.now(timezone.utc).isoformat(),
    )
    state['previous_error'] = state.pop('error', None)
    state['previous_traceback'] = state.pop('traceback', None)
    state['controller_pid'] = os.getpid()

    def save(**fields):
        state.update(fields, updated_utc=datetime.now(timezone.utc).isoformat())
        temporary = STATUS.with_suffix('.tmp')
        write(temporary, state)
        temporary.replace(STATUS)

    def cpu(label, script, *args):
        log = LOGS / (label + '.log')
        assert not log.exists(), log
        save(status='running', stage=label, active_gpu_children=[])
        started = time.time()
        with log.open('w', encoding='utf-8') as handle:
            process = subprocess.run([sys.executable, 'harness/' + script, *args], cwd=ROOT,
                                     env=os.environ.copy(), stdout=handle, stderr=subprocess.STDOUT)
        state['stages'].append(dict(stage=label, exit_code=process.returncode,
                                    seconds=time.time() - started, log=str(log)))
        save()
        if process.returncode:
            raise RuntimeError(label + ' failed: ' + str(log))

    try:
        cpu('spice_resource_recovery', 'recover_spice_resources.py', '--execute')
        cpu('final_analysis_recovery', 'analyze_results.py', '--stage', 'all')
        cpu('spice_recovery_validation', 'validate_spice_resource_recovery.py')
        cpu('final_validation', 'validate_single.py', '--phase', 'all')
        cpu('full_report', 'compile_full_report.py')
        cpu('report_resource_addendum', 'append_recovery_addendum.py')
        save(status='completed', stage='completed', recovery_completed=True,
             completed_utc=datetime.now(timezone.utc).isoformat(),
             report=str(ROOT / 'reports/FULL_EXPERIMENT_REPORT.md'))
    except BaseException as error:
        save(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
