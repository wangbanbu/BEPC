"""Isolated BEPC experiment, inheriting frozen forward/KV/timing code unchanged."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'harness'))
spec = importlib.util.spec_from_file_location('_isolated_frozen_generation', ROOT/'harness/run_benchmark.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
from bepc_experimental import ExperimentalBEPC

original_controller = core.SpeedController


def controller_factory(*args, **kwargs):
    variant = args[2]
    cls = ExperimentalBEPC if variant.get('experimental_bepc', False) else original_controller
    return cls(*args, **kwargs)


core.SpeedController = controller_factory


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_once(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if path.exists():
        assert path.read_text(encoding='utf-8') == payload, path
    else:
        path.write_text(payload, encoding='utf-8')


core.write = write_once


def sample_key(row):
    return str(row['question_id']), row['method'], int(row['repeat'])


def samples_for(cfg, phase, task, shard):
    split = 'test' if phase == 'main' else 'dev'
    return [s for s in read(cfg['data'][task][split+'_path']) if s['group_ordinal'] % 2 == shard]


def schedule(cfg, task, sample):
    names = list(cfg['methods_by_task'][task])
    # Preserve the task-specific seed offsets used in the five-task experiment.
    task_index = cfg['parent_task_order'].index(task)
    random.Random(cfg['order_seed'] + task_index*1000003 + sample['ordinal']*101).shuffle(names)
    return names


class ExperimentRuntime(core.Runtime):
    def run(self):
        assert self.cfg['repetitions'] == 1
        directory = ROOT/'results'/self.phase
        directory.mkdir(parents=True, exist_ok=True)
        for self.task in self.cfg['tasks']:
            target = directory/f'{self.phase}_{self.task}_shard{self.shard}.jsonl'
            samples = samples_for(self.cfg, self.phase, self.task, self.shard)
            methods = self.cfg['methods_by_task'][self.task]
            expected = {(str(s['question_id']), m, 0) for s in samples for m in methods}
            rows = {}
            if target.exists():
                for line in target.read_text(encoding='utf-8').splitlines():
                    row = json.loads(line)
                    key = sample_key(row)
                    assert key in expected and key not in rows, (target, key)
                    assert row['config_sha256'] == self.meta['config_sha256']
                    rows[key] = row
            marker = target.with_suffix('.complete.json')
            if marker.exists():
                assert set(rows) == expected and read(marker)['result_sha256'] == sha(target)
                continue
            missing = expected-set(rows)
            warmup_path = ROOT/f'audit/{self.label}_{self.task}.warmup.json'
            if missing and not warmup_path.exists():
                first = next(s for s in samples if any((str(s['question_id']), m, 0) in missing for m in methods))
                inputs = self.inputs(first)
                identity = core.input_digest(inputs)
                warmups = [self.generate(first, inputs, name, variant, -1, i, False, identity)
                           for i, (name, variant) in enumerate(methods.items())]
                # Check the changed equivalence path immediately, outside timed evidence.
                by_method = {r['method']: r for r in warmups}
                for field in ('raw_token_ids', 'returned_token_ids', 'acceptance_length', 'acceptance_paths'):
                    assert by_method['bepc_original'][field] == by_method['bepc_fast'][field], ('warmup parity', self.task, field)
                write_once(warmup_path, dict(requests=len(warmups), seconds=sum(r['seconds'] for r in warmups),
                    original_fast_exact_parity=True, excluded_from_measurements=True))
            start = time.perf_counter()
            with target.open('a', encoding='utf-8') as handle:
                for sample in samples:
                    names = schedule(self.cfg, self.task, sample)
                    needed = [m for m in names if (str(sample['question_id']), m, 0) not in rows]
                    if not needed:
                        continue
                    inputs = self.inputs(sample)
                    identity = core.input_digest(inputs)
                    for method in needed:
                        try:
                            row = self.generate(sample, inputs, method, methods[method], 0, names.index(method), False, identity)
                        except Exception as error:
                            write_once(ROOT/f'audit/{self.label}_{self.task}.failure.json', dict(
                                exception=repr(error), traceback=traceback.format_exc(), task=self.task,
                                question_id=sample['question_id'], method=method, completed_records=len(rows)))
                            raise
                        row.update(measurement_origin='new_concurrent_single_pass',
                            isolated_entry_sha256=sha(__file__), inherited_generation_core_sha256=sha(ROOT/'harness/run_benchmark.py'))
                        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                        handle.flush()
                        rows[sample_key(row)] = row
                    old = rows[(str(sample['question_id']), 'bepc_original', 0)]
                    fast = rows[(str(sample['question_id']), 'bepc_fast', 0)]
                    for field in ('raw_token_ids', 'returned_token_ids', 'acceptance_length', 'acceptance_paths'):
                        assert old[field] == fast[field], ('optimization changed output', self.task, sample['question_id'], field)
                    print(json.dumps(dict(phase=self.phase, task=self.task, shard=self.shard,
                        ordinal=sample['ordinal'], records=len(rows), expected=len(expected),
                        elapsed_seconds=time.perf_counter()-start)), flush=True)
            assert set(rows) == expected
            self.verify_checkout()
            write_once(marker, dict(complete=True, phase=self.phase, task=self.task, shard=self.shard,
                records=len(rows), questions=len(samples), repetitions=1, result_sha256=sha(target),
                metadata=self.meta, exact_original_fast_parity=True,
                seconds_excluding_load_warmup=time.perf_counter()-start))
        self.guard.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['development', 'main'], required=True)
    parser.add_argument('--shard', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    ExperimentRuntime(args.phase, args.shard).main()


if __name__ == '__main__':
    main()
