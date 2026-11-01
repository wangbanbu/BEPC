"""Single-factor response-budget ablation around unchanged native generation."""
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
PREVIOUS = ROOT.parent/'bepc_optimization_20261101'
sys.path.insert(0, str(ROOT/'harness'))
spec = importlib.util.spec_from_file_location('_uncapped_frozen_generation', ROOT/'harness/run_benchmark.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
from bepc_experimental import ExperimentalBEPC
from uncapped_controller import NoResponseCapBEPC


def factory(*args, **kwargs):
    cls = NoResponseCapBEPC if args[2].get('no_response_cap', False) else ExperimentalBEPC
    return cls(*args, **kwargs)


core.SpeedController = factory


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_once(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if path.exists():
        assert path.read_text(encoding='utf-8') == text, path
    else:
        path.write_text(text, encoding='utf-8')


core.write = write_once


def key(row):
    return str(row['question_id']), row['method'], int(row['repeat'])


def samples_for(cfg, phase, task, shard):
    split = 'test' if phase == 'main' else 'dev'
    return [s for s in read(cfg['data'][task][split+'_path']) if s['group_ordinal'] % 2 == shard]


def schedule(cfg, task, sample):
    names = list(cfg['methods_by_task'][task])
    index = cfg['parent_task_order'].index(task)
    random.Random(cfg['order_seed']+index*1000003+sample['ordinal']*101).shuffle(names)
    return names


def previous_rows(phase, task, shard):
    path = PREVIOUS/f'results/{phase}/{phase}_{task}_shard{shard}.jsonl'
    assert read(path.with_suffix('.complete.json'))['result_sha256'] == sha(path)
    return {(str(r['question_id']), r['method']): r for r in
            (json.loads(line) for line in path.read_text().splitlines()) if r['repeat'] == 0}


def historical_parity(row, history):
    source_name = {'bepc_cap4': 'bepc_fast', 'strict_shared': 'strict_shared'}.get(row['method'])
    if source_name is None:
        return
    source = history[str(row['question_id']), source_name]
    for field in ('input_tensor_sha256', 'prompt_token_ids', 'image_grid_thw', 'prompt_text',
                  'raw_token_ids', 'returned_token_ids', 'acceptance_length', 'acceptance_paths'):
        assert row[field] == source[field], ('control reproducibility', row['task'], row['question_id'], field)


class UncappedRuntime(core.Runtime):
    def run(self):
        assert self.cfg['repetitions'] == 1
        directory = ROOT/'results'/self.phase
        directory.mkdir(parents=True, exist_ok=True)
        for self.task in self.cfg['tasks']:
            path = directory/f'{self.phase}_{self.task}_shard{self.shard}.jsonl'
            samples = samples_for(self.cfg, self.phase, self.task, self.shard)
            methods = self.cfg['methods_by_task'][self.task]
            expected = {(str(s['question_id']), m, 0) for s in samples for m in methods}
            history = previous_rows(self.phase, self.task, self.shard)
            rows = {}
            if path.exists():
                for line in path.read_text(encoding='utf-8').splitlines():
                    row = json.loads(line)
                    identity = key(row)
                    assert identity in expected and identity not in rows
                    assert row['config_sha256'] == self.meta['config_sha256']
                    rows[identity] = row
            marker = path.with_suffix('.complete.json')
            if marker.exists():
                assert set(rows) == expected and read(marker)['result_sha256'] == sha(path)
                continue
            missing = expected-set(rows)
            # Warm up each live process; retained old markers never replace a
            # new process warmup if a failed process must be resumed.
            if missing:
                first = next(s for s in samples if any((str(s['question_id']), m, 0) in missing for m in methods))
                inputs = self.inputs(first)
                identity = core.input_digest(inputs)
                warmups = [self.generate(first, inputs, name, variant, -1, i, False, identity)
                           for i, (name, variant) in enumerate(methods.items())]
                for row in warmups:
                    historical_parity(row, history)
                import os
                write_once(ROOT/f'audit/{self.label}_{self.task}.warmup_pid{os.getpid()}.json',
                    dict(requests=len(warmups), seconds=sum(r['seconds'] for r in warmups),
                         control_historical_parity=True, excluded_from_measurements=True))
            start = time.perf_counter()
            with path.open('a', encoding='utf-8') as handle:
                for sample in samples:
                    names = schedule(self.cfg, self.task, sample)
                    needed = [m for m in names if (str(sample['question_id']), m, 0) not in rows]
                    if not needed:
                        continue
                    inputs = self.inputs(sample)
                    identity = core.input_digest(inputs)
                    for name in needed:
                        try:
                            row = self.generate(sample, inputs, name, methods[name], 0, names.index(name), False, identity)
                            historical_parity(row, history)
                        except Exception as error:
                            write_once(ROOT/f'audit/{self.label}_{self.task}.failure.json', dict(
                                exception=repr(error), traceback=traceback.format_exc(), task=self.task,
                                question_id=sample['question_id'], method=name, completed_records=len(rows)))
                            raise
                        row.update(measurement_origin='new_concurrent_single_pass',
                            isolated_entry_sha256=sha(__file__), inherited_generation_core_sha256=sha(ROOT/'harness/run_benchmark.py'))
                        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                        handle.flush()
                        rows[key(row)] = row
                    print(json.dumps(dict(phase=self.phase, task=self.task, shard=self.shard,
                        ordinal=sample['ordinal'], records=len(rows), expected=len(expected),
                        elapsed_seconds=time.perf_counter()-start)), flush=True)
            assert set(rows) == expected
            self.verify_checkout()
            write_once(marker, dict(complete=True, phase=self.phase, task=self.task, shard=self.shard,
                records=len(rows), questions=len(samples), repetitions=1, result_sha256=sha(path),
                metadata=self.meta, capped_and_strict_historical_parity=True,
                seconds_excluding_load_warmup=time.perf_counter()-start))
        self.guard.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['development', 'main'], required=True)
    parser.add_argument('--shard', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    UncappedRuntime(args.phase, args.shard).main()
