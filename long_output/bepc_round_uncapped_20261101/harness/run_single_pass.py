"""Single-pass scheduling around the byte-identical, frozen long-output runtime.

Only repeat 0 is retained from the stopped three-repeat run. Generation and
measurement are inherited without modification from its SHA-pinned runtime.
"""
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
PARENT = ROOT.parent
CORE = PARENT / 'harness/run_benchmark.py'
CORE_SHA256 = 'a8ffa14bcef3eda7f9436945b13f24c820d7897fc34bfb89fc9c4369f14f899a'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_once(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists():
        assert path.read_text(encoding='utf-8') == payload, f'Preserve existing evidence: {path}'
    else:
        path.write_text(payload, encoding='utf-8')


assert digest(CORE) == CORE_SHA256, 'Frozen parent generation runtime changed'
sys.path.insert(0, str(ROOT / 'harness'))
spec = importlib.util.spec_from_file_location('_single_pass_frozen_core', CORE)
core = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(core)
core.ROOT = ROOT
# Loading a resumed PID may encounter the same immutable metadata seal. Reuse
# only byte-identical metadata; conflicting evidence still raises. No model,
# generation, attention, numerical, or timer function is replaced.
core.write = write_once


def schedule(cfg, task, sample):
    names = list(cfg['methods_by_task'][task] if cfg.get('phase') == 'main' else cfg['development_methods'])
    random.Random(cfg['order_seed'] + cfg['tasks'].index(task)*1000003 + sample['ordinal']*101).shuffle(names)
    return names


def key(row):
    return str(row['question_id']), row['method'], int(row['repeat'])


def jsonl_rows(path):
    if not path.exists():
        return []
    result = []
    for line_no, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        # A stopped write can leave a trailing partial line; preserve it as an
        # invalid source record instead of silently treating it as a generation.
        try:
            result.append((line_no, json.loads(line)))
        except json.JSONDecodeError:
            assert line_no == len(path.read_text(encoding='utf-8').splitlines()), (path, line_no)
            result.append((line_no, None))
    return result


def task_samples(cfg, phase, task, shard):
    split = 'test' if phase == 'main' else 'dev'
    return [s for s in read(cfg['data'][task][split+'_path']) if s['group_ordinal'] % 2 == shard]


def methods_for(cfg, phase, task):
    return cfg['methods_by_task'][task] if phase == 'main' else cfg['development_methods']


def expected_keys(cfg, phase, task, shard):
    return {(str(s['question_id']), m, 0) for s in task_samples(cfg, phase, task, shard)
            for m in methods_for(cfg, phase, task)}


def validated_existing(path, expected):
    found = {}
    for line_no, row in jsonl_rows(path):
        assert row is not None, f'Incomplete new single-pass JSONL; preserve and repair explicitly: {path}:{line_no}'
        identity = key(row)
        assert identity in expected, (path, line_no, identity)
        assert identity not in found, f'Duplicate single-pass record: {path}:{line_no}:{identity}'
        found[identity] = row
    return found


def seal_task(cfg, phase, task, shard, target, rows, *, metadata=None, seconds=None):
    expected = expected_keys(cfg, phase, task, shard)
    assert set(rows) == expected
    config_path = ROOT/'configs'/('frozen_config.json' if phase == 'main' else 'design_config.json')
    meta = dict(metadata or {})
    meta.update(config_sha256=digest(config_path), phase=phase,
                source_sha256=cfg['harness_sha256'], single_pass_runtime_entry_sha256=digest(__file__),
                single_pass_runtime_entry_path=str(Path(__file__).resolve()),
                inherited_generation_core_sha256=CORE_SHA256)
    imported = sum(r.get('measurement_origin') == 'reused_parent_repeat0' for r in rows.values())
    result = dict(complete=True, task=task, phase=phase, records=len(rows),
                  questions=len(task_samples(cfg, phase, task, shard)),
                  result_sha256=digest(target), metadata=meta, repetitions=1,
                  parent_repeat0_reused_records=imported, newly_generated_records=len(rows)-imported,
                  all_measurements_single_pass=True)
    if seconds is not None:
        result['seconds_excluding_load_warmup'] = seconds
    write_once(target.with_suffix('.complete.json'), result)


def import_development(cfg, shard):
    """CPU-only evidence reuse. Selection depends solely on repeat == 0."""
    assert cfg['repetitions'] == 1 and cfg['warmups'] == 1
    config_path = ROOT/'configs/design_config.json'
    parent_config = PARENT/'configs/design_config.json'
    parent_cfg = read(parent_config)
    assert cfg['development_methods'] == parent_cfg['development_methods']
    for task in cfg['tasks']:
        for split in ('dev', 'test'):
            assert cfg['data'][task][split+'_sha256'] == parent_cfg['data'][task][split+'_sha256']
    directory = ROOT/'results/development'
    directory.mkdir(parents=True, exist_ok=True)
    manifests = []
    for task in cfg['tasks']:
        source = PARENT/'results/development'/f'development_{task}_shard{shard}.jsonl'
        target = directory/source.name
        expected = expected_keys(cfg, 'development', task, shard)
        existing = validated_existing(target, expected)
        source_sha = digest(source) if source.exists() else None
        imports = skipped_repeats = incomplete = 0
        parent_keys = set()
        with target.open('a', encoding='utf-8') as handle:
            for line_no, row in jsonl_rows(source):
                if row is None:
                    incomplete += 1
                    continue
                if row['repeat'] != 0:
                    skipped_repeats += 1
                    continue
                identity = key(row)
                assert identity in expected, (source, line_no, identity)
                assert identity not in parent_keys, (source, line_no, 'duplicate parent repeat0')
                parent_keys.add(identity)
                assert row['config_sha256'] == digest(parent_config)
                sample = next(s for s in task_samples(cfg, 'development', task, shard)
                              if str(s['question_id']) == str(row['question_id']))
                assert row['prompt_text'] == sample['prompt_text']
                assert row['image_sha256'] == sample['image_sha256']
                assert row['schedule_position'] == schedule(cfg, task, sample).index(row['method'])
                if identity in existing:
                    previous = existing[identity]
                    assert previous['measurement_origin'] == 'reused_parent_repeat0'
                    assert previous['single_pass_provenance']['parent_record_sha256'] == hashlib.sha256(
                        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
                    continue
                new = dict(row)
                new['config_sha256'] = digest(config_path)
                new['measurement_origin'] = 'reused_parent_repeat0'
                new['single_pass_provenance'] = dict(parent_path=str(source), parent_line=line_no,
                    parent_file_sha256=source_sha, parent_config_sha256=row['config_sha256'],
                    parent_record_sha256=hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True,
                        separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
                    parent_record_sha256_encoding='UTF8 canonical JSON sorted keys compact separators, no newline',
                    normalized_fields={'config_sha256': {'original': row['config_sha256'], 'normalized': digest(config_path)}},
                    parent_runtime_sha256=CORE_SHA256,
                    selection_rule='all successfully persisted repeat == 0, no output/quality filtering')
                handle.write(json.dumps(new, ensure_ascii=False, allow_nan=False)+'\n')
                handle.flush()
                existing[identity] = new
                imports += 1
        manifests.append(dict(task=task, shard=shard, source_path=str(source), source_sha256=source_sha,
            newly_imported_records=imports, total_single_pass_records=len(existing),
            excluded_parent_repeat1_or_repeat2_records=skipped_repeats,
            incomplete_parent_trailing_lines=incomplete, expected_records=len(expected)))
        if set(existing) == expected and not target.with_suffix('.complete.json').exists():
            parent_marker = source.with_suffix('.complete.json')
            meta = read(parent_marker).get('metadata', {}) if parent_marker.exists() else {}
            seal_task(cfg, 'development', task, shard, target, existing, metadata=meta)
    # Initial import ledger is immutable; subsequent CLI resumes check the rows,
    # and do not overwrite this historical first-import accounting.
    audit = ROOT/'audit'/f'development_shard{shard}.parent_repeat0_import.json'
    if not audit.exists():
        write_once(audit, dict(status='pass', records=manifests, new_config_sha256=digest(config_path),
            parent_config_sha256=digest(parent_config), inherited_generation_core_sha256=CORE_SHA256))
    return manifests


class SinglePassRuntime(core.Runtime):
    """Only the scheduler differs; generation/KV/attention/timers are inherited."""
    def __init__(self, phase, shard):
        super().__init__(phase, shard)
        self.meta.update(single_pass_runtime_entry_sha256=digest(__file__),
            single_pass_runtime_entry_path=str(Path(__file__).resolve()),
            inherited_generation_core_sha256=CORE_SHA256)
        write_once(ROOT/f'audit/{self.label}.single_pass_loaded.json', self.meta)

    def run(self):
        assert self.cfg['repetitions'] == 1 and self.cfg['warmups'] == 1
        directory = ROOT/'results'/self.phase
        directory.mkdir(parents=True, exist_ok=True)
        for self.task in self.cfg['tasks']:
            target = directory/f'{self.phase}_{self.task}_shard{self.shard}.jsonl'
            samples = task_samples(self.cfg, self.phase, self.task, self.shard)
            methods = methods_for(self.cfg, self.phase, self.task)
            expected = expected_keys(self.cfg, self.phase, self.task, self.shard)
            rows = validated_existing(target, expected)
            if target.with_suffix('.complete.json').exists():
                marker = read(target.with_suffix('.complete.json'))
                assert set(rows) == expected and marker['result_sha256'] == digest(target)
                print(json.dumps(dict(phase=self.phase, task=self.task, shard=self.shard,
                    skipped_complete=True, records=len(rows))), flush=True)
                continue
            missing = expected-set(rows)
            if not missing:
                seal_task(self.cfg, self.phase, self.task, self.shard, target, rows, metadata=self.meta)
                continue
            warmup_path = ROOT/f'audit/{self.label}_{self.task}.warmup.json'
            if not warmup_path.exists():
                first = next(s for s in samples if any((str(s['question_id']),m,0) in missing for m in methods))
                inputs = self.inputs(first)
                identity = core.input_digest(inputs)
                warmups = [self.generate(first, inputs, name, variant, -1, j, False, identity)
                           for j, (name, variant) in enumerate(methods.items())]
                write_once(warmup_path, dict(requests=len(warmups),
                    seconds=sum(r['seconds'] for r in warmups), lifecycle_checked_every_request=True,
                    one_warmup_per_method=True, resumed_partial_task=bool(rows)))
            count = len(rows)
            started = time.perf_counter()
            cfg = dict(self.cfg, phase=self.phase)
            with target.open('a', encoding='utf-8') as handle:
                for sample in samples:
                    scheduled = schedule(cfg, self.task, sample)
                    missing_names = [m for m in scheduled if (str(sample['question_id']),m,0) not in rows]
                    if not missing_names:
                        continue
                    inputs = self.inputs(sample)
                    identity = core.input_digest(inputs)
                    for method in missing_names:
                        order = scheduled.index(method)
                        try:
                            row = self.generate(sample, inputs, method, methods[method], 0, order, False, identity)
                        except Exception as error:
                            failure = directory/f'{self.label}_{self.task}.failure.json'
                            write_once(failure, dict(exception=repr(error), traceback=traceback.format_exc(),
                                image_id=sample['image_id'], question_id=sample['question_id'], method=method,
                                repeat=0, completed_records=count, config_sha256=self.meta['config_sha256']))
                            raise
                        row['measurement_origin'] = 'new_single_pass_generation'
                        row['single_pass_provenance'] = dict(runtime_entry_sha256=digest(__file__),
                            inherited_generation_core_sha256=CORE_SHA256,
                            resumed_partial_question=len(missing_names) != len(methods))
                        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                        handle.flush()
                        rows[key(row)] = row
                        count += 1
                    print(json.dumps(dict(phase=self.phase, task=self.task, shard=self.shard,
                        ordinal=sample['ordinal'], group_ordinal=sample['group_ordinal'], records=count,
                        seconds=time.perf_counter()-started, repeat=0)), flush=True)
            self.verify_checkout()
            seal_task(self.cfg, self.phase, self.task, self.shard, target, rows,
                      metadata=self.meta, seconds=time.perf_counter()-started)
        self.guard.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['development', 'main'], default='development')
    parser.add_argument('--shard', type=int, choices=[0, 1], required=True)
    parser.add_argument('--import-only', action='store_true')
    args = parser.parse_args()
    cfg_path = ROOT/'configs'/('frozen_config.json' if args.phase == 'main' else 'design_config.json')
    cfg = read(cfg_path)
    if args.phase == 'development':
        print(json.dumps(dict(parent_repeat0_import=import_development(cfg, args.shard))), flush=True)
    if args.import_only:
        assert args.phase == 'development'
        return
    if all((ROOT/'results'/args.phase/f'{args.phase}_{t}_shard{args.shard}.complete.json').exists()
           for t in cfg['tasks']):
        print(json.dumps(dict(phase=args.phase, shard=args.shard, all_tasks_already_complete=True)), flush=True)
        return
    SinglePassRuntime(args.phase, args.shard).main()


if __name__ == '__main__':
    main()
