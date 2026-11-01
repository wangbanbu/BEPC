"""Independent CPU accounting, preservation and token-equivalence validation."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent/'single_pass'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]


def validate(phase):
    cfgpath = ROOT/'configs'/('frozen_config.json' if phase == 'main' else 'design_config.json')
    cfg, before = read(cfgpath), read(ROOT/'audit/PRIOR_PRESERVATION.json')
    changed = [p['path'] for p in before['files'] if sha(p['path']) != p['sha256']]
    assert not changed, ('Previously frozen code/results changed', changed)
    for name, expected in cfg['harness_sha256'].items():
        assert sha(ROOT/'harness'/name) == expected, name
    oldcfg = read(OLD/'configs/frozen_config.json')
    split, count = ('test', 100) if phase == 'main' else ('dev', 20)
    methods = {'strict_shared', 'bepc_original', 'bepc_fast', 'bepc_rescue0625'}
    tasks, traces = {}, []
    for task in cfg['tasks']:
        assert cfg['data'][task] == oldcfg['data'][task]
        assert cfg['methods_by_task'][task]['bepc_original'] == oldcfg['methods_by_task'][task]['bepc_algorithm']
        assert cfg['methods_by_task'][task]['strict_shared'] == oldcfg['methods_by_task'][task]['strict_shared']
        samples = read(cfg['data'][task][split+'_path'])
        assert len(samples) == count
        lookup = {}
        for shard in (0, 1):
            path = ROOT/'results'/phase/f'{phase}_{task}_shard{shard}.jsonl'
            marker = read(path.with_suffix('.complete.json'))
            rows = jsonl(path)
            assert marker['complete'] and marker['records'] == len(rows)
            assert marker['result_sha256'] == sha(path)
            assert marker['metadata']['config_sha256'] == sha(cfgpath)
            for row in rows:
                key = (str(row['question_id']), row['method'])
                assert key not in lookup and row['method'] in methods
                assert row['repeat'] == 0 and row['temperature'] == 0
                assert row['shard'] == row['group_ordinal'] % 2 == shard
                assert row['model_calls']['vision'] == 1 and row['extra_target_forwards'] == 0
                assert row['model_calls']['target'] == row['raw_verification_rounds']+1
                assert row['config_sha256'] == sha(cfgpath)
                assert row['request_lifecycle']['live_controllers_after_request'] == 0
                assert row['request_lifecycle']['live_logits_after_request'] == 0
                assert row['effective_accepted_draft_tokens']+row['effective_verification_rounds'] == row['completion_tokens']
                lookup[key] = row
        expected = {(str(s['question_id']), m) for s in samples for m in methods}
        assert set(lookup) == expected
        oldrows = {}
        for path in sorted((OLD/'results'/phase).glob(f'{phase}_{task}_shard*.jsonl')):
            for row in jsonl(path):
                if row['repeat'] == 0:
                    oldrows[(str(row['question_id']), row['method'])] = row
        retired, rounds = Counter(), Counter()
        for sample in samples:
            qid = str(sample['question_id'])
            paired = {m: lookup[qid, m] for m in methods}
            for row in paired.values():
                for field in ('input_tensor_sha256', 'prompt_token_ids', 'image_grid_thw', 'prompt_text', 'image_sha256'):
                    assert row[field] == paired['bepc_original'][field], (task, qid, field)
            for field in ('raw_token_ids', 'returned_token_ids', 'acceptance_length', 'acceptance_paths'):
                assert paired['bepc_original'][field] == paired['bepc_fast'][field], (task, qid, field)
            for method in ('bepc_fast', 'bepc_rescue0625'):
                stats = paired[method]['signal_stats']
                retired[method] += stats.get('exhausted_fast_rounds', 0)
                rounds[method] += paired[method]['raw_verification_rounds']
                assert stats.get('unsafe_bound_decisions', 0) == 0
                assert stats['committed_response_mismatches'] <= 4
            for method, historical in [('strict_shared', 'strict_shared'), ('bepc_original', 'bepc_algorithm')]:
                source = oldrows.get((qid, historical))
                if source is None:
                    continue
                traces.append(dict(task=task, question_id=qid, method=method,
                    prior_input_equal=source['input_tensor_sha256'] == paired[method]['input_tensor_sha256'],
                    prior_raw_tokens_equal=source['raw_token_ids'] == paired[method]['raw_token_ids'],
                    prior_visible_tokens_equal=source['returned_token_ids'] == paired[method]['returned_token_ids']))
        tasks[task] = dict(questions=count, records=len(lookup), original_fast_exact_parity=count,
            exhausted_fast_rounds=dict(retired), raw_rounds=dict(rounds),
            historical_comparisons=[x for x in traces if x['task'] == task])
    resources = []
    for shard in (0, 1):
        resource = read(ROOT/f'audit/{phase}_shard{shard}.gpu_resource.json')
        assert not resource['events'], resource
        resources.append(resource)
    payload = dict(status='pass', phase=phase, tasks=tasks,
        records=sum(x['records'] for x in tasks.values()), old_files_unchanged=len(before['files']),
        frozen_runtime_files_verified=len(cfg['harness_sha256']),
        historical_input_mismatches=sum(not x['prior_input_equal'] for x in traces),
        historical_visible_output_differences=sum(not x['prior_visible_tokens_equal'] for x in traces),
        resource_checks=resources, config_sha256=sha(cfgpath))
    destination = ROOT/'audit'/f'{phase.upper()}_VALIDATION.json'
    assert not destination.exists(), destination
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status='pass', phase=phase, records=payload['records'],
        old_files_unchanged=payload['old_files_unchanged'],
        historical_input_mismatches=payload['historical_input_mismatches'],
        historical_visible_output_differences=payload['historical_visible_output_differences'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['development', 'main'], required=True)
    validate(parser.parse_args().phase)
