"""CPU-only evidence accounting for response-budget removal."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT.parent/'bepc_optimization_20261101'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def validate(phase):
    cfgpath = ROOT/'configs'/('frozen_config.json' if phase == 'main' else 'design_config.json')
    cfg = read(cfgpath)
    preservation = read(ROOT/'audit/PRIOR_PRESERVATION.json')
    for entry in preservation['files']:
        assert sha(entry['path']) == entry['sha256'], entry['path']
    for name, expected in cfg['harness_sha256'].items():
        assert sha(ROOT/'harness'/name) == expected, name
    count = 100 if phase == 'main' else 20
    split = 'test' if phase == 'main' else 'dev'
    tasks = {}
    for task in cfg['tasks']:
        samples = read(cfg['data'][task][split+'_path'])
        assert len(samples) == count
        methods = cfg['methods_by_task'][task]
        assert methods['bepc_no_response_cap']['response_mismatch_budget'] is None
        assert methods['bepc_no_response_cap']['round_mismatch_budget'] == 2
        rows, historic = {}, {}
        for shard in (0, 1):
            path = ROOT/f'results/{phase}/{phase}_{task}_shard{shard}.jsonl'
            marker = read(path.with_suffix('.complete.json'))
            assert marker['complete'] and marker['result_sha256'] == sha(path)
            assert marker['metadata']['config_sha256'] == sha(cfgpath)
            batch = lines(path)
            assert len(batch) == marker['records']
            for row in batch:
                identity = str(row['question_id']), row['method']
                assert identity not in rows
                assert row['repeat'] == 0 and row['temperature'] == 0
                assert row['config_sha256'] == sha(cfgpath)
                assert row['shard'] == row['group_ordinal'] % 2 == shard
                assert row['extra_target_forwards'] == 0 and row['model_calls']['vision'] == 1
                assert row['model_calls']['target'] == row['raw_verification_rounds']+1
                assert row['effective_accepted_draft_tokens']+row['effective_verification_rounds'] == row['completion_tokens']
                for field in ('live_controllers_after_request', 'live_logits_after_request'):
                    assert row['request_lifecycle'][field] == 0
                if row['method'] != 'strict_shared':
                    paths = row['acceptance_paths']
                    assert len(paths) == row['raw_verification_rounds']
                    committed = 0
                    for i, selected in enumerate(paths):
                        assert selected['round'] == i
                        assert selected['response_mismatches_before'] == committed
                        assert 0 <= selected['mismatches'] <= 2
                        assert selected['accepted_draft'] == row['acceptance_length'][i]
                        committed += selected['mismatches']
                    stats = row['signal_stats']
                    assert committed == stats['committed_response_mismatches']
                    assert stats.get('unsafe_bound_decisions', 0) == 0
                    if row['method'] == 'bepc_cap4':
                        assert committed <= 4 and stats['response_mismatch_budget'] == 4
                    else:
                        assert stats['response_mismatch_budget'] is None
                        assert stats['response_cap_enabled'] is False
                        assert stats.get('observer_retirements', 0) == 0
                        assert stats.get('exhausted_fast_rounds', 0) == 0
                        assert stats.get('response_budget_binding', 0) == 0
                rows[identity] = row
            oldpath = PREVIOUS/f'results/{phase}/{phase}_{task}_shard{shard}.jsonl'
            assert read(oldpath.with_suffix('.complete.json'))['result_sha256'] == sha(oldpath)
            for row in lines(oldpath):
                historic[str(row['question_id']), row['method']] = row
        expected = {(str(s['question_id']), m) for s in samples for m in methods}
        assert set(rows) == expected
        divergent = []
        counts = []
        for sample in samples:
            qid = str(sample['question_id'])
            paired = {m: rows[qid, m] for m in methods}
            for field in ('input_tensor_sha256', 'image_sha256', 'prompt_text', 'prompt_token_ids', 'image_grid_thw'):
                assert all(r[field] == paired['bepc_cap4'][field] for r in paired.values())
            for method, prior in [('strict_shared', 'strict_shared'), ('bepc_cap4', 'bepc_fast')]:
                source = historic[qid, prior]
                for field in ('raw_token_ids', 'returned_token_ids', 'acceptance_length', 'acceptance_paths', 'input_tensor_sha256'):
                    assert paired[method][field] == source[field], (task, qid, method, field)
            cap, no = paired['bepc_cap4'], paired['bepc_no_response_cap']
            counts.append(no['signal_stats']['committed_response_mismatches'])
            # Before accumulated count3, even two new mismatches cannot hit
            # the response4 boundary. The first policy divergence must respect
            # that property, using actual selected native-tree paths.
            for a, b in zip(cap['acceptance_paths'], no['acceptance_paths']):
                if a != b:
                    assert a['response_mismatches_before'] == b['response_mismatches_before'] >= 3, (task, qid, a, b)
                    divergent.append(dict(question_id=qid, first_divergent_round=a['round'],
                        shared_committed_mismatches=a['response_mismatches_before']))
                    break
        tasks[task] = dict(records=len(rows), questions=count,
            historical_control_parity=2*count, uncapped_over_four=sum(v>4 for v in counts),
            uncapped_mean_mismatches=sum(counts)/count, uncapped_max_mismatches=max(counts),
            first_divergent_paths=divergent)
    resource = []
    for shard in (0, 1):
        value = read(ROOT/f'audit/{phase}_shard{shard}.gpu_resource.json')
        assert not value['events'], value
        resource.append(value)
    result = dict(status='pass', phase=phase, tasks=tasks, records=sum(v['records'] for v in tasks.values()),
        historical_files_unchanged=len(preservation['files']),
        frozen_sources_verified=len(cfg['harness_sha256']), resource_checks=resource,
        config_sha256=sha(cfgpath))
    output = ROOT/'audit'/f'{phase.upper()}_VALIDATION.json'
    assert not output.exists(), output
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(status='pass', phase=phase, records=result['records'],
        old_files_unchanged=result['historical_files_unchanged'],
        uncapped_over_four={t:v['uncapped_over_four'] for t,v in tasks.items()})))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['development', 'main'], required=True)
    validate(parser.parse_args().phase)
