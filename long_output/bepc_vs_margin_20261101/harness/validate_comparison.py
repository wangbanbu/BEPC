"""CPU-only runtime accounting, unchanged BEPC and fair-input validation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PREVIOUS=ROOT.parent/'bepc_response_uncapped_20261101'
METHODS=('bepc_no_response_cap','margin_no_response_cap')


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lines(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s]


def validate(phase):
    cp=ROOT/'configs'/('frozen_config.json' if phase=='main' else 'design_config.json')
    cfg=read(cp)
    for name,digest in cfg['harness_sha256'].items(): assert sha(ROOT/'harness'/name)==digest,name
    prior=read(ROOT/'audit/PRIOR_PRESERVATION.json')
    for entry in prior['files']: assert sha(entry['path'])==entry['sha256'],entry['path']
    count=100 if phase=='main' else 20
    split='test' if phase=='main' else 'dev'
    tasks={}
    for task in cfg['tasks']:
        samples=read(cfg['data'][task][split+'_path'])
        rows,history={},{}
        for shard in (0,1):
            p=ROOT/f'results/{phase}/{phase}_{task}_shard{shard}.jsonl'
            mark=read(p.with_suffix('.complete.json'))
            assert mark['complete'] and mark['result_sha256']==sha(p)
            assert mark['metadata']['config_sha256']==sha(cp)
            batch=lines(p)
            assert len(batch)==mark['records']==count
            for r in batch:
                key=str(r['question_id']),r['method']
                assert key not in rows
                assert r['repeat']==0 and r['temperature']==0 and not r['diagnostic_recording']
                assert r['config_sha256']==sha(cp)
                assert r['shard']==r['group_ordinal']%2==shard
                assert r['extra_target_forwards']==0 and r['model_calls']['vision']==1
                assert r['model_calls']['target']==r['raw_verification_rounds']+1
                assert r['effective_accepted_draft_tokens']+r['effective_verification_rounds']==r['completion_tokens']
                assert not r['request_lifecycle']['live_controllers_after_request']
                assert not r['request_lifecycle']['live_logits_after_request']
                n=0
                for i,pth in enumerate(r['acceptance_paths']):
                    assert pth['round']==i and pth['response_mismatches_before']==n
                    assert 0<=pth['mismatches']<=2
                    assert pth['accepted_draft']==r['acceptance_length'][i]
                    n+=pth['mismatches']
                stats=r['signal_stats']
                assert n==stats['committed_response_mismatches']
                assert stats['response_mismatch_budget'] is None and stats['response_cap_enabled'] is False
                assert stats['round_mismatch_budget']==2 and not stats.get('response_budget_binding',0)
                assert not stats.get('unsafe_bound_decisions',0)
                assert not stats.get('observer_retirements',0)
                if r['method']=='margin_no_response_cap':
                    assert stats['plain_margin_without_representation_observers'] is True
                    assert not stats['spaces'] and not stats['observer_context_counters']
                    assert not stats['visual_observer_counters'] and stats['full_candidate_scores']==0
                rows[key]=r
            old=PREVIOUS/f'results/{phase}/{phase}_{task}_shard{shard}.jsonl'
            assert read(old.with_suffix('.complete.json'))['result_sha256']==sha(old)
            for r in lines(old):
                if r['method']=='bepc_no_response_cap':history[str(r['question_id'])]=r
        assert len(samples)==count
        assert set(rows)=={(str(s['question_id']),m) for s in samples for m in METHODS}
        changed=0
        for s in samples:
            q=str(s['question_id'])
            a,b=rows[q,METHODS[0]],rows[q,METHODS[1]]
            for f in ('input_tensor_sha256','image_sha256','prompt_text','prompt_token_ids','image_grid_thw'):
                assert a[f]==b[f],(task,q,f)
            for f in ('input_tensor_sha256','prompt_token_ids','raw_token_ids','returned_token_ids','acceptance_length','acceptance_paths'):
                assert a[f]==history[q][f],(task,q,'BEPC reproducibility',f)
            changed+=a['output_text']!=b['output_text']
        tasks[task]=dict(records=len(rows),questions=count,BEPC_historical_exact=count,
            paired_inputs_exact=count,outputs_changed=changed)
    resources=[read(ROOT/f'audit/{phase}_shard{s}.gpu_resource.json') for s in (0,1)]
    assert all(not r['events'] for r in resources)
    value=dict(status='pass',phase=phase,tasks=tasks,records=sum(t['records'] for t in tasks.values()),
        prior_files_unchanged=len(prior['files']),config_sha256=sha(cp),resources=resources)
    out=ROOT/f'audit/{phase.upper()}_VALIDATION.json'
    assert not out.exists()
    out.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(value),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['development','main'],required=True)
    validate(parser.parse_args().phase)
