"""Frozen long-output development-only LVSpec selection and CPU score loading.

No GPU/model import. Complete three-repeat runtime grids and source/config
seals are checked before fixed repeat0 scoring. A lambda0/PST policy is never
mislabelled Strict; all failed candidates cause explicit disable_relaxation.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
RATIOS = (0., .3, .5, .7)
DEFAULTS = dict(loose_ratio=.7, similarity_top_n=10, draft_window=10,
                pst_enabled=True, pst_include_root=True, visual_reference='image_tokens',
                stable_rank_ties=True, disable_relaxation=False)
REPEAT_FIELDS = ('raw_token_ids','returned_token_ids','output_text','acceptance_length','acceptance_paths','eos_index','cap_reached')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, obj, preserve=True):
    path=Path(path)
    require(not preserve or not path.exists(), 'Preserve existing evidence: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def sample_id(sample):
    return str(sample.get('question_id',sample.get('sample_id')))


def load_phase(config, config_path, phase):
    split='test' if phase=='main' else 'dev'
    require(config['repetitions']==3,'All benchmark methods require three repetitions')
    all_rows, samples, manifests, differences={}, {}, {}, []
    cfg_sha=sha(config_path)
    for name,expected in config['harness_sha256'].items():
        require(sha(ROOT/'harness'/name)==expected,'Frozen runtime source changed: '+name)
    if config.get('model_identity_path'):
        require(sha(config['model_identity_path'])==config['model_identity_sha256'],'Checkpoint identity manifest changed')
    if config.get('checkout_manifest_sha256'):
        require(sha(ROOT/'audit/CLEAN_CHECKOUT.json')==config['checkout_manifest_sha256'],'Fresh checkout manifest changed')
    for task in config['tasks']:
        samples[task]=read(config['data'][task][split+'_path'])
        require(sha(config['data'][task][split+'_path'])==config['data'][task][split+'_sha256'],'Input SHA '+task)
        require(sha(config['data'][task][split+'_labels_path'])==config['data'][task][split+'_labels_sha256'],'Official label SHA '+task)
        group_count=100 if phase=='main' else 20
        require(len({s['group_id'] for s in samples[task]})==group_count,'Group count '+task)
        methods=config['methods_by_task'][task] if phase=='main' else config['development_methods']
        rows=[]
        for shard in (0,1):
            path=ROOT/'results'/phase/f'{phase}_{task}_shard{shard}.jsonl'
            marker=read(path.with_suffix('.complete.json'))
            require(marker['complete'] is True and sha(path)==marker['result_sha256'],'Complete result SHA '+str(path))
            batch=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            require(marker['records']==len(batch) and marker['metadata']['config_sha256']==cfg_sha,'Complete record/config '+task)
            require(marker['metadata']['source_sha256']==config['harness_sha256'],'Runtime source manifest '+task)
            for row in batch:
                require(row['phase']==phase and row['task']==task and row['config_sha256']==cfg_sha,'Row phase/task/config')
                require(row['shard']==shard==row['group_ordinal']%2,'Shard/group mapping')
                require(row['diagnostic_recording'] is False,'Scoring requires production records')
                sample=samples[task][row['ordinal']]
                require(sample_id(sample)==str(row['question_id']) and sample['group_id']==row['group_id']
                        and sample['image_sha256']==row['image_sha256'],'Sample/record identity')
            rows.extend(batch)
            manifests[str(path)]=sha(path)
        expected={(method,ordinal,rep) for method in methods for ordinal in range(len(samples[task])) for rep in range(3)}
        actual=[(row['method'],row['ordinal'],row['repeat']) for row in rows]
        require(len(actual)==len(set(actual)) and set(actual)==expected,'Complete method/question/repeat grid '+task)
        lookup={(row['method'],row['ordinal'],row['repeat']):row for row in rows}
        for ordinal in range(len(samples[task])):
            paired=[row for row in rows if row['ordinal']==ordinal]
            for field in ('prompt_token_ids','input_tensor_sha256','image_grid_thw','prompt_text'):
                require(all(row[field]==paired[0][field] for row in paired),'Cross-method input mismatch '+task+'/'+str(ordinal)+'/'+field)
        for method in methods:
            for ordinal in range(len(samples[task])):
                for rep in (1,2):
                    changed=[f for f in REPEAT_FIELDS if lookup[method,ordinal,0][f]!=lookup[method,ordinal,rep][f]]
                    if changed:
                        differences.append(dict(task=task,method=method,question_id=sample_id(samples[task][ordinal]),
                                                ordinal=ordinal,repeat=rep,changed_fields=changed))
        all_rows[task]=rows
    return samples,all_rows,manifests,differences


def score_phase(config, samples, rows_by_task, phase, config_path, result_hashes):
    from evaluate_long import score_task, OFFICIAL_SHA256, OFFICIAL_SOURCE
    evaluator_sha=sha(ROOT/'harness/evaluate_long.py')
    parser_freeze=ROOT/'audit/METRIC_PARSER_FREEZE.json'
    require(parser_freeze.exists() and read(parser_freeze)['evaluator_sha256']==evaluator_sha,'Scoring evaluator does not match pre-quality parser freeze')
    require(read(parser_freeze)['original_evaluator_sha256']==OFFICIAL_SHA256==sha(OFFICIAL_SOURCE),'Original sealed official evaluator changed')
    parser_freeze_sha=sha(parser_freeze)
    result={}
    for task in config['tasks']:
        result[task]={}
        methods=config['methods_by_task'][task] if phase=='main' else config['development_methods']
        for method in methods:
            selected=sorted((row for row in rows_by_task[task] if row['method']==method and row['repeat']==0),key=lambda row:row['ordinal'])
            predictions={str(row['question_id']):row['output_text'] for row in selected}
            folder=ROOT/'metrics'/phase/task/method
            seal=dict(config_sha256=sha(config_path),inputs_sha256=config['data'][task][('test' if phase=='main' else 'dev')+'_sha256'],
                result_sha256={path:value for path,value in result_hashes.items() if f'_{task}_shard' in path},
                evaluator_sha256=evaluator_sha,original_evaluator_sha256=OFFICIAL_SHA256,
                metric_parser_freeze_sha256=parser_freeze_sha,quality_repeat=0,
                predictions_sha256=hashlib.sha256(json.dumps(predictions,sort_keys=True,ensure_ascii=False).encode()).hexdigest())
            seal_path=folder/'SCORING_SEAL.json'
            if seal_path.exists():
                existing=read(seal_path)
                require(all(existing.get(key)==value for key,value in seal.items()),'Existing score seal differs '+str(folder))
                require(sha(folder/'aggregate.json')==existing['aggregate_sha256'] and sha(folder/'per_question_scores.json')==existing['per_question_sha256'],'Scored results mutated')
                per_question,aggregate=read(folder/'per_question_scores.json'),read(folder/'aggregate.json')
            else:
                per_question,aggregate=score_task(task,samples[task],predictions,folder)
                write(seal_path,dict(seal,aggregate_sha256=sha(folder/'aggregate.json'),per_question_sha256=sha(folder/'per_question_scores.json')))
            require(len(per_question)==len(samples[task]) and len({str(row['question_id']) for row in per_question})==len(per_question),'Scored grid '+task+'/'+method)
            result[task][method]=dict(rows=per_question,aggregate=aggregate,seal_sha256=sha(seal_path),folder=str(folder))
    return result


def quality_pass(task, candidate, baseline):
    if task=='coco':
        # Long descriptive paragraphs are outside the short-caption length
        # regime of CIDEr-D's sigma6 penalty. Keep exact CIDEr as auxiliary;
        # never let an absolute1e-12 epsilon turn its tiny values into a pass.
        tests={key:(candidate.get(key) is not None and baseline.get(key) is not None) for key in ('SPICE','CHAIRs')}
        if not all(tests.values()):
            return False,dict(status='undefined_required_metric',available=tests)
        conditions=dict(SPICE=candidate['SPICE']+1e-12>=.95*baseline['SPICE'],
                        CHAIRs=candidate['CHAIRs']<=baseline['CHAIRs']+1e-12)
    elif task=='mme':
        conditions={key:candidate[key]+1e-12>=baseline[key] for key in ('MME_subset_score','accuracy')}
    else:
        conditions=dict(accuracy=candidate['accuracy']+1e-12>=baseline['accuracy'])
    return all(conditions.values()),dict(status='defined',conditions=conditions,
        candidate={key:candidate[key] for key in conditions},baseline={key:baseline[key] for key in conditions},floating_roundoff_tolerance=1e-12)


def select_task(task, config, runtime_rows, scores):
    baseline=scores['vispec']['aggregate']
    candidates=[]
    for method,variant in config['development_methods'].items():
        if variant.get('family')!='LVSPEC':
            continue
        ratio=float(variant['loose_ratio'])
        require(ratio in RATIOS and not variant.get('disable_relaxation',False),'Unexpected LVSpec development policy')
        passed,diagnostic=quality_pass(task,scores[method]['aggregate'],baseline)
        batch=[row for row in runtime_rows if row['method']==method]
        # Equal repetitions per question: mean all requests == question means.
        seconds=statistics.mean(row['seconds'] for row in batch)
        candidates.append(dict(method=method,loose_ratio=ratio,qualitypass=passed,
                               mean_seconds=seconds,diagnostic=diagnostic,score_seal_sha256=scores[method]['seal_sha256']))
    require(sorted(row['loose_ratio'] for row in candidates)==list(RATIOS),'LVSpec grid must be exactly0/.3/.5/.7')
    paper=next(row for row in candidates if row['loose_ratio']==.7)
    if paper['qualitypass']:
        chosen=paper
        reason='paper_ratio_passes_keep_paper_before_speed_selection'
    else:
        eligible=[row for row in candidates if row['loose_ratio']<.7 and row['qualitypass']]
        chosen=min(eligible,key=lambda row:(row['mean_seconds'],row['loose_ratio'])) if eligible else None
        reason='lower_ratio_quality_eligible_fastest' if chosen else 'all_four_fail_explicit_strict_fallback'
    if chosen is None:
        params=dict(DEFAULTS,disable_relaxation=True)
        selected='vispec'
        label='LVSpec-ViSpec-adapted-ratio (explicit Strict fallback; no LV relaxation)'
    else:
        params=dict(DEFAULTS,loose_ratio=chosen['loose_ratio'])
        selected=chosen['method']
        label='LVSpec-ViSpec-adapted-ratio'
    return dict(parameters=params,selecteddevmethod=selected,qualitypass=bool(chosen),reason=reason,label=label,
                strict_fallback=chosen is None,paper_qualitypass=paper['qualitypass'],candidates=candidates,
                quality_constraint_is_proxy_not_population_noninferiority=True)


def self_test():
    require(quality_pass('gqa',{'accuracy':1.},{'accuracy':1.})[0],'EqualQA score must pass')
    require(not quality_pass('textvqa',{'accuracy':.9},{'accuracy':1.})[0],'QA decline must fail')
    require(not quality_pass('mme',{'MME_subset_score':200.,'accuracy':.5},{'MME_subset_score':100.,'accuracy':.6})[0],'MME question accuracy gate independent')
    require(quality_pass('coco',{'CIDEr':.95,'SPICE':.95,'CHAIRs':.1},{'CIDEr':1.,'SPICE':1.,'CHAIRs':.1})[0],'COCO95percent boundary')
    require(quality_pass('coco',{'CIDEr':0.,'SPICE':.95,'CHAIRs':.1},{'CIDEr':1.,'SPICE':1.,'CHAIRs':.1})[0],'Long-output CIDEr is auxiliary, never a development gate')
    require(quality_pass('coco',{'CIDEr':1e-100,'SPICE':.95,'CHAIRs':.1},{'CIDEr':1e-30,'SPICE':1.,'CHAIRs':.1})[0],'Tiny CIDEr does not invoke an absolute epsilon gate')
    require(not quality_pass('coco',{'CIDEr':1.,'SPICE':None,'CHAIRs':0.},{'CIDEr':1.,'SPICE':1.,'CHAIRs':.1})[0],'UndefinedSPICE cannot pass')
    methods={f'lv{ratio}':dict(family='LVSPEC',loose_ratio=ratio) for ratio in RATIOS}
    config=dict(development_methods=methods)
    runtime=[dict(method=name,seconds=(2. if ratio==.7 else 1.)) for name,ratio in zip(methods,RATIOS)]
    def scores(values):
        result={'vispec':dict(aggregate={'accuracy':1.})}
        result.update({name:dict(aggregate={'accuracy':value},seal_sha256='test-only') for name,value in zip(methods,values)})
        return result
    selected=select_task('gqa',config,runtime,scores([1,1,1,1]))
    require(selected['parameters']['loose_ratio']==.7,'Paper-first takes precedence over speed')
    selected=select_task('gqa',config,runtime,scores([0,1,1,0]))
    require(selected['parameters']['loose_ratio']==.3,'Equal timing chooses smaller eligible lower ratio')
    selected=select_task('gqa',config,runtime,scores([0,0,0,0]))
    require(selected['parameters']['disable_relaxation'] is True and selected['strict_fallback'] and selected['selecteddevmethod']=='vispec','Allfail mustdisable, notlambda0/PST')
    print('CPU LVSpec selection self-tests passed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/design_config.json')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test();return
    cfg=read(args.config)
    require(not (ROOT/'configs/lvspec_selection.json').exists(),'Selection already frozen')
    samples,rows,hashes,repeat_differences=load_phase(cfg,args.config,'development')
    scores=score_phase(cfg,samples,rows,'development',args.config,hashes)
    selections={task:select_task(task,cfg,rows[task],scores[task]) for task in cfg['tasks']}
    payload=dict(status='development_only_selection',ready_for_main=not repeat_differences,
        design_config_sha256=sha(args.config),selector_sha256=sha(__file__),evaluator_sha256=sha(ROOT/'harness/evaluate_long.py'),
        result_sha256=hashes,all_methods_repetitions=3,quality_repeat=0,
        metric_parser_freeze_sha256=sha(ROOT/'audit/METRIC_PARSER_FREEZE.json'),
        selections=selections,repeat_differences=repeat_differences,
        rule='Paper.7 passes=>keep; otherwise fastest eligible lower ratio; exact timing tie smaller ratio; all fail=>explicit disable_relaxation Strict fallback',
        coco_development_gate='SPICE >=0.95*Strict and CHAIRs <=Strict; full-output CIDEr auxiliary, not a gate',
        lambda_zero_with_PST_is_not_strict=True,main_outputs_used=False,ground_truth_or_human_safety_claim=False)
    write(ROOT/'configs/lvspec_selection.json',payload)
    print(json.dumps(dict(status=payload['status'],ready_for_main=payload['ready_for_main'],
          selections={task:{key:value[key] for key in ('selecteddevmethod','parameters','qualitypass','strict_fallback')} for task,value in selections.items()}),ensure_ascii=False))
    if repeat_differences:
        raise SystemExit('Development repetition differences retained; MAIN No-Go until investigated')


if __name__=='__main__':
    main()
