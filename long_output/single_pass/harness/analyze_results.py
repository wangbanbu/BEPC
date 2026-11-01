"""Paired image/group statistics for the long-output clean ViSpec benchmark.

CPU only. Quality uses official score_task and the predeclared repeat0.
5000 shared paired bootstrap draws per task; MME draws complete image-pairs
within category. No pooled cross-task Accuracy, test tuning, Judge or classifier.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from select_lvspec import ROOT, load_phase, score_phase, read, sha, write, require, sample_id

PRIMARY='bepc_algorithm'
METHOD_ORDER=('ar','vispec','strict_shared','bepc_algorithm','margin4_ccve','lvspec_paper','lvspec_adapted')
TASK_TITLES=dict(mme='MME subset',gqa='GQA subset',textvqa='TextVQA subset',scienceqa='ScienceQA image subset',coco='COCO Caption subset')
LOWER_BETTER={'mean_seconds','mean_decode_seconds','CHAIRs','CHAIRi','mean_peak_allocated_bytes',
              'answer_extraction_failure_fraction','parse_uncertain_fraction','truncation_fraction'}


def csv_write(path, rows):
    path=Path(path)
    require(not path.exists(),'Preserve existing CSV: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader();writer.writerows(rows)


def bootstrap_indices(groups, categories, iterations, seed, stratified=False):
    rng=np.random.default_rng(seed)
    if not stratified:
        return rng.integers(0,len(groups),size=(iterations,len(groups)),dtype=np.int64)
    blocks=[]
    for category in sorted(set(categories)):
        indices=np.asarray([index for index,c in enumerate(categories) if c==category],dtype=np.int64)
        blocks.append(indices[rng.integers(0,len(indices),size=(iterations,len(indices)))])
    return np.concatenate(blocks,axis=1)


def holm(values):
    # Unavailable prespecified tests count as p1 in the family, preserving its
    # original size; their own adjusted p remains unavailable.
    order=sorted(range(len(values)),key=lambda i:1. if values[i] is None else values[i])
    result=[None]*len(values);running=0.
    for rank,index in enumerate(order):
        p=1. if values[index] is None else values[index]
        running=max(running,min(1.,(len(values)-rank)*p))
        if values[index] is not None:result[index]=running
    return result


def evaluate_vector(numerator, denominator, draws):
    numerator=np.asarray(numerator,dtype=np.float64)
    if denominator is None:
        point=float(numerator.sum());replicates=numerator[draws].sum(1)
    else:
        denominator=np.asarray(denominator,dtype=np.float64)
        total=denominator.sum()
        point=float(numerator.sum()/total) if total>0 else None
        den=denominator[draws].sum(1)
        replicates=np.divide(numerator[draws].sum(1),den,out=np.full(len(draws),np.nan),where=den>0)
    finite=np.isfinite(replicates)
    # Do not silently drop undefined replicates to manufacture an interval.
    interval=np.quantile(replicates,[.025,.975]).tolist() if finite.all() and point is not None else [None,None]
    return dict(point=point,ci95=interval,replicates=replicates,undefined_bootstrap=int((~finite).sum()))


def unavailable(draws, reason):
    return dict(point=None,ci95=[None,None],replicates=np.full(len(draws),np.nan),undefined_bootstrap=len(draws),reason=reason)


def group_structure(samples):
    groups=list(dict.fromkeys(str(sample['group_id']) for sample in samples))
    mapping={group:index for index,group in enumerate(groups)}
    categories=[]
    for group in groups:
        values={sample.get('category','all') for sample in samples if str(sample['group_id'])==group}
        require(len(values)==1,'One image group appears in different category strata')
        categories.append(next(iter(values)))
    return groups,mapping,categories


def method_metrics(task, method, samples, runtime_rows, scored, draws, groups, group_map, categories):
    n=len(groups)
    runtime={(row['ordinal'],row['repeat']):row for row in runtime_rows if row['method']==method}
    quality={str(row['question_id']):row for row in scored['rows']}
    question_groups=[group_map[str(sample['group_id'])] for sample in samples]
    vectors={}
    def sums(values):
        out=np.zeros(n)
        for group,value in zip(question_groups,values):out[group]+=value
        return out
    def ratio_metric(name, numerator, denominator):
        vectors[name]=evaluate_vector(sums(numerator),sums(denominator),draws)
    qs=[quality[sample_id(sample)] for sample in samples]
    if task=='coco':
        for metric in ('CIDEr','SPICE','CHAIRs'):
            values=[row.get(metric) for row in qs]
            if any(value is None for value in values):
                vectors[metric]=unavailable(draws,'undefined_official_per_caption_metric')
            else:ratio_metric(metric,values,[1]*len(qs))
        ratio_metric('CHAIRi',[row['hallucinated_mentions'] for row in qs],[row['object_mentions'] for row in qs])
    else:
        ratio_metric('accuracy',[row['score'] for row in qs],[1]*len(qs))
    if task=='mme':
        category_counts=Counter(categories)
        contributions=np.zeros(n);pairs=np.zeros(n)
        from evaluate_long import official
        MME_CATEGORIES=official().MME_CATEGORIES
        for group_id,index in group_map.items():
            batch=[quality[sample_id(sample)] for sample in samples if str(sample['group_id'])==group_id]
            require(len(batch)==2 and all(row['category']==categories[index] for row in batch),'MME complete category-image pair')
            pair=int(all(row['score']==1 for row in batch))
            pairs[index]=pair
            contributions[index]=100*(sum(row['score'] for row in batch)/2+pair)/category_counts[categories[index]]
        vectors['MME_subset_score']=evaluate_vector(contributions,None,draws)
        for title,keep in MME_CATEGORIES.items():
            vector=contributions*np.asarray([cat in keep for cat in categories])
            vectors[title+'_subset_score']=evaluate_vector(vector,None,draws)
        vectors['acc_plus']=evaluate_vector(pairs,np.ones(n),draws)
    times=[];decodes=[];tokens=[];completion_tokens=[];accepted=[];rounds=[];native_accepted=[];native_rounds=[]
    peaks=[];raw_tokens=[];truncated=[];preprocessing=[];words=[]
    for ordinal in range(len(samples)):
        batch=[runtime[ordinal,0]]
        times.append(np.mean([row['seconds'] for row in batch]))
        decodes.append(np.mean([row['decode_seconds'] for row in batch]))
        tokens.append(np.mean([len(row['returned_token_ids']) for row in batch]))
        completion_tokens.append(np.mean([row['completion_tokens'] for row in batch]))
        accepted.append(np.mean([row['effective_accepted_draft_tokens'] for row in batch]))
        rounds.append(np.mean([row['effective_verification_rounds'] for row in batch]))
        native_accepted.append(np.mean([row['raw_accepted_draft_tokens'] for row in batch]))
        native_rounds.append(np.mean([row['raw_verification_rounds'] for row in batch]))
        peaks.append(np.mean([row['peak_allocated_bytes'] for row in batch]))
        raw_tokens.append(np.mean([len(row['raw_token_ids']) for row in batch]))
        truncated.append(np.mean([row['cap_reached'] for row in batch]))
        preprocessing.append(np.mean([row.get('input_preprocessing_seconds',np.nan) for row in batch]))
        words.append(np.mean([len(row['output_text'].split()) for row in batch]))
    ratio_metric('mean_seconds',times,[1]*len(samples))
    ratio_metric('mean_decode_seconds',decodes,[1]*len(samples))
    ratio_metric('throughput_tokens_per_second',tokens,times)
    ratio_metric('decode_throughput_tokens_per_second',tokens,decodes)
    ratio_metric('mean_peak_allocated_bytes',peaks,[1]*len(samples))
    ratio_metric('mean_output_words',words,[1]*len(samples))
    ratio_metric('truncation_fraction',truncated,[1]*len(samples))
    ratio_metric('parse_uncertain_fraction',[int(row.get('uncertain',False)) for row in qs],[1]*len(samples))
    if task=='coco':
        vectors['answer_extraction_failure_fraction']=unavailable(draws,'COCO_full_description_has_no_final_answer_extraction')
    else:
        ratio_metric('answer_extraction_failure_fraction',
                     [int(row.get('answer_extraction_status') in ('invalid','uncertain')) for row in qs], [1]*len(samples))
    if method=='ar':
        vectors['native_mean_accepted_draft_length']=unavailable(draws,'AR_has_no_Draft_acceptance')
        vectors['effective_mean_accepted_draft_length']=unavailable(draws,'AR_has_no_Draft_acceptance')
        vectors['effective_mean_committed_tokens_per_round']=unavailable(draws,'AR_has_no_Draft_acceptance')
    else:
        ratio_metric('native_mean_accepted_draft_length',native_accepted,native_rounds)
        ratio_metric('effective_mean_accepted_draft_length',accepted,rounds)
        ratio_metric('effective_mean_committed_tokens_per_round',completion_tokens,rounds)
    aggregate=scored['aggregate']
    for key in ('accuracy','MME_subset_score','Perception_subset_score','Cognition_subset_score','acc_plus','CIDEr','SPICE','CHAIRs','CHAIRi'):
        if key in aggregate and key in vectors:
            a,b=aggregate[key],vectors[key]['point']
            tolerance=0. if key=='CIDEr' else 1e-8
            require((a is None and b is None) or (a is not None and b is not None and math.isclose(a,b,rel_tol=1e-12,abs_tol=tolerance)),'Independent aggregate disagrees with official scorer '+task+'/'+method+'/'+key)
    summary=dict(task=task,method=method,images_or_groups=n,questions=len(samples),requests=len(samples),
        repetitions=1,timing_repeatability='not_assessed_single_pass',
        mean_visible_tokens=float(np.mean(tokens)),visible_tokens_exclude_EOS=True,mean_raw_tokens=float(np.mean(raw_tokens)),
        mean_completion_tokens_including_EOS=float(np.mean(completion_tokens)),
        completion_tokens_including_EOS_throughput_per_second=float(np.sum(completion_tokens)/np.sum(times)),
        completion_tokens_including_EOS_decode_throughput_per_second=float(np.sum(completion_tokens)/np.sum(decodes)),
        truncation_fraction=float(np.mean(truncated)),parse_invalid_count=aggregate['parse_invalid_count'],
        parse_uncertain_count=aggregate['uncertain_count'],quality_repeat=0,
        answer_extraction_failures=aggregate.get('answer_extraction_failures',0),
        answer_extraction_counts=json.dumps(aggregate.get('answer_extraction_counts',{}),sort_keys=True),
        max_peak_allocated_bytes=max(row['peak_allocated_bytes'] for row in runtime.values()),
        max_peak_reserved_bytes=max(row['peak_reserved_bytes'] for row in runtime.values()),
        max_peak_increment_bytes=max(row['peak_increment_bytes'] for row in runtime.values()),
        max_peak_reserved_increment_bytes=max(row['peak_reserved_increment_bytes'] for row in runtime.values()),
        mean_recorded_preprocessing_seconds=(float(np.mean(preprocessing)) if np.isfinite(preprocessing).all() else None),
        preprocessing_reused_across_methods=True,scoring_seal_sha256=scored['seal_sha256'],
        mean_committed_mismatches=float(np.mean([row.get('signal_stats',{}).get('committed_mismatches',0) for row in runtime.values()])),
        response_fraction_with_committed_mismatch=float(np.mean([row.get('signal_stats',{}).get('committed_mismatches',0)>0 for row in runtime.values()])),
        mean_lvspec_visual_only_relaxations=float(np.mean([row.get('signal_stats',{}).get('selected_visual_relaxations',0) for row in runtime.values()])),
        mean_lvspec_pst_only_relaxations=float(np.mean([row.get('signal_stats',{}).get('selected_pst_relaxations',0) for row in runtime.values()])),
        mean_lvspec_visual_and_pst_relaxations=float(np.mean([row.get('signal_stats',{}).get('selected_visual_and_pst_relaxations',0) for row in runtime.values()])))
    for name,result in vectors.items():
        summary[name]=result['point'];summary[name+'_ci95_low'],summary[name+'_ci95_high']=result['ci95']
        summary[name+'_undefined_bootstrap']=result['undefined_bootstrap']
    summary['mat_native_draft_only']=summary['native_mean_accepted_draft_length']
    summary['mat_effective_draft_only']=summary['effective_mean_accepted_draft_length']
    summary['mat_effective_with_root']=summary['effective_mean_committed_tokens_per_round']
    return vectors,summary


def paired_comparison(task, method, baseline, metric, one, two, family):
    a,b=one[metric],two[metric]
    item=dict(task=task,method=method,baseline=baseline,metric=metric,family=family,
              lower_is_better=metric in LOWER_BETTER,method_point=a['point'],baseline_point=b['point'])
    if a['point'] is None or b['point'] is None or not np.isfinite(a['replicates']).all() or not np.isfinite(b['replicates']).all():
        return dict(item,status='unavailable',difference=None,ci95_low=None,ci95_high=None,p=None)
    delta=a['point']-b['point'];boot=a['replicates']-b['replicates']
    lo,hi=np.quantile(boot,[.025,.975])
    # Shift the paired empirical bootstrap to a zero-effect null. Approximate,
    # two-sided, add-one Monte Carlo p; not an exact randomization test.
    # Scale-relative floating tolerance preserves tests for tiny nonzero
    # long-description CIDEr-D scores; a fixed1e-15 would flatten them to p1.
    centered=np.abs(boot-delta)
    threshold=max(0.,abs(delta)-8*np.finfo(float).eps*max(abs(delta),float(centered.max())))
    p=float((1+np.count_nonzero(centered>=threshold))/(len(boot)+1))
    result=dict(item,status='ok',difference=delta,ci95_low=float(lo),ci95_high=float(hi),p=p)
    if metric=='mean_seconds':
        ratio=b['replicates']/a['replicates']
        rlo,rhi=np.quantile(ratio,[.025,.975])
        result.update(latency_speedup=b['point']/a['point'],latency_speedup_ci95_low=float(rlo),latency_speedup_ci95_high=float(rhi))
    return result


def self_test():
    groups=['a','b','c','d'];cats=['x','x','y','y']
    draws=bootstrap_indices(groups,cats,5000,31,True)
    require(draws.shape==(5000,4) and np.all((draws<2).sum(1)==2),'MME strata draw sizes')
    value=evaluate_vector([1,2,3,4],[2,2,2,2],draws)
    require(abs(value['point']-1.25)<1e-12,'Ratio aggregate')
    require(holm([.01,.04,.03])==[.03,.06,.06],'Holm known example')
    require(holm([.01,None])[0]==.02,'Unavailable tests retain family size')
    constant=evaluate_vector([1,1,1,1],[1,1,1,1],draws)
    pair=paired_comparison('x','a','b','accuracy',{'accuracy':constant},{'accuracy':constant},'test')
    require(pair['p']==1 and pair['difference']==0 and pair['ci95_low']==pair['ci95_high']==0,'Identical paired control')
    double=evaluate_vector([2,2,2,2],[1,1,1,1],draws)
    tiny_one=evaluate_vector([1e-100]*4,[1]*4,draws)
    tiny_two=evaluate_vector([2e-100]*4,[1]*4,draws)
    ordinary=paired_comparison('x','a','b','CIDEr',{'CIDEr':double},{'CIDEr':constant},'test')
    tiny=paired_comparison('x','a','b','CIDEr',{'CIDEr':tiny_two},{'CIDEr':tiny_one},'test')
    require(tiny['p']==ordinary['p'] and tiny['p']<.001,'Bootstrap p is scale invariant for tiny nonzero CIDEr')
    require(evaluate_vector([0,9],[1,3],np.array([[0,1]]))['point']==2.25,'Pooled MAT must not equal unweighted request MAT mean1.5')
    require('e-' in f(1e-100,6),'Tiny nonzero CIDEr remains visible in report')
    samples=[dict(question_id=str(i),group_id='g'+str(i//2),category='existence') for i in range(4)]
    questions=[dict(question_id=str(i),score=score,category='existence') for i,score in enumerate([1,1,1,0])]
    aggregate=dict(accuracy=.75,MME_subset_score=125.,Perception_subset_score=125.,Cognition_subset_score=0.,acc_plus=.5,parse_invalid_count=0,uncertain_count=0)
    runtime=[]
    for i in range(4):
        for rep in range(1):
            runtime.append(dict(method='vispec',ordinal=i,repeat=rep,seconds=1.,decode_seconds=.5,returned_token_ids=[2],effective_accepted_draft_tokens=1,effective_verification_rounds=1,raw_accepted_draft_tokens=3,raw_verification_rounds=2,output_text='A blue object',peak_allocated_bytes=10,peak_reserved_bytes=20,peak_increment_bytes=3,peak_reserved_increment_bytes=4,raw_token_ids=[2,1],completion_tokens=2,cap_reached=False,input_preprocessing_seconds=.1))
    groups,mapping,categories=group_structure(samples)
    mme_draws=bootstrap_indices(groups,categories,5000,19,True)
    vectors,_=method_metrics('mme','vispec',samples,runtime,dict(rows=questions,aggregate=aggregate,seal_sha256='test-only'),mme_draws,groups,mapping,categories)
    require(vectors['MME_subset_score']['point']==125. and vectors['accuracy']['point']==.75 and vectors['acc_plus']['point']==.5,'MME complete-pair/category formula')
    require(vectors['native_mean_accepted_draft_length']['point']==1.5 and vectors['effective_mean_accepted_draft_length']['point']==1. and vectors['effective_mean_committed_tokens_per_round']['point']==2.,'Native/effective Draft MAT and root inclusion remain distinct')
    require(vectors['mean_output_words']['point']==3.,'Output length is recorded descriptively')
    zero=evaluate_vector([0,1,0],[1,9,0],np.asarray([[0,0,0],[2,2,2]]))
    require(zero['point']==.1 and zero['undefined_bootstrap']==1 and zero['ci95']==[None,None],'Ratio denominators cannot be replaced by caption means or undefined draws dropped')
    print('CPU statistical self-tests passed')


def analyze(config,config_path,samples,rows,scores,hashes,repeat_differences):
    require(config['bootstrap']==5000,'Predeclared5000 draws')
    summaries=[];comparisons=[];manifests={};all_vectors={}
    for task_index,task in enumerate(config['tasks']):
        groups,mapping,categories=group_structure(samples[task])
        seed=int(config['bootstrap_seed'])+task_index*100003
        draws=bootstrap_indices(groups,categories,5000,seed,task=='mme')
        manifests[task]=dict(seed=seed,draw_sha256=hashlib.sha256(draws.tobytes()).hexdigest(),shape=list(draws.shape),
                            groups=groups,categories=categories,stratified=task=='mme',stratum_counts=dict(Counter(categories)))
        all_vectors[task]={}
        methods=config['methods_by_task'][task]
        for method in METHOD_ORDER:
            require(method in methods,'Predeclared7 main arms')
            vectors,summary=method_metrics(task,method,samples[task],rows[task],scores[task][method],draws,groups,mapping,categories)
            variant=methods[method]
            summary.update(loose_ratio=variant.get('loose_ratio'),disable_relaxation=variant.get('disable_relaxation',False),
                           residual_attenuation=variant.get('residual_attenuation'),tau=variant.get('tau'))
            all_vectors[task][method]=vectors;summaries.append(summary)
        metric_names=list(all_vectors[task][PRIMARY])
        for metric in metric_names:
            primary=[]
            for baseline in METHOD_ORDER:
                if baseline==PRIMARY:continue
                primary.append(paired_comparison(task,PRIMARY,baseline,metric,all_vectors[task][PRIMARY],all_vectors[task][baseline],'BEPC_six_prespecified'))
            for row,p in zip(primary,holm([row['p'] for row in primary])):row['holm_adjusted_p']=p
            comparisons.extend(primary)
            secondary=[paired_comparison(task,method,baseline,metric,all_vectors[task][method],all_vectors[task][baseline],'LV_adaptation_two_prespecified')
                       for method,baseline in config['secondary_comparisons']]
            require(len(secondary)==2,'Exactly two prespecified LV-adaptation comparisons')
            for row,p in zip(secondary,holm([row['p'] for row in secondary])):row['holm_adjusted_p']=p
            comparisons.extend(secondary)
        # Descriptive paired speed vs native for every arm; never choose winner.
        for summary in [row for row in summaries if row['task']==task]:
            a=all_vectors[task][summary['method']]['mean_seconds'];b=all_vectors[task]['vispec']['mean_seconds']
            lo,hi=np.quantile(b['replicates']/a['replicates'],[.025,.975])
            summary.update(speedup_vs_native=b['point']/a['point'],speedup_vs_native_ci95_low=float(lo),speedup_vs_native_ci95_high=float(hi))
    manifest=dict(status='completed_single_pass_long_output_clean_five_task_paired_analysis',created_utc=datetime.now(timezone.utc).isoformat(),
        config_sha256=sha(config_path),analyzer_sha256=sha(__file__),selector_sha256=sha(ROOT/'harness/select_lvspec.py'),
        evaluator_sha256=sha(ROOT/'harness/evaluate_long.py'),result_sha256=hashes,
        immutable_runtime_sources=config['harness_sha256'],
        runtime_identity={key:config.get(key) for key in ('upstream','upstream_commit','target','draft','processor','tree','temperature','max_new_tokens','max_length','model_identity_path','model_identity_sha256','checkout_manifest_sha256')},
        selection_sha256=sha(ROOT/'configs/lvspec_selection.json'),frozen_data_sources=config['data'],
        scoring_seals={task:{method:entry['seal_sha256'] for method,entry in batch.items()} for task,batch in scores.items()},
        measured_requests=sum(len(batch) for batch in rows.values()),bootstrap=5000,bootstrap_unit='physical image/group',
        MME_stratified_by_category=True,draws=manifests,quality_repeat=0,
        repetitions=1,repeat_differences=repeat_differences,repeat_determinism_passed=None,
        repeated_run_determinism='not_assessed_single_pass',
        inherited_repeat0_records=sum(bool(row.get('origin') or row.get('import_provenance')) for batch in rows.values() for row in batch),
        p_method='two-sided centered paired-bootstrap add-one Monte Carlo approximation',
        multiple_comparisons='Holm within each task/metric: BEPCvs6 family; separate LVadaptvs2 family; unavailable tests retain original family size',
        prespecified_comparisons=config['comparisons'],prespecified_secondary_comparisons=config['secondary_comparisons'],
        numerical_environment=dict(python=sys.version,numpy=np.__version__),
        CI_scope='pointwise percentile95%; image/group sampling only; no repeated-run timing variability; fixed reference metric per-image scores; no global across-task/multi-metric correction',
        automatic_quality_does_not_establish_human_caption_factual_safety=True)
    from evaluate_long import OFFICIAL_SHA256
    manifest['original_evaluator_sha256']=OFFICIAL_SHA256
    manifest['MAT_definition']='native/effective accepted Draft tokens divided by corresponding pooled rounds; auxiliary with-root =1+effective; AR NA'
    manifest['QA_quality_scope']='unique terminal Final answer task accuracy; explanation factuality/reasoning not assessed'
    manifest['coco_development_gate']='SPICE >=0.95*Strict and CHAIRs <=Strict; CIDEr auxiliary only'
    manifest['supplementary_data_audit_sha256']={str(path):sha(path) for path in (ROOT/'audit').glob('DATA*json')}
    crosscheck=ROOT/'audit/COCO_DEVELOPMENT_METRIC_CROSSCHECK.md'
    manifest['coco_development_metric_crosscheck_sha256']=sha(crosscheck) if crosscheck.exists() else None
    manifest['metric_parser_freeze_sha256']=sha(ROOT/'audit/METRIC_PARSER_FREEZE.json')
    manifest['actual_runtime_environments']={}
    for shard in (0,1):
        path=ROOT/f'audit/main_shard{shard}.loaded.json'
        loaded=read(path)
        manifest['actual_runtime_environments'][str(shard)]=dict(
            source_path=str(path),source_sha256=sha(path),
            **{key:loaded.get(key) for key in ('physical_gpu','gpu','torch','cuda','target_dtype','executable','AR_memory_includes_idle_loaded_Draft')})
    history=read(ROOT/'audit/DATA_HISTORICAL_OVERLAP_SUPPLEMENT.json')
    manifest['effective_historical_freshness_counts']=history['effective_group_counts']
    manifest['project_history_unseen_claim']=history['strict_project_history_unseen_claim']
    return summaries,comparisons,manifest


def f(value,digits=2,scale=1):
    return 'NA' if value is None else (f'{value*scale:.{max(3,digits)}e}' if 0<abs(value*scale)<10**(-digits) else f'{value*scale:.{digits}f}')


def report(config,summaries,comparisons,manifest):
    selection=read(ROOT/'configs/lvspec_selection.json')
    lines=['# 长输出 ViSpec 五任务单次测试完整实验报告', '',
        '本轮在相同图像和问题上，将提示改为约100–160词解释/详细图像描述，最大新token预算512。'
        '五个任务各20组开发、100组测试，按用户修订每种方法仅完成1次生成；MME每图保留两问。'
        '本报告首先比较内容token/s、原生/有效Draft MAT和官方任务质量，同时保留请求时延、长度、显存、格式失败和负结果。', '',
        '日期：2026-11-01（Asia/Shanghai）。使用前轮fresh GitHub clone和相同模型、数据身份与来源划分；'
        '仅长提示和预算改变，开发集重新按同一网格选择LVSpec参数。BEPC/CCVE阈值未重调。'
        '所有成绩属于固定子集，不能写成完整官方split或独立全新图像的成绩。', '',
        f'完成测试模型请求 {manifest["measured_requests"]} 次；不执行repeat1/2。'
        '任务质量使用唯一的repeat0；全部失败、截断、无效格式及未定义官方指标均保留。'
        '由于每个配置只测一次，本轮没有重复运行稳定性证据；时延和吞吐置信区间只反映图像抽样，不包含重复计时波动。', '',
        'QA先生成解释，再以唯一的末行Final answer给出最终答案；固定格式提取后调用原官方任务指标。'
        '因此QA结果衡量最终答案正确性，无法证明解释正确或推理忠实。COCO整段原样计算SPICE、CIDEr-D、CHAIR，'
        '它们是辅助自动质量/对象幻觉指标，不能代替人工事实与信息覆盖评价。', '']
    environments=list(manifest['actual_runtime_environments'].values())
    lines += ['实际运行：Target=Qwen2.5-VL-7B-Instruct，Draft=对应ViSpec checkpoint；'
        + '；'.join(f'GPU{entry["physical_gpu"]}: {entry["gpu"]}, {entry["target_dtype"]}, PyTorch {entry["torch"]}, CUDA {entry["cuda"]}' for entry in environments)
        + '。单请求在一张GPU执行，两卡仅分片不同image/group。', '',
        f'温度={config["temperature"]}，max_new_tokens={config["max_new_tokens"]}，max_length={config["max_length"]}；'
        f'Draft树={json.dumps(config["tree"],ensure_ascii=False)}。精确文本、输入及源码SHA见冻结配置和协议。', '',
        '原生ViSpec保留原完整leaf logits展开和最后未使用Draft构树；strict_shared才应用公共工程优化。'
        'BEPC固定wR=1、tau1、每轮/每响应2/4 mismatch预算；Margin4+CCVE同预算。'
        'LVSpec采用合法branch相对视觉排名和PST，没有相同2/4 cap；这些差别不能作为预算匹配消融解释。', '',
        'AR使用同一Target custom KV/mRoPE的串行loop，不调用Draft；其显存仍包括常驻但未用的Draft权重，'
        '因此不是纯Target生产部署footprint。原短实验已记录AR与native Strict的17题输出差异，'
        '本轮保留实际输出，不能假设两个执行图在BF16下逐token等价。', '']
    history=read(ROOT/'audit/DATA_HISTORICAL_OVERLAP_SUPPLEMENT.json')
    lines += ['| 任务 | 开发无已知历史曝光/组数 | 测试无已知历史曝光/组数 |','|---|---:|---:|']
    for task in config['tasks']:
        dev=history['effective_group_counts'][task]['dev'];test=history['effective_group_counts'][task]['test']
        lines.append(f'| {task} | {dev["effective_no_recorded_history_overlap_groups"]}/{dev["groups"]} | {test["effective_no_recorded_history_overlap_groups"]}/{test["groups"]} |')
    lines += ['', '这沿用原历史审计相对短实验启动前的状态；现在全部图像已参加短实验，不能再宣称项目未见。'
        '本轮开发与测试图像仍分离，历史曝光不用于筛样或改变统计分母。', '']
    for task in config['tasks']:
        choice=selection['selections'][task]
        batch=[row for row in summaries if row['task']==task]
        lines += [f'## {TASK_TITLES[task]}', '',
            f'长输出开发选择：{choice["selecteddevmethod"]}；lambda={choice["parameters"]["loose_ratio"]}，'
            f'disable_relaxation={choice["parameters"]["disable_relaxation"]}。'
            +('这是显式Strict fallback，不能称为成功宽松策略。' if choice['parameters']['disable_relaxation'] else '测试集未改变配置。'), '']
        if choice['paper_qualitypass']:
            lines += ['paper与adapted因预先的paper优先规则使用相同接受参数；它们是独立计时重复臂，不能将时延波动解释为算法差异。', '']
        if task=='coco':
            lines += ['开发质量门预先固定为SPICE≥0.95×Strict且CHAIRs≤Strict。CIDEr-D完整保留但不参与开发门：'
                '短参考的sigma6长度惩罚可将长段落分数压到极小值，不能用绝对epsilon将它视为相等。'
                '这里的门仍只是20图自动代理，不能证明总体或人工质量非劣效。', '']
        headers=['方法','内容token/s [95%CI]','原生Draft MAT','有效Draft MAT','1+有效MAT','模型请求ms [95%CI]','最大allocated GiB']
        lines += ['| '+' | '.join(headers)+' |','| '+' | '.join(['---']+['---:']*(len(headers)-1))+' |']
        for row in batch:
            throughput=f'{f(row["throughput_tokens_per_second"])} [{f(row["throughput_tokens_per_second_ci95_low"])}, {f(row["throughput_tokens_per_second_ci95_high"])}]'
            latency=f'{f(row["mean_seconds"],scale=1000)} [{f(row["mean_seconds_ci95_low"],scale=1000)}, {f(row["mean_seconds_ci95_high"],scale=1000)}]'
            line=[row['method'],throughput,f(row['native_mean_accepted_draft_length']),
                f(row['effective_mean_accepted_draft_length']),f(row['effective_mean_committed_tokens_per_round']),
                latency,f(row['max_peak_allocated_bytes'],scale=1/2**30)]
            lines.append('| '+' | '.join(line)+' |')
        if task=='coco':
            headers=['方法','CIDEr-D原尺度','SPICE%','CHAIRs%','CHAIRi%']
        elif task=='mme':
            headers=['方法','MME子集总分','感知','认知','答案Accuracy%']
        else:
            headers=['方法','最终答案Accuracy%']
        lines += ['', '| '+' | '.join(headers)+' |','| '+' | '.join(['---']+['---:']*(len(headers)-1))+' |']
        for row in batch:
            quality=([f(row['MME_subset_score']),f(row['Perception_subset_score']),f(row['Cognition_subset_score']),f(row['accuracy'],scale=100)] if task=='mme' else
                [f(row['CIDEr'],digits=6),f(row['SPICE'],scale=100),f(row['CHAIRs'],scale=100),f(row['CHAIRi'],scale=100)] if task=='coco' else [f(row['accuracy'],scale=100)])
            lines.append('| '+' | '.join([row['method']]+quality)+' |')
        lines += ['', '| 方法 | 可见token/响应 | 输出词/响应 | 触顶% | 答案提取失败/题 | 全部invalid/uncertain | 含committed mismatch响应% |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for row in batch:
            extraction='NA' if task=='coco' else str(row['answer_extraction_failures'])+'/'+str(row['questions'])
            line=[row['method'],f(row['mean_visible_tokens']),f(row['mean_output_words']),f(row['truncation_fraction'],scale=100),
                extraction,str(row['parse_invalid_count'])+'/'+str(row['parse_uncertain_count']),f(row['response_fraction_with_committed_mismatch'],scale=100)]
            lines.append('| '+' | '.join(line)+' |')
        primary=[row for row in comparisons if row['task']==task and row['family']=='BEPC_six_prespecified' and row['metric']=='mean_seconds']
        lines += ['', 'BEPC相对其余六方法的实际请求时延比（>1为BEPC快）：'+
            '；'.join(f'{row["baseline"]} {f(row.get("latency_speedup"),3)}×，Holm p={f(row.get("holm_adjusted_p"),4)}' for row in primary)+'。', '',
            'token/s提高不必然代表每题更快：宽松方法可能生成更长文本或更容易触顶。'
            '请将token/s、整题时延、长度、截断与质量一起比较；接受长度本身不能证明用户等待时间缩短。', '']
    lines += ['## 统计方法与结论边界', '',
        '- 每问题只有一次测量，在image/group上配对bootstrap5000次。MME在category内保留完整两问group重采样。时延与吞吐CI仅反映这些图像的抽样变化，不能估计重复运行、GPU状态和计时噪声；质量使用唯一的repeat0。',
        '- 主token/s=可见内容token总数/同步模型请求秒数，排除EOS；模型请求包括controller构造、视觉编码、prefill、Target/Draft及在线特征。CSV另存decode-only和含EOS completion吞吐。PIL/processor/H2D独立计时且方法间复用，因此不是完整客户端E2E。',
        '- 原生Draft MAT=全部原生接受Draft token总数/全部原生验证轮数，可能包含最后block在EOS或预算后的计算。有效Draft MAT按统一首EOS/512预算裁切；另报1+有效MAT表示含每轮root的推进长度。三者均为pooled轮数比率，AR全部NA；不平均每请求MAT。',
        '- 所有点式95%CI条件于当前子集/硬件和固定官方每图质量分数。六个BEPC比较与两个LV比较分别按task/metric作Holm校正；不控制跨全部任务和指标的全局错误率。中心化paired-bootstrap p是Monte Carlo近似，不是精确随机化检验。',
        '- QA最终答案通过冻结末行格式提取，全部格式失败、uncertain和越界保留并计0；Acc不评价解释中的事实或逻辑。TextVQA执行官方十答案leave-one-out归一化，GQA保留exact equality。MME每类100×(acc+acc_plus)求和，14类上限2800，不是完整官方split。',
        '- COCO整段原样计算；CIDEr-D仅作辅助，以科学计数保留极小非零值。参考只有短caption，CIDEr长度惩罚、SPICE参考覆盖不足及CHAIR对象机会数随长度变化会影响比较。自动指标不构成人类事实/覆盖Ground Truth。CIDEr DF使用当前完整100图参考，bootstrap不重新拟合DF；开发与测试绝对CIDEr不可跨split比较。',
        '- 未定义SPICE/CHAIRi保留NA，不删除事件或填0；不按输出长度、格式成功、历史曝光或预期质量筛样。词数100–160仅是提示要求，没有最小生成强制或长度有效性过滤。',
        '- 显存最大值是观测峰值，AR含常驻未用Draft；reserved受allocator历史影响，不是独占需求。显存增量另存CSV，不执行每请求empty_cache。',
        '- 原生与共享Strict控制公共工程收益。LVSpec是原论文相对视觉/PST在ViSpec短树及静态图像上的适配，不是原视频/H200/K10配置的完整复现。100图及20图开发代理不足以证明总体非劣效或罕见幻觉安全；未训练、融合、利用测试质量调参或选择更快重复。', '',
        '依据：[冻结实验协议](../EXPERIMENT_PROTOCOL.md)、[质量与解析协议](QUALITY_PROTOCOL.md)、[MAT/长提示/AR对齐审计](../audit/MAT_AND_LONG_PROMPT_AUDIT.md)。', '',
        '机器可读结果：BENCHMARK_SUMMARY.csv、PAIRED_COMPARISONS.csv；图：BENCHMARK_PERFORMANCE.pdf（吞吐、时延、MAT）和BENCHMARK_QUALITY.pdf。'
        '全部配置、输入、结果、官方评分来源和统计SHA见audit/STATISTICAL_ANALYSIS.json及各方法SCORING_SEAL.json。', '']
    return '\n'.join(lines)


def plots(config,summaries):
    if importlib.util.find_spec('matplotlib') is None:
        plotting_python=Path('/home/wbb/miniconda3/bin/python')
        require(plotting_python.exists(),'Configured CPU Matplotlib Python missing')
        # Official scores/statistics stay in the verified scoring environment;
        # only already-computed numeric figure data enter the plotting process.
        with tempfile.TemporaryDirectory(prefix='plot-only-',dir=ROOT/'audit') as folder:
            payload=Path(folder)/'figures.json'
            payload.write_text(json.dumps(dict(config=config,summaries=summaries),allow_nan=False))
            subprocess.run([str(plotting_python),str(Path(__file__).resolve()),'--plot-payload',str(payload)],check=True)
        return str(plotting_python)
    _plots_inline(config,summaries)
    return sys.executable


def _plots_inline(config,summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    titles={m:m.replace('lvspec_','LV ').replace('bepc_','BEPC ').replace('_',' ') for m in METHOD_ORDER}
    colors=['#C6CCD2' if m!='bepc_algorithm' else '#3B7DAF' for m in METHOD_ORDER]
    def style(ax,task,batch,label):
        x=np.arange(len(batch))
        ax.set_xticks(x,[titles[r['method']] for r in batch],rotation=55,ha='right',fontsize=8)
        ax.set_title(TASK_TITLES[task]);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
    with PdfPages(ROOT/'reports/BENCHMARK_PERFORMANCE.pdf') as pdf:
        for metric,scale,label in [('throughput_tokens_per_second',1,'Visible content tokens/s (full model request)'),
                                   ('mean_seconds',1000,'Model request latency (ms)')]:
            fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
            for ax,task in zip(axes.flat,config['tasks']):
                batch=[r for r in summaries if r['task']==task];x=np.arange(len(batch))
                y=np.array([r[metric]*scale for r in batch])
                lo=np.array([r[metric+'_ci95_low']*scale for r in batch]);hi=np.array([r[metric+'_ci95_high']*scale for r in batch])
                ax.bar(x,y,color=colors)
                ax.errorbar(x,y,yerr=np.maximum(0,np.vstack((y-lo,hi-y))),fmt='none',color='black',capsize=3,lw=1)
                style(ax,task,batch,label)
            axes.flat[-1].axis('off')
            fig.suptitle('Single pass / long output / 512-token budget: image/group sampling 95% CI')
            pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
        for ax,task in zip(axes.flat,config['tasks']):
            batch=[r for r in summaries if r['task']==task];x=np.arange(len(batch))
            for offset,metric,title,color in [(-.18,'native_mean_accepted_draft_length','Native Draft MAT','#98A8B8'),
                                            (.18,'effective_mean_accepted_draft_length','Effective Draft MAT','#3B7DAF')]:
                valid=[i for i,row in enumerate(batch) if row[metric] is not None]
                y=np.array([batch[i][metric] for i in valid])
                lo=np.array([batch[i][metric+'_ci95_low'] for i in valid]);hi=np.array([batch[i][metric+'_ci95_high'] for i in valid])
                ax.bar(np.asarray(valid)+offset,y,width=.36,color=color,label=title)
                ax.errorbar(np.asarray(valid)+offset,y,yerr=np.maximum(0,np.vstack((y-lo,hi-y))),fmt='none',color='black',capsize=2,lw=1)
            style(ax,task,batch,'Accepted Draft tokens / verification round')
            ax.legend(fontsize=8);ax.text(0,.02,'AR: NA',transform=ax.get_xaxis_transform(),ha='center',fontsize=8)
        axes.flat[-1].axis('off');fig.suptitle('Pooled native/effective Draft MAT; each round root excluded')
        pdf.savefig(fig);plt.close(fig)
    with PdfPages(ROOT/'reports/BENCHMARK_QUALITY.pdf') as pdf:
        fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
        for ax,task in zip(axes.flat,config['tasks']):
            metric='MME_subset_score' if task=='mme' else 'SPICE' if task=='coco' else 'accuracy'
            scale=1 if task=='mme' else 100
            batch=[r for r in summaries if r['task']==task]
            for i,row in enumerate(batch):
                if row.get(metric) is None:continue
                y=row[metric]*scale;lo=row.get(metric+'_ci95_low');hi=row.get(metric+'_ci95_high')
                ax.scatter(i,y,color='#3B7DAF' if row['method']=='bepc_algorithm' else '#697681')
                if lo is not None:ax.errorbar(i,y,yerr=np.maximum(0,np.array([[y-lo*scale],[hi*scale-y]])),color='black',capsize=3,lw=1)
            style(ax,task,batch,'MME subset score' if task=='mme' else 'Full-description SPICE (%)' if task=='coco' else 'Extracted final-answer accuracy (%)')
            ax.set_ylim(0,2800 if task=='mme' else 100)
        axes.flat[-1].axis('off');fig.suptitle('Long-output automatic task metrics; explanation factuality not established')
        pdf.savefig(fig);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('all','score','analyze'),default='all')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/frozen_config.json')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--plot-payload',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.self_test:self_test();return
    if args.plot_payload:
        payload=read(args.plot_payload);_plots_inline(payload['config'],payload['summaries']);return
    cfg=read(args.config)
    require(cfg['config_finalized_before_generation'] is True,'Main config must be frozen')
    samples,rows,hashes,repeats=load_phase(cfg,args.config,'main')
    scores=score_phase(cfg,samples,rows,'main',args.config,hashes)
    if args.stage=='score':
        print(json.dumps(dict(status='official_main_scores_complete',requests=sum(len(x) for x in rows.values()))));return
    summaries,comparisons,manifest=analyze(cfg,args.config,samples,rows,scores,hashes,repeats)
    output=ROOT/'reports'
    targets=[output/'BENCHMARK_SUMMARY.csv',output/'PAIRED_COMPARISONS.csv',output/'BENCHMARK_REPORT.md',output/'BENCHMARK_PERFORMANCE.pdf',output/'BENCHMARK_QUALITY.pdf',ROOT/'audit/STATISTICAL_ANALYSIS.json']
    require(not any(path.exists() for path in targets),'Preserve existing final analysis outputs')
    output.mkdir(parents=True,exist_ok=True)
    csv_write(targets[0],summaries);csv_write(targets[1],comparisons)
    targets[2].write_text(report(cfg,summaries,comparisons,manifest))
    manifest['plotting_python']=plots(cfg,summaries)
    manifest['output_sha256']={str(path):sha(path) for path in targets[:-1]}
    write(targets[-1],manifest)
    print(json.dumps(dict(status=manifest['status'],requests=manifest['measured_requests'],repeat_differences=len(repeats),outputs=[str(p) for p in targets])))


if __name__=='__main__':
    main()
