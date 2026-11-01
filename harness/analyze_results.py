"""Task-specific paired image/group statistics for the clean ViSpec benchmark.

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
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from select_lvspec import ROOT, load_phase, score_phase, read, sha, write, require, sample_id

PRIMARY='bepc_algorithm'
METHOD_ORDER=('ar','vispec','strict_shared','bepc_algorithm','margin4_ccve','lvspec_paper','lvspec_adapted')
TASK_TITLES=dict(mme='MME subset',gqa='GQA subset',textvqa='TextVQA subset',scienceqa='ScienceQA image subset',coco='COCO Caption subset')
LOWER_BETTER={'mean_seconds','mean_decode_seconds','CHAIRs','CHAIRi','mean_peak_allocated_bytes'}


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
        from evaluate_tasks import MME_CATEGORIES
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
    times=[];decodes=[];tokens=[];completion_tokens=[];accepted=[];rounds=[];peaks=[];raw_tokens=[];truncated=[];preprocessing=[]
    for ordinal in range(len(samples)):
        batch=[runtime[ordinal,rep] for rep in range(3)]
        times.append(np.mean([row['seconds'] for row in batch]))
        decodes.append(np.mean([row['decode_seconds'] for row in batch]))
        tokens.append(np.mean([len(row['returned_token_ids']) for row in batch]))
        completion_tokens.append(np.mean([row['completion_tokens'] for row in batch]))
        accepted.append(np.mean([row['effective_accepted_draft_tokens'] for row in batch]))
        rounds.append(np.mean([row['effective_verification_rounds'] for row in batch]))
        peaks.append(np.mean([row['peak_allocated_bytes'] for row in batch]))
        raw_tokens.append(np.mean([len(row['raw_token_ids']) for row in batch]))
        truncated.append(np.mean([row['cap_reached'] for row in batch]))
        preprocessing.append(np.mean([row.get('input_preprocessing_seconds',np.nan) for row in batch]))
    ratio_metric('mean_seconds',times,[1]*len(samples))
    ratio_metric('mean_decode_seconds',decodes,[1]*len(samples))
    ratio_metric('throughput_tokens_per_second',tokens,times)
    ratio_metric('decode_throughput_tokens_per_second',tokens,decodes)
    ratio_metric('mean_peak_allocated_bytes',peaks,[1]*len(samples))
    if method=='ar':
        vectors['effective_mean_accepted_draft_length']=unavailable(draws,'AR_has_no_Draft_acceptance')
    else:ratio_metric('effective_mean_accepted_draft_length',accepted,rounds)
    aggregate=scored['aggregate']
    for key in ('accuracy','MME_subset_score','Perception_subset_score','Cognition_subset_score','acc_plus','CIDEr','SPICE','CHAIRs','CHAIRi'):
        if key in aggregate and key in vectors:
            a,b=aggregate[key],vectors[key]['point']
            require((a is None and b is None) or (a is not None and b is not None and abs(a-b)<1e-8),'Independent aggregate disagrees with official scorer '+task+'/'+method+'/'+key)
    summary=dict(task=task,method=method,images_or_groups=n,questions=len(samples),requests=len(samples)*3,
        mean_visible_tokens=float(np.mean(tokens)),visible_tokens_exclude_EOS=True,mean_raw_tokens=float(np.mean(raw_tokens)),
        mean_completion_tokens_including_EOS=float(np.mean(completion_tokens)),
        completion_tokens_including_EOS_throughput_per_second=float(np.sum(completion_tokens)/np.sum(times)),
        completion_tokens_including_EOS_decode_throughput_per_second=float(np.sum(completion_tokens)/np.sum(decodes)),
        truncation_fraction=float(np.mean(truncated)),parse_invalid_count=aggregate['parse_invalid_count'],
        parse_uncertain_count=aggregate['uncertain_count'],quality_repeat=0,
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
    p=float((1+np.count_nonzero(np.abs(boot-delta)>=abs(delta)-1e-15))/(len(boot)+1))
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
    samples=[dict(question_id=str(i),group_id='g'+str(i//2),category='existence') for i in range(4)]
    questions=[dict(question_id=str(i),score=score,category='existence') for i,score in enumerate([1,1,1,0])]
    aggregate=dict(accuracy=.75,MME_subset_score=125.,Perception_subset_score=125.,Cognition_subset_score=0.,acc_plus=.5,parse_invalid_count=0,uncertain_count=0)
    runtime=[]
    for i in range(4):
        for rep in range(3):
            runtime.append(dict(method='vispec',ordinal=i,repeat=rep,seconds=1.,decode_seconds=.5,returned_token_ids=[2],effective_accepted_draft_tokens=1,effective_verification_rounds=2,peak_allocated_bytes=10,peak_reserved_bytes=20,peak_increment_bytes=3,peak_reserved_increment_bytes=4,raw_token_ids=[2,1],completion_tokens=2,cap_reached=False,input_preprocessing_seconds=.1))
    groups,mapping,categories=group_structure(samples)
    mme_draws=bootstrap_indices(groups,categories,5000,19,True)
    vectors,_=method_metrics('mme','vispec',samples,runtime,dict(rows=questions,aggregate=aggregate,seal_sha256='test-only'),mme_draws,groups,mapping,categories)
    require(vectors['MME_subset_score']['point']==125. and vectors['accuracy']['point']==.75 and vectors['acc_plus']['point']==.5,'MME complete-pair/category formula')
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
    manifest=dict(status='completed_clean_five_task_paired_analysis',created_utc=datetime.now(timezone.utc).isoformat(),
        config_sha256=sha(config_path),analyzer_sha256=sha(__file__),selector_sha256=sha(ROOT/'harness/select_lvspec.py'),
        evaluator_sha256=sha(ROOT/'harness/evaluate_tasks.py'),result_sha256=hashes,
        immutable_runtime_sources=config['harness_sha256'],
        runtime_identity={key:config.get(key) for key in ('upstream','upstream_commit','target','draft','processor','tree','temperature','max_new_tokens','max_length','model_identity_path','model_identity_sha256','checkout_manifest_sha256')},
        selection_sha256=sha(ROOT/'configs/lvspec_selection.json'),frozen_data_sources=config['data'],
        scoring_seals={task:{method:entry['seal_sha256'] for method,entry in batch.items()} for task,batch in scores.items()},
        measured_requests=sum(len(batch) for batch in rows.values()),bootstrap=5000,bootstrap_unit='physical image/group',
        MME_stratified_by_category=True,draws=manifests,quality_repeat=0,
        repeat_differences=repeat_differences,repeat_determinism_passed=not repeat_differences,
        p_method='two-sided centered paired-bootstrap add-one Monte Carlo approximation',
        multiple_comparisons='Holm within each task/metric: BEPCvs6 family; separate LVadaptvs2 family; unavailable tests retain original family size',
        prespecified_comparisons=config['comparisons'],prespecified_secondary_comparisons=config['secondary_comparisons'],
        numerical_environment=dict(python=sys.version,numpy=np.__version__),
        CI_scope='pointwise percentile95%; fixed reference metric per-image scores; no global across-task/multi-metric correction',
        automatic_quality_does_not_establish_human_caption_factual_safety=True)
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
    return 'NA' if value is None else f'{value*scale:.{digits}f}'


def report(config,summaries,comparisons,manifest):
    selection=read(ROOT/'configs/lvspec_selection.json')
    lines=['# 干净 ViSpec 五任务解码实测', '',
        '本报告使用新 clone ViSpec、冻结的任务短答案/单句 Caption 提示和独立开发/测试划分。'
        '每任务100组测试、每方法3次完整生成；MME每组保留原始两个问题。全部成绩属于固定子集，不能写成完整官方split成绩。', '',
        '日期：2026-10-31（Asia/Shanghai）。MME历史曝光与另发现的任务前缀ID别名均保留，冻结输入未重抽样。'
        '所有方法仍在同一批图像上配对；历史曝光与本轮dev/test独立是不同问题。'
        '因此不能宣称全部测试图为项目未见，统计与质量不以历史曝光状态筛样。'
        '以下数量动态读取 audit/DATA_HISTORICAL_OVERLAP_SUPPLEMENT.json，其effective字段覆盖冻结输入中的原freshness标记。', '',
        f'完成模型请求 {manifest["measured_requests"]} 次；跨重复轨迹差异 {len(manifest["repeat_differences"])} 对。'
        '质量固定使用repeat0，未挑选较优重复。所有失败、解析失败、截断和官方未定义指标保留。', '',
        '原生ViSpec保留原完整leaf logits展开和最后未使用Draft构树；strict_shared对照才享用共享工程优化。'
        'BEPC为wR=1、tau1、2/4 mismatch预算；Margin4+CCVE同预算。LVSpec没有2/4 cap，采用每合法branch相对视觉排名与PST，不能将其结果解释为严格预算匹配消融。', '']
    environments=list(manifest['actual_runtime_environments'].values())
    lines += ['实际运行配置：Target为Qwen2.5-VL-7B-Instruct，Draft为对应ViSpec checkpoint；'
              + '；'.join(f'GPU{entry["physical_gpu"]}: {entry["gpu"]}, {entry["target_dtype"]}, PyTorch {entry["torch"]}, CUDA {entry["cuda"]}' for entry in environments)
              + '。每个请求只在一张GPU运行，两卡仅分片不同image/group。', '',
              f'温度={config["temperature"]}，max_new_tokens={config["max_new_tokens"]}，max_length={config["max_length"]}；'
              f'原Draft树配置={json.dumps(config["tree"],ensure_ascii=False)}。'
              'QA短答案与Caption单句提示、图像处理及模型来源见冻结配置和data各split文件；本轮未因任务质量结果修改提示或模型。', '']
    lines += ['AR基线是Target原生串行loop，复用正确的custom KV与mRoPE，不调用Draft。'
              '外置runner修正了上游AR辅助脚本new_token未递增、EOS后仍进行无用Target调用的问题；upstream源码没有修改。'
              'AR显存仍包含本进程常驻但未使用的Draft模型，因此不是Target-only生产部署的footprint。', '']
    history=read(ROOT/'audit/DATA_HISTORICAL_OVERLAP_SUPPLEMENT.json')
    lines += ['| 任务 | 开发无已知历史曝光/组数 | 测试无已知历史曝光/组数 |','|---|---:|---:|']
    for task in config['tasks']:
        dev=history['effective_group_counts'][task]['dev'];test=history['effective_group_counts'][task]['test']
        lines.append(f'| {task} | {dev["effective_no_recorded_history_overlap_groups"]}/{dev["groups"]} | {test["effective_no_recorded_history_overlap_groups"]}/{test["groups"]} |')
    lines += ['', str(history['dev_test_image_independence']), str(history['strict_project_history_unseen_claim']),
              '上述补充审计未使用历史质量/时延调整样本、答案或标签；新图泛化主张为No-Go，不妨碍本轮内部配对工程比较。', '']
    for task in config['tasks']:
        choice=selection['selections'][task]
        lines += [f'## {TASK_TITLES[task]}', '',
            f'LVSpec开发选择：`{choice["selecteddevmethod"]}`；lambda={choice["parameters"]["loose_ratio"]}，'
            f'disable_relaxation={choice["parameters"]["disable_relaxation"]}。'
            +('这是显式Strict fallback，不能作为成功宽松策略。' if choice['parameters']['disable_relaxation'] else '测试集未改变该配置。'), '']
        if choice['paper_qualitypass']:
            lines += ['开发规则保留论文比例，因此paper与adapted两臂的接受参数相同；它们不能被解释为两个不同算法，其时延差异反映独立请求的测量波动。', '']
        quality_headers=({'mme':['MME总分','感知','认知','Acc%'], 'coco':['CIDEr-D×100','SPICE×100','CHAIRs%']} .get(task,['Accuracy%']))
        headers=['方法']+quality_headers+['模型请求ms [95%CI]','相对原生速度 [95%CI]','Draft/轮','峰值allocated GiB','峰值reserved GiB','无效解析']
        lines += ['| '+' | '.join(headers)+' |','| '+' | '.join(['---']+['---:']*(len(headers)-1))+' |']
        for row in [r for r in summaries if r['task']==task]:
            quality=([f(row['MME_subset_score']),f(row['Perception_subset_score']),f(row['Cognition_subset_score']),f(row['accuracy'],scale=100)] if task=='mme' else
                     [f(row['CIDEr'],scale=100),f(row['SPICE'],scale=100),f(row['CHAIRs'],scale=100)] if task=='coco' else [f(row['accuracy'],scale=100)])
            latency=f'{f(row["mean_seconds"],scale=1000)} [{f(row["mean_seconds_ci95_low"],scale=1000)}, {f(row["mean_seconds_ci95_high"],scale=1000)}]'
            speed=f'{f(row["speedup_vs_native"])} [{f(row["speedup_vs_native_ci95_low"])}, {f(row["speedup_vs_native_ci95_high"])}]'
            line=[row['method']]+quality+[latency,speed,f(row['effective_mean_accepted_draft_length']),f(row['max_peak_allocated_bytes'],scale=1/2**30),f(row['max_peak_reserved_bytes'],scale=1/2**30),str(row['parse_invalid_count'])]
            lines.append('| '+' | '.join(line)+' |')
        primary=[row for row in comparisons if row['task']==task and row['family']=='BEPC_six_prespecified' and row['metric']=='mean_seconds']
        lines += ['', 'BEPC相对其余六方法的时延比较：'+
            '；'.join(f'{row["baseline"]} {f(row.get("latency_speedup"),3)}×，Holm p={f(row.get("holm_adjusted_p"),4)}' for row in primary)+'。', '']
        lines += ['实际宽松暴露与输出长度（响应比例以3次生成的全部请求为分母；计数仅指最终提交的mismatch）：', '',
                  '| 方法 | 含committed mismatch的响应% | committed mismatch/响应 | 可见token/响应 |',
                  '|---|---:|---:|---:|']
        for row in [r for r in summaries if r['task']==task]:
            lines.append('| '+' | '.join([row['method'],f(row['response_fraction_with_committed_mismatch'],scale=100),
                f(row['mean_committed_mismatches']),f(row['mean_visible_tokens'])])+' |')
        if task in ('mme','gqa','scienceqa'):
            lines += ['', '这些短答案的首token是各方法共同的严格Target root。零或极少committed mismatch时，'
                      '相同任务质量只能表明当前提示和长度设置下无观测下降；它没有充分暴露宽松门，不能证明门的判别力或事实安全。', '']
        lines += ['', 'AR的完整模型请求可因短响应而快于投机方法；接受Draft长度提高也不保证请求提速。'
                  '上表保留AR及所有慢于基线的负结果，不按接受长度替代实际计时排序。', '']
    lines += ['## 统计口径与局限', '',
        '- 每问题先平均3次时延；在相同image/group抽样索引下配对bootstrap5000次。MME按category分层、完整双问题group重采样，MME总分及分域分数按官方category公式重算。',
        '- 表中95%CI是点式区间。六个BEPC比较与两个LV适配比较分别按task/metric作Holm校正；不控制所有任务和所有指标合并后的全局错误率。p为中心化配对bootstrap的双侧Monte Carlo近似，并非精确随机化检验。',
        '- Accuracy分母包括全部问题；不把不同任务Accuracy、MME量纲或COCO指标混合平均。TextVQA使用官方十答案VQA-style分数；MME总分为所含category的100×(acc+acc_plus)之和，14类别上限2800。',
        '- COCO使用当前100图完整参考语料计算一次正式CIDEr-D，bootstrap重采样其固定每图得分；未在每个bootstrap内重新拟合document frequency。开发20与测试100的CIDEr绝对值不跨split比较。SPICE或CHAIRi未定义时保留NA，不能用替代指标或删除样本制造结果。',
        '- [COCO开发指标独立复核](../audit/COCO_DEVELOPMENT_METRIC_CROSSCHECK.md)确认真实image ID、每图5条参考和作者原始CIDEr实现逐图结果一致，无重复缩放或截断错误。开发生成与参考词数不同只是描述性关联；没有据此将低分因果归于提示词、长度或小样本DF。',
        '- token/s使用首EOS前、128-budget内可见文本token数，不含EOS；模型请求分母包含controller/vision/prefill/Draft/Target/在线特征。CSV另存包含EOS的completion吞吐，供与前轮口径对照。PIL/processor/transfer单独计时并跨方法复用，不属于表中模型请求延迟；该列不是完整用户端E2E。decode-only口径和输出长度见CSV。',
        '- Draft/轮为有效接受Draft token总数/有效验证轮数，未加每轮root token；AR为NA。显存列是观测最大值，不附伪精确最大值置信区间；AR包括常驻未使用Draft模型。reserved受CUDA allocator历史影响，不是方法独占需求；allocated/reserved增量另存CSV，不通过每请求empty_cache改变口径。',
        '- 原生ViSpec与strict_shared的差异反映公共工程优化；不能把宽松方法相对原生的全部收益归因于信号或算法。LVSpec-paper-ratio仍是ViSpec短树/静态图像适配，未宣称复现原K10视频/H200设定。',
        '- 每任务100组是Pilot。CI条件于当前固定硬件、每问题3次计时均值和当前参考指标；MME各category仅约7/8组，不能据此证明强非劣效或罕见事实错误安全。20组开发的非下降代理门不证明总体非劣效，测试差异不显著也不等于等效；本轮没有完成有功效保证的非劣效检验。Caption CIDEr/SPICE/CHAIR不能证明人工事实或显著遗漏安全。历史曝光数量和别名例外动态来自补充审计；各任务来源/缺图/排除数量见数据审计。',
        '- 全部方法、全部预设配对和负结果保存在CSV；没有根据测试结果选择风险方向、重新调lambda、选择更快重复、训练分类器或融合信号。', '',
        '实现与数据依据：[冻结实验协议](../EXPERIMENT_PROTOCOL.md)、[LVSpec移植审计](LVSPEC_IMPLEMENTATION_AUDIT.md)、'
        '[官方指标实现审计](METRIC_AUDIT.md)、[数据审计](../audit/DATA_AUDIT.md)。', '',
        '输出：`BENCHMARK_SUMMARY.csv`、`PAIRED_COMPARISONS.csv`、`BENCHMARK_PERFORMANCE.pdf`、`BENCHMARK_QUALITY.pdf`；配置、输入、评分、源码和原始结果SHA见 `audit/STATISTICAL_ANALYSIS.json` 与各方法评分seal。', '']
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
    with PdfPages(ROOT/'reports/BENCHMARK_PERFORMANCE.pdf') as pdf:
        fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
        for ax,task in zip(axes.flat,config['tasks']):
            batch=[r for r in summaries if r['task']==task]
            x=np.arange(len(batch));y=np.array([r['mean_seconds']*1000 for r in batch])
            lo=np.array([r['mean_seconds_ci95_low']*1000 for r in batch]);hi=np.array([r['mean_seconds_ci95_high']*1000 for r in batch])
            ax.bar(x,y,color=['#C6CCD2' if r['method']!='bepc_algorithm' else '#3B7DAF' for r in batch])
            ax.errorbar(x,y,yerr=np.maximum(0,np.vstack((y-lo,hi-y))),fmt='none',color='black',capsize=3,lw=1)
            ax.set_xticks(x,[titles[r['method']] for r in batch],rotation=55,ha='right',fontsize=8)
            ax.set_title(TASK_TITLES[task]);ax.set_ylabel('Model request latency (ms)');ax.grid(axis='y',alpha=.2)
        axes.flat[-1].axis('off');fig.suptitle('Fixed subsets: mean latency and paired image/group bootstrap 95% CI')
        pdf.savefig(fig);plt.close(fig)
    with PdfPages(ROOT/'reports/BENCHMARK_QUALITY.pdf') as pdf:
        fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
        for ax,task in zip(axes.flat,config['tasks']):
            metric='MME_subset_score' if task=='mme' else 'CIDEr' if task=='coco' else 'accuracy'
            scale=1 if task=='mme' else 100
            batch=[r for r in summaries if r['task']==task];x=np.arange(len(batch))
            for i,row in enumerate(batch):
                if row.get(metric) is None:continue
                y=row[metric]*scale;lo=row.get(metric+'_ci95_low');hi=row.get(metric+'_ci95_high')
                ax.scatter(i,y,color='#3B7DAF' if row['method']=='bepc_algorithm' else '#697681')
                if lo is not None:ax.errorbar(i,y,yerr=np.maximum(0,np.array([[y-lo*scale],[hi*scale-y]])),color='black',capsize=3,lw=1)
            ax.set_xticks(x,[titles[r['method']] for r in batch],rotation=55,ha='right',fontsize=8)
            ax.set_title(TASK_TITLES[task]);ax.set_ylabel('MME subset score' if task=='mme' else 'CIDEr-D x100' if task=='coco' else 'Official accuracy (%)')
            ax.grid(axis='y',alpha=.2)
            if task=='mme':ax.set_ylim(0,2800)
            elif task=='coco':ax.set_ylim(bottom=0)
            else:ax.set_ylim(0,100)
        axes.flat[-1].axis('off');fig.suptitle('Official task metrics on fixed subsets; no cross-task averaging')
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
