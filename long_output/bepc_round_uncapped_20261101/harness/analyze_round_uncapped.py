"""Pre-frozen paired analysis: BEPC round2 versus no mismatch count caps.

All scores, thresholds, prompts, trees and stopping rules remain fixed. Only the
per-round mismatch count limit changes; neither arm has a response count cap.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np

from evaluate_round_uncapped import (ROOT, PREVIOUS, COMPARISON, UNCAPPED, METHODS,
    TASKS, require, read, sha, write_once, load_phase, score_phase, sample_id)

LEGACY_ANALYZER = PREVIOUS / 'harness/analyze_results.py'
CONTROL, NEW = METHODS
NI_ARITHMETIC_TOLERANCE = 1e-12
TITLES = {'scienceqa': 'ScienceQA', 'coco': 'COCO Caption', 'textvqa': 'TextVQA'}
PARITY_FIELDS = ('raw_token_ids', 'returned_token_ids', 'output_text',
    'acceptance_length', 'acceptance_paths', 'eos_index', 'cap_reached')
PATH_FIELDS = ('row', 'accepted_draft', 'mismatches', 'nodes', 'tokens')


def legacy_math():
    previous = list(sys.path)
    sys.path.insert(0, str(LEGACY_ANALYZER.parent))
    try:
        spec = importlib.util.spec_from_file_location('_read_only_prior_statistical_math', LEGACY_ANALYZER)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    finally:
        sys.path[:] = previous
    return module


def csv_once(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=sorted({key for row in rows for key in row}))
    writer.writeheader(); writer.writerows(rows); payload = buffer.getvalue()
    require(not path.exists() or path.read_text() == payload.replace('\r\n', '\n'), 'Preserve CSV: '+str(path))
    if not path.exists(): path.write_text(payload, encoding='utf-8')


def numeric_stats(mapping, prefix=''):
    result = {}
    for key, value in mapping.items():
        name = prefix + key
        if isinstance(value, dict): result.update(numeric_stats(value, name+'.'))
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value): result[name] = value
    return result


def runtime_diagnostics(task, method, rows):
    selected = sorted((r for r in rows if r['method']==method), key=lambda r:r['ordinal'])
    counts, paths, numeric, affected = [], [], [], []
    old_binding = []
    for row in selected:
        stats = row.get('signal_stats', {})
        count = int(stats.get('committed_mismatches', stats.get('committed_response_mismatches', 0)))
        selected_paths = [int(p.get('mismatches', 0)) for p in row['acceptance_paths']]
        require(all(0 <= n <= p['accepted_draft'] for n,p in zip(selected_paths,row['acceptance_paths'])),
                'Mismatch count must fit selected Draft tokens')
        require(sum(selected_paths)==count, 'Selected path and response mismatch counts differ')
        require(stats.get('response_cap_enabled') is False and stats.get('response_mismatch_budget') is None,
                'Both methods explicitly disable response cap')
        if method==CONTROL:
            require(stats.get('round_mismatch_budget')==2, 'Control remains round2')
            require(all(n<=2 for n in selected_paths), 'Control per-round budget violation')
        else:
            require(stats.get('round_mismatch_budget') is None and stats.get('round_cap_enabled') is False,
                    'New policy explicitly disables round cap')
            require(stats.get('row_budget_binding',0)==0, 'Removed round cap cannot bind')
        counts.append(count); paths.extend(selected_paths); numeric.append(numeric_stats(stats))
        affected.append(any(n>2 for n in selected_paths)); old_binding.append(stats.get('row_budget_binding',0))
    costs = []
    for key in sorted(set().union(*(r.keys() for r in numeric))):
        present = [r[key] for r in numeric if key in r]
        costs.append(dict(task=task, method=method, counter=key, present_questions=len(present), total_questions=len(selected),
            sum=float(sum(present)), mean=float(np.mean(present)), maximum=float(max(present)),
            interpretation='diagnostic counter; not separately synchronized GPU feature latency'))
    summary = dict(committed_mismatch_p50=float(np.quantile(counts,.5)), committed_mismatch_p95=float(np.quantile(counts,.95)),
        committed_mismatch_max=max(counts), response_fraction_above_four_mismatches=float(np.mean(np.asarray(counts)>4)),
        actual_per_round_mismatch_max=max(paths,default=0), selected_rounds=len(paths),
        selected_rounds_above_two_mismatches=sum(n>2 for n in paths),
        selected_round_fraction_above_two_mismatches=float(np.mean(np.asarray(paths)>2)) if paths else 0.,
        responses_with_round_above_two_mismatches=sum(affected),
        response_fraction_with_round_above_two_mismatches=float(np.mean(affected)),
        per_round_budget_violations=sum(n>2 for n in paths) if method==CONTROL else 0,
        round_budget_binding_candidate_visits=int(sum(old_binding)),
        responses_with_round_budget_binding=sum(n>0 for n in old_binding),
        binding_counter_scope='scanned candidate visits; not unique natural rejected events or selected rounds')
    return summary, costs


def noninferiority(task, vectors):
    one, two = vectors[NEW], vectors[CONTROL]
    conditions = []
    if task != 'coco':
        lower = float(np.quantile(one['accuracy']['replicates']-two['accuracy']['replicates'], .05))
        conditions.append(dict(metric='accuracy', difference=one['accuracy']['point']-two['accuracy']['point'],
            bound_type='one_sided_95_lower', bound=lower, permitted_decline=.02,
            proxy_pass=lower >= -.02-NI_ARITHMETIC_TOLERANCE))
    else:
        for key,multiplier,limit,upper in (('SPICE',.95,0.,False),('CHAIRs',1.,.02,True)):
            a,b=one[key],two[key]
            if a['point'] is None or b['point'] is None:
                conditions.append(dict(metric=key,proxy_pass=False,status='undefined_required_metric')); continue
            bound=float(np.quantile(a['replicates']-multiplier*b['replicates'],.95 if upper else .05))
            conditions.append(dict(metric=key,difference=a['point']-b['point'],baseline_multiplier=multiplier,
                bound_type='one_sided_95_upper' if upper else 'one_sided_95_lower',bound=bound,
                permitted_difference=limit,proxy_pass=bound <= limit+NI_ARITHMETIC_TOLERANCE if upper else bound >= limit-NI_ARITHMETIC_TOLERANCE))
    return dict(task=task,method=NEW,baseline=CONTROL,conditions=conditions,
        automatic_proxy_pass=all(r['proxy_pass'] for r in conditions),
        arithmetic_boundary_tolerance=NI_ARITHMETIC_TOLERANCE,raw_bounds_preserved=True,
        scope='predeclared fixed-pilot automatic-metric diagnostic; not human factual quality guarantee',
        zero_discordance_bootstrap_can_be_degenerate=True)


def historical_parity(samples, runtime):
    result={}
    for task in samples:
        history={}
        for shard in (0,1):
            path=COMPARISON/'results/main'/f'main_{task}_shard{shard}.jsonl'
            require(read(path.with_suffix('.complete.json'))['result_sha256']==sha(path),'Historical sealed result changed')
            for line in path.read_text().splitlines():
                row=json.loads(line)
                if row['method']=='bepc_no_response_cap':history[str(row['question_id'])]=row
        current={str(r['question_id']):r for r in runtime[task] if r['method']==CONTROL}
        differences=[];inputs=[]
        for sample in samples[task]:
            qid=sample_id(sample);a,b=current[qid],history[qid]
            changed=[key for key in PARITY_FIELDS if a[key]!=b[key]]
            changed_inputs=[key for key in ('prompt_token_ids','input_tensor_sha256','image_grid_thw','prompt_text') if a[key]!=b[key]]
            if changed:differences.append(dict(question_id=qid,fields=changed))
            if changed_inputs:inputs.append(dict(question_id=qid,fields=changed_inputs))
        result[task]=dict(questions=len(samples[task]),exact_questions=len(samples[task])-len(differences),
            differences=differences,input_differences=inputs,historical_runtime_timings_reused=False)
    return result


def path_changes(control,new):
    one,two=control['acceptance_paths'],new['acceptance_paths']
    first=next((i for i,(a,b) in enumerate(zip(one,two))
                if any(a.get(k)!=b.get(k) for k in PATH_FIELDS)),None)
    if first is None and len(one)!=len(two):first=min(len(one),len(two))
    return dict(first_selected_path_divergence_round=first,
        control_first_divergent_path=one[first] if first is not None and first<len(one) else None,
        uncapped_first_divergent_path=two[first] if first is not None and first<len(two) else None,
        changed=first is not None)


def transitions(task,samples,scores,rows):
    scored={method:{str(r['question_id']):r for r in scores[method]['rows']} for method in METHODS}
    runtime={(r['method'],str(r['question_id'])):r for r in rows}
    counts=Counter();cases=[];path_cases=[]
    for sample in samples:
        qid=sample_id(sample);a,b=scored[NEW][qid],scored[CONTROL][qid]
        new,control=runtime[NEW,qid],runtime[CONTROL,qid]
        changed=new['output_text']!=control['output_text'];counts['outputs_changed' if changed else 'outputs_unchanged']+=1
        path=path_changes(control,new);counts['acceptance_path_changed' if path['changed'] else 'acceptance_path_unchanged']+=1
        if path['changed']:path_cases.append(dict(question_id=qid,group_id=sample['group_id'],**path))
        if task=='coco':
            comparison={}
            for key in ('SPICE','CHAIRs','CHAIRi','CIDEr'):
                av,bv=a.get(key),b.get(key)
                comparison.update({key+'_uncapped':av,key+'_round2':bv,key+'_delta':None if av is None or bv is None else av-bv})
                suffix='_undefined' if av is None or bv is None else '_equal' if av==bv else '_increased' if av>bv else '_decreased'
                counts[key+suffix]+=1
        else:
            delta=a['score']-b['score'];counts['score_improved' if delta>0 else 'score_degraded' if delta<0 else 'score_tie']+=1
            comparison=dict(score_delta=delta,uncapped_score=a['score'],round2_score=b['score'],
                uncapped_answer=a.get('extracted_answer'),round2_answer=b.get('extracted_answer'),
                uncapped_parse_status=a.get('answer_extraction_status'),round2_parse_status=b.get('answer_extraction_status'))
            if task=='scienceqa':counts[f'round2_{int(b["score"])}_to_uncapped_{int(a["score"])}']+=1
        if changed:
            cases.append(dict(task=task,question_id=qid,group_id=sample['group_id'],image_id=sample.get('image_id'),
                image_path=sample.get('image_path'),prompt_text=new['prompt_text'],uncapped_text=new['output_text'],round2_text=control['output_text'],
                uncapped_committed_mismatches=new['signal_stats'].get('committed_mismatches',0),
                round2_committed_mismatches=control['signal_stats'].get('committed_mismatches',0),
                uncapped_truncated=new['cap_reached'],round2_truncated=control['cap_reached'],**comparison,**path))
    return dict(counts=dict(counts),changed_cases=cases,path_changes=path_cases,
        selection='all changed outputs and all path changes retained, no score/harm/length filtering')


def analyze(config,config_path,samples,runtime,scores,result_hashes):
    old=legacy_math();require(config.get('bootstrap',5000)==5000,'Frozen 5000 draws')
    summaries=[];comparisons=[];features=[];diagnostics=[];draw_seals={}
    for index,task in enumerate(config['tasks']):
        groups,mapping,categories=old.group_structure(samples[task])
        seed=int(config.get('bootstrap_seed',2026103109))+index*100003
        draws=old.bootstrap_indices(groups,categories,5000,seed,False)
        draw_seals[task]=dict(seed=seed,unit='physical image/group',groups=groups,shape=list(draws.shape),sha256=hashlib.sha256(draws.tobytes()).hexdigest())
        vectors={}
        for method in METHODS:
            vector,summary=old.method_metrics(task,method,samples[task],runtime[task],scores[task][method],draws,groups,mapping,categories)
            variant=config['methods_by_task'][task][method];diag,costs=runtime_diagnostics(task,method,runtime[task])
            summary.update(diag);summary.update(gamma=variant.get('gamma'),tau=variant.get('tau'),
                response_mismatch_budget=None,round_mismatch_budget=variant['round_mismatch_budget'],
                round_cap_enabled=method==CONTROL,measurement_scope='new same-run single-pass paired timing')
            vectors[method]=vector;summaries.append(summary);features.extend(costs)
        for metric in vectors[NEW]:
            row=old.paired_comparison(task,NEW,CONTROL,metric,vectors[NEW],vectors[CONTROL],'single_prespecified_no_round_cap_minus_round2')
            row['multiplicity']='one planned method comparison per task/metric; no global cross-metric correction'
            if row['status']=='ok' and row['baseline_point']:
                row['relative_change_percent']=100*row['difference']/row['baseline_point']
                if np.all(vectors[CONTROL][metric]['replicates']!=0):
                    lo,hi=np.quantile(100*(vectors[NEW][metric]['replicates']/vectors[CONTROL][metric]['replicates']-1),[.025,.975])
                    row['relative_change_percent_ci95_low']=float(lo);row['relative_change_percent_ci95_high']=float(hi)
            comparisons.append(row)
        diagnostics.append(noninferiority(task,vectors))
    prior=ROOT/'audit/STATISTICAL_ANALYSIS.json'
    created=read(prior)['created_utc'] if prior.exists() else datetime.now(timezone.utc).isoformat()
    manifest=dict(status='completed_BEPC_round_cap_ablation',created_utc=created,config_sha256=sha(config_path),
        analyzer_sha256=sha(__file__),evaluator_sha256=sha(ROOT/'harness/evaluate_round_uncapped.py'),
        legacy_statistical_math_sha256=sha(LEGACY_ANALYZER),result_sha256=result_hashes,
        scoring_seals={task:{method:r['seal_sha256'] for method,r in batch.items()} for task,batch in scores.items()},
        bootstrap=5000,bootstrap_unit='physical image/group',draws=draw_seals,
        comparisons=[[NEW,CONTROL]],p_method='two-sided centered paired-bootstrap add-one Monte Carlo approximation',
        multiple_comparisons='one frozen pair per task/metric; no global correction across tasks/metrics',
        CI_scope='paired image sampling only; no repeated-run timing uncertainty; fixed official per-image scores',
        measured_main_requests=sum(len(batch) for batch in runtime.values()),repetitions=1,
        numerical_environment=dict(python=sys.version,numpy=np.__version__),
        immutable_previous_source_manifest_sha256=sha(ROOT/'audit/PRIOR_PRESERVATION.json'),
        historical_parity=historical_parity(samples,runtime),quality_diagnostics=diagnostics,
        noninferiority_arithmetic_boundary_tolerance=NI_ARITHMETIC_TOLERANCE,
        output_transitions={task:transitions(task,samples[task],scores[task],runtime[task]) for task in config['tasks']},
        human_ground_truth_status='pending: no newly submitted double-blind human caption assessment',
        historical_test_reused=True,independent_project_unseen_test_claim=False,prior_timing_not_used_in_primary_comparisons=True)
    return summaries,comparisons,manifest,features


def number(value,digits=2,scale=1):
    if value is None or value=='':return 'NA'
    value=float(value)*scale
    return f'{value:.3e}' if 0<abs(value)<10**(-digits) else f'{value:.{digits}f}'


def interval(row,key,scale=1):
    return f'{number(row[key],scale=scale)} [{number(row[key+"_ci95_low"],scale=scale)}, {number(row[key+"_ci95_high"],scale=scale)}]'


def report(config,summaries,comparisons,manifest):
    lines=['# BEPC 取消每轮 mismatch 上限：完整实验报告','',
        '比较原 BEPC（每轮最多2个mismatch、每题无累计上限）与取消每轮上限后的BEPC。'
        '唯一政策变化是删除每轮mismatch计数门；候选接受评分、tau=1、双向gamma=.5、树、Prompt、EOS和512-token停止预算保持一致。'
        '无mismatch计数上限仍须逐个通过BEPC接受门，并受有限草稿树和停止规则约束。','',
        'ScienceQA、COCO、TextVQA各20开发组+100测试组，正式模型请求共'+str(manifest['measured_main_requests'])+'次，两臂均重新计时一次。'
        '开发样本用于实现验证，没有搜索阈值或按质量挑选方法。样本是沿用项目已有图像，不能宣称项目未见测试。','',
        '## 直接结果','']
    for task in config['tasks']:
        by={r['method']:r for r in summaries if r['task']==task};a,b=by[NEW],by[CONTROL]
        speed=100*(a['throughput_tokens_per_second']/b['throughput_tokens_per_second']-1)
        latency=100*(a['mean_seconds']/b['mean_seconds']-1)
        mat=100*(a['effective_mean_accepted_draft_length']/b['effective_mean_accepted_draft_length']-1)
        q=('SPICE '+number(b['SPICE'],scale=100)+'→'+number(a['SPICE'],scale=100)+'；CHAIRs '+number(b['CHAIRs'],scale=100)+'%→'+number(a['CHAIRs'],scale=100)+'%') if task=='coco' else '官方Accuracy '+number(b['accuracy'],scale=100)+'%→'+number(a['accuracy'],scale=100)+'%'
        lines.append(f'- {TITLES[task]}：取消每轮上限后token/s {speed:+.2f}%，请求时延 {latency:+.2f}%，有效MAT {mat:+.2f}%；{q}。')
    lines+=['','吞吐和整题时延分别评价，输出长度改变可能使二者方向不同。以下95%CI反映图像组抽样，不包含重复运行计时波动。','',
        '## 方法与执行口径','',
        '同一合法前缀固定Target候选a与Draft候选d，以原生g=z(a)−z(d)为基础。BEPC将父节点Target末层RMSNorm前状态h投影到'
        '图像prefill子空间U与正交化的近期文本子空间P，候选方向q=w(a)−w(d)，用完整h对应的RMSNorm缩放读取mU、mP。'
        'U/P是几何代理，不能解释为因果分离的视觉事实与语言偏见。mU·mP<0且两者幅度均≥1e−4时，'
        'δ=.5·sign(mP)·min(|mU|,|mP|,1)，否则δ=0；按g−δ≤1放行。residual_attenuation=false。','',
        '两臂都使用U/P秩8/4、视觉池64、文本窗口32，保留既有安全边界预筛、缓存、同一原生Target/Draft forward、'
        'tree attention与KV提交。每条合法分支连续扫描至首次拒绝，选择首个最长可接受前缀。'
        '对照在第三个mismatch前截断该分支，新方法取消这个计数截断。仅最终选中的分支提交历史与mismatch计数；'
        '其他分支的预算检查次数不是自然事件数量。没有额外Target forward。','',
        '## 冻结设置与测量','',
        f'- 上游ViSpec commit：`{config.get("upstream_commit","见配置")}`。',
        f'- Target：`{config.get("target")}`；Draft：`{config.get("draft")}`。',
        '- T=0、seed=0、BF16；两张L20按物理图像组分片，同题两方法顺序随机化；单次测量，warmup排除。',
        '- 原长输出Prompt、max_new_tokens=512、原Draft树和EOS不变；实际每条树分支仍有限长，不会无限接受。',
        '- 请求时间包含视觉编码、prefill、Target/Draft、验证与BEPC特征；共享PIL/processor/H2D准备不计入，非客户端全链路E2E。',
        '- token/s=可见内容token总数/同步模型请求总秒数，排除EOS，按512可见预算裁切。另列decode-only token/s。',
        '- 有效MAT=可见范围接受Draft token总数/验证轮数，不含每轮root；原生MAT保留官方整块统计。',
        '- 每task100个物理image/group配对bootstrap5000次；同一重采样同时用于两臂。只有一个预定方法对比/task/metric，无跨任务跨指标全局校正。','']
    for task in config['tasks']:
        batch=[r for r in summaries if r['task']==task]
        lines+=['## '+TITLES[task],'','| 方法 | token/s [95% CI] | 请求ms [95% CI] | decode token/s | 原生MAT | 有效MAT | 峰值allocated GiB | 峰值reserved GiB |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in batch:lines.append('| '+' | '.join([r['method'],interval(r,'throughput_tokens_per_second'),interval(r,'mean_seconds',1000),number(r['decode_throughput_tokens_per_second']),number(r['native_mean_accepted_draft_length']),number(r['effective_mean_accepted_draft_length']),number(r['max_peak_allocated_bytes'],scale=1/2**30),number(r['max_peak_reserved_bytes'],scale=1/2**30)])+' |')
        if task=='coco':
            lines+=['','| 方法 | CIDEr-D原尺度 | SPICE % [95% CI] | CHAIRs % [95% CI] | CHAIRi % |','|---|---:|---:|---:|---:|']
            for r in batch:lines.append('| '+' | '.join([r['method'],number(r['CIDEr'],6),interval(r,'SPICE',100),interval(r,'CHAIRs',100),number(r['CHAIRi'],scale=100)])+' |')
        else:
            lines+=['','| 方法 | 官方Accuracy % [95% CI] | 提取失败总数 | invalid | uncertain |','|---|---:|---:|---:|---:|---:|']
            for r in batch:lines.append(f'| {r["method"]} | {interval(r,"accuracy",100)} | {r["answer_extraction_failures"]}/{r["questions"]} | {r["parse_invalid_count"]} | {r["parse_uncertain_count"]} |')
        lines+=['','| 方法 | token/题 | 词/题 | 截断% | mismatch均值 | p50/p95/max | 每轮max | >2 mismatch轮数/总轮数 | 受影响响应数 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in batch:lines.append('| '+' | '.join([r['method'],number(r['mean_visible_tokens']),number(r['mean_output_words']),number(r['truncation_fraction'],scale=100),number(r['mean_committed_mismatches']),'/'.join(number(r[k],1) for k in ('committed_mismatch_p50','committed_mismatch_p95','committed_mismatch_max')),str(r['actual_per_round_mismatch_max']),str(r['selected_rounds_above_two_mismatches'])+'/'+str(r['selected_rounds']),str(r['responses_with_round_above_two_mismatches'])+'/'+str(r['questions'])])+' |')
        lines+=['','| 无上限−round2指标 | 差值 [95% CI] | 相对变化% [95% CI] | 近似双侧p |','|---|---:|---:|---:|']
        wanted=('throughput_tokens_per_second','mean_seconds','effective_mean_accepted_draft_length','SPICE','CHAIRs','accuracy','answer_extraction_failure_fraction')
        for r in comparisons:
            if r['task']!=task or r['metric'] not in wanted:continue
            lines.append('| '+' | '.join([r['metric'],f'{number(r["difference"],4)} [{number(r["ci95_low"],4)}, {number(r["ci95_high"],4)}]',f'{number(r.get("relative_change_percent"))} [{number(r.get("relative_change_percent_ci95_low"))}, {number(r.get("relative_change_percent_ci95_high"))}]',number(r['p'],4)])+' |')
        lines+=['','全部转移（round2→无mismatch上限）：`'+json.dumps(manifest['output_transitions'][task]['counts'],ensure_ascii=False,sort_keys=True)+'`。','']
        control=next(r for r in batch if r['method']==CONTROL)
        lines.append(f'原round2在{control["responses_with_round_budget_binding"]}/100个响应中记录计数门触发，候选扫描访问累计{control["round_budget_binding_candidate_visits"]}次；这些访问不是去重事件、选中轮次或独立mismatch。')
    lines+=['','## 自动代理非劣效检查','',
        '沿用预先固定容忍门：QA最多下降2个百分点；COCO同时满足新SPICE≥原SPICE的95%、CHAIRs增幅≤2个百分点。'
        '使用配对bootstrap单侧95%界；比较边界预先规定1e−12纯算术舍入容差，原始未舍入界保留在JSON。'
        '这不是额外质量容忍放宽，不作用于CIDEr或配对差异；不得将“不显著”视为相等。'
        '这些容忍门是pilot研究扩展，不是官方指标规定的安全标准。','',
        '| 任务 | 自动代理门 | 原始单侧界 |','|---|---|---|']
    for item in manifest['quality_diagnostics']:
        detail='; '.join(f'{r["metric"]}: {r.get("bound_type",r.get("status"))}={r.get("bound")!r}, pass={r["proxy_pass"]}' for r in item['conditions'])
        lines.append(f'| {item["task"]} | {"PASS" if item["automatic_proxy_pass"] else "NOT ESTABLISHED"} | {detail} |')
    lines+=['','QA仅评价最终答案，不验证全部解释；COCO双人独立人工事实/充分性审核仍pending。'
        '原样保留invalid、uncertain、截断与提取失败，失败计0，不删除差例。零分歧的bootstrap可能退化，不能推出罕见风险为0。','',
        '## 工程成本与复现审计','',
        'FEATURE_COST_COUNTERS.csv保留候选、子空间、缓存等实际计数覆盖。它们不是单独同步GPU延迟；请求总时间与显存已包含真实成本。'
        'allocated为活跃张量峰值；reserved受allocator历史影响，不能解释为算法最低独占显存。两臂均无额外Target forward。','']
    for task,item in manifest['historical_parity'].items():lines.append(f'- {TITLES[task]} round2对照与上一轮BEPC：完整输出/接受路径精确一致{item["exact_questions"]}/{item["questions"]}，输入差异{len(item["input_differences"])}；未复用历史时延。')
    lines+=['','audit/MAIN_VALIDATION.json与DEVELOPMENT_VALIDATION.json审计运行期预算字段、提交路径和首次分叉。'
        'STATISTICAL_ANALYSIS.json另保留所有首次选中路径分叉，不按质量挑选。CHANGED_OUTPUT_PAIRS.jsonl包含全部改变完整输出、输入图像引用、Prompt及官方质量分数。','',
        'CPU官方评分仅在全部GPU生成退出、GPU_COMPLETION.json确认无GPU PID后开始。整语料官方分数仅在完整有序样本、标签、解析器、实现和原始输出签名完全相同时复用；'
        '记录来源与SHA，不复用计时。CIDEr-D的文档频率语料不变，bootstrap使用固定官方每图分数，不在每次抽样重拟合DF。','',
        '## 评价局限','',
        '- ScienceQA使用冻结末行解析器和官方Accuracy；TextVQA执行官方VQA-style归一化与leave-one-out评分。全部提取失败计0。',
        '- COCO全文原样进入官方PTB、CIDEr-D、SPICE和作者CHAIR；SPICE仅使用既有64G Java资源上限，评分模型/公式不变。',
        '- 长段落对短参考使CIDEr-D可能接近0，不得单独解释为事实差；SPICE受参考覆盖限制；CHAIR主要覆盖COCO对象幻觉，无法验证全部属性、动作、关系、数量与遗漏。',
        '- 本轮只有一次运行，复用项目已有图像；CI不涵盖重复运行噪声、设备间推广或罕见错误，没有独立未见图像与人工caption质量证据。',
        '- 政策变化会改变后续上下文与输出长度；这是完整解码策略对比，不是单事件mismatch的因果有害标签。',
        '- 未调整BEPC算法思想、阈值或候选信号；没有训练、融合、扩样或根据测试成绩挑选更优参数。',
        '- 若自动代理非劣效未建立，应保留原方法，不能用吞吐或MAT增长替代质量证据。','',
        '证据：configs/frozen_config.json、results/main/、metrics/main/与audit/STATISTICAL_ANALYSIS.json。'
        '数值见BENCHMARK_SUMMARY.csv、PAIRED_COMPARISONS.csv；图见BENCHMARK_PERFORMANCE.pdf、BENCHMARK_QUALITY.pdf。','']
    return '\n'.join(lines)


def plot_payload(config,summaries,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    colors=['#287eaf','#d58b3e'];labels=['BEPC round2','BEPC no count caps']
    with PdfPages(destination/'BENCHMARK_PERFORMANCE.pdf') as pdf:
        for metric,scale,title in [('throughput_tokens_per_second',1,'Visible content tokens/s'),('mean_seconds',1000,'Full model request (ms)'),('effective_mean_accepted_draft_length',1,'Effective Draft MAT (root excluded)')]:
            fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
            for ax,task in zip(axes,config['tasks']):
                rows=[r for r in summaries if r['task']==task];x=np.arange(len(rows));y=np.array([r[metric]*scale for r in rows]);lo=np.array([r[metric+'_ci95_low']*scale for r in rows]);hi=np.array([r[metric+'_ci95_high']*scale for r in rows])
                ax.bar(x,y,color=colors);ax.errorbar(x,y,yerr=np.maximum(0,np.vstack((y-lo,hi-y))),fmt='none',color='black',capsize=3)
                ax.set_xticks(x,labels,fontsize=9);ax.set_title(TITLES[task]);ax.set_ylabel(title);ax.grid(axis='y',alpha=.2)
            fig.suptitle('Single pass / long prompts / paired image bootstrap 95% CI');pdf.savefig(fig);plt.close(fig)
    with PdfPages(destination/'BENCHMARK_QUALITY.pdf') as pdf:
        fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
        for ax,task in zip(axes,config['tasks']):
            metric='SPICE' if task=='coco' else 'accuracy';rows=[r for r in summaries if r['task']==task];x=np.arange(len(rows));y=np.array([r[metric]*100 for r in rows]);lo=np.array([r[metric+'_ci95_low']*100 for r in rows]);hi=np.array([r[metric+'_ci95_high']*100 for r in rows])
            ax.bar(x,y,color=colors);ax.errorbar(x,y,yerr=np.maximum(0,np.vstack((y-lo,hi-y))),fmt='none',color='black',capsize=3);ax.set_xticks(x,labels);ax.set_title(TITLES[task]);ax.set_ylabel(metric+' (%)');ax.grid(axis='y',alpha=.2)
        fig.suptitle('Automatic quality metrics / human caption audit pending');pdf.savefig(fig);plt.close(fig)
        rows=[r for r in summaries if r['task']=='coco'];fig,ax=plt.subplots(figsize=(6,4),constrained_layout=True);x=np.arange(len(rows));y=np.array([r['CHAIRs']*100 for r in rows]);lo=np.array([r['CHAIRs_ci95_low']*100 for r in rows]);hi=np.array([r['CHAIRs_ci95_high']*100 for r in rows]);ax.bar(x,y,color=colors);ax.errorbar(x,y,yerr=np.maximum(0,np.vstack((y-lo,hi-y))),fmt='none',color='black',capsize=3);ax.set_xticks(x,labels);ax.set_ylabel('CHAIRs (%) - lower is better');ax.set_title('COCO object hallucination only');pdf.savefig(fig);plt.close(fig)


def self_test():
    old=legacy_math();previous=list(sys.path);sys.path.insert(0,str(LEGACY_ANALYZER.parent))
    try:old.self_test()
    finally:sys.path[:]=previous
    vector={method:{'accuracy':dict(point=.8,replicates=np.full(5000,.8))} for method in METHODS}
    require(noninferiority('scienceqa',vector)['automatic_proxy_pass'],'Equal QA diagnostic')
    vector[NEW]['accuracy']=dict(point=.78,replicates=np.full(5000,.78))
    require(noninferiority('scienceqa',vector)['automatic_proxy_pass'],'Exact2pp boundary tolerates only FP64 roundoff')
    vector[NEW]['accuracy']=dict(point=.779999,replicates=np.full(5000,.779999))
    require(not noninferiority('scienceqa',vector)['automatic_proxy_pass'],'Real decline past2pp rejected')
    vector={method:{'SPICE':dict(point=.2,replicates=np.full(5000,.2)),
        'CHAIRs':dict(point=.2,replicates=np.full(5000,.2))} for method in METHODS}
    vector[NEW]['SPICE']=dict(point=.19,replicates=np.full(5000,.19));vector[NEW]['CHAIRs']=dict(point=.22,replicates=np.full(5000,.22))
    require(noninferiority('coco',vector)['automatic_proxy_pass'],'COCO exact declared boundaries')
    require('e-' in number(1e-50,6),'Tiny nonzero CIDEr preserved')
    require(numeric_stats({'a':2,'nested':{'b':3},'bool':True})=={'a':2,'nested.b':3},'Diagnostic flattening')
    stats=dict(committed_mismatches=6,response_cap_enabled=False,response_mismatch_budget=None,
        round_mismatch_budget=None,round_cap_enabled=False,row_budget_binding=0)
    synthetic=[dict(method=NEW,ordinal=0,signal_stats=stats,acceptance_paths=[dict(mismatches=3,accepted_draft=4)]*2)]
    diag,_=runtime_diagnostics('scienceqa',NEW,synthetic)
    require(diag['selected_rounds_above_two_mismatches']==2 and diag['responses_with_round_above_two_mismatches']==1,
            'No round cap admits>2 selected mismatch rounds')
    print('Paired-analysis numerical tests passed; no inference or official scoring')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=ROOT/'configs/frozen_config.json');parser.add_argument('--self-test',action='store_true');parser.add_argument('--plot-payload',type=Path);args=parser.parse_args()
    if args.self_test:self_test();return
    if args.plot_payload:
        data=read(args.plot_payload);plot_payload(data['config'],data['summaries'],ROOT/'reports');return
    config=read(args.config);samples,rows,result_hashes=load_phase(config,args.config,'main');scores=score_phase(config,args.config,'main',samples,rows,result_hashes)
    summaries,comparisons,manifest,features=analyze(config,args.config,samples,rows,scores,result_hashes)
    destination=ROOT/'reports';destination.mkdir(parents=True,exist_ok=True)
    csv_once(destination/'BENCHMARK_SUMMARY.csv',summaries);csv_once(destination/'PAIRED_COMPARISONS.csv',comparisons);csv_once(destination/'FEATURE_COST_COUNTERS.csv',features)
    write_once(ROOT/'audit/STATISTICAL_ANALYSIS.json',manifest)
    changed=[row for task in config['tasks'] for row in manifest['output_transitions'][task]['changed_cases']]
    changed_path=destination/'CHANGED_OUTPUT_PAIRS.jsonl';payload=''.join(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n' for row in changed)
    require(not changed_path.exists() or changed_path.read_text()==payload,'Preserve all changed outputs')
    if not changed_path.exists():changed_path.write_text(payload,encoding='utf-8')
    path=destination/'FULL_EXPERIMENT_REPORT.md';text=report(config,summaries,comparisons,manifest)
    require(not path.exists() or path.read_text()==text,'Preserve full report')
    if not path.exists():path.write_text(text,encoding='utf-8')
    plot=ROOT/'audit/PLOT_PAYLOAD.json';write_once(plot,dict(config=config,summaries=summaries));pdfs=[destination/'BENCHMARK_PERFORMANCE.pdf',destination/'BENCHMARK_QUALITY.pdf']
    if not all(item.exists() and item.stat().st_size for item in pdfs):
        if importlib.util.find_spec('matplotlib'):plot_payload(config,summaries,destination)
        else:subprocess.run(['/home/wbb/miniconda3/bin/python',str(Path(__file__).resolve()),'--plot-payload',str(plot)],check=True)
    artifacts=[path,changed_path,destination/'BENCHMARK_SUMMARY.csv',destination/'PAIRED_COMPARISONS.csv',destination/'FEATURE_COST_COUNTERS.csv',*pdfs,ROOT/'audit/STATISTICAL_ANALYSIS.json']
    write_once(ROOT/'audit/REPORT_DELIVERY.json',dict(status='completed',artifacts_sha256={str(item):sha(item) for item in artifacts}))
    print(json.dumps(dict(status='completed',main_requests=manifest['measured_main_requests'],report=str(path),changed_outputs=len(changed)),ensure_ascii=False))


if __name__=='__main__':main()
