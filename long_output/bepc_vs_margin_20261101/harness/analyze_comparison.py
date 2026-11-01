"""Pre-frozen paired CPU analysis of BEPC versus ordinary Margin.

Both policies retain round2, tau1, and no response cap. No test-driven tuning,
new sampling, historical timing reuse, or output rewriting is performed.
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

from evaluate_comparison import (ROOT, PREVIOUS, UNCAPPED, METHODS, TASKS,
    require, read, sha, write_once, load_phase, score_phase, sample_id)

LEGACY_ANALYZER = PREVIOUS / 'harness/analyze_results.py'
BEPC, MARGIN = METHODS
TITLES = {'scienceqa': 'ScienceQA', 'coco': 'COCO Caption', 'textvqa': 'TextVQA'}
PARITY_FIELDS = ('raw_token_ids', 'returned_token_ids', 'output_text',
    'acceptance_length', 'acceptance_paths', 'eos_index', 'cap_reached')


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
    counts, paths, numeric = [], [], []
    for row in selected:
        stats = row.get('signal_stats', {})
        count = int(stats.get('committed_mismatches', stats.get('committed_response_mismatches', 0)))
        selected_paths = [int(p.get('mismatches', 0)) for p in row['acceptance_paths']]
        require(all(0 <= value <= 2 for value in selected_paths), 'Per-round mismatch budget violated')
        require(sum(selected_paths)==count, 'Selected path and response mismatch counts differ')
        require(stats.get('response_cap_enabled') is False and stats.get('response_mismatch_budget') is None,
                'Both methods must explicitly disable response cap')
        counts.append(count); paths.extend(selected_paths); numeric.append(numeric_stats(stats))
    costs = []
    for key in sorted(set().union(*(r.keys() for r in numeric))):
        present = [r[key] for r in numeric if key in r]
        costs.append(dict(task=task, method=method, counter=key, present_questions=len(present), total_questions=len(selected),
            sum=float(sum(present)), mean=float(np.mean(present)), maximum=float(max(present)),
            interpretation='diagnostic counter; not separately synchronized GPU feature latency'))
    summary = dict(committed_mismatch_p50=float(np.quantile(counts,.5)), committed_mismatch_p95=float(np.quantile(counts,.95)),
        committed_mismatch_max=max(counts), response_fraction_above_four_mismatches=float(np.mean(np.asarray(counts)>4)),
        actual_per_round_mismatch_max=max(paths,default=0), per_round_budget_violations=sum(n>2 for n in paths))
    return summary, costs


def noninferiority(task, vectors):
    one, two = vectors[BEPC], vectors[MARGIN]
    conditions = []
    if task != 'coco':
        lower = float(np.quantile(one['accuracy']['replicates']-two['accuracy']['replicates'], .05))
        conditions.append(dict(metric='accuracy', difference=one['accuracy']['point']-two['accuracy']['point'],
            bound_type='one_sided_95_lower', bound=lower, permitted_decline=.02, proxy_pass=lower >= -.02))
    else:
        for key, multiplier, bound_limit, upper in (('SPICE',.95,0.,False),('CHAIRs',1.,.02,True)):
            a,b = one[key],two[key]
            if a['point'] is None or b['point'] is None:
                conditions.append(dict(metric=key,proxy_pass=False,status='undefined_required_metric')); continue
            bound = float(np.quantile(a['replicates']-multiplier*b['replicates'], .95 if upper else .05))
            conditions.append(dict(metric=key,difference=a['point']-b['point'],baseline_multiplier=multiplier,
                bound_type='one_sided_95_upper' if upper else 'one_sided_95_lower', bound=bound,
                permitted_difference=bound_limit,proxy_pass=bound <= bound_limit if upper else bound >= bound_limit))
    return dict(task=task,method=BEPC,baseline=MARGIN,conditions=conditions,
        automatic_proxy_pass=all(value['proxy_pass'] for value in conditions),
        scope='predeclared fixed-pilot automatic-metric diagnostic; not human factual quality guarantee',
        zero_discordance_bootstrap_can_be_degenerate=True)


def historical_parity(samples, runtime):
    result={}
    for task in samples:
        history={}
        for shard in (0,1):
            path=UNCAPPED/'results/main'/f'main_{task}_shard{shard}.jsonl'
            require(read(path.with_suffix('.complete.json'))['result_sha256']==sha(path),'Previous sealed result changed')
            for line in path.read_text().splitlines():
                row=json.loads(line)
                if row['method']==BEPC: history[str(row['question_id'])]=row
        current={str(r['question_id']):r for r in runtime[task] if r['method']==BEPC}
        differences=[]; input_differences=[]
        for sample in samples[task]:
            qid=sample_id(sample); a,b=current[qid],history[qid]
            changed=[key for key in PARITY_FIELDS if a[key]!=b[key]]
            inputs=[key for key in ('prompt_token_ids','input_tensor_sha256','image_grid_thw','prompt_text') if a[key]!=b[key]]
            if changed:differences.append(dict(question_id=qid,fields=changed))
            if inputs:input_differences.append(dict(question_id=qid,fields=inputs))
        result[task]=dict(questions=len(samples[task]),exact_questions=len(samples[task])-len(differences),
            differences=differences,input_differences=input_differences,historical_runtime_timings_reused=False)
    return result


def transitions(task,samples,scores,rows):
    scored={method:{str(r['question_id']):r for r in scores[method]['rows']} for method in METHODS}
    runtime={(r['method'],str(r['question_id'])):r for r in rows}
    counts=Counter(); cases=[]
    for sample in samples:
        qid=sample_id(sample); a,b=scored[BEPC][qid],scored[MARGIN][qid]
        bepc,margin=runtime[BEPC,qid],runtime[MARGIN,qid]
        changed=bepc['output_text']!=margin['output_text']; counts['outputs_changed' if changed else 'outputs_unchanged']+=1
        if task=='coco':
            comparison={}
            for key in ('SPICE','CHAIRs','CHAIRi','CIDEr'):
                av,bv=a.get(key),b.get(key)
                comparison.update({key+'_bepc':av,key+'_margin':bv,key+'_delta':None if av is None or bv is None else av-bv})
                suffix='_undefined' if av is None or bv is None else '_equal' if av==bv else '_increased' if av>bv else '_decreased'
                counts[key+suffix]+=1
        else:
            delta=a['score']-b['score']; counts['score_improved' if delta>0 else 'score_degraded' if delta<0 else 'score_tie']+=1
            comparison=dict(score_delta=delta,bepc_score=a['score'],margin_score=b['score'],
                bepc_answer=a.get('extracted_answer'),margin_answer=b.get('extracted_answer'),
                bepc_parse_status=a.get('answer_extraction_status'),margin_parse_status=b.get('answer_extraction_status'))
            if task=='scienceqa':counts[f'margin_{int(b["score"])}_to_bepc_{int(a["score"])}']+=1
        if changed:
            cases.append(dict(task=task,question_id=qid,group_id=sample['group_id'],image_id=sample.get('image_id'),
                image_path=sample.get('image_path'),prompt_text=bepc['prompt_text'],bepc_text=bepc['output_text'],margin_text=margin['output_text'],
                bepc_committed_mismatches=bepc['signal_stats'].get('committed_mismatches',0),
                margin_committed_mismatches=margin['signal_stats'].get('committed_mismatches',0),
                bepc_truncated=bepc['cap_reached'],margin_truncated=margin['cap_reached'],**comparison))
    return dict(counts=dict(counts),changed_cases=cases,selection='all changed outputs, no score/harm/length filtering')


def analyze(config,config_path,samples,runtime,scores,result_hashes):
    old=legacy_math(); require(config.get('bootstrap',5000)==5000,'Frozen 5000 draws')
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
                response_mismatch_budget=None,round_mismatch_budget=2,measurement_scope='new same-run single-pass paired timing')
            vectors[method]=vector;summaries.append(summary);features.extend(costs)
        for metric in vectors[BEPC]:
            row=old.paired_comparison(task,BEPC,MARGIN,metric,vectors[BEPC],vectors[MARGIN],'single_prespecified_BEPC_minus_Margin')
            row['multiplicity']='one planned method comparison per task/metric; no global cross-metric correction'
            if row['status']=='ok' and row['baseline_point']:
                row['relative_change_percent']=100*row['difference']/row['baseline_point']
                if np.all(vectors[MARGIN][metric]['replicates'] != 0):
                    lo,hi=np.quantile(100*(vectors[BEPC][metric]['replicates']/vectors[MARGIN][metric]['replicates']-1),[.025,.975])
                    row['relative_change_percent_ci95_low']=float(lo);row['relative_change_percent_ci95_high']=float(hi)
            comparisons.append(row)
        diagnostics.append(noninferiority(task,vectors))
    prior=ROOT/'audit/STATISTICAL_ANALYSIS.json'
    created=read(prior)['created_utc'] if prior.exists() else datetime.now(timezone.utc).isoformat()
    manifest=dict(status='completed_BEPC_vs_Margin',created_utc=created,config_sha256=sha(config_path),
        analyzer_sha256=sha(__file__),evaluator_sha256=sha(ROOT/'harness/evaluate_comparison.py'),
        legacy_statistical_math_sha256=sha(LEGACY_ANALYZER),result_sha256=result_hashes,
        scoring_seals={task:{method:item['seal_sha256'] for method,item in batch.items()} for task,batch in scores.items()},
        bootstrap=5000,bootstrap_unit='physical image/group',draws=draw_seals,
        comparisons=[[BEPC,MARGIN]],p_method='two-sided centered paired-bootstrap add-one Monte Carlo approximation',
        multiple_comparisons='one frozen BEPC-Margin pair per task/metric; no global correction across tasks/metrics',
        CI_scope='paired image sampling only; no repeated-run timing uncertainty; fixed official per-image scores',
        measured_main_requests=sum(len(batch) for batch in runtime.values()),repetitions=1,
        numerical_environment=dict(python=sys.version,numpy=np.__version__),
        immutable_previous_source_manifest_sha256=sha(ROOT/'audit/PRIOR_PRESERVATION.json'),
        historical_parity=historical_parity(samples,runtime),quality_diagnostics=diagnostics,
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
    lines=['# BEPC 与简单 Margin：生成速度和质量完整实验报告','',
        '比较刚完成解释的双向有界 BEPC 与普通 Target Margin。两种方法均为每轮最多2个 mismatch、每题无累计上限、tau=1，'
        '相同模型、输入、原长输出 Prompt、树、EOS 与512-token预算，每题各运行一次；本轮重新测两臂时延。', '',
        'ScienceQA、COCO、TextVQA 各20开发组+100测试组，正式请求共'+str(manifest['measured_main_requests'])+'次。'
        '沿用已有样本而非项目未见测试集。开发数据用于完整性与数值验证，没有搜索阈值或选择最佳方法。','',
        '## 本轮直接结果','']
    for task in config['tasks']:
        by={r['method']:r for r in summaries if r['task']==task};a,b=by[BEPC],by[MARGIN]
        speed=100*(a['throughput_tokens_per_second']/b['throughput_tokens_per_second']-1)
        latency=100*(a['mean_seconds']/b['mean_seconds']-1)
        mat=100*(a['effective_mean_accepted_draft_length']/b['effective_mean_accepted_draft_length']-1)
        q=('SPICE '+number(b['SPICE'],scale=100)+'→'+number(a['SPICE'],scale=100)+'；CHAIRs '+number(b['CHAIRs'],scale=100)+'%→'+number(a['CHAIRs'],scale=100)+'%') if task=='coco' else '官方 Accuracy '+number(b['accuracy'],scale=100)+'%→'+number(a['accuracy'],scale=100)+'%'
        lines.append(f'- {TITLES[task]}：BEPC 相对 Margin，token/s {speed:+.2f}%，整题模型时延 {latency:+.2f}%，有效MAT {mat:+.2f}%；{q}。')
    lines+=['','token/s、接受长度和每题完成时间分别解释；内容变长可以同时提高token/s和增加每题时延。以下配对CI仅反映当前图像组抽样，单次运行没有重复计时证据。','',
        '## 算法与对照口径','',
        '两种方法共享真正执行的 ViSpec 验证树、候选合法前缀、原生 Target logits、逐轮两个 mismatch 限制与首个最长前缀选择。'
        'Target top1为a、Draft候选为d，原生margin g=z(a)−z(d)，普通Margin直接按g≤1放行。','',
        'BEPC把同一父节点的末层RMSNorm前状态h投影到图像prefill子空间U和正交化的近期文本子空间P，'
        '候选方向q=w(a)−w(d)，用完整h对应的RMSNorm缩放读出mU、mP。U/P是表示几何代理，不是因果分离的事实/先验。'
        'mU·mP<0且两者幅度≥1e−4时，δ=0.5·sign(mP)·min(|mU|,|mP|,1)，否则δ=0；按g−δ≤1接受。'
        'residual_attenuation=false，双向gamma=.5，U/P秩8/4，视觉池64，文本窗口32；无新增Target forward。','',
        'BEPC使用已有精确边界预筛及缓存，保留其真实观察器/子空间成本；Margin不构造BEPC观察器或子空间。'
        '双方都不在累计4次后退回Strict。区别是接受评分及实现该评分必需的成本，没有修改原生Target/Draft forward。','',
        '为保留历史BEPC控制器字节，继承的BEPC统计字段all_relaxed_methods_same_budget=false仍记录上轮跨臂说明；'
        '它是遗留元数据，不表示本轮两臂预算不同。本轮权威依据是冻结config和运行期response_cap_enabled=false、'
        'response_mismatch_budget=null、round_mismatch_budget=2，两臂均经路径计数核验。','',
        '## 配置与测量','',
        f'- 上游ViSpec commit：`{config.get("upstream_commit","见配置")}`。',
        f'- Target：`{config.get("target")}`；Draft：`{config.get("draft")}`。',
        '- T=0、seed=0、BF16；两张L20按图像组分片，同题两方法顺序随机化；每方法单次测量，warmup排除。',
        '- 请求时间包含视觉编码、prefill、Target/Draft解码、验证与BEPC特征；不含预先共享的PIL/processor/H2D准备，不是客户端全链路E2E。',
        '- token/s=可见内容token总数/同步请求秒数总和，内容token排除EOS并遵守原512可见边界；decode token/s另行列出。',
        '- 有效MAT=可见范围接受Draft token总数/验证轮数，排除每轮root；原生MAT沿用官方整块计数。',
        '- 按每task100个物理图像/group成对bootstrap5000次；同一抽样同时用于两方法，95%百分位CI。只有一项预定方法对比/task/metric，不作跨所有指标的全局校正。','']
    for task in config['tasks']:
        rows=[r for r in summaries if r['task']==task]
        lines += ['## '+TITLES[task], '', '| 方法 | token/s [95% CI] | 请求ms [95% CI] | decode token/s | 原生MAT | 有效MAT | 峰值allocated GiB | 峰值reserved GiB |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            lines.append('| '+' | '.join([r['method'],interval(r,'throughput_tokens_per_second'),interval(r,'mean_seconds',1000),number(r['decode_throughput_tokens_per_second']),number(r['native_mean_accepted_draft_length']),number(r['effective_mean_accepted_draft_length']),number(r['max_peak_allocated_bytes'],scale=1/2**30),number(r['max_peak_reserved_bytes'],scale=1/2**30)])+' |')
        if task=='coco':
            lines+=['','| 方法 | CIDEr-D原尺度 | SPICE % [95% CI] | CHAIRs % [95% CI] | CHAIRi % |','|---|---:|---:|---:|---:|']
            for r in rows:lines.append('| '+' | '.join([r['method'],number(r['CIDEr'],6),interval(r,'SPICE',100),interval(r,'CHAIRs',100),number(r['CHAIRi'],scale=100)])+' |')
        else:
            lines+=['','| 方法 | 官方Accuracy % [95% CI] | 提取失败总数 | invalid | uncertain |','|---|---:|---:|---:|---:|']
            for r in rows:lines.append(f'| {r["method"]} | {interval(r,"accuracy",100)} | {r["answer_extraction_failures"]}/{r["questions"]} | {r["parse_invalid_count"]} | {r["parse_uncertain_count"]} |')
        lines+=['','| 方法 | 内容token/题 | 词/题 | 截断% | mismatch均值 | p50/p95/max | >4次比例% | 每轮max |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in rows:lines.append('| '+' | '.join([r['method'],number(r['mean_visible_tokens']),number(r['mean_output_words']),number(r['truncation_fraction'],scale=100),number(r['mean_committed_mismatches']),'/'.join(number(r[k],1) for k in ('committed_mismatch_p50','committed_mismatch_p95','committed_mismatch_max')),number(r['response_fraction_above_four_mismatches'],scale=100),str(r['actual_per_round_mismatch_max'])])+' |')
        lines+=['','| BEPC−Margin指标 | 差值 [95% CI] | 相对变化% [95% CI] | 近似双侧p |','|---|---:|---:|---:|']
        wanted=('throughput_tokens_per_second','mean_seconds','effective_mean_accepted_draft_length','SPICE','CHAIRs','accuracy','answer_extraction_failure_fraction')
        for r in comparisons:
            if r['task']!=task or r['metric'] not in wanted:continue
            lines.append('| '+' | '.join([r['metric'],f'{number(r["difference"],4)} [{number(r["ci95_low"],4)}, {number(r["ci95_high"],4)}]',f'{number(r.get("relative_change_percent"))} [{number(r.get("relative_change_percent_ci95_low"))}, {number(r.get("relative_change_percent_ci95_high"))}]',number(r['p'],4)])+' |')
        lines+=['','全部输出与分数转移（方向Margin→BEPC）：`'+json.dumps(manifest['output_transitions'][task]['counts'],ensure_ascii=False,sort_keys=True)+'`。','']
    lines+=['## 质量容忍检查与结论边界','',
        '预先固定的自动代理非劣效检查：QA最多下降2个百分点；COCO同时要求BEPC的SPICE≥Margin的95%、CHAIRs增幅≤2个百分点。'
        '采用配对bootstrap单侧95%界；不把“不显著”解释为相等。阈值是沿用pilot容忍规则的研究扩展，不是官方指标定义的安全标准。','',
        '| 任务 | BEPC相对Margin自动代理门 | 单侧界 |','|---|---|---|']
    for item in manifest['quality_diagnostics']:
        detail='; '.join(f'{r["metric"]}: {r.get("bound_type",r.get("status"))}={number(r.get("bound"),4)}, pass={r["proxy_pass"]}' for r in item['conditions'])
        lines.append(f'| {item["task"]} | {"PASS" if item["automatic_proxy_pass"] else "NOT ESTABLISHED"} | {detail} |')
    lines+=['','QA只评价最终答案，不自动验证全部长解释；COCO仍未收到新的双人独立盲评，自动指标不能替代人工事实质量。'
        '原样保留所有失败、uncertain与截断；提取失败计0，不删除差例。零分歧时bootstrap可能退化，不意味着罕见质量风险为0。','',
        '## 工程成本与复现审计','',
        'FEATURE_COST_COUNTERS.csv汇总实际特征/子空间/候选统计的覆盖、总量和均值。计数器不等于单独同步GPU耗时；'
        '总请求时间与峰值显存包含真实成本。allocated是活跃张量峰值，reserved包含缓存分配器保留区，受此前请求影响；'
        'reserved不能解释为每种算法的最低显存需求。两方法都没有额外Target forward，Margin零BEPC子空间构建由运行验证审计检查。','']
    for task,item in manifest['historical_parity'].items():
        lines.append(f'- {TITLES[task]} BEPC与上一轮uncapped：输出/接受路径精确一致 {item["exact_questions"]}/{item["questions"]}；输入差异 {len(item["input_differences"])}。历史时延未复用。')
    lines+=['','CPU官方评分在全部GPU任务退出后执行，防止SPICE/PTB评分与计时争用CPU。'
        '若全量有序样本、标签、解析器、官方实现和完整原始输出文本签名完全相同，可以复用整语料官方质量评分；每次复用记录来源及SHA，不复用计时。', '',
        '## 评价方法、局限与下一步','',
        '- ScienceQA使用冻结末行答案解析器和官方多选Accuracy；TextVQA使用冻结解析器与官方VQA-style归一化/leave-one-out评分。失败计0，大小写等处理沿用官方实现。',
        '- COCO完整段落原样进入官方PTB、CIDEr-D、SPICE和作者CHAIR。SPICE仅沿用64G Java资源上限，未改模型、实现或公式。',
        '- 长段落与短COCO参考的长度差异使CIDEr-D可能近乎为0；其数值变大不能单独作为事实改善。SPICE依赖参考覆盖；CHAIR主要覆盖COCO对象幻觉，不覆盖全部属性、关系、动作和遗漏。',
        '- 无完整独立图像测试、没有重复运行计时、没有人工caption充分性/事实精度审核，不能宣布一般性的质量保持或稳定加速优势。',
        '- 两种接受策略改变后续前缀与输出长度；这是整条解码政策比较，不是单个mismatch的因果危害标注。',
        '- 所有改变样本均保留在CHANGED_OUTPUT_PAIRS.jsonl，包含图像引用、原始Prompt、两条完整文本及分数，不按预期选择正例。',
        '- 最终选择应同时考虑token/s、整题时延、输出长度、MAT、最终答案、对象幻觉和人工caption质量；本轮不训练分类器、不融合规则、不根据测试调参。','',
        '证据：configs/frozen_config.json、results/main/、metrics/main/、audit/STATISTICAL_ANALYSIS.json。'
        '主表见BENCHMARK_SUMMARY.csv，全部配对差值见PAIRED_COMPARISONS.csv，图见BENCHMARK_PERFORMANCE.pdf和BENCHMARK_QUALITY.pdf。','']
    return '\n'.join(lines)


def plot_payload(config,summaries,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    colors=['#287eaf','#d58b3e'];labels=['BEPC no cap','Margin no cap']
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
    vector[BEPC]['accuracy']=dict(point=.7,replicates=np.full(5000,.7))
    require(not noninferiority('textvqa',vector)['automatic_proxy_pass'],'Declining QA diagnostic')
    require('e-' in number(1e-50,6),'Tiny nonzero CIDEr preserved')
    require(numeric_stats({'a':2,'nested':{'b':3},'bool':True})=={'a':2,'nested.b':3},'Diagnostic flattening')
    synthetic=[dict(method=BEPC,ordinal=0,signal_stats={'committed_mismatches':6,'response_cap_enabled':False,'response_mismatch_budget':None},acceptance_paths=[{'mismatches':2}]*3)]
    diag,_=runtime_diagnostics('scienceqa',BEPC,synthetic)
    require(diag['committed_mismatch_max']==6 and diag['response_fraction_above_four_mismatches']==1,'No response cap with round2')
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
