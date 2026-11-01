"""Frozen CPU analysis of removal of BEPC's per-response mismatch cap.

Only the response cap changes. No parameter selection, new sampling, training,
output normalization or rewriting of historical sources/results is performed.
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
from evaluate_uncapped import (ROOT, PREVIOUS, OPTIMIZATION, METHODS, TASKS,
    require, read, sha, write_once, load_phase, score_phase, sample_id)

LEGACY_ANALYZER = PREVIOUS / 'harness/analyze_results.py'
COMPARISONS = (
    ('bepc_no_response_cap', 'bepc_cap4', 'response_cap_removal'),
    ('bepc_cap4', 'strict_shared', 'capped_relaxation_vs_strict'),
    ('bepc_no_response_cap', 'strict_shared', 'uncapped_relaxation_vs_strict'),
)
PARITY_FIELDS = ('raw_token_ids', 'returned_token_ids', 'output_text',
    'acceptance_length', 'acceptance_paths', 'eos_index', 'cap_reached')
TITLES = {'scienceqa': 'ScienceQA', 'textvqa': 'TextVQA', 'coco': 'COCO Caption'}


def legacy_math():
    previous_path = list(sys.path)
    sys.path.insert(0, str(LEGACY_ANALYZER.parent))
    try:
        spec = importlib.util.spec_from_file_location('_read_only_prior_statistical_math', LEGACY_ANALYZER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = previous_path
    return module


def csv_once(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=sorted({key for row in rows for key in row}))
    writer.writeheader(); writer.writerows(rows)
    payload = buffer.getvalue()
    if path.exists():
        require(path.read_text() == payload.replace('\r\n', '\n'), 'Preserve CSV: ' + str(path))
    else:
        path.write_text(payload)


def historical_parity(samples, runtime_rows):
    result = {}
    for task in samples:
        lookup = {(r['method'], str(r['question_id'])): r for r in runtime_rows[task]}
        history = {}
        for shard in (0, 1):
            path = OPTIMIZATION / 'results/main' / f'main_{task}_shard{shard}.jsonl'
            require(read(path.with_suffix('.complete.json'))['result_sha256'] == sha(path), 'Previous result seal changed')
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row['repeat'] == 0 and row['method'] in ('strict_shared', 'bepc_fast'):
                    history[row['method'], str(row['question_id'])] = row
        entries = {}
        for current, prior in (('strict_shared', 'strict_shared'), ('bepc_cap4', 'bepc_fast')):
            differences, inputs = [], []
            for sample in samples[task]:
                qid = sample_id(sample)
                a, b = lookup[current, qid], history[prior, qid]
                changed = [key for key in PARITY_FIELDS if a[key] != b[key]]
                inp = [key for key in ('prompt_token_ids', 'input_tensor_sha256', 'image_grid_thw', 'prompt_text') if a[key] != b[key]]
                if changed: differences.append(dict(question_id=qid, fields=changed))
                if inp: inputs.append(dict(question_id=qid, fields=inp))
            entries[current] = dict(historical_method=prior, questions=len(samples[task]),
                exact_questions=len(samples[task])-len(differences), differences=differences, input_differences=inputs)
        result[task] = entries
    return result


def noninferiority(task, method, baseline, vectors):
    one, two = vectors[method], vectors[baseline]
    conditions = []
    if task != 'coco':
        delta = one['accuracy']['replicates'] - two['accuracy']['replicates']
        lower = float(np.quantile(delta, .05))
        conditions.append(dict(metric='accuracy', difference=one['accuracy']['point']-two['accuracy']['point'],
            bound_type='one_sided_95_lower', bound=lower, permitted_decline=.02, proxy_pass=lower >= -.02))
    else:
        for key, factor, cap, upper in (('SPICE', .95, 0., False), ('CHAIRs', 1., .02, True)):
            a, b = one[key], two[key]
            if a['point'] is None or b['point'] is None:
                conditions.append(dict(metric=key, proxy_pass=False, status='undefined_required_metric'))
                continue
            distribution = a['replicates'] - factor*b['replicates']
            bound = float(np.quantile(distribution, .95 if upper else .05))
            conditions.append(dict(metric=key, difference=a['point']-b['point'],
                bound_type='one_sided_95_upper' if upper else 'one_sided_95_lower', bound=bound,
                baseline_multiplier=factor, permitted_difference=cap,
                proxy_pass=bound <= cap if upper else bound >= cap))
    return dict(task=task, method=method, baseline=baseline,
        automatic_proxy_pass=all(r['proxy_pass'] for r in conditions), conditions=conditions,
        scope='predeclared fixed-pilot automatic-metric diagnostic, no human quality guarantee',
        zero_discordance_bootstrap_can_be_degenerate=True,
        human_caption_reliability='pending' if task == 'coco' else 'not_evaluated')


def numeric_stats(mapping, prefix=''):
    """Flatten diagnostic counters only; no synchronization or added timing."""
    result = {}
    for key, value in mapping.items():
        name = prefix + key
        if isinstance(value, dict):
            result.update(numeric_stats(value, name + '.'))
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            result[name] = value
    return result


def runtime_diagnostics(task, method, samples, rows):
    selected = sorted((r for r in rows if r['method'] == method), key=lambda r: r['ordinal'])
    counts, every_path, numeric = [], [], []
    for row in selected:
        stats = row.get('signal_stats', {})
        n = int(stats.get('committed_mismatches', stats.get('committed_response_mismatches', 0)))
        counts.append(n)
        paths = row.get('acceptance_paths', [])
        mismatches = [int(p.get('mismatches', 0)) for p in paths]
        require(all(0 <= x <= 2 for x in mismatches), 'Per-round mismatch cap violated')
        if method != 'strict_shared':
            require(sum(mismatches) == n, 'Committed mismatch count disagrees with selected-path log')
        if method == 'bepc_cap4': require(n <= 4, 'Capped control exceeds four')
        if method == 'strict_shared': require(n == 0, 'Strict unexpectedly relaxed')
        every_path.extend(mismatches)
        numeric.append(numeric_stats(stats))
    features = []
    for key in sorted(set().union(*(r.keys() for r in numeric))):
        values = [r.get(key) for r in numeric]
        present = [v for v in values if v is not None]
        features.append(dict(task=task, method=method, counter=key, present_questions=len(present),
            total_questions=len(selected), sum=float(sum(present)), mean=float(np.mean(present)),
            maximum=float(max(present)), interpretation='diagnostic counter; not standalone synchronized feature latency'))
    summary = dict(committed_mismatch_p50=float(np.quantile(counts, .5)),
        committed_mismatch_p95=float(np.quantile(counts, .95)), committed_mismatch_max=max(counts),
        response_fraction_above_four_mismatches=float(np.mean(np.array(counts)>4)),
        actual_per_round_mismatch_max=max(every_path, default=0),
        per_round_budget_violations=sum(n>2 for n in every_path))
    return summary, features


def transitions(task, samples, scores, rows):
    score_lookup = {method: {str(r['question_id']): r for r in scores[method]['rows']} for method in METHODS}
    runtime = {(r['method'], str(r['question_id'])): r for r in rows}
    counts, cases = Counter(), []
    for sample in samples:
        qid = sample_id(sample)
        a, b = score_lookup['bepc_no_response_cap'][qid], score_lookup['bepc_cap4'][qid]
        uncapped, capped = runtime['bepc_no_response_cap', qid], runtime['bepc_cap4', qid]
        changed = uncapped['output_text'] != capped['output_text']
        counts['outputs_changed' if changed else 'outputs_unchanged'] += 1
        if task == 'coco':
            comparison = {}
            for key in ('SPICE', 'CHAIRs', 'CHAIRi', 'CIDEr'):
                av, bv = a.get(key), b.get(key)
                comparison[key+'_delta'] = None if av is None or bv is None else av-bv
                comparison[key+'_cap4'] = bv; comparison[key+'_uncapped'] = av
                counts[key+('_undefined' if av is None or bv is None else '_equal' if av==bv else '_increased' if av>bv else '_decreased')] += 1
        else:
            delta = a['score']-b['score']
            counts['score_improved' if delta>0 else 'score_degraded' if delta<0 else 'score_tie'] += 1
            comparison = dict(score_delta=delta, cap4_score=b['score'], uncapped_score=a['score'],
                cap4_answer=b.get('extracted_answer'), uncapped_answer=a.get('extracted_answer'),
                cap4_parse_status=b.get('answer_extraction_status'), uncapped_parse_status=a.get('answer_extraction_status'))
            if task == 'scienceqa': counts[f'{int(b["score"])}_to_{int(a["score"])}'] += 1
        if changed:
            cases.append(dict(task=task, question_id=qid, group_id=sample['group_id'], image_id=sample.get('image_id'),
                image_path=sample.get('image_path'), prompt_text=uncapped['prompt_text'],
                cap4_caption=capped['output_text'], uncapped_caption=uncapped['output_text'],
                cap4_committed_mismatches=capped.get('signal_stats', {}).get('committed_mismatches', 0),
                uncapped_committed_mismatches=uncapped.get('signal_stats', {}).get('committed_mismatches', 0),
                cap4_truncated=capped['cap_reached'], uncapped_truncated=uncapped['cap_reached'], **comparison))
    return dict(counts=dict(counts), changed_cases=cases,
        selection='all changed outputs retained without score/harm/length filtering')


def analyze(config, config_path, samples, runtime, scores, result_hashes):
    old = legacy_math()
    require(config.get('bootstrap', 5000) == 5000, '5000 paired bootstrap draws fixed')
    summaries, comparisons, quality, draw_seals, features = [], [], [], {}, []
    all_vectors = {}
    for index, task in enumerate(config['tasks']):
        groups, mapping, categories = old.group_structure(samples[task])
        seed = int(config.get('bootstrap_seed', 2026103109)) + index*100003
        draws = old.bootstrap_indices(groups, categories, 5000, seed, False)
        draw_seals[task] = dict(seed=seed, unit='physical image/group', groups=groups,
            shape=list(draws.shape), sha256=hashlib.sha256(draws.tobytes()).hexdigest())
        all_vectors[task] = {}
        for method in METHODS:
            vectors, summary = old.method_metrics(task, method, samples[task], runtime[task], scores[task][method],
                draws, groups, mapping, categories)
            variant = config['methods_by_task'][task][method]
            diag, costs = runtime_diagnostics(task, method, samples[task], runtime[task])
            summary.update(diag)
            summary.update(gamma=variant.get('gamma'), tau=variant.get('tau'),
                response_mismatch_budget=variant.get('response_mismatch_budget'),
                round_mismatch_budget=variant.get('round_mismatch_budget'),
                measurement_scope='new same-run single-pass paired timing')
            features.extend(costs); all_vectors[task][method] = vectors; summaries.append(summary)
        for metric in all_vectors[task]['bepc_cap4']:
            family = [old.paired_comparison(task, method, baseline, metric,
                all_vectors[task][method], all_vectors[task][baseline], family)
                for method, baseline, family in COMPARISONS]
            for row, adjusted in zip(family, old.holm([r['p'] for r in family])):
                row['holm_adjusted_p'] = adjusted
                row['holm_family'] = 'three_prespecified_comparisons_within_task_metric'
                if row['status']=='ok' and row['baseline_point']:
                    row['relative_change_percent'] = 100*row['difference']/row['baseline_point']
            comparisons.extend(family)
        for method, baseline, _ in COMPARISONS:
            quality.append(noninferiority(task, method, baseline, all_vectors[task]))
    parity = historical_parity(samples, runtime)
    transition_rows = {task: transitions(task, samples[task], scores[task], runtime[task]) for task in config['tasks']}
    prior = ROOT / 'audit/STATISTICAL_ANALYSIS.json'
    created = read(prior)['created_utc'] if prior.exists() else datetime.now(timezone.utc).isoformat()
    manifest = dict(status='completed_BEPC_response_cap_ablation', created_utc=created,
        config_sha256=sha(config_path), analyzer_sha256=sha(__file__), evaluator_sha256=sha(ROOT/'harness/evaluate_uncapped.py'),
        legacy_statistical_math_sha256=sha(LEGACY_ANALYZER), result_sha256=result_hashes,
        scoring_seals={task: {method: entry['seal_sha256'] for method, entry in batch.items()} for task,batch in scores.items()},
        bootstrap=5000, bootstrap_unit='physical image/group', draws=draw_seals,
        comparisons=[list(pair) for pair in COMPARISONS],
        p_method='two-sided centered paired-bootstrap add-one Monte Carlo approximation',
        multiple_comparisons='Holm within task/metric, three frozen pairs; not global across all metrics',
        CI_scope='paired image/group sampling; no repeated-run timing uncertainty; fixed official per-image scores',
        measured_main_requests=sum(len(rows) for rows in runtime.values()), repetitions=1,
        numerical_environment=dict(python=sys.version, numpy=np.__version__),
        immutable_previous_source_manifest_sha256=sha(ROOT/'audit/PRIOR_PRESERVATION.json'),
        historical_parity=parity, quality_diagnostics=quality, output_transitions=transition_rows,
        human_ground_truth_status='not available; no new human annotations or VLM Judge labels',
        historical_data_reused_by_user_request=True, independent_unseen_test_claim=False,
        prior_timing_not_used_in_primary_comparisons=True,
        engineering_note='uncapped never retires features for response-budget exhaustion; extra feature/controller cost stays in timing')
    return summaries, comparisons, manifest, features


def number(value, digits=2, scale=1):
    if value is None or value=='': return 'NA'
    value = float(value)*scale
    return f'{value:.3e}' if 0<abs(value)<10**(-digits) else f'{value:.{digits}f}'


def interval(row, key, scale=1):
    return f'{number(row[key], scale=scale)} [{number(row[key+"_ci95_low"], scale=scale)}, {number(row[key+"_ci95_high"], scale=scale)}]'


def historical_rows(tasks):
    path = OPTIMIZATION/'reports/BENCHMARK_SUMMARY.csv'
    with path.open() as handle: rows = [r for r in csv.DictReader(handle) if r['task'] in tasks]
    for row in rows:
        row.update(measurement_scope='historical_previous_optimization_descriptive_only', source_path=str(path), source_sha256=sha(path))
    return rows


def report(config, summaries, comparisons, manifest, historical):
    lines = ['# BEPC 去除每题四次 mismatch 上限：完整实验报告', '',
        '本轮在新目录运行，保留原 BEPC、上一轮工程优化与放宽版的源码/配置/结果。'
        '仅取消每题累计最多接受 4 个 mismatch 的约束；每轮最多 2 个 mismatch、双向 gamma=0.5、tau=1、'
        'U/P 秩8/4、文本窗口32、视觉池64、冲突门、EOS 保护、Target/Draft forward、KV 与树设置均保持不变。', '',
        '复用 ScienceQA、COCO、TextVQA 各20开发/100测试组；不重新抽样。T=0，原长输出提示、512-token预算，'
        '同一问题内三种方法随机执行顺序，每种方法只测一次。开发数据不用于选择新的 gamma/阈值或接受预算。', '',
        f'正式测试共 {manifest["measured_main_requests"]} 次请求；三种方法各300次。'
        '主比较使用本轮重新运行的 capped BEPC 与 Strict，历史时延仅描述性列出。', '',
        '## 核心结果', '']
    for task in config['tasks']:
        b = {r['method']: r for r in summaries if r['task']==task}
        a,c,s = (b[k] for k in METHODS[::-1])
        tps = 100*(a['throughput_tokens_per_second']/c['throughput_tokens_per_second']-1)
        latency = 100*(a['mean_seconds']/c['mean_seconds']-1)
        mat = 100*(a['effective_mean_accepted_draft_length']/c['effective_mean_accepted_draft_length']-1)
        q = ('SPICE '+number(c['SPICE'],scale=100)+'→'+number(a['SPICE'],scale=100)+
             '；CHAIRs '+number(c['CHAIRs'],scale=100)+'%→'+number(a['CHAIRs'],scale=100)+'%') if task=='coco' else (
             '最终答案 Accuracy '+number(c['accuracy'],scale=100)+'%→'+number(a['accuracy'],scale=100)+'%')
        lines.append(f'- {TITLES[task]}：uncapped 相对 cap4，token/s {tps:+.2f}%，请求时延 {latency:+.2f}%，'
            f'有效 MAT {mat:+.2f}%；{q}。平均提交 mismatch {c["mean_committed_mismatches"]:.2f}→'
            f'{a["mean_committed_mismatches"]:.2f}/题，uncapped 超过4次的比例 {100*a["response_fraction_above_four_mismatches"]:.1f}%。')
    lines += ['', 'token/s、MAT 和整题时延分别报告。输出长度改变时，更高 token/s 不一定意味着更快完成问题。'
        '质量不显著下降不能证明质量相等；以下非劣效门仅是预先规定的自动指标 pilot 检查。', '',
        '## 方法与唯一算法干预', '', '| 方法 | 每题累计 mismatch 上限 | 每轮上限 | BEPC 打分 |', '|---|---:|---:|---|',
        '| strict_shared | 0 | 0 | Target greedy 连续前缀 |',
        '| bepc_cap4 | 4 | 2 | 原双向 BEPC，gamma=0.5、tau=1 |',
        '| bepc_no_response_cap | 无 | 2 | 完全相同 |', '',
        '对原生 Target–Draft Margin g，冲突有效时 δ=0.5·sign(mP)·min(|mU|,|mP|,1)，否则δ=0；'
        '接受条件 g−δ≤1。仅合法选中分支的实际 mismatch 才计数；扫描兄弟分支不消耗预算。'
        '取消 response cap 不等于取消逐轮保护，也不等于接受所有 Draft。U/P 是几何代理，不是视觉/语言因果分离。', '',
        'cap4 预算耗尽后可退休 BEPC observer 并进入 Strict 快速路径；uncapped 因没有累计预算，不会执行这种退休。'
        '因此 uncapped 额外特征维护的代价全部计入时延；不能把理论接受增长直接当作净加速。', '',
        '## 速度、MAT、显存与生成质量', '',
        'token/s=首 EOS/512 可见范围内内容 token 总数÷同步模型请求总秒数，不含EOS；模型请求包括视觉编码、prefill、'
        '控制器和Target/Draft解码。PIL/processor/H2D预处理单独记录并在方法间复用，所以不是完整客户端E2E。'
        '有效MAT=实际可见范围内接受的Draft token总数÷验证轮数，排除每轮root；原生MAT保留官方整块计数。', '']
    for task in config['tasks']:
        batch = [r for r in summaries if r['task']==task]
        lines += [f'### {TITLES[task]}', '',
            '| 方法 | token/s [95% CI] | 请求ms [95% CI] | decode token/s | 原生MAT | 有效MAT | 峰值allocated GiB |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for r in batch:
            lines.append('| '+' | '.join([r['method'],interval(r,'throughput_tokens_per_second'),interval(r,'mean_seconds',1000),
                number(r['decode_throughput_tokens_per_second']),number(r['native_mean_accepted_draft_length']),
                number(r['effective_mean_accepted_draft_length']),number(r['max_peak_allocated_bytes'],scale=1/2**30)])+' |')
        if task=='coco':
            lines += ['', '| 方法 | CIDEr-D原尺度 | SPICE % [95% CI] | CHAIRs % [95% CI] | CHAIRi % |', '|---|---:|---:|---:|---:|']
            for r in batch:
                lines.append('| '+' | '.join([r['method'],number(r['CIDEr'],6),interval(r,'SPICE',100),interval(r,'CHAIRs',100),number(r['CHAIRi'],scale=100)])+' |')
        else:
            lines += ['', '| 方法 | 官方最终答案Accuracy % [95% CI] | 答案提取失败 |', '|---|---:|---:|']
            for r in batch: lines.append(f'| {r["method"]} | {interval(r,"accuracy",100)} | {r["answer_extraction_failures"]}/{r["questions"]} |')
        lines += ['', '| 方法 | token/题 | 词/题 | 截断% | mismatch均值 | p50/p95/max | 超过4次% | 实际每轮max |', '|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in batch:
            lines.append('| '+' | '.join([r['method'],number(r['mean_visible_tokens']),number(r['mean_output_words']),number(r['truncation_fraction'],scale=100),
                number(r['mean_committed_mismatches']),'/'.join(number(r[k],1) for k in ('committed_mismatch_p50','committed_mismatch_p95','committed_mismatch_max')),
                number(r['response_fraction_above_four_mismatches'],scale=100),str(r['actual_per_round_mismatch_max'])])+' |')
        lines += ['', '| 对比 | 指标 | 差值 [95% CI] | Holm p |', '|---|---|---:|---:|']
        wanted = ('mean_seconds','throughput_tokens_per_second','effective_mean_accepted_draft_length','SPICE' if task=='coco' else 'accuracy','CHAIRs' if task=='coco' else 'answer_extraction_failure_fraction')
        for r in comparisons:
            if r['task']!=task or r['metric'] not in wanted: continue
            lines.append(f'| {r["method"]}−{r["baseline"]} | {r["metric"]} | {number(r["difference"],4)} '
                f'[{number(r["ci95_low"],4)}, {number(r["ci95_high"],4)}] | {number(r.get("holm_adjusted_p"),4)} |')
        lines += ['', 'uncapped 相对 cap4 的完整转移计数：`'+json.dumps(manifest['output_transitions'][task]['counts'],ensure_ascii=False,sort_keys=True)+'`。', '']
    lines += ['## 预先规定的质量容忍检查', '',
        'QA允许最多2个百分点下降；COCO同时要求SPICE不少于对照95%、CHAIRs增幅不超过2个百分点。'
        '使用配对bootstrap单侧95%界，既比较uncapped/cap4，也分别比较两种BEPC/Strict；不能以p>0.05替代非劣效。', '',
        '| 任务 | 方法/对照 | 自动代理门 | 单侧界 |', '|---|---|---|---|']
    for r in manifest['quality_diagnostics']:
        details='; '.join(f'{v["metric"]}: {v.get("bound_type",v.get("status"))}={number(v.get("bound"),4)}, pass={v["proxy_pass"]}' for v in r['conditions'])
        lines.append(f'| {r["task"]} | {r["method"]}/{r["baseline"]} | {"PASS" if r["automatic_proxy_pass"] else "NOT ESTABLISHED"} | {details} |')
    lines += ['', '所有门仅限当前100题/任务固定pilot自动评价；零分歧时非参数bootstrap可能退化，不能证明总体罕见退化率为0。'
        'COCO没有新增双人盲评，不能把这些指标称为人工Ground Truth，也不能给出生成事实安全保证。', '',
        '## 额外特征成本与历史一致性', '',
        'FEATURE_COST_COUNTERS.csv保存每个数值计数器（含子空间、full-score、视觉预筛、observer构建/退休）的覆盖、总数、均值与最大值。'
        '这些是诊断计数，不将CPU enqueue耗时当作独立同步GPU feature时延。总请求/decode时间与显存已包含全部实际开销。', '']
    for task, entries in manifest['historical_parity'].items():
        for method, item in entries.items():
            lines.append(f'- {TITLES[task]} {method} 与上轮 {item["historical_method"]}：输出/路径字段一致 '
                f'{item["exact_questions"]}/{item["questions"]}；输入差异{len(item["input_differences"])}。所有差异保留，不擅自归因为BF16/KV/mask。')
    lines += ['', '## 与上轮工程优化实验的描述性对照', '',
        '以下历史速度来自不同时段，不参与本轮配对检验，也不用于选择最佳算法。原长提示、数据、生成预算相同。', '',
        '| 任务 | 上轮方法 | token/s | 请求ms | 有效MAT | 主质量 |', '|---|---|---:|---:|---:|---|']
    for r in historical:
        q=('SPICE '+number(r.get('SPICE'),scale=100)+' / CHAIRs '+number(r.get('CHAIRs'),scale=100)) if r['task']=='coco' else number(r.get('accuracy'),scale=100)+'%'
        lines.append('| '+' | '.join([r['task'],r['method'],number(r['throughput_tokens_per_second']),number(r['mean_seconds'],scale=1000),number(r['effective_mean_accepted_draft_length']),q])+' |')
    lines += ['', '## 评价实现、统计与局限', '',
        '- ScienceQA沿用官方多选Accuracy和冻结末行答案解析器；TextVQA沿用官方VQA-style leave-one-out归一化。提取失败和歧义保留并计0；只评最终答案，不审计长解释。',
        '- COCO整段输出原样进入官方PTB、CIDEr-D、SPICE和作者CHAIR。SPICE沿用已验证的64G资源上限、新缓存，未改scorer/模型/公式。全量有序样本、标签、原文与签名逐字相同时才复用已有完整语料评分；不复用速度。',
        '- 原始解析失败、生成截断和所有正负改变样本均保留；CHANGED_OUTPUT_PAIRS.jsonl包含全部改变输出的图像引用、原文及分数，不能把自动分数变化直接等同具体事实伤害。',
        '- 图像/group配对bootstrap5000次，三个预先对比按task/metric做Holm校正，不控制全部任务/指标全局错误率；CI只量化图像抽样，不量化单次计时运行噪声。',
        '- 当前样本沿用旧实验，没有项目未见独立性；没有根据输出、标签或是否变好重新抽样，也没有根据测试结果调gamma/预算/阈值。',
        '- 长描述相对短参考使CIDEr极小；SPICE受参考覆盖限制，CHAIR只覆盖COCO80对象幻觉，不覆盖全部属性/关系/动作/遗漏。',
        '- 取消累计预算会改变后续前缀和可见事件，是完整解码政策比较；不将所有新增mismatch称为单事件因果效应，也不训练新的验证器。',
        '- 采用uncapped必须综合整题时延、token/s、MAT、输出长度、最终答案、对象幻觉和人工质量。当前固定pilot无法承诺所有输入上更快且保质。', '',
        '证据入口：configs/frozen_config.json、results/main/、metrics/main/、audit/STATISTICAL_ANALYSIS.json。'
        '图表见BENCHMARK_PERFORMANCE.pdf、BENCHMARK_QUALITY.pdf；主表与配对差值分别为BENCHMARK_SUMMARY.csv、PAIRED_COMPARISONS.csv。', '']
    return '\n'.join(lines)


def plot_payload(config, summaries, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    colors=['#a8afb8','#287eaf','#d58b3e']; labels=['Strict','BEPC cap4','BEPC no response cap']
    with PdfPages(destination/'BENCHMARK_PERFORMANCE.pdf') as pdf:
        for metric,scale,title in [('throughput_tokens_per_second',1,'Visible content tokens/s'),('mean_seconds',1000,'Full model request latency (ms)'),('effective_mean_accepted_draft_length',1,'Effective accepted Draft tokens / round')]:
            fig,axes=plt.subplots(1,3,figsize=(13,4.5),constrained_layout=True)
            for ax,task in zip(axes,config['tasks']):
                rows=[r for r in summaries if r['task']==task]; x=np.arange(len(rows))
                y=np.array([r[metric]*scale for r in rows]); lo=np.array([r[metric+'_ci95_low']*scale for r in rows]); hi=np.array([r[metric+'_ci95_high']*scale for r in rows])
                ax.bar(x,y,color=colors); ax.errorbar(x,y,yerr=np.maximum(0,np.vstack((y-lo,hi-y))),fmt='none',color='black',capsize=3)
                ax.set_xticks(x,labels,rotation=30,ha='right',fontsize=8); ax.set_title(TITLES[task]); ax.set_ylabel(title); ax.grid(axis='y',alpha=.2)
            fig.suptitle('One pass / long prompts / 512 tokens: paired image-bootstrap 95% CI'); pdf.savefig(fig); plt.close(fig)
    with PdfPages(destination/'BENCHMARK_QUALITY.pdf') as pdf:
        fig,axes=plt.subplots(1,3,figsize=(13,4.5),constrained_layout=True)
        for ax,task in zip(axes,config['tasks']):
            metric='SPICE' if task=='coco' else 'accuracy'; rows=[r for r in summaries if r['task']==task]; x=np.arange(len(rows))
            y=np.array([r[metric]*100 for r in rows]); lo=np.array([r[metric+'_ci95_low']*100 for r in rows]); hi=np.array([r[metric+'_ci95_high']*100 for r in rows])
            ax.bar(x,y,color=colors); ax.errorbar(x,y,yerr=np.maximum(0,np.vstack((y-lo,hi-y))),fmt='none',color='black',capsize=3)
            ax.set_xticks(x,labels,rotation=30,ha='right',fontsize=8); ax.set_title(TITLES[task]); ax.set_ylabel(metric+' (%)'); ax.grid(axis='y',alpha=.2)
        fig.suptitle('Automatic final-answer/caption metrics; no human caption quality guarantee'); pdf.savefig(fig); plt.close(fig)
        rows=[r for r in summaries if r['task']=='coco']; fig,ax=plt.subplots(figsize=(7,4.5),constrained_layout=True); x=np.arange(len(rows))
        y=np.array([r['CHAIRs']*100 for r in rows]); lo=np.array([r['CHAIRs_ci95_low']*100 for r in rows]); hi=np.array([r['CHAIRs_ci95_high']*100 for r in rows])
        ax.bar(x,y,color=colors); ax.errorbar(x,y,yerr=np.maximum(0,np.vstack((y-lo,hi-y))),fmt='none',color='black',capsize=3)
        ax.set_xticks(x,labels,rotation=25,ha='right'); ax.set_ylabel('CHAIRs (%) — lower is better'); ax.set_title('COCO80 object hallucination only'); pdf.savefig(fig); plt.close(fig)


def self_test():
    old=legacy_math(); previous=list(sys.path); sys.path.insert(0,str(LEGACY_ANALYZER.parent))
    try: old.self_test()
    finally: sys.path[:]=previous
    vec={'a':{'accuracy':dict(point=.8,replicates=np.full(5000,.8))},'b':{'accuracy':dict(point=.8,replicates=np.full(5000,.8))}}
    require(noninferiority('scienceqa','a','b',vec)['automatic_proxy_pass'],'Equal QA pilot proxy')
    vec['a']['accuracy']=dict(point=.7,replicates=np.full(5000,.7))
    require(not noninferiority('textvqa','a','b',vec)['automatic_proxy_pass'],'Declining QA fails')
    require('e-' in number(1e-50,6),'Tiny CIDEr nonzero preserved')
    require(numeric_stats({'a':2,'nested':{'b':3},'not_counter':True})=={'a':2,'nested.b':3},'Nested cost counter flattening')
    synthetic=[dict(method='bepc_no_response_cap',ordinal=0,signal_stats={'committed_mismatches':6},acceptance_paths=[{'mismatches':2}]*3)]
    diag,_=runtime_diagnostics('scienceqa','bepc_no_response_cap',[{}],synthetic)
    require(diag['committed_mismatch_max']==6 and diag['response_fraction_above_four_mismatches']==1 and diag['actual_per_round_mismatch_max']==2,'No response cap still round2')
    print('Uncapped paired analysis numerical tests passed; no inference or scoring')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/frozen_config.json')
    parser.add_argument('--self-test',action='store_true'); parser.add_argument('--plot-payload',type=Path)
    args=parser.parse_args()
    if args.self_test: self_test(); return
    if args.plot_payload:
        data=read(args.plot_payload); plot_payload(data['config'],data['summaries'],ROOT/'reports'); return
    config=read(args.config); samples,rows,result_hashes=load_phase(config,args.config,'main')
    scores=score_phase(config,args.config,'main',samples,rows,result_hashes)
    summaries,comparisons,manifest,features=analyze(config,args.config,samples,rows,scores,result_hashes)
    historical=historical_rows(config['tasks']); destination=ROOT/'reports'; destination.mkdir(parents=True,exist_ok=True)
    csv_once(destination/'BENCHMARK_SUMMARY.csv',summaries); csv_once(destination/'PAIRED_COMPARISONS.csv',comparisons)
    csv_once(destination/'HISTORICAL_PREVIOUS_SUMMARY.csv',historical); csv_once(destination/'FEATURE_COST_COUNTERS.csv',features)
    write_once(ROOT/'audit/STATISTICAL_ANALYSIS.json',manifest)
    changed=[case for task in config['tasks'] for case in manifest['output_transitions'][task]['changed_cases']]
    changed_payload=''.join(json.dumps(case,ensure_ascii=False,allow_nan=False)+'\n' for case in changed)
    changed_path=destination/'CHANGED_OUTPUT_PAIRS.jsonl'
    require(not changed_path.exists() or changed_path.read_text()==changed_payload,'Preserve all changed outputs')
    changed_path.write_text(changed_payload)
    text=report(config,summaries,comparisons,manifest,historical); path=destination/'FULL_EXPERIMENT_REPORT.md'
    require(not path.exists() or path.read_text()==text,'Preserve report'); path.write_text(text,encoding='utf-8')
    payload=ROOT/'audit/PLOT_PAYLOAD.json'; write_once(payload,dict(config=config,summaries=summaries))
    pdfs=[destination/'BENCHMARK_PERFORMANCE.pdf',destination/'BENCHMARK_QUALITY.pdf']
    if not all(p.exists() and p.stat().st_size for p in pdfs):
        if importlib.util.find_spec('matplotlib'): plot_payload(config,summaries,destination)
        else:
            plotting_python=Path('/home/wbb/miniconda3/bin/python'); require(plotting_python.exists(),'Plotting environment absent')
            subprocess.run([str(plotting_python),str(Path(__file__).resolve()),'--plot-payload',str(payload)],check=True)
    artifacts=[path,changed_path,destination/'BENCHMARK_SUMMARY.csv',destination/'PAIRED_COMPARISONS.csv',destination/'FEATURE_COST_COUNTERS.csv',destination/'HISTORICAL_PREVIOUS_SUMMARY.csv',*pdfs,ROOT/'audit/STATISTICAL_ANALYSIS.json']
    write_once(ROOT/'audit/REPORT_DELIVERY.json',dict(status='completed',artifacts_sha256={str(p):sha(p) for p in artifacts}))
    print(json.dumps(dict(status='completed',main_requests=manifest['measured_main_requests'],report=str(path),changed_outputs=len(changed)),ensure_ascii=False))


if __name__=='__main__': main()
