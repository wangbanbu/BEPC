"""Frozen three-task paired analysis for isolated BEPC optimization.

No model inference, classifier, signal fusion, threshold selection, output
normalization, resampling of questions or modification of previous artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

from evaluate_experiment import (ROOT, PREVIOUS, METHODS, TASKS, require, read, sha,
    write_once, load_phase, score_phase, sample_id)

LEGACY_ANALYZER = PREVIOUS / 'harness/analyze_results.py'
COMPARISONS = (
    ('bepc_fast', 'bepc_original', 'engineering_equivalence_speed'),
    ('bepc_rescue0625', 'bepc_fast', 'directional_rescue_increment'),
    ('bepc_original', 'strict_shared', 'original_relaxation_vs_strict'),
    ('bepc_rescue0625', 'strict_shared', 'rescue_vs_strict'),
)
PARITY_FIELDS = ('raw_token_ids', 'returned_token_ids', 'output_text', 'acceptance_length',
                 'acceptance_paths', 'eos_index', 'cap_reached')
TITLES = {'scienceqa': 'ScienceQA', 'textvqa': 'TextVQA', 'coco': 'COCO Caption'}


def legacy_math():
    # Only pure statistical functions and task metrics are used. Their own main
    # entry point, previous output ROOT, selectors and write functions are not run.
    previous_path = list(sys.path)
    sys.path.insert(0, str(LEGACY_ANALYZER.parent))
    try:
        spec = importlib.util.spec_from_file_location('_read_only_legacy_statistical_math', LEGACY_ANALYZER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = previous_path
    return module


def csv_once(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import io
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=sorted({key for row in rows for key in row}))
    writer.writeheader()
    writer.writerows(rows)
    payload = buffer.getvalue()
    if path.exists():
        require(path.read_text() == payload.replace('\r\n', '\n'), 'Preserve existing CSV: ' + str(path))
    else:
        path.write_text(payload)


def exact_parity(samples, runtime_rows):
    result = {}
    for task in samples:
        lookup = {(row['method'], row['ordinal']): row for row in runtime_rows[task]}
        differences = []
        for ordinal, sample in enumerate(samples[task]):
            one, two = lookup['bepc_fast', ordinal], lookup['bepc_original', ordinal]
            changed = [key for key in PARITY_FIELDS if one[key] != two[key]]
            if changed:
                differences.append(dict(question_id=sample_id(sample), ordinal=ordinal, fields=changed))
        historical = {}
        for shard in (0, 1):
            path = PREVIOUS / 'results/main' / f'main_{task}_shard{shard}.jsonl'
            marker = read(path.with_suffix('.complete.json'))
            require(marker['result_sha256'] == sha(path), 'Historical result SHA changed')
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row['method'] == 'bepc_algorithm' and row['repeat'] == 0:
                    historical[str(row['question_id'])] = row
        old_differences, input_differences = [], []
        for ordinal, sample in enumerate(samples[task]):
            one = lookup['bepc_original', ordinal]
            two = historical[sample_id(sample)]
            inputs = [key for key in ('prompt_token_ids', 'input_tensor_sha256', 'image_grid_thw', 'prompt_text') if one[key] != two[key]]
            changed = [key for key in PARITY_FIELDS if one[key] != two[key]]
            if inputs:
                input_differences.append(dict(question_id=sample_id(sample), fields=inputs))
            if changed:
                old_differences.append(dict(question_id=sample_id(sample), fields=changed))
        result[task] = dict(questions=len(samples[task]), exact_fields=list(PARITY_FIELDS),
            fast_original_exact_questions=len(samples[task]) - len(differences),
            fast_original_differences=differences,
            original_historical_exact_questions=len(samples[task]) - len(old_differences),
            original_historical_differences=old_differences, historical_input_differences=input_differences)
    return result


def noninferiority(task, method, baseline, vectors):
    one, two = vectors[method], vectors[baseline]
    conditions = []
    if task != 'coco':
        delta = one['accuracy']['replicates'] - two['accuracy']['replicates']
        lower = float(np.quantile(delta, .05))
        conditions.append(dict(metric='accuracy', difference=one['accuracy']['point'] - two['accuracy']['point'],
            bound_type='one_sided_95_lower', bound=lower, permitted_decline=.02, proxy_pass=lower >= -.02))
    else:
        for key, factor, cap, upper in (('SPICE', .95, 0., False), ('CHAIRs', 1., .02, True)):
            a, b = one[key], two[key]
            if a['point'] is None or b['point'] is None:
                conditions.append(dict(metric=key, proxy_pass=False, status='undefined_required_metric'))
                continue
            distribution = a['replicates'] - factor * b['replicates']
            bound = float(np.quantile(distribution, .95 if upper else .05))
            conditions.append(dict(metric=key, difference=a['point'] - b['point'],
                bound_type='one_sided_95_upper' if upper else 'one_sided_95_lower', bound=bound,
                baseline_multiplier=factor, permitted_difference=cap,
                proxy_pass=bound <= cap if upper else bound >= cap))
    return dict(task=task, method=method, baseline=baseline,
        automatic_proxy_pass=all(row['proxy_pass'] for row in conditions), conditions=conditions,
        scope='predeclared finite pilot automatic-metric diagnostic; no human safety/equivalence claim',
        zero_discordance_nonparametric_bootstrap_can_be_degenerate=True,
        human_caption_reliability='pending' if task == 'coco' else 'not_evaluated')


def transitions(task, samples, scores, rows):
    one = {str(row['question_id']): row for row in scores['bepc_rescue0625']['rows']}
    two = {str(row['question_id']): row for row in scores['bepc_fast']['rows']}
    runtime = {(row['method'], str(row['question_id'])): row for row in rows}
    counts, cases = Counter(), []
    for sample in samples:
        qid = sample_id(sample)
        a, b = one[qid], two[qid]
        changed = runtime['bepc_rescue0625', qid]['output_text'] != runtime['bepc_fast', qid]['output_text']
        counts['outputs_changed' if changed else 'outputs_unchanged'] += 1
        if task == 'coco':
            comparison = {}
            for key in ('SPICE', 'CHAIRs', 'CHAIRi', 'CIDEr'):
                av, bv = a.get(key), b.get(key)
                comparison[key] = None if av is None or bv is None else av - bv
                if av is None or bv is None:
                    counts[key + '_undefined'] += 1
                elif av == bv:
                    counts[key + '_equal'] += 1
                else:
                    counts[key + ('_increased' if av > bv else '_decreased')] += 1
        else:
            delta = a['score'] - b['score']
            counts['score_improved' if delta > 0 else 'score_degraded' if delta < 0 else 'score_tie'] += 1
            comparison = dict(score_delta=delta, baseline_score=b['score'], rescue_score=a['score'],
                baseline_answer=b.get('extracted_answer'), rescue_answer=a.get('extracted_answer'))
            if task == 'scienceqa':
                counts[f'{int(b["score"])}_to_{int(a["score"])}'] += 1
        if changed:
            cases.append(dict(question_id=qid, group_id=sample['group_id'], image_id=sample.get('image_id'),
                image_path=sample.get('image_path'), **comparison))
    return dict(counts=dict(counts), changed_cases=cases,
        selection='all changed outputs retained; no harmful-case filtering')


def analyze(config, config_path, samples, runtime, scores, result_hashes):
    old = legacy_math()
    require(config.get('bootstrap', 5000) == 5000, 'Fixed 5000 paired bootstrap draws')
    summaries, comparisons, ni_rows, draw_seals = [], [], [], {}
    all_vectors = {}
    for index, task in enumerate(config['tasks']):
        groups, mapping, categories = old.group_structure(samples[task])
        seed = int(config.get('bootstrap_seed', 20261101)) + index * 100003
        draws = old.bootstrap_indices(groups, categories, 5000, seed, False)
        draw_seals[task] = dict(seed=seed, unit='physical image/group', groups=groups,
            shape=list(draws.shape), sha256=hashlib.sha256(draws.tobytes()).hexdigest())
        all_vectors[task] = {}
        for method in METHODS:
            vectors, summary = old.method_metrics(task, method, samples[task], runtime[task], scores[task][method],
                draws, groups, mapping, categories)
            variant = config['methods_by_task'][task][method]
            summary.update(gamma_rescue=variant.get('gamma_rescue', variant.get('gamma')),
                gamma_protect=variant.get('gamma_protect', variant.get('gamma')), tau=variant.get('tau'),
                measurement_scope='new same-run single-pass paired timing')
            all_vectors[task][method] = vectors
            summaries.append(summary)
        for metric in all_vectors[task]['bepc_fast']:
            family = [old.paired_comparison(task, method, baseline, metric,
                all_vectors[task][method], all_vectors[task][baseline], family)
                for method, baseline, family in COMPARISONS]
            for row, adjusted in zip(family, old.holm([row['p'] for row in family])):
                row['holm_adjusted_p'] = adjusted
                row['holm_family'] = 'four_prespecified_comparisons_within_task_metric'
                if row['status'] == 'ok' and row['baseline_point']:
                    row['relative_change_percent'] = 100 * row['difference'] / row['baseline_point']
            comparisons.extend(family)
        for method, baseline, _ in COMPARISONS:
            ni_rows.append(noninferiority(task, method, baseline, all_vectors[task]))
    parity = exact_parity(samples, runtime)
    transition_rows = {task: transitions(task, samples[task], scores[task], runtime[task]) for task in config['tasks']}
    previous_manifest = ROOT / 'audit/STATISTICAL_ANALYSIS.json'
    created_utc = read(previous_manifest)['created_utc'] if previous_manifest.exists() else datetime.now(timezone.utc).isoformat()
    manifest = dict(status='completed_three_task_isolated_BEPC_analysis', created_utc=created_utc,
        config_sha256=sha(config_path), analyzer_sha256=sha(__file__), evaluator_sha256=sha(ROOT / 'harness/evaluate_experiment.py'),
        legacy_statistical_math_sha256=sha(LEGACY_ANALYZER), result_sha256=result_hashes,
        scoring_seals={task: {method: entry['seal_sha256'] for method, entry in methods.items()} for task, methods in scores.items()},
        bootstrap=5000, bootstrap_unit='physical image/group', draws=draw_seals,
        comparisons=[list(pair) for pair in COMPARISONS],
        p_method='two-sided centered paired-bootstrap add-one Monte Carlo approximation',
        multiple_comparisons='Holm within task/metric, four frozen pairs; not global across tasks/metrics',
        CI_scope='pointwise paired image/group sampling; no repeated-run timing uncertainty; fixed official per-image scores',
        measured_main_requests=sum(len(rows) for rows in runtime.values()), repetitions=1,
        numerical_environment=dict(python=sys.version, numpy=np.__version__),
        immutable_previous_source_manifest_sha256=sha(ROOT / 'audit/PRIOR_PRESERVATION.json'),
        engineering_parity=parity, quality_diagnostics=ni_rows, output_transitions=transition_rows,
        human_ground_truth_status='not available; no new human annotations or VLM Judge labels',
        independent_unseen_test_claim=False, historical_data_reused_by_user_request=True,
        prior_timing_not_used_in_primary_comparisons=True)
    return summaries, comparisons, manifest


def number(value, digits=2, scale=1):
    if value is None or value == '':
        return 'NA'
    value = float(value) * scale
    return f'{value:.3e}' if 0 < abs(value) < 10 ** (-digits) else f'{value:.{digits}f}'


def interval(row, key, scale=1):
    return f'{number(row[key], scale=scale)} [{number(row[key+"_ci95_low"], scale=scale)}, {number(row[key+"_ci95_high"], scale=scale)}]'


def historical_rows(tasks):
    path = PREVIOUS / 'reports/BENCHMARK_SUMMARY.csv'
    with path.open() as handle:
        rows = [row for row in csv.DictReader(handle) if row['task'] in tasks]
    for row in rows:
        row['measurement_scope'] = 'historical_previous_single_pass_descriptive_only'
        row['source_path'] = str(path)
        row['source_sha256'] = sha(path)
    return rows


def report(config, summaries, comparisons, manifest, historical):
    parity = manifest['engineering_parity']
    exact = sum(item['fast_original_exact_questions'] for item in parity.values())
    total = sum(item['questions'] for item in parity.values())
    lines = ['# BEPC 独立代码优化与定向放宽：三任务完整实验报告', '',
        '本轮保留原 BEPC 代码、原五任务实验配置和结果，另建独立源码目录。复用上轮 ScienceQA、COCO、TextVQA '
        '各 20 组开发、100 组测试图像及原长输出提示，T=0、512-token 上限、Draft 树、EOS、KV 更新与每轮/每题 2/4 mismatch 预算不变。'
        '原版 BEPC、工程优化版和单向小幅增强 rescue 版在同轮重新计时；历史七方法仅作描述性对照。', '',
        f'主测试共 {manifest["measured_main_requests"]} 次请求，四个方法各 300 次；每问题每方法只测一次。'
        f'工程版与本轮原版在 {exact}/{total} 题的全部冻结输出/接受路径字段完全一致。'
        '这验证当前样本上的工程等价性，不能代替对全部输入的形式证明。', '',
        '## 核心结论与边界', '']
    for task in config['tasks']:
        batch = {row['method']: row for row in summaries if row['task'] == task}
        fast, original, rescue = (batch[name] for name in ('bepc_fast', 'bepc_original', 'bepc_rescue0625'))
        throughput = 100 * (fast['throughput_tokens_per_second'] / original['throughput_tokens_per_second'] - 1)
        latency = 100 * (fast['mean_seconds'] / original['mean_seconds'] - 1)
        q = ('SPICE ' + number(rescue['SPICE'], scale=100) + '%、CHAIRs ' + number(rescue['CHAIRs'], scale=100) + '%'
             if task == 'coco' else '最终答案 Accuracy ' + number(rescue['accuracy'], scale=100) + '%')
        p = parity[task]
        lines.append(f'- {TITLES[task]}：工程版 token/s 相对原版变化 {throughput:+.2f}%，整题请求时延变化 {latency:+.2f}%；'
            f'工程版逐字段一致 {p["fast_original_exact_questions"]}/{p["questions"]}。rescue 版为 {q}，'
            f'改变输出 {manifest["output_transitions"][task]["counts"].get("outputs_changed", 0)}/{p["questions"]} 题。')
    lines += ['', '这些结果同时保留增益与退化。token/s、MAT 增加不代表整题更快，质量差异不显著也不能推出质量相等。'
        '所有 CI 仅反映图像抽样变化；一次计时无法量化运行间噪声。COCO 未新增人工盲评，自动指标通过仍不能称为事实安全保证。', '',
        '## 方法与冻结的改动', '',
        '| 方法 | 接受规则/改动 |', '|---|---|',
        '| strict_shared | 同一公共工程路径的 T=0 Strict；仅接受 Target greedy token |',
        '| bepc_original | 原双向 BEPC，tau=1、gamma=0.5、rank U/P=8/4、历史文本窗口32、视觉池64、wR=1 |',
        '| bepc_fast | 保持同一打分与决策；独立代码实现工程优化，具体实现和校验见本目录协议/审计 |',
        '| bepc_rescue0625 | 与工程版相同，只将视觉支持 Draft 且文本支持 Target 的 rescue 幅度 0.5→0.625；protection=0.5 |', '',
        '记 Target–Draft 原生 Margin 为 g，视觉/文本正交子空间候选读出为 mU、mP。'
        '冲突门、有效性门、min 截断、tau=1 和预算均不变：rescue 分支 delta=+0.625·min(|mU|,|mP|,1)，'
        'protection 分支 delta=-0.5·min(|mU|,|mP|,1)，其余 delta=0；接受条件 g-delta≤1。'
        '因此 rescue 理论 Margin 上界从 1.5 提至 1.625，不是所有 g≤1.625 一律放行。'
        'U/P 是几何代理，不是纯视觉证据/纯语言先验的因果分解。', '',
        '本轮没有训练分类器、融合新信号、改 Strict 内核、增加 Target forward 或从测试质量挑阈值；'
        '测试集及开发集沿用旧样本，不再具有项目未见性质。开发与测试图像仍按冻结清单分离。', '',
        '## 本轮配对速度、接受长度与显存', '',
        'token/s=可见内容 token 总数/同步模型请求秒数，EOS 不计；请求包含视觉编码、prefill、controller 和 Target/Draft 解码。'
        'PIL/processor/H2D 预处理单独计时并在方法间复用，故不是完整客户端 E2E。'
        '有效 MAT=首 EOS/512 预算内实际接受的 Draft token 总数/验证轮数，排除每轮 root；不对每请求 MAT 求平均。', '']
    for task in config['tasks']:
        batch = [row for row in summaries if row['task'] == task]
        lines += [f'### {TITLES[task]}', '',
            '| 方法 | 内容 token/s [95% CI] | 请求 ms [95% CI] | 原生 MAT | 有效 MAT | 含 root 推进量 | 峰值 allocated GiB |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for row in batch:
            lines.append('| ' + ' | '.join([row['method'], interval(row, 'throughput_tokens_per_second'),
                interval(row, 'mean_seconds', 1000), number(row['native_mean_accepted_draft_length']),
                number(row['effective_mean_accepted_draft_length']), number(row['effective_mean_committed_tokens_per_round']),
                number(row['max_peak_allocated_bytes'], scale=1 / 2**30)]) + ' |')
        if task == 'coco':
            lines += ['', '| 方法 | CIDEr-D 原尺度 | SPICE % [95% CI] | CHAIRs % [95% CI] | CHAIRi % |',
                '|---|---:|---:|---:|---:|']
            for row in batch:
                lines.append('| ' + ' | '.join([row['method'], number(row['CIDEr'], 6), interval(row, 'SPICE', 100),
                    interval(row, 'CHAIRs', 100), number(row['CHAIRi'], scale=100)]) + ' |')
        else:
            lines += ['', '| 方法 | 官方最终答案 Accuracy % [95% CI] | 答案提取失败 |', '|---|---:|---:|']
            for row in batch:
                lines.append(f'| {row["method"]} | {interval(row, "accuracy", 100)} | {row["answer_extraction_failures"]}/{row["questions"]} |')
        lines += ['', '| 方法 | token/题 | 词/题 | 截断 % | committed mismatch/题 | mismatch 响应 % |', '|---|---:|---:|---:|---:|---:|']
        for row in batch:
            lines.append('| ' + ' | '.join([row['method'], number(row['mean_visible_tokens']), number(row['mean_output_words']),
                number(row['truncation_fraction'], scale=100), number(row['mean_committed_mismatches']),
                number(row['response_fraction_with_committed_mismatch'], scale=100)]) + ' |')
        lines += ['', '| 对比 | 指标 | 差值 [95% CI] | Holm p |', '|---|---|---:|---:|']
        metrics = ('mean_seconds', 'throughput_tokens_per_second', 'effective_mean_accepted_draft_length',
                   'SPICE' if task == 'coco' else 'accuracy', 'CHAIRs' if task == 'coco' else 'answer_extraction_failure_fraction')
        for row in comparisons:
            if row['task'] != task or row['metric'] not in metrics:
                continue
            lines.append(f'| {row["method"]} − {row["baseline"]} | {row["metric"]} | '
                f'{number(row["difference"], 4)} [{number(row["ci95_low"], 4)}, {number(row["ci95_high"], 4)}] | '
                f'{number(row.get("holm_adjusted_p"), 4)} |')
        transition = manifest['output_transitions'][task]
        lines += ['', '放宽版相对工程版的全部转移统计：`' + json.dumps(transition['counts'], ensure_ascii=False, sort_keys=True) + '`。'
            '所有改变的 question_id、image/group_id 和分数差值完整保存在 audit/STATISTICAL_ANALYSIS.json；没有按正负结果筛选。', '']
    lines += ['## 事先规定的质量容忍检查', '',
        'QA 容忍最多 2 个百分点下降；COCO 要求 SPICE 不低于对照的 95%，CHAIRs 增幅不超过 2 个百分点。'
        '这里用配对 bootstrap 的单侧 95% 界作 pilot 自动代理检查，不将“p>0.05”当作非劣效。'
        'COCO 两门必须同时通过。所有界都在当前固定样本和自动指标条件下解释。', '',
        '| 任务 | 方法 / 对照 | 自动代理门 | 逐指标单侧界 |', '|---|---|---|---|']
    for row in manifest['quality_diagnostics']:
        details = '; '.join(f'{item["metric"]}: {item.get("bound_type", item.get("status"))}={number(item.get("bound"), 4)}, pass={item["proxy_pass"]}' for item in row['conditions'])
        lines.append(f'| {row["task"]} | {row["method"]} / {row["baseline"]} | {"PASS" if row["automatic_proxy_pass"] else "NOT ESTABLISHED"} | {details} |')
    lines += ['', '零分歧时非参数 bootstrap 可能产生退化为 0 的差值区间，无法证明未见图像上的罕见退化率为 0。'
        '尤其每任务仅 100 题，不能据此承诺总体生成质量不变。工程版还应依靠实现等价测试；定向放宽仍需独立样本及人工事实/覆盖审核。', '',
        '## 与上轮七方法的历史对照', '',
        '下表原样引用上轮同一批图像、长提示、512-token、单次运行的最终报告 CSV。'
        '它们属于不同运行时段，不能混入本轮配对显著性检验或据此给全部方法重新排名。'
        '本轮重新运行的 Strict 与原 BEPC 是连接两轮的控制，不复用历史速度。', '',
        '| 任务 | 上轮方法 | token/s | 请求 ms | 有效 MAT | 主质量 |', '|---|---|---:|---:|---:|---:|']
    for row in historical:
        quality = ('SPICE ' + number(row.get('SPICE'), scale=100) + '% / CHAIRs ' + number(row.get('CHAIRs'), scale=100) + '%'
                   if row['task'] == 'coco' else number(row.get('accuracy'), scale=100) + '%')
        lines.append('| ' + ' | '.join([row['task'], row['method'], number(row['throughput_tokens_per_second']),
            number(row['mean_seconds'], scale=1000), number(row['effective_mean_accepted_draft_length']), quality]) + ' |')
    lines += ['', '本轮原版与上轮原版的生成一致性：', '']
    for task, item in parity.items():
        lines.append(f'- {TITLES[task]}：冻结输出/路径字段一致 {item["original_historical_exact_questions"]}/{item["questions"]}；'
            f'输入不同 {len(item["historical_input_differences"])} 题。若有输出差异，完整保留，不能擅自归因为 BF16、KV 或 mask。')
    lines += ['', '## 官方评价与复用证据', '',
        'ScienceQA 使用原官方多选 Accuracy 与冻结末行答案提取；TextVQA 使用官方 VQA-style leave-one-out 归一化分数，'
        '未提取/多末行/越界答案保留并计 0。QA 评价最终答案，不审计长解释的事实或推理。', '',
        'COCO 使用相同官方 PTB、CIDEr-D、SPICE 与作者 CHAIR 实现，整段描述原样输入。'
        'SPICE 仅把 Java 最大 heap 从 16G 改为 64G，防止前轮长段落 OOM；scorer、模型、公式、线程与输入均不改变。'
        '仅完整有序样本、官方标签、原始输出文字逐字一致时复用既有完整语料分数；CIDEr DF 不改变。'
        '每方法 SCORE_PROVENANCE.json 区分新计算/历史或同轮复用并封存 SHA，不复用计时。'
        '缓存写入警告与 OOM/丢句分开：后者拒绝封存评分；前者仅影响 LMDB 持久化，不改变已解析元组的内存评分。', '',
        '## 局限与下一步', '',
        '- 一次请求计时、L20 双卡按图像分片；没有重复运行置信度。方法在样本内随机排序减少顺序影响，仍可能受热状态和 allocator 历史影响。',
        '- 点式 95% CI 基于 image/group 配对重采样 5000 次，所有方法共享同一组抽样；四组预先比较按 task/metric Holm 校正，不控制全部指标的全局错误率。',
        '- 图像沿用旧实验，开发/测试分离但不是新独立测试集。没有按输出长度、正确率、是否改善或失败状态剔除样本。',
        '- 长段落对短 caption 参考会使 CIDEr-D 极小；SPICE 受参考覆盖限制，CHAIR 仅测 COCO80 对象幻觉，不能代替属性、关系、动作、遗漏的人工评价。',
        '- 未新增自动 Judge 或人工 Ground Truth，不由自动分数推断具体 harmful mismatch 标签。U/P 候选分量不能解释为纯视觉/语言因果效应。',
        '- 只有当工程版路径/输出等价且实际请求时延改善时，才适合将其作为实现升级；定向放宽的采用须同时看速度、答案分数、对象幻觉和人工质量，不能只看 MAT。', '',
        '原始数据、模型身份、配置、source SHA、输入、结果封印见本目录 configs/、results/ 与 audit/。'
        '主表为 BENCHMARK_SUMMARY.csv，配对差值为 PAIRED_COMPARISONS.csv，历史表为 HISTORICAL_PREVIOUS_SUMMARY.csv；'
        '图见 BENCHMARK_PERFORMANCE.pdf 与 BENCHMARK_QUALITY.pdf。', '']
    return '\n'.join(lines)


def plot_payload(config, summaries, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    colors = ['#a8afb8', '#63758a', '#287eaf', '#d58b3e']
    labels = ['Strict', 'BEPC original', 'BEPC fast', 'BEPC rescue .625']
    with PdfPages(destination / 'BENCHMARK_PERFORMANCE.pdf') as pdf:
        for metric, scale, title in [('throughput_tokens_per_second', 1, 'Visible content tokens/s'),
                ('mean_seconds', 1000, 'Full model request latency (ms)'),
                ('effective_mean_accepted_draft_length', 1, 'Effective accepted Draft tokens / round')]:
            fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
            for ax, task in zip(axes, config['tasks']):
                rows = [row for row in summaries if row['task'] == task]
                y = np.array([row[metric] * scale for row in rows])
                lo = np.array([row[metric + '_ci95_low'] * scale for row in rows])
                hi = np.array([row[metric + '_ci95_high'] * scale for row in rows])
                x = np.arange(len(rows))
                ax.bar(x, y, color=colors)
                ax.errorbar(x, y, yerr=np.maximum(0, np.vstack((y-lo, hi-y))), fmt='none', color='black', capsize=3)
                ax.set_xticks(x, labels, rotation=35, ha='right', fontsize=8)
                ax.set_title(TITLES[task]); ax.set_ylabel(title); ax.grid(axis='y', alpha=.2)
            fig.suptitle('One pass / long prompts / 512 tokens: paired image bootstrap 95% CI')
            pdf.savefig(fig); plt.close(fig)
    with PdfPages(destination / 'BENCHMARK_QUALITY.pdf') as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
        for ax, task in zip(axes, config['tasks']):
            metric = 'SPICE' if task == 'coco' else 'accuracy'
            rows = [row for row in summaries if row['task'] == task]
            y = np.array([row[metric] * 100 for row in rows]); x = np.arange(len(rows))
            ax.bar(x, y, color=colors)
            lo = np.array([row[metric + '_ci95_low'] * 100 for row in rows]); hi = np.array([row[metric + '_ci95_high'] * 100 for row in rows])
            ax.errorbar(x, y, yerr=np.maximum(0, np.vstack((y-lo, hi-y))), fmt='none', color='black', capsize=3)
            ax.set_xticks(x, labels, rotation=35, ha='right', fontsize=8)
            ax.set_title(TITLES[task]); ax.set_ylabel(metric + ' (%)'); ax.grid(axis='y', alpha=.2)
        fig.suptitle('Automatic metrics; QA explanations and human caption quality not evaluated')
        pdf.savefig(fig); plt.close(fig)
        rows = [row for row in summaries if row['task'] == 'coco']
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        y = np.array([row['CHAIRs'] * 100 for row in rows]); x = np.arange(len(rows))
        ax.bar(x, y, color=colors)
        lo = np.array([row['CHAIRs_ci95_low'] * 100 for row in rows]); hi = np.array([row['CHAIRs_ci95_high'] * 100 for row in rows])
        ax.errorbar(x, y, yerr=np.maximum(0, np.vstack((y-lo, hi-y))), fmt='none', color='black', capsize=3)
        ax.set_xticks(x, labels, rotation=25, ha='right'); ax.set_ylabel('CHAIRs (%) — lower is better')
        ax.set_title('COCO80 object hallucination only; not complete factual ground truth')
        pdf.savefig(fig); plt.close(fig)


def self_test():
    old = legacy_math()
    previous_path = list(sys.path)
    sys.path.insert(0, str(LEGACY_ANALYZER.parent))
    try:
        old.self_test()
    finally:
        sys.path[:] = previous_path
    replicates = np.array([.8] * 5000)
    vectors = {'a': {'accuracy': dict(point=.8, replicates=replicates)},
               'b': {'accuracy': dict(point=.8, replicates=replicates)}}
    require(noninferiority('scienceqa', 'a', 'b', vectors)['automatic_proxy_pass'], 'Equal finite pilot proxy')
    vectors['a']['accuracy'] = dict(point=.7, replicates=np.array([.7] * 5000))
    require(not noninferiority('scienceqa', 'a', 'b', vectors)['automatic_proxy_pass'], 'QA harm fails proxy')
    require('e-' in number(1e-50, 6), 'Tiny nonzero CIDEr retained')
    print('New paired analysis self-tests passed; no inference or scoring')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/frozen_config.json')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--plot-payload', type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if args.plot_payload:
        data = read(args.plot_payload)
        plot_payload(data['config'], data['summaries'], ROOT / 'reports'); return
    config = read(args.config)
    samples, rows, result_hashes = load_phase(config, args.config, 'main')
    scores = score_phase(config, args.config, 'main', samples, rows, result_hashes)
    summaries, comparisons, manifest = analyze(config, args.config, samples, rows, scores, result_hashes)
    historical = historical_rows(config['tasks'])
    destination = ROOT / 'reports'
    destination.mkdir(parents=True, exist_ok=True)
    csv_once(destination / 'BENCHMARK_SUMMARY.csv', summaries)
    csv_once(destination / 'PAIRED_COMPARISONS.csv', comparisons)
    csv_once(destination / 'HISTORICAL_PREVIOUS_SUMMARY.csv', historical)
    write_once(ROOT / 'audit/STATISTICAL_ANALYSIS.json', manifest)
    text = report(config, summaries, comparisons, manifest, historical)
    path = destination / 'FULL_EXPERIMENT_REPORT.md'
    require(not path.exists() or path.read_text() == text, 'Preserve existing final report')
    path.write_text(text, encoding='utf-8')
    payload = ROOT / 'audit/PLOT_PAYLOAD.json'
    write_once(payload, dict(config=config, summaries=summaries))
    pdf_paths = [destination / 'BENCHMARK_PERFORMANCE.pdf', destination / 'BENCHMARK_QUALITY.pdf']
    if not all(path.exists() and path.stat().st_size > 0 for path in pdf_paths):
        if importlib.util.find_spec('matplotlib'):
            plot_payload(config, summaries, destination)
        else:
            plotting_python = Path('/home/wbb/miniconda3/bin/python')
            require(plotting_python.exists(), 'Read-only plot environment absent')
            subprocess.run([str(plotting_python), str(Path(__file__).resolve()), '--plot-payload', str(payload)], check=True)
    artifacts = [path, destination / 'BENCHMARK_SUMMARY.csv', destination / 'PAIRED_COMPARISONS.csv',
        destination / 'HISTORICAL_PREVIOUS_SUMMARY.csv', destination / 'BENCHMARK_PERFORMANCE.pdf',
        destination / 'BENCHMARK_QUALITY.pdf', ROOT / 'audit/STATISTICAL_ANALYSIS.json']
    write_once(ROOT / 'audit/REPORT_DELIVERY.json', dict(status='completed',
        artifacts_sha256={str(path): sha(path) for path in artifacts}))
    print(json.dumps(dict(status='completed', main_requests=manifest['measured_main_requests'],
        report=str(path), fast_original_exact=sum(row['fast_original_exact_questions'] for row in manifest['engineering_parity'].values())), ensure_ascii=False))


if __name__ == '__main__':
    main()
