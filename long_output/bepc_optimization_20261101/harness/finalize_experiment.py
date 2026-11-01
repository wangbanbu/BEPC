"""Seal final evidence and export all changed outputs without selecting examples."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(path, text):
    if path.exists():
        assert path.read_text(encoding='utf-8') == text, path
    else:
        path.write_text(text, encoding='utf-8')


def main():
    cfg = read(ROOT/'configs/frozen_config.json')
    records, score_seals, diagnostics = [], [], []
    for phase in ('development', 'main'):
        validation = read(ROOT/'audit'/f'{phase.upper()}_VALIDATION.json')
        assert validation['status'] == 'pass'
        for task in cfg['tasks']:
            for method in cfg['methods_by_task'][task]:
                folder = ROOT/'metrics'/phase/task/method
                seal = read(folder/'SCORING_SEAL.json')
                for field, filename in [('aggregate_sha256', 'aggregate.json'),
                    ('per_question_sha256', 'per_question_scores.json'),
                    ('score_provenance_sha256', 'SCORE_PROVENANCE.json')]:
                    assert sha(folder/filename) == seal[field], folder/filename
                score_seals.append(dict(path=str(folder/'SCORING_SEAL.json'), sha256=sha(folder/'SCORING_SEAL.json')))
    for task in cfg['tasks']:
        samples = read(cfg['data'][task]['test_path'])
        runtime = {}
        for shard in (0, 1):
            for line in (ROOT/f'results/main/main_{task}_shard{shard}.jsonl').read_text().splitlines():
                row = json.loads(line)
                runtime[str(row['question_id']), row['method']] = row
        scores = {method: {str(row['question_id']): row for row in read(
            ROOT/'metrics/main'/task/method/'per_question_scores.json')} for method in cfg['methods_by_task'][task]}
        for method in ('bepc_original', 'bepc_fast', 'bepc_rescue0625'):
            subset = [runtime[str(s['question_id']), method] for s in samples]
            used = [r['signal_stats']['committed_response_mismatches'] for r in subset]
            diagnostics.append(dict(task=task, method=method,
                mean_committed_mismatches=sum(used)/len(used),
                budget_exhausted_questions=sum(v == 4 for v in used),
                changed_outputs_vs_original=sum(r['returned_token_ids'] != runtime[str(r['question_id']), 'bepc_original']['returned_token_ids'] for r in subset),
                skipped_feature_rounds=sum(r['signal_stats'].get('exhausted_fast_rounds', 0) for r in subset),
                total_raw_rounds=sum(r['raw_verification_rounds'] for r in subset),
                mean_full_candidate_scores=sum(r['signal_stats'].get('full_candidate_scores', 0) for r in subset)/len(subset)))
        for sample in samples:
            qid = str(sample['question_id'])
            old, new = runtime[qid, 'bepc_fast'], runtime[qid, 'bepc_rescue0625']
            if old['returned_token_ids'] == new['returned_token_ids']:
                continue
            left, right = old['returned_token_ids'], new['returned_token_ids']
            divergence = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))
            entry = dict(task=task, image_id=sample['image_id'], question_id=qid,
                image_path=sample['image_path'], image_sha256=sample['image_sha256'],
                original_prompt=sample['prompt_text'], first_divergence_visible_token=divergence,
                original_bepc_output=old['output_text'], rescue0625_output=new['output_text'],
                strict_output=runtime[qid, 'strict_shared']['output_text'],
                comparison_scope='whole decoding-policy trajectory; not a single-event counterfactual label',
                automatic_only=True, human_quality_label=None,
                original_eos_index=old['eos_index'], rescue_eos_index=new['eos_index'],
                original_truncated=old['cap_reached'], rescue_truncated=new['cap_reached'])
            metric_keys = ('SPICE', 'CIDEr', 'CHAIRs', 'CHAIRi', 'hallucinated_mentions', 'object_mentions') if task == 'coco' else (
                'score', 'extracted_answer', 'answer_extraction_status', 'answer_extraction_reason')
            entry['original_official_metrics'] = {key: scores['bepc_fast'][qid].get(key) for key in metric_keys}
            entry['rescue_official_metrics'] = {key: scores['bepc_rescue0625'][qid].get(key) for key in metric_keys}
            records.append(entry)
    once(ROOT/'reports/CHANGED_OUTPUT_PAIRS.jsonl', ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in records))
    once(ROOT/'audit/MECHANISM_DIAGNOSTICS.json', json.dumps(diagnostics, ensure_ascii=False, indent=2)+'\n')
    report = ROOT/'reports/FULL_EXPERIMENT_REPORT.md'
    text = report.read_text(encoding='utf-8')
    heading = '\n## 执行与交付核验补充\n'
    if heading not in text:
        text += heading + (
            '\n旧代码与上轮结果的137项SHA在开发阶段、主测试结束及最终交付时重新核验；均未修改。'
            '工程版与原版在全部60组开发和300组测试上的原始token、可见token、接受长度及路径完全一致。'
            '完整变化案例逐条导出为 [CHANGED_OUTPUT_PAIRS.jsonl](CHANGED_OUTPUT_PAIRS.jsonl)，包含图像引用、原始提示、两条输出与自动评分，不按好坏筛选。\n'
            '\nCIDEr的图像bootstrap固定本次完整参考语料的document frequency和官方逐图分数；没有在每次重采样中重建DF，'
            '因此该CI是固定DF条件下的图像抽样区间。不能将其解释为重新构建参考语料的全部不确定性。\n'
            '\n开发集CPU官方评分与主测试早期的GPU生成有短暂重叠，可能产生主机资源竞争。GPU由每个分片进程独占，'
            '方法顺序按问题随机化；这些措施仍不能消除一次计时中的全部主机/频率噪声。小幅速度差异需重复计时再确认。\n'
            '\n统计方案中的三个主要对比及原BEPC相对Strict的背景对比全部保留；后者在读取新质量分数前通过'
            '`audit/STATISTICAL_PLAN_CLARIFICATION.json`明确纳入同一四比较Holm家族。没有据结果挑选系数。\n')
        text += '\n### 预算与实际放宽幅度\n\n| 数据集 | 方法 | 平均提交mismatch | 用满4次预算的题数/100 | 相对原版输出变化数/100 | 跳过BEPC特征的轮数/总轮数 |\n|---|---|---:|---:|---:|---:|\n'
        for item in diagnostics:
            text += (f"| {item['task']} | {item['method']} | {item['mean_committed_mismatches']:.2f} | "
                     f"{item['budget_exhausted_questions']} | {item['changed_outputs_vs_original']} | "
                     f"{item['skipped_feature_rounds']}/{item['total_raw_rounds']} |\n")
        text += ('\n救援系数增加只改变局部条件：对同一有效救援候选，新增接受区间为 '
                 '`1+0.5*v < g <= 1+0.625*v`，其中 `v=min(|m_U|,|m_P|,1)`。'
                 '区间最大宽度0.125；不增加每题预算。因此分数更宽松不必然增加整题提交的mismatch总数或显著减少验证轮数。'
                 '上表是已生成轨迹的描述性证据，不把预算饱和与速度变化直接解释为已识别的因果机制。\n')
        report.write_text(text, encoding='utf-8')
    before = read(ROOT/'audit/PRIOR_PRESERVATION.json')
    for item in before['files']:
        assert sha(item['path']) == item['sha256'], item['path']
    files = sorted([p for p in (ROOT/'reports').iterdir() if p.is_file()])
    files += [ROOT/'audit/MAIN_VALIDATION.json', ROOT/'audit/DEVELOPMENT_VALIDATION.json',
              ROOT/'audit/STATISTICAL_ANALYSIS.json', ROOT/'audit/MECHANISM_DIAGNOSTICS.json',
              ROOT/'configs/frozen_config.json', ROOT/'EXPERIMENT_PROTOCOL.md', Path(__file__)]
    manifest = dict(status='ready_for_independent_final_review', created_utc=datetime.now(timezone.utc).isoformat(),
        old_files_unchanged=len(before['files']), score_seals=score_seals,
        changed_output_pairs=len(records), artifacts=[dict(path=str(p), sha256=sha(p)) for p in files])
    path = ROOT/'audit/FINAL_DELIVERY.json'
    assert not path.exists(), path
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=manifest['status'], artifacts=len(files), changed_output_pairs=len(records),
        official_score_seals=len(score_seals), old_files_unchanged=len(before['files']))))


if __name__ == '__main__':
    main()
