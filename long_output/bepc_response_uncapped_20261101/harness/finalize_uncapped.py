"""Final preservation and scoring seals after the uncapped ablation report."""
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


def once(path, obj):
    assert not path.exists(), path
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    cfg = read(ROOT/'configs/frozen_config.json')
    source = read(ROOT/'audit/PRIOR_PRESERVATION.json')
    for entry in source['files']:
        assert sha(entry['path']) == entry['sha256'], entry['path']
    for name, expected in cfg['harness_sha256'].items():
        assert sha(ROOT/'harness'/name) == expected, name
    seals = []
    for phase in ('development', 'main'):
        validation = read(ROOT/'audit'/f'{phase.upper()}_VALIDATION.json')
        assert validation['status'] == 'pass'
        assert validation['records'] == (900 if phase == 'main' else 180)
        for task in cfg['tasks']:
            for method in cfg['methods_by_task'][task]:
                folder = ROOT/'metrics'/phase/task/method
                seal = read(folder/'SCORING_SEAL.json')
                for field, file in [('aggregate_sha256', 'aggregate.json'),
                    ('per_question_sha256', 'per_question_scores.json'),
                    ('score_provenance_sha256', 'SCORE_PROVENANCE.json')]:
                    assert sha(folder/file) == seal[field], folder/file
                seals.append(dict(path=str(folder/'SCORING_SEAL.json'), sha256=sha(folder/'SCORING_SEAL.json')))
    initial = ROOT/'audit/REPORT_DELIVERY.json'
    delivery = read(initial)
    for path, expected in delivery['artifacts_sha256'].items():
        assert sha(path) == expected, path
    report = ROOT/'reports/FULL_EXPERIMENT_REPORT.md'
    before = sha(report)
    validation = read(ROOT/'audit/MAIN_VALIDATION.json')
    text = report.read_text(encoding='utf-8')
    text += '\n## 最终执行与操纵核验\n\n'
    text += (f"旧代码、前两轮结果的{len(source['files'])}项SHA在运行前后均一致。"
             '180次开发与900次主测试全部保留；保留上限BEPC和Strict的输入、原始/可见token及选中路径均与上轮一致。'
             '本轮CPU官方评分仅在两个GPU生成进程都结束后运行，无官方评分与GPU计时重叠。\n\n')
    text += '| 任务 | uncapped实际超过4次/100 | 累计mismatch均值 | 最大值 | 首个选中路径差异已核验题数 |\n|---|---:|---:|---:|---:|\n'
    for task, entry in validation['tasks'].items():
        text += (f"| {task} | {entry['uncapped_over_four']} | {entry['uncapped_mean_mismatches']:.2f} | "
                 f"{entry['uncapped_max_mismatches']} | {len(entry['first_divergent_paths'])} |\n")
    text += ('\n在首个选中路径发生差异前，两策略采用相同前缀。核验要求差异出现时共享累计mismatch至少为3：'
             '累计不超过2时，一轮至多2次新接受不会碰到整题4次限制，因此两策略不应提前分叉。'
             '这项检查支持干预位置正确，不将整条轨迹的质量差异当作单事件Harmful因果标签。\n\n'
             'CIDEr的bootstrap固定完整参考语料DF与官方逐图分数，并非每次重采样重建DF。'
             '自动指标与QA最终答案只覆盖各自评价范围，人工图像条件事实/充分性评价仍未提供。\n\n'
             '本报告及全部当前文件的最终哈希以`audit/FINAL_DELIVERY.json`为准；'
             '`audit/REPORT_DELIVERY.json`保留补充章节前的生成器阶段封印，来源关系记录在REPORT_ADDENDUM_PROVENANCE。\n')
    report.write_text(text, encoding='utf-8')
    once(ROOT/'audit/REPORT_ADDENDUM_PROVENANCE.json', dict(
        initial_delivery_sha256=sha(initial), original_report_sha256=before,
        final_report_sha256=sha(report), reason='append independent execution/manipulation accounting only; unchanged scores/statistics'))
    helpers = ['run_uncapped.py', 'prepare_uncapped.py', 'validate_uncapped.py',
               'run_uncapped_pipeline.py', 'finalize_uncapped.py']
    files = sorted(p for p in (ROOT/'reports').iterdir() if p.is_file())
    files += [ROOT/'audit/DEVELOPMENT_VALIDATION.json', ROOT/'audit/MAIN_VALIDATION.json',
        ROOT/'audit/STATISTICAL_ANALYSIS.json', ROOT/'audit/REPORT_ADDENDUM_PROVENANCE.json',
        ROOT/'configs/frozen_config.json', ROOT/'EXPERIMENT_PROTOCOL.md']
    files += [ROOT/'harness'/name for name in helpers]
    once(ROOT/'audit/FINAL_DELIVERY.json', dict(status='ready_for_independent_final_review',
        created_utc=datetime.now(timezone.utc).isoformat(), old_files_unchanged=len(source['files']),
        official_score_seals=seals, artifacts=[dict(path=str(p), sha256=sha(p)) for p in files]))
    print(json.dumps(dict(status='ready_for_independent_final_review', artifacts=len(files), official_score_seals=len(seals))))


if __name__ == '__main__':
    main()
