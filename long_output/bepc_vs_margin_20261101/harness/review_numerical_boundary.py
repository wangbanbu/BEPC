"""Exact integer-count review of the ScienceQA noninferiority boundary.

Retains frozen inference, scorer, analysis source, thresholds and all raw
metrics. Only a derived pass/fail at an exactly equal integer-count boundary
is corrected. The original FP64 result and pre/post hashes are retained.
"""
from __future__ import annotations
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out=ROOT/'audit/NUMERICAL_BOUNDARY_REVIEW.json'
    assert not out.exists(),'Review already applied; preserve original record'
    manifest_path=ROOT/'audit/STATISTICAL_ANALYSIS.json'
    report_path=ROOT/'reports/FULL_EXPERIMENT_REPORT.md'
    before={str(p):sha(p) for p in (manifest_path,report_path)}
    cfg=read(ROOT/'configs/frozen_config.json')
    manifest=read(manifest_path)
    samples=read(cfg['data']['scienceqa']['test_path'])
    groups=manifest['draws']['scienceqa']['groups']
    assert len(groups)==len(samples)==100
    assert len(set(s['group_id'] for s in samples))==100
    seed=manifest['draws']['scienceqa']['seed']
    draws=np.random.default_rng(seed).integers(0,100,size=(5000,100),dtype=np.int64)
    draw_sha=hashlib.sha256(draws.tobytes()).hexdigest()
    assert draw_sha==manifest['draws']['scienceqa']['sha256']
    sample_for_group={str(s['group_id']):str(s['question_id']) for s in samples}
    values=[]
    for method in ('bepc_no_response_cap','margin_no_response_cap'):
        rows=read(ROOT/f'metrics/main/scienceqa/{method}/per_question_scores.json')
        scores={str(r['question_id']):r['score'] for r in rows}
        assert all(v in (0,1) for v in scores.values())
        values.append(np.array([int(scores[sample_for_group[g]]) for g in groups],dtype=np.int64))
    counts=(values[0]-values[1])[draws].sum(axis=1)
    ordered=np.sort(counts)
    # Exact counterpart of NumPy's fixed linear 5th percentile convention.
    position=Fraction(len(ordered)-1,20)
    lower_index=position.numerator//position.denominator
    weight=position-lower_index
    lower_count=(1-weight)*int(ordered[lower_index])+weight*int(ordered[lower_index+1])
    lower=lower_count/len(groups)
    limit=-Fraction(str(cfg['noninferiority_plan']['scienceqa']['max_absolute_drop']))
    reviewed_pass=lower>=limit
    assert lower==limit==Fraction(-1,50) and reviewed_pass
    entry=next(x for x in manifest['quality_diagnostics'] if x['task']=='scienceqa')
    original=copy.deepcopy(entry)
    assert len(entry['conditions'])==1
    assert abs(entry['conditions'][0]['bound']-float(lower))<1e-15
    entry['conditions'][0].update(bound=float(lower),proxy_pass=reviewed_pass,
        exact_bound_fraction=str(lower),exact_paired_count_quantile=str(lower_count),
        boundary_review='same frozen draws; integer paired correct counts; unchanged linear percentile')
    entry['automatic_proxy_pass']=reviewed_pass
    entry['raw_FP64_before_boundary_review']=original
    manifest['numerical_boundary_review']=dict(path=str(out),implementation_sha256=sha(__file__),
        affected='ScienceQA automatic NI gate only; unchanged metrics, labels, confidence intervals and thresholds')
    text=report_path.read_text(encoding='utf-8')
    old='| scienceqa | NOT ESTABLISHED | accuracy: one_sided_95_lower=-0.0200, pass=False |'
    new='| scienceqa | BOUNDARY PASS | accuracy: exact paired-count one_sided_95_lower=-0.0200 (=−2/100), pass=True |'
    assert text.count(old)==1
    text=text.replace(old,new)
    anchor='QA只评价最终答案，不自动验证全部长解释；'
    note=('ScienceQA 的自动非劣效界恰好等于预设 −2pp：原始FP64代码得到 −0.020000000000000018，'
        '直接比较误判为False。使用相同5000组抽样、相同线性5%分位数和整数正确题数独立重算，精确界为 −2/100，'
        '因此修正为边界通过；没有改变容忍阈值、指标、标签或推理结果。原始判定与修正来源保存在'
        '`audit/NUMERICAL_BOUNDARY_REVIEW.json`和统计清单。此边界通过不支持优越性；COCO及TextVQA质量门仍未建立。\n\n'
        '最终交付哈希以`audit/FINAL_DELIVERY.json`为准；`audit/REPORT_DELIVERY.json`保留数值边界修正前的生成器封印。\n\n')
    assert text.count(anchor)==1
    text=text.replace(anchor,note+anchor)
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    report_path.write_text(text,encoding='utf-8')
    value=dict(status='exact_boundary_correction_only',implementation_sha256=sha(__file__),
        original_FP64_diagnostic=original,reviewed_diagnostic=entry,
        exact_count_quantile=str(lower_count),exact_accuracy_bound=str(lower),
        exact_predeclared_limit=str(limit),bootstrap_draw_sha256=draw_sha,
        original_report_delivery_sha256=sha(ROOT/'audit/REPORT_DELIVERY.json'),
        pre_correction_sha256=before,post_correction_sha256={str(p):sha(p) for p in (manifest_path,report_path)},
        inference_scorer_analyzer_and_all_metric_values_unchanged=True)
    out.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(exact_lower=str(lower),exact_limit=str(limit),reviewed_pass=reviewed_pass)))


if __name__=='__main__':main()
