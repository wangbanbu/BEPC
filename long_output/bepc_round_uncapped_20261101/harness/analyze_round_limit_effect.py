"""Posthoc descriptive mechanism check; no metric/label/rule changes.

Read every paired native-path log, distinguish a longer accepted prefix from
a different first-longest leaf at unchanged length, and retain all identities.
Append a transparent addendum only after the frozen report is generated.
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert not (ROOT/'audit/REPORT_ADDENDUM_PROVENANCE.json').exists()
    report=ROOT/'reports/FULL_EXPERIMENT_REPORT.md'
    generated=read(ROOT/'audit/REPORT_DELIVERY.json')
    for p,h in generated['artifacts_sha256'].items():assert sha(p)==h
    before=sha(report)
    cfg=read(ROOT/'configs/frozen_config.json')
    summaries,all_cases,inputs={},[],{}
    for task in cfg['tasks']:
        rows={}
        for shard in (0,1):
            p=ROOT/f'results/main/main_{task}_shard{shard}.jsonl'
            inputs[str(p)]=sha(p)
            assert read(p.with_suffix('.complete.json'))['result_sha256']==sha(p)
            for line in p.read_text(encoding='utf-8').splitlines():
                r=json.loads(line);key=str(r['question_id']),r['method']
                assert key not in rows;rows[key]=r
        samples=read(cfg['data'][task]['test_path'])
        counts=Counter()
        for s in samples:
            q=str(s['question_id'])
            a,b=rows[q,'bepc_round2'],rows[q,'bepc_no_mismatch_caps']
            assert a['input_tensor_sha256']==b['input_tensor_sha256']
            counts['total_uncapped_rounds']+=len(b['acceptance_paths'])
            over=sum(x['mismatches']>2 for x in b['acceptance_paths'])
            counts['uncapped_rounds_above_two']+=over
            counts['responses_with_round_above_two']+=over>0
            counts['outputs_changed']+=a['output_text']!=b['output_text']
            for old,new in zip(a['acceptance_paths'],b['acceptance_paths']):
                if old!=new:
                    gain=new['accepted_draft']-old['accepted_draft']
                    assert new['mismatches']>=3 and gain>=0
                    equal=gain==0
                    if equal:assert new['row']<old['row']
                    counts['first_path_divergences']+=1
                    counts['first_divergence_same_length_earlier_leaf' if equal
                           else 'first_divergence_longer_prefix']+=1
                    all_cases.append(dict(task=task,question_id=q,image_id=s['image_id'],
                        image_path=s['image_path'],first_divergent_round=new['round'],
                        accepted_length_gain=gain,
                        interpretation='same-length earlier leaf' if equal else 'longer accepted prefix',
                        old_selected_path=old,new_selected_path=new))
                    break
            else:
                assert a['acceptance_paths']==b['acceptance_paths']
                assert a['raw_token_ids']==b['raw_token_ids']
        summaries[task]=dict(counts)
    effect=ROOT/'audit/ROUND_LIMIT_EFFECT.json'
    assert not effect.exists()
    effect.write_text(json.dumps(dict(
        scope='posthoc descriptive native-path mechanism check; no labels or quality values used',
        implementation_sha256=sha(__file__),result_sha256=inputs,tasks=summaries,cases=all_cases,
        original_analysis_sha256=sha(ROOT/'audit/STATISTICAL_ANALYSIS.json'),
        first_divergence_comparison='same inputs and identical committed native paths before divergence',
        caveat='local first-round path effect does not isolate full-request timing or imply factual quality'),
        ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    text=report.read_text(encoding='utf-8')
    text+='\n## 事后描述性机制检查：多接受 token 与同长度换分支\n\n'
    text+=('本节读取全部600条主测试原生日志，不使用质量标签筛选案例，也不修改评分或接受规则。'
        '在两臂共同历史仍完全一致的首次路径分叉处，分别比较选中前缀长度与树行号。\n\n')
    text+='| 任务 | 实际>2 mismatch轮数/总轮数 | 影响题数/100 | 首次分叉长度增加 | 同长度换更早分支 |\n|---|---:|---:|---:|---:|\n'
    for task,s in summaries.items():
        n=s['uncapped_rounds_above_two'];total=s['total_uncapped_rounds']
        text+=(f'| {task} | {n}/{total} ({100*n/total:.2f}%) | '
            f"{s['responses_with_round_above_two']} | {s.get('first_divergence_longer_prefix',0)} | "
            f"{s.get('first_divergence_same_length_earlier_leaf',0)} |\n")
    text+=('\n同长度换分支的原因是保留的first-longest规则：取消次数门后，原来较早的某条分支能够达到已有最佳长度，'
        '于是取代原来较晚的分支；当前轮提交token数量相同，但文本和后续上下文发生变化。'
        '因此不能把所有输出变化都解释成多接受了token。只有少量轮次实际越过原次数门，'
        '这提供了本轮MAT和吞吐收益较小的结构性解释；整段时延还受后续轨迹、输出长度与特征开销影响，不能由该计数单独分离。\n\n'
        '全部首次分叉与图像/题号见`audit/ROUND_LIMIT_EFFECT.json`。冻结的官方评分、统计清单、CSV和曲线均未修改。'
        '本补充章节的前后哈希见`audit/REPORT_ADDENDUM_PROVENANCE.json`；最终交付以`audit/FINAL_DELIVERY.json`为准。'
        '`audit/REPORT_DELIVERY.json`保留添加本章节前的生成器封印。\n')
    report.write_text(text,encoding='utf-8')
    provenance=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        reason='append posthoc descriptive path-length/tie mechanism check only',
        original_report_sha256=before,final_report_sha256=sha(report),
        original_report_delivery_sha256=sha(ROOT/'audit/REPORT_DELIVERY.json'),
        supplemental_source_sha256=sha(__file__),supplemental_effect_sha256=sha(effect),
        unchanged_generated_artifacts={p:h for p,h in generated['artifacts_sha256'].items() if Path(p)!=report})
    for p,h in provenance['unchanged_generated_artifacts'].items():assert sha(p)==h
    (ROOT/'audit/REPORT_ADDENDUM_PROVENANCE.json').write_text(json.dumps(provenance,
        ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(summaries),flush=True)


if __name__=='__main__':main()
