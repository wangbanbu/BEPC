"""Unblind the completed, immutable single-model visual review for reporting."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
keys={x['case_id']:x for x in json.loads((ROOT/'audit/COCO_BLIND_KEY.json').read_text())}
pairs={(x['task'],str(x['question_id'])):x for x in json.loads((ROOT/'data/paired_records.json').read_text())}
path=ROOT/'blind_review/COCO_MODEL_REVIEW.jsonl'
reviews=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
assert len(reviews)==26==len(keys)
assert len(set(x['case_id'] for x in reviews))==26
rows=[]
for x in reviews:
    k=keys[x['case_id']]
    p=pairs['coco',k['question_id']]
    assert x['image_id']==k['image_id']
    assert x['image_checked'] and not x['method_identity_seen'] and not x['automatic_metric_values_seen']
    assert hashlib.sha256(Path(x['image_path']).read_bytes()).hexdigest()==x['image_sha256']
    rel=x['overall_relation']
    winner=k[rel[0]] if rel.startswith(('A_','B_')) else None
    row=dict(x,question_id=k['question_id'],A_method=k['A'],B_method=k['B'],
        bepc_relation='better' if winner=='bepc' else 'worse' if winner=='margin' else rel,
        bepc_CHAIRi=p['bepc_score']['CHAIRi'],margin_CHAIRi=p['margin_score']['CHAIRi'],
        bepc_CHAIRs=p['bepc_score']['CHAIRs'],margin_CHAIRs=p['margin_score']['CHAIRs'])
    row['CHAIRi_direction']='improved' if row['bepc_CHAIRi']<row['margin_CHAIRi'] else 'degraded'
    rows.append(row)
(ROOT/'audit/COCO_UNBLINDED_REVIEW.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
summary=dict(n=len(rows),bepc_relations=dict(Counter(r['bepc_relation'] for r in rows)),
    CHAIRi_by_relation=dict(Counter(r['CHAIRi_direction']+'/'+r['bepc_relation'] for r in rows)),
    review_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    scope='All26 unequal CHAIRi cases; metric-enriched selection, NOT prevalence estimate.',
    reviewer='one blinded image-conditioned model-assisted review; NOT human Ground Truth; no formal agreement estimate')
(ROOT/'audit/COCO_UNBLINDING_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
pd.DataFrame([{k:r[k] for k in ['case_id','question_id','image_id','A_method','B_method','bepc_relation','CHAIRi_direction','bepc_CHAIRi','margin_CHAIRi','rationale']} for r in rows]).to_csv(ROOT/'data/coco_unblinded_review.csv',index=False)
print(json.dumps(summary,ensure_ascii=False,indent=2))
for r in rows:
    print(r['question_id'],r['A_method'],r['bepc_relation'],r['CHAIRi_direction'],r['rationale'])
