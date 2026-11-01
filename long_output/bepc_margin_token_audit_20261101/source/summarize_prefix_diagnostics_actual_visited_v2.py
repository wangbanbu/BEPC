"""CPU-only attribution: never infer semantic truth from policy token edits."""
import argparse,csv,json,hashlib,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def lcp(a,b):
 n=0
 while n<min(len(a),len(b)) and a[n]==b[n]:n+=1
 return n

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--scope',choices=('pilot49','allchanged'),required=True);args=ap.parse_args()
 pairs=read(ROOT/'data/paired_records.json');mapping={(p['task'],str(p['question_id'])):p for p in pairs}
 if args.scope=='pilot49':keys=[(p['task'],str(p['question_id'])) for p in read(ROOT/'data/qa_answer_change_cases.json')]+[('coco',str(p['question_id'])) for p in read(ROOT/'audit/COCO_BLIND_KEY.json')]
 else:keys=[k for k,p in mapping.items() if p['bepc']['returned_token_ids']!=p['margin']['returned_token_ids']]
 cases=[];flips=[];missing=[]
 for task,qid in keys:
  path=ROOT/f'diagnostics/{task}_{qid}.json'
  if not path.exists():missing.append((task,qid));continue
  r=read(path);p=mapping[task,qid];bt=p['bepc']['returned_token_ids'];mt=p['margin']['returned_token_ids'];n=lcp(bt,mt)
  base=dict(task=task,question_id=qid,status=r['status'],first_visible_divergence_index=n,bepc_first_different_token=bt[n] if n<len(bt) else None,margin_first_different_token=mt[n] if n<len(mt) else None,first_path_divergence_round=r.get('first_path_divergence_round'),elapsed_seconds=r.get('elapsed_seconds'),diagnostic_file=str(path.relative_to(ROOT)))
  if r['status']!='valid':cases.append(dict(base,reason=r.get('reason')));continue
  all_dif=[d for d in r['candidate_diagnostics'] if d['decision_relation']!='same']
  dif=[d for d in all_dif if d['production_bepc_visited']]
  shadow_only=[d for d in all_dif if not d['production_bepc_visited']]
  relevant=[d for d in dif if (d['decision_relation']=='protection' and d['selected_margin_edge']) or(d['decision_relation']=='rescue' and d['selected_bepc_edge'])]
  direct=[]
  for d in dif:
   pos=d['candidate_generated_position']
   prefix_hash=hashlib.sha256(json.dumps(p['bepc']['prompt_token_ids']+bt[:pos],separators=(',',':')).encode()).hexdigest()
   shared_prefix=pos<=n and d['legal_prefix_sha256']==prefix_hash
   category=None
   if pos==n and shared_prefix:
    if d['decision_relation']=='protection' and d['selected_margin_edge'] and n<len(mt) and mt[n]==d['draft_token']:
     category='direct_protection_Target_retained' if n<len(bt) and bt[n]==d['target_token'] else 'direct_protection_alternative_token'
    if d['decision_relation']=='rescue' and d['selected_bepc_edge'] and n<len(bt) and bt[n]==d['draft_token']:
     category='direct_rescue_Draft_committed' if n<len(mt) and mt[n]==d['target_token'] else 'direct_rescue_alternative_comparison'
   if category:direct.append(category)
   flips.append(dict(task=task,question_id=qid,first_visible_divergence_index=n,position_offset_from_first_visible_difference=pos-n,position_relation='same_position' if pos==n else 'deeper_than_first_visible' if pos>n else 'earlier_than_first_visible',prefix_matches_observed_common_prefix=shared_prefix,direct_classification=category,**d))
  if direct:attribution='+'.join(sorted(set(direct)))
  elif any(d['candidate_generated_position']>n for d in relevant):attribution='descendant_gate_changes_earlier_ancestor_selection'
  elif all(d['candidate_generated_position']<n for d in relevant) and relevant:attribution='later_token_divergence_after_earlier_gate_or_chunk_change'
  else:attribution='tree_selection_change_not_directly_attributed'
  base.update(first_tree_flip_types='+'.join(sorted({d['decision_relation'] for d in dif})),winner_related_flip_types='+'.join(sorted({d['decision_relation'] for d in relevant})),all_first_tree_flip_count=len(dif),shadow_only_hypothetical_flip_count=len(shadow_only),shadow_only_hypothetical_types='+'.join(sorted({d['decision_relation'] for d in shadow_only})),winner_related_flip_count=len(relevant),first_visible_attribution=attribution,visible_divergence_offset_from_tree_start=n-r['committed_generated_tokens'],model_calls=r['model_calls'],supplemental_readout_extra_target_forwards=r['extra_diagnostic_target_forwards'],production_BEPC_prior_and_first_path_exact=True,shadow_Margin_historical_first_path_exact=True)
  cases.append(base)
 summary=dict(scope=args.scope,requested=len(keys),available=len(cases),missing=missing,status_counts=dict(collections.Counter(c['status'] for c in cases)),total_prefix_replay_seconds=sum(c.get('elapsed_seconds') or 0 for c in cases),model_forward_counts=dict(collections.Counter({k:sum(c.get('model_calls',{}).get(k,0) for c in cases) for k in ('target','draft','vision')})),note='Strict v2: only production_bepc_visited=true gate flips enter causal gate attribution; shadow-only full scores are hypothetical. model_forward_counts count repeated original prefill/tree for prefix audit; supplemental BEPC readouts add no Target forward. Pilot49 is enriched and never a prevalence estimator.',by_task={})
 for t in ('scienceqa','textvqa','coco'):
  cc=[c for c in cases if c['task']==t]
  summary['by_task'][t]=dict(n=len(cc),first_tree_flip_types=dict(collections.Counter(c.get('first_tree_flip_types','invalid') for c in cc)),winner_related_flip_types=dict(collections.Counter(c.get('winner_related_flip_types','invalid') for c in cc)),first_visible_attribution=dict(collections.Counter(c.get('first_visible_attribution','invalid') for c in cc)))
 out=ROOT/'diagnostics'/(args.scope+'_actual_visited_v2');out.mkdir(exist_ok=True)
 for name,obj in [('CASE_ATTRIBUTIONS.json',cases),('CANDIDATE_FLIPS.json',flips),('SUMMARY.json',summary)]:
  path=out/name
  if path.exists():raise RuntimeError(f'preserve {path}')
  path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
