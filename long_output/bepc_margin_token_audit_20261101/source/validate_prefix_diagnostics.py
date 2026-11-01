"""Independent CPU numerical and provenance checks of native prefix diagnostics."""
from pathlib import Path
import hashlib,json,math,collections
ROOT=Path(__file__).resolve().parents[1];PARENT=ROOT.parent/'bepc_vs_margin_20261101'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
pairs={(p['task'],str(p['question_id'])):p for p in read(ROOT/'data/paired_records.json')}
expected=[k for k,p in pairs.items() if p['bepc']['returned_token_ids']!=p['margin']['returned_token_ids']]
checks=collections.Counter();fails=[];max_score_error=max_delta_error=0.
def check(name,value,case=None):
 checks[name]+=1
 if not value:fails.append(dict(check=name,case=case))
for key in expected:
 r=read(ROOT/f'diagnostics/{key[0]}_{key[1]}.json');p=pairs[key];b=p['bepc'];m=p['margin'];f=r['first_path_divergence_round'];case='/'.join(key)
 check('status_valid',r['status']=='valid',case)
 check('same_processor_input_hash',r['input_tensor_sha256']==b['input_tensor_sha256']==m['input_tensor_sha256'],case)
 check('historical_BEPC_first_path_exact',r['replayed_bepc_path']==b['acceptance_paths'][f],case)
 check('historical_Margin_shadow_first_path_exact',r['shadow_margin_path']==m['acceptance_paths'][f],case)
 check('prior_path_count_exact',r['prior_common_path_rounds_verified']==f,case)
 check('prior_both_policies_common',b['acceptance_paths'][:f]==m['acceptance_paths'][:f],case)
 check('stopped_before_commit',r['stopped_before_divergent_tree_commit'] and not r['new_full_continuation_generated'] and not r['used_future_continuation'],case)
 check('actual_forward_count',r['model_calls']==dict(vision=1,target=f+2,draft=4*(f+1)),case)
 check('supplemental_readout_no_forward',r['extra_diagnostic_target_forwards']==0,case)
 for d in r['candidate_diagnostics']:
  ctx=case+f"/parent{d['parent']}/d{d['draft_token']}"
  for row,col in d['positions']:
   check('candidate_token_alignment',r['retrieve_rows'][row][col-1]==d['parent'] and r['candidate_rows'][row][col]==d['draft_token'] and r['target_top1_rows'][row][col-1]==d['target_token'] and r['native_margin_rows'][row][col-1]==d['g'],ctx)
  check('candidate_position_alignment',d['candidate_generated_position']==d['candidate_absolute_position']-len(b['prompt_token_ids']) and d['candidate_absolute_position']==d['parent_absolute_position']+1,ctx)
  check('score_finite',all(d[k] is not None and math.isfinite(d[k]) for k in ('g','m_U','m_P','m_R','delta','score')),ctx)
  if all(d[k] is not None for k in ('g','m_U','m_P','delta','score')):
   err=abs(d['score']-(d['g']-d['delta']));max_score_error=max(max_score_error,err)
   check('score_equals_g_minus_delta',err<=1e-6,ctx)
   expected_delta=(.5*math.copysign(1,d['m_P'])*min(abs(d['m_U']),abs(d['m_P']),1.)) if d['gate'] else 0.
   err=abs(d['delta']-expected_delta);max_delta_error=max(max_delta_error,err)
   check('frozen_symmetric_correction_formula',err<=1e-6,ctx)
  check('Margin_decision_formula',d['margin_accept']==(d['g']<=1.),ctx)
  check('BEPC_full_decision_formula',d['bepc_accept']==(d['score']<=1.),ctx)
  if d['production_bepc_visited']:check('production_gate_full_readout_parity',d['production_bepc_decision']==d['bepc_accept'],ctx)
for rel,h in read(ROOT/'audit/ATTRIBUTION_EVIDENCE_AUDIT.json')['source_result_sha256'].items():check('sealed_parent_results_unchanged',sha(PARENT/rel)==h,rel)
cfg=read(PARENT/'configs/frozen_config.json')
for name,h in cfg['harness_sha256'].items():check('sealed_parent_harness_unchanged',sha(PARENT/'harness'/name)==h,name)
v2=read(ROOT/'diagnostics/allchanged_actual_visited_v2/CANDIDATE_FLIPS.json')
check('strict_summary_all_flips_production_visited',all(d['production_bepc_visited'] for d in v2))
report=dict(status='pass' if not fails else 'fail',cases=len(expected),checks=dict(checks),total_checks=sum(checks.values()),failures=fails,max_score_rounding_error=max_score_error,max_delta_formula_error=max_delta_error,score_tolerance=1e-6,scope='CPU validation of source/provenance/path/alignment/frozen-score diagnostics; no label truth assertion')
out=ROOT/'diagnostics/DIAGNOSTIC_VALIDATION.json';assert not out.exists();out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
