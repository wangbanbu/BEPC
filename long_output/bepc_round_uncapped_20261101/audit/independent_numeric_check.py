"""Read-only independent arithmetic from sealed request/per-image records.

Writes only the audit result when explicitly called; imports no experiment
analyzer or official scorer and performs no inference/metric execution.
"""
from pathlib import Path
import csv,hashlib,json,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
METHODS=('bepc_round2','bepc_no_mismatch_caps')
TASKS=('scienceqa','coco','textvqa')
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def flatten(x,prefix=''):
 out={}
 for k,v in x.items():
  if isinstance(v,dict):out.update(flatten(v,prefix+k+'.'))
  elif isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v):out[prefix+k]=v
 return out

def main():
 cfg=read(ROOT/'configs/frozen_config.json')
 summary=list(csv.DictReader((ROOT/'reports/BENCHMARK_SUMMARY.csv').open()))
 comparisons=list(csv.DictReader((ROOT/'reports/PAIRED_COMPARISONS.csv').open()))
 costs=list(csv.DictReader((ROOT/'reports/FEATURE_COST_COUNTERS.csv').open()))
 assert len(summary)==6 and len(comparisons)==42
 sm={(x['task'],x['method']):x for x in summary}
 cp={(x['task'],x['metric']):x for x in comparisons}
 fc={(x['task'],x['method'],x['counter']):x for x in costs}
 checks=[]
 def equal(label,a,b):
  if a is None:
   assert b in ('',None), (label,a,b)
  else:
   b=float(b);assert math.isfinite(b) and math.isclose(float(a),b,rel_tol=2e-12,abs_tol=0 if a and abs(a)<1e-12 else 2e-12),(label,a,b)
  checks.append(label)
 def metric(n,d,draws):
  n=np.asarray(n,dtype=float);d=np.asarray(d,dtype=float)
  if not d.sum():return None,np.full(len(draws),np.nan)
  den=d[draws].sum(1);rep=np.divide(n[draws].sum(1),den,out=np.full(len(draws),np.nan),where=den>0)
  return n.sum()/d.sum(),rep
 evidence={};bootstrap={};ni={};transitions={};seal_count=0;reuse_count=0
 for phase in ('development','main'):
  for ti,task in enumerate(TASKS):
   samples=read(cfg['data'][task][('dev' if phase=='development' else 'test')+'_path'])
   assert len({str(s['group_id']) for s in samples})==len(samples)
   for m in METHODS:
    folder=ROOT/f'metrics/{phase}/{task}/{m}';seal=read(folder/'SCORING_SEAL.json')
    assert seal['config_sha256']==sha(ROOT/'configs/frozen_config.json')
    for field,name in [('aggregate_sha256','aggregate.json'),('per_question_sha256','per_question_scores.json'),('score_provenance_sha256','SCORE_PROVENANCE.json')]:assert seal[field]==sha(folder/name)
    prov=read(folder/'SCORE_PROVENANCE.json');seal_count+=1
    if prov['mode']=='exact_whole_corpus_official_score_reuse':
     source=Path(prov['source']);ss=read(source/'SCORING_SEAL.json')
     assert prov['source_seal_sha256']==sha(source/'SCORING_SEAL.json')
     for f,n in [('aggregate_sha256','aggregate.json'),('per_question_sha256','per_question_scores.json')]:assert ss[f]==sha(source/n)==sha(folder/n)
     reuse_count+=1
   if phase!='main':continue
   rows=[]
   for shard in (0,1):
    p=ROOT/f'results/main/main_{task}_shard{shard}.jsonl';assert sha(p)==read(p.with_suffix('.complete.json'))['result_sha256']
    rows.extend(json.loads(s) for s in p.read_text().splitlines())
   ids=[str(s['question_id']) for s in samples]
   runtime={(r['method'],str(r['question_id'])):r for r in rows}
   quality={m:{str(q['question_id']):q for q in read(ROOT/f'metrics/main/{task}/{m}/per_question_scores.json')} for m in METHODS}
   draws=np.random.default_rng(int(cfg['bootstrap_seed'])+ti*100003).integers(0,len(samples),size=(5000,len(samples)),dtype=np.int64)
   bootstrap[task]=dict(sha256=hashlib.sha256(draws.tobytes()).hexdigest(),shape=list(draws.shape))
   vectors={}
   for m in METHODS:
    rr=[runtime[m,q] for q in ids];qq=[quality[m][q] for q in ids];ones=[1]*len(ids)
    for r,q in zip(rr,qq):assert r['output_text']==q['raw_output_text']
    direct={
     'mean_seconds':([r['seconds'] for r in rr],ones),
     'mean_decode_seconds':([r['decode_seconds'] for r in rr],ones),
     'throughput_tokens_per_second':([len(r['returned_token_ids']) for r in rr],[r['seconds'] for r in rr]),
     'decode_throughput_tokens_per_second':([len(r['returned_token_ids']) for r in rr],[r['decode_seconds'] for r in rr]),
     'mean_peak_allocated_bytes':([r['peak_allocated_bytes'] for r in rr],ones),
     'mean_output_words':([len(r['output_text'].split()) for r in rr],ones),
     'truncation_fraction':([r['cap_reached'] for r in rr],ones),
     'parse_uncertain_fraction':([int(q.get('uncertain',False)) for q in qq],ones),
     'answer_extraction_failure_fraction':([int(q.get('answer_extraction_status') in ('invalid','uncertain')) for q in qq],ones),
     'native_mean_accepted_draft_length':([r['raw_accepted_draft_tokens'] for r in rr],[r['raw_verification_rounds'] for r in rr]),
     'effective_mean_accepted_draft_length':([r['effective_accepted_draft_tokens'] for r in rr],[r['effective_verification_rounds'] for r in rr]),
     'effective_mean_committed_tokens_per_round':([r['completion_tokens'] for r in rr],[r['effective_verification_rounds'] for r in rr]),
    }
    if task=='coco':
     for key in ('CIDEr','SPICE','CHAIRs'):direct[key]=([q[key] for q in qq],ones)
     direct['CHAIRi']=([q['hallucinated_mentions'] for q in qq],[q['object_mentions'] for q in qq])
    else:direct['accuracy']=([q['score'] for q in qq],ones)
    vectors[m]={k:metric(n,d,draws) for k,(n,d) in direct.items()}
    if task=='coco':vectors[m]['answer_extraction_failure_fraction']=(None,np.full(len(draws),np.nan))
    for key,(point,reps) in vectors[m].items():
     equal(f'{task}/{m}/{key}',point,sm[task,m][key])
     lohi=np.quantile(reps,[.025,.975]) if point is not None and np.isfinite(reps).all() else (None,None)
     for side,value in zip(('low','high'),lohi):equal(f'{task}/{m}/{key}/CI{side}',value,sm[task,m][key+'_ci95_'+side])
     equal(f'{task}/{m}/{key}/undefined',int(np.sum(~np.isfinite(reps))),sm[task,m][key+'_undefined_bootstrap'])
    stats=[r['signal_stats'] for r in rr];cnt=[int(s['committed_mismatches']) for s in stats];paths=[p['mismatches'] for r in rr for p in r['acceptance_paths']]
    extras=dict(mean_visible_tokens=np.mean([len(r['returned_token_ids']) for r in rr]),
     mean_raw_tokens=np.mean([len(r['raw_token_ids']) for r in rr]),mean_completion_tokens_including_EOS=np.mean([r['completion_tokens'] for r in rr]),
     mean_committed_mismatches=np.mean(cnt),committed_mismatch_p50=np.quantile(cnt,.5),committed_mismatch_p95=np.quantile(cnt,.95),committed_mismatch_max=max(cnt),
     response_fraction_with_committed_mismatch=np.mean(np.array(cnt)>0),response_fraction_above_four_mismatches=np.mean(np.array(cnt)>4),actual_per_round_mismatch_max=max(paths),selected_rounds=len(paths),
     selected_rounds_above_two_mismatches=sum(n>2 for n in paths),selected_round_fraction_above_two_mismatches=np.mean(np.array(paths)>2),
     responses_with_round_above_two_mismatches=sum(any(p['mismatches']>2 for p in r['acceptance_paths']) for r in rr),
     response_fraction_with_round_above_two_mismatches=np.mean([any(p['mismatches']>2 for p in r['acceptance_paths']) for r in rr]),
     round_budget_binding_candidate_visits=sum(s['row_budget_binding'] for s in stats),responses_with_round_budget_binding=sum(s['row_budget_binding']>0 for s in stats),
     max_peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rr),max_peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in rr),
     max_peak_increment_bytes=max(r['peak_increment_bytes'] for r in rr),max_peak_reserved_increment_bytes=max(r['peak_reserved_increment_bytes'] for r in rr),
     completion_tokens_including_EOS_throughput_per_second=sum(r['completion_tokens'] for r in rr)/sum(r['seconds'] for r in rr),
     completion_tokens_including_EOS_decode_throughput_per_second=sum(r['completion_tokens'] for r in rr)/sum(r['decode_seconds'] for r in rr),
     mean_recorded_preprocessing_seconds=np.mean([r['input_preprocessing_seconds'] for r in rr]),
     parse_invalid_count=sum(q['parse_invalid'] for q in qq),parse_uncertain_count=sum(q['uncertain'] for q in qq),
     answer_extraction_failures=sum(q.get('answer_extraction_status') in ('invalid','uncertain') for q in qq))
    for key,value in extras.items():equal(f'{task}/{m}/{key}',value,sm[task,m][key])
    keys=set().union(*(flatten(s).keys() for s in stats))
    for key in keys:
     a=[flatten(s)[key] for s in stats if key in flatten(s)];row=fc[task,m,key]
     for name,value in dict(present_questions=len(a),total_questions=len(rr),sum=sum(a),mean=np.mean(a),maximum=max(a)).items():equal(f'counter/{task}/{m}/{key}/{name}',value,row[name])
   for k in vectors[METHODS[0]]:
    b,br=vectors[METHODS[0]][k];a,ar=vectors[METHODS[1]][k];row=cp[task,k]
    equal(task+'/'+k+'/new',a,row['method_point']);equal(task+'/'+k+'/control',b,row['baseline_point'])
    if a is None or b is None:assert row['status']=='unavailable';continue
    assert row['status']=='ok';delta=a-b;rep=ar-br;lo,hi=np.quantile(rep,[.025,.975])
    centered=np.abs(rep-delta);threshold=max(0.,abs(delta)-8*np.finfo(float).eps*max(abs(delta),float(centered.max())))
    p=(1+np.sum(centered>=threshold))/(len(draws)+1)
    for name,value in dict(difference=delta,ci95_low=lo,ci95_high=hi,p=p).items():equal(task+'/'+k+'/'+name,value,row[name])
    if b:
     equal(task+'/'+k+'/relative',100*delta/b,row['relative_change_percent'])
     if np.all(br!=0):
      low,high=np.quantile(100*(ar/br-1),[.025,.975])
      for side,value in [('low',low),('high',high)]:equal(task+'/'+k+'/relativeCI'+side,value,row['relative_change_percent_ci95_'+side])
     else:
      assert row['relative_change_percent_ci95_low']==row['relative_change_percent_ci95_high']==''
    if k=='mean_seconds':
     for name,value in dict(latency_speedup=b/a,latency_speedup_ci95_low=np.quantile(br/ar,.025),latency_speedup_ci95_high=np.quantile(br/ar,.975)).items():equal(task+'/'+k+'/'+name,value,row[name])
   if task=='coco':
    lower=float(np.quantile(vectors[METHODS[1]]['SPICE'][1]-.95*vectors[METHODS[0]]['SPICE'][1],.05));upper=float(np.quantile(vectors[METHODS[1]]['CHAIRs'][1]-vectors[METHODS[0]]['CHAIRs'][1],.95));ni[task]=dict(SPICE_lower=lower,CHAIRs_upper=upper,passed=lower>=-1e-12 and upper<=.02+1e-12)
   else:
    lower=float(np.quantile(vectors[METHODS[1]]['accuracy'][1]-vectors[METHODS[0]]['accuracy'][1],.05));ni[task]=dict(accuracy_lower=lower,passed=lower>=-.02-1e-12)
   evidence[task]={m:{k:float(sm[task,m][k]) for k in ('throughput_tokens_per_second','effective_mean_accepted_draft_length','mean_seconds','mean_visible_tokens')} for m in METHODS}
   c={}
   for q in ids:
    a=quality[METHODS[0]][q];b=quality[METHODS[1]][q]
    if task!='coco':
     key='improved' if b['score']>a['score'] else 'degraded' if b['score']<a['score'] else 'tie';c[key]=c.get(key,0)+1
     if task=='scienceqa':
      key=f'{int(a["score"])}->{int(b["score"])}';c[key]=c.get(key,0)+1
    else:
     for k in ('SPICE','CHAIRs','CHAIRi','CIDEr'):
      key=k+('_improved' if (b[k]<a[k] if k.startswith('CHAIR') else b[k]>a[k]) else '_degraded' if b[k]!=a[k] else '_tie');c[key]=c.get(key,0)+1
   transitions[task]=c
 assert seal_count==12
 manifest=read(ROOT/'audit/STATISTICAL_ANALYSIS.json')
 for t in TASKS:assert manifest['draws'][t]['sha256']==bootstrap[t]['sha256']
 for diag in manifest['quality_diagnostics']:
  t=diag['task'];assert diag['automatic_proxy_pass']==ni[t]['passed']
  for cond in diag['conditions']:
   key=('accuracy_lower' if t!='coco' else 'SPICE_lower' if cond['metric']=='SPICE' else 'CHAIRs_upper');equal(t+'/NI/'+key,ni[t][key],cond['bound'])
 result=dict(status='pass',independent_numeric_comparisons=len(checks),summary_rows=len(summary),paired_comparison_rows=len(comparisons),feature_counter_rows=len(costs),score_seals=seal_count,exact_corpus_reused_seals=reuse_count,evidence=evidence,transitions=transitions,noninferiority=ni,bootstrap=bootstrap,inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'reports/BENCHMARK_SUMMARY.csv',ROOT/'reports/PAIRED_COMPARISONS.csv',ROOT/'reports/FEATURE_COST_COUNTERS.csv',ROOT/'audit/STATISTICAL_ANALYSIS.json']})
 (ROOT/'audit/INDEPENDENT_NUMERIC_CHECK.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
