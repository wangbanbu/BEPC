"""Exact production-prefix replay; stop before committing first divergent tree.
Read-only imports from sealed baseline; all writes stay under this new run.
No caption regeneration and no label computation. Extra scores are diagnostic.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, os, sys, time, traceback
from pathlib import Path
sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parents[1]
PARENT = OUT.parent/'bepc_vs_margin_20261101'
sys.path.insert(0, str(PARENT/'harness'))
spec=importlib.util.spec_from_file_location('_read_only_original_core',PARENT/'harness/run_benchmark.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
import torch
from uncapped_controller import NoResponseCapBEPC
from refined_controller import budgeted_scan
from compact_logits import tree_decoding
from terminal_update import TerminalUpdate


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():raise RuntimeError(f'preserve existing {p}')
 p.write_text(json.dumps(core.clean(o),ensure_ascii=False,indent=2,allow_nan=False)+'\n')

class PrefixStopped(Exception): pass

class DiagnosticRuntime:
 def __init__(self):
  self.cfg=read(PARENT/'configs/frozen_config.json')
  for name,digest in self.cfg['harness_sha256'].items():
   assert sha(PARENT/'harness'/name)==digest,name
  for task,entry in self.cfg['data'].items():
   for split in ('dev','test'):assert sha(entry[split+'_path'])==entry[split+'_sha256']
  sys.path.insert(0,self.cfg['upstream']);os.chdir(self.cfg['upstream'])
  from vispec.model import spec_model_ours
  from transformers import AutoProcessor
  self.module=spec_model_ours
  self.model=spec_model_ours.SpecModel.from_pretrained(base_model_path=self.cfg['target'],spec_model_path=self.cfg['draft'],torch_dtype='auto',low_cpu_mem_usage=True,device_map='auto',**self.cfg['tree']).eval()
  self.processor=AutoProcessor.from_pretrained(self.cfg['target'],**self.cfg['processor'])
  self.tokenizer=self.model.tokenizer
  self.original={k:getattr(self.module,k) for k in NoResponseCapBEPC.FUNCTIONS}
  self.samples={(t,str(s['question_id'])):s for t in self.cfg['tasks'] for s in read(self.cfg['data'][t]['test_path'])}
  self.calls={};self.handles=[]
  for key,module in [('target',self.model.base_model),('draft',self.model.spec_layer),('vision',self.model.base_model.visual)]:
   def hook(mod,args,out,key=key):self.calls[key]=self.calls.get(key,0)+1
   self.handles.append(module.register_forward_hook(hook))
  write(OUT/'diagnostics'/f'loaded_pid{os.getpid()}.json',dict(parent=str(PARENT),config_sha256=sha(PARENT/'configs/frozen_config.json'),script_sha256=sha(__file__),torch=torch.__version__,target_dtype=str(self.model.base_model.dtype),physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],attention=type(self.model.base_model.model.layers[-1].self_attn).__name__,all_writes_new_root=True))
 inputs=core.Runtime.inputs
 def restore(self):
  for k,f in self.original.items():setattr(self.module,k,f)
 def decode(self,x):return self.tokenizer.decode(x,skip_special_tokens=False)
 def diagnose(self,pair):
  task,qid=pair['task'],str(pair['question_id']);b,m=pair['bepc'],pair['margin']
  started=time.perf_counter();result=dict(task=task,question_id=qid,image_id=pair['image_id'],status='invalid',new_full_continuation_generated=False,diagnostic_only=True)
  first=next((i for i,(x,y) in enumerate(zip(b['acceptance_paths'],m['acceptance_paths'])) if x!=y),None)
  if first is None:
   return dict(result,reason='no shared round with differing historical path')
  result['first_path_divergence_round']=first
  sample=self.samples[task,qid]
  self.restore();self.module.tree_decoding=tree_decoding;self.calls={}
  inputs=self.inputs(sample)
  assert inputs.input_ids[0].tolist()==b['prompt_token_ids']==m['prompt_token_ids'],'prompt tokens'
  assert inputs.image_grid_thw.tolist()==b['image_grid_thw']==m['image_grid_thw'],'image grid'
  digest=core.input_digest(inputs)
  assert digest==b['input_tensor_sha256']==m['input_tensor_sha256'],'input hash'
  result.update(input_tensor_sha256=digest,image_sha256=sha(sample['image_path']),prompt_text=sample['prompt_text'])
  terminal=TerminalUpdate(self.original['update_inference_inputs'],prompt_length=inputs.input_ids.shape[1],max_new_tokens=self.cfg['max_new_tokens'],eos_token_id=self.tokenizer.eos_token_id,audit=False)
  self.module.update_inference_inputs=terminal
  variant=self.cfg['methods_by_task'][task]['bepc_no_response_cap']
  controller=NoResponseCapBEPC(self.module,self.model,variant,inputs.input_ids,inputs.image_grid_thw,audit=False,max_new_tokens=self.cfg['max_new_tokens'],image_id=sample['image_id'],prompt_kind=task)
  production_evaluate=self.module.evaluate_posterior
  path_checks=[]
  def evaluate(logits,candidates,processor):
   production_result=production_evaluate(logits,candidates,processor)
   current=controller.acceptance_paths[-1];round_=int(current['round'])
   assert current==b['acceptance_paths'][round_],('historical BEPC path mismatch',round_,current,b['acceptance_paths'][round_])
   if round_<first:
    assert current==m['acceptance_paths'][round_],('prior common path mismatch',round_)
    path_checks.append(round_);return production_result
   assert round_==first
   # Freeze production observations before any additional lazy feature work.
   actual_cache=dict(controller.decision_cache);actual_reasons=dict(controller.decision_reasons)
   prior_tokens=[t for p in b['acceptance_paths'][:first] for t in p['tokens']]
   assert controller.ids[0].tolist()==b['prompt_token_ids']+prior_tokens,'committed prefix parity'
   margin_visited=set();margin_rows=[]
   def margin_allowed(row,col):
    parent=controller.retrieve_rows[row][col-1];a=controller.targets[row][col-1];d=controller.candidate_rows[row][col]
    margin_visited.add((parent,a,d));g=controller.margin_rows[row][col-1]
    return math.isfinite(g) and g<=1.
   best,bestlen,bestmm=0,-1,0
   for row,seq in enumerate(controller.candidate_rows):
    n,mm=budgeted_scan(seq,controller.targets[row],eos=controller.eos,committed_length=controller.ids.shape[1],visible_boundary=controller.visible_boundary,response_used=controller.committed_mismatches,response_budget=controller.response_budget,round_budget=controller.round_budget,allowed=lambda col,row=row:margin_allowed(row,col))
    margin_rows.append(dict(row=row,accepted_draft=n,mismatches=mm))
    if n>bestlen:best,bestlen,bestmm=row,n,mm
   shadow=dict(round=round_,row=best,accepted_draft=bestlen,mismatches=bestmm,response_mismatches_before=controller.committed_mismatches,nodes=controller.retrieve_rows[best][:bestlen+1],tokens=controller.candidate_rows[best][:bestlen+1])
   assert shadow==m['acceptance_paths'][first],('historical Margin shadow path mismatch',shadow,m['acceptance_paths'][first])
   loc={}
   for row,seq in enumerate(controller.candidate_rows):
    for col in range(1,len(seq)):
     if seq[col]<0:continue
     key=(controller.retrieve_rows[row][col-1],controller.targets[row][col-1],seq[col])
     loc.setdefault(key,[]).append((row,col))
   keyset=set(actual_cache)|margin_visited
   diagnostics=[]
   for key in sorted(keyset):
    parent,a,d=key;row,col=loc[key][0];g=controller.margin_rows[row][col-1]
    margin=controller.margin_tensor[row,col-1]
    full=controller._bepc_score(parent,a,d,margin)
    full_clean=core.clean(full);be_decision=bool(full['score']<=1.);ma_decision=math.isfinite(g) and g<=1.
    if key in actual_cache:assert be_decision==actual_cache[key],('full diagnostic changed production gate',key)
    ctx=controller.epcc.context(parent)
    prefix_tokens=controller.epcc.committed_ids.tolist()+[controller.epcc.token_rows[node] for node in ctx['path']]
    be_path=current['nodes'];ma_path=shadow['nodes']
    be_selected=any(be_path[i]==parent and controller.candidate_rows[current['row']][i+1]==d for i in range(len(be_path)-1))
    ma_selected=any(ma_path[i]==parent and controller.candidate_rows[shadow['row']][i+1]==d for i in range(len(ma_path)-1))
    diagnostics.append(dict(parent=parent,target_token=a,draft_token=d,target_text=self.decode([a]),draft_text=self.decode([d]),g=g,m_U=full_clean.get('m_U'),m_P=full_clean.get('m_P'),m_R=full_clean.get('m_R'),delta=full_clean.get('delta'),score=full_clean.get('score'),gate=full_clean.get('gate'),available=full_clean.get('available'),protection_component=full_clean.get('protection'),rescue_component=full_clean.get('rescue'),margin_accept=ma_decision,bepc_accept=be_decision,decision_relation='protection' if ma_decision and not be_decision else 'rescue' if be_decision and not ma_decision else 'same',production_bepc_visited=key in actual_cache,production_bepc_decision=actual_cache.get(key),production_reason=actual_reasons.get(key),shadow_margin_visited=key in margin_visited,selected_bepc_edge=be_selected,selected_margin_edge=ma_selected,positions=loc[key],parent_absolute_position=ctx['parent_absolute_position'],candidate_absolute_position=ctx['parent_absolute_position']+1,candidate_generated_position=ctx['parent_absolute_position']+1-len(b['prompt_token_ids']),parent_path=list(ctx['path']),legal_prefix_sha256=hashlib.sha256(json.dumps(prefix_tokens,separators=(',',':')).encode()).hexdigest(),legal_prefix_last_64_tokens=prefix_tokens[-64:],legal_prefix_tail=self.decode(prefix_tokens[-64:])))
   candidates_cpu=controller.candidate_rows
   result.update(status='valid',historical_BEPC_path_parity=True,prior_common_path_rounds_verified=len(path_checks),historical_Margin_same_tree_shadow_parity=True,committed_generated_tokens=len(prior_tokens),committed_generated_token_ids=prior_tokens,committed_prefix_text=self.decode(prior_tokens),historical_bepc_path=b['acceptance_paths'][first],historical_margin_path=m['acceptance_paths'][first],replayed_bepc_path=current,shadow_margin_path=shadow,candidate_rows=candidates_cpu,retrieve_rows=controller.retrieve_rows,target_top1_rows=controller.targets,native_margin_rows=controller.margin_rows,shadow_margin_scan_rows=margin_rows,candidate_diagnostics=diagnostics,production_bepc_visit_count=len(actual_cache),shadow_margin_visit_count=len(margin_visited),supplemental_full_scores=len(diagnostics),stopped_before_divergent_tree_commit=True,used_future_continuation=False,new_label_generated=False)
   raise PrefixStopped()
  self.module.evaluate_posterior=evaluate
  try:
   torch.manual_seed(self.cfg['seed'])
   with torch.inference_mode():
    self.model.specgenerate(**inputs,temperature=0.,max_new_tokens=self.cfg['max_new_tokens'],max_length=self.cfg['max_length'],log=True,return_acceptance_len=True,return_decode_time=True)
   result.update(status='invalid',reason='native generation ended before expected divergence')
  except PrefixStopped:pass
  except Exception as e:result.update(status='invalid',reason=repr(e),traceback=traceback.format_exc())
  finally:
   controller.restore();self.restore()
  torch.cuda.synchronize()
  result.update(elapsed_seconds=time.perf_counter()-started,model_calls=dict(self.calls),extra_diagnostic_target_forwards=0,loaded_original_sources_unchanged=True)
  return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('six','all'),required=True);args=parser.parse_args()
 pairs=read(OUT/'data/paired_records.json');mapping={(p['task'],str(p['question_id'])):p for p in pairs}
 first=[('scienceqa','13461'),('scienceqa','17228'),('textvqa','38417'),('textvqa','35085'),('textvqa','35696'),('textvqa','39338')]
 if args.mode=='all':
  keys=[(p['task'],str(p['question_id'])) for p in read(OUT/'data/qa_answer_change_cases.json')]+[('coco',str(p['question_id'])) for p in read(OUT/'audit/COCO_BLIND_KEY.json')]
  keys=first+[k for k in keys if k not in first]
 else:keys=first
 todo=[k for k in keys if not (OUT/f'diagnostics/{k[0]}_{k[1]}.json').exists()]
 if not todo:print('all requested prefix diagnostics already present',flush=True);return
 runtime=DiagnosticRuntime()
 for key in todo:
  print(json.dumps(dict(event='begin',task=key[0],qid=key[1])),flush=True)
  try:result=runtime.diagnose(mapping[key])
  except Exception as e:result=dict(task=key[0],question_id=key[1],status='invalid',reason=repr(e),traceback=traceback.format_exc())
  write(OUT/f'diagnostics/{key[0]}_{key[1]}.json',result)
  print(json.dumps(dict(event='done',task=key[0],qid=key[1],status=result['status'],elapsed=result.get('elapsed_seconds'),reason=result.get('reason'))),flush=True)
if __name__=='__main__':main()
