"""Freeze long-prompt inputs/configuration without changing the short benchmark."""
from pathlib import Path
import argparse,copy,hashlib,json,shutil
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT.parent
TASKS=['mme','gqa','textvqa','scienceqa','coco']
EXPLANATION=('Provide a detailed explanation of about 100 to 160 words, using the image and the question to support your answer. '
 'Do not invent details that are not visible or supported by the question. ')
FINAL={
 'mme':'End with exactly one final line in the format "Final answer: yes" or "Final answer: no". Put that final answer on a separate line, with no text after it.',
 'gqa':'End with exactly one final line in the format "Final answer: <answer>", replacing <answer> with a single word or short phrase. Put that final answer on a separate line, with no text after it.',
 'textvqa':'End with exactly one final line in the format "Final answer: <answer>", replacing <answer> with a single word or short phrase. Put that final answer on a separate line, with no text after it.',
 'scienceqa':'End with exactly one final line in the format "Final answer: X", replacing X with the selected option letter. Put that final answer on a separate line, with no text after it.'}
SHORT_SUFFIX={'mme':'\nPlease answer yes or no.','gqa':'\nAnswer the question using a single word or phrase.',
 'textvqa':'\nAnswer the question using a single word or phrase.','scienceqa':'\nAnswer with the option letter only.'}
COCO=('Please provide a detailed description of the given image in about 100 to 160 words. '
 'Describe visible objects, attributes, actions, quantities, and spatial relations, and distinguish uncertainty from visible facts.')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),p
 p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def long_prompt(task,text):
 if task=='coco':
  assert text=='Describe this image in one concise sentence.'
  return COCO
 suffix=SHORT_SUFFIX[task];assert text.endswith(suffix),(task,text)
 return text[:-len(suffix)]+'\n\n'+EXPLANATION+FINAL[task]

def data():
 original=read(PARENT/'configs/frozen_config.json');evidence={}
 for task in TASKS:
  evidence[task]={}
  for split,count in [('dev',20),('test',100)]:
   source=Path(original['data'][task][split+'_path']);assert sha(source)==original['data'][task][split+'_sha256']
   rows=read(source);changed=[]
   assert len({r['group_id'] for r in rows})==count
   for old in rows:
    new=copy.deepcopy(old);new['prompt_text']=long_prompt(task,old['prompt_text']);new['prompt_kind']='task_long'
    assert not {'answer','answers','label','references'}&set(new)
    changed.append(new)
   output=ROOT/f'data/{task}/{split}.json';write(output,changed)
   evidence[task][split]=dict(parent_path=str(source),parent_sha256=sha(source),path=str(output),sha256=sha(output),
     questions=len(rows),groups=count,identity_order_images_gold_unchanged=True,only_changed_fields=['prompt_text','prompt_kind'])
 write(ROOT/'audit/LONG_INPUT_PROVENANCE.json',dict(parent_config_sha256=sha(PARENT/'configs/frozen_config.json'),
  resampled=False,generated_new_gold=False,all_image_files_reused_readonly=True,inputs=evidence))
 print('Long prompts prepared on unchanged20/100 groups per task.')

def design():
 cfg=read(PARENT/'configs/design_config.json')
 cfg.update(name='Clean GitHub ViSpec five-task long-output decoding comparison',created_utc=datetime.now(timezone.utc).isoformat(),
  experiment_kind='long_prompt_512_budget',prompt_kind='task_long',max_new_tokens=512,max_length=2048,
  parent_short_design_path=str(PARENT/'configs/design_config.json'),parent_short_design_sha256=sha(PARENT/'configs/design_config.json'),
  parent_short_final_path=str(PARENT/'configs/frozen_config.json'),parent_short_final_sha256=sha(PARENT/'configs/frozen_config.json'),
  word_target_instructions=dict(min_words=100,max_words=160,not_a_filter=True,no_forced_minimum_generation=True),
  config_finalized_before_generation=False,preparation_sha256=sha(__file__),protocol_sha256=sha(ROOT/'EXPERIMENT_PROTOCOL.md'))
 for task in TASKS:
  entry=cfg['data'][task]
  for split in ('dev','test'):
   entry['parent_'+split+'_path']=entry[split+'_path'];entry['parent_'+split+'_sha256']=entry[split+'_sha256']
   entry[split+'_path']=str(ROOT/f'data/{task}/{split}.json');entry[split+'_sha256']=sha(entry[split+'_path'])
 cfg['selection']['COCO']='SPICE>=.95Strict and CHAIRs<=Strict; full-description CIDEr is auxiliary and not a gate due known length floor'
 cfg['selection']['long_prompt_recalibration']=True
 cfg['mat_definition']=dict(native='sum(raw accepted Draft tokens)/sum(raw verification rounds)',
  effective='sum(firstEOS/budget effective accepted Draft tokens)/sum(effective verification rounds)',
  root_excluded=True,effective_completion_includes_first_EOS=True,AR='not applicable',auxiliary_progress='1+effective Draft MAT')
 cfg['quality_definition']=dict(QA='unique terminal Final answer line -> same official task evaluator; failures score0 and stay in denominator',
  COCO='full description unmodified; CIDEr-D/SPICE/CHAIR, no main label fusion',explanation_factual_safety='not established by QA answer accuracy')
 cfg['harness_sha256']={name:sha(ROOT/'harness'/name) for name in cfg['harness_sha256']}
 cfg['test_sha256']={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'tests').glob('*.py'))}
 for name in ['CLEAN_CHECKOUT.json','DATA_HISTORICAL_OVERLAP_SUPPLEMENT.json']:
  output=ROOT/'audit'/name;assert not output.exists();shutil.copyfile(PARENT/'audit'/name,output)
 cfg['checkout_manifest_sha256']=sha(ROOT/'audit/CLEAN_CHECKOUT.json')
 cfg['quality_protocol_path']=str(ROOT/'reports/QUALITY_PROTOCOL.md');cfg['quality_protocol_sha256']=sha(cfg['quality_protocol_path'])
 cfg['long_input_provenance_sha256']=sha(ROOT/'audit/LONG_INPUT_PROVENANCE.json')
 write(ROOT/'configs/design_config.json',cfg)
 evaluator=ROOT/'harness/evaluate_long.py'
 write(ROOT/'audit/METRIC_PARSER_FREEZE.json',dict(evaluator_sha256=sha(evaluator),
  original_evaluator_sha256=sha(PARENT/'harness/evaluate_tasks.py'),quality_protocol_sha256=cfg['quality_protocol_sha256'],
  frozen_before_any_long_official_score=True,parser='terminal_final_answer_v1',no_gold_assisted_extraction=True))
 print(json.dumps(dict(status='sealed',runtime_sources=len(cfg['harness_sha256']),development_requests=3240,main_requests=12600,max_new_tokens=512)))

def parity_gate():
 cfg=read(ROOT/'configs/design_config.json');rows=[];checks=0
 for task in TASKS:
  for shard in [0,1]:
   p=ROOT/f'results/parity/parity_{task}_shard{shard}.jsonl';done=read(p.with_suffix('.complete.json'))
   assert done['complete'] and done['result_sha256']==sha(p)
   own=[json.loads(x) for x in p.read_text().splitlines()];assert done['records']==len(own)
   rows+=own;checks+=len(read(p.with_suffix('.checks.json'))['checks'])
 assert len(rows)==cfg['expected_parity_requests']
 for r in rows:
  st=r['signal_stats']
  assert st.get('unsafe_bound_decisions',0)==st.get('visual_bound_decision_disagreements',0)==st.get('reference_decision_disagreements',0)==0
  for key,limit in [('reference_audit_projection_relative_error_max',1e-3),('reference_audit_component_abs_error_max',1e-2),('reference_audit_score_abs_error_max',1e-2)]:assert st.get(key) is None or st[key]<=limit
 write(ROOT/'audit/PARITY_GATE.json',dict(status='pass',requests=len(rows),checks=checks,config_sha256=sha(ROOT/'configs/design_config.json')))
 print('Long-output GPU parity passed')

def freeze():
 cfg=read(ROOT/'configs/design_config.json');assert read(ROOT/'audit/PARITY_GATE.json')['status']=='pass'
 selection=read(ROOT/'configs/lvspec_selection.json');assert selection['ready_for_main']
 for task in TASKS:
  parameters=selection['selections'][task]['parameters'];assert parameters['loose_ratio'] in [0.,.3,.5,.7]
  cfg['methods_by_task'][task]['lvspec_adapted'].update(parameters)
 cfg.update(config_finalized_before_generation=True,selection_sha256=sha(ROOT/'configs/lvspec_selection.json'),design_config_sha256=sha(ROOT/'configs/design_config.json'))
 write(ROOT/'configs/frozen_config.json',cfg)
 write(ROOT/'audit/MAIN_READY.json',dict(status='pass',config_sha256=sha(ROOT/'configs/frozen_config.json'),development_only_selection=True,test_outputs_inspected=False))
 print('Long-output main config frozen from development only')

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['data','design','parity-gate','freeze'],required=True)
 args=parser.parse_args();{'data':data,'design':design,'parity-gate':parity_gate,'freeze':freeze}[args.stage]()
