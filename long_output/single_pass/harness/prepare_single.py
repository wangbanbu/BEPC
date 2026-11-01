"""User-authorized one-pass continuation; keep the stopped run immutable."""
from pathlib import Path
import argparse,copy,hashlib,json,shutil
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,data):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),p
 p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def design():
 source=PARENT/'configs/design_config.json';cfg=read(source)
 assert cfg['repetitions']==3 and cfg['max_new_tokens']==512
 cfg.update(name='Clean ViSpec five-task long-output single-pass comparison',created_utc=datetime.now(timezone.utc).isoformat(),
  experiment_kind='long_prompt_512_budget_single_pass',repetitions=1,warmups=1,
  expected_development_requests=1080,expected_main_requests=4200,expected_parity_requests=0,inherited_parity_requests=120,
  parent_long_design_path=str(source),parent_long_design_sha256=sha(source),
  parent_original_run_inventory_path=str(ROOT/'audit/ORIGINAL_RUN_INVENTORY.json'),
  parent_original_run_inventory_sha256=sha(ROOT/'audit/ORIGINAL_RUN_INVENTORY.json'),
  original_run_reuse='all completed development repeat0; no repeat1/2 in new main comparison',
  user_requested_repetitions=1,user_request='现在先不进行重复测试，只测一遍，测完之后生成完整的实验报告',
  config_finalized_before_generation=False,preparation_sha256=sha(__file__),protocol_sha256=sha(ROOT/'EXPERIMENT_PROTOCOL.md'),
  timing_CI_scope='image/group sampling only; no repeated-run timing variability',
  run_repeatability_scope='not assessed by one-pass main; inherited parity/development evidence separate')
 cfg['harness_sha256']={n:sha(ROOT/'harness'/n) for n in cfg['harness_sha256']}
 cfg['harness_sha256']['run_single_pass.py']=sha(ROOT/'harness/run_single_pass.py')
 cfg['test_sha256']={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'tests').glob('*.py'))}
 cfg['quality_protocol_path']=str(ROOT/'reports/QUALITY_PROTOCOL.md');cfg['quality_protocol_sha256']=sha(cfg['quality_protocol_path'])
 cfg['parent_runtime_provenance_path']=str(PARENT/'audit/LONG_RUNTIME_COPY_PROVENANCE.json')
 cfg['parent_runtime_provenance_sha256']=sha(cfg['parent_runtime_provenance_path'])
 for n in ['CLEAN_CHECKOUT.json','DATA_HISTORICAL_OVERLAP_SUPPLEMENT.json']:
  p=ROOT/'audit'/n;assert not p.exists();shutil.copyfile(PARENT/'audit'/n,p)
 cfg['checkout_manifest_sha256']=sha(ROOT/'audit/CLEAN_CHECKOUT.json')
 assert read(ROOT/'audit/METRIC_PARSER_FREEZE.json')['evaluator_sha256']==sha(ROOT/'harness/evaluate_long.py')
 assert all(sha(x[k+'_path'])==x[k+'_sha256'] for x in cfg['data'].values() for k in ['dev','test','dev_labels','test_labels'])
 write(ROOT/'configs/design_config.json',cfg)
 gate=PARENT/'audit/PARITY_GATE.json';validation=PARENT/'audit/PARITY_VALIDATION.json'
 assert read(gate)['status']==read(validation)['status']=='pass'
 reuse=dict(status='pass',mode='inherited_frozen_core_parity_not_rerun',inherited_requests=120,new_parity_requests=0,
  parent_gate_path=str(gate),parent_gate_sha256=sha(gate),parent_validation_path=str(validation),parent_validation_sha256=sha(validation),
  unchanged_generation_core_sha256=cfg['harness_sha256']['run_benchmark.py'],single_pass_entry_sha256=cfg['harness_sha256']['run_single_pass.py'],
  config_sha256=sha(ROOT/'configs/design_config.json'),scope='same core generation/math/inputs/stop; new entry only scheduling/reuse/provenance')
 write(ROOT/'audit/PARITY_REUSE.json',reuse)
 write(ROOT/'audit/PARITY_GATE.json',reuse)
 print(json.dumps(dict(status='single_pass_design_frozen',runtime_sources=17,development=1080,main=4200,repetitions=1)))
def freeze():
 cfg=read(ROOT/'configs/design_config.json');selection=read(ROOT/'configs/lvspec_selection.json')
 assert cfg['repetitions']==1 and selection['ready_for_main']
 assert read(ROOT/'audit/PARITY_REUSE.json')['status']=='pass'
 for task in cfg['tasks']:
  p=selection['selections'][task]['parameters'];assert p['loose_ratio'] in [0.,.3,.5,.7]
  cfg['methods_by_task'][task]['lvspec_adapted'].update(p)
 cfg.update(config_finalized_before_generation=True,selection_sha256=sha(ROOT/'configs/lvspec_selection.json'),design_config_sha256=sha(ROOT/'configs/design_config.json'))
 write(ROOT/'configs/frozen_config.json',cfg)
 write(ROOT/'audit/MAIN_READY.json',dict(status='pass',config_sha256=sha(ROOT/'configs/frozen_config.json'),development_only_selection=True,test_outputs_inspected=False,repetitions=1))
 print('Single-pass main config frozen from development only.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['design','freeze'],required=True);a=p.parse_args();{'design':design,'freeze':freeze}[a.stage]()
