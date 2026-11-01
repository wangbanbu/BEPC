"""Independent CPU-only audit of the user-requested single-pass long benchmark.

The pinned earlier auditor supplies unchanged token/call/MAT, official-metric,
selection and grouped-statistics contracts. This file audits the changed
scheduler, exact repeat-0 reuse and genuinely single-pass population separately.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT.parent
SHORT=PARENT.parent
LONG_VALIDATOR=PARENT/'harness/validate_long.py'
LONG_VALIDATOR_SHA='b9fc98d6070b067ba468d91731e0ee842981e549ab2669ab62d9ff4cc18c331a'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def need(condition,message):
    if not condition:raise AssertionError(message)
def canonical_sha(row):return hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()

need(sha(LONG_VALIDATOR)==LONG_VALIDATOR_SHA,'read-only long validator changed')
spec=importlib.util.spec_from_file_location('_single_pinned_long_validator',LONG_VALIDATOR)
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
v.ROOT=ROOT;v.BASE=SHORT;v.base.ROOT=ROOT
TASKS,METHODS,row_contract=v.TASKS,v.METHODS,v.row_contract

@contextmanager
def old_long_root():
    before=v.ROOT;v.ROOT=PARENT
    try:yield
    finally:v.ROOT=before

def parent_function_ast(name):
    tree=ast.parse(LONG_VALIDATOR.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='LongAudit')
    return next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)

def single_analysis_contract():
    """Only change the pinned count12600→4200; keep all draw/Holm/point checks."""
    function=parent_function_ast('analysis')
    replacements=0
    class Count(ast.NodeTransformer):
        def visit_Constant(self,node):
            nonlocal replacements
            if node.value==12600:
                replacements+=1
                return ast.copy_location(ast.Constant(value=4200),node)
            return node
    function=Count().visit(function)
    need(replacements==1,'exactly one inherited analysis count adaptation')
    namespace=dict(v.__dict__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(__file__)+'::pinned_analysis_contract','exec'),namespace)
    return namespace['analysis']

def imported_row_contract(row,original,entry,cfg_sha):
    need(row['measurement_origin']=='reused_parent_repeat0','import origin')
    expected=dict(original,config_sha256=cfg_sha)
    need({k:value for k,value in row.items() if k not in ('measurement_origin','single_pass_provenance')}==expected,'import preserves every original measurement and output')
    p=row['single_pass_provenance']
    need(p['parent_path']==entry['source_path'] and p['parent_line']==entry['source_line'],'source identity/line map')
    need(p['parent_record_sha256']==entry['source_row_sha256']==canonical_sha(original),'canonical original row SHA')
    need(p['parent_config_sha256']==original['config_sha256']==entry['source_config_sha256'],'original config mapping')
    need(p['normalized_fields']=={'config_sha256':{'original':original['config_sha256'],'normalized':cfg_sha}},'only config normalization')
    need(original['repeat']==row['repeat']==0,'all reused rows are repeat0')

class SingleAudit(v.LongAudit):
    def inventory(self):
        inv=read(ROOT/'audit/ORIGINAL_RUN_INVENTORY.json')
        need(inv['main_records']==0 and inv['successful_reusable_repeat0']==1003,'original stopping point 1003dev0/0main')
        need(not inv['failed_rows'] and not inv['partial_or_malformed_records'],'all original failed/partial cases retained and resolved')
        need(inv['gold_contents_read'] is False and inv['source_files_modified'] is False,'inventory label blind/read only')
        for path,item in inv['original_manifest'].items():need(sha(path)==item['sha256'],'preserved original file: '+path)
        need(all(not p['proc_exists'] for p in inv['stop_evidence']['original_pids'].values()),'original workers stopped before scheduling change')
        self.inventory_data=inv
        self.import_map={(r['task'],r['shard'],r['ordinal'],r['method']):r for r in inv['reusable_rows']}
        need(len(self.import_map)==1003,'unique original repeat0 inventory')
        return dict(reusable_repeat0=1003,missing_repeat0=77,main_started=False,old_files=len(inv['original_manifest']),all_old_file_sha_verified=True)

    def configuration(self,phase):
        path=ROOT/'configs'/('frozen_config.json' if phase=='main' else 'design_config.json')
        cfg,old=read(path),read(PARENT/'configs/design_config.json')
        for key in ('upstream','upstream_commit','target','draft','model_identity_path','model_identity_sha256','tree','temperature','seed','order_seed','bootstrap_seed','system_prompt','processor','max_length','eos_token_ids','tasks','shards','bootstrap','primary_quality_repeat','no_training','no_new_judge_calls','main_method_templates','development_methods','parity_methods','parity_pairs','data','max_new_tokens','selection','comparisons','secondary_comparisons','long_input_provenance_sha256'):
            need(cfg[key]==old[key],'unchanged long setting: '+key)
        need(cfg['repetitions']==cfg['warmups']==1,'one measurement and at most one warmup per method/task/GPU')
        need(cfg['expected_development_requests']==1080 and cfg['expected_main_requests']==4200,'single-pass counts')
        need(cfg['max_new_tokens']==512 and cfg['bootstrap']==5000,'512budget/5000group draws')
        need(set(cfg['harness_sha256'])==set(old['harness_sha256'])|{'run_single_pass.py'},'16originalruntime plus scheduler')
        for name,digest in cfg['harness_sha256'].items():
            need(sha(ROOT/'harness'/name)==digest,'single runtime seal: '+name)
            if name!='run_single_pass.py':need(digest==old['harness_sha256'][name]==sha(PARENT/'harness'/name),'byte-identical original generation runtime: '+name)
        tree=ast.parse((ROOT/'harness/run_single_pass.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SinglePassRuntime')
        need(len(cls.bases)==1 and ast.unparse(cls.bases[0])=='core.Runtime','frozen Runtime inheritance')
        need({n.name for n in cls.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}=={'__init__','run'},'generate/inputs/serial_ar/restore/verify_checkout/main all inherited unchanged')
        for name,digest in cfg.get('test_sha256',{}).items():need(sha(ROOT/name)==digest,'single test source: '+name)
        need(sha(ROOT/'EXPERIMENT_PROTOCOL.md')==cfg['protocol_sha256'],'single protocol SHA')
        model=read(cfg['model_identity_path'])
        need(sha(cfg['model_identity_path'])==cfg['model_identity_sha256'] and sha(model['original_manifest'])==model['original_sha256'],'checkpoint manifests')
        for item in model['files']:
            stat,expected=Path(item['path']).stat(),item['stat_after']
            need((stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,stat.st_ino,stat.st_dev)==(expected['size_bytes'],expected['mtime_ns'],expected['ctime_ns'],expected['inode'],expected['device']),'unchanged checkpoint '+item['path'])
        freeze=read(ROOT/'audit/METRIC_PARSER_FREEZE.json')
        need(freeze['evaluator_sha256']==sha(ROOT/'harness/evaluate_long.py') and freeze['original_evaluator_sha256']==sha(SHORT/'harness/evaluate_tasks.py'),'frozen current and official scorer')
        need(freeze['parent_long_evaluator_sha256']==sha(PARENT/'harness/evaluate_long.py') and freeze['parser']=='terminal_final_answer_v1' and freeze['no_gold_assisted_extraction'] and freeze['parser_function_AST_identical_to_parent'],'same frozen final-answer parser')
        need(freeze['quality_protocol_sha256']==cfg['quality_protocol_sha256']==sha(ROOT/'reports/QUALITY_PROTOCOL.md'),'quality contract')
        if phase=='main':
            need(cfg['config_finalized_before_generation'],'main pre-frozen')
            need(cfg['design_config_sha256']==sha(ROOT/'configs/design_config.json') and cfg['selection_sha256']==sha(ROOT/'configs/lvspec_selection.json'),'design/selection freeze')
            ready=read(ROOT/'audit/MAIN_READY.json')
            need(ready['status']=='pass' and ready['config_sha256']==sha(path) and ready['development_only_selection'] and not ready['test_outputs_inspected'],'main readiness before test')
        self.configs[phase]=cfg
        return dict(sha256=sha(path),runtime_sources=17,original_runtime_byte_identical=16,repetitions=1,warmups=1,max_new_tokens=512)

    def data(self):
        # Config intentionally references existing read-only long input paths.
        with old_long_root():return super().data()

    def parity_reuse(self):
        evidence=read(ROOT/'audit/PARITY_REUSE.json')
        inv=self.inventory_data
        status=read(PARENT/'audit/PARITY_VALIDATION.json')
        gate=read(PARENT/'audit/PARITY_GATE.json')
        need(status['status']=='pass' and all(x['passed'] for x in status['checks'] if x['required_for_execution_integrity']),'original independently audited parity')
        need(gate['status']=='pass','original parity gate')
        cfg=read(PARENT/'configs/design_config.json')
        rows=[];comparisons=0
        for task in TASKS:
            samples={str(s['question_id']):s for s in self.samples[task,'dev'] if s['group_ordinal']<2}
            taskrows=[]
            for shard in (0,1):
                path=PARENT/f'results/parity/parity_{task}_shard{shard}.jsonl'
                need(sha(path)==inv['original_manifest'][str(path)]['sha256'],'old parity result SHA')
                for line in path.read_text().splitlines():
                    r=json.loads(line);row_contract(r,samples[str(r['question_id'])],cfg['parity_methods'][r['method']],cfg,'parity',shard);taskrows.append(r)
            lookup={(r['question_id'],r['method']):r for r in taskrows}
            for question in samples:
                for left,right in cfg['parity_pairs']:
                    a,b=lookup[question,left],lookup[question,right]
                    need(all(a[k]==b[k] for k in v.base.PARITY_FIELDS),'inherited original parity tokens/MAT')
                    comparisons+=1
            rows.extend(taskrows)
        need(len(rows)==120 and comparisons==48,'reuse actual120records/48comparisons without rerunning')
        return dict(inherited_requests=120,inherited_pairwise_checks=48,new_parity_requests=0,source_gate_sha256=sha(PARENT/'audit/PARITY_GATE.json'),source_validation_sha256=sha(PARENT/'audit/PARITY_VALIDATION.json'),reuse_seal_sha256=sha(ROOT/'audit/PARITY_REUSE.json'))

    def loaded(self,phase):
        cfg=self.configs[phase];out=[]
        for shard in (0,1):
            meta=read(ROOT/f'audit/{phase}_shard{shard}.single_pass_loaded.json')
            path=ROOT/'configs'/('frozen_config.json' if phase=='main' else 'design_config.json')
            need(meta['config_sha256']==sha(path) and meta['source_sha256']==cfg['harness_sha256'],'actual loaded single config/runtime')
            need(meta['single_pass_runtime_entry_sha256']==cfg['harness_sha256']['run_single_pass.py'] and meta['inherited_generation_core_sha256']==cfg['harness_sha256']['run_benchmark.py'],'loaded scheduler/inherited core')
            need(meta['physical_gpu']==str(shard) and meta['AR_memory_includes_idle_loaded_Draft'],'fixed GPU/memory disclosure')
            for name,item in meta['loaded_vispec_sources'].items():
                need(Path(item['path']).resolve().is_relative_to(Path(cfg['upstream']).resolve()) and sha(item['path'])==item['sha256'],'fresh imported official file '+name)
            resource=read(ROOT/f'audit/{phase}_shard{shard}.gpu_resource.json')
            need(resource['checks']>0 and not resource['events'] and resource['physical_gpu']==str(shard),'exclusive GPU')
            out.append(dict(shard=shard,checks=resource['checks']))
        return out

    def phase(self,phase):
        cfg=self.configs[phase];split='test' if phase=='main' else 'dev'
        config_sha=sha(ROOT/'configs'/('frozen_config.json' if phase=='main' else 'design_config.json'))
        rows_by_task={};manifest={};totals=Counter();used=set();changed_preprocessing=[];warmups=0
        for task in TASKS:
            samples=self.samples[task,split];byid={str(s['question_id']):s for s in samples}
            methods=cfg['methods_by_task'][task] if phase=='main' else cfg['development_methods'];rows=[]
            for shard in (0,1):
                path=ROOT/f'results/{phase}/{phase}_{task}_shard{shard}.jsonl';marker=read(path.with_suffix('.complete.json'))
                batch=[json.loads(l) for l in path.read_text().splitlines() if l.strip()]
                own=[s for s in samples if s['group_ordinal']%2==shard]
                expected={(str(s['question_id']),m,0) for s in own for m in methods}
                actual=[(str(r['question_id']),r['method'],r['repeat']) for r in batch]
                need(len(actual)==len(set(actual)) and set(actual)==expected,'complete single-pass grid, no duplicate/repeat')
                need(marker['complete'] and marker['records']==len(batch) and marker['questions']==len(own) and marker['phase']==phase and marker['task']==task,'complete group count')
                need(marker['result_sha256']==sha(path) and marker['metadata']['config_sha256']==config_sha and marker['metadata']['source_sha256']==cfg['harness_sha256'],'complete file/config/runtime seal')
                need(marker['repetitions']==1 and marker['all_measurements_single_pass'],'single-pass marker')
                imported=new=0
                for r in batch:
                    need(r['config_sha256']==config_sha,'row config normalized to single request plan')
                    row_contract(r,byid[str(r['question_id'])],methods[r['method']],cfg,phase,shard)
                    key=(task,shard,r['ordinal'],r['method'])
                    if r['measurement_origin']=='reused_parent_repeat0':
                        need(phase=='development' and key in self.import_map and key not in used,'one-to-one inventoried import')
                        entry=self.import_map[key];source=Path(entry['source_path'])
                        original=json.loads(source.read_text().splitlines()[entry['source_line']-1])
                        imported_row_contract(r,original,entry,config_sha)
                        need(r['single_pass_provenance']['parent_file_sha256']==sha(source),'source file provenance')
                        used.add(key);imported+=1
                    else:
                        need(r['measurement_origin']=='new_single_pass_generation' and (phase!='development' or key not in self.import_map),'never regenerate successful imported row')
                        p=r['single_pass_provenance']
                        need(p['runtime_entry_sha256']==cfg['harness_sha256']['run_single_pass.py'] and p['inherited_generation_core_sha256']==cfg['harness_sha256']['run_benchmark.py'],'new generation source seal')
                        new+=1
                need(marker['parent_repeat0_reused_records']==imported and marker['newly_generated_records']==new,'complete origins')
                totals['imported']+=imported;totals['new']+=new
                warmfile=ROOT/f'audit/{phase}_shard{shard}_{task}.warmup.json'
                if new:
                    warm=read(warmfile)
                    need(warm['requests']==len(methods) and warm['lifecycle_checked_every_request'] and warm['one_warmup_per_method'] and warm['seconds']>0,'one warmup/method on actual generation shards')
                    warmups+=warm['requests']
                else:need(not warmfile.exists(),'no new model warmups for import-only completed task')
                rows.extend(batch);manifest[str(path)]=sha(path)
            lookup={(str(r['question_id']),r['method']):r for r in rows}
            for s in samples:
                schedule=list(methods);random.Random(cfg['order_seed']+TASKS.index(task)*1000003+s['ordinal']*101).shuffle(schedule)
                selected=[lookup[str(s['question_id']),m] for m in schedule]
                need([r['schedule_position'] for r in selected]==list(range(len(methods))),'same frozen randomized repeat0 schedule')
                for field in ('prompt_token_ids','input_tensor_sha256','image_grid_thw'):
                    need(all(r[field]==selected[0][field] for r in selected),'same paired actual input '+field)
                prep={r['input_preprocessing_seconds'] for r in selected}
                if len(prep)>1:
                    need({r['measurement_origin'] for r in selected}=={'reused_parent_repeat0','new_single_pass_generation'},'preprocessing variation limited to documented resume boundary')
                    changed_preprocessing.append(dict(task=task,question_id=s['question_id'],observations=len(prep)))
                a,b=lookup[str(s['question_id']),'strict_shared'],lookup[str(s['question_id']),'vispec']
                changed=[k for k in v.base.PARITY_FIELDS if a[k]!=b[k]]
                if changed:self.equivalence_differences.append(dict(phase=phase,task=task,question_id=s['question_id'],fields=changed))
            rows_by_task[task]=rows
        count=sum(len(r) for r in rows_by_task.values())
        need(count=={'development':1080,'main':4200}[phase],'1080dev/4200main')
        need(warmups=={'development':18,'main':70}[phase],'18gap-dev/70main single warmups')
        if phase=='development':need(used==set(self.import_map) and totals['imported']==1003 and totals['new']==77,'all1003 original repeat0 reused plus77 gaps only')
        else:need(totals['imported']==0 and totals['new']==4200,'all4200 main measured once')
        self.phase_rows[phase]=rows_by_task;self.hashes[phase]=manifest
        return dict(measured_requests=count,warmup_requests=warmups,origins=dict(totals),repeat_run_reproducibility='not_assessed_single_pass',resume_preprocessing_differences=changed_preprocessing)

    analysis=single_analysis_contract()

    def determinism(self):
        need(not self.equivalence_differences,'shared/native Strict mismatch retained: investigate')
        return dict(repeat_run_reproducibility='not_assessed_single_pass',equivalence_differences=0,original_small_GPU_parity_inherited=True)

    def run(self,phase):
        self.check('preserved_original_run_and_label_blind_inventory',self.inventory)
        self.check('fresh_official_sources_and_readonly_originals',self.sources)
        phases=['development','main'] if phase in ('all','main') else ['development'] if phase=='development' else []
        for p in (['development','main'] if 'main' in phases else ['development']):self.check(p+'_configuration',lambda p=p:self.configuration(p))
        if self.configs:self.check('exact_same_inputs_and_official_labels',self.data)
        if self.samples:self.check('reuse_original120_GPU_parity_no_new_requests',self.parity_reuse)
        for p in phases:
            self.check(p+'_actual_imports_GPU_exclusivity',lambda p=p:self.loaded(p))
            self.check(p+'_one_pass_grid_origins_MAT_calls_budgets',lambda p=p:self.phase(p))
        scores={}
        for p in phases:
            if p in self.phase_rows:
                score=self.check(p+'_official_quality_and_seals',lambda p=p:self.metrics(p))
                if score is not None:scores[p]=score
        if 'development' in scores and (ROOT/'configs/frozen_config.json').exists():self.check('development_only_threshold_selection',lambda:self.selection(scores['development']))
        if 'main' in self.phase_rows:self.check('grouped5000_bootstrap_points_Holm_output_seals',self.analysis)
        self.check('failures_retained',self.failures)
        self.check('strict_shared_native_equivalence',self.determinism)
        self.check('project_history_unseen_claim',self.strict_freshness,required=False)
        passed=all(x['passed'] for x in self.checks if x['required_for_execution_integrity'])
        need('torch' not in sys.modules,'CPU-only validator imported torch')
        return dict(status='pass' if passed else 'No-Go',requested_phase=phase,execution_and_data_integrity_status='pass' if passed else 'No-Go',strict_project_history_unseen_claim_status='No-Go',independent_cpu_audit=True,gpu_or_checkpoint_loaded=False,judge_or_training=False,validator_sha256=sha(__file__),pinned_long_validator_sha256=LONG_VALIDATOR_SHA,checks=self.checks,evidence=self.evidence,warnings=self.warnings,limitations=['Only one measured request per method/question; run-to-run timing variability and repeat determinism are not assessed.','5000 confidence intervals concern paired image/group sampling, not independent repeated timing runs.','Development reuse spans original and resumed sessions; source measurements and preprocessing differences remain explicit.','Long QA answer accuracy does not verify explanation truthfulness; automatic caption metrics are not human factual Ground Truth.','Same existing images as the short-output cohort, including all declared historical overlaps; no unseen-image generalization claim.','Token/call/MAT point estimates and grouped bootstrap draws/Holm correction are independently checked; every interval is not independently reimplemented.'])

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=['preflight','development','main','all'],default='all');parser.add_argument('--output',type=Path,default=ROOT/'audit/FINAL_DELIVERY_VALIDATION.json');args=parser.parse_args()
    result=SingleAudit().run(args.phase);args.output.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if args.output.exists():need(args.output.read_text()==payload,'preserve previous audit; use a new output for changed stage')
    else:args.output.write_text(payload)
    md=['# 单次长输出实验独立交付审计','',f'Status: **{result["status"]}**；phase `{args.phase}`。CPU-only。','', '| Check | Result |','|---|---|']
    md += ['| '+c['name']+' | '+('PASS' if c['passed'] else 'No-Go: '+c['exception'].replace('|','/'))+' |' for c in result['checks']]
    md += ['','本轮每题/方法仅一次测量；不能据空的重复差异列表声称已验证重复确定性。完整源库存、来源映射、MAT、调用预算、官方评分和统计封印见JSON。','','## 局限','']+['- '+s for s in result['limitations']]
    args.output.with_suffix('.md').write_text('\n'.join(md)+'\n')
    print(json.dumps(dict(status=result['status'],checks=len(result['checks']),required_failures=sum(not c['passed'] and c['required_for_execution_integrity'] for c in result['checks']),output=str(args.output))))
    return 0 if result['status']=='pass' else 1

if __name__=='__main__':raise SystemExit(main())
