"""Predeclare five-task method identity and freeze dev-only LVSpec selection."""
from pathlib import Path
import argparse,copy,hashlib,json,subprocess
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
TASKS=['mme','gqa','textvqa','scienceqa','coco']
RUNTIME_FILES=['run_benchmark.py','resources.py','compact_logits.py','online_controller.py','epcc_online.py','visual_online.py','optimized_visual.py','refined_controller.py','refined_math.py','refined_observer.py','speed_controller.py','speed_math.py','speed_observer.py','terminal_update.py','bounds.py','lvspec_controller.py']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists(),p
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def templates():
    shared=dict(compact_logits=True,terminal_skip=True)
    relaxed=dict(shared,controller='speed',round_mismatch_budget=2,response_mismatch_budget=4,rich_diagnostics=False)
    lv=dict(shared,family='LVSPEC',loose_ratio=.7,similarity_top_n=10,draft_window=10,pst_enabled=True,pst_include_root=True,visual_reference='image_tokens',stable_rank_ties=True,disable_relaxation=False)
    return dict(ar=dict(family='AR',compact_logits=False,terminal_skip=False),
        vispec=dict(family='vispec',compact_logits=False,terminal_skip=False),
        strict_shared=dict(shared,family='vispec'),
        bepc_algorithm=dict(relaxed,family='REFINED_BEPC',rank_u=8,rank_p=4,text_window=32,visual_pool_size=64,gamma=.5,kappa=1.,eta=1e-4,residual_attenuation=False,correction_mode='both',tau=1.,solver='fast',prune=True,visual_first_prune=True),
        margin4_ccve=dict(relaxed,family='ME',tau_M=4.,tau_E=-.165220245718956,threshold=-.165220245718956),
        lvspec_paper=lv,lvspec_adapted=copy.deepcopy(lv))
def design():
    methods=templates();dev={k:copy.deepcopy(methods[k]) for k in ['ar','vispec','strict_shared','bepc_algorithm','margin4_ccve']}
    for ratio in [0.,.3,.5,.7]:dev['lvspec_ratio'+format(ratio,'g')]=dict(methods['lvspec_paper'],loose_ratio=ratio)
    parity=copy.deepcopy(methods)
    parity.update(ar_hidden_reference=dict(methods['ar'],ar_output_hidden_states=True),
        bepc_reference=dict(methods['bepc_algorithm'],controller='legacy',solver='reference',prune=False,visual_first_prune=False),
        lvspec_disabled=dict(methods['lvspec_paper'],disable_relaxation=True))
    data={}
    for task in TASKS:
        entry={}
        for split,count in [('dev',20),('test',100)]:
            path=ROOT/f'data/{task}/{split}.json';rows=read(path)
            assert len(rows)==count*(2 if task=='mme' else 1)
            assert len({s['group_id'] for s in rows})==count
            assert [s['ordinal'] for s in rows]==list(range(len(rows)))
            for row in rows:
                assert sha(row['image_path'])==row['image_sha256']
                assert 'answer' not in row and 'answers' not in row and 'label' not in row
            entry[split+'_path']=str(path);entry[split+'_sha256']=sha(path)
            entry[split+'_labels_path']=str(path.with_name(split+'_labels.json'));entry[split+'_labels_sha256']=sha(entry[split+'_labels_path'])
        assert not {r['image_sha256'] for r in read(entry['dev_path'])}&{r['image_sha256'] for r in read(entry['test_path'])},task
        data[task]=entry
    oldcfg=read('/data/wbb/vispec-official-reproduction/research/harmful_mismatch/coco_only/bepc_speed_optimization/configs/frozen_config.json')
    source=Path(oldcfg['model_identity_manifest_path']);model=read(source)
    for item in model['files']:
        st=Path(item['path']).stat();prev=item['stat_after']
        assert (st.st_size,st.st_mtime_ns,st.st_ctime_ns,st.st_ino,st.st_dev)==(prev['size_bytes'],prev['mtime_ns'],prev['ctime_ns'],prev['inode'],prev['device'])
    write(ROOT/'audit/MODEL_IDENTITY.json',dict(original_manifest=str(source),original_sha256=sha(source),files=model['files'],stat_identity_verified=True))
    cfg=dict(name='Clean GitHub ViSpec five-task decoding benchmark',created_utc=datetime.now(timezone.utc).isoformat(),
        upstream=str(ROOT/'upstream/ViSpec'),upstream_commit='49a33ff0ca31e281b9dc8803fd6f3041c6c0e135',
        target='/data/wbb/model/Qwen2.5-VL-7B-Instruct',draft='/data/wbb/model/ViSpec-Qwen2.5-VL-7B-Instruct',
        model_identity_path=str(ROOT/'audit/MODEL_IDENTITY.json'),model_identity_sha256=sha(ROOT/'audit/MODEL_IDENTITY.json'),
        tree=dict(total_token=30,depth=3,top_k=8,num_q=2),temperature=0.,seed=0,order_seed=2026103109,bootstrap_seed=2026103109,
        system_prompt='You are a helpful assistant.',processor=dict(use_fast=False,min_pixels=256*28*28,max_pixels=1280*28*28),
        max_new_tokens=128,max_length=2048,eos_token_ids=[151645],tasks=TASKS,data=data,development_methods=dev,methods_by_task={t:copy.deepcopy(methods) for t in TASKS},main_method_templates=methods,
        parity_methods=parity,parity_pairs=[['strict_shared','vispec'],['bepc_algorithm','bepc_reference'],['ar','ar_hidden_reference'],['lvspec_disabled','strict_shared']],
        repetitions=3,warmups=3,shards=2,bootstrap=5000,config_finalized_before_generation=False,
        harness_sha256={n:sha(ROOT/'harness'/n) for n in RUNTIME_FILES},test_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'tests').glob('*.py'))},
        protocol_sha256=sha(ROOT/'EXPERIMENT_PROTOCOL.md'),checkout_manifest_sha256=sha(ROOT/'audit/CLEAN_CHECKOUT.json'),
        preparation_sha256=sha(__file__),expected_development_requests=9*120*3,expected_main_requests=7*600*3,expected_parity_requests=10*12,
        comparisons=[['bepc_algorithm',m] for m in methods if m!='bepc_algorithm'],secondary_comparisons=[['lvspec_adapted','lvspec_paper'],['lvspec_adapted','strict_shared']],
        selection=dict(grid=[0.,.3,.5,.7],paper_ratio=.7,conditional_on_paper_quality_failure=True,QA='no decrease from nativeStrict official task score',MME='total acc+accplus and overall accuracy no decrease',COCO='CIDEr/SPICE >=.95Strict and CHAIRs <=Strict',none_eligible='explicit disable_relaxation Strict fallback',no_test_tuning=True),
        primary_quality_repeat=0,reference_method='vispec',primary_method='bepc_algorithm',no_training=True,no_new_judge_calls=True)
    write(ROOT/'configs/design_config.json',cfg)
    print(json.dumps(dict(status='sealed',runtime_sources=len(cfg['harness_sha256']),development_requests=cfg['expected_development_requests'],main_requests=cfg['expected_main_requests'])))
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
        for k,b in [('reference_audit_projection_relative_error_max',1e-3),('reference_audit_component_abs_error_max',1e-2),('reference_audit_score_abs_error_max',1e-2)]:assert st.get(k) is None or st[k]<=b
    write(ROOT/'audit/PARITY_GATE.json',dict(status='pass',requests=len(rows),checks=checks,config_sha256=sha(ROOT/'configs/design_config.json')))
    print('parity pass')
def freeze():
    cfg=read(ROOT/'configs/design_config.json');assert read(ROOT/'audit/PARITY_GATE.json')['status']=='pass'
    selection=read(ROOT/'configs/lvspec_selection.json');assert selection['ready_for_main']
    for task in TASKS:
        selected=selection['selections'][task];parameters=selected['parameters']
        assert parameters['loose_ratio'] in [0.,.3,.5,.7]
        cfg['methods_by_task'][task]['lvspec_adapted'].update(parameters)
    cfg.update(config_finalized_before_generation=True,selection_sha256=sha(ROOT/'configs/lvspec_selection.json'),design_config_sha256=sha(ROOT/'configs/design_config.json'))
    write(ROOT/'configs/frozen_config.json',cfg)
    write(ROOT/'audit/MAIN_READY.json',dict(status='pass',config_sha256=sha(ROOT/'configs/frozen_config.json'),development_only_selection=True,test_outputs_inspected=False))
    print(json.dumps(dict(status='frozen',lvspec={t:cfg['methods_by_task'][t]['lvspec_adapted'] for t in TASKS})))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['design','parity-gate','freeze'],required=True);a=p.parse_args();{'design':design,'parity-gate':parity_gate,'freeze':freeze}[a.stage]()
