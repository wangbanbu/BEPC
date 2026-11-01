"""Independent CPU-only audit of the official SPICE heap-resource recovery.

No Java, GPU, checkpoint, scorer or benchmark request is executed. The audit
checks the preserved failed attempt, identical input, exact command change,
dependencies, actual new log/output and frozen evaluator's subsequent cache hit.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import tempfile
import unittest
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
EXPECTED_OFFICIAL_SHA='1f491b0f1dba0a148b5a65ddc574c73c76878c098c967fc1f45fc08ba2a83fc3'
FATAL=re.compile(r'OutOfMemoryError|Java\s+heap\s+space|out\s+of\s+memory|\bOOM\b|\bignored?\b|Could\s+not\s+score\s+batched|NullPointerException|Exception\s+in\s+thread|Error:\s+Could\s+not\s+(?!cache\s+item\b)',re.I)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def need(condition,message):
    if not condition:raise AssertionError(message)
def close(a,b):return a is None and b is None or a is not None and b is not None and math.isfinite(a) and math.isfinite(b) and abs(a-b)<=1e-12*max(1.,abs(a),abs(b))

def command_contract(before,after):
    need(isinstance(before,list) and isinstance(after,list) and len(before)==len(after),'same Java argument shape')
    changes=[dict(index=i,original=a,recovery=b) for i,(a,b) in enumerate(zip(before,after)) if a!=b]
    need(changes==[dict(index=2,original='-Xmx16G',recovery='-Xmx64G')],'only JVM heap may change')
    need(before[:2]==['java','-XX:ActiveProcessorCount=8'] and after[-2:]==['-threads','1'],'same8CPU/threads1')
    need('-subset' in after and '-silent' in after and 'edu.anu.spice.SpiceScorer' in after,'same official scorer/options')
    return changes

def output_contract(payload,expected_ids,log):
    hits=FATAL.findall(log)
    need(not hits,'fatal or skipped parsing indicators: '+repr(hits))
    need(isinstance(payload,list) and len(payload)==len(expected_ids),'full output batch')
    ids=[int(r['image_id']) for r in payload]
    need(len(set(ids))==len(ids) and set(ids)==set(expected_ids),'complete unique output IDs')
    scores={}
    for row in payload:
        need('f' in row['scores']['All'],'official All.f exists')
        value=row['scores']['All']['f']
        try:value=float(value)
        except (TypeError,ValueError):value=float('nan')
        scores[int(row['image_id'])]=value if math.isfinite(value) else None
    return scores

def unchanged(manifest):
    for path,digest in manifest.items():
        expected=digest['sha256'] if isinstance(digest,dict) else digest
        need(Path(path).is_file() and sha(path)==expected,'protected existing evidence changed: '+path)
    return len(manifest)

class RecoveryAudit:
    def __init__(self):self.checks=[];self.evidence={}
    def check(self,name,function):
        try:
            evidence=function();self.checks.append(dict(name=name,passed=True,evidence=evidence));return evidence
        except Exception as e:self.checks.append(dict(name=name,passed=False,exception=repr(e)));return None

    def manifest(self):
        self.path=ROOT/'audit/SPICE_RESOURCE_RECOVERY.json';self.record=read(self.path)
        self.baseline=read(ROOT/'audit/SPICE_RECOVERY_INDEPENDENT_BASELINE.json')
        r=self.record
        need(r['status']=='pass' and r['java_exit_code']==0 and r['no_gpu_inference'],'successful isolated CPU attempt')
        need(r['main_gpu_records']==self.baseline['main_requests']==4200 and self.baseline['sealed_methods_before']==78,'4200GPU rows/78 prior successful score seals')
        need(r['actual_java_producer_heap']=='64G' and r['cache_ready_for_frozen_analyzer'],'actual64G producer, no synthetic output')
        need(sha(r['recovery_source']['path'])==r['recovery_source']['sha256'],'actual recovery source seal')
        need(sha(r['official_wrapper']['path'])==r['official_wrapper']['sha256']==EXPECTED_OFFICIAL_SHA,'original official evaluator unchanged')
        changes=command_contract(r['original_command'],r['recovery_command'])
        need(r['command_difference']==changes and not r['caption_references_ptb_and_metric_rules_changed'],'resource-only recovery')
        self.folder=Path(r['recovery_command'][r['recovery_command'].index('-out')+1]).parent
        need(self.folder==ROOT/'metrics/main/coco/lvspec_paper/official_scoring/spice','recovered exact method path')
        return dict(recovery_manifest_sha256=sha(self.path),recovery_source_sha256=r['recovery_source']['sha256'],actual_heap='64G',command_changes=changes)

    def dependencies(self):
        r=self.record
        count=unchanged(r['scorer_dependencies'])+unchanged(r['scorer_library_sha256'])
        cp=r['recovery_command'][r['recovery_command'].index('-cp')+1]
        expected=set()
        for part in cp.split(':'):
            p=Path(part)
            expected.update(str(f) for f in p.parent.glob(p.name) if f.is_file() and f.suffix=='.jar')
        sealed={p for p in r['scorer_dependencies'] if Path(p).suffix=='.jar'}|set(r['scorer_library_sha256'])
        need(expected==sealed,'all actual classpath JARs individually sealed')
        return dict(dependency_files=count,actual_classpath_jars=len(expected),all_sha256_verified=True)

    def archive_and_input(self):
        r=self.record;archive=Path(r['preserved_failure_directory'])
        need(archive==ROOT/'audit/spice_recovery/failed_attempt_16g/lvspec_paper','failed attempt retained under audit, outside active metric tree')
        index=archive.parent/'FAILURE_INVENTORY.json'
        need(sha(index)==r['preserved_failure_inventory_sha256'] and read(index)==r['original_failure_inventory'],'failure inventory preserved')
        actual={str(p.relative_to(archive)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in archive.rglob('*') if p.is_file()}
        need(actual==r['original_failure_inventory'],'entire failed attempt byte-identical')
        for original,expected in self.baseline['original_failed_attempt']['files'].items():
            relative=Path(original).relative_to(Path(self.baseline['original_failed_attempt']['directory']).parent)
            need(actual[str(relative)]==expected,'independently snapshotted failure file preserved')
        need(self.baseline['original_failed_attempt']['spice_output_exists'] is False and self.baseline['original_failed_attempt']['score_seal_exists'] is False,'failed attempt supplied no usable score')
        need(not (archive/'official_scoring/spice/output.json').exists() and not (archive/'SCORING_SEAL.json').exists(),'archive does not fabricate prior output/seal')
        old_input=archive/'official_scoring/spice/input.json';new_input=self.folder/'input.json'
        need(old_input.read_bytes()==new_input.read_bytes() and sha(new_input)==r['input_sha256'],'exact original input bytes, not shortened/filtered')
        ids=[int(row['image_id']) for row in read(new_input)]
        need(len(ids)==100 and len(set(ids))==100 and set(ids)==set(range(100)),'all100 unchanged input IDs')
        need(r['fresh_cache_initially_empty'] and r['fresh_cache_path']==str(self.folder/'cache'),'fresh cache attested by frozen recovery source')
        old_log=(archive/'official_scoring/spice/scorer.log').read_text(errors='replace')
        need('OutOfMemoryError' in old_log and FATAL.search(old_log),'archived documented original failure')
        self.ids=ids
        return dict(archived_files=len(actual),archive_index_sha256=sha(index),input_sha256=sha(new_input),input_ids=100,old_failed_output_absent=True,fresh_cache_at_start='source-and-runtime-attested, not independently observed in real time')

    def original_records_unchanged(self):
        r=self.record;cfg=read(ROOT/'configs/frozen_config.json')
        protected={}
        for path,digest in self.baseline['sha256'].items():
            p=Path(path)
            if '/results/main/' in path or '/configs/' in path or '/metrics/' in path or p.name in cfg['harness_sha256'] or p.name in ('evaluate_long.py','select_lvspec.py','analyze_results.py','validate_single.py'):
                protected[path]=digest
        independent=unchanged(protected)
        need(r['protected_evidence_before']==r['protected_evidence_after'],'recovery full protected tree snapshot unchanged')
        runtime=unchanged(r['protected_evidence_after'])
        need(r['other_methods_unchanged'] and r['archived_failure_unchanged'] and r['protected_gpu_config_runtime_and_successful_metrics_unchanged'],'runtime protective assertions')
        main=list((ROOT/'results/main').glob('*.jsonl'))
        need(len(main)==10 and sum(len(p.read_text().splitlines()) for p in main)==4200,'all original4200 main rows retained')
        return dict(independent_pre_recovery_protected_files=independent,recovery_protected_files=runtime,main_records=4200,previously_successful_score_seals=78,no_gpu_regeneration=True)

    def new_output_and_log(self):
        r=self.record;output=self.folder/'output.json';log=self.folder/'scorer.log';signature=self.folder/'input.sha256'
        need(sha(output)==r['output_sha256'] and sha(log)==r['recovery_log_sha256'] and sha(signature)==r['signature_sha256'],'actual output/log/cache signature seals')
        need(signature.read_text().strip()==r['input_sha256'],'only identical input can reuse result')
        text=log.read_text(errors='replace');self.scores=output_contract(read(output),self.ids,text)
        need(r['validation']['status']=='pass' and r['validation']['output_ids']==100 and not r['validation']['fatal_or_ignored_parse_log_matches'] and r['validation']['score_values_unmodified'],'full-batch validation, no partial parse')
        return dict(output_sha256=sha(output),log_sha256=sha(log),output_ids=100,undefined_All_f=sum(v is None for v in self.scores.values()),fatal_or_ignored_parse_log_matches=[],cache_persistence_warning_count=text.count('Could not cache item'),cache_caption_length_warning_count=text.count('Caption may be too long'),planned_wrapper_heap_is_not_actual_producer_heap=True)

    def final_scores_and_seals(self):
        folder=ROOT/'metrics/main/coco/lvspec_paper';seal=read(folder/'SCORING_SEAL.json');rows=read(folder/'per_question_scores.json');aggregate=read(folder/'aggregate.json')
        need(seal['aggregate_sha256']==sha(folder/'aggregate.json') and seal['per_question_sha256']==sha(folder/'per_question_scores.json'),'recovered official score seals')
        need(seal['config_sha256']==sha(ROOT/'configs/frozen_config.json') and seal['quality_repeat']==0,'unchanged frozen config and source repeat')
        need(seal['evaluator_sha256']==sha(ROOT/'harness/evaluate_long.py') and seal['original_evaluator_sha256']==EXPECTED_OFFICIAL_SHA,'same actual evaluator source')
        cfg=read(ROOT/'configs/frozen_config.json');samples=read(cfg['data']['coco']['test_path'])
        need(len(rows)==len(samples)==100 and [r['sample_id'] for r in rows]==[s['question_id'] for s in samples],'all original100 questions, aligned')
        need(all(close(row['SPICE'],self.scores[i]) for i,row in enumerate(rows)),'exact official output All.f mapped to every original row')
        point=None if any(s is None for s in self.scores.values()) else statistics.mean(self.scores.values())
        need(close(aggregate['SPICE'],point),'official aggregate mean/undefined convention')
        provenance=aggregate['additional_provenance']['spice']
        need(provenance['input_sha256']==self.record['input_sha256'] and provenance['exact_same_input_result_reused'],'frozen evaluator reused validated exact-input result')
        need(provenance['command']==self.record['original_command'],'frozen wrapper records its nominal16G command, not actual64G run')
        for field,ending in [('spice_jar_sha256','spice-1.0.jar'),('stanford_models_sha256','stanford-corenlp-3.6.0-models.jar'),('wrapper_sha256','spice.py')]:
            matching=[digest for path,digest in self.record['scorer_dependencies'].items() if path.endswith('/'+ending)]
            need(matching==[provenance[field]],'actual and reported official dependency: '+field)
        scoreseals=list((ROOT/'metrics/main').glob('*/*/SCORING_SEAL.json'))
        need(len(scoreseals)==35,'all35 main official scoring arms finished')
        return dict(main_score_seals=35,recovered_method_score_seal_sha256=sha(folder/'SCORING_SEAL.json'),All_f_to_per_question_and_mean_verified=True,nominal_wrapper_command='16G cache-hit plan only',actual_producer_command='64G recorded in resource recovery manifest')

    def failure_scan_scope(self):
        active=sorted(set(ROOT.glob('results/**/*.failure.json'))|set(ROOT.glob('metrics/**/metric_failure.json')))
        need(not active,'active request/scorer failure remains unresolved')
        archived=list((ROOT/'audit/spice_recovery').glob('**/metric_failure.json'))
        need(len(archived)==1,'original scorer failure remains preserved outside active metric tree')
        return dict(active_failure_files=0,preserved_archived_metric_failures=1,original_validator_failure_globs_exclude_audit_archive=True)

    def run(self):
        if self.check('successful_CPU_recovery_manifest_and_only_heap_change',self.manifest) is not None:
            self.check('unchanged_official_JARs_wrapper_models_and_classpath',self.dependencies)
            if self.check('preserved_failure_and_identical100_caption_input',self.archive_and_input) is not None:
                self.check('original4200_GPU_records_runtime_config_successful_metrics_unchanged',self.original_records_unchanged)
                if self.check('complete_new_output_no_ignored_parse_and_sealed_log',self.new_output_and_log) is not None:self.check('same_official_scorer_output_to_final_scores_and_seals',self.final_scores_and_seals)
            self.check('active_failure_scan_excludes_preserved_audit_archive',self.failure_scan_scope)
        need('torch' not in sys.modules,'validator imported torch')
        return dict(status='pass' if all(c['passed'] for c in self.checks) else 'No-Go',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),validator_sha256=sha(__file__),independent_cpu_audit=True,no_gpu_or_java_execution=True,checks=self.checks,limitations=['The audit independently checks preserved files, command differences, logs, IDs and score joins; it does not independently rerun Stanford parsing or certify every extracted scene graph.','Fresh-cache emptiness at start is attested by the sealed recovery implementation and recorded runtime state, not by a separate real-time observer.','The frozen official wrapper reports its nominal16G command on a cache hit; the resource manifest and preserved log/output identify the actual64G producer.','Long descriptions and automatic caption metrics retain their previously reported validity limitations.'])

class Contracts(unittest.TestCase):
    def commands(self):
        a=['java','-XX:ActiveProcessorCount=8','-Xmx16G','-cp','jars','edu.anu.spice.SpiceScorer','input','-cache','cache','-out','output','-subset','-silent','-threads','1'];b=list(a);b[2]='-Xmx64G';return a,b
    def test_heap_only(self):self.assertEqual(len(command_contract(*self.commands())),1)
    def test_other_changes_rejected(self):
        for index,value in [(1,'-XX:ActiveProcessorCount=16'),(-1,'2'),(6,'other_input')]:
            a,b=self.commands();b[index]=value
            with self.assertRaises(AssertionError):command_contract(a,b)
    def payload(self):return [dict(image_id=i,scores={'All':{'f':.5}}) for i in range(100)]
    def test_complete_and_cache_warning_accepted(self):self.assertEqual(len(output_contract(self.payload(),range(100),'Error: Could not cache item\nCaption may be too long')),100)
    def test_fatal_or_ignored_parse_rejected(self):
        for text in ['OutOfMemoryError','Will ignore and continue','NullPointerException','Could not score batched file','Java heap space','Exception in thread main']:
            with self.subTest(text=text),self.assertRaises(AssertionError):output_contract(self.payload(),range(100),text)
    def test_missing_duplicate_output_rejected(self):
        rows=self.payload()
        with self.assertRaises(AssertionError):output_contract(rows[:-1],range(100),'')
        rows[-1]['image_id']=0
        with self.assertRaises(AssertionError):output_contract(rows,range(100),'')
    def test_undefined_f_not_fabricated(self):
        rows=self.payload();rows[0]['scores']['All']['f']=None
        self.assertIsNone(output_contract(rows,range(100),'')[0])
    def test_changed_protected_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'record';p.write_text('original');manifest={str(p):sha(p)};self.assertEqual(unchanged(manifest),1);p.write_text('changed')
            with self.assertRaises(AssertionError):unchanged(manifest)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true');parser.add_argument('--output',type=Path,default=ROOT/'audit/SPICE_RESOURCE_RECOVERY_VALIDATION.json');args=parser.parse_args()
    if args.self_test:
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts));return 0 if result.wasSuccessful() else 1
    result=RecoveryAudit().run();payload=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n';need(not args.output.exists(),'preserve existing resource validation output');args.output.write_text(payload)
    md=['# SPICE资源恢复独立审计','',f'Status: **{result["status"]}**。仅CPU文件审计；不执行Java、GPU推理或重新生成。','','| Check | Result |','|---|---|']
    md += ['| '+c['name']+' | '+('PASS' if c['passed'] else 'No-Go: '+c['exception'].replace('|','/'))+' |' for c in result['checks']]
    md += ['','实际producer采用64G JVM heap；冻结wrapper在精确input的cache hit中仍展示nominal16G计划命令，两者由资源恢复manifest明确区分。原16G失败日志、输入和部分缓存保留在audit归档；失败评分未混入最终结果。','','## 局限','']+['- '+s for s in result['limitations']]
    args.output.with_suffix('.md').write_text('\n'.join(md)+'\n');print(json.dumps(dict(status=result['status'],checks=len(result['checks']),output=str(args.output))));return 0 if result['status']=='pass' else 1

if __name__=='__main__':raise SystemExit(main())
