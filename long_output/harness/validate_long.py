"""Independent CPU audit of the fixed-sample long-output benchmark.

Reuses the SHA-pinned short benchmark's pure row, phase-grid and determinism
contracts. Long-prompt data, configuration, terminal-answer scoring, MAT and
analysis seals are checked separately. Never loads torch, a GPU or a checkpoint.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
BASE_VALIDATOR = BASE/'harness/validate_delivery.py'
BASE_VALIDATOR_SHA = '3132ed9a7b421d2dc4e7043847862b20d0ccaed4eb83fe3c0796f965f9ae7cc7'

def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def need(condition, message):
    if not condition:
        raise AssertionError(message)

need(sha(BASE_VALIDATOR) == BASE_VALIDATOR_SHA, 'original pure validator dependency changed')
spec = importlib.util.spec_from_file_location('_long_base_validator', BASE_VALIDATOR)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.ROOT = ROOT
TASKS, METHODS, close = base.TASKS, base.MAIN_METHODS, base.close

def expected_long_prompt(task, short):
    """Independent reconstruction from the preregistered wording contract."""
    if task=='coco':
        need(short=='Describe this image in one concise sentence.', 'original short caption instruction')
        return ('Please provide a detailed description of the given image in about 100 to 160 words. '
            'Describe visible objects, attributes, actions, quantities, and spatial relations, and distinguish uncertainty from visible facts.')
    suffixes={'mme':'\nPlease answer yes or no.', 'gqa':'\nAnswer the question using a single word or phrase.',
        'textvqa':'\nAnswer the question using a single word or phrase.', 'scienceqa':'\nAnswer with the option letter only.'}
    suffix=suffixes[task]
    need(short.endswith(suffix), 'old prompt has fixed short-format suffix')
    explanation=('Provide a detailed explanation of about 100 to 160 words, using the image and the question to support your answer. '
        'Do not invent details that are not visible or supported by the question. ')
    ending={'mme':'End with exactly one final line in the format "Final answer: yes" or "Final answer: no".',
        'gqa':'End with exactly one final line in the format "Final answer: <answer>", replacing <answer> with a single word or short phrase.',
        'textvqa':'End with exactly one final line in the format "Final answer: <answer>", replacing <answer> with a single word or short phrase.',
        'scienceqa':'End with exactly one final line in the format "Final answer: X", replacing X with the selected option letter.'}[task]
    return short[:-len(suffix)]+'\n\n'+explanation+ending+' Put that final answer on a separate line, with no text after it.'

@contextmanager
def base_root(path):
    before = base.ROOT
    base.ROOT = Path(path)
    try:
        yield
    finally:
        base.ROOT = before

def module_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Called by the inherited complete-grid contract; retain its original checks.
original_row_contract = base.row_contract

def row_contract(row, sample, variant, cfg, phase, shard):
    original_row_contract(row, sample, variant, cfg, phase, shard)
    need(row['prompt_kind'] == cfg.get('prompt_kind', 'task_long'), 'long prompt marker')
    raw, lengths = row['raw_token_ids'], row['acceptance_length']
    accepted, rounds = sum(lengths), len(lengths)
    effective, visible_rounds = row['effective_accepted_draft_tokens'], row['effective_verification_rounds']
    need(row['raw_generated_tokens'] == len(raw) and row['returned_content_tokens'] == len(row['returned_token_ids']), 'raw/visible token counts')
    need(row['raw_accepted_draft_tokens'] == accepted and row['raw_verification_rounds'] == rounds, 'native Draft MAT counts')
    need(row['raw_committed_root_tokens'] == rounds and row['effective_committed_root_tokens'] == visible_rounds, 'separate root accounting')
    need(effective+visible_rounds == row['completion_tokens'], 'effective Draft+root is completion')
    ar = variant['family'] == 'AR'
    expected = dict(mat_native_draft_only=None if ar else accepted/rounds,
        mat_native_with_root=None if ar else len(raw)/rounds,
        mat_effective_draft_only=None if ar else effective/visible_rounds,
        mat_effective_with_root=None if ar else row['completion_tokens']/visible_rounds)
    for key, value in expected.items():
        need(close(row[key], value), 'explicit long MAT: '+key)
    eos = next((i for i,value in enumerate(raw) if value in cfg['eos_token_ids']), None)
    need(row['raw_first_eos_index'] == eos, 'raw EOS retained beyond cap')
    need(row['post_eos_raw_tokens'] == (0 if eos is None else len(raw)-eos-1), 'post-EOS tail count')
    need(row['budget_excluded_raw_tokens'] == max(0,len(raw)-cfg['max_new_tokens']), 'over-budget tail count')
    context = row['context_capacity_audit']
    need(context['status'] == 'pass' and context['prompt_tokens'] == row['prompt_tokens'] and context['max_new_tokens'] == 512, 'context budget check')
    need(context['conservative_maximum_context_tokens'] == context['prompt_tokens']+512+context['conservative_tree_and_stop_headroom'], 'context headroom arithmetic')
    need(context['conservative_maximum_context_tokens'] < context['native_target_KV_capacity'], 'no context truncation or KV overflow')
base.row_contract = row_contract

class LongAudit(base.Audit):
    def sources(self):
        # Same clean official checkout; all old measurements/artifacts read-only.
        with base_root(BASE):
            evidence = super().sources()
        evidence['base_validator_sha256'] = sha(BASE_VALIDATOR)
        evidence['short_configuration_sha256'] = sha(BASE/'configs/frozen_config.json')
        evidence['same_clean_checkout_reused'] = True
        return evidence

    def configuration(self, phase):
        path = ROOT/'configs'/('frozen_config.json' if phase == 'main' else 'design_config.json')
        cfg, old = read(path), read(BASE/'configs/design_config.json')
        for key in ('upstream','upstream_commit','target','draft','model_identity_path','model_identity_sha256',
                    'tree','temperature','seed','system_prompt','processor','max_length','eos_token_ids',
                    'tasks','repetitions','shards','bootstrap','primary_quality_repeat','no_training',
                    'no_new_judge_calls','main_method_templates','development_methods','parity_methods','parity_pairs',
                    'expected_development_requests','expected_main_requests','expected_parity_requests'):
            need(cfg[key] == old[key], 'unchanged model/method/design: '+key)
        need(cfg['max_new_tokens'] == 512 and cfg['max_length'] == 2048, 'long budget512/native round limit2048')
        need(set(cfg['harness_sha256']) == set(old['harness_sha256']) and len(cfg['harness_sha256']) == 16, 'all16 runtime files sealed')
        for name, digest in cfg['harness_sha256'].items():
            need(sha(ROOT/'harness'/name) == digest, 'long runtime seal: '+name)
            if name != 'run_benchmark.py':
                need(digest == old['harness_sha256'][name] == sha(BASE/'harness'/name), '15 unchanged algorithm/helper sources: '+name)
        for name, digest in cfg['test_sha256'].items():
            need(sha(ROOT/name) == digest, 'test source SHA: '+name)
        need(sha(ROOT/'EXPERIMENT_PROTOCOL.md') == cfg['protocol_sha256'], 'long protocol frozen')
        need(sha(ROOT/'harness/prepare_experiment.py') == cfg['preparation_sha256'], 'long preparation source frozen')
        need(sha(ROOT/'audit/CLEAN_CHECKOUT.json') == cfg['checkout_manifest_sha256'], 'clean source manifest copied')
        need(sha(cfg['model_identity_path']) == cfg['model_identity_sha256'], 'checkpoint identity manifest')
        model = read(cfg['model_identity_path'])
        need(sha(model['original_manifest']) == model['original_sha256'], 'original model SHA manifest')
        for item in model['files']:
            stat, expected = Path(item['path']).stat(), item['stat_after']
            need((stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,stat.st_ino,stat.st_dev) ==
                 (expected['size_bytes'],expected['mtime_ns'],expected['ctime_ns'],expected['inode'],expected['device']), 'unchanged checkpoint file: '+item['path'])
        freeze_path = ROOT/'audit/METRIC_PARSER_FREEZE.json'
        freeze = read(freeze_path)
        need(freeze['evaluator_sha256'] == sha(ROOT/'harness/evaluate_long.py'), 'long parser frozen source')
        need(freeze['original_evaluator_sha256'] == sha(BASE/'harness/evaluate_tasks.py'), 'official evaluator remains sealed')
        need(freeze['frozen_before_any_long_official_score'] is True and freeze['parser']=='terminal_final_answer_v1'
             and freeze['no_gold_assisted_extraction'] is True, 'parser freeze before quality')
        need(freeze['quality_protocol_sha256']==sha(ROOT/'reports/QUALITY_PROTOCOL.md')==cfg['quality_protocol_sha256'], 'quality protocol seal')
        if phase == 'main':
            need(cfg['config_finalized_before_generation'] is True, 'main config frozen')
            need(cfg['design_config_sha256'] == sha(ROOT/'configs/design_config.json') and cfg['selection_sha256'] == sha(ROOT/'configs/lvspec_selection.json'), 'unchanged design and dev selection')
            ready = read(ROOT/'audit/MAIN_READY.json')
            need(ready['status'] == 'pass' and ready['config_sha256'] == sha(path) and ready['development_only_selection'] and not ready['test_outputs_inspected'], 'main ready before test inspection')
        self.configs[phase] = cfg
        return dict(config_path=str(path),sha256=sha(path),runtime_sources=16,unchanged_algorithm_helpers=15,
                    model_files=len(model['files']),max_new_tokens=512,model_verification='prior SHA plus current stat identity')

    def data(self):
        # Fully verify the original data/history audit, then exact long/short join.
        original = base.Audit(self.verify_history_bytes)
        original.configs['development'] = read(BASE/'configs/design_config.json')
        with base_root(BASE):
            old_evidence = original.data()
        cfg = next(iter(self.configs.values()))
        provenance_path=ROOT/'audit/LONG_INPUT_PROVENANCE.json'
        provenance=read(provenance_path)
        need(sha(provenance_path)==cfg['long_input_provenance_sha256'] and not provenance['resampled']
             and not provenance['generated_new_gold'] and provenance['all_image_files_reused_readonly'], 'long input transformation seal/no resampling')
        need(provenance['parent_config_sha256']==sha(BASE/'configs/frozen_config.json'), 'original short population seal')
        statistics_out = {}
        for task in TASKS:
            statistics_out[task] = {}
            for split, groups in [('dev',20),('test',100)]:
                entry, old_entry = cfg['data'][task], original.configs['development']['data'][task]
                path = Path(entry[split+'_path'])
                need(path == ROOT/f'data/{task}/{split}.json' and sha(path) == entry[split+'_sha256'], 'long split path/SHA')
                need(Path(entry[split+'_labels_path']) == Path(old_entry[split+'_labels_path']) and entry[split+'_labels_sha256'] == old_entry[split+'_labels_sha256'], 'reuse unchanged original separate labels')
                samples, previous = read(path), original.samples[task,split]
                source_seal=provenance['inputs'][task][split]
                need(source_seal['sha256']==sha(path) and source_seal['parent_sha256']==sha(old_entry[split+'_path'])
                     and source_seal['only_changed_fields']==['prompt_text','prompt_kind'], 'exact authorized transformation fields')
                need(len(samples) == len(previous), 'same sample count without resampling')
                for row, before in zip(samples,previous):
                    allowed = {'prompt_text','prompt_kind','source_short_prompt_text'}
                    need({k:v for k,v in row.items() if k not in allowed} == {k:v for k,v in before.items() if k not in allowed}, 'all original identities and metadata unchanged')
                    need(not (set(row)&base.FORBIDDEN_RUNTIME_LABELS), 'runtime has no labels/solutions/references')
                    prompt = row['prompt_text']
                    need(prompt==expected_long_prompt(task,before['prompt_text']), 'original question/hint/options unchanged with exact long instruction')
                    need(prompt != before['prompt_text'] and '100 to 160 words' in prompt, 'actual long prompt')
                    if task == 'coco':
                        need(prompt == 'Please provide a detailed description of the given image in about 100 to 160 words. Describe visible objects, attributes, actions, quantities, and spatial relations, and distinguish uncertainty from visible facts.', 'frozen detailed COCO prompt')
                    else:
                        need('Final answer:' in prompt and 'End with exactly one final line' in prompt, 'terminal-answer prompt grammar')
                        need('Provide a detailed explanation' in prompt, 'explanation before final answer')
                self.samples[task,split] = samples
                statistics_out[task][split] = dict(questions=len(samples),groups=groups,exact_same_original_identity=True)
        self.freshness_findings = original.freshness_findings
        self.warnings += original.warnings
        self.warnings.append('All long-output images are reused from the declared short-output benchmark; no new independent-image generalization claim.')
        return dict(task_splits=statistics_out,original_data_audit=old_evidence,original_and_long_dev_test_images_identical=True,
                    all_long_questions=720,all_original_global_RGB_groups=600,resampled=False)

    def loaded(self, phase):
        cfg = self.configs[phase]
        cfg_path = ROOT/'configs'/('frozen_config.json' if phase == 'main' else 'design_config.json')
        result = []
        for shard in (0,1):
            meta = read(ROOT/f'audit/{phase}_shard{shard}.loaded.json')
            need(meta['config_sha256'] == sha(cfg_path) and meta['runner_sha256'] == cfg['harness_sha256']['run_benchmark.py'], 'actually loaded long config/runner')
            need(meta['source_sha256'] == cfg['harness_sha256'] and meta['processor_settings'] == cfg['processor'], 'loaded runtime/processor')
            need(meta['physical_gpu'] == str(shard) and meta['AR_memory_includes_idle_loaded_Draft'], 'physical GPU and AR memory disclosure')
            for name,item in meta['loaded_vispec_sources'].items():
                need(Path(item['path']).resolve().is_relative_to(Path(cfg['upstream']).resolve()), 'fresh official actual import: '+name)
                need(sha(item['path']) == item['sha256'], 'actual imported source SHA: '+name)
            resource = read(ROOT/f'audit/{phase}_shard{shard}.gpu_resource.json')
            need(resource['physical_gpu'] == str(shard) and resource['checks'] > 0 and not resource['events'], 'exclusive GPU resource log')
            result.append(dict(shard=shard,imports=len(meta['loaded_vispec_sources']),resource_checks=resource['checks']))
        return result

    def metrics(self, phase):
        scorer = module_file(ROOT/'harness/evaluate_long.py','_long_scoring_audit')
        cfg = self.configs[phase]
        cfg_path = ROOT/'configs'/('frozen_config.json' if phase == 'main' else 'design_config.json')
        results = {}
        for task in TASKS:
            results[task] = {}
            samples = self.samples[task,'test' if phase == 'main' else 'dev']
            labeled,_ = scorer.attach_labels(task,samples)
            methods = cfg['methods_by_task'][task] if phase == 'main' else cfg['development_methods']
            for method in methods:
                folder = ROOT/f'metrics/{phase}/{task}/{method}'
                seal,rows,aggregate = [read(folder/name) for name in ('SCORING_SEAL.json','per_question_scores.json','aggregate.json')]
                need(seal['config_sha256'] == sha(cfg_path) and seal['inputs_sha256'] == cfg['data'][task][('test' if phase=='main' else 'dev')+'_sha256'], 'score exact long config/input')
                need(seal['quality_repeat'] == 0 and seal['evaluator_sha256'] == sha(ROOT/'harness/evaluate_long.py'), 'fixed repeat0/long scorer')
                need(seal['original_evaluator_sha256'] == scorer.OFFICIAL_SHA256 == sha(BASE/'harness/evaluate_tasks.py'), 'unchanged official evaluator')
                freeze_path = ROOT/'audit/METRIC_PARSER_FREEZE.json'
                need(seal['metric_parser_freeze_sha256'] == sha(freeze_path) and (folder/'SCORING_SEAL.json').stat().st_mtime_ns >= freeze_path.stat().st_mtime_ns, 'parser frozen before scoring')
                need(seal['aggregate_sha256'] == sha(folder/'aggregate.json') and seal['per_question_sha256'] == sha(folder/'per_question_scores.json'), 'immutable scores')
                need(seal['result_sha256'] == {p:v for p,v in self.hashes[phase].items() if f'_{task}_shard' in p}, 'exact actual runtime shards scored')
                selected = sorted([r for r in self.phase_rows[phase][task] if r['method']==method and r['repeat']==0],key=lambda r:r['ordinal'])
                predictions = {str(r['question_id']):r['output_text'] for r in selected}
                need(seal['predictions_sha256'] == hashlib.sha256(json.dumps(predictions,sort_keys=True,ensure_ascii=False).encode()).hexdigest(), 'raw long prediction identity')
                need(len(rows)==len(samples) and [r['sample_id'] for r in rows]==[r['question_id'] for r in samples], 'all original questions retained')
                extracts = [scorer.extract_final_answer(task,predictions[r['question_id']]) for r in samples]
                need(all(r['prediction']==predictions[r['sample_id']] and r['raw_output_text']==predictions[r['sample_id']] for r in rows), 'full raw explanation retained')
                need(aggregate['long_answer_adapter_sha256']==sha(ROOT/'harness/evaluate_long.py') and aggregate['all_questions_retained'] and not aggregate['automatic_judge_used'], 'no judge / full denominator')
                need(aggregate['parse_invalid_count']==sum(r['parse_invalid'] for r in rows) and aggregate['uncertain_count']==sum(r['uncertain'] for r in rows), 'all invalid/uncertain retained')
                need(aggregate['answer_extraction_failures']==sum(e['status'] in ('invalid','uncertain') for e in extracts), 'format failures explicitly counted')
                official = scorer.official()
                if task != 'coco':
                    fun = {'mme':official._score_mme,'gqa':official._score_gqa,'textvqa':official._score_textvqa,'scienceqa':official._score_scienceqa}[task]
                    again_rows,again = fun(labeled,[e['answer'] for e in extracts])
                    for row,entry,check in zip(rows,extracts,again_rows):
                        need(row['extracted_answer']==entry['answer'] and row['answer_extraction_status']==entry['status'] and row['answer_extraction_reason']==entry['reason'], 'frozen terminal parser replay')
                        need(close(row['score'],check['score']), 'official extracted-answer score replay')
                        need(entry['status'] not in ('invalid','uncertain') or row['score']==0, 'failed format contributes zero')
                    for key in ('accuracy','MME_subset_score','Perception_subset_score','Cognition_subset_score','acc_plus'):
                        if key in again: need(close(aggregate[key],again[key]), 'official task aggregate replay: '+key)
                else:
                    need(aggregate['CIDEr_document_frequency_images']==len(samples), 'COCO current split DF')
                    need(close(aggregate['CIDEr'],statistics.mean(r['CIDEr'] for r in rows)), 'CIDEr mean')
                    spice = None if any(r['SPICE'] is None for r in rows) else statistics.mean(r['SPICE'] for r in rows)
                    need(close(aggregate['SPICE'],spice), 'SPICE undefined/mean preserved')
                    need(close(aggregate['CHAIRs'],statistics.mean(r['CHAIRs'] for r in rows)), 'CHAIRs corpus ratio')
                    objects,hallucinations = sum(r['object_mentions'] for r in rows),sum(r['hallucinated_mentions'] for r in rows)
                    need(close(aggregate['CHAIRi'],hallucinations/objects if objects else None), 'CHAIRi mention ratio')
                    need((folder/'official_scoring/spice/scorer.log').exists() and (folder/'official_scoring/chair/author.log').exists(), 'real official SPICE/CHAIR process evidence')
                    need(aggregate['additional_provenance']['spice']['input_sha256']==sha(folder/'official_scoring/spice/input.json'), 'full long description SPICE inputs')
                results[task][method] = dict(rows=rows,aggregate=aggregate,seal_sha256=sha(folder/'SCORING_SEAL.json'))
        self.metric_scores[phase] = results
        return results

    def selection(self,scores):
        cfg = self.configs['development']
        selection,frozen = read(ROOT/'configs/lvspec_selection.json'),read(ROOT/'configs/frozen_config.json')
        need(selection['status']=='development_only_selection' and selection['ready_for_main'] and not selection['main_outputs_used'] and not selection['repeat_differences'], 'dev-only selection/determinism')
        for key,path in [('design_config_sha256',ROOT/'configs/design_config.json'),('selector_sha256',ROOT/'harness/select_lvspec.py'),('evaluator_sha256',ROOT/'harness/evaluate_long.py'),('metric_parser_freeze_sha256',ROOT/'audit/METRIC_PARSER_FREEZE.json')]:
            need(selection[key]==sha(path), 'selector source seal: '+key)
        need(selection['result_sha256']==self.hashes['development'], 'selection includes exactly frozen development grid')
        decisions = {}
        for task in TASKS:
            baseline = scores[task]['vispec']['aggregate']
            candidates = []
            for method,variant in cfg['development_methods'].items():
                if variant['family']!='LVSPEC':continue
                current = scores[task][method]['aggregate']
                if task=='coco':
                    # Long paragraphs have a known CIDEr-D short-reference
                    # Gaussian length floor. Its auxiliary-only role is fixed
                    # before the first long score, not selected from outcomes.
                    ok = all(current.get(k) is not None and baseline.get(k) is not None for k in ('SPICE','CHAIRs'))
                    ok = ok and current['SPICE']+1e-12>=.95*baseline['SPICE'] and current['CHAIRs']<=baseline['CHAIRs']+1e-12
                elif task=='mme':ok = all(current[k]+1e-12>=baseline[k] for k in ('MME_subset_score','accuracy'))
                else:ok = current['accuracy']+1e-12>=baseline['accuracy']
                times = [r['seconds'] for r in self.phase_rows['development'][task] if r['method']==method]
                candidates.append(dict(method=method,ratio=variant['loose_ratio'],passed=bool(ok),mean=statistics.mean(times)))
            need(sorted(r['ratio'] for r in candidates)==[0.,.3,.5,.7], 'four fixed LV ratios')
            paper = next(r for r in candidates if r['ratio']==.7)
            eligible = [r for r in candidates if r['ratio']<.7 and r['passed']]
            chosen = paper if paper['passed'] else min(eligible,key=lambda r:(r['mean'],r['ratio'])) if eligible else None
            actual = selection['selections'][task]
            need(actual['paper_qualitypass']==paper['passed'] and actual['strict_fallback']==(chosen is None), 'paper-first selection and explicit fallback')
            need(actual['selecteddevmethod']==(chosen['method'] if chosen else 'vispec') and actual['qualitypass']==bool(chosen), 'independent dev quality-constrained selection')
            need(actual['parameters']['disable_relaxation']==(chosen is None) and actual['parameters']['loose_ratio']==(chosen['ratio'] if chosen else .7), 'lambda0/PST not mistaken for Strict')
            for candidate in actual['candidates']:
                own = next(r for r in candidates if r['method']==candidate['method'])
                need(candidate['qualitypass']==own['passed'] and close(candidate['mean_seconds'],own['mean']), 'all candidate quality/latency retained')
                need(candidate['score_seal_sha256']==scores[task][candidate['method']]['seal_sha256'], 'candidate score seal')
            expected = dict(cfg['main_method_templates']['lvspec_adapted'],**actual['parameters'])
            need(frozen['methods_by_task'][task]['lvspec_adapted']==expected, 'main LV matches development-only freeze')
            for method in METHODS-{'lvspec_adapted'}:
                need(frozen['methods_by_task'][task][method]==cfg['main_method_templates'][method], 'no other main method adjustment')
            decisions[task] = dict(selected=actual['selecteddevmethod'],strict_fallback=actual['strict_fallback'],independently_recomputed=True)
        return decisions

    def analysis(self):
        import numpy as np
        report = read(ROOT/'audit/STATISTICAL_ANALYSIS.json')
        cfg = self.configs['main']
        for key,path in [('config_sha256',ROOT/'configs/frozen_config.json'),('analyzer_sha256',ROOT/'harness/analyze_results.py'),('selector_sha256',ROOT/'harness/select_lvspec.py'),('evaluator_sha256',ROOT/'harness/evaluate_long.py'),('selection_sha256',ROOT/'configs/lvspec_selection.json'),('metric_parser_freeze_sha256',ROOT/'audit/METRIC_PARSER_FREEZE.json')]:
            need(report[key]==sha(path), 'analysis source/config seal: '+key)
        need(report['result_sha256']==self.hashes['main'] and report['measured_requests']==12600 and report['bootstrap']==5000, 'full12600 main records/bootstrap5000')
        need(not report['repeat_differences'] and report['MME_stratified_by_category'] and report['immutable_runtime_sources']==cfg['harness_sha256'], 'determinism/strata/runtime seal')
        need(report['prespecified_comparisons']==cfg['comparisons'] and report['prespecified_secondary_comparisons']==cfg['secondary_comparisons'], 'predeclared comparison families')
        need(set(report['draws'])==set(TASKS), 'all five bootstrap task populations')
        for task,drawing in report['draws'].items():
            samples = self.samples[task,'test']
            groups = list(dict.fromkeys(r['group_id'] for r in samples))
            categories = [next(r.get('category','all') for r in samples if r['group_id']==group) for group in groups]
            need(drawing['shape']==[5000,100] and drawing['groups']==groups and drawing['categories']==categories, 'actual paired image groups in draws')
            need(drawing['stratified']==(task=='mme') and drawing['stratum_counts']==dict(Counter(categories)), 'exact MME complete-pair strata')
            seed = cfg['bootstrap_seed']+TASKS.index(task)*100003
            need(drawing['seed']==seed, 'frozen bootstrap task seed')
            rng = np.random.default_rng(seed)
            if task=='mme':
                blocks=[]
                for category in sorted(set(categories)):
                    indices=np.asarray([i for i,value in enumerate(categories) if value==category],dtype=np.int64)
                    blocks.append(indices[rng.integers(0,len(indices),size=(5000,len(indices)))])
                draws=np.concatenate(blocks,axis=1)
            else:draws=rng.integers(0,100,size=(5000,100),dtype=np.int64)
            need(hashlib.sha256(draws.tobytes()).hexdigest()==drawing['draw_sha256'], 'independent grouped bootstrap draws')
        for name,digest in report['output_sha256'].items():
            path=Path(name) if Path(name).is_absolute() else ROOT/name
            need(path.resolve().is_relative_to(ROOT.resolve()) and sha(path)==digest, 'immutable actual report/table/figure: '+name)
        with (ROOT/'reports/BENCHMARK_SUMMARY.csv').open(newline='') as f:rows=list(csv.DictReader(f))
        summary={(r['task'],r['method']):r for r in rows}
        need(len(rows)==len(summary)==35 and set(summary)=={(t,m) for t in TASKS for m in METHODS}, 'all35 task/method results')
        def number(value):return None if value in ('',None) else float(value)
        for (task,method),row in summary.items():
            records=[r for r in self.phase_rows['main'][task] if r['method']==method]
            aggregate=self.metric_scores['main'][task][method]['aggregate']
            times=[r['seconds'] for r in records]
            decode=[r['decode_seconds'] for r in records]
            points=dict(mean_seconds=statistics.mean(times),mean_decode_seconds=statistics.mean(decode),
                throughput_tokens_per_second=sum(len(r['returned_token_ids']) for r in records)/sum(times),
                decode_throughput_tokens_per_second=sum(len(r['returned_token_ids']) for r in records)/sum(decode),
                completion_tokens_including_EOS_throughput_per_second=sum(r['completion_tokens'] for r in records)/sum(times),
                mean_peak_allocated_bytes=statistics.mean(r['peak_allocated_bytes'] for r in records),
                mean_visible_tokens=statistics.mean(len(r['returned_token_ids']) for r in records),
                mean_raw_tokens=statistics.mean(len(r['raw_token_ids']) for r in records),
                truncation_fraction=statistics.mean(r['cap_reached'] for r in records),
                effective_mean_accepted_draft_length=None if method=='ar' else sum(r['effective_accepted_draft_tokens'] for r in records)/sum(r['effective_verification_rounds'] for r in records))
            points['native_mean_accepted_draft_length']=None if method=='ar' else sum(r['raw_accepted_draft_tokens'] for r in records)/sum(r['raw_verification_rounds'] for r in records)
            points['effective_mean_committed_tokens_per_round']=None if method=='ar' else sum(r['completion_tokens'] for r in records)/sum(r['effective_verification_rounds'] for r in records)
            points['mat_native_draft_only']=points['native_mean_accepted_draft_length']
            points['mat_effective_draft_only']=points['effective_mean_accepted_draft_length']
            points['mat_effective_with_root']=points['effective_mean_committed_tokens_per_round']
            points['mean_output_words']=statistics.mean(len(r['output_text'].split()) for r in records)
            need(int(row['requests'])==len(records) and int(row['images_or_groups'])==100 and int(row['quality_repeat'])==0, 'summary full paired population')
            for key,value in points.items():need(close(number(row[key]),value), 'independent throughput/MAT/cost point: '+key)
            for key in ('accuracy','MME_subset_score','Perception_subset_score','Cognition_subset_score','acc_plus','CIDEr','SPICE','CHAIRs','CHAIRi'):
                if key in aggregate:need(close(number(row[key]),aggregate[key]), 'official quality point: '+key)
            need(int(row['parse_invalid_count'])==aggregate['parse_invalid_count'] and int(row['parse_uncertain_count'])==aggregate['uncertain_count'], 'summary failures retain denominator')
            need(row['scoring_seal_sha256']==self.metric_scores['main'][task][method]['seal_sha256'], 'summary score mapping')
        with (ROOT/'reports/PAIRED_COMPARISONS.csv').open(newline='') as f:comparisons=list(csv.DictReader(f))
        family=defaultdict(list)
        keys=[]
        for row in comparisons:
            key=tuple(row[k] for k in ('task','method','baseline','metric','family'));keys.append(key)
            family[row['task'],row['metric'],row['family']].append(row)
            a=number(summary[row['task'],row['method']][row['metric']]);b=number(summary[row['task'],row['baseline']][row['metric']])
            need(close(number(row['method_point']),a) and close(number(row['baseline_point']),b), 'comparison correct signed mapping')
            if row['status']=='ok':need(a is not None and b is not None and close(number(row['difference']),a-b) and 0<=number(row['p'])<=1 and number(row['ci95_low'])<=number(row['ci95_high']), 'difference/CI/p finite and ordered')
            else:need(row['status']=='unavailable' and number(row['p']) is None and number(row['difference']) is None, 'unavailable preserved')
        need(len(keys)==len(set(keys)), 'no duplicate selective comparisons')
        allowed={'BEPC_six_prespecified':cfg['comparisons'],'LV_adaptation_two_prespecified':cfg['secondary_comparisons']}
        for (task,metric,name),batch in family.items():
            need(name in allowed and len(batch)==len(allowed[name]) and {(r['method'],r['baseline']) for r in batch}=={tuple(pair) for pair in allowed[name]}, 'complete prespecified family per metric')
            ordered=sorted(batch,key=lambda r:1. if number(r['p']) is None else number(r['p']))
            running=0.
            for rank,row in enumerate(ordered):
                p=1. if number(row['p']) is None else number(row['p']);running=max(running,min(1.,(len(ordered)-rank)*p))
                need(close(number(row['holm_adjusted_p']),None if number(row['p']) is None else running), 'independent Holm correction')
        for task in TASKS:
            for metric in ('mean_seconds','throughput_tokens_per_second','effective_mean_accepted_draft_length'):
                for name in allowed:need((task,metric,name) in family, 'required speed/MAT families not omitted')
        return dict(summary_rows=len(summary),comparison_rows=len(comparisons),bootstrap_draws_independently_verified=True,
                    throughput_and_MAT_points_independently_recomputed=True,Holm_independently_recomputed=True,
                    every_interval_independently_reimplemented=False,artifact_sha256=sha(ROOT/'audit/STATISTICAL_ANALYSIS.json'))

    def run(self,phase):
        result=super().run(phase)
        result['validator_sha256']=sha(__file__)
        result['base_validator_dependency_sha256']=sha(BASE_VALIDATOR)
        result['long_output_extension']=True
        result['limitations'] += ['All images reuse the declared short-output cohort; long/short changes prompt AND budget, not a pure prompt-only causal experiment.',
            'Terminal QA accuracy does not evaluate explanation truthfulness; full long-description CIDEr/SPICE/CHAIR do not constitute human factual Ground Truth.',
            'AR/native Strict token divergence is separately reported, not silently assumed lossless.']
        need('torch' not in sys.modules, 'validator unexpectedly loaded torch')
        return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['preflight','parity','development','main','all'],default='all')
    parser.add_argument('--output',type=Path,default=ROOT/'audit/FINAL_DELIVERY_VALIDATION.json')
    parser.add_argument('--verify-history-bytes',action='store_true')
    args=parser.parse_args()
    result=LongAudit(args.verify_history_bytes).run(args.phase)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if args.output.exists():need(args.output.read_text()==payload,'preserve existing audit; select a new output path for a different phase')
    else:args.output.write_text(payload,encoding='utf-8')
    markdown=['# 长输出独立交付审计','',f"Status: **{result['status']}**；phase: `{args.phase}`。CPU-only，无 GPU、Judge 或训练调用。",'',
        '| Check | Result |','|---|---|']
    for item in result['checks']:
        markdown.append('| '+item['name']+' | '+('PASS' if item['passed'] else ('No-Go: '+item['exception'].replace('|','/')))+' |')
    markdown += ['','逐请求 MAT/停止/模型调用、完整网格、输入与gold封印、确定性、评分和统计证据见相邻JSON。历史未见图像主张单列，不等同执行失败。',
        '','## 局限','']+['- '+s for s in result['limitations']]
    args.output.with_suffix('.md').write_text('\n'.join(markdown)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],checks=len(result['checks']),required_failures=sum(not c['passed'] and c['required_for_execution_integrity'] for c in result['checks']),output=str(args.output))))
    return 0 if result['status']=='pass' else 1

if __name__=='__main__':raise SystemExit(main())
