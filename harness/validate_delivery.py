"""Independent CPU evidence audit of the fresh five-task decoding benchmark.

This program never imports torch, loads a checkpoint, generates answers, calls
an external judge, changes a score, or removes failed requests. Run after the
requested phase has completed. A missing or inconsistent evidence item yields
No-Go, with all findings retained. --self-test exercises the pure record checks.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
TASKS = ('mme', 'gqa', 'textvqa', 'scienceqa', 'coco')
MAIN_METHODS = {'ar', 'vispec', 'strict_shared', 'bepc_algorithm', 'margin4_ccve',
                'lvspec_paper', 'lvspec_adapted'}
REPEAT_FIELDS = ('raw_token_ids', 'returned_token_ids', 'output_text',
                 'acceptance_length', 'acceptance_paths', 'eos_index', 'cap_reached')
PARITY_FIELDS = ('raw_token_ids', 'returned_token_ids', 'acceptance_length',
                 'eos_index', 'cap_reached')
FORBIDDEN_RUNTIME_LABELS = {'answer', 'answers', 'answer_index', 'answer_letter',
                            'gold', 'gold_answer', 'gt_answers', 'label', 'labels',
                            'lecture', 'solution', 'references', 'sentences', 'captions'}
RELAXED_BUDGET_METHODS = {'bepc_algorithm', 'bepc_reference', 'margin4_ccve'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def close(left, right, tol=1e-10):
    return (left is None and right is None) or (left is not None and right is not None
        and math.isfinite(float(left)) and math.isfinite(float(right))
        and abs(float(left)-float(right)) <= tol*max(1., abs(float(left)), abs(float(right))))


def need(condition, message):
    if not condition:
        raise AssertionError(message)


def canonical(value):
    value = str(value).strip()
    return str(int(value)) if re.fullmatch(r'\d+', value) else value.lower()


def historical_identity(value):
    """Resolve recorded task namespace aliases for historical overlap audit."""
    value = str(value).strip()
    value = re.sub(r'^(?:mme|gqa|textvqa|scienceqa|sqa|coco|coco_caption):', '', value, flags=re.I)
    return canonical(value)


def load_local(name):
    spec = importlib.util.spec_from_file_location('_delivery_' + name, ROOT/'harness'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row_contract(row, sample, variant, cfg, phase, shard):
    """Independent token/stop/call/path/budget arithmetic, no model imports."""
    need(row['phase'] == phase and row['task'] == sample['task'], 'row phase/task')
    need(row['temperature'] == 0., 'temperature must be zero')
    for key in ('question_id', 'image_id', 'image_path', 'image_sha256', 'group_id',
                'group_ordinal', 'ordinal', 'prompt_text'):
        need(row[key] == sample[key], f'input identity: {key}')
    need(row['shard'] == shard == sample['group_ordinal'] % 2, 'group/shard assignment')
    need(str(row['physical_gpu']) == str(shard) == str(row['gpu']), 'fixed physical GPU')
    need(bool(row['diagnostic_recording']) == (phase == 'parity'), 'audit mode / timing')
    need(row['prompt_tokens'] == len(row['prompt_token_ids']) > 0, 'prompt token length')
    grid = row['image_grid_thw']
    need(len(grid) == 1 and len(grid[0]) == 3 and grid[0][0] == 1
         and all(isinstance(x, int) and x > 0 for x in grid[0]), 'single-image visual grid')
    need(len(row['input_tensor_sha256']) == 64, 'processed input tensor identity')
    raw, lengths = row['raw_token_ids'], row['acceptance_length']
    need(raw and lengths and all(isinstance(x, int) and x >= 0 for x in raw), 'raw token IDs')
    need(all(isinstance(x, int) and x >= 0 for x in lengths), 'nonnegative native accept lengths')
    need(sum(lengths)+len(lengths) == len(raw), 'native output / accept-block accounting')
    need(row['native_new_tokens'] == len(raw), 'native generated token count')
    need(row['native_index'] == len(lengths)-1, 'native loop index')
    eos_ids = set(cfg['eos_token_ids'])
    budget = raw[:cfg['max_new_tokens']]
    eos = next((i for i, value in enumerate(budget) if value in eos_ids), None)
    returned = budget[:eos] if eos is not None else budget
    completion = len(returned)+int(eos is not None)
    need(row['returned_token_ids'] == returned and row['eos_index'] == eos, 'first EOS /128 cap')
    need(row['cap_reached'] == (eos is None and len(budget) == cfg['max_new_tokens']), 'truncation status')
    need(row['completion_tokens'] == completion, 'effective completion length')
    effective = rounds = committed = 0
    for length in lengths:
        if committed >= completion:
            break
        rounds += 1
        effective += max(0, min(length, completion-committed-1))
        committed += length+1
    need(row['effective_accepted_draft_tokens'] == effective
         and row['effective_verification_rounds'] == rounds, 'effective acceptance after EOS/cap')
    calls, stats = row['model_calls'], row['signal_stats']
    need(calls.get('vision', 0) == 1, 'one visual encoder call')
    need(row['extra_target_forwards'] == 0, 'no extra Target forward')
    need(stats.get('extra_target_forwards', 0) == stats.get('additional_target_forwards', 0) == 0,
         'no hidden additional Target forward')
    if variant['family'] == 'AR':
        need(not any(lengths) and calls.get('target', 0) == len(raw)
             and calls.get('draft', 0) == 0, 'AR call accounting')
    else:
        need(calls.get('target', 0) == len(lengths)+1, 'prefill plus native Target rounds')
        terminal = stats.get('terminal_update', {})
        skipped = terminal.get('terminal_draft_skips', 0)
        need(skipped in (0, 1), 'at most one unused final Draft tree')
        if variant.get('terminal_skip', False):
            need(terminal.get('update_calls') == len(lengths), 'native KV updates retained')
            need(terminal.get('original_update_KV_and_sample_p_preserved') is True, 'native KV/sample_p unchanged')
            need(terminal.get('additional_target_forwards') == 0, 'terminal adapter has no forward')
            native_terminal = any(value in eos_ids for value in raw) or len(raw) > cfg['max_new_tokens']
            need(skipped == int(native_terminal), 'skip exactly the unused tree on native terminal')
            need(not skipped or terminal.get('eos_terminal_skips', 0)
                 or terminal.get('budget_terminal_skips', 0), 'record skip reason')
        else:
            need(skipped == 0 and terminal.get('enabled') is False, 'unmodified final native Draft tree')
        expected_draft = (len(lengths)+1-skipped)*(cfg['tree']['depth']+1)
        need(calls.get('draft', 0) == expected_draft, 'native depth+1 Draft forwards per constructed tree')
    lifecycle = row['request_lifecycle']
    need(lifecycle.get('immediate_release_checked') is True
         and lifecycle.get('live_controllers_after_request') == 0
         and lifecycle.get('live_logits_after_request') == 0
         and lifecycle.get('explicit_gc_or_empty_cache') is False, 'request ownership / no forced GC')
    for key in ('seconds', 'decode_seconds', 'baseline_allocated_bytes', 'baseline_reserved_bytes',
                'peak_allocated_bytes', 'peak_reserved_bytes', 'peak_increment_bytes',
                'peak_reserved_increment_bytes'):
        need(isinstance(row[key], (int, float)) and math.isfinite(row[key]) and row[key] >= 0,
             f'finite nonnegative cost: {key}')
    need(row['seconds'] > 0 and row['decode_seconds'] <= row['seconds']+1e-6, 'complete request contains decode')
    need(row['peak_allocated_bytes']-row['baseline_allocated_bytes'] == row['peak_increment_bytes'], 'allocated peak delta')
    need(row['peak_reserved_bytes']-row['baseline_reserved_bytes'] == row['peak_reserved_increment_bytes'], 'reserved peak delta')
    need(row['preprocessing_reused_for_paired_requests'] is True, 'preprocessing cost disclosure')
    need(row['input_preprocessing_seconds'] is not None and row['input_preprocessing_seconds'] >= 0,
         'separately recorded preprocessing')
    if variant.get('compact_logits', False) and variant['family'] != 'AR':
        compact = stats['compact_logits']
        need(compact.get('native_argmax_reductions') == len(lengths), 'one compact argmax per tree')
        need(compact.get('audit_reference_materializations') == (len(lengths) if phase == 'parity' else 0),
             'production never expands audit logits')
    if row['acceptance_paths']:
        paths = row['acceptance_paths']
        need(len(paths) == len(lengths), 'one actually selected path per native round')
        reconstructed, response = [], 0
        for index, (path, length) in enumerate(zip(paths, lengths)):
            need(path['round'] == index and path['accepted_draft'] == length, 'selected path round/length')
            need(len(path['tokens']) == len(path['nodes']) == length+1, 'selected real tree path length')
            need(path['nodes'][0] == 0 and len(set(path['nodes'])) == len(path['nodes'])
                 and min(path['nodes']) >= 0, 'rooted nonpadding path')
            need(path['response_mismatches_before'] == response, 'only committed path spends budget')
            need(0 <= path['mismatches'] <= length, 'mismatch count is candidate-specific')
            if row['method'] in RELAXED_BUDGET_METHODS:
                need(path['mismatches'] <= 2 and response+path['mismatches'] <= 4, 'BEPC/ME2/4 budget')
            response += path['mismatches']
            reconstructed.extend(path['tokens'])
        need(reconstructed == raw, 'submitted tree path reconstructs complete raw output')
        need(stats['committed_response_mismatches'] == response, 'committed mismatch total')
    if row['method'] in RELAXED_BUDGET_METHODS:
        need(stats['round_mismatch_budget'] == 2 and stats['response_mismatch_budget'] == 4,
             'BEPC/ME unchanged mismatch budgets')
    if variant['family'] == 'LVSPEC':
        need(stats['round_mismatch_budget'] is None and stats['response_mismatch_budget'] is None,
             'LVSpec has no invented2/4budget')
        need(stats['original_lvspec_mismatch_budget'] == 'unlimited'
             and stats['introduced_probability_or_entropy_gate'] is False, 'LVSpec strategy identity')
        need(stats['lvspec_parameters']['loose_ratio'] == variant['loose_ratio']
             and stats['lvspec_parameters']['disable_relaxation'] == variant['disable_relaxation'], 'LVSpec frozen policy')
        need('one legal actual root-to-leaf path' in stats['local_rank_set'], 'legal branch-local LV rank sets')
    for key in ('unsafe_bound_decisions', 'visual_bound_decision_disagreements', 'reference_decision_disagreements'):
        need(stats.get(key, 0) == 0, 'numeric parity: '+key)
    for key, limit in (('reference_audit_projection_relative_error_max', 1e-3),
                       ('reference_audit_component_abs_error_max', 1e-2),
                       ('reference_audit_score_abs_error_max', 1e-2)):
        need(stats.get(key) is None or stats[key] <= limit, 'numeric parity tolerance: '+key)


class Audit:
    def __init__(self, verify_history_bytes=False):
        self.checks, self.evidence, self.warnings = [], {}, []
        self.phase_rows, self.samples, self.configs, self.hashes = {}, {}, {}, {}
        self.metric_scores = {}
        self.repeat_differences, self.equivalence_differences = [], []
        self.verify_history_bytes = bool(verify_history_bytes)
        self.freshness_findings = []

    def check(self, name, function, required=True):
        try:
            result = function()
            self.checks.append(dict(name=name, passed=True, required_for_execution_integrity=required, evidence=result))
            return result
        except Exception as error:
            self.checks.append(dict(name=name, passed=False, required_for_execution_integrity=required, exception=repr(error)))
            return None

    def sources(self):
        manifest = read(ROOT/'audit/CLEAN_CHECKOUT.json')
        upstream = ROOT/'upstream/ViSpec'
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=upstream, text=True).strip()
        status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=upstream, text=True).strip()
        tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=upstream).decode().strip('\0').split('\0')
        need(head == manifest['head'] == '49a33ff0ca31e281b9dc8803fd6f3041c6c0e135', 'official fresh commit')
        need(not status and manifest['working_tree_clean'], 'unmodified upstream worktree')
        need(set(tracked) == set(manifest['tracked_sha256']), 'complete tracked file manifest')
        for name, expected in manifest['tracked_sha256'].items():
            need(sha(upstream/name) == expected, 'upstream file changed: '+name)
        adapters = read(ROOT/'audit/ADAPTER_SOURCE_PROVENANCE.json')
        for name, item in adapters.items():
            need(sha(item['original_source']) == item['original_sha256'], 'old read-only adapter source: '+name)
            need(sha(ROOT/'harness'/name) == item['copied_sha256'], 'new adapter provenance: '+name)
        lvroot = ROOT/'reference/LVSpec'
        lvhead = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=lvroot, text=True).strip()
        need(lvhead == '84dde16ea16b39402552eb4eb95d26f9c5101d15', 'fresh official LVSpec reference commit')
        need(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=lvroot, text=True).strip(), 'LVSpec reference remains read-only')
        lvfiles = {
            'decoding/loose_match.py': 'e899f8d346024409e40b427c941b587cdada44c938f845b8084e1b0883f41bdc',
            'decoding/tree_decoding_qwen2_5.py': 'a8ecb27fc1c7c43b9bcadb1b8560e6df5c623ba8153933e48a66049b6f6b14e5',
            'inference.py': '607f48306606d2a1dcf5f9dd761be45586ae144eb2c7f87485608bdcd36c00ff',
            'evaluation/qwen2_5_vl_lvspec.py': '55560d75eb9183a18bd554a2552db0bd5df571b1a13da33d2e97190c707d00cf',
            'models/modeling_qwen2_5_vl.py': 'c2d30524377c785d437e3c983c6624bdecfb32d2ef4a04b3d1b2c2890cf19457'}
        for name, expected in lvfiles.items():
            need(sha(lvroot/name) == expected, 'LVSpec official reference source: '+name)
        return dict(upstream_head=head, tracked_files=len(tracked), adapter_sources=len(adapters),
                    clean_checkout_manifest_sha256=sha(ROOT/'audit/CLEAN_CHECKOUT.json'),
                    official_lvspec_commit=lvhead, official_lvspec_sources=len(lvfiles))

    def configuration(self, phase):
        path = ROOT/'configs'/('frozen_config.json' if phase == 'main' else 'design_config.json')
        cfg = read(path)
        need(tuple(cfg['tasks']) == TASKS, 'fixed five tasks')
        need(cfg['temperature'] == 0 and cfg['repetitions'] == 3 and cfg['shards'] == 2, 'deterministic3repeat2GPU')
        need(cfg['tree'] == dict(total_token=30, depth=3, top_k=8, num_q=2), 'frozen native Draft tree')
        need(cfg['max_new_tokens'] == 128 and cfg['max_length'] == 2048, 'shared generation limits')
        need(cfg['processor'] == dict(use_fast=False, min_pixels=256*28*28, max_pixels=1280*28*28), 'fixed visual processor')
        need(cfg['system_prompt'] == 'You are a helpful assistant.', 'fixed system prompt')
        need(cfg['upstream'] == str(ROOT/'upstream/ViSpec'), 'fresh upstream import root')
        for name, expected in cfg['harness_sha256'].items():
            need(sha(ROOT/'harness'/name) == expected, 'runtime source SHA: '+name)
        for name, expected in cfg['test_sha256'].items():
            need(sha(ROOT/name) == expected, 'test SHA: '+name)
        parser_freeze = read(ROOT/'audit/METRIC_PARSER_FREEZE.json')
        need(parser_freeze['status'] == 'frozen_before_first_quality_scoring'
             and parser_freeze['evaluator_sha256'] == sha(ROOT/'harness/evaluate_tasks.py'), 'fixed answer parser freeze')
        need(parser_freeze['new_test_sha256'] == sha(parser_freeze['new_test_path']), 'parser boundary test source')
        need(parser_freeze['unchanged_original_five_test_sha256'] == cfg['test_sha256']
             and parser_freeze['frozen_runtime_source_sha256'] == cfg['harness_sha256'], 'parser fix leaves sealed decoder/tests unchanged')
        need(parser_freeze['development_quality_scoring_started'] is False
             and parser_freeze['benchmark_QA_generation_outputs_inspected'] is False
             and parser_freeze['quality_or_latency_results_used_to_choose_rule'] is False
             and parser_freeze['runtime_or_existing_sealed_test_modified'] is False, 'format-only correction precedes outcome inspection')
        need(sha(ROOT/'EXPERIMENT_PROTOCOL.md') == cfg['protocol_sha256'], 'frozen protocol SHA')
        need(sha(ROOT/'harness/prepare_experiment.py') == cfg['preparation_sha256'], 'preparation SHA')
        need(sha(ROOT/'audit/CLEAN_CHECKOUT.json') == cfg['checkout_manifest_sha256'], 'checkout seal')
        need(sha(cfg['model_identity_path']) == cfg['model_identity_sha256'], 'checkpoint identity manifest')
        model = read(cfg['model_identity_path'])
        need(sha(model['original_manifest']) == model['original_sha256'], 'original model identity manifest')
        for item in model['files']:
            stat, expected = Path(item['path']).stat(), item['stat_after']
            need((stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino, stat.st_dev) ==
                 (expected['size_bytes'], expected['mtime_ns'], expected['ctime_ns'], expected['inode'], expected['device']),
                 'checkpoint file identity changed: '+item['path'])
        templates = cfg['main_method_templates']
        need(set(templates) == MAIN_METHODS, 'seven actual main methods')
        need(not templates['vispec']['compact_logits'] and not templates['vispec']['terminal_skip'], 'original native ViSpec baseline')
        need(templates['strict_shared']['compact_logits'] and templates['strict_shared']['terminal_skip'], 'separate shared engineering baseline')
        need(not templates['ar']['compact_logits'] and not templates['ar']['terminal_skip'], 'serial AR identity')
        for name in ('bepc_algorithm', 'margin4_ccve'):
            need(templates[name]['round_mismatch_budget'] == 2
                 and templates[name]['response_mismatch_budget'] == 4, 'fixed BEPC/ME2/4')
        bepc = templates['bepc_algorithm']
        need(bepc['tau'] == 1 and bepc['rank_u'] == 8 and bepc['rank_p'] == 4
             and bepc['text_window'] == 32 and bepc['visual_pool_size'] == 64
             and bepc['gamma'] == .5 and bepc['kappa'] == 1 and bepc['eta'] == 1e-4
             and bepc['residual_attenuation'] is False and bepc['correction_mode'] == 'both', 'unchanged BEPC algorithm')
        me = templates['margin4_ccve']
        need(me['tau_M'] == 4 and me['tau_E'] == -.165220245718956, 'unchanged Margin4+CCVE thresholds')
        lv = templates['lvspec_paper']
        need(lv['loose_ratio'] == .7 and lv['similarity_top_n'] == 10 and lv['draft_window'] == 10
             and lv['pst_enabled'] and lv['pst_include_root'] and lv['stable_rank_ties']
             and lv['visual_reference'] == 'image_tokens' and not lv['disable_relaxation'], 'paper LVSpec identity')
        need('round_mismatch_budget' not in lv and 'response_mismatch_budget' not in lv, 'unbounded original LVSpec budget')
        need(sorted(v['loose_ratio'] for v in cfg['development_methods'].values() if v['family'] == 'LVSPEC') == [0., .3, .5, .7], 'predeclared LVSpec dev grid')
        need(cfg['bootstrap'] == 5000 and cfg['primary_quality_repeat'] == 0 and cfg['no_training']
             and cfg['no_new_judge_calls'], 'statistics and scope preregistration')
        need(cfg['expected_development_requests'] == 3240 and cfg['expected_main_requests'] == 12600
             and cfg['expected_parity_requests'] == 120, 'declared request totals')
        if phase == 'main':
            need(cfg['config_finalized_before_generation'] is True, 'main configuration frozen before generation')
            need(sha(ROOT/'configs/design_config.json') == cfg['design_config_sha256'], 'design unchanged after dev')
            need(sha(ROOT/'configs/lvspec_selection.json') == cfg['selection_sha256'], 'dev selection frozen')
            ready = read(ROOT/'audit/MAIN_READY.json')
            need(ready['status'] == 'pass' and ready['config_sha256'] == sha(path)
                 and ready['development_only_selection'] and ready['test_outputs_inspected'] is False, 'main readiness seal')
        self.configs[phase] = cfg
        return dict(config_path=str(path), sha256=sha(path), runtime_sources=len(cfg['harness_sha256']),
                    tests=len(cfg['test_sha256']), model_files=len(model['files']), model_verification='prior SHA identity plus current full stat tuple; no checkpoint rehash')

    def data(self):
        cfg = next(iter(self.configs.values()))
        manifest = read(ROOT/'audit/DATA_MANIFEST.json')
        prior_path = ROOT/'audit/DATA_PRIOR_IDENTITIES_FAST.json'
        prior = read(prior_path)
        allowed = manifest['explicit_history_reuse_tasks']
        need(set(allowed) <= {'mme'}, 'only MME historical reuse explicitly authorized by protocol')
        need(manifest['runtime_records_contain_gold_labels'] is False
             and manifest['old_results_quality_not_used'] is True, 'label and outcome selection separation')
        for entry in prior['sources']:
            need(Path(entry['path']).is_file() and len(entry['sha256']) == 64, 'preserved historical identity source')
            if self.verify_history_bytes:
                need(sha(entry['path']) == entry['sha256'], 'history exclusion source changed: '+entry['path'])
        amendment = read(ROOT/'audit/DATA_AUDIT_EXTRACTION_AMENDMENT.json')
        need('before any sample selection' in amendment['initial_python_per_row_inventory'], 'history extraction amendment predates selection')
        supplement_path = ROOT/'audit/DATA_PARQUET_IDENTITY_SUPPLEMENT.json'
        supplement = read(supplement_path)
        need(supplement['only_identity_columns_read'] is True, 'parquet exclusion audit uses identities, not quality')
        for entry in supplement['sources']:
            need(Path(entry['path']).is_file() and len(entry['sha256']) == 64, 'preserved parquet identity source')
            if self.verify_history_bytes:
                need(sha(entry['path']) == entry['sha256'], 'parquet history exclusion source changed')
        ignored_path = ROOT/'audit/DATA_IGNORED_IDENTITY_SUPPLEMENT.json'
        ignored = read(ignored_path)
        need(ignored['only_identity_fields_used'] is True, 'ignored files audited without old generated quality')
        for entry in ignored['sources']:
            need(Path(entry['path']).is_file() and len(entry['sha256']) == 64, 'preserved ignored-file identity source')
            if self.verify_history_bytes:
                need(sha(entry['path']) == entry['sha256'], 'ignored history exclusion source changed')
        history_sources = (prior, ignored)
        previous_ids = {task: {key: {historical_identity(value) for source in history_sources
            for value in source['used'][task][key]} for key in ('questions', 'images')} for task in TASKS}
        prior_image_bytes = {value for source in history_sources for value in source['image_sha256']}
        prior_rgb = {value for source in history_sources for value in source['image_rgb_sha256']}
        erratum_path = ROOT/'audit/DATA_HISTORICAL_OVERLAP_SUPPLEMENT.json'
        erratum = read(erratum_path)
        need(erratum['frozen_data_manifest_sha256'] == sha(ROOT/'audit/DATA_MANIFEST.json')
             and erratum['frozen_runtime_and_gold_file_sha256_unchanged'] is True
             and erratum['resampled_events_or_images'] is False
             and erratum['repeated_inference_to_address_erratum'] is False, 'explicit historical erratum retains frozen cohort')
        need(erratum['prior_identity_manifest_sha256'] == sha(prior_path)
             and erratum['additional_rg_visibility']['supplement_sha256'] == sha(ignored_path), 'historical erratum source seals')
        need(erratum['additional_rg_visibility']['additional_exact_selected_id_or_hash_overlaps'] == 0, 'ignored-file supplemental exact identity check')
        overrides = {(r['task'], r['split'], str(r['question_id'])): r for r in erratum['field_overrides_apply_at_reporting_only']}
        need(len(overrides) == len(erratum['field_overrides_apply_at_reporting_only']), 'unique historical metadata errata')
        for name, digest in read(ROOT/'audit/DATA_AUDIT_SEAL.json').items():
            need(sha(ROOT/'audit'/name) == digest, 'preserved original data audit seal: '+name)
        verified_prior_sources = set()
        for item in overrides.values():
            source = item['prior_source']
            path = Path(source['path'])
            if str(path) not in verified_prior_sources:
                need(sha(path) == source['sha256'], 'historical overlap original source bytes')
                verified_prior_sources.add(str(path))
            with path.open(encoding='utf-8') as stream:
                for line_number, line in enumerate(stream, 1):
                    if line_number == source['line_number']:
                        record = json.loads(line)
                        break
                else:
                    raise AssertionError('Historical overlap line absent')
            need(record.get('image_id') == source['image_id'], 'historical overlap original line image ID')
            need(historical_identity(record['image_id']) == historical_identity(item['image_id']), 'historical namespace image alias')
            need(source['generation_event_field_presence'] and all(key in record for key in source['generation_event_field_presence']), 'historical source is actual generation/event record')
            if source.get('image_sha256') is not None:
                need(source['image_sha256'] == record.get('image_sha256') == item['image_sha256'], 'historical overlap exact image bytes')
        discovered = set()
        all_samples, image_cache, statistics_out = {}, {}, {}
        self.samples = all_samples
        for task in TASKS:
            statistics_out[task] = {}
            for split, groups in (('dev', 20), ('test', 100)):
                data = cfg['data'][task]
                path, labels_path = Path(data[split+'_path']), Path(data[split+'_labels_path'])
                need(path == ROOT/f'data/{task}/{split}.json', 'canonical runtime data path')
                need(sha(path) == data[split+'_sha256'] and sha(labels_path) == data[split+'_labels_sha256'], 'input/label SHA')
                meta = manifest['splits'][f'{task}/{split}']
                need(meta['sha256'] == sha(path) and meta['labels_sha256'] == sha(labels_path), 'independent data manifest SHA')
                samples, labels = read(path), read(labels_path)
                all_samples[task, split] = samples
                need(len(samples) == groups*(2 if task == 'mme' else 1), 'fixed question count')
                need(len({r['group_id'] for r in samples}) == groups, 'fixed group count')
                need([r['ordinal'] for r in samples] == list(range(len(samples))), 'stable sample ordinals')
                need(set(labels) == {str(r['question_id']) for r in samples}, 'one official label per runtime question')
                grouped = defaultdict(list)
                for row in samples:
                    need(row['task'] == task and not (set(row) & FORBIDDEN_RUNTIME_LABELS), 'runtime excludes gold labels')
                    need(row['system_prompt'] == cfg['system_prompt'], 'system prompt agreement')
                    need(isinstance(row['question_id'], str) and row['prompt_text'].strip(), 'string question ID / nonempty prompt')
                    need(0 <= row['group_ordinal'] < groups, 'group ordinal range')
                    grouped[row['group_id']].append(row)
                    label = labels[row['question_id']]
                    need(label['image_sha256'] == row['image_sha256'], 'label image alignment')
                    need(row['history_reuse_required'] == (task in allowed)
                         and row['fresh_to_project'] == (task not in allowed), 'historical reuse disclosure')
                    if task not in allowed:
                        previous = previous_ids[task]
                        overlap = (historical_identity(row['question_id']) in previous['questions']
                            or historical_identity(row['image_id']) in previous['images']
                            or row['image_sha256'] in prior_image_bytes or row['image_rgb_sha256'] in prior_rgb)
                        key = task, split, str(row['question_id'])
                        if overlap:
                            discovered.add(key)
                            need(key in overrides, f'undeclared historical overlap: {key}')
                            override = overrides[key]
                            need(all(override[name] == row[name] for name in ('image_id', 'image_sha256', 'image_rgb_sha256')), 'historical erratum exact frozen image')
                            need(all(row[name] == value for name, value in override['stored_fields'].items()), 'preserve original metadata rather than rewrite history')
                            need(override['effective_field_overrides'] == dict(fresh_to_project=False, historical_image_overlap=True), 'effective report discloses history overlap')
                    image_path = Path(row['image_path'])
                    if str(image_path) not in image_cache:
                        from PIL import Image
                        with Image.open(image_path) as image:
                            rgb = image.convert('RGB')
                            rgb_id = hashlib.sha256(rgb.tobytes()).hexdigest()
                            image_cache[str(image_path)] = (sha(image_path), rgb_id, list(rgb.size))
                    original, rgb_digest, size = image_cache[str(image_path)]
                    need(original == row['image_sha256'], 'original image bytes')
                    need(size == row['image_size'], 'decoded image dimensions')
                    # Exact RGB hash convention checked against preparation below.
                    need(rgb_digest == row['image_rgb_sha256'], 'decoded RGB image identity')
                for group in grouped.values():
                    need(len(group) == (2 if task == 'mme' else 1), 'complete question group')
                    need(len({r['image_rgb_sha256'] for r in group}) == 1
                         and len({r['group_ordinal'] for r in group}) == 1, 'group image identity')
                    if task == 'mme':
                        need(len({r['category'] for r in group}) == 1, 'same-category paired MME questions')
                        need({labels[r['question_id']]['answer'] for r in group} == {'yes', 'no'}, 'MME official positive/negative pair')
                need(len({group[0]['image_rgb_sha256'] for group in grouped.values()}) == groups, 'unique image groups including across MME categories')
                all_samples[task, split] = samples
                statistics_out[task][split] = dict(groups=groups, questions=len(samples),
                    categories=dict(Counter(r.get('category') for r in samples)) if task == 'mme' else None,
                    historical_reuse_explicit=task in allowed)
            dev, test = all_samples[task, 'dev'], all_samples[task, 'test']
            for key in ('image_sha256', 'image_rgb_sha256'):
                need(not {r[key] for r in dev} & {r[key] for r in test}, 'dev/test image independence: '+task+'/'+key)
            need(not {r['question_id'] for r in dev} & {r['question_id'] for r in test}, 'dev/test question independence')
            if task == 'mme':
                need(len({r['category'] for r in dev}) == len({r['category'] for r in test}) == 14, 'all14MME categories stratified')
        need(discovered == set(overrides), 'canonical history scan and explicit errata cover exactly the same seven images')
        for task in TASKS:
            for split in ('dev', 'test'):
                samples = all_samples[task, split]
                count = len({r['group_id'] for r in samples}) if task in allowed else sum(key[0] == task and key[1] == split for key in discovered)
                expected = erratum['effective_group_counts'][task][split]
                need(expected['groups'] == len({r['group_id'] for r in samples})
                     and expected['effective_history_overlapping_groups'] == count
                     and expected['effective_no_recorded_history_overlap_groups'] == expected['groups']-count, 'effective historical overlap counts')
                if count:
                    self.freshness_findings.append(dict(task=task, split=split, historical_image_groups=count,
                                                       total_groups=expected['groups']))
        global_rgb_groups = defaultdict(set)
        for (task, split), samples in all_samples.items():
            for row in samples:
                global_rgb_groups[row['image_rgb_sha256']].add((task, split, row['group_id']))
        need(len(global_rgb_groups) == 600 and all(len(groups) == 1 for groups in global_rgb_groups.values()),
             '600 globally distinct physical image groups across both splits and all tasks')
        self.samples = all_samples
        self.warnings.append('MME history reuse is explicitly authorized; this is not an unseen-image generalization claim.')
        if not self.verify_history_bytes:
            self.warnings.append('Historical exclusion sources are19.43GB/20929files: preserved source SHA entries and chosen-ID exclusions checked; all historical bytes are not rehashed in this invocation.')
        self.warnings.append('Seven non-MME historical image aliases are disclosed in the immutable reporting erratum; COCO test98/100 and TextVQA test97/100 have no recorded project history. No images are resampled.')
        return dict(task_splits=statistics_out, history_source_files=len(prior['sources']),
                    historical_reuse_tasks=allowed, data_manifest_sha256=sha(ROOT/'audit/DATA_MANIFEST.json'),
                    prior_identity_manifest_sha256=sha(prior_path), parquet_supplement_sha256=sha(supplement_path),
                    historical_source_bytes_rehashed=self.verify_history_bytes,
                    historical_erratum_sha256=sha(erratum_path), canonical_nonMME_history_overlaps=len(discovered),
                    ignored_file_supplement_sha256=sha(ignored_path))

    def loaded(self, phase):
        cfg, cfg_path = self.configs[phase], ROOT/'configs'/('frozen_config.json' if phase == 'main' else 'design_config.json')
        output = []
        for shard in (0, 1):
            meta = read(ROOT/f'audit/{phase}_shard{shard}.loaded.json')
            need(meta['config_sha256'] == sha(cfg_path) and meta['runner_sha256'] == cfg['harness_sha256']['run_benchmark.py'], 'loaded config/runner identity')
            need(meta['source_sha256'] == cfg['harness_sha256'] and meta['processor_settings'] == cfg['processor'], 'loaded harness/processor identity')
            need(meta['physical_gpu'] == str(shard) and meta['AR_memory_includes_idle_loaded_Draft'], 'fixed GPU / AR memory disclosure')
            need(meta['loaded_vispec_sources'], 'actual imported ViSpec sources recorded')
            for name, item in meta['loaded_vispec_sources'].items():
                path = Path(item['path']).resolve()
                need(path.is_relative_to((ROOT/'upstream/ViSpec').resolve()), 'all actual vispec imports from fresh clone: '+name)
                need(sha(path) == item['sha256'], 'actually imported source unchanged: '+name)
            resource = read(ROOT/f'audit/{phase}_shard{shard}.gpu_resource.json')
            need(resource['physical_gpu'] == str(shard) and resource['checks'] > 0 and not resource['events'], 'exclusive physical GPU resource log')
            output.append(dict(shard=shard, imported_sources=len(meta['loaded_vispec_sources']), resource_checks=resource['checks']))
        return output

    def phase(self, phase):
        cfg = self.configs[phase]
        split = 'test' if phase == 'main' else 'dev'
        cfg_path = ROOT/'configs'/('frozen_config.json' if phase == 'main' else 'design_config.json')
        rows_by_task, manifest, total, warmups = {}, {}, 0, 0
        for task in TASKS:
            samples = self.samples[task, split]
            if phase == 'parity':
                samples = [r for r in samples if r['group_ordinal'] < 2]
            byid = {str(r['question_id']): r for r in samples}
            methods = cfg['methods_by_task'][task] if phase == 'main' else cfg['parity_methods'] if phase == 'parity' else cfg['development_methods']
            repeats = 1 if phase == 'parity' else 3
            rows = []
            for shard in (0, 1):
                path = ROOT/f'results/{phase}/{phase}_{task}_shard{shard}.jsonl'
                complete = read(path.with_suffix('.complete.json'))
                need(complete['complete'] and complete['task'] == task and complete['phase'] == phase, 'completed phase/task marker')
                need(sha(path) == complete['result_sha256'], 'complete result hash')
                batch = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                need(complete['records'] == len(batch), 'complete record count')
                need(complete['metadata']['config_sha256'] == sha(cfg_path), 'complete config seal')
                need(complete['metadata'] == read(ROOT/f'audit/{phase}_shard{shard}.loaded.json'), 'complete and process metadata agreement')
                own = [r for r in samples if r['group_ordinal'] % 2 == shard]
                need(complete['questions'] == len(own), 'shard full group/question count')
                expected = {(method, r['ordinal'], rep) for method in methods for r in own for rep in range(repeats)}
                actual = [(r['method'], r['ordinal'], r['repeat']) for r in batch]
                need(len(actual) == len(set(actual)) and set(actual) == expected, 'complete actual question/method/repeat grid')
                for row in batch:
                    need(row['config_sha256'] == sha(cfg_path), 'row config SHA')
                    need(str(row['question_id']) in byid, 'frozen question ID')
                    row_contract(row, byid[str(row['question_id'])], methods[row['method']], cfg, phase, shard)
                warm = read(ROOT/f'audit/{phase}_shard{shard}_{task}.warmup.json')
                expected_warm = len(methods)*(cfg['warmups'] if phase == 'main' else 1)
                need(warm['requests'] == expected_warm and warm['lifecycle_checked_every_request'], 'warmup requests retained')
                need(warm['seconds'] > 0, 'warmup duration retained separately')
                warmups += warm['requests']
                if phase == 'parity':
                    check_file = read(path.with_suffix('.checks.json'))
                    need(check_file['status'] == 'pass' and len(check_file['checks']) == len(own)*len(cfg['parity_pairs']), 'parity audit result count')
                    need(all(all(v is True for k, v in check.items() if k not in ('question_id', 'method', 'baseline'))
                             for check in check_file['checks']), 'retained parity booleans')
                rows.extend(batch)
                manifest[str(path)] = sha(path)
            lookup = {(r['method'], r['ordinal'], r['repeat']): r for r in rows}
            for sample in samples:
                for rep in range(repeats):
                    scheduled = list(methods)
                    random.Random(cfg['order_seed']+TASKS.index(task)*1000003+sample['ordinal']*101+rep*10007).shuffle(scheduled)
                    selected = [lookup[method, sample['ordinal'], rep] for method in scheduled]
                    need([row['schedule_position'] for row in selected] == list(range(len(methods))), 'predeclared randomized method order')
                    for key in ('prompt_token_ids', 'input_tensor_sha256', 'image_grid_thw', 'input_preprocessing_seconds'):
                        need(all(row[key] == selected[0][key] for row in selected), 'paired methods share original processed input: '+key)
                for method in methods:
                    for rep in range(1, repeats):
                        a, b = lookup[method, sample['ordinal'], 0], lookup[method, sample['ordinal'], rep]
                        fields = [key for key in REPEAT_FIELDS if a[key] != b[key]]
                        if fields:
                            self.repeat_differences.append(dict(phase=phase, task=task, method=method,
                                question_id=sample['question_id'], repeat=rep, changed_fields=fields))
                pairs = cfg['parity_pairs'] if phase == 'parity' else [('strict_shared', 'vispec')]
                for left, right in pairs:
                    for rep in range(repeats):
                        a, b = lookup[left, sample['ordinal'], rep], lookup[right, sample['ordinal'], rep]
                        fields = list(PARITY_FIELDS)
                        if a['acceptance_paths'] and b['acceptance_paths']:
                            fields.append('acceptance_paths')
                        changed = [key for key in fields if a[key] != b[key]]
                        if a['model_calls']['target'] != b['model_calls']['target']:
                            changed.append('target_calls')
                        if changed:
                            self.equivalence_differences.append(dict(phase=phase, task=task, question_id=sample['question_id'],
                                repeat=rep, method=left, baseline=right, changed_fields=changed))
            rows_by_task[task] = rows
            total += len(rows)
        expected = {'parity': 120, 'development': 3240, 'main': 12600}[phase]
        need(total == expected, 'measured phase request total')
        need(warmups == {'parity': 100, 'development': 90, 'main': 210}[phase], 'phase warmup total')
        self.phase_rows[phase], self.hashes[phase] = rows_by_task, manifest
        return dict(measured_requests=total, warmup_requests=warmups, result_sha256=manifest,
                    repeated_outputs_changed=sum(r['phase'] == phase for r in self.repeat_differences),
                    equivalence_differences=sum(r['phase'] == phase for r in self.equivalence_differences))

    def metrics(self, phase):
        cfg = self.configs[phase]
        cfg_path = ROOT/'configs'/('frozen_config.json' if phase == 'main' else 'design_config.json')
        scorer = load_local('evaluate_tasks')
        results = {}
        for task in TASKS:
            results[task] = {}
            split = 'test' if phase == 'main' else 'dev'
            samples = self.samples[task, split]
            labeled, _ = scorer.attach_official_labels(task, samples)
            methods = cfg['methods_by_task'][task] if phase == 'main' else cfg['development_methods']
            for method in methods:
                folder = ROOT/f'metrics/{phase}/{task}/{method}'
                seal, rows, aggregate = (read(folder/name) for name in ('SCORING_SEAL.json', 'per_question_scores.json', 'aggregate.json'))
                need(seal['config_sha256'] == sha(cfg_path) and seal['inputs_sha256'] == cfg['data'][task][split+'_sha256'], 'score config/input seal')
                need(seal['quality_repeat'] == 0 and seal['evaluator_sha256'] == sha(ROOT/'harness/evaluate_tasks.py'), 'frozen evaluator / fixed quality repeat')
                freeze_path = ROOT/'audit/METRIC_PARSER_FREEZE.json'
                need(seal['metric_parser_freeze_sha256'] == sha(freeze_path), 'score sealed parser provenance')
                need((folder/'SCORING_SEAL.json').stat().st_mtime_ns >= freeze_path.stat().st_mtime_ns,
                     'parser frozen before actual task scoring')
                need(seal['aggregate_sha256'] == sha(folder/'aggregate.json')
                     and seal['per_question_sha256'] == sha(folder/'per_question_scores.json'), 'immutable metric results')
                need(seal['result_sha256'] == {p: v for p, v in self.hashes[phase].items() if f'_{task}_shard' in p}, 'score references exactly actual runtime shards')
                selected = sorted((r for r in self.phase_rows[phase][task] if r['method'] == method and r['repeat'] == 0), key=lambda r: r['ordinal'])
                predictions = {str(r['question_id']): r['output_text'] for r in selected}
                identity = hashlib.sha256(json.dumps(predictions, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                need(seal['predictions_sha256'] == identity, 'score prediction identity')
                need(len(rows) == len(samples) and [r['sample_id'] for r in rows] == [r['question_id'] for r in samples], 'complete aligned score population')
                need(all(r['prediction'] == predictions[r['sample_id']] for r in rows), 'each score uses frozen repeat0 text')
                need(aggregate['evaluator_sha256'] == sha(ROOT/'harness/evaluate_tasks.py')
                     and aggregate['all_questions_retained'] and not aggregate['automatic_judge_used'], 'metric provenance / no autojudge')
                need(aggregate['parse_invalid_count'] == sum(r['parse_invalid'] for r in rows)
                     and aggregate['uncertain_count'] == sum(r['uncertain'] for r in rows), 'invalid/uncertain denominator retention')
                if task != 'coco':
                    function = {'mme': scorer._score_mme, 'gqa': scorer._score_gqa,
                                'textvqa': scorer._score_textvqa, 'scienceqa': scorer._score_scienceqa}[task]
                    again_rows, again = function(labeled, [predictions[r['question_id']] for r in samples])
                    need(again_rows == rows, 'independent repeat0 official scorer replay')
                    for key in ('accuracy', 'MME_subset_score', 'Perception_subset_score', 'Cognition_subset_score', 'acc_plus'):
                        if key in again:
                            need(close(aggregate[key], again[key]), 'official aggregate recomputed: '+key)
                else:
                    need(aggregate['CIDEr_document_frequency_images'] == len(samples), 'fresh current split DF')
                    need(close(aggregate['CIDEr'], statistics.mean(r['CIDEr'] for r in rows)), 'CIDEr average')
                    spice = None if any(r['SPICE'] is None for r in rows) else statistics.mean(r['SPICE'] for r in rows)
                    need(close(aggregate['SPICE'], spice), 'SPICE undefined retention / mean')
                    need(close(aggregate['CHAIRs'], statistics.mean(r['CHAIRs'] for r in rows)), 'CHAIR sentence ratio')
                    objects = sum(r['object_mentions'] for r in rows)
                    hallucinated = sum(r['hallucinated_mentions'] for r in rows)
                    need(aggregate['object_mentions'] == objects and aggregate['hallucinated_mentions'] == hallucinated,
                         'CHAIR corpus numerator / denominator')
                    need(close(aggregate['CHAIRi'], hallucinated/objects if objects else None), 'CHAIR object ratio rather than per-caption mean')
                    spice_meta = aggregate['additional_provenance']['spice']
                    need(spice_meta['input_sha256'] == sha(folder/'spice/input.json'), 'current SPICE input')
                    need(sha(folder/'spice/output.json') and (folder/'spice/scorer.log').exists(), 'real SPICE output and process log retained')
                    need((folder/'chair/author.log').exists(), 'real author CHAIR audit retained')
                    need(all(r['SPICE_status'] == ('ok' if r['SPICE'] is not None else 'undefined_official_All_f') for r in rows), 'SPICE parse states')
                results[task][method] = dict(rows=rows, aggregate=aggregate, seal_sha256=sha(folder/'SCORING_SEAL.json'))
        self.evidence[phase+'_metric_scores'] = {task: {method: entry['seal_sha256'] for method, entry in own.items()} for task, own in results.items()}
        self.metric_scores[phase] = results
        return results

    def selection(self, scores):
        cfg = self.configs['development']
        frozen = read(ROOT/'configs/frozen_config.json')
        selection = read(ROOT/'configs/lvspec_selection.json')
        need(selection['status'] == 'development_only_selection' and selection['ready_for_main']
             and selection['main_outputs_used'] is False and not selection['repeat_differences'], 'selection uses deterministic dev only')
        need(selection['design_config_sha256'] == sha(ROOT/'configs/design_config.json')
             and selection['selector_sha256'] == sha(ROOT/'harness/select_lvspec.py')
             and selection['evaluator_sha256'] == sha(ROOT/'harness/evaluate_tasks.py'), 'selection implementation seals')
        need(selection['metric_parser_freeze_sha256'] == sha(ROOT/'audit/METRIC_PARSER_FREEZE.json'), 'selection fixed parser provenance')
        need(selection['result_sha256'] == self.hashes['development'], 'selection dev population only')
        decisions = {}
        for task in TASKS:
            baseline = scores[task]['vispec']['aggregate']
            candidates = []
            for method, variant in cfg['development_methods'].items():
                if variant['family'] != 'LVSPEC':
                    continue
                current = scores[task][method]['aggregate']
                if task == 'coco':
                    passed = all(current.get(key) is not None and baseline.get(key) is not None for key in ('CIDEr', 'SPICE', 'CHAIRs'))
                    passed = passed and current['CIDEr']+1e-12 >= .95*baseline['CIDEr'] and current['SPICE']+1e-12 >= .95*baseline['SPICE'] and current['CHAIRs'] <= baseline['CHAIRs']+1e-12
                elif task == 'mme':
                    passed = all(current[key]+1e-12 >= baseline[key] for key in ('MME_subset_score', 'accuracy'))
                else:
                    passed = current['accuracy']+1e-12 >= baseline['accuracy']
                times = [r['seconds'] for r in self.phase_rows['development'][task] if r['method'] == method]
                candidates.append(dict(method=method, ratio=variant['loose_ratio'], passed=bool(passed), mean=statistics.mean(times)))
            need(sorted(r['ratio'] for r in candidates) == [0., .3, .5, .7], 'independent four-ratio selection grid')
            paper = next(r for r in candidates if r['ratio'] == .7)
            eligible = [r for r in candidates if r['ratio'] < .7 and r['passed']]
            chosen = paper if paper['passed'] else min(eligible, key=lambda r: (r['mean'], r['ratio'])) if eligible else None
            actual = selection['selections'][task]
            need(actual['paper_qualitypass'] == paper['passed'] and actual['strict_fallback'] == (chosen is None), 'paper-first / explicit fallback rule')
            need(actual['selecteddevmethod'] == (chosen['method'] if chosen else 'vispec'), 'correct dev quality-constrained fastest selection')
            need(actual['qualitypass'] == bool(chosen), 'quality-pass selection status')
            need(actual['parameters']['disable_relaxation'] == (chosen is None), 'lambda0/PST is not mislabelled strict fallback')
            need(actual['parameters']['loose_ratio'] == (chosen['ratio'] if chosen else .7), 'selected ratio')
            for candidate in actual['candidates']:
                own = next(r for r in candidates if r['method'] == candidate['method'])
                need(candidate['qualitypass'] == own['passed'] and close(candidate['mean_seconds'], own['mean']), 'stored candidate quality /3repeat latency')
                need(candidate['score_seal_sha256'] == scores[task][candidate['method']]['seal_sha256'], 'selection metric seal')
            expected = dict(cfg['main_method_templates']['lvspec_adapted'], **actual['parameters'])
            need(frozen['methods_by_task'][task]['lvspec_adapted'] == expected, 'test LVSpec parameters equal frozen dev choice')
            for name in MAIN_METHODS-{'lvspec_adapted'}:
                need(frozen['methods_by_task'][task][name] == cfg['main_method_templates'][name], 'no other test method tuning')
            decisions[task] = dict(selected=actual['selecteddevmethod'], strict_fallback=actual['strict_fallback'], independently_recomputed=True)
        return decisions

    def analysis(self):
        import csv
        import numpy as np
        path = ROOT/'audit/STATISTICAL_ANALYSIS.json'
        report = read(path)
        cfg = self.configs['main']
        need(report['config_sha256'] == sha(ROOT/'configs/frozen_config.json'), 'statistical analysis config seal')
        for key, name in (('analyzer_sha256', 'analyze_results.py'), ('selector_sha256', 'select_lvspec.py'), ('evaluator_sha256', 'evaluate_tasks.py')):
            need(report[key] == sha(ROOT/'harness'/name), 'analysis source seal: '+name)
        need(report['result_sha256'] == self.hashes['main'], 'statistical analysis full main population')
        need(report['measured_requests'] == 12600 and report['bootstrap'] == 5000, 'analysis sample/repetition/bootstrap counts')
        need(not report['repeat_differences'], 'analysis retains zero actual repeat differences')
        need(report['MME_stratified_by_category'] is True, 'MME preserves strata and paired groups')
        need(report['immutable_runtime_sources'] == cfg['harness_sha256'], 'analysis frozen runtime source identity')
        need(report['selection_sha256'] == sha(ROOT/'configs/lvspec_selection.json'), 'analysis frozen dev selection')
        need(report['metric_parser_freeze_sha256'] == sha(ROOT/'audit/METRIC_PARSER_FREEZE.json'), 'analysis fixed parser provenance')
        need(report['prespecified_comparisons'] == cfg['comparisons']
             and report['prespecified_secondary_comparisons'] == cfg['secondary_comparisons'], 'prespecified statistical comparison families')
        need(set(report['draws']) == set(TASKS), 'draws for all five tasks')
        for task, drawing in report['draws'].items():
            need(task in TASKS and drawing['shape'] == [5000, 100], '100 group bootstrap draws')
            need(drawing['stratified'] == (task == 'mme'), 'stratification only where declared')
            samples = self.samples[task, 'test']
            groups = list(dict.fromkeys(row['group_id'] for row in samples))
            categories = [next(row.get('category', 'all') for row in samples if row['group_id'] == group) for group in groups]
            need(drawing['groups'] == groups and drawing['categories'] == categories, 'draw population matches actual physical groups')
            need(drawing['stratum_counts'] == dict(Counter(categories)), 'exact category stratum counts')
            seed = cfg['bootstrap_seed']+TASKS.index(task)*100003
            need(drawing['seed'] == seed, 'fixed task-specific bootstrap seed')
            rng = np.random.default_rng(seed)
            if task == 'mme':
                blocks = []
                for category in sorted(set(categories)):
                    indices = np.asarray([i for i, value in enumerate(categories) if value == category], dtype=np.int64)
                    blocks.append(indices[rng.integers(0, len(indices), size=(5000, len(indices)))])
                draws = np.concatenate(blocks, axis=1)
            else:
                draws = rng.integers(0, 100, size=(5000, 100), dtype=np.int64)
            need(hashlib.sha256(draws.tobytes()).hexdigest() == drawing['draw_sha256'], 'independent deterministic bootstrap draw reconstruction')
        for name, expected in report['output_sha256'].items():
            target = Path(name) if Path(name).is_absolute() else ROOT/name
            need(target.resolve().is_relative_to(ROOT.resolve()) and sha(target) == expected, 'analysis output immutable: '+name)
        need(len(report['output_sha256']) >= 5, 'summary/comparisons/report/figures retained')
        with (ROOT/'reports/BENCHMARK_SUMMARY.csv').open(newline='') as stream:
            summaries = list(csv.DictReader(stream))
        with (ROOT/'reports/PAIRED_COMPARISONS.csv').open(newline='') as stream:
            comparisons = list(csv.DictReader(stream))
        summary = {(row['task'], row['method']): row for row in summaries}
        need(len(summary) == len(summaries) == 35 and set(summary) == {(task, method) for task in TASKS for method in MAIN_METHODS}, 'summary contains all35 task/method results')
        def number(value):
            return None if value in ('', None) else float(value)
        for (task, method), row in summary.items():
            runtime = [r for r in self.phase_rows['main'][task] if r['method'] == method]
            metric = self.metric_scores['main'][task][method]
            need(int(row['requests']) == len(runtime) and int(row['questions']) == len(self.samples[task, 'test'])
                 and int(row['images_or_groups']) == 100 and int(row['quality_repeat']) == 0, 'summary population / repeat0')
            for key in ('accuracy', 'MME_subset_score', 'Perception_subset_score', 'Cognition_subset_score',
                        'acc_plus', 'CIDEr', 'SPICE', 'CHAIRs', 'CHAIRi'):
                if key in metric['aggregate']:
                    need(close(number(row[key]), metric['aggregate'][key]), 'summary official metric point: '+key)
            times = [r['seconds'] for r in runtime]
            decodes = [r['decode_seconds'] for r in runtime]
            points = dict(mean_seconds=statistics.mean(times), mean_decode_seconds=statistics.mean(decodes),
                throughput_tokens_per_second=sum(len(r['returned_token_ids']) for r in runtime)/sum(times),
                decode_throughput_tokens_per_second=sum(len(r['returned_token_ids']) for r in runtime)/sum(decodes),
                mean_peak_allocated_bytes=statistics.mean(r['peak_allocated_bytes'] for r in runtime),
                completion_tokens_including_EOS_throughput_per_second=sum(r['completion_tokens'] for r in runtime)/sum(times),
                completion_tokens_including_EOS_decode_throughput_per_second=sum(r['completion_tokens'] for r in runtime)/sum(decodes),
                mean_visible_tokens=statistics.mean(len(r['returned_token_ids']) for r in runtime),
                mean_raw_tokens=statistics.mean(len(r['raw_token_ids']) for r in runtime),
                truncation_fraction=statistics.mean(r['cap_reached'] for r in runtime),
                effective_mean_accepted_draft_length=None if method == 'ar' else
                    sum(r['effective_accepted_draft_tokens'] for r in runtime)/sum(r['effective_verification_rounds'] for r in runtime))
            for key, value in points.items():
                need(close(number(row[key]), value), 'independent reported cost/length point: '+key)
            need(row['scoring_seal_sha256'] == metric['seal_sha256'], 'summary method score seal')
            need(int(row['parse_invalid_count']) == metric['aggregate']['parse_invalid_count']
                 and int(row['parse_uncertain_count']) == metric['aggregate']['uncertain_count'], 'summary parse failures retained')
            baseline = [r for r in self.phase_rows['main'][task] if r['method'] == 'vispec']
            need(close(number(row['speedup_vs_native']), statistics.mean(r['seconds'] for r in baseline)/statistics.mean(times)), 'native-relative speedup')
        families = defaultdict(list)
        comparison_keys = []
        for row in comparisons:
            key = tuple(row[name] for name in ('task', 'method', 'baseline', 'metric', 'family'))
            comparison_keys.append(key)
            families[row['task'], row['metric'], row['family']].append(row)
            a = number(summary[row['task'], row['method']][row['metric']])
            b = number(summary[row['task'], row['baseline']][row['metric']])
            need(close(number(row['method_point']), a) and close(number(row['baseline_point']), b), 'comparison point mapping')
            if row['status'] == 'ok':
                need(a is not None and b is not None and close(number(row['difference']), a-b), 'paired signed difference')
                need(0 <= number(row['p']) <= 1 and number(row['ci95_low']) <= number(row['ci95_high']), 'valid paired interval/p')
            else:
                need(row['status'] == 'unavailable' and number(row['p']) is None
                     and number(row['difference']) is None, 'unavailable comparison retained without numeric fabrication')
        need(len(comparison_keys) == len(set(comparison_keys)), 'no duplicate or selectively repeated comparisons')
        def metric_names(task):
            quality = ['CIDEr', 'SPICE', 'CHAIRs', 'CHAIRi'] if task == 'coco' else ['accuracy']
            if task == 'mme':
                quality += ['MME_subset_score', 'Perception_subset_score', 'Cognition_subset_score', 'acc_plus']
            return quality + ['mean_seconds', 'mean_decode_seconds', 'throughput_tokens_per_second',
                              'decode_throughput_tokens_per_second', 'mean_peak_allocated_bytes',
                              'effective_mean_accepted_draft_length']
        for task in TASKS:
            metrics = metric_names(task)
            for metric in metrics:
                for family, expected in (('BEPC_six_prespecified', cfg['comparisons']),
                                         ('LV_adaptation_two_prespecified', cfg['secondary_comparisons'])):
                    batch = families[task, metric, family]
                    need(len(batch) == len(expected) and {(r['method'], r['baseline']) for r in batch} == {tuple(pair) for pair in expected}, 'complete prespecified family incl unavailable tests')
                    ordered = sorted(batch, key=lambda r: 1. if number(r['p']) is None else number(r['p']))
                    running = 0.
                    for rank, own in enumerate(ordered):
                        p = 1. if number(own['p']) is None else number(own['p'])
                        running = max(running, min(1., (len(ordered)-rank)*p))
                        expected_adjusted = None if number(own['p']) is None else running
                        need(close(number(own['holm_adjusted_p']), expected_adjusted), 'independent Holm family correction')
        expected_family_keys = {(task, metric, family)
            for task in TASKS
            for metric in metric_names(task)
            for family in ('BEPC_six_prespecified', 'LV_adaptation_two_prespecified')}
        need(set(families) == expected_family_keys, 'no unregistered family or missing task/metric')
        return dict(analysis_sha256=sha(path), bootstrap=5000, output_files=len(report['output_sha256']),
                    summaries=len(summaries), comparisons=len(comparisons), independently_reconstructed_draw_sha=True,
                    scope='Checks source/input/output seals, full grouped draws, point estimates and Holm adjustments; does not claim independent replication of every interval or raw bootstrap p')

    def failures(self):
        files = sorted(set(ROOT.glob('results/**/*.failure.json')) | set(ROOT.glob('metrics/**/metric_failure.json')))
        inventory = [dict(path=str(path.relative_to(ROOT)), sha256=sha(path), payload=read(path)) for path in files]
        self.evidence['failure_files'] = inventory
        need(not files, 'failed requests/scorers retained; investigate every failure before claiming clean delivery')
        return dict(retained_failure_files=inventory)

    def determinism(self):
        need(not self.repeat_differences, 'repeat differences retained in report; deterministic quality No-Go')
        need(not self.equivalence_differences, 'native/shared or reference parity differences retained; equivalence No-Go')
        return dict(repeat_differences=0, equivalence_differences=0)

    def run(self, phase):
        self.check('fresh_source_and_readonly_adapter_provenance', self.sources)
        phases = ['parity', 'development', 'main'] if phase == 'all' else [phase] if phase != 'preflight' else []
        config_phases = ['development', 'main'] if 'main' in phases else ['development']
        for item in config_phases:
            self.check(item+'_configuration_and_checkpoint_identity', lambda item=item: self.configuration(item))
        if self.configs:
            self.check('data_images_prompts_labels_history_and_groups', self.data)
        for item in phases:
            if item not in self.configs and 'development' in self.configs:
                self.configs[item] = self.configs['development']
            if item in self.configs:
                self.check(item+'_actual_imports_and_exclusive_gpu_logs', lambda item=item: self.loaded(item))
                self.check(item+'_complete_runtime_grid_and_native_semantics', lambda item=item: self.phase(item))
        scores = {}
        for item in ('development', 'main'):
            if item in self.phase_rows:
                own = self.check(item+'_official_metrics_and_score_seals', lambda item=item: self.metrics(item))
                if own is not None:
                    scores[item] = own
        if 'development' in scores and (ROOT/'configs/frozen_config.json').exists():
            self.check('independent_development_only_lvspec_selection', lambda: self.selection(scores['development']))
        if 'main' in self.phase_rows:
            self.check('grouped_statistical_analysis_and_output_seals', self.analysis)
        self.check('all_failures_retained_and_resolved', self.failures)
        self.check('repeat_determinism_and_claimed_equivalence', self.determinism)
        self.check('strict_project_unseen_image_claim', self.strict_freshness, required=False)
        self.evidence.update(repeat_differences=self.repeat_differences,
                             equivalence_differences=self.equivalence_differences)
        if phase == 'all' and len(self.phase_rows) == 3:
            self.check('grand_request_total', lambda: self.grand_total())
        passed = all(item['passed'] for item in self.checks if item['required_for_execution_integrity'])
        return dict(status='pass' if passed else 'No-Go', requested_phase=phase,
                    execution_and_data_integrity_status='pass' if passed else 'No-Go',
                    strict_project_history_unseen_claim_status='No-Go' if self.freshness_findings else 'not_established',
                    independent_cpu_audit=True, gpu_or_checkpoint_loaded=False,
                    judge_or_training=False, validator_sha256=sha(__file__),
                    checks=self.checks, evidence=self.evidence, warnings=self.warnings,
                    limitations=['MME history reuse limits unseen-image generalization.',
                      'Seven additional non-MME aliases are historically seen; the strict all-project-unseen claim remains No-Go.',
                      '20/100-image subsets and automatic metrics do not establish population noninferiority or human factual safety.',
                      'Checkpoint bytes use a preserved prior SHA manifest plus current stat tuple; this audit does not rehash all checkpoint shards.',
                      'Production ancestry correctness relies on sealed source, CPU contracts and explicit small GPU parity; full hidden-state replay is not rerun.',
                      'This validator audits the delivered bootstrap inputs/draw seals; a second implementation of every confidence interval is not claimed.'])

    def strict_freshness(self):
        self.evidence['strict_project_freshness_findings'] = self.freshness_findings
        need(not self.freshness_findings, 'Strict all-project-unseen condition is No-Go; declared historical overlaps retained without resampling')
        return dict(status='no_known_overlap')

    def grand_total(self):
        measured = sum(len(rows) for phase in self.phase_rows.values() for rows in phase.values())
        warmups = sum(read(ROOT/f'audit/{phase}_shard{shard}_{task}.warmup.json')['requests']
                      for phase in ('parity', 'development', 'main') for shard in (0, 1) for task in TASKS)
        need(measured == 15960 and warmups == 400 and measured+warmups == 16360, '16360 total requests if no retries')
        return dict(measured=measured, warmup=warmups, total=measured+warmups)


def self_test():
    """Metric-free arithmetic tests: valid native trace and three corruptions."""
    import copy
    sample = dict(task='coco', question_id='q', image_id=1, image_path='fixture',
                  image_sha256='a'*64, group_id='g', group_ordinal=0, ordinal=0, prompt_text='caption')
    cfg = dict(eos_token_ids=[9], max_new_tokens=128, tree=dict(depth=3))
    variant = dict(family='vispec', compact_logits=False, terminal_skip=False)
    row = dict(sample, method='vispec', phase='main', temperature=0., shard=0, physical_gpu='0', gpu='0',
        diagnostic_recording=False, prompt_token_ids=[1], prompt_tokens=1, image_grid_thw=[[1, 4, 4]],
        input_tensor_sha256='b'*64, raw_token_ids=[2, 9], acceptance_length=[1], native_new_tokens=2,
        native_index=0, returned_token_ids=[2], eos_index=1, cap_reached=False, completion_tokens=2,
        effective_accepted_draft_tokens=1, effective_verification_rounds=1, model_calls=dict(target=2, draft=8, vision=1),
        extra_target_forwards=0, signal_stats=dict(terminal_update=dict(enabled=False)),
        request_lifecycle=dict(immediate_release_checked=True, live_controllers_after_request=0,
            live_logits_after_request=0, explicit_gc_or_empty_cache=False),
        seconds=1., decode_seconds=.5, baseline_allocated_bytes=10, baseline_reserved_bytes=20,
        peak_allocated_bytes=30, peak_reserved_bytes=40, peak_increment_bytes=20, peak_reserved_increment_bytes=20,
        preprocessing_reused_for_paired_requests=True, input_preprocessing_seconds=.1, acceptance_paths=[])
    row_contract(row, sample, variant, cfg, 'main', 0)
    checks = 1
    for key, replacement in [('returned_token_ids', [2, 9]), ('model_calls', dict(target=3, draft=8, vision=1)),
                             ('request_lifecycle', dict(row['request_lifecycle'], live_logits_after_request=1))]:
        broken = copy.deepcopy(row)
        broken[key] = replacement
        try:
            row_contract(broken, sample, variant, cfg, 'main', 0)
        except AssertionError:
            checks += 1
        else:
            raise AssertionError('Validator failed to reject corrupted '+key)
    compact_variant = dict(family='REFINED_BEPC', compact_logits=True, terminal_skip=True)
    compact = copy.deepcopy(row)
    compact.update(method='bepc_algorithm', raw_token_ids=[2, 3, 4, 9], returned_token_ids=[2, 3, 4],
        native_new_tokens=4, acceptance_length=[3], eos_index=3, completion_tokens=4,
        effective_accepted_draft_tokens=3, model_calls=dict(target=2, draft=4, vision=1),
        acceptance_paths=[dict(round=0, row=0, accepted_draft=3, mismatches=2,
                              response_mismatches_before=0, nodes=[0, 1, 2, 3], tokens=[2, 3, 4, 9])],
        signal_stats=dict(round_mismatch_budget=2, response_mismatch_budget=4, committed_response_mismatches=2,
            compact_logits=dict(native_argmax_reductions=1, audit_reference_materializations=0),
            terminal_update=dict(update_calls=1, terminal_draft_skips=1, eos_terminal_skips=1,
                budget_terminal_skips=0, original_update_KV_and_sample_p_preserved=True, additional_target_forwards=0)))
    row_contract(compact, sample, compact_variant, cfg, 'main', 0)
    checks += 1
    for corrupt_budget in (True, False):
        broken = copy.deepcopy(compact)
        if corrupt_budget:
            broken['acceptance_paths'][0]['mismatches'] = 3
        else:
            broken['model_calls']['draft'] = 8
        try:
            row_contract(broken, sample, compact_variant, cfg, 'main', 0)
        except AssertionError:
            checks += 1
        else:
            raise AssertionError('Validator missed mismatch-budget or terminal-tree accounting corruption')
    print(json.dumps(dict(self_test='pass', contracts=checks, gpu_imported='torch' in sys.modules)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('preflight', 'parity', 'development', 'main', 'all'), default='all')
    parser.add_argument('--output', type=Path, default=ROOT/'audit/FINAL_DELIVERY_VALIDATION.json')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--verify-history-bytes', action='store_true', help='Also rehash19.43GB of historical exclusion sources; avoid during GPU timing')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    result = Audit(verify_history_bytes=args.verify_history_bytes).run(args.phase)
    output = args.output.resolve()
    need(output.is_relative_to(ROOT.resolve()), 'validator output must stay in new benchmark root')
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n'
    if output.exists():
        need(output.read_text(encoding='utf-8') == encoded, 'Preserve previous audit; use a new --output for a changed audit')
    else:
        output.write_text(encoded, encoding='utf-8')
    markdown = ['# Independent delivery validation', '', f"Status: **{result['status']}**. Phase: `{args.phase}`.", '',
                '| Check | Result |', '|---|---|']
    markdown += [f"| {entry['name']} | {'PASS' if entry['passed'] else 'No-Go: '+entry['exception'].replace('|', '/')} |" for entry in result['checks']]
    markdown += ['', 'Repeat and equivalence differences, errors and source seals are retained in the adjacent JSON.', '',
                 *['- '+item for item in result['warnings']+result['limitations']], '']
    md_path, md = output.with_suffix('.md'), '\n'.join(markdown)
    if md_path.exists():
        need(md_path.read_text(encoding='utf-8') == md, 'Preserve previous Markdown audit')
    else:
        md_path.write_text(md, encoding='utf-8')
    print(json.dumps(dict(status=result['status'], checks=len(result['checks']),
                         required_failures=sum(not entry['passed'] and entry['required_for_execution_integrity'] for entry in result['checks']),
                         strict_project_history_unseen_claim_status=result['strict_project_history_unseen_claim_status'], output=str(output))))
    if result['status'] != 'pass':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
