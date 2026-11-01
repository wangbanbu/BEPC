"""CPU-only scoring of the isolated BEPC response-cap ablation.

The previous evaluator, final-answer parser, official task implementations and
gold files are imported read-only. Only the SPICE Java heap changes to 64 GiB;
its scorer, Stanford models, inputs, options and scoring equations do not change.
Exact whole-corpus score reuse is audited, never inferred from method names.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT.parent / 'single_pass'
ADAPTER_SOURCE = PREVIOUS / 'harness/evaluate_long.py'
# The exact read-only source seal is captured by freeze() before any scoring.
METHODS = ('strict_shared', 'bepc_cap4', 'bepc_no_response_cap')
OPTIMIZATION = ROOT.parent / 'bepc_optimization_20261101'
TASKS = ('scienceqa', 'coco', 'textvqa')
_ADAPTER = None


def require(value, message):
    if not value:
        raise AssertionError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists():
        require(path.read_text(encoding='utf-8') == payload, 'Preserve existing evidence: ' + str(path))
    else:
        path.write_text(payload, encoding='utf-8')


def sample_id(sample):
    return str(sample.get('question_id', sample.get('sample_id')))


def prediction_sha(predictions):
    return hashlib.sha256(json.dumps(predictions, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def freeze():
    source = ROOT.parent.parent / 'harness/evaluate_tasks.py'
    parent_adapter = ROOT.parent / 'harness/evaluate_long.py'
    payload = dict(
        status='pre_quality_scoring_implementation_freeze',
        sources={str(path): sha(path) for path in (Path(__file__), ROOT / 'harness/analyze_uncapped.py',
            ADAPTER_SOURCE, PREVIOUS / 'harness/analyze_results.py', source, parent_adapter)},
        parser='terminal_final_answer_v1 unchanged',
        tasks=list(TASKS), methods=list(METHODS),
        spice_change='Java maximum heap 16G -> 64G only; same official scorer/models/options',
        score_reuse='only exact ordered sample corpus and exact raw output text; whole-corpus CIDEr DF preserved',
        no_test_quality_parameter_selection=True,
    )
    write_once(ROOT / 'audit/METRIC_IMPLEMENTATION_FREEZE.json', payload)
    return payload


def load_adapter():
    global _ADAPTER
    frozen = read(ROOT / 'audit/METRIC_IMPLEMENTATION_FREEZE.json')
    for path, expected in frozen['sources'].items():
        require(sha(path) == expected, 'Frozen metric source changed: ' + path)
    if _ADAPTER is None:
        spec = importlib.util.spec_from_file_location('_read_only_previous_long_evaluator', ADAPTER_SOURCE)
        _ADAPTER = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_ADAPTER)
    _ADAPTER.ROOT = ROOT
    official = _ADAPTER.official()
    official._spice = spice64
    return _ADAPTER


def spice64(references, predicted, output_dir):
    """Same official SPICE invocation/parsing with a larger resource ceiling."""
    official = _ADAPTER.official()
    package = official.COCO_DEPS / 'pycocoevalcap/spice'
    jar = package / 'spice-1.0.jar'
    folder = Path(output_dir) / 'spice'
    require(folder.resolve().is_relative_to(ROOT.resolve()), 'SPICE output scope')
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'cache').mkdir(exist_ok=True)
    payload = [dict(image_id=index, test=predicted[index][0], refs=references[index]) for index in references]
    input_path, output_path = folder / 'input.json', folder / 'output.json'
    write_once(input_path, payload)
    identity = sha(input_path)
    signature = folder / 'input.sha256'
    command = ['java', '-XX:ActiveProcessorCount=8', '-Xmx64G', '-cp',
               os.pathsep.join([str(official.SPICE_MODELS / '*'), str(jar), str(package / 'lib/*')]),
               'edu.anu.spice.SpiceScorer', str(input_path), '-cache', str(folder / 'cache'),
               '-out', str(output_path), '-subset', '-silent', '-threads', '1']
    reused = output_path.exists() and signature.exists() and signature.read_text().strip() == identity
    if not reused:
        import fcntl
        lock_path = ROOT / 'metrics/.spice.lock'
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        started = datetime.now(timezone.utc).isoformat()
        import time
        clock = time.monotonic()
        with lock_path.open('a', encoding='utf-8') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with (folder / 'scorer.log').open('w', encoding='utf-8') as log:
                subprocess.run(command, cwd=folder, stdout=log, stderr=subprocess.STDOUT, check=True)
        log = (folder / 'scorer.log').read_text(encoding='utf-8', errors='replace')
        require(not re.search(r'OutOfMemoryError|NullPointerException|Will ignore and continue|Could not score batched file', log),
                'SPICE scoring failure/ignored sentence; no signature sealed: ' + str(folder))
        require(output_path.exists(), 'Official SPICE output missing')
        # LMDB cache persistence warnings do not skip in-memory parsed tuples.
        write_once(folder / 'RESOURCE_EXECUTION.json', dict(
            command=command, start_utc=started, elapsed_seconds=time.monotonic() - clock,
            input_sha256=identity, output_sha256=sha(output_path), log_sha256=sha(folder / 'scorer.log'),
            ignored_sentence_or_java_failure=False,
            cache_persistence_warnings=log.count('Could not cache item'),
            cache_warning_scope='author Java LMDB persistence warning; not sentence omission',
        ))
        signature.write_text(identity + '\n', encoding='utf-8')
    values = read(output_path)
    scores = {int(value['image_id']): value['scores'] for value in values}
    require(set(scores) == set(references) and len(scores) == len(values), 'SPICE output IDs incomplete/duplicate')
    result = {}
    for key, value in scores.items():
        try:
            number = float(value['All']['f'])
        except (TypeError, ValueError):
            number = float('nan')
        result[key] = number if math.isfinite(number) else None
    classpath_files = sorted(set(list(official.SPICE_MODELS.glob('*.jar')) + [jar] + list((package / 'lib').glob('*.jar'))))
    return result, dict(
        command=command, input_sha256=identity, output_sha256=sha(output_path),
        spice_jar_sha256=sha(jar), wrapper_sha256=sha(package / 'spice.py'),
        classpath_sha256={str(path): sha(path) for path in classpath_files},
        cache_scope='new experiment/output only; no historical writable Java cache imported',
        exact_same_input_result_reused=reused, java_heap_change_only=True,
        undefined_aggregate_policy='preserve null; any undefined item makes aggregate undefined',
    )


def load_phase(config, config_path, phase):
    require(config['repetitions'] == 1, 'Single pass only')
    require(set(config['tasks']) == set(TASKS), 'Only three requested datasets')
    split = 'test' if phase == 'main' else 'dev'
    expected_groups = 100 if phase == 'main' else 20
    samples, all_rows, result_hashes = {}, {}, {}
    config_sha = sha(config_path)
    for name, expected in config.get('harness_sha256', {}).items():
        require(sha(ROOT / 'harness' / name) == expected, 'Frozen inference source changed: ' + name)
    for task in config['tasks']:
        data = config['data'][task]
        for suffix in (split + '_path', split + '_labels_path'):
            require(sha(data[suffix]) == data[suffix.replace('_path', '_sha256')], 'Frozen data/label SHA: ' + task)
        samples[task] = read(data[split + '_path'])
        require(len(samples[task]) == expected_groups and len({sample['group_id'] for sample in samples[task]}) == expected_groups,
                'Expected one question per independent group: ' + task)
        methods = config['methods_by_task'][task] if phase == 'main' else config['development_methods']
        require(set(methods) == set(METHODS), 'Fixed three method arms: ' + task)
        rows = []
        for shard in (0, 1):
            path = ROOT / 'results' / phase / f'{phase}_{task}_shard{shard}.jsonl'
            marker = read(path.with_suffix('.complete.json'))
            require(marker['complete'] is True and marker['result_sha256'] == sha(path), 'Completed result seal: ' + str(path))
            batch = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            require(marker['records'] == len(batch), 'Completed record count')
            require(marker['metadata']['config_sha256'] == config_sha, 'Completed configuration SHA')
            require(marker['metadata']['source_sha256'] == config['harness_sha256'], 'Completed source identity')
            for row in batch:
                require(row['task'] == task and row['phase'] == phase and row['config_sha256'] == config_sha,
                        'Row task/phase/config')
                require(row['shard'] == shard == row['group_ordinal'] % 2, 'Physical group shard')
                require(row['diagnostic_recording'] is False and row['temperature'] == 0, 'Production T=0 only')
                sample = samples[task][row['ordinal']]
                require(sample_id(sample) == str(row['question_id']) and sample['group_id'] == row['group_id']
                        and sample['image_sha256'] == row['image_sha256'], 'Row sample identity')
                require(math.isfinite(row['seconds']) and row['seconds'] > 0, 'Finite positive request timing')
                require(row['extra_target_forwards'] == 0, 'No additional Target forwards')
            rows.extend(batch)
            result_hashes[str(path)] = sha(path)
        actual = [(row['method'], row['ordinal'], row['repeat']) for row in rows]
        expected = {(method, ordinal, 0) for method in methods for ordinal in range(len(samples[task]))}
        require(len(actual) == len(set(actual)) and set(actual) == expected, 'Complete unique method/question grid: ' + task)
        for ordinal in range(len(samples[task])):
            paired = [row for row in rows if row['ordinal'] == ordinal]
            for key in ('prompt_token_ids', 'input_tensor_sha256', 'image_grid_thw', 'prompt_text'):
                require(all(row[key] == paired[0][key] for row in paired), 'Paired input equality: ' + key)
        all_rows[task] = rows
    return samples, all_rows, result_hashes


def reusable_corpus(task, phase, samples, predictions, destination):
    """Whole-corpus reuse only: protects CIDEr's corpus document frequencies."""
    candidates = []
    local = ROOT / 'metrics' / phase / task
    if local.exists():
        candidates.extend(path.parent for path in sorted(local.glob('*/SCORING_SEAL.json')))
    candidates.extend(OPTIMIZATION / 'metrics' / phase / task / method
                      for method in ('strict_shared', 'bepc_original', 'bepc_fast', 'bepc_rescue0625'))
    candidates.extend(PREVIOUS / 'metrics' / phase / task / method
                      for method in ('strict_shared', 'bepc_algorithm'))
    adapter = load_adapter()
    labeled, _ = adapter.attach_labels(task, samples)
    ordered = [predictions[sample_id(sample)] for sample in samples]
    signature = hashlib.sha256(json.dumps(dict(task=task, samples=labeled, raw_texts=ordered,
        parser=adapter.PARSER_VERSION, official_evaluator_sha256=adapter.OFFICIAL_SHA256),
        sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    for folder in candidates:
        if folder == destination or not (folder / 'SCORING_SEAL.json').exists():
            continue
        seal = read(folder / 'SCORING_SEAL.json')
        for key, name in (('aggregate_sha256', 'aggregate.json'), ('per_question_sha256', 'per_question_scores.json')):
            require(sha(folder / name) == seal[key], 'Source score seal mutated: ' + str(folder))
        aggregate = read(folder / 'aggregate.json')
        if aggregate.get('input_sha256') != signature:
            continue
        rows = read(folder / 'per_question_scores.json')
        require(len(rows) == len(samples) and [str(row['question_id']) for row in rows] == [sample_id(sample) for sample in samples],
                'Reuse ordered IDs differ')
        require([row.get('raw_output_text') for row in rows] == ordered, 'Reuse raw text must match exactly')
        return rows, aggregate, dict(mode='exact_whole_corpus_official_score_reuse', source=str(folder),
            source_seal_sha256=sha(folder / 'SCORING_SEAL.json'),
            source_aggregate_sha256=sha(folder / 'aggregate.json'),
            source_per_question_sha256=sha(folder / 'per_question_scores.json'),
            raw_text_equal=True, labeled_ordered_corpus_signature=signature,
            source_is_historical=not folder.is_relative_to(ROOT),
            source_runtime_timings_reused=False, CIDEr_complete_DF_corpus_unchanged=True)
    return None


def score_phase(config, config_path, phase, samples=None, rows_by_task=None, result_hashes=None):
    if samples is None:
        samples, rows_by_task, result_hashes = load_phase(config, config_path, phase)
    adapter = load_adapter()
    scored = {}
    for task in config['tasks']:
        scored[task] = {}
        methods = config['methods_by_task'][task] if phase == 'main' else config['development_methods']
        for method in methods:
            selected = sorted((row for row in rows_by_task[task] if row['method'] == method), key=lambda row: row['ordinal'])
            predictions = {str(row['question_id']): row['output_text'] for row in selected}
            folder = ROOT / 'metrics' / phase / task / method
            seal_path = folder / 'SCORING_SEAL.json'
            expected = dict(config_sha256=sha(config_path), task=task, method=method, phase=phase,
                evaluator_sha256=sha(__file__), previous_evaluator_sha256=sha(ADAPTER_SOURCE),
                official_evaluator_sha256=adapter.OFFICIAL_SHA256,
                metric_implementation_freeze_sha256=sha(ROOT / 'audit/METRIC_IMPLEMENTATION_FREEZE.json'),
                result_sha256={p: value for p, value in result_hashes.items() if f'_{task}_shard' in p},
                predictions_sha256=prediction_sha(predictions), quality_repeat=0)
            if seal_path.exists():
                seal = read(seal_path)
                require(all(seal[key] == value for key, value in expected.items()), 'Existing score inputs changed')
                require(sha(folder / 'aggregate.json') == seal['aggregate_sha256'] and
                        sha(folder / 'per_question_scores.json') == seal['per_question_sha256'], 'Existing score content changed')
                rows, aggregate = read(folder / 'per_question_scores.json'), read(folder / 'aggregate.json')
            else:
                reuse = reusable_corpus(task, phase, samples[task], predictions, folder)
                if reuse is None:
                    rows, aggregate = adapter.score_task(task, samples[task], predictions, folder)
                    provenance = dict(mode='new_official_score', spice_heap='64G' if task == 'coco' else None,
                        parser_unchanged=True, official_scorer_unchanged=True)
                else:
                    rows, aggregate, provenance = reuse
                    folder.mkdir(parents=True, exist_ok=True)
                    write_once(folder / 'per_question_scores.json', rows)
                    write_once(folder / 'aggregate.json', aggregate)
                    write_once(folder / 'answer_extraction.json', [dict(question_id=row['question_id'],
                        raw_output_text=row['raw_output_text'], answer=row['extracted_answer'],
                        status=row['answer_extraction_status'], reason=row['answer_extraction_reason']) for row in rows])
                write_once(folder / 'SCORE_PROVENANCE.json', provenance)
                write_once(seal_path, dict(expected, aggregate_sha256=sha(folder / 'aggregate.json'),
                    per_question_sha256=sha(folder / 'per_question_scores.json'),
                    score_provenance_sha256=sha(folder / 'SCORE_PROVENANCE.json')))
            require(len(rows) == len(samples[task]) and len({str(row['question_id']) for row in rows}) == len(rows), 'Scored sample grid')
            scored[task][method] = dict(rows=rows, aggregate=aggregate, seal_sha256=sha(seal_path), folder=str(folder))
            print(json.dumps(dict(status='scored', phase=phase, task=task, method=method,
                quality={key: aggregate[key] for key in ('accuracy', 'SPICE', 'CHAIRs') if key in aggregate}), ensure_ascii=False), flush=True)
    return scored


def self_test():
    # No model, official metric execution, label access or generated outputs.
    spec = importlib.util.spec_from_file_location('_parser_self_test', ADAPTER_SOURCE)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    require(adapter.extract_final_answer('scienceqa', 'Reason.\nFinal answer: B')['answer'] == 'B', 'Frozen ScienceQA parser')
    require(adapter.extract_final_answer('scienceqa', 'Final answer: B\nMore')['status'] == 'invalid', 'Nonterminal answer invalid')
    require(adapter.extract_final_answer('textvqa', 'Final answer: X\nFinal answer: Y')['status'] == 'uncertain', 'Ambiguity retained')
    require(adapter.extract_final_answer('coco', 'Full caption.')['answer'] == 'Full caption.', 'Whole COCO text unchanged')
    require(prediction_sha({'b': '2', 'a': '1'}) == prediction_sha({'a': '1', 'b': '2'}), 'Canonical prediction identity')
    require(prediction_sha({'a': 'X'}) != prediction_sha({'a': 'x'}), 'No hidden output normalization')
    print('CPU parser/provenance tests passed; no generation/scoring performed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('dev', 'development', 'main'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.freeze:
        freeze()
        print('Metric implementation frozen before scoring')
        return
    require(args.phase is not None, '--phase required')
    if args.phase == 'dev':
        args.phase = 'development'
    path = args.config or ROOT / 'configs' / ('frozen_config.json' if args.phase == 'main' else 'design_config.json')
    score_phase(read(path), path, args.phase)


if __name__ == '__main__':
    main()
