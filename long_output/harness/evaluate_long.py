"""Frozen-format long-answer adapter around the sealed official task scorers.

Only the terminal ``Final answer: ...`` line supplies a QA answer. No learned
extraction, answer search, gold-assisted parsing, fallback guess or test tuning
occurs. All generated metric files and writable locks stay in long_output.
The old benchmark, labels and official metric dependencies remain read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BASE_ROOT = ROOT.parent
OFFICIAL_SOURCE = BASE_ROOT / 'harness/evaluate_tasks.py'
OFFICIAL_SHA256 = '1f491b0f1dba0a148b5a65ddc574c73c76878c098c967fc1f45fc08ba2a83fc3'
LABEL_CONFIG = BASE_ROOT / 'configs/frozen_config.json'
LABEL_CONFIG_SHA256 = '5c582799d4b0098ed8334fc5e80f753c73896783f406c93f07c3c6c57e47b188'
PARSER_VERSION = 'terminal_final_answer_v1'
_OFFICIAL = None


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def official():
    global _OFFICIAL
    if sha(OFFICIAL_SOURCE) != OFFICIAL_SHA256:
        raise ValueError('Sealed original official task evaluator source changed')
    if _OFFICIAL is None:
        spec = importlib.util.spec_from_file_location('_sealed_long_official_tasks', OFFICIAL_SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # ROOT alone controls output restriction, lock and label lookup. All
        # official source/dependency constants were resolved at original load.
        # Gold is explicitly joined below; no old output/cache is imported.
        _OFFICIAL = module
    _OFFICIAL.ROOT = ROOT
    return _OFFICIAL


def canonical_task(task):
    name = str(task).lower().replace('_', '').replace('-', '')
    name = {'sqa': 'scienceqa', 'cococaption': 'coco', 'cococaptions': 'coco'}.get(name, name)
    if name not in ('mme', 'gqa', 'textvqa', 'scienceqa', 'coco'):
        raise ValueError('Unsupported task')
    return name


def extract_final_answer(task, text):
    """Formatting only. Return answer, status, reason; no access to labels."""
    task = canonical_task(task)
    if not isinstance(text, str):
        raise TypeError('Output must be a string')
    if task == 'coco':
        return dict(answer=text, status='not_applicable', reason='full_description_scored_unchanged')
    lines = text.splitlines()
    nonempty = [index for index, line in enumerate(lines) if line.strip()]
    matches = []
    for index, line in enumerate(lines):
        match = re.fullmatch(r'[ \t]*Final[ \t]+answer[ \t]*:[ \t]*(.*?)[ \t]*', line, re.IGNORECASE)
        if match:
            matches.append((index, match.group(1).strip()))
    if not matches:
        return dict(answer='', status='invalid', reason='missing_standalone_final_answer_line')
    if len(matches) != 1:
        return dict(answer='', status='uncertain', reason='multiple_final_answer_lines')
    index, answer = matches[0]
    if not nonempty or index != nonempty[-1]:
        return dict(answer='', status='invalid', reason='final_answer_is_not_last_nonempty_line')
    if not answer:
        return dict(answer='', status='invalid', reason='empty_final_answer')
    if task == 'mme' and not re.fullmatch(r'(?:yes|no)[.!?]?', answer, re.IGNORECASE):
        return dict(answer='', status='invalid', reason='mme_final_answer_not_yes_or_no')
    if task == 'scienceqa' and not re.fullmatch(r'(?:[A-Za-z]|\([A-Za-z]\)|\[[A-Za-z]\])[.!?]?', answer):
        return dict(answer='', status='invalid', reason='scienceqa_final_answer_not_single_option')
    return dict(answer=answer, status='parsed', reason='unique_terminal_final_answer_line')


def attach_labels(task, samples):
    """Read the same SHA-sealed gold; never modify generation inputs."""
    module = official()
    samples = [dict(sample) for sample in samples]
    expected = {'scienceqa': ('answer_index', 'answer', 'gold_answer', 'gold', 'label'),
                'textvqa': ('answers', 'gt_answers', 'gold_answers', 'gold'),
                'coco': ('references', 'captions', 'sentences'),
                'gqa': ('answer', 'gold_answer', 'gold', 'label'),
                'mme': ('answer', 'gold_answer', 'gold', 'label')}[task]
    if all(any(key in sample for key in expected) for sample in samples):
        return samples, dict(mode='explicit labeled scoring samples', labels_used_for_generation=False)
    if sha(LABEL_CONFIG) != LABEL_CONFIG_SHA256:
        raise ValueError('Original task/label configuration seal changed')
    config = read(LABEL_CONFIG)
    ids = {module.sample_id(sample) for sample in samples}
    matches = []
    for split in ('dev', 'test'):
        path = Path(config['data'][task][split+'_labels_path'])
        expected_sha = config['data'][task][split+'_labels_sha256']
        if sha(path) != expected_sha:
            raise ValueError('Original official label file changed: '+str(path))
        labels = read(path)
        if ids <= set(labels):
            matches.append((path, labels, split, expected_sha))
    if len(matches) != 1:
        raise ValueError('Samples do not match exactly one sealed official-label split')
    path, labels, split, expected_sha = matches[0]
    fields = {'answer', 'answer_index', 'answer_letter', 'answers', 'choices', 'references',
              'captions', 'sentences', 'category', 'cocoid'}
    for sample in samples:
        label = labels[module.sample_id(sample)]
        if sample.get('image_sha256') is not None and label.get('image_sha256') != sample['image_sha256']:
            raise ValueError('Official label image SHA does not match generation image')
        for key in fields & set(label):
            if key in sample and sample[key] != label[key]:
                raise ValueError('Official label conflicts with scoring sample: '+key)
            sample[key] = label[key]
    return samples, dict(mode='post-generation join of original sealed separate official-label file',
        split=split, path=str(path), sha256=expected_sha, labels_used_for_generation=False,
        label_config_sha256=LABEL_CONFIG_SHA256)


def score_task(task, samples, predictions, output_dir):
    """Drop-in score_task API; all failures remain in the task denominator."""
    task = canonical_task(task)
    module = official()
    samples, raw_texts = module.align(samples, predictions)
    samples, label_provenance = attach_labels(task, samples)
    destination = Path(output_dir).resolve()
    if not destination.is_relative_to(ROOT.resolve()):
        raise ValueError('Long-output scoring must stay within long_output')
    destination.mkdir(parents=True, exist_ok=True)
    extracted = [extract_final_answer(task, text) for text in raw_texts]
    signature = hashlib.sha256(json.dumps(dict(task=task, samples=samples, raw_texts=raw_texts,
        parser=PARSER_VERSION, official_evaluator_sha256=OFFICIAL_SHA256),
        sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    signature_path = destination/'metric_inputs.sha256'
    if signature_path.exists() and signature_path.read_text().strip() != signature:
        raise ValueError('Existing long-output metric files have different raw inputs; use another output_dir')
    signature_path.write_text(signature+'\n', encoding='utf-8')
    rows, aggregate = module.score_task(task, samples, [entry['answer'] for entry in extracted],
                                       destination/'official_scoring')
    for row, raw, entry in zip(rows, raw_texts, extracted):
        row['raw_output_text'] = raw
        row['extracted_answer'] = entry['answer'] if task != 'coco' else None
        row['answer_extraction_status'] = entry['status']
        row['answer_extraction_reason'] = entry['reason']
        row['official_parse_status'] = row['parse_status']
        row['official_parse_reason'] = row['parse_reason']
        row['prediction'] = raw
        if entry['status'] in ('invalid', 'uncertain'):
            if row['score'] != 0:
                raise AssertionError('Unextractable QA answer must score0')
            row.update(parse_status=entry['status'], parse_invalid=entry['status'] == 'invalid',
                       uncertain=entry['status'] == 'uncertain', parse_reason=entry['reason'])
    counts = Counter(entry['status'] for entry in extracted)
    aggregate.update(input_sha256=signature, official_label_join=label_provenance,
        parse_invalid_count=sum(row['parse_invalid'] for row in rows),
        uncertain_count=sum(row['uncertain'] for row in rows),
        long_answer_parser=PARSER_VERSION, long_answer_adapter_sha256=sha(__file__),
        official_evaluator_sha256=OFFICIAL_SHA256,
        answer_extraction_counts=dict(counts),
        answer_extraction_failures=sum(entry['status'] in ('invalid', 'uncertain') for entry in extracted),
        answer_extraction_failure_policy='score0 and retain denominator; no fallback, gold lookup or learned extraction',
        quality_scope=('full long description; automatic reference/object metrics, not human factual ground truth'
                       if task == 'coco' else 'extracted final-answer task accuracy; explanation quality not assessed'),
        automatic_judge_used=False, all_questions_retained=True)
    write(destination/'answer_extraction.json', [dict(question_id=module.sample_id(sample),
        raw_output_text=raw, **entry) for sample, raw, entry in zip(samples, raw_texts, extracted)])
    write(destination/'per_question_scores.json', rows)
    write(destination/'aggregate.json', aggregate)
    return rows, aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=['mme', 'gqa', 'textvqa', 'scienceqa', 'coco'])
    parser.add_argument('--samples', required=True, type=Path)
    parser.add_argument('--predictions', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    content = args.predictions.read_text(encoding='utf-8')
    try:
        predictions = json.loads(content)
    except json.JSONDecodeError:
        predictions = [json.loads(line) for line in content.splitlines() if line.strip()]
    _, aggregate = score_task(args.task, read(args.samples), predictions, args.output_dir)
    print(json.dumps(aggregate, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
