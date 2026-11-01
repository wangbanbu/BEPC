"""Official task scoring, with explicit subset/answer-parser adaptations.

Public API: score_task(task, samples, predictions, output_dir) ->
    (per_question_rows: list[dict], aggregate: dict), both JSON serializable.
One call scores one method and one split; choose repeat0 before calling. Samples
may use sample_id/question_id/id, group_id/image_id, category, gold/answer,
answers(10 for TextVQA), choices(ScienceQA), sentences/references(COCO).
Predictions may be an ID->string/record mapping, aligned list of strings, or
records carrying the same ID and output_text/prediction/text/answer. Duplicate
or missing IDs fail; invalid answers remain score0 rather than being dropped.
No inference, Judge, learned extraction, random guessing or test tuning occurs.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import types

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
REPRO = WORKSPACE / 'vispec-official-reproduction'
MME_SOURCE = WORKSPACE / 'data/MME/eval_tool/calculation.py'
TEXTVQA_SOURCE = REPRO / 'research/harmful_mismatch/cache/literature/code/facebookresearch/mmf/mmf/utils/m4c_evaluators.py'
GQA_SOURCE = WORKSPACE / 'ViSpec/ccfa-workfiles/experiments/clean-verifier-comparison/reference/gqa_eval.py'
SCIENCEQA_SOURCE = WORKSPACE / 'ScienceQA/tools/evaluate_acc.py'
COCO_DEPS = WORKSPACE / 'ViSpec/ccfa-workfiles/experiments/lvspec-comparison/context-verifier/official-quality/cache/deps'
SPICE_MODELS = WORKSPACE / 'ViSpec/ccfa-workfiles/experiments/context-evidence-autoresearch/cache/spice-models'
CHAIR_BASE = REPRO / 'visual-margin-validation'
ANNOTATIONS = WORKSPACE / 'data/COCO/annotations'
MME_CATEGORIES = {
    'Perception': ('existence', 'count', 'position', 'color', 'posters', 'celebrity',
                   'scene', 'landmark', 'artwork', 'OCR'),
    'Cognition': ('commonsense_reasoning', 'numerical_calculation', 'text_translation', 'code_reasoning'),
}
SOURCE_URLS = {
    'mme': 'https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation',
    'gqa': 'https://cs.stanford.edu/people/dorarad/gqa/evaluate.html',
    'textvqa': 'https://github.com/facebookresearch/mmf/blob/main/mmf/utils/m4c_evaluators.py',
    'scienceqa': 'https://github.com/lupantech/ScienceQA/blob/main/tools/evaluate_acc.py',
    'cider': 'https://github.com/tylin/coco-caption',
    'spice': 'https://github.com/peteanderson80/SPICE',
    'chair': 'https://github.com/LisaAnne/Hallucination',
}
_MODULES = {}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, 'tolist'):
        return clean(value.tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def load_module(path, name):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Required original evaluation implementation unavailable: {path}')
    key = (str(path.resolve()), sha(path))
    if key not in _MODULES:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULES[key] = module
    return _MODULES[key]


def sample_id(row):
    for key in ('sample_id', 'question_id', 'questionId', 'id', 'qid', 'cocoid', 'image_id'):
        if key in row:
            return str(row[key])
    raise ValueError('Sample/prediction has no stable ID')


def prediction_text(value):
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        raise TypeError('Prediction must be text or an identified text record')
    for key in ('output_text', 'prediction', 'text', 'answer', 'pred_answer'):
        if key in value and isinstance(value[key], str):
            return value[key]
    raise ValueError('Prediction record has no textual output')


def align(samples, predictions):
    samples = list(samples)
    ids = [sample_id(row) for row in samples]
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError('Expected nonempty samples with unique question IDs')
    if isinstance(predictions, dict):
        mapping = {str(key): prediction_text(value) for key, value in predictions.items()}
    else:
        predictions = list(predictions)
        if all(isinstance(value, str) for value in predictions):
            if len(predictions) != len(samples):
                raise ValueError('Aligned prediction list length differs')
            mapping = dict(zip(ids, predictions))
        else:
            mapping = {}
            for row in predictions:
                identifier = sample_id(row)
                if identifier in mapping:
                    raise ValueError('Duplicate predicted question; select one method/repeat before scoring')
                mapping[identifier] = prediction_text(row)
    if set(mapping) != set(ids):
        raise ValueError(f'Prediction/sample ID mismatch: missing={len(set(ids)-set(mapping))}, extra={len(set(mapping)-set(ids))}')
    return samples, [mapping[identifier] for identifier in ids]


def _gold(sample, fields=('gold_answer', 'gold', 'answer', 'label')):
    for field in fields:
        if field in sample:
            return sample[field]
    raise ValueError('Sample has no ground-truth answer')


def _row(sample, prediction, score, parsed, status, reason=None):
    return dict(sample_id=sample_id(sample), question_id=sample.get('question_id', sample_id(sample)),
                image_id=sample.get('image_id', sample.get('cocoid')),
                group_id=sample.get('group_id', sample.get('image_id', sample_id(sample))),
                prediction=prediction, parsed_answer=parsed, score=float(score),
                parse_status=status, parse_invalid=status == 'invalid',
                uncertain=status == 'uncertain', parse_reason=reason)


def parse_scienceqa(text, number_of_choices):
    """Frozen formatting-only parser. Ambiguity stays uncertain; no guess."""
    if not 2 <= number_of_choices <= 26:
        raise ValueError('Invalid ScienceQA choices count')
    valid = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:number_of_choices])
    stripped = text.strip()
    # Explicit answers take precedence, but contradictory answer markers do
    # not select a convenient last answer or inspect the correct option.
    ambiguous_pair = re.search(r'(?i)\banswer\s*(?:is|:|=)\s*[\(\[]?([a-z])[\)\]]?\s*(?:or|and|/)\s*[\(\[]?([a-z])\b', stripped)
    if ambiguous_pair and len({value.upper() for value in ambiguous_pair.groups()}) > 1:
        return None, 'uncertain', 'multiple_options_in_explicit_answer'
    # A following word boundary is insufficient: the article in "answer is a
    # lion" is not an option marker. Require paired brackets, punctuation, or
    # the end of the answer line. "answer is A because ..." is deliberately
    # invalid without a delimiter; this is a fixed formatting-only rule.
    marked = re.findall(r'(?im)\b(?:(?:the|my)\s+)?(?:(?:final|correct)\s+)?answer\s*(?:is|:|=)\s*(?:\(([a-z])\)|\[([a-z])\]|([a-z])(?=[ \t]*(?:$|[.,:;!?])))', stripped)
    marked = [next(value for value in group if value).upper() for group in marked]
    if marked:
        unique = set(marked)
        if len(unique) > 1:
            return None, 'uncertain', 'conflicting_explicit_answer_markers'
        letter = marked[0]
        return (ord(letter)-65, 'parsed', 'explicit_answer_marker') if letter in valid else (None, 'invalid', 'out_of_range_option')
    bare = re.fullmatch(r'[\(\[]?([A-Za-z])[\)\]]?[\s.!:;]*', stripped)
    if bare:
        letter = bare[1].upper()
        return (ord(letter)-65, 'parsed', 'standalone_option_letter') if letter in valid else (None, 'invalid', 'out_of_range_option')
    prefix = re.match(r'^\s*(?:\(([A-Za-z])\)|\[([A-Za-z])\]|([A-Za-z])[.:])\s+', stripped)
    if prefix:
        line_labels = re.findall(r'(?m)^\s*(?:\(([A-Za-z])\)|\[([A-Za-z])\]|([A-Za-z])[.:])\s+', stripped)
        if len({next(value for value in label if value).upper() for label in line_labels}) > 1:
            return None, 'uncertain', 'multiple_leading_option_lines'
        letter = next(value for value in prefix.groups() if value).upper()
        return (ord(letter)-65, 'parsed', 'leading_delimited_option_letter') if letter in valid else (None, 'invalid', 'out_of_range_option')
    return None, 'invalid', 'no_unambiguous_option_letter'


def _score_scienceqa(samples, texts):
    rows = []
    for sample, text in zip(samples, texts):
        choices = sample.get('choices', sample.get('options'))
        if not isinstance(choices, list) or not 2 <= len(choices) <= 26:
            raise ValueError('ScienceQA requires original choices')
        gold = _gold(sample, ('answer_index', 'gold_answer', 'gold', 'answer', 'label'))
        if isinstance(gold, str) and len(gold) == 1 and gold.upper() in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            gold = ord(gold.upper())-65
        if isinstance(gold, bool) or not isinstance(gold, int) or not 0 <= gold < len(choices):
            raise ValueError('ScienceQA gold must be an in-range original option index')
        if not (sample.get('image') or sample.get('image_path') or sample.get('has_image')):
            raise ValueError('This benchmark restricts ScienceQA to image questions')
        parsed, status, reason = parse_scienceqa(text, len(choices))
        row = _row(sample, text, status == 'parsed' and parsed == gold, parsed, status, reason)
        row.update(gold_answer_index=gold, choices=len(choices))
        rows.append(row)
    return rows, dict(accuracy=sum(row['score'] for row in rows)/len(rows), primary_metric='accuracy',
                      official_metric='multichoice accuracy',
                      scope='frozen image-question subset; not full4241 ScienceQA test score',
                      parser='fixed format-only option parser; invalid/uncertain count0; no random fallback')


def _score_gqa(samples, texts):
    rows = []
    for sample, text in zip(samples, texts):
        gold = _gold(sample)
        if not isinstance(gold, str):
            raise ValueError('GQA requires official textual answer')
        # GQA official evaluator compares supplied prediction and gold exactly.
        # No VQA normalization, lowercase, punctuation removal or LLM parser.
        status = 'parsed' if text else 'invalid'
        rows.append({**_row(sample, text, text == gold, text, status,
                            None if text else 'empty_answer'), 'gold_answer': gold})
    return rows, dict(accuracy=sum(row['score'] for row in rows)/len(rows), primary_metric='accuracy',
                      official_metric='exact answer accuracy',
                      scope='frozen GQA val_balanced subset; not full official challenge',
                      normalization='none beyond the already-frozen generation text policy')


def _score_textvqa(samples, texts):
    module = load_module(TEXTVQA_SOURCE, '_official_mmf_textvqa')
    evaluator = module.TextVQAAccuracyEvaluator()
    rows, official_inputs = [], []
    for sample, text in zip(samples, texts):
        answers = sample.get('answers', sample.get('gt_answers', sample.get('gold_answers')))
        if answers is None:
            answers = _gold(sample)
        if not isinstance(answers, list) or len(answers) != 10:
            raise ValueError('Official TextVQA scoring requires all10 human answers')
        answers = [answer['answer'] if isinstance(answer, dict) else answer for answer in answers]
        if not all(isinstance(answer, str) for answer in answers):
            raise ValueError('TextVQA annotations must be text')
        normalized = evaluator.answer_processor(text)
        scores = evaluator._compute_answer_scores(answers)
        score = scores.get(normalized, 0.0)
        status = 'parsed' if normalized else 'invalid'
        rows.append({**_row(sample, text, score, normalized, status,
                            None if normalized else 'empty_after_official_normalization'),
                     'normalized_answers': [evaluator.answer_processor(answer) for answer in answers]})
        official_inputs.append(dict(pred_answer=text, gt_answers=answers))
    accuracy = evaluator.eval_pred_list(official_inputs)
    if abs(accuracy-sum(row['score'] for row in rows)/len(rows)) > 1e-12:
        raise AssertionError('Per-question TextVQA scores differ from original MMF evaluation')
    return rows, dict(accuracy=accuracy, primary_metric='accuracy',
                      official_metric='MMF TextVQA VQA-style ten-answer leave-one-out accuracy',
                      normalization='original EvalAIAnswerProcessor punctuation/digits/articles/contractions',
                      scope='frozen TextVQA validation subset')


def mme_calculator():
    """Execute author's original parser class without importing unused sklearn.

    The official script imports sklearn for additional precision/confusion
    metrics. This benchmark needs its unmodified parse_pred_ans and the simple
    exact acc/paired acc+ formulas; no substitute answer parser is introduced.
    """
    key = ('original_mme_parser_ast', sha(MME_SOURCE))
    if key not in _MODULES:
        tree = ast.parse(MME_SOURCE.read_text(encoding='utf-8'))
        node = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == 'calculate_metrics')
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(MME_SOURCE), 'exec'), namespace)
        _MODULES[key] = namespace['calculate_metrics']
    return _MODULES[key]()


def _score_mme(samples, texts):
    calculator = mme_calculator()
    known = {category for categories in MME_CATEGORIES.values() for category in categories}
    rows, grouped = [], defaultdict(list)
    for sample, text in zip(samples, texts):
        category = sample.get('category', sample.get('mme_category'))
        if category not in known:
            raise ValueError(f'Unknown official MME category: {category}')
        group = sample.get('group_id', sample.get('image_id', sample.get('image_path')))
        if group is None:
            raise ValueError('MME needs the actual shared image/group ID for its two questions')
        gold = str(_gold(sample)).lower()
        if gold not in ('yes', 'no'):
            raise ValueError('MME gold must be yes/no')
        parsed = calculator.parse_pred_ans(text.lower())
        status = 'invalid' if parsed == 'other' else 'parsed'
        row = {**_row(sample, text, parsed == gold, parsed, status,
                       'official_first4_characters_no_yesno' if status == 'invalid' else None),
               'category': category, 'gold_answer': gold, 'group_id': str(group)}
        rows.append(row)
        grouped[(category, str(group))].append(row)
    if any(len(group) != 2 for group in grouped.values()):
        raise ValueError('Every MME image/category group must retain exactly its original two questions')
    for group in grouped.values():
        pair_correct = int(all(row['score'] == 1 for row in group))
        for row in group:
            row['group_pair_correct'] = pair_correct
    by_category = {}
    for category in sorted({row['category'] for row in rows}):
        questions = [row for row in rows if row['category'] == category]
        groups = [group for (cat, _), group in grouped.items() if cat == category]
        acc = sum(row['score'] for row in questions)/len(questions)
        acc_plus = sum(all(row['score'] == 1 for row in group) for group in groups)/len(groups)
        # Same original accuracy/acc_plus formula, grouped by identity rather
        # than relying on adjacent lines; sklearn precision metrics are unused.
        by_category[category] = dict(questions=len(questions), groups=len(groups), acc=acc,
                                     acc_plus=acc_plus, score=100*(acc+acc_plus),
                                     other_num=sum(row['parse_invalid'] for row in questions))
    aggregate = dict(primary_metric='MME_subset_score',
        MME_subset_score=sum(value['score'] for value in by_category.values()),
        accuracy=sum(row['score'] for row in rows)/len(rows),
        acc_plus=sum(all(row['score'] == 1 for row in group) for group in grouped.values())/len(grouped),
        groups=len(grouped), categories=by_category,
        group_scores=[dict(category=category, group_id=group_id,
                           acc=sum(row['score'] for row in group)/2,
                           acc_plus=int(all(row['score'] == 1 for row in group)))
                      for (category, group_id), group in grouped.items()],
        absent_categories=sorted(known-set(by_category)),
        official_metric='per-category100*(acc+acc_plus), then category sum',
        scope='stratified image-group subset; not full MME/leaderboard score',
        maximum_present_category_sum=200*len(by_category))
    for name, categories in MME_CATEGORIES.items():
        aggregate[name+'_subset_score'] = sum(by_category[cat]['score'] for cat in categories if cat in by_category)
    return rows, aggregate


def _coco_references(sample):
    values = sample.get('references', sample.get('captions', sample.get('sentences')))
    if not isinstance(values, list) or not values:
        raise ValueError('COCO requires the original human reference captions')
    result = []
    for value in values:
        caption = value.get('raw', value.get('caption')) if isinstance(value, dict) else value
        if not isinstance(caption, str) or not caption:
            raise ValueError('COCO reference caption is empty/invalid')
        result.append(caption)
    return result


def _local_copy(source, target):
    source, target = Path(source), Path(target)
    if not source.is_file():
        raise FileNotFoundError(f'Official metric dependency missing: {source}')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha(source) != sha(target):
            raise ValueError(f'Existing local metric source differs: {target}')
    else:
        shutil.copyfile(source, target)
    return sha(source)


def _ptb(output_dir):
    """Original wrapper writes its tmp file beside itself: copy wrapper first."""
    source = COCO_DEPS/'pycocoevalcap/tokenizer'
    destination = output_dir/'reference/tokenizer'
    _local_copy(source/'ptbtokenizer.py', destination/'ptbtokenizer.py')
    jar = destination/'stanford-corenlp-3.4.1.jar'
    original_jar = source/jar.name
    if not original_jar.is_file():
        raise FileNotFoundError(original_jar)
    if not jar.exists():
        os.symlink(original_jar, jar)
    elif jar.resolve() != original_jar.resolve() and sha(jar) != sha(original_jar):
        raise ValueError('PTB tokenizer jar differs')
    return load_module(destination/'ptbtokenizer.py', '_fresh_official_ptb').PTBTokenizer()


def _spice(references, predicted, output_dir):
    package = COCO_DEPS/'pycocoevalcap/spice'
    jar = package/'spice-1.0.jar'
    if not jar.is_file() or not (SPICE_MODELS/'stanford-corenlp-3.6.0-models.jar').is_file():
        raise FileNotFoundError('Official SPICE jar/Stanford models unavailable; no replacement scorer allowed')
    folder = output_dir/'spice'
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'cache').mkdir(exist_ok=True)
    payload = [dict(image_id=index, test=predicted[index][0], refs=references[index]) for index in references]
    input_path, output_path = folder/'input.json', folder/'output.json'
    write(input_path, payload)
    identity = sha(input_path)
    signature = folder/'input.sha256'
    command = ['java', '-XX:ActiveProcessorCount=8', '-Xmx16G', '-cp',
               os.pathsep.join([str(SPICE_MODELS/'*'), str(jar), str(package/'lib/*')]),
               'edu.anu.spice.SpiceScorer', str(input_path), '-cache', str(folder/'cache'),
               '-out', str(output_path), '-subset', '-silent', '-threads', '1']
    reused = output_path.exists() and signature.exists() and signature.read_text(encoding='utf-8').strip() == identity
    if not reused:
        import fcntl
        lock_path = ROOT/'metrics/.spice.lock'
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open('a', encoding='utf-8') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with (folder/'scorer.log').open('w', encoding='utf-8') as log:
                subprocess.run(command, cwd=folder, stdout=log, stderr=subprocess.STDOUT, check=True)
        signature.write_text(identity+'\n', encoding='utf-8')
    values = json.loads(output_path.read_text(encoding='utf-8'))
    scores = {int(value['image_id']): value['scores'] for value in values}
    if set(scores) != set(references) or len(values) != len(scores):
        raise ValueError('Official SPICE output IDs incomplete/duplicated')
    result = {}
    for key, value in scores.items():
        raw = value['All']['f']
        try:
            number = float(raw)
        except (TypeError, ValueError):
            number = float('nan')
        result[key] = number if math.isfinite(number) else None
    return result, dict(command=command, input_sha256=identity, spice_jar_sha256=sha(jar),
                        wrapper_sha256=sha(package/'spice.py'),
                        stanford_models_sha256=sha(SPICE_MODELS/'stanford-corenlp-3.6.0-models.jar'),
                        cache_scope='new experiment/output only; no historical score or writable cache import',
                        exact_same_input_result_reused=reused,
                        undefined_aggregate_policy='preserve null; any undefined item makes aggregate undefined')


def _chair(samples, texts, output_dir):
    """Verified Python3 adapter invokes original author CHAIR code unchanged."""
    folder = output_dir/'chair'
    records = {}
    for relative in ('pattern_inflect.py', 'CHAIR/utils/chair.py', 'CHAIR/data/synonyms.txt'):
        records[relative] = _local_copy(CHAIR_BASE/'reference'/relative, folder/'reference'/relative)
    source = CHAIR_BASE/'source/chair.py'
    common = types.ModuleType('common')
    common.ROOT = folder
    common.digest = sha
    common.read = lambda path: json.loads(Path(path).read_text(encoding='utf-8'))
    def scoped_write(path, value):
        path = Path(path).resolve()
        if not path.is_relative_to(folder.resolve()):
            raise ValueError('CHAIR adapter attempted write outside current experiment output')
        write(path, value)
    common.write = scoped_write
    previous = sys.modules.get('common')
    sys.modules['common'] = common
    try:
        # Each output gets its own adapter globals so author debug writes never
        # inherit another method's output root or a historical experiment root.
        spec = importlib.util.spec_from_file_location('_fresh_author_chair_' + sha(source)[:12], source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop('common', None)
        else:
            sys.modules['common'] = previous
    adapted = []
    for sample in samples:
        image_id = sample.get('cocoid', sample.get('image_id'))
        if image_id is None:
            raise ValueError('COCO/CHAIR requires original numeric COCO image_id')
        adapted.append(dict(sample, cocoid=int(image_id)))
    (folder/'reference').mkdir(parents=True, exist_ok=True)
    with (folder/'author.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        chair = module.Chair(adapted)
        rows = [chair.score(sample['cocoid'], text) for sample, text in zip(adapted, texts)]
        objects = sum(row['object_mentions'] for row in rows)
        hallucinations = sum(row['hallucinated_mentions'] for row in rows)
        overall_s = sum(row['CHAIRs'] for row in rows)/len(rows)
        overall_i = hallucinations/objects if objects else None
        if objects:
            # Original author compute_chair must agree on both aggregate and
            # every sentence; zero denominator stays undefined, never fake0.
            official = chair.check_author([dict(output_text=text) for text in texts], adapted, 'current')
            if abs(official['CHAIRs']-overall_s) > 1e-12 or abs(official['CHAIRi']-overall_i) > 1e-12:
                raise AssertionError('Independent CHAIR rows disagree with original author aggregation')
    return rows, dict(CHAIRs=overall_s, CHAIRi=overall_i, hallucinated_mentions=hallucinations,
                      object_mentions=objects), dict(author_sources=records, adapter_sha256=sha(source),
                      annotations={path.name: sha(path) for path in sorted(ANNOTATIONS.glob('*2014.json'))
                                   if path.name.startswith(('captions_', 'instances_'))},
                      scope='author80 COCO object categories plus synonyms; not general factuality',
                      zero_corpus_object_denominator='null, undefined; per-caption original no-object score remains0')


def _score_coco(samples, texts, output_dir):
    if len({int(sample.get('cocoid', sample.get('image_id'))) for sample in samples}) != len(samples):
        raise ValueError('COCO reference corpus must contain each original image exactly once')
    tokenizer = _ptb(output_dir)
    reference_raw = {index: [dict(caption=text) for text in _coco_references(sample)] for index, sample in enumerate(samples)}
    predicted_raw = {index: [dict(caption=text)] for index, text in enumerate(texts)}
    references = tokenizer.tokenize(reference_raw)
    predicted = tokenizer.tokenize(predicted_raw)
    write(output_dir/'ptb_references.json', references)
    write(output_dir/'ptb_predictions.json', predicted)
    # No tokenizer/tmp/cache writes are directed into the original dependency.
    if str(COCO_DEPS) not in sys.path:
        sys.path.insert(0, str(COCO_DEPS))
    from pycocoevalcap.cider.cider import Cider
    cider, individual = Cider().compute_score(references, predicted)
    if len(individual) != len(samples) or not math.isfinite(float(cider)):
        raise ValueError('Official CIDEr output invalid')
    spice, spice_provenance = _spice(references, predicted, output_dir)
    chair, chair_aggregate, chair_provenance = _chair(samples, texts, output_dir)
    rows = []
    for index, (sample, text) in enumerate(zip(samples, texts)):
        status = 'parsed' if text.strip() else 'invalid'
        row = _row(sample, text, float(individual[index]), text, status,
                   None if status == 'parsed' else 'empty_caption_retained')
        row.update(CIDEr=float(individual[index]), SPICE=spice[index],
                   SPICE_status='ok' if spice[index] is not None else 'undefined_official_All_f',
                   **chair[index])
        rows.append(row)
    aggregate = dict(primary_metric='CIDEr', CIDEr=float(cider),
        SPICE=None if any(value is None for value in spice.values()) else sum(spice.values())/len(spice),
        **chair_aggregate,
        metric_units='CIDEr native scorer scale; SPICE/CHAIR fractions; no implicitx100',
        CIDEr_implementation='official pycocoevalcap Cider: clipping and sigma6 Gaussian length penalty (CIDEr-D implementation)',
        CIDEr_document_frequency_images=len(samples),
        CIDEr_DF_policy='fresh current complete subset references only; dev20 and test100 are separate DF corpora; compare methods within split',
        scope='subset caption quality/object hallucination; not human factual/omission ground truth',
        additional_provenance=dict(spice=spice_provenance, chair=chair_provenance,
            cider_scorer_sha256=sha(COCO_DEPS/'pycocoevalcap/cider/cider_scorer.py'),
            tokenizer_sha256=sha(COCO_DEPS/'pycocoevalcap/tokenizer/ptbtokenizer.py')))
    return rows, aggregate


def attach_official_labels(task, samples):
    """Join prepared labels only during CPU scoring, never into model inputs."""
    expected = {'scienceqa': ('answer_index', 'answer', 'gold_answer', 'gold', 'label'),
                'textvqa': ('answers', 'gt_answers', 'gold_answers', 'gold'),
                'coco': ('references', 'captions', 'sentences'),
                'gqa': ('answer', 'gold_answer', 'gold', 'label'),
                'mme': ('answer', 'gold_answer', 'gold', 'label')}[task]
    samples = [dict(sample) for sample in samples]
    for sample in samples:
        if isinstance(sample.get('label'), dict):
            nested = sample.pop('label')
            for key, value in nested.items():
                if key in sample and sample[key] != value:
                    raise ValueError('Nested evaluation label conflicts with runtime metadata')
                sample[key] = value
    if all(any(key in sample for key in expected) for sample in samples):
        return samples, dict(mode='explicit labeled scoring samples', labels_used_for_generation=False)
    identifiers = {sample_id(sample) for sample in samples}
    directory = ROOT/'data'/task
    candidates = []
    for split in ('dev', 'test'):
        path = directory/f'{split}_labels.json'
        if path.is_file():
            labels = json.loads(path.read_text(encoding='utf-8'))
            if identifiers <= set(labels):
                candidates.append((path, labels))
    if len(candidates) != 1:
        raise ValueError('Score samples lack labels and do not identify exactly one prepared official-label split')
    path, labels = candidates[0]
    fields = {'answer', 'answer_index', 'answer_letter', 'answers', 'choices', 'references',
              'captions', 'sentences', 'category', 'cocoid'}
    for sample in samples:
        label = labels[sample_id(sample)]
        if sample.get('image_sha256') is not None and label.get('image_sha256') != sample['image_sha256']:
            raise ValueError('Prepared evaluation label refers to a different original image')
        for key in fields & set(label):
            if key in sample and sample[key] != label[key]:
                raise ValueError('Prepared evaluation label conflicts with supplied sample metadata')
            sample[key] = label[key]
    return samples, dict(mode='post-generation join of separate official-label file',
                         path=str(path), sha256=sha(path), labels_used_for_generation=False)


def score_task(task, samples, predictions, output_dir):
    """Return (per_question, aggregate), and write their UTF-8 JSON evidence.

    Only one method/split and one deterministic quality repeat per call. MME
    samples remain both questions per selected category/image group. Official
    dependencies are read-only; all generated caches/files stay in new ROOT.
    """
    task = str(task).lower().replace('_', '').replace('-', '')
    task = {'sqa': 'scienceqa', 'cococaption': 'coco', 'cococaptions': 'coco'}.get(task, task)
    if task not in ('mme', 'gqa', 'textvqa', 'scienceqa', 'coco'):
        raise ValueError('Unsupported task')
    samples, texts = align(samples, predictions)
    samples, label_provenance = attach_official_labels(task, samples)
    destination = Path(output_dir).resolve()
    if not destination.is_relative_to(ROOT.resolve()):
        raise ValueError('Evaluation writes must stay within this new benchmark root')
    destination.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(json.dumps(dict(task=task, samples=samples, predictions=texts),
                                         sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    signature_path = destination/'metric_inputs.sha256'
    if signature_path.exists() and signature_path.read_text(encoding='utf-8').strip() != signature:
        raise ValueError('Existing metric output belongs to different task/reference/predictions; use a distinct output_dir')
    signature_path.write_text(signature+'\n', encoding='utf-8')
    try:
        if task == 'coco':
            rows, aggregate = _score_coco(samples, texts, destination)
        else:
            scorer = {'mme': _score_mme, 'gqa': _score_gqa, 'textvqa': _score_textvqa, 'scienceqa': _score_scienceqa}[task]
            rows, aggregate = scorer(samples, texts)
    except Exception as error:
        write(destination/'metric_failure.json', dict(task=task, exception=repr(error), input_sha256=signature,
                                                      substitute_metric_used=False))
        raise
    source = {'mme': MME_SOURCE, 'gqa': GQA_SOURCE, 'textvqa': TEXTVQA_SOURCE, 'scienceqa': SCIENCEQA_SOURCE}.get(task)
    aggregate.update(task=task, questions=len(rows),
        parse_invalid_count=sum(row['parse_invalid'] for row in rows),
        uncertain_count=sum(row['uncertain'] for row in rows),
        all_questions_retained=True, input_sha256=signature,
        source_url=SOURCE_URLS.get(task, SOURCE_URLS['cider']),
        original_implementation_sha256=sha(source) if source is not None else None,
        official_label_join=label_provenance,
        evaluator_sha256=sha(__file__), automatic_judge_used=False,
        invalid_policy='score0 and keep denominator; COCO undefined metrics retained null')
    rows, aggregate = clean(rows), clean(aggregate)
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
    samples = json.loads(args.samples.read_text(encoding='utf-8'))
    text = args.predictions.read_text(encoding='utf-8')
    try:
        predictions = json.loads(text)
    except json.JSONDecodeError:
        predictions = [json.loads(line) for line in text.splitlines() if line.strip()]
    _, aggregate = score_task(args.task, samples, predictions, args.output_dir)
    print(json.dumps(aggregate, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
