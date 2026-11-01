"""CPU metric/parsing tests; synthetic fixtures are never benchmark evidence."""
from pathlib import Path
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1]/'harness/evaluate_tasks.py'
spec = importlib.util.spec_from_file_location('clean_task_metrics', SOURCE)
metric = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metric)


class TaskMetricTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.patch = patch.object(metric, 'ROOT', self.root)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def score(self, task, samples, predictions):
        return metric.score_task(task, samples, predictions, self.root/'metrics'/task)

    def test_gqa_original_exact_equality_keeps_case_and_punctuation(self):
        samples = [dict(question_id='1', answer='blue'), dict(question_id='2', answer='yes')]
        rows, aggregate = self.score('gqa', samples, {'1': 'Blue', '2': 'yes'})
        self.assertEqual([row['score'] for row in rows], [0, 1])
        self.assertEqual(aggregate['accuracy'], .5)
        self.assertFalse(rows[0]['parse_invalid'])
        self.assertEqual(json.loads((self.root/'metrics/gqa/aggregate.json').read_text())['accuracy'], .5)

    def test_textvqa_uses_original_normalization_and_leave_one_out(self):
        samples = [dict(question_id=str(index), answers=['red']*count + ['blue']*(10-count))
                   for index, count in enumerate([1, 2, 3, 4])]
        rows, aggregate = self.score('textvqa', samples, ['RED.']*4)
        for row, expected in zip(rows, [.3, .6, .9, 1.]):
            self.assertAlmostEqual(row['score'], expected, places=12)
        self.assertAlmostEqual(aggregate['accuracy'], .7, places=12)
        evaluator = metric.load_module(metric.TEXTVQA_SOURCE, 'original_textvqa_test').TextVQAAccuracyEvaluator()
        self.assertEqual(evaluator.answer_processor('The TWO.'), '2')
        self.assertEqual(evaluator.answer_processor('a cat'), 'cat')
        self.assertEqual(evaluator.answer_processor('dont'), "don't")

    def test_textvqa_requires_original_ten_answers(self):
        with self.assertRaises(ValueError):
            self.score('textvqa', [dict(question_id='1', answers=['yes'])], ['yes'])

    def test_scienceqa_fixed_parser_does_not_guess(self):
        base = dict(choices=['first', 'second', 'third'], image_path='fixture.png', answer_index=1)
        samples = [dict(base, question_id=str(index)) for index in range(5)]
        predictions = ['B', 'The answer is (B).', 'second', 'Answer: A. The answer is B.', 'Z']
        rows, aggregate = self.score('scienceqa', samples, predictions)
        self.assertEqual([row['score'] for row in rows], [1, 1, 0, 0, 0])
        self.assertEqual(aggregate['accuracy'], .4)
        self.assertEqual(aggregate['parse_invalid_count'], 2)
        self.assertEqual(aggregate['uncertain_count'], 1)
        self.assertEqual(rows[3]['parse_reason'], 'conflicting_explicit_answer_markers')

    def test_scienceqa_multianswer_and_out_of_range_remain_invalid_or_uncertain(self):
        self.assertEqual(metric.parse_scienceqa('Answer is A or B', 3)[1], 'uncertain')
        self.assertEqual(metric.parse_scienceqa('A. first\nB. second', 3)[1], 'uncertain')
        self.assertEqual(metric.parse_scienceqa('The answer is D.', 3)[1], 'invalid')
        self.assertEqual(metric.parse_scienceqa('The answer is blue.', 3)[1], 'invalid')
        self.assertEqual(metric.parse_scienceqa('(C)', 3)[:2], (2, 'parsed'))

    def test_mme_preserves_two_question_group_accplus_and_official_prefix(self):
        samples = [dict(question_id=str(index), group_id='img'+str(index//2), category='color',
                        answer='yes' if index%2 == 0 else 'no') for index in range(4)]
        rows, aggregate = self.score('mme', samples, ['Yes.', 'No, it is not.', 'yes', 'yes'])
        self.assertEqual([row['score'] for row in rows], [1, 1, 1, 0])
        self.assertEqual(aggregate['categories']['color']['acc'], .75)
        self.assertEqual(aggregate['categories']['color']['acc_plus'], .5)
        self.assertEqual(aggregate['MME_subset_score'], 125.)
        self.assertEqual(aggregate['maximum_present_category_sum'], 200)
        original = metric.mme_calculator()
        self.assertEqual(original.parse_pred_ans('nobody knows'), 'no')
        self.assertEqual(original.parse_pred_ans('i do not know'), 'other')
        with self.assertRaises(ValueError):
            metric._score_mme(samples[:3], ['yes', 'no', 'yes'])

    def test_duplicate_or_missing_prediction_ids_fail_without_resampling(self):
        samples = [dict(question_id='1', answer='yes'), dict(question_id='2', answer='no')]
        with self.assertRaises(ValueError):
            metric.align(samples, [{'question_id': '1', 'output_text': 'yes'}])
        with self.assertRaises(ValueError):
            metric.align(samples, [{'question_id': '1', 'output_text': 'yes'}]*2)

    def test_separate_official_labels_are_joined_only_for_scoring(self):
        samples = [dict(question_id='q1', image_id=4, image_sha256='abc', image_path='fixture.png')]
        folder = self.root/'data/scienceqa'; folder.mkdir(parents=True)
        labels = {'q1': dict(answer_index=0, answer_letter='A', choices=['x', 'y'], image_sha256='abc')}
        (folder/'test_labels.json').write_text(json.dumps(labels))
        rows, aggregate = self.score('scienceqa', samples, ['A'])
        self.assertEqual(rows[0]['score'], 1.)
        self.assertNotIn('answer_index', samples[0])
        self.assertEqual(aggregate['official_label_join']['mode'], 'post-generation join of separate official-label file')

    def test_coco_reference_formats_and_spice_undefined_retention(self):
        samples = [dict(question_id='1', image_id=1, sentences=[{'raw': 'A cat sits on a mat.'}, {'raw': 'A small cat sits.'}]),
                   dict(question_id='2', image_id=2, references=['A dog runs on grass.', 'A running dog.'])]
        self.assertEqual(metric._coco_references(samples[0])[0], 'A cat sits on a mat.')
        chair_rows = [dict(CHAIRs=0, CHAIRi=0, hallucinated_mentions=0, object_mentions=1,
                           hallucinated_words=[], generated_objects=['cat'], groundtruth_objects=['cat']),
                      dict(CHAIRs=1, CHAIRi=1, hallucinated_mentions=1, object_mentions=1,
                           hallucinated_words=['cat'], generated_objects=['cat'], groundtruth_objects=['dog'])]
        # Exercise original PTB/CIDEr, with explicitly synthetic other metric
        # outputs to test output plumbing only; never claim these as real scores.
        old_names = {p.name for p in (metric.COCO_DEPS/'pycocoevalcap/tokenizer').iterdir()}
        with patch.object(metric, '_spice', return_value=({0: .5, 1: None}, {'synthetic_test_only': True})), \
             patch.object(metric, '_chair', return_value=(chair_rows, {'CHAIRs': .5, 'CHAIRi': .5}, {'synthetic_test_only': True})):
            rows, aggregate = self.score('coco', samples, ['A cat sits.', 'A cat runs.'])
        self.assertEqual(aggregate['CIDEr_document_frequency_images'], 2)
        self.assertIsNone(aggregate['SPICE'])
        self.assertIsNone(rows[1]['SPICE'])
        self.assertEqual(rows[1]['SPICE_status'], 'undefined_official_All_f')
        self.assertEqual(len(rows), 2)
        self.assertEqual(old_names, {p.name for p in (metric.COCO_DEPS/'pycocoevalcap/tokenizer').iterdir()})


if __name__ == '__main__':
    unittest.main()
