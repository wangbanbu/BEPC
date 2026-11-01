"""Synthetic CPU-only format/official-score boundary tests, not experiment data."""
from pathlib import Path
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1]/'harness/evaluate_long.py'
spec = importlib.util.spec_from_file_location('long_quality_metrics', SOURCE)
metric = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metric)


class LongQualityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.patch = patch.object(metric, 'ROOT', self.root)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def score(self, task, samples, outputs):
        return metric.score_task(task, samples, outputs, self.root/'metrics'/task)

    def test_only_unique_terminal_marker_accepted(self):
        parse = metric.extract_final_answer
        self.assertEqual(parse('gqa', 'The shirt appears blue.\n\nFinal answer: blue\n')['answer'], 'blue')
        self.assertEqual(parse('gqa', 'Blue')['status'], 'invalid')
        self.assertEqual(parse('gqa', 'Final answer: blue\nMore rationale')['status'], 'invalid')
        self.assertEqual(parse('gqa', 'Final answer: blue\nFinal answer: blue')['status'], 'uncertain')
        self.assertEqual(parse('gqa', 'Final answer: blue\nFinal answer: red')['status'], 'uncertain')
        self.assertEqual(parse('gqa', 'Final answer:  ')['status'], 'invalid')
        self.assertEqual(parse('gqa', 'Explanation; Final answer: blue')['status'], 'invalid')

    def test_mme_does_not_inherit_nobody_parser_bug(self):
        self.assertEqual(metric.extract_final_answer('mme', 'Explanation.\nFinal answer: nobody')['status'], 'invalid')
        self.assertEqual(metric.extract_final_answer('mme', 'Explanation.\nFinal answer: No.')['answer'], 'No.')

    def test_gqa_exact_case_and_punctuation_preserved_and_failed_denominator(self):
        samples = [dict(question_id=str(i), answer='blue') for i in range(4)]
        rows, aggregate = self.score('gqa', samples, [
            'Reasoning.\nFinal answer: blue', 'Reasoning.\nFinal answer: Blue',
            'Reasoning.\nFinal answer: blue.', 'blue'])
        self.assertEqual([row['score'] for row in rows], [1, 0, 0, 0])
        self.assertEqual(aggregate['accuracy'], .25)
        self.assertEqual(aggregate['answer_extraction_failures'], 1)
        self.assertEqual(rows[0]['prediction'], 'Reasoning.\nFinal answer: blue')
        self.assertEqual(rows[0]['extracted_answer'], 'blue')

    def test_textvqa_executes_original_mmf_normalization(self):
        samples = [dict(question_id=str(i), answers=['red']*count+['blue']*(10-count))
                   for i, count in enumerate([1, 2, 3, 4])]
        rows, aggregate = self.score('textvqa', samples, ['Visual evidence.\nFinal answer: RED.']*4)
        for row, expected in zip(rows, [.3, .6, .9, 1.]):
            self.assertAlmostEqual(row['score'], expected, places=12)
        self.assertAlmostEqual(aggregate['accuracy'], .7, places=12)

    def test_scienceqa_uses_option_not_rationale_or_article(self):
        fixture = dict(choices=['lion', 'tiger'], image_path='fixture.png', answer_index=0)
        samples = [dict(fixture, question_id=str(i)) for i in range(4)]
        outputs = ['Rationale mentions option B.\nFinal answer: A',
                   'Reasoning.\nFinal answer: a lion',
                   'Reasoning.\nFinal answer: Z',
                   'Final answer: A\nFinal answer: B']
        rows, aggregate = self.score('scienceqa', samples, outputs)
        self.assertEqual([row['score'] for row in rows], [1, 0, 0, 0])
        self.assertEqual(aggregate['uncertain_count'], 1)
        self.assertEqual(aggregate['parse_invalid_count'], 2)
        self.assertEqual(aggregate['accuracy'], .25)

    def test_mme_preserves_full_question_pairs(self):
        samples = [dict(question_id=str(i), group_id='image', category='color',
                        answer='yes' if i == 0 else 'no') for i in range(2)]
        rows, aggregate = self.score('mme', samples, ['Visible blue.\nFinal answer: yes',
                                                   'Visible blue.\nFinal answer: no'])
        self.assertEqual(aggregate['MME_subset_score'], 200.)
        self.assertTrue(all(row['group_pair_correct'] == 1 for row in rows))

    def test_coco_full_text_is_never_reduced_to_first_sentence(self):
        text = 'A person stands outside. A bicycle leans against a wall.\nFinal answer: irrelevant marker.'
        entry = metric.extract_final_answer('coco', text)
        self.assertEqual(entry['answer'], text)
        self.assertEqual(entry['status'], 'not_applicable')

    def test_original_source_sha_checked_and_output_cannot_escape(self):
        self.assertEqual(metric.sha(metric.OFFICIAL_SOURCE), metric.OFFICIAL_SHA256)
        with self.assertRaises(ValueError):
            metric.score_task('gqa', [dict(question_id='1', answer='blue')],
                              ['Final answer: blue'], self.root.parent/'escape-quality-test')

    def test_new_raw_output_cannot_overwrite_same_extracted_score(self):
        sample = [dict(question_id='1', answer='blue')]
        self.score('gqa', sample, ['First rationale.\nFinal answer: blue'])
        with self.assertRaises(ValueError):
            self.score('gqa', sample, ['Different rationale.\nFinal answer: blue'])

    def test_sealed_label_join_is_read_only_and_bound_to_image(self):
        original = metric.read(metric.BASE_ROOT/'data/scienceqa/dev.json')[0]
        snapshot = json.dumps(original, sort_keys=True)
        joined, provenance = metric.attach_labels('scienceqa', [original])
        self.assertIn('answer_index', joined[0])
        self.assertEqual(json.dumps(original, sort_keys=True), snapshot)
        self.assertEqual(provenance['split'], 'dev')
        damaged = dict(original, image_sha256='incorrect')
        with self.assertRaises(ValueError):
            metric.attach_labels('scienceqa', [damaged])


if __name__ == '__main__':
    unittest.main()
