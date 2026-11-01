"""Fixed format boundary tests added before the first task-quality scoring.

No generated benchmark answers, labels, model calls, or expected quality
changes are used. The original five sealed test sources remain unchanged.
"""
import importlib.util
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('official_metrics_parser_boundary', ROOT/'harness/evaluate_tasks.py')
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


class ScienceQAFormatBoundary(unittest.TestCase):
    def test_article_is_never_an_explicit_option_marker(self):
        for text in ('The answer is a lion.', 'The answer is A lion.',
                     'The answer is a butterfly.', 'Answer: A because it is visible.'):
            with self.subTest(text=text):
                self.assertEqual(metrics.parse_scienceqa(text, 4)[:2], (None, 'invalid'))

    def test_delimited_markers_and_bare_letters(self):
        for text in ('A', 'a', '(a)', 'A.', 'The answer is A.', 'The answer is a.',
                     'Answer: (a)', 'Answer is [A] because it is visible.',
                     'The answer is A', 'The answer is A\nExplanation follows.',
                     'Answer = A: explanation.', 'Answer is A .'):
            with self.subTest(text=text):
                self.assertEqual(metrics.parse_scienceqa(text, 4)[:2], (0, 'parsed'))

    def test_ambiguous_or_out_of_range_remain_in_denominator(self):
        self.assertEqual(metrics.parse_scienceqa('The answer is A or B.', 4)[:2], (None, 'uncertain'))
        self.assertEqual(metrics.parse_scienceqa('Answer is (A). Answer is (B).', 4)[:2], (None, 'uncertain'))
        self.assertEqual(metrics.parse_scienceqa('Answer is Z.', 4)[:2], (None, 'invalid'))

    def test_unpunctuated_explanation_is_explicitly_invalid(self):
        for text in ('The answer is A because of the picture.', 'Answer: B since it is red.'):
            with self.subTest(text=text):
                self.assertEqual(metrics.parse_scienceqa(text, 4)[:2], (None, 'invalid'))


if __name__ == '__main__':
    unittest.main()
