"""CPU-only checks of single-pass adaptation; no experiment outputs read."""
import ast
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'harness'))
import analyze_results as analysis
import select_lvspec as selector


class SinglePassStatisticsTests(unittest.TestCase):
    def test_parent_parser_function_is_identical(self):
        def function(path, name):
            tree=ast.parse(path.read_text())
            return ast.dump(next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==name),include_attributes=False)
        self.assertEqual(function(ROOT/'harness/evaluate_long.py','extract_final_answer'),
                         function(ROOT.parent/'harness/evaluate_long.py','extract_final_answer'))

    def test_declared_selection_gate_unchanged(self):
        def function(path):
            tree=ast.parse(path.read_text())
            return ast.dump(next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='quality_pass'),include_attributes=False)
        self.assertEqual(function(ROOT/'harness/select_lvspec.py'),
                         function(ROOT.parent/'harness/select_lvspec.py'))

    def test_no_repeat_averaging_or_false_determinism_claim(self):
        source=(ROOT/'harness/analyze_results.py').read_text()
        self.assertIn("batch=[runtime[ordinal,0]]",source)
        self.assertIn("repeat_determinism_passed=None",source)
        self.assertNotIn("requests=len(samples)*3",source)

    def test_single_pass_pooled_mat_and_mme_statistics(self):
        analysis.self_test()

    def test_selection_paper_first_and_explicit_strict_fallback(self):
        selector.self_test()


if __name__=='__main__':
    unittest.main()
