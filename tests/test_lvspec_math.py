"""CPU checks against freshly cloned LVSpec algorithms and legal-tree cases."""
import ast
import importlib.util
from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'harness'))
from lvspec_controller import (visual_relevance, branch_low_relevance_mask,
    scan_branch, legal_branch_paths)

spec = importlib.util.spec_from_file_location('fresh_lvspec_loose_match', ROOT/'reference/LVSpec/decoding/loose_match.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
source = ast.parse((ROOT/'reference/LVSpec/decoding/tree_decoding_qwen2_5.py').read_text())
function = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == 'visual_similarity_hidden_state')
namespace = dict(torch=torch, F=F)
exec(compile(ast.Module(body=[function], type_ignores=[]), '<fresh-official-visual-function>', 'exec'), namespace)


class LVSpecMathTests(unittest.TestCase):
    def test_top10_cosine_matches_actual_official_function_fp32(self):
        torch.manual_seed(104)
        for visual_n in (3, 10, 21):
            visual, text = torch.randn(visual_n, 16), torch.randn(7, 16)
            actual = visual_relevance(visual, text)
            expected = namespace['visual_similarity_hidden_state'](visual[None], text[None], topk=10)[0]
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5))
        with self.assertRaises(ValueError):
            visual_relevance(torch.empty(0, 16), text)
        with self.assertRaises(ValueError):
            visual_relevance(torch.randn(3, 15), text)

    def test_chain10_ratio_point7_matches_fresh_official_k7_and_pst(self):
        torch.manual_seed(208)
        for _ in range(100):
            candidates = torch.randint(2, 17, (3, 11))
            logits = torch.randn(3, 10, 18)
            logits[:, :, :2] = -100 # exclude EOS/sentinel guards for policy equivalence
            targets = logits.argmax(-1)
            scores = torch.randn(3, 10)
            posterior = (candidates[:, 1:] == targets).int()
            expected = reference.loose_match_hidden_topk_or_search(posterior, scores, candidates, logits, k=7)
            for row in range(3):
                allowed = branch_low_relevance_mask(scores[row], .7)
                actual, _, _ = scan_branch(candidates[row].tolist(), targets[row].tolist(), allowed.tolist(),
                    eos=1, committed_length=0, visible_boundary=1000, pst_enabled=True)
                self.assertEqual(actual, int(expected[row]))

    def test_rank_comparison_is_one_legal_path_not_a_tree_wide_pool(self):
        retrieve = [[0, 1, 3], [0, 2, 4]]
        paths = legal_branch_paths(retrieve, 5)
        self.assertEqual(paths[3], (0, 1, 3))
        native_scores = torch.tensor([.5, .1, .9, .3, .3])
        a = branch_low_relevance_mask(native_scores[retrieve[0][:-1]], .5)
        b = branch_low_relevance_mask(native_scores[retrieve[1][:-1]], .5)
        # The common parent has identical absolute score, different branch rank.
        self.assertFalse(bool(a[0]))
        self.assertTrue(bool(b[0]))
        # Changing unrelated branch2 cannot alter branch1's mask.
        native_scores[2] = -100
        self.assertTrue(torch.equal(a, branch_low_relevance_mask(native_scores[retrieve[0][:-1]], .5)))
        with self.assertRaises(ValueError):
            legal_branch_paths([[0, 1, 3], [0, 2, 3]], 4)
        with self.assertRaises(ValueError):
            legal_branch_paths([[0, -1, 1]], 2)

    def test_lambda_zero_is_not_strict_with_pst_and_explicit_fallback_is_strict(self):
        tokens, targets = [7, 3, 4], [7, 9]
        mask = branch_low_relevance_mask(torch.tensor([.1, .9]), 0).tolist()
        self.assertEqual(mask, [False, False])
        kwargs = dict(eos=1, committed_length=0, visible_boundary=100)
        self.assertEqual(scan_branch(tokens, targets, mask, pst_enabled=True, **kwargs)[0], 1)
        self.assertEqual(scan_branch(tokens, targets, mask, pst_enabled=False, **kwargs)[0], 0)
        self.assertEqual(scan_branch(tokens, targets, [True, True], disable_relaxation=True, **kwargs)[0], 0)
        self.assertEqual(scan_branch(tokens, targets, mask, pst_include_root=False, **kwargs)[0], 0)

    def test_pst_does_not_search_siblings_and_limits_window(self):
        kwargs = dict(eos=1, committed_length=0, visible_boundary=100)
        self.assertEqual(scan_branch([0, 3, 4], [8, 9], [False, False], **kwargs)[0], 0)
        # An unrelated sibling could contain8; it is absent from the legal row.
        self.assertEqual(scan_branch([0, 3, 8], [8, 9], [False, False], **kwargs)[0], 1)
        tokens = list(range(20, 33))
        allowed = branch_low_relevance_mask(torch.arange(12.).float(), 1.)
        targets = [token+100 for token in tokens[1:]]
        self.assertEqual(scan_branch(tokens, targets, allowed.tolist(), pst_enabled=False, **kwargs)[0], 10)

    def test_common_eos_budget_padding_guards_and_original_unlimited_mismatches(self):
        kwargs = dict(eos=1, committed_length=0, visible_boundary=100)
        self.assertEqual(scan_branch([0, 1, 4], [2, 4], [True, True], **kwargs)[0], 0)
        self.assertEqual(scan_branch([0, 1, 4], [1, 4], [True, True], **kwargs)[0], 2)
        self.assertEqual(scan_branch([0, 3, -1], [2, 4], [True, True], **kwargs)[0], 1)
        self.assertEqual(scan_branch([0, 3, 4], [2, 9], [True, True],
                                    eos=1, committed_length=10, visible_boundary=11)[0], 0)
        tokens = list(range(20, 31))
        length, mismatches, _ = scan_branch(tokens, [100]*10, [True]*10, pst_enabled=False, **kwargs)
        self.assertEqual((length, mismatches), (10, 10)) # no added2/4 caps

    def test_fraction_uses_actual_length_and_ties_have_declared_position_order(self):
        self.assertEqual(branch_low_relevance_mask(torch.tensor([.1, .2, .3]), .7).tolist(), [True, True, False])
        self.assertEqual(branch_low_relevance_mask(torch.tensor([.1, .2]), .3).tolist(), [False, False])
        self.assertEqual(branch_low_relevance_mask(torch.zeros(4), .5).tolist(), [True, True, False, False])
        with self.assertRaises(ValueError):
            branch_low_relevance_mask(torch.ones(3), 1.1)


if __name__ == '__main__':
    unittest.main()
