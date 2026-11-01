"""CPU integration: native-postnorm capture, branch/logit alignment and release."""
from pathlib import Path
from types import SimpleNamespace
import gc
import sys
import unittest
import weakref

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'harness'))
from lvspec_controller import LVSpecController, visual_relevance, branch_low_relevance_mask, scan_branch


class NativeFixture:
    def __init__(self):
        torch.manual_seed(201)
        self.norm = torch.nn.LayerNorm(8, bias=False)
        self.model = SimpleNamespace(base_model=SimpleNamespace(
            config=SimpleNamespace(image_token_id=99),
            model=SimpleNamespace(norm=self.norm, tree_mask=None)),
            tokenizer=SimpleNamespace(eos_token_id=1))
        self.ids = torch.tensor([[98, 99, 99, 10, 97]])
        self.tokens = torch.tensor([[0, 4, 5, 6, 7, 8, 9]])
        self.retrieve = torch.tensor([[0, 1, 3], [0, 1, 4], [0, 2, 5], [0, 2, 6]])
        self.positions = torch.tensor([0, 1, 1, 2, 2, 2, 2])
        mask = torch.zeros(7, 7, dtype=torch.long)
        for nodes in self.retrieve.tolist():
            for j, node in enumerate(nodes):
                mask[node, nodes[:j+1]] = 1
        self.model.base_model.model.tree_mask = mask
        self.prefill_raw, self.tree_raw = torch.randn(1, 5, 8), torch.randn(1, 7, 8)
        self.head = torch.randn(8, 15)
        self.native_target_calls = 0
        self.selection_hints = []
        self.runtime = SimpleNamespace(initialize_tree=self.prefill, tree_decoding=self.tree,
            evaluate_posterior=self.strict, update_inference_inputs=self.update)
        # Callable object permits the same metadata hint interface as TerminalUpdate.
        fixture = self
        class Update:
            def note_selection(self, best, length, tokens):
                fixture.selection_hints.append((best, length, tokens))
            def __call__(self, *args, **kwargs):
                return fixture.update(*args, **kwargs)
        self.runtime.update_inference_inputs = Update()

    def prefill(self, ids):
        self.native_target_calls += 1
        self.prefill_hidden = self.norm(self.prefill_raw)
        return 'native-prefill-result'

    def tree(self, model, tokens, cache, positions, ids, retrieve):
        self.native_target_calls += 1
        self.tree_hidden = self.norm(self.tree_raw)
        self.logits = (self.tree_hidden @ self.head)[0, retrieve]
        return self.logits, self.tree_hidden, None

    @staticmethod
    def strict(logits, candidates, processor):
        lengths = (candidates[:, 1:] == logits[:, :-1].argmax(-1)).int().cumprod(-1).sum(-1)
        best = lengths.argmax()
        length = lengths[best]
        return best, length, logits[best, length]

    @staticmethod
    def update(ids, candidates, best, length):
        return (torch.cat((ids, candidates[int(best):int(best)+1, :int(length)+1]), -1),)

    def initialize(self, variant=None, audit=True):
        self.controller = LVSpecController(self.runtime, self.model, variant or {}, self.ids,
                                           torch.tensor([[1, 2, 4]]), audit=audit)
        self.runtime.initialize_tree(self.ids)
        result = self.runtime.tree_decoding(self.model, self.tokens, None, self.positions, self.ids, self.retrieve)
        return result[0], self.tokens[0, self.retrieve]


class LVSpecControllerTests(unittest.TestCase):
    def test_real_postnorm_image_rows_and_parent_rows_no_extra_target_calls(self):
        f = NativeFixture()
        logits, candidates = f.initialize(dict(loose_ratio=.7))
        c = f.controller
        try:
            self.assertTrue(torch.equal(c.visual, f.prefill_hidden[0, [1, 2]].float()))
            self.assertTrue(torch.equal(c.tree_hidden, f.tree_hidden[0]))
            self.assertFalse(torch.allclose(c.visual, f.prefill_raw[0, [1, 2]]))
            scores = visual_relevance(c.visual, c.tree_hidden)
            targets = logits[:, :-1].argmax(-1).tolist()
            expected = []
            for row, nodes in enumerate(f.retrieve.tolist()):
                mask = branch_low_relevance_mask(scores[nodes[:-1]], .7).tolist()
                expected.append(scan_branch(candidates[row].tolist(), targets[row], mask,
                    eos=1, committed_length=5, visible_boundary=133)[0])
            best, length, output = f.runtime.evaluate_posterior(logits, candidates, None)
            self.assertEqual((int(best), int(length)), (expected.index(max(expected)), max(expected)))
            self.assertTrue(torch.equal(output, logits[best, length]))
            self.assertEqual(f.native_target_calls, 2) # prefill and tree only
            self.assertEqual(f.selection_hints[-1], (int(best), int(length), candidates[best, :int(length)+1].tolist()))
            stats = c.final_stats()
            self.assertEqual(stats['extra_target_forwards'], 0)
            self.assertIsNone(stats['response_mismatch_budget'])
            self.assertEqual(stats['branch_ranking_sets'], 4)
            self.assertFalse(stats['introduced_probability_or_entropy_gate'])
        finally:
            c.restore()

    def test_explicit_strict_fallback_matches_native_selection_and_commit(self):
        f = NativeFixture()
        logits, candidates = f.initialize(dict(disable_relaxation=True))
        c = f.controller
        try:
            expected = f.strict(logits, candidates, None)
            actual = f.runtime.evaluate_posterior(logits, candidates, None)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected)))
            self.assertEqual(c.stats['similarity_computations'], 0)
            result = f.runtime.update_inference_inputs(f.ids, candidates, actual[0], actual[1])
            self.assertTrue(torch.equal(c.committed_ids, result[0]))
            self.assertEqual(c.stats['committed_mismatches'], 0)
            self.assertIsNone(c.tree_hidden)
        finally:
            c.restore()

    def test_tree_mask_position_and_candidate_alignment_audits_fail(self):
        f = NativeFixture()
        logits, candidates = f.initialize()
        c = f.controller
        try:
            bad = candidates.clone()
            bad[1, 1] += 1
            with self.assertRaises(ValueError):
                f.runtime.evaluate_posterior(logits, bad, None)
            f.model.base_model.model.tree_mask[1, 2] = 1
            with self.assertRaises(ValueError):
                f.runtime.tree_decoding(f.model, f.tokens, None, f.positions, f.ids, f.retrieve)
        finally:
            c.restore()

    def test_stochastic_processor_rejected_not_silently_used_as_greedy(self):
        f = NativeFixture()
        logits, candidates = f.initialize()
        try:
            with self.assertRaises(ValueError):
                f.runtime.evaluate_posterior(logits, candidates, lambda *args: args)
        finally:
            f.controller.restore()

    def test_restore_removes_hook_and_no_owner_cycle_with_gc_disabled(self):
        f = NativeFixture()
        _, _ = f.initialize()
        c, f.controller = f.controller, None
        self.assertEqual(len(f.norm._forward_hooks), 1)
        ref = weakref.ref(c)
        enabled = gc.isenabled()
        gc.disable()
        try:
            c.restore()
            self.assertEqual(len(f.norm._forward_hooks), 0)
            del c
            self.assertIsNone(ref())
        finally:
            if enabled:
                gc.enable()


if __name__ == '__main__':
    unittest.main()
