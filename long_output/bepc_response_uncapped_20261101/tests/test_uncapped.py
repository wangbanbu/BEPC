"""CPU-only response-cap ablation invariants; no labels or checkpoints read."""
from collections import Counter
import json
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'harness'))
import refined_controller as frozen
from bepc_experimental import ExperimentalBEPC
from uncapped_controller import NoResponseCapBEPC


def make_controller(cls=NoResponseCapBEPC, **overrides):
    block = torch.nn.Identity()
    base = SimpleNamespace(
        config=SimpleNamespace(image_token_id=5),
        model=SimpleNamespace(layers=[block], tree_mask=torch.tril(torch.ones(4, 4))),
    )
    model = SimpleNamespace(base_model=base, tokenizer=SimpleNamespace(
        eos_token_id=6, all_special_ids=[5, 6]))

    def prefill(ids, **kwargs):
        return block(torch.ones(1, ids.shape[-1], 8))

    def tree(model, tokens, cache, positions, ids, retrieve):
        return block(torch.ones(1, tokens.shape[-1], 8))

    def update(ids, candidates, best, length, retrieve, **kwargs):
        return (torch.cat((ids, candidates[int(best):int(best) + 1, :int(length) + 1]), -1),)

    runtime = SimpleNamespace(initialize_tree=prefill, tree_decoding=tree,
                              evaluate_posterior=lambda *args: None,
                              update_inference_inputs=update)
    original = dict(vars(runtime))
    ids = torch.tensor([[5, 1, 2]])
    variant = dict(family='REFINED_BEPC', experimental_bepc=True,
                   gamma=.5, gamma_rescue=.5, gamma_protection=.5,
                   residual_attenuation=False, round_mismatch_budget=2,
                   response_mismatch_budget=None, no_response_cap=True)
    if cls is ExperimentalBEPC:
        variant.update(response_mismatch_budget=4, no_response_cap=False)
    variant.update(overrides)
    controller = cls(runtime, model, variant, ids, torch.tensor([[1, 2, 2]]))
    return controller, block, original, ids


def reference_scan(sequence, targets, *, eos, committed_length, visible_boundary,
                   allowed, round_budget=2):
    """Independent direct definition: every rule except a response-wide cap."""
    accepted = mismatches = 0
    eos_seen = sequence[0] == eos
    for j, draft in enumerate(sequence[1:]):
        target = targets[j]
        if draft < 0:
            break
        if draft != target:
            if (eos_seen or committed_length + j + 1 >= visible_boundary
                    or target == eos or draft == eos or mismatches >= round_budget
                    or not allowed(j + 1)):
                break
            mismatches += 1
        accepted += 1
        eos_seen |= draft == eos
    return accepted, mismatches


class ScannerTests(unittest.TestCase):
    def test_uncapped_scan_matches_independent_reference(self):
        rng = random.Random(110142)
        for _ in range(10000):
            width = rng.randrange(1, 13)
            sequence = [rng.randrange(7) for _ in range(width)]
            first_pad = rng.randrange(1, width + 1)
            sequence[first_pad:] = [-1] * (width - first_pad)
            targets = [sequence[j + 1] if rng.random() < .5 else rng.randrange(7)
                       for j in range(width - 1)]
            decisions = [rng.random() < .8 for _ in range(width)]
            kwargs = dict(eos=6, committed_length=rng.randrange(1, 110),
                          visible_boundary=rng.randrange(1, 115),
                          allowed=lambda column: decisions[column])
            actual = frozen.budgeted_scan(
                sequence, targets, response_used=rng.randrange(1000),
                response_budget=math.inf, round_budget=2, **kwargs)
            self.assertEqual(actual, reference_scan(sequence, targets, **kwargs))
            self.assertLessEqual(actual[1], 2)

    def test_padding_eos_boundary_and_round_limit(self):
        kwargs = dict(eos=6, committed_length=10, visible_boundary=100,
                      response_used=500, response_budget=math.inf,
                      round_budget=2, allowed=lambda _: True)
        self.assertEqual(frozen.budgeted_scan([0, 1, 2, 3], [4, 4, 4], **kwargs), (2, 2))
        self.assertEqual(frozen.budgeted_scan([0, 1, -1, -1], [1, 2, 3], **kwargs), (1, 0))
        self.assertEqual(frozen.budgeted_scan([0, 6, 2], [4, 2], **kwargs), (0, 0))
        self.assertEqual(frozen.budgeted_scan([0, 1, 2], [6, 2], **kwargs), (0, 0))
        self.assertEqual(frozen.budgeted_scan([6, 1, 2], [3, 2], **kwargs), (0, 0))
        # Preserve the frozen convention: exact post-EOS matches are scanned,
        # with native output clipping handling their visible semantics later.
        self.assertEqual(frozen.budgeted_scan([6, 1, 2], [1, 2], **kwargs), (2, 0))
        kwargs['visible_boundary'] = 11
        self.assertEqual(frozen.budgeted_scan([0, 1, 2], [3, 2], **kwargs), (0, 0))
        self.assertEqual(frozen.budgeted_scan([0, 1, 2], [1, 2], **kwargs), (2, 0))


class LifecycleTests(unittest.TestCase):
    def test_many_rounds_exceed_four_without_observer_retirement(self):
        controller, block, originals, ids = make_controller()
        try:
            self.assertTrue(math.isinf(controller.response_budget))
            self.assertIsNone(controller.variant['response_mismatch_budget'])
            self.assertEqual(controller.round_budget, 2)
            self.assertEqual(len(block._forward_hooks), 1)
            controller.prefill(ids)
            candidates = torch.tensor([[3, 1, 2, 3]])
            retrieve = torch.tensor([[0, 1, 2, 3]])
            logits = torch.zeros(1, 4, 7)
            logits[:, :, 4] = 5.
            controller.allowed = lambda *args: True
            for round_index in range(6):
                controller.tree(controller.model, candidates, None, torch.arange(4), ids, retrieve)
                best, length, _ = controller.evaluate(logits, candidates, None)
                self.assertEqual((int(best), int(length)), (0, 2))
                ids = controller.update(ids, candidates, best, length, retrieve)[0]
                self.assertEqual(controller.committed_mismatches, 2 * (round_index + 1))
                self.assertIsNotNone(controller.epcc)
                self.assertEqual(len(block._forward_hooks), 1)
            stats = controller.final_stats()
            self.assertEqual(stats['committed_response_mismatches'], 12)
            self.assertEqual(stats['observer_retirements'], 0)
            self.assertEqual(stats['exhausted_fast_rounds'], 0)
            self.assertEqual(stats['response_budget_binding'], 0)
            self.assertGreater(stats['row_budget_binding'], 0)
            self.assertIsNone(stats['response_mismatch_budget'])
            self.assertFalse(stats['response_cap_enabled'])
            self.assertFalse(stats['all_relaxed_methods_same_budget'])
            self.assertFalse(stats['budget_exhaustion_feature_retirement'])
            json.dumps(stats, allow_nan=False)
        finally:
            controller.restore()
        for key, value in originals.items():
            self.assertIs(getattr(controller.runtime, key), value)
        self.assertEqual(len(block._forward_hooks), 0)

    def test_first_longest_off_path_scan_does_not_spend_budget(self):
        controller, _, _, ids = make_controller()
        try:
            controller.prefill(ids)
            candidates = torch.tensor([[3, 1, 2, 3], [3, 4, 4, 4], [3, 4, 4, 4]])
            retrieve = torch.tensor([[0, 1, 2, 3], [0, 4, 5, 6], [0, 7, 8, 9]])
            tokens = torch.tensor([[3, 1, 2, 3, 4, 4, 4, 4, 4, 4]])
            controller.tree(controller.model, tokens, None, torch.arange(10), ids, retrieve)
            logits = torch.zeros(3, 4, 7)
            logits[:, :, 4] = 5.
            controller.allowed = lambda *args: True
            best, length, _ = controller.evaluate(logits, candidates, None)
            self.assertEqual((int(best), int(length)), (1, 3))
            self.assertEqual(controller._selected_mismatches, 0)
            self.assertEqual(controller.committed_mismatches, 0)
            self.assertEqual(controller.acceptance_paths[-1]['mismatches'], 0)
            # First branch examined two acceptable mismatches, but none commit.
            self.assertGreater(controller.stats['row_budget_binding'], 0)
        finally:
            controller.restore()

    def test_capped_control_still_stops_at_four(self):
        controller, block, _, ids = make_controller(ExperimentalBEPC)
        try:
            controller.prefill(ids)
            candidates = torch.tensor([[3, 1, 2, 3]])
            retrieve = torch.tensor([[0, 1, 2, 3]])
            logits = torch.zeros(1, 4, 7)
            logits[:, :, 4] = 5.
            controller.allowed = lambda *args: True
            for _ in range(2):
                controller.tree(controller.model, candidates, None, torch.arange(4), ids, retrieve)
                best, length, _ = controller.evaluate(logits, candidates, None)
                ids = controller.update(ids, candidates, best, length, retrieve)[0]
            self.assertEqual(controller.committed_mismatches, 4)
            self.assertIsNone(controller.epcc)
            self.assertEqual(len(block._forward_hooks), 0)
            controller.tree(controller.model, candidates, None, torch.arange(4), ids, retrieve)
            _, length, _ = controller.evaluate(logits, candidates, None)
            self.assertEqual(int(length), 0)
            self.assertEqual(controller.response_budget, 4)
        finally:
            controller.restore()

    def test_other_algorithm_changes_rejected(self):
        for overrides in (dict(no_response_cap=False), dict(response_mismatch_budget=8),
                          dict(round_mismatch_budget=3), dict(gamma_rescue=.625),
                          dict(gamma=.625), dict(kappa=2.), dict(eta=.01),
                          dict(tau=1.5), dict(rank_u=4), dict(rank_p=8),
                          dict(text_window=16), dict(visual_pool_size=32),
                          dict(correction_mode='rescue'), dict(residual_attenuation=True)):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                make_controller(**overrides)


if __name__ == '__main__':
    unittest.main(verbosity=2)
