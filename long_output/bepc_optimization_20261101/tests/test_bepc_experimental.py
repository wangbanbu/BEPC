"""CPU invariants for isolated BEPC optimization; no dataset labels are read."""
from collections import Counter
from pathlib import Path
import math
import random
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'harness'))
import refined_controller as frozen
from refined_math import score_components
from speed_math import visual_first_decision
from bepc_experimental import (
    ExperimentalBEPC, adjust_rescue_result, asymmetric_bound_decision,
    asymmetric_visual_decision, strict_exhausted_selection)


def _reference_selection(rows, targets, eos, committed, boundary):
    stats = Counter()
    values = [frozen.budgeted_scan(
        row, target, eos=eos, committed_length=committed,
        visible_boundary=boundary, response_used=4, response_budget=4,
        round_budget=2,
        allowed=lambda _: (_ for _ in ()).throw(AssertionError('Gate called after budget')),
        stats=stats) for row, target in zip(rows, targets)]
    best = max(range(len(values)), key=lambda index: values[index][0])
    return (best, values[best][0]), stats


class SelectionTests(unittest.TestCase):
    def test_randomized_exhausted_scan_matches_frozen(self):
        rng = random.Random(341901)
        for _ in range(4000):
            count, width = rng.randrange(1, 13), rng.randrange(1, 9)
            rows, targets = [], []
            for _ in range(count):
                row = [rng.randrange(7) for _ in range(width)]
                # Valid trailing padding, plus frequent matches and EOS edges.
                first_pad = rng.randrange(1, width + 1)
                row[first_pad:] = [-1] * (width - first_pad)
                target = [row[j + 1] if rng.random() < .65 and row[j + 1] >= 0
                          else rng.randrange(7) for j in range(width - 1)]
                rows.append(row)
                targets.append(target)
            committed, boundary = rng.randrange(1, 100), rng.randrange(1, 110)
            reference, reference_stats = _reference_selection(rows, targets, 6, committed, boundary)
            stats = Counter()
            actual = strict_exhausted_selection(
                rows, targets, eos=6, committed_length=committed,
                visible_boundary=boundary, stats=stats)
            self.assertEqual(actual, reference)
            self.assertEqual(stats, reference_stats)

    def test_first_longest_and_native_post_eos_matches(self):
        rows = [[6, 6, 3, -1], [2, 4, 5, -1], [1, 4, 5, -1]]
        targets = [[6, 3, 0], [4, 5, 0], [4, 5, 0]]
        # The frozen rule accepts exact matches after EOS/visible boundary.
        self.assertEqual(strict_exhausted_selection(
            rows, targets, eos=6, committed_length=99, visible_boundary=100), (0, 2))


class BoundTests(unittest.TestCase):
    def test_symmetric_helpers_are_verbatim_frozen(self):
        for g in (-5., .4999, .5, .5001, 1., 1.4999, 1.5, 1.5001, 4.):
            self.assertEqual(asymmetric_bound_decision(g, 1., .5, .5, 1.),
                             frozen.bound_decision(g, 1., .5, 1.))
            for m_u in (-10., -.3, -1e-5, 0., 1e-5, .3, 10.):
                self.assertEqual(asymmetric_visual_decision(
                    g, 1., m_u, gamma_rescue=.5, gamma_protection=.5),
                    visual_first_decision(g, 1., m_u))

    def test_asymmetric_bounds_contain_independent_scores(self):
        rng = random.Random(9537)
        for _ in range(20000):
            g, u, p = rng.uniform(0, 3), rng.uniform(-4, 4), rng.uniform(-4, 4)
            valid = rng.random() > .1
            active = valid and u * p < 0 and min(abs(u), abs(p)) >= 1e-4
            gamma = .625 if p > 0 else .5
            delta = gamma * math.copysign(min(abs(u), abs(p), 1.), p) if active else 0.
            score = g - delta
            global_decision = asymmetric_bound_decision(g, 1., .625, .5, 1.)
            if global_decision is not None:
                self.assertEqual(global_decision, score <= 1.)
            bounds = asymmetric_visual_decision(g, 1., u, gamma_rescue=.625,
                                                 gamma_protection=.5)
            self.assertLessEqual(bounds['lower'] - 1e-12, score)
            self.assertGreaterEqual(bounds['upper'] + 1e-12, score)
            if bounds['decision'] is not None:
                self.assertEqual(bounds['decision'], score <= 1.)

    def test_boundaries_do_not_shortcut_and_invalid_values(self):
        self.assertIsNone(asymmetric_bound_decision(.5, 1., .625, .5, 1.))
        self.assertIsNone(asymmetric_bound_decision(1.625, 1., .625, .5, 1.))
        self.assertIsNone(asymmetric_bound_decision(float('nan'), 1., .625, .5, 1.))
        for gamma in (-.1, 1.1, float('inf')):
            with self.assertRaises(ValueError):
                asymmetric_bound_decision(1., 1., gamma, .5, 1.)

    def test_actual_fp32_readouts_and_rescue_only_change(self):
        gen = torch.Generator().manual_seed(701)
        for _ in range(500):
            h = torch.randn(8, generator=gen)
            U = torch.eye(8)[:, :2]
            P = torch.eye(8)[:, 2:4]
            h_u, h_p = U @ (U.T @ h), P @ (P.T @ h)
            state = dict(U=U, P=P, h=h, h_U=h_u, h_P=h_p,
                         h_R=h - h_u - h_p, valid=True)
            a, d = torch.randn(2, 8, generator=gen)
            g = torch.rand((), generator=gen) * 2
            base = score_components(state, a, d, torch.ones(8), 1e-6, g,
                                    gamma=.5, residual_attenuation=False)
            self.assertIs(adjust_rescue_result(base, g, gamma_rescue=.5,
                                               gamma_protection=.5), base)
            new = adjust_rescue_result(base, g, gamma_rescue=.625,
                                       gamma_protection=.5)
            if bool(base['rescue']):
                self.assertTrue(torch.equal(new['delta'], base['delta'] * 1.25))
                self.assertLessEqual(float(new['score']), float(base['score']))
            else:
                self.assertTrue(torch.equal(new['delta'], base['delta']))
                self.assertTrue(torch.equal(new['score'], base['score']))
            bound = asymmetric_visual_decision(float(g), 1., float(base['m_U']),
                                              gamma_rescue=.625, gamma_protection=.5)
            score = float(new['score'])
            self.assertGreaterEqual(score + 1e-5, bound['lower'])
            self.assertLessEqual(score - 1e-5, bound['upper'])
            if bound['decision'] is not None:
                self.assertEqual(bound['decision'], score <= 1.)
            self.assertTrue(torch.isfinite(new['score']))


def make_controller(gamma_rescue=.5):
    block = torch.nn.Identity()
    base = SimpleNamespace(
        config=SimpleNamespace(image_token_id=5),
        model=SimpleNamespace(layers=[block], tree_mask=torch.tril(torch.ones(3, 3))),
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
    originals = dict(vars(runtime))
    ids = torch.tensor([[5, 1, 2]])
    controller = ExperimentalBEPC(
        runtime, model, dict(family='REFINED_BEPC', experimental_bepc=True,
                            gamma_rescue=gamma_rescue, gamma_protection=.5,
                            residual_attenuation=False), ids, torch.tensor([[1, 2, 2]]))
    return controller, block, originals, ids


class LifecycleTests(unittest.TestCase):
    def test_budget_commit_retires_hook_and_skips_margin(self):
        controller, block, originals, ids = make_controller()
        self.assertEqual(len(block._forward_hooks), 1)
        controller.prefill(ids)
        retrieve = torch.tensor([[0, 1, 2]])
        candidates = torch.tensor([[3, 4, 1]])
        controller.tree(controller.model, candidates, None, torch.arange(3), ids, retrieve)
        logits = torch.zeros(1, 3, 7)
        logits[0, 0, 4], logits[0, 1, 0] = 5., 5.
        controller.committed_mismatches = 3
        controller.allowed = lambda *args: True
        best, length, _ = controller.evaluate(logits, candidates, None)
        self.assertEqual((int(best), int(length)), (0, 2))
        updated = controller.update(ids, candidates, best, length, retrieve)[0]
        self.assertEqual(controller.committed_mismatches, 4)
        self.assertIsNone(controller.epcc)
        self.assertEqual(len(block._forward_hooks), 0)
        self.assertEqual(controller.stats['observer_retirements'], 1)
        counters = dict(controller._retired_observer_counters)
        controller.tree(controller.model, candidates, None, torch.arange(3), updated, retrieve)
        with mock.patch.object(frozen, 'native_margins', side_effect=AssertionError('Margin requested')):
            best, length, _ = controller.evaluate(logits, candidates, None)
        self.assertEqual((int(best), int(length)), (0, 1))
        controller.update(updated, candidates, best, length, retrieve)
        self.assertEqual(controller._retired_observer_counters, counters)
        self.assertEqual(controller.stats['exhausted_fast_rounds'], 1)
        self.assertEqual(controller.stats['skipped_native_margin_rounds'], 1)
        self.assertEqual(controller.stats['observer_retirements'], 1)
        final = controller.final_stats()
        self.assertEqual(final['observer_context_counters'], counters)
        self.assertEqual(final['committed_response_mismatches'], 4)
        controller.restore()
        for key, value in originals.items():
            self.assertIs(getattr(controller.runtime, key), value)
        self.assertEqual(len(block._forward_hooks), 0)

    def test_restoration_without_budget_exhaustion(self):
        controller, block, originals, _ = make_controller(.625)
        controller.restore()
        self.assertEqual(len(block._forward_hooks), 0)
        for key, value in originals.items():
            self.assertIs(getattr(controller.runtime, key), value)


if __name__ == '__main__':
    unittest.main(verbosity=2)
