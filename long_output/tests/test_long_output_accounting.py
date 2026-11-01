"""Actual-runner accounting at EOS and native block budget boundaries; CPU only."""
from __future__ import annotations

import ast
import json
import os
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from test_runner_contract import fake_tree_runtime

ROOT = Path(__file__).resolve().parents[1]


def record(raw, acceptance, budget=512):
    runtime, inputs, sample, variant = fake_tree_runtime('vispec')
    runtime.cfg.update(max_new_tokens=budget, prompt_kind='task_long')
    variant['compact_logits'] = False

    def generate(**kwargs):
        runtime.counts = dict(target=len(acceptance)+1, vision=1)
        output = torch.cat((inputs.input_ids, torch.tensor([raw])), dim=-1)
        return output, len(raw), len(acceptance)-1, acceptance, 0.001

    runtime.model.specgenerate = generate
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, CUDA_VISIBLE_DEVICES='cpu-contract'))
        for name in ('synchronize', 'reset_peak_memory_stats'):
            stack.enter_context(patch.object(torch.cuda, name, return_value=None))
        for name in ('memory_allocated', 'memory_reserved', 'max_memory_allocated', 'max_memory_reserved'):
            stack.enter_context(patch.object(torch.cuda, name, return_value=0))
        stack.enter_context(patch.object(torch, 'manual_seed', return_value=None))
        return runtime.generate(sample, inputs, 'vispec', variant, 0, 0, False, 'same-input')


class LongOutputAccounting(unittest.TestCase):
    def test_eos_inside_accepted_draft_keeps_native_and_visible_counts(self):
        row = record([6, 4, 1, 8], [3])
        self.assertEqual(row['prompt_kind'], 'task_long')
        self.assertEqual(row['returned_token_ids'], [6, 4])
        self.assertEqual(row['completion_tokens'], 3)
        self.assertEqual(row['raw_generated_tokens'], 4)
        self.assertEqual(row['raw_accepted_draft_tokens'], 3)
        self.assertEqual(row['effective_accepted_draft_tokens'], 2)
        self.assertEqual(row['mat_native_draft_only'], 3)
        self.assertEqual(row['mat_effective_draft_only'], 2)
        self.assertEqual(row['mat_effective_with_root'], 3)
        self.assertEqual(row['post_eos_raw_tokens'], 1)

    def test_budget_boundary_retains_native_after_block_overshoot(self):
        row = record([6, 4, 7, 8, 9, 10], [2, 2], budget=4)
        self.assertEqual(row['raw_token_ids'], [6, 4, 7, 8, 9, 10])
        self.assertEqual(row['returned_token_ids'], [6, 4, 7, 8])
        self.assertTrue(row['cap_reached'])
        self.assertEqual(row['budget_excluded_raw_tokens'], 2)
        self.assertEqual(row['raw_verification_rounds'], 2)
        self.assertEqual(row['effective_verification_rounds'], 2)
        self.assertEqual(row['mat_native_draft_only'], 2)
        self.assertEqual(row['mat_effective_draft_only'], 1)

    def test_eos_after_budget_does_not_turn_truncated_view_into_completed(self):
        row = record([6, 4, 7, 1], [3], budget=3)
        self.assertTrue(row['cap_reached'])
        self.assertIsNone(row['eos_index'])
        self.assertEqual(row['raw_first_eos_index'], 3)
        self.assertEqual(row['completion_tokens'], 3)
        self.assertEqual(row['mat_effective_draft_only'], 2)

    def test_first_root_eos_has_no_effective_draft_acceptance(self):
        row = record([1, 4, 7], [2])
        self.assertEqual(row['returned_token_ids'], [])
        self.assertEqual(row['completion_tokens'], 1)
        self.assertEqual(row['mat_effective_draft_only'], 0)
        self.assertEqual(row['mat_effective_with_root'], 1)
        self.assertEqual(row['post_eos_raw_tokens'], 2)

    def test_all_other_runtime_files_are_byte_identical_to_short_run(self):
        names = json.loads((ROOT.parent/'configs/design_config.json').read_text())['harness_sha256']
        for name in names:
            if name == 'run_benchmark.py':
                continue
            source = ROOT/'harness'/name
            short = ROOT.parent/'harness'/source.name
            self.assertTrue(short.exists(), source.name)
            self.assertEqual(source.read_bytes(), short.read_bytes(), source.name)

    def test_serial_ar_and_native_restore_functions_are_structurally_unchanged(self):
        def functions(path):
            module = ast.parse(path.read_text())
            cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'Runtime')
            return {n.name: ast.dump(n, include_attributes=False) for n in cls.body if isinstance(n, ast.FunctionDef)}
        short = functions(ROOT.parent/'harness/run_benchmark.py')
        long = functions(ROOT/'harness/run_benchmark.py')
        for name in ('serial_ar', 'restore', 'wrap_evaluate', 'verify_checkout', 'run', 'main'):
            self.assertEqual(long[name], short[name], name)


if __name__ == '__main__':
    unittest.main()
