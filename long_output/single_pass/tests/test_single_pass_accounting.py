"""Scheduler/evidence tests; they do not instantiate a model or access a GPU."""
import ast
import hashlib
import importlib.util
import json
import pathlib
import random
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('single_pass_test_runtime', ROOT/'harness/run_single_pass.py')
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


class SinglePassAccountingTests(unittest.TestCase):
    def test_generation_methods_inherited_byte_identically(self):
        for name in ['generate', 'inputs', 'serial_ar', 'restore', 'verify_checkout']:
            self.assertIs(getattr(runtime.SinglePassRuntime, name), getattr(runtime.core.Runtime, name))
        source = ast.parse((ROOT/'harness/run_single_pass.py').read_text())
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'SinglePassRuntime')
        self.assertEqual({n.name for n in cls.body if isinstance(n, ast.FunctionDef)}, {'__init__', 'run'})

    def test_repeat_zero_schedule_matches_frozen_rule(self):
        cfg = dict(tasks=['mme', 'gqa', 'textvqa'], order_seed=2026103109,
                   development_methods={x: {} for x in ['ar', 'vispec', 'a', 'b']})
        for ordinal in range(20):
            expected = list(cfg['development_methods'])
            random.Random(cfg['order_seed']+1000003+ordinal*101+0*10007).shuffle(expected)
            self.assertEqual(runtime.schedule(cfg, 'gqa', {'ordinal': ordinal}), expected)

    def test_existing_rows_reject_duplicate_or_out_of_grid(self):
        row = {'question_id': 'q', 'method': 'ar', 'repeat': 0}
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory)/'x.jsonl'
            path.write_text(json.dumps(row)+'\n')
            self.assertEqual(len(runtime.validated_existing(path, {('q', 'ar', 0)})), 1)
            with self.assertRaises(AssertionError):
                runtime.validated_existing(path, {('q', 'ar', 1)})
            path.write_text((json.dumps(row)+'\n')*2)
            with self.assertRaises(AssertionError):
                runtime.validated_existing(path, {('q', 'ar', 0)})

    def test_stopped_parent_trailing_partial_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory)/'x.jsonl'
            valid = {'question_id': 'q', 'method': 'ar', 'repeat': 0}
            path.write_text(json.dumps(valid)+'\n{"repeat":')
            self.assertEqual(runtime.jsonl_rows(path), [(1, valid), (2, None)])
            with self.assertRaises(AssertionError):
                runtime.validated_existing(path, {('q', 'ar', 0)})

    def test_parent_core_source_matches_pinned_digest(self):
        self.assertEqual(hashlib.sha256(runtime.CORE.read_bytes()).hexdigest(), runtime.CORE_SHA256)


if __name__ == '__main__':
    unittest.main()
