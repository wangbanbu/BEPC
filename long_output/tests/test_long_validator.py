"""CPU-only independent validator corruption tests; no inference fixtures."""
from __future__ import annotations
import ast
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('long_validator',ROOT/'harness/validate_long.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)

def fixture():
    # Reuse only the already checked synthetic literal trace from the old
    # validator, never runtime/score outputs and never implementation math.
    tree=ast.parse(v.BASE_VALIDATOR.read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='self_test')
    nodes=[]
    for node in function.body:
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='row_contract':break
        nodes.append(node)
    namespace={}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'literal_fixture','exec'),namespace)
    row,sample,cfg,variant=[namespace[k] for k in ('row','sample','cfg','variant')]
    cfg['max_new_tokens']=512
    row.update(prompt_kind='task_long',raw_generated_tokens=2,returned_content_tokens=1,
        raw_accepted_draft_tokens=1,raw_verification_rounds=1,raw_committed_root_tokens=1,
        effective_committed_root_tokens=1,mat_native_draft_only=1.,mat_native_with_root=2.,
        mat_effective_draft_only=1.,mat_effective_with_root=2.,raw_first_eos_index=1,
        post_eos_raw_tokens=0,budget_excluded_raw_tokens=0,
        context_capacity_audit=dict(status='pass',prompt_tokens=1,max_new_tokens=512,
            conservative_tree_and_stop_headroom=68,conservative_maximum_context_tokens=581,native_target_KV_capacity=4096))
    return row,sample,cfg,variant

class LongValidatorContracts(unittest.TestCase):
    def check(self,row,sample,cfg,variant):v.row_contract(row,sample,variant,cfg,'main',0)
    def test_valid_native_trace(self):
        self.check(*fixture());self.assertNotIn('torch',sys.modules)
    def test_rejects_wrong_Draft_or_root_MAT(self):
        for key in ('mat_native_draft_only','mat_native_with_root','mat_effective_draft_only','mat_effective_with_root'):
            row,sample,cfg,variant=fixture();row[key]=10.
            with self.subTest(key=key),self.assertRaises(AssertionError):self.check(row,sample,cfg,variant)
    def test_rejects_corrupt_tail_counts(self):
        for key in ('raw_first_eos_index','post_eos_raw_tokens','budget_excluded_raw_tokens'):
            row,sample,cfg,variant=fixture();row[key]=55
            with self.subTest(key=key),self.assertRaises(AssertionError):self.check(row,sample,cfg,variant)
    def test_rejects_wrong_event_mapping(self):
        row,sample,cfg,variant=fixture();row['question_id']='other'
        with self.assertRaises(AssertionError):self.check(row,sample,cfg,variant)
    def test_rejects_wrong_Target_call_count(self):
        row,sample,cfg,variant=fixture();row['model_calls']['target']=3
        with self.assertRaises(AssertionError):self.check(row,sample,cfg,variant)
    def test_rejects_unsafe_context(self):
        row,sample,cfg,variant=fixture();row['context_capacity_audit']['native_target_KV_capacity']=580
        with self.assertRaises(AssertionError):self.check(row,sample,cfg,variant)
    def test_rejects_wrong_effective_root_or_token_count(self):
        for key in ('effective_committed_root_tokens','raw_generated_tokens','returned_content_tokens'):
            row,sample,cfg,variant=fixture();row[key]=7
            with self.subTest(key=key),self.assertRaises(AssertionError):self.check(row,sample,cfg,variant)
    def test_AR_MAT_is_null(self):
        row,sample,cfg,variant=fixture();variant['family']='AR';row.update(method='ar',acceptance_length=[0,0],
            native_index=1,raw_accepted_draft_tokens=0,raw_verification_rounds=2,
            raw_committed_root_tokens=2,effective_committed_root_tokens=2,
            effective_accepted_draft_tokens=0,effective_verification_rounds=2,
            model_calls=dict(target=2,draft=0,vision=1),mat_native_draft_only=None,
            mat_native_with_root=None,mat_effective_draft_only=None,mat_effective_with_root=None)
        self.check(row,sample,cfg,variant)
        row['mat_effective_draft_only']=0.
        with self.assertRaises(AssertionError):self.check(row,sample,cfg,variant)

if __name__=='__main__':unittest.main()
