"""Independent corruption tests for single-pass provenance and MAT contracts."""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('single_validator',ROOT/'harness/validate_single.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)

def fixture():
    tree=ast.parse(v.v.BASE_VALIDATOR.read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='self_test')
    nodes=[]
    for node in function.body:
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='row_contract':break
        nodes.append(node)
    namespace={};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'synthetic_trace','exec'),namespace)
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

def provenance_fixture():
    old=dict(question_id='q',repeat=0,config_sha256='old',seconds=1.23,raw_token_ids=[1,2])
    entry=dict(source_path='source',source_line=8,source_row_sha256=v.canonical_sha(old),source_config_sha256='old')
    row=dict(old,config_sha256='new',measurement_origin='reused_parent_repeat0',single_pass_provenance=dict(parent_path='source',parent_line=8,parent_record_sha256=v.canonical_sha(old),parent_config_sha256='old',normalized_fields={'config_sha256':{'original':'old','normalized':'new'}}))
    return row,old,entry,'new'

class SingleValidatorTests(unittest.TestCase):
    def test_valid_row_CPU_only(self):
        row,sample,cfg,variant=fixture();v.row_contract(row,sample,variant,cfg,'main',0)
        self.assertNotIn('torch',sys.modules)
    def test_wrong_MAT_rejected(self):
        for name in ('mat_native_draft_only','mat_native_with_root','mat_effective_draft_only','mat_effective_with_root'):
            row,sample,cfg,variant=fixture();row[name]=99
            with self.subTest(name=name),self.assertRaises(AssertionError):v.row_contract(row,sample,variant,cfg,'main',0)
    def test_bad_calls_or_context_rejected(self):
        row,sample,cfg,variant=fixture();row['model_calls']['target']=3
        with self.assertRaises(AssertionError):v.row_contract(row,sample,variant,cfg,'main',0)
        row,sample,cfg,variant=fixture();row['context_capacity_audit']['native_target_KV_capacity']=580
        with self.assertRaises(AssertionError):v.row_contract(row,sample,variant,cfg,'main',0)
    def test_import_preserves_all_original_measurements(self):v.imported_row_contract(*provenance_fixture())
    def test_import_changed_time_or_token_rejected(self):
        for name,value in [('seconds',9.1),('raw_token_ids',[3]),('repeat',1)]:
            row,old,entry,digest=provenance_fixture();row[name]=value
            with self.subTest(name=name),self.assertRaises(AssertionError):v.imported_row_contract(row,old,entry,digest)
    def test_import_source_mismatch_rejected(self):
        for name,value in [('parent_line',9),('parent_record_sha256','wrong'),('parent_config_sha256','wrong')]:
            row,old,entry,digest=provenance_fixture();row['single_pass_provenance'][name]=value
            with self.subTest(name=name),self.assertRaises(AssertionError):v.imported_row_contract(row,old,entry,digest)
    def test_analysis_count_is_4200_original5000draw_contract_kept(self):
        const=v.SingleAudit.analysis.__code__.co_consts
        self.assertIn(4200,const);self.assertNotIn(12600,const)
        self.assertIn(5000,const)

if __name__=='__main__':unittest.main()
