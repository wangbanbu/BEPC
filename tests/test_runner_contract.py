"""CPU contracts executing actual runner/position AST with fake model backends.

No checkpoint imports, ResourceGuard, nvidia-smi, CUDA initialization or kernels.
The tests extract actual function bodies, rather than copy their loop logic.
"""
from __future__ import annotations

import ast
import copy
import gc
import os
import sys
import time
import unittest
import weakref
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT
sys.path.insert(0, str(ROOT/'harness'))
from compact_logits import CompactTreeLogits
from refined_controller import RefinedController


class Inputs(dict):
    def __getattr__(self, key):
        return self[key]


def actual_position_helper():
    """Compile the two actual custom-Qwen cache/mRoPE inference if blocks."""
    path = PROJECT/'upstream/ViSpec/vispec/model/modeling_qwen2_5_vl_kv.py'
    tree = ast.parse(path.read_text())
    model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Qwen2_5_VLForConditionalGeneration')
    forward = next(n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    selected = []
    for node in forward.body:
        if not isinstance(node, ast.If):
            continue
        names = {x.id for x in ast.walk(node.test) if isinstance(x, ast.Name)}
        if {'use_cache', 'past_key_values', 'cache_position'} <= names or {'position_ids', 'attention_mask'} <= names:
            selected.append(copy.deepcopy(node))
    if len(selected) != 2:
        raise AssertionError('Actual native cache/mRoPE inference contract changed')
    signature = ast.parse('def positions(self, input_ids, inputs_embeds, past_key_values, use_cache=True, cache_position=None, position_ids=None, attention_mask=None, image_grid_thw=None, video_grid_thw=None, second_per_grid_ts=None):\n    pass').body[0]
    signature.body = selected + ast.parse('return position_ids, cache_position').body
    module = ast.fix_missing_locations(ast.Module(body=[signature], type_ignores=[]))
    namespace = {'torch': torch}
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace['positions']


NATIVE_POSITION_HELPER = actual_position_helper()


def actual_runtime_methods(tree_decoding):
    """Compile actual pure methods without running the runner's module imports."""
    path = ROOT/'harness/run_benchmark.py'
    tree = ast.parse(path.read_text())
    source_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Runtime')
    selected = {'restore', 'wrap_evaluate', 'serial_ar', 'generate'}
    cls = copy.deepcopy(source_class)
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in selected]
    clean = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'clean'))
    namespace = dict(torch=torch, time=time, os=os, weakref=weakref,
                     RefinedController=RefinedController,
                     CompactTreeLogits=CompactTreeLogits, tree_decoding=tree_decoding)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[clean, cls],type_ignores=[])), str(path),'exec'),namespace)
    return namespace['Runtime']


class FakeVision(torch.nn.Module):
    def forward(self, pixels):
        return pixels


class FakeSerialTarget(torch.nn.Module):
    def __init__(self, predictions, delta=-3):
        super().__init__()
        self.predictions, self.delta = predictions, delta
        self.visual = FakeVision()
        self.rope_deltas = None
        self.positions, self.cache_objects, self.call_keys, self.before_lengths = [], [], [], []
        self.model = SimpleNamespace(tree_mask='stale tree',tree_mode='stale tree')

    def get_rope_index(self, input_ids, image_grid_thw, video_grid_thw,
                       second_per_grid_ts, attention_mask):
        size = input_ids.shape[1]
        positions = (torch.arange(size)+self.delta).view(1,1,-1).expand(3,1,-1)
        return positions, torch.tensor([[self.delta]])

    def forward(self, input_ids, past_key_values, **kwargs):
        prefill = 'pixel_values' in kwargs
        if prefill:
            self.prediction_index = 0
            self.visual(kwargs['pixel_values'])
        before = int(past_key_values[0][0].current_length)
        embeds = torch.zeros(1,input_ids.shape[1],4)
        positions, _ = NATIVE_POSITION_HELPER(self,input_ids,embeds,past_key_values,
            use_cache=kwargs['use_cache'],attention_mask=kwargs.get('attention_mask'),
            image_grid_thw=kwargs.get('image_grid_thw'))
        self.positions.append(positions.clone())
        self.cache_objects.append(id(past_key_values))
        self.before_lengths.append(before)
        self.call_keys.append(set(kwargs))
        next_token = self.predictions[min(self.prediction_index,len(self.predictions)-1)]
        self.prediction_index += 1
        logits = torch.zeros(1,input_ids.shape[1],32)
        logits[:,:,-1] = -1
        logits[:,:,next_token] = 10
        past_key_values[0][0].current_length.add_(input_ids.shape[1])
        return SimpleNamespace(logits=logits)


class FakeModule(SimpleNamespace):
    def __init__(self):
        super().__init__()
        self.allocations = self.resets = 0

    def reset_tree_mode(self, model):
        self.resets += 1
        model.base_model.model.tree_mask = model.base_model.model.tree_mode = None

    def initialize_past_key_values(self, model):
        self.allocations += 1
        lengths = torch.zeros(1,dtype=torch.long)
        cache = [[SimpleNamespace(current_length=lengths[0])]]
        return cache, [torch.zeros(1)], lengths


def fake_ar_runtime(predictions, *, budget=8, loop_count=20):
    runtime_type = actual_runtime_methods(lambda *args: None)
    runtime = runtime_type.__new__(runtime_type)
    runtime.module = FakeModule()
    runtime.model = SimpleNamespace(base_model=FakeSerialTarget(predictions),
                                   spec_layer=SimpleNamespace(total_tokens=2))
    runtime.tokenizer = SimpleNamespace(eos_token_id=1)
    # Deliberately differ by one, just like native Draft total_tokens=total_token-1.
    runtime.cfg = dict(max_length=loop_count+2+10,tree=dict(total_token=3),max_new_tokens=budget)
    runtime.counts = {}
    def count(mod,args,out,key):
        runtime.counts[key]=runtime.counts.get(key,0)+1
    runtime.model.base_model.register_forward_hook(lambda m,a,o:count(m,a,o,'target'))
    runtime.model.base_model.visual.register_forward_hook(lambda m,a,o:count(m,a,o,'vision'))
    inputs=Inputs(input_ids=torch.tensor([[8,9,10,11,12]]),pixel_values=torch.ones(1),
                  attention_mask=torch.ones(1,5,dtype=torch.long),image_grid_thw=torch.tensor([[1,4,4]]))
    return runtime, inputs


class SerialARContracts(unittest.TestCase):
    def call(self,runtime,inputs):
        with patch.object(torch.cuda,'synchronize',return_value=None):
            return runtime.serial_ar(inputs)

    def test_incremental_native_mrope_positions_and_KV_reuse(self):
        runtime,inputs=fake_ar_runtime([4,5,6,1])
        out,count,index,acceptance,decode=self.call(runtime,inputs)
        self.assertEqual(out[0,5:].tolist(),[4,5,6,1])
        self.assertEqual((count,index,acceptance),(4,3,[0,0,0,0]))
        target=runtime.model.base_model
        self.assertEqual([int(p[0,0,0]) for p in target.positions[1:]],[2,3,4])
        self.assertEqual([tuple(p.shape) for p in target.positions[1:]],[(3,1,1)]*3)
        self.assertEqual(target.before_lengths,[0,5,6,7])
        self.assertEqual(len(set(target.cache_objects)),1)
        self.assertEqual(runtime.counts,dict(target=4,vision=1))
        self.assertIsNone(target.model.tree_mask)
        self.assertIsNone(target.model.tree_mode)
        self.assertGreaterEqual(decode,0)
        self.assertTrue(all('pixel_values' not in keys for keys in target.call_keys[1:]))

    def test_immediate_EOS_has_no_extra_forward(self):
        runtime,inputs=fake_ar_runtime([1,5])
        out,count,index,acceptance,_=self.call(runtime,inputs)
        self.assertEqual(out[0,5:].tolist(),[1])
        self.assertEqual((count,index,acceptance),(1,0,[0]))
        self.assertEqual(runtime.counts,dict(target=1,vision=1))

    def test_budget_preserves_native_strict_greater_than_stop(self):
        runtime,inputs=fake_ar_runtime([5],budget=2)
        out,count,index,acceptance,_=self.call(runtime,inputs)
        self.assertEqual(out[0,5:].tolist(),[5,5,5])
        self.assertEqual((count,index,len(acceptance)),(3,2,3))
        self.assertEqual(runtime.counts,dict(target=3,vision=1))

    def test_loop_upper_bound_uses_actual_native_draft_total_tokens(self):
        runtime,inputs=fake_ar_runtime([5],budget=100,loop_count=2)
        _,count,_,_,_=self.call(runtime,inputs)
        self.assertEqual(count,2)
        self.assertEqual(runtime.counts['target'],2)

    def test_existing_KV_reset_for_next_request_without_reallocation(self):
        runtime,inputs=fake_ar_runtime([4,1])
        self.call(runtime,inputs)
        cache_id=id(runtime.model.past_key_values)
        runtime.counts={}
        self.call(runtime,inputs)
        self.assertEqual(id(runtime.model.past_key_values),cache_id)
        self.assertEqual(runtime.module.allocations,1)
        self.assertEqual(runtime.module.resets,2)
        self.assertEqual(runtime.model.base_model.before_lengths,[0,5,0,5])
        self.assertEqual(runtime.counts,dict(target=2,vision=1))


class FakeTreeModel:
    def __init__(self):
        d=16
        self.tokenizer=SimpleNamespace(eos_token_id=1,all_special_ids=[1,97,98,99],
            decode=lambda ids,skip_special_tokens=True:' '.join(map(str,ids)))
        self.base_model=SimpleNamespace(config=SimpleNamespace(image_token_id=99),
            model=SimpleNamespace(layers=[torch.nn.Identity()],norm=SimpleNamespace(weight=torch.ones(d),variance_epsilon=1e-6),tree_mask=torch.tensor([[1,0],[1,1]])),
            lm_head=torch.nn.Linear(d,100,bias=False))

    def specgenerate(self, **kwargs):
        inputs=kwargs['input_ids']
        self.runtime.module.initialize_tree(inputs)
        tokens=torch.tensor([[6,4]])
        retrieve=torch.tensor([[0,1]])
        logits,_,_=self.runtime.module.tree_decoding(self,tokens,None,torch.tensor([0,1]),inputs,retrieve)
        best,length,_=self.runtime.module.evaluate_posterior(logits,tokens,None)
        updated=self.runtime.module.update_inference_inputs(inputs,tokens,best,length,retrieve)
        return updated[0],int(length)+1,0,[int(length)],.001


def fake_tree_runtime(family):
    runtime=None
    model=FakeTreeModel()
    def native_tree(model,tokens,cache,positions,ids,retrieve):
        runtime.counts['target']=runtime.counts.get('target',0)+1
        model.base_model.model.layers[-1](torch.arange(tokens.numel()*16).reshape(1,-1,16).float()/17)
        native=torch.zeros(2,100);native[:,2]=1
        return CompactTreeLogits(native,retrieve),None,None
    runtime_type=actual_runtime_methods(native_tree)
    runtime=runtime_type.__new__(runtime_type)
    model.runtime=runtime
    runtime.model,runtime.tokenizer=model,model.tokenizer
    def prefill(ids):
        runtime.counts['target']=runtime.counts.get('target',0)+1
        runtime.counts['vision']=runtime.counts.get('vision',0)+1
        model.base_model.model.layers[-1](torch.arange(ids.numel()*16).reshape(1,-1,16).float()/31)
    def update(ids,candidates,best,length,retrieve):
        return (torch.cat((ids,candidates[int(best):int(best)+1,:int(length)+1]),-1),)
    runtime.original=dict(initialize_tree=prefill,tree_decoding=native_tree,
        evaluate_posterior=lambda *args:None,update_inference_inputs=update)
    runtime.module=SimpleNamespace(**runtime.original)
    runtime.guard=SimpleNamespace(assert_free=lambda:None)
    runtime.cfg=dict(seed=1701,max_new_tokens=128,max_length=2048)
    runtime.meta=dict(config_sha256='cpu-contract')
    runtime.phase,runtime.shard='contract',0
    runtime.task='coco'
    inputs=Inputs(input_ids=torch.tensor([[98,99,99,99,99,10,11,12,13,97]]),image_grid_thw=torch.tensor([[1,4,4]]))
    sample=dict(ordinal=0,image_id=7,group_id='7',group_ordinal=0,question_id=7,image_path='cpu-only',image_sha256='cpu-only',prompt_text='caption')
    variant=dict(family=family,compact_logits=True,threshold=1.)
    return runtime,inputs,sample,variant


class LifecycleContracts(unittest.TestCase):
    def test_actual_generate_module_only_wrapper_and_immediate_release(self):
        for family in ('T2','REFINED_BEPC'):
            with self.subTest(family=family):
                runtime,inputs,sample,variant=fake_tree_runtime(family)
                previous=gc.isenabled()
                gc.disable()
                try:
                    with ExitStack() as stack:
                        stack.enter_context(patch.dict(os.environ,CUDA_VISIBLE_DEVICES='cpu-contract'))
                        for name in ('synchronize','reset_peak_memory_stats'):
                            stack.enter_context(patch.object(torch.cuda,name,return_value=None))
                        for name in ('memory_allocated','memory_reserved','max_memory_allocated','max_memory_reserved'):
                            stack.enter_context(patch.object(torch.cuda,name,return_value=0))
                        stack.enter_context(patch.object(torch,'manual_seed',return_value=None))
                        row=runtime.generate(sample,inputs,family,variant,0,0,False,'same-input')
                    self.assertEqual(row['request_lifecycle']['live_controllers_after_request'],0)
                    self.assertEqual(row['request_lifecycle']['live_logits_after_request'],0)
                    self.assertEqual(row['model_calls'],dict(target=2,vision=1))
                    self.assertEqual(row['signal_stats']['compact_logits']['audit_reference_materializations'],0)
                    self.assertTrue(all(ref() is None for ref in runtime.logit_refs))
                    self.assertEqual(len(runtime.model.base_model.model.layers[-1]._forward_hooks),0)
                    for name,original in runtime.original.items():
                        self.assertIs(getattr(runtime.module,name),original)
                finally:
                    if previous:gc.enable()

    def test_source_never_attaches_bound_method_wrapper_to_controller_instance(self):
        tree=ast.parse((ROOT/'harness/run_benchmark.py').read_text())
        writes=[]
        for node in ast.walk(tree):
            if isinstance(node,(ast.Assign,ast.AnnAssign,ast.AugAssign)):
                targets=node.targets if isinstance(node,ast.Assign) else [node.target]
                writes.extend(target for target in targets if isinstance(target,ast.Attribute)
                              and isinstance(target.value,ast.Name) and target.value.id=='controller'
                              and target.attr=='evaluate')
        self.assertEqual(writes,[])


if __name__=='__main__':
    unittest.main()
