"""Critical-path tests for the new no-observer, no-response-cap Margin arm."""
from collections import Counter
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'harness'))
from compact_logits import CompactTreeLogits
from margin_controller import NoResponseCapMargin
from refined_controller import native_margins


def fixture():
    block=torch.nn.Identity()
    model=SimpleNamespace(base_model=SimpleNamespace(config=SimpleNamespace(image_token_id=5),
        model=SimpleNamespace(layers=[block],tree_mask=torch.tril(torch.ones(4,4)))),
        tokenizer=SimpleNamespace(eos_token_id=6,all_special_ids=[5,6]))
    def prefill(ids,**kwargs):return block(torch.ones(1,ids.shape[-1],8))
    def tree(model,tokens,cache,positions,ids,retrieve):return block(torch.ones(1,tokens.shape[-1],8))
    def update(ids,candidates,best,length,retrieve,**kwargs):
        return (torch.cat((ids,candidates[int(best):int(best)+1,:int(length)+1]),-1),)
    runtime=SimpleNamespace(initialize_tree=prefill,tree_decoding=tree,
        evaluate_posterior=lambda *args:None,update_inference_inputs=update)
    originals=dict(vars(runtime))
    ids=torch.tensor([[5,1,2]])
    variant=dict(family='T2',tau=1.,round_mismatch_budget=2,
        response_mismatch_budget=None,no_response_cap=True,visual_first_prune=False)
    obj=NoResponseCapMargin(runtime,model,variant,ids,torch.tensor([[1,2,2]]))
    return obj,block,originals,ids


class MarginTests(unittest.TestCase):
    def test_native_compact_and_expanded_FP32_boundary(self):
        # Parent logits come from the native dtype; each scalar is cast before
        # subtraction, even where the BF16 difference itself would be rounded.
        native=torch.tensor([[16.,14.9375,0.,0.,0.,0.,0.],
            [3.,2.,0.,0.,0.,0.,0.],[2.,1.0078125,0.,0.,0.,0.,0.],
            [3.,0.,0.,0.,0.,0.,0.]],dtype=torch.bfloat16)
        retrieve=torch.tensor([[0,1,2,3]])
        candidates=torch.tensor([[4,1,1,1]])
        compact=CompactTreeLogits(native,retrieve)
        targets=compact[:,:-1].argmax(-1)
        a=native_margins(compact,candidates,targets)
        b=native_margins(native[retrieve],candidates,targets)
        self.assertEqual(a.dtype,torch.float32)
        self.assertTrue(torch.equal(a,b))
        self.assertEqual(a.tolist(),[[1.0625,1.,.9921875]])

    def test_gate_boundary_nonfinite_and_no_representation_hooks(self):
        obj,block,originals,ids=fixture()
        try:
            self.assertTrue(math.isinf(obj.response_budget))
            self.assertIsNone(obj.epcc);self.assertIsNone(obj.visual);self.assertIsNone(obj.draft)
            self.assertFalse(block._forward_hooks)
            obj.candidate_rows=[[3,1]];obj.targets=[[4]];obj.retrieve_rows=[[0,1]]
            for value,expected in [(1.,True),(.999,True),(1.001,False),
                                   (float('nan'),False),(float('inf'),False)]:
                obj.decision_cache={}
                obj.margin_tensor=torch.tensor([[value]])
                obj.margin_rows=[[value]]
                self.assertEqual(obj.allowed(0,1,None,None),expected)
            stats=obj.final_stats()
            self.assertIsNone(stats['response_mismatch_budget'])
            json.dumps(stats,allow_nan=False)
        finally:obj.restore()
        for k,v in originals.items():self.assertIs(getattr(obj.runtime,k),v)

    def test_round_budget_unselected_branches_and_cumulative_over_four(self):
        obj,block,_,ids=fixture()
        try:
            obj.prefill(ids)
            # top1=4, draft=1 has margin .5, so two mismatches are accepted,
            # and the third is rejected by the shared per-round limit.
            candidates=torch.tensor([[3,1,1,1],[3,1,1,1]])
            retrieve=torch.tensor([[0,1,2,3],[0,1,2,3]])
            logits=torch.zeros(2,4,7);logits[:,:,4]=1.;logits[:,:,1]=.5
            for r in range(4):
                obj.tree(obj.model,candidates[:1],None,torch.arange(4),ids,retrieve)
                best,length,_=obj.evaluate(logits,candidates,None)
                self.assertEqual((int(best),int(length)),(0,2))
                self.assertEqual(obj.committed_mismatches,r*2)
                ids=obj.update(ids,candidates,best,length,retrieve)[0]
            self.assertEqual(obj.committed_mismatches,8)
            self.assertEqual(obj.stats['response_budget_binding'],0)
            self.assertFalse(block._forward_hooks)
            self.assertEqual(obj.stats['full_candidate_scores'],0)
        finally:obj.restore()


if __name__=='__main__':unittest.main(verbosity=2)
