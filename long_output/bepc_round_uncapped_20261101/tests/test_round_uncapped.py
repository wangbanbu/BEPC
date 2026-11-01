"""Critical-path tests for the per-round cap removal; no model/labels read."""
from collections import Counter
import json
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest

import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'harness'))
from refined_controller import budgeted_scan
from round_uncapped_controller import NoRoundCapBEPC
from uncapped_controller import NoResponseCapBEPC


def fixture(unbounded=True):
    block=torch.nn.Identity()
    model=SimpleNamespace(base_model=SimpleNamespace(config=SimpleNamespace(image_token_id=5),
        model=SimpleNamespace(layers=[block],tree_mask=torch.tril(torch.ones(5,5)))),
        tokenizer=SimpleNamespace(eos_token_id=6,all_special_ids=[5,6]))
    def prefill(ids,**kwargs):return block(torch.ones(1,ids.shape[-1],8))
    def tree(model,tokens,cache,positions,ids,retrieve):return block(torch.ones(1,tokens.shape[-1],8))
    def update(ids,candidates,best,length,retrieve,**kwargs):
        return (torch.cat((ids,candidates[int(best):int(best)+1,:int(length)+1]),-1),)
    runtime=SimpleNamespace(initialize_tree=prefill,tree_decoding=tree,
        evaluate_posterior=lambda *args:None,update_inference_inputs=update)
    originals=dict(vars(runtime));ids=torch.tensor([[5,1,2]])
    variant=dict(family='REFINED_BEPC',experimental_bepc=True,tau=1.,
        gamma=.5,gamma_rescue=.5,gamma_protection=.5,kappa=1.,eta=1e-4,
        residual_attenuation=False,round_mismatch_budget=None if unbounded else 2,
        response_mismatch_budget=None,no_response_cap=True,no_round_cap=unbounded)
    cls=NoRoundCapBEPC if unbounded else NoResponseCapBEPC
    obj=cls(runtime,model,variant,ids,torch.tensor([[1,2,2]]))
    return obj,block,originals,ids


class RoundCapTests(unittest.TestCase):
    def test_four_mismatches_allowed_only_in_new_arm_and_history_commits(self):
        for unbounded,expected in [(False,2),(True,4)]:
            obj,block,originals,ids=fixture(unbounded)
            try:
                obj.prefill(ids)
                candidates=torch.tensor([[3,1,2,3,1]])
                retrieve=torch.tensor([[0,1,2,3,4]])
                logits=torch.zeros(1,5,7);logits[:,:,4]=.25
                # Uses the real BEPC global bound / finite margin gate.
                for i in range(3):
                    obj.tree(obj.model,candidates,None,torch.arange(5),ids,retrieve)
                    best,length,_=obj.evaluate(logits,candidates,None)
                    self.assertEqual(int(length),expected)
                    self.assertEqual(obj.committed_mismatches,i*expected)
                    ids=obj.update(ids,candidates,best,length,retrieve)[0]
                self.assertEqual(obj.committed_mismatches,expected*3)
                self.assertEqual(obj.stats['response_budget_binding'],0)
                self.assertIsNotNone(obj.epcc)
                self.assertEqual(len(block._forward_hooks),1)
                stats=obj.final_stats();json.dumps(stats,allow_nan=False)
                if unbounded:
                    self.assertTrue(math.isinf(obj.round_budget))
                    self.assertIsNone(stats['round_mismatch_budget'])
                    self.assertFalse(stats['round_cap_enabled'])
                    self.assertEqual(stats['row_budget_binding'],0)
            finally:obj.restore()
            self.assertFalse(block._forward_hooks)
            for k,v in originals.items():self.assertIs(getattr(obj.runtime,k),v)

    def test_score_eos_visible_boundary_and_first_longest_still_apply(self):
        obj,_,_,ids=fixture()
        try:
            obj.prefill(ids)
            candidates=torch.tensor([[3,1,2,3,1],[3,4,4,4,4]])
            retrieve=torch.tensor([[0,1,2,3,4],[0,5,6,7,8]])
            tokens=torch.tensor([[3,1,2,3,1,4,4,4,4]])
            mask=torch.zeros(9,9)
            for path in retrieve.tolist():
                for depth,node in enumerate(path):mask[node,path[:depth+1]]=1
            obj.model.base_model.model.tree_mask=mask
            logits=torch.zeros(2,5,7);logits[:,:,4]=2.
            obj.tree(obj.model,tokens,None,torch.arange(9),ids,retrieve)
            best,length,_=obj.evaluate(logits,candidates,None)
            self.assertEqual((int(best),int(length)),(1,4))
            self.assertEqual(obj._selected_mismatches,0)
            self.assertEqual(obj.committed_mismatches,0)
            # With small finite margins the first tied full-length row wins.
            logits[:,:,4]=.25
            obj.tree(obj.model,tokens,None,torch.arange(9),ids,retrieve)
            best,length,_=obj.evaluate(logits,candidates,None)
            self.assertEqual((int(best),int(length)),(0,4))
            self.assertEqual(obj.committed_mismatches,0)
            opts=dict(eos=6,committed_length=3,visible_boundary=100,
                response_used=1000,response_budget=math.inf,round_budget=math.inf,allowed=lambda _:True)
            self.assertEqual(budgeted_scan([0,1,6,2],[4,4,4],**opts),(1,1))
            self.assertEqual(budgeted_scan([0,1,2,3],[4,6,4],**opts),(1,1))
            opts['visible_boundary']=5
            self.assertEqual(budgeted_scan([0,1,2,3],[4,4,4],**opts),(1,1))
        finally:obj.restore()

    def test_unbounded_scanner_matches_independent_direct_definition(self):
        rng=random.Random(202611012)
        for _ in range(10000):
            n=rng.randrange(1,12);sequence=[rng.randrange(7) for _ in range(n)]
            pad=rng.randrange(1,n+1);sequence[pad:]=[-1]*(n-pad)
            targets=[sequence[j+1] if rng.random()<.5 else rng.randrange(7) for j in range(n-1)]
            allowed=[rng.random()<.75 for _ in range(n)]
            current=rng.randrange(10,100);boundary=rng.randrange(10,110)
            length=count=0;eos_seen=sequence[0]==6
            for j,d in enumerate(sequence[1:]):
                a=targets[j]
                if d<0:break
                if d!=a:
                    if eos_seen or current+j+1>=boundary or a==6 or d==6 or not allowed[j+1]:break
                    count+=1
                length+=1;eos_seen|=d==6
            actual=budgeted_scan(sequence,targets,eos=6,committed_length=current,
                visible_boundary=boundary,response_used=1000,response_budget=math.inf,
                round_budget=math.inf,allowed=lambda j:allowed[j])
            self.assertEqual(actual,(length,count))


if __name__=='__main__':unittest.main(verbosity=2)
