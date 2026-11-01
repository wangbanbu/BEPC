"""CPU-only tests of frozen algebra, numerical validity and spatial pooling."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from refined_math import (ABS_RANK_SQUARED, REL_RANK_SQUARED, RefinedSpaces,
                          pool_visual_rows, score_components)


def projector(B):
    return B @ B.T


def manual_state(*, h=None, valid=True):
    h = torch.tensor([1., 1., 2., 0.]) if h is None else h
    U, P = torch.eye(4)[:, :1], torch.eye(4)[:, 1:2]
    h_U, h_P = U @ (U.T @ h), P @ (P.T @ h)
    return dict(U=U, P=P, h=h, h_U=h_U, h_P=h_P,
                h_R=h-h_U-h_P, valid=valid)


class PoolingTests(unittest.TestCase):
    def test_real_merged_grid_identity_and_bf16_promotion(self):
        values = torch.arange(60*5).reshape(60, 5).to(torch.bfloat16)
        pooled, meta = pool_visual_rows(values, torch.tensor([[1, 12, 20]]))
        self.assertTrue(torch.equal(pooled, values.float()))
        self.assertEqual(pooled.dtype, torch.float32)
        self.assertEqual(meta['merged_grid_h_w'], [6, 10])
        self.assertEqual(meta['pooled_grid_h_w'], [6, 10])
        self.assertEqual(meta['pooling'], 'identity_fp32')

    def test_pooling_is_exact_2d_adaptive_average_not_flattened(self):
        values = torch.arange(60*3).reshape(60, 3).float()
        pooled, meta = pool_visual_rows(values, [1, 12, 20], max_vectors=8)
        self.assertEqual(meta['pooled_grid_h_w'], [2, 4])
        reference = F.adaptive_avg_pool2d(values.reshape(6,10,3).permute(2,0,1).unsqueeze(0), (2,4))
        reference = reference[0].permute(1,2,0).reshape(8,3)
        self.assertTrue(torch.equal(pooled, reference))

    def test_default_cap_and_repeatability(self):
        values = torch.arange(14*20*3).reshape(280,3).float()
        first, meta = pool_visual_rows(values, [[1, 28, 40]])
        second, second_meta = pool_visual_rows(values, [[1, 28, 40]])
        self.assertEqual(meta['pooled_grid_h_w'], [7, 9])
        self.assertLessEqual(first.shape[0], 64)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(meta, second_meta)

    def test_invalid_grid_and_unsupported_video_fail_clearly(self):
        values = torch.zeros(6, 4)
        for grid in ([2,4,6], [1,5,6], [1,4,8], [1,4,6.5], [[1],[4],[6]], [[1,4,6], [1,4,6]]):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                pool_visual_rows(values, grid)


class SubspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, seed=1701):
        generator = torch.Generator().manual_seed(seed)
        return (torch.randn(40,64,generator=generator),
                torch.randn(24,64,generator=generator),
                torch.randn(64,generator=generator))

    def test_fast_reference_projection_and_score_agreement(self):
        for seed in range(10):
            V, T, h = self.fixture(seed)
            fast = RefinedSpaces(V, solver='fast')
            reference = RefinedSpaces(V, solver='reference')
            a, b = fast.parent(T,h), reference.parent(T,h)
            self.assertTrue(a['valid'] and b['valid'])
            self.assertTrue(torch.allclose(projector(a['U']), projector(b['U']), atol=2e-5, rtol=2e-5))
            self.assertTrue(torch.allclose(projector(a['P']), projector(b['P']), atol=5e-5, rtol=5e-5))
            generator = torch.Generator().manual_seed(seed+100)
            wa, wd = torch.randn(64,generator=generator), torch.randn(64,generator=generator)
            options = dict(norm_weight=torch.ones(64), norm_eps=1e-6, native_margin=torch.tensor(.7))
            fa, rb = score_components(a,wa,wd,**options), score_components(b,wa,wd,**options)
            for key in ('m_U','m_P','m_R','w_R','score'):
                self.assertTrue(torch.allclose(fa[key],rb[key],atol=3e-4,rtol=3e-4), (seed,key,fa[key],rb[key]))
            self.assertEqual(bool(fa['gate']),bool(rb['gate']))
            self.assertEqual(fa['score'].dtype,torch.float32)

    def test_orthogonality_and_independent_hidden_reconstruction(self):
        V,T,h = self.fixture()
        for solver in ('reference','fast'):
            state = RefinedSpaces(V,solver=solver).parent(T,h)
            self.assertTrue(torch.allclose(state['U'].T@state['U'],torch.eye(8),atol=2e-6))
            self.assertTrue(torch.allclose(state['P'].T@state['P'],torch.eye(4),atol=2e-6))
            self.assertLess(float((state['U'].T@state['P']).abs().max()),2e-6)
            self.assertTrue(torch.allclose(state['h_U']+state['h_P']+state['h_R'],h,atol=2e-6))
            self.assertTrue(torch.allclose(state['U'].T@state['h_R'],torch.zeros(8),atol=3e-6))
            self.assertTrue(torch.allclose(state['P'].T@state['h_R'],torch.zeros(4),atol=3e-6))

    def test_zero_and_duplicate_rows_do_not_create_spurious_rank(self):
        e = torch.eye(12)
        visual = torch.stack((e[0],e[0],torch.zeros(12)))
        text = torch.stack((e[1],e[1],e[1],torch.zeros(12)))
        for solver in ('fast','reference'):
            spaces = RefinedSpaces(visual,solver=solver)
            state = spaces.parent(text,e[0]+e[1])
            self.assertEqual(state['U'].shape[1],1)
            self.assertEqual(state['P'].shape[1],1)
            self.assertTrue(state['valid'])
            self.assertEqual(spaces.visual_normalization['zero_rows'],1)
            self.assertEqual(state['info']['normalization']['zero_rows'],1)

    def test_rank_threshold_common_squared_rule(self):
        A = torch.zeros(4,8)
        A[0,0] = 1
        A[1,1] = math.sqrt(2e-5)
        A[2,2] = math.sqrt(5e-6)
        for solver in ('fast','reference'):
            spaces=RefinedSpaces(torch.eye(8),rank_u=1,solver=solver)
            basis, info=spaces._basis(A,4,kind='rank_rule_test')
            self.assertEqual(basis.shape[1],2)
            self.assertAlmostEqual(info['squared_cutoff'], max(ABS_RANK_SQUARED,REL_RANK_SQUARED),places=10)

    def test_text_inside_U_and_empty_text_have_no_P(self):
        e = torch.eye(8)
        for solver in ('fast','reference'):
            spaces = RefinedSpaces(e[:1],solver=solver)
            for text in (e[:1].repeat(5,1),torch.empty(0,8)):
                state = spaces.parent(text,e[0]+e[1])
                self.assertFalse(state['valid'])
                self.assertEqual(state['P'].shape[1],0)
                score=score_components(state,e[0],e[1],torch.ones(8),1e-6,torch.tensor(.4))
                self.assertEqual(float(score['delta']),0)
                self.assertEqual(score['score'],torch.tensor(.4))

    def test_ambiguous_fast_selected_boundary_explicitly_falls_back(self):
        visual = torch.eye(12,20)
        fast=RefinedSpaces(visual,rank_u=8,solver='fast')
        reference=RefinedSpaces(visual,rank_u=8,solver='reference')
        self.assertEqual(fast.visual_info['solver_used'],'full_svd')
        self.assertTrue(fast.visual_info['fallback'])
        self.assertEqual(fast.counters['fast_fallbacks'],1)
        self.assertEqual(fast.visual_info['fallback_reason'],'ambiguous_selected_rank_boundary')
        self.assertTrue(torch.equal(projector(fast.U),projector(reference.U)))

    def test_eigh_exception_has_recorded_exact_fallback(self):
        V,_,_=self.fixture()
        with patch('torch.linalg.eigh',side_effect=RuntimeError('synthetic numerical failure')):
            fast=RefinedSpaces(V,solver='fast')
        self.assertTrue(fast.visual_info['fallback'])
        self.assertEqual(fast.visual_info['solver_used'],'full_svd')
        self.assertEqual(fast.counters['full_svd_calls'],1)
        self.assertIn('eigh_or_cleanup_failure',fast.visual_info['fallback_reason'])

    def test_failed_exact_solver_is_explicit_invalid_margin_fallback(self):
        V,T,h=self.fixture()
        with patch('torch.linalg.eigh',side_effect=RuntimeError('synthetic failure')), patch('torch.linalg.svd',side_effect=RuntimeError('synthetic SVD convergence failure')):
            spaces=RefinedSpaces(V,solver='fast')
        self.assertFalse(spaces.visual_info['valid'])
        self.assertEqual(spaces.counters['fast_fallbacks'],1)
        self.assertEqual(spaces.counters['full_svd_calls'],1)
        self.assertEqual(spaces.counters['full_svd_failures'],1)
        self.assertIn('full_svd_numerical_failure',spaces.visual_info['reason'])
        state=spaces.parent(T,h)
        result=score_components(state,torch.ones(64),torch.zeros(64),torch.ones(64),1e-6,.45)
        self.assertFalse(bool(result['available']))
        self.assertEqual(result['score'],torch.tensor(.45))

    def test_out_of_memory_is_not_hidden_as_numerical_fallback(self):
        V,_,_=self.fixture()
        with patch('torch.linalg.eigh',side_effect=RuntimeError('CUDA out of memory')):
            with self.assertRaisesRegex(RuntimeError,'out of memory'):
                RefinedSpaces(V,solver='fast')

    def test_nonfinite_inputs_and_history_window_enforced(self):
        V,T,h=self.fixture()
        spaces=RefinedSpaces(V,solver='fast')
        with self.assertRaises(ValueError): spaces.parent(torch.zeros(33,64),h)
        bad=T.clone();bad[0,0]=float('nan')
        self.assertFalse(spaces.parent(bad,h)['valid'])
        badV=V.clone();badV[0,0]=float('inf')
        invalid=RefinedSpaces(badV,solver='fast')
        self.assertEqual(invalid.U.shape[1],0)
        self.assertFalse(invalid.parent(T,h)['valid'])


class ReadoutTests(unittest.TestCase):
    def score(self,state=None,**kwargs):
        return score_components(manual_state() if state is None else state,
                                torch.tensor([2.,-3.,4.,0.]),torch.zeros(4),
                                torch.ones(4),1e-6,torch.tensor(20.,dtype=torch.bfloat16),**kwargs)

    def test_mR_is_independent_and_native_margin_is_not_repaired(self):
        state=manual_state()
        result=self.score(state,bias_difference=.25)
        D=torch.rsqrt(state['h'].square().mean()+1e-6)
        expected=torch.tensor([2.,-3.,4.,0.])*D
        self.assertEqual(result['m_R'],expected@state['h_R'])
        self.assertNotAlmostEqual(float(result['m_R']),float(result['g']-result['m_U']-result['m_P']-.25),places=4)
        self.assertEqual(result['g'],torch.tensor(20.))
        self.assertGreater(float(result['native_lens_error']),1)
        self.assertLess(float(result['lens_closure_error']),2e-6)
        self.assertEqual(result['lens_margin'],expected@state['h']+.25)

    def test_residual_bias_attenuation_and_ablation(self):
        result=self.score(bias_difference=.25)
        expected=(result['m_U'].abs()+result['m_P'].abs())/(result['m_U'].abs()+result['m_P'].abs()+result['m_R'].abs()+.25+1e-8)
        self.assertEqual(result['w_R'],expected)
        self.assertTrue(bool(result['protection']))
        self.assertGreater(result['score'],result['g'])
        ablation=self.score(bias_difference=.25,residual_attenuation=False)
        self.assertEqual(float(ablation['w_R']),1)
        self.assertGreater(ablation['delta'].abs(),result['delta'].abs())

    def test_rescue_protection_gate_and_correction_bound(self):
        first=self.score()
        self.assertTrue(bool(first['gate']) and bool(first['protection']))
        state=manual_state(h=torch.tensor([-1.,-1.,2.,0.]))
        second=self.score(state)
        self.assertTrue(bool(second['gate']) and bool(second['rescue']))
        self.assertLess(second['score'],second['g'])
        for state in (manual_state(),state):
            for gamma,kappa in ((.5,1.),(1.,2.),(0.,1.)):
                result=self.score(state,gamma=gamma,kappa=kappa)
                self.assertLessEqual(float(result['delta'].abs()),gamma*kappa+1e-7)
                self.assertGreaterEqual(float(result['w_R']),0)
                self.assertLessEqual(float(result['w_R']),1)

    def test_weak_same_sign_invalid_and_nonfinite_are_margin_fallbacks(self):
        states=[manual_state(valid=False),manual_state(h=torch.tensor([1.,-1.,2.,0.])),
                manual_state(h=torch.tensor([1e-8,1.,2.,0.]))]
        for state in states:
            result=self.score(state)
            self.assertFalse(bool(result['gate']))
            self.assertEqual(result['score'],result['g'])
        state=manual_state();state['h_R'][2]=float('nan')
        result=self.score(state)
        self.assertFalse(bool(result['available']))
        self.assertEqual(result['score'],result['g'])

    def test_independent_residual_detects_deliberately_inconsistent_projection(self):
        state=manual_state();state['h_R']=state['h_R']+torch.tensor([0.,0.,1.,0.])
        result=self.score(state)
        self.assertGreater(float(result['lens_closure_error']),1)
        self.assertEqual(result['g'],torch.tensor(20.))


if __name__=='__main__':
    unittest.main()
