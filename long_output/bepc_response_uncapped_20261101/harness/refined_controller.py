"""Budgeted refined BEPC and matched single-rule controls on native ViSpec.

No Target/Draft forward, native logits, KV update, tree setting or EOS handling
is replaced. All relaxed arms share the same independent-per-branch mismatch
budgets, first-longest continuous-prefix selection and actual-path commits.
"""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import sys
import time

import torch

from online_controller import Controller as FrozenController, DraftDistributions
from bounds import bound_decision
from optimized_visual import OptimizedVisualScores
from refined_observer import RefinedContextObserver


class PrefixCheckedDraftDistributions(DraftDistributions):
    """Native Draft logits plus the full input identity of their current tree."""
    def tree(self, *args, **kwargs):
        ids = args[1] if len(args) > 1 else kwargs['input_ids']
        self.input_prefix = ids.detach().clone()
        return super().tree(*args, **kwargs)


def native_margins(logits, candidates, targets):
    method = getattr(logits, 'native_candidate_margins', None)
    if callable(method):
        return method(candidates, targets)
    rows = logits[:, :-1]
    a = rows.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    d = rows.gather(-1, candidates[:, 1:].clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return a.float() - d.float()


def budgeted_scan(sequence, targets, *, eos, committed_length, visible_boundary,
                  response_used, response_budget, round_budget, allowed, stats=None):
    """Pure selection primitive; scanning an uncommitted branch spends nothing."""
    length = mismatch_count = 0
    eos_seen = sequence[0] == eos
    for j, d in enumerate(sequence[1:]):
        a = targets[j]
        if d < 0:
            break
        if d == a:
            accept = True
        elif eos_seen or committed_length + j + 1 >= visible_boundary:
            accept = False
        elif a == eos or d == eos:
            accept = False
        elif mismatch_count >= round_budget or response_used + mismatch_count >= response_budget:
            if stats is not None:
                stats['row_budget_binding' if mismatch_count >= round_budget else 'response_budget_binding'] += 1
            accept = False
        else:
            accept = bool(allowed(j + 1))
        if not accept:
            break
        length += 1
        if d != a:
            mismatch_count += 1
        if d == eos:
            eos_seen = True
    return length, mismatch_count


class RefinedController(FrozenController):
    def __init__(self, runtime, model, variant, prompt_ids, image_grid, *,
                 collect=False, collect_variants=None, audit=False,
                 max_new_tokens=128, image_id=None, prompt_kind=None):
        if collect:
            raise ValueError('Refined decoding does not collect future-continuation probes')
        original_variant = dict(variant)
        family = {'MarginE': 'ME', 'BEPC': 'REFINED_BEPC',
                  'BEPC_REFINED': 'REFINED_BEPC'}.get(variant['family'], variant['family'])
        if family not in ('T2', 'MCONF', 'VLINE', 'REFINED_BEPC', 'ME'):
            raise ValueError(f'Unknown refined-control family {family}')
        import online_controller
        previous_visual = online_controller.BatchVisualScores
        try:
            if family in ('ME', 'VLINE'):
                online_controller.BatchVisualScores = OptimizedVisualScores
            surrogate = dict(variant, family='CCVE' if family in ('ME', 'VLINE') else 'T2')
            super().__init__(runtime, model, surrogate, prompt_ids, image_grid,
                             collect=False, audit=audit, max_new_tokens=max_new_tokens)
        finally:
            online_controller.BatchVisualScores = previous_visual
        self.variant, self.family = original_variant, family
        self.audit_enabled = bool(audit)
        self.image_id, self.prompt_kind = image_id, prompt_kind
        self.round_budget = int(variant.get('round_mismatch_budget', 2))
        self.response_budget = int(variant.get('response_mismatch_budget', 4))
        if self.round_budget != 2 or self.response_budget != 4:
            raise ValueError('Matched relaxed controls require budgets round2/response4')
        self.committed_mismatches = 0
        self.decision_cache = {}
        self.decision_reasons = {}
        self.acceptance_paths = []
        self.parent_cache = {}
        self.spaces = None
        self.reference_spaces = None
        self.reference_parents = {}
        self.reference_audit_rows = []
        self.diagnostic_rows = []
        self.timing_events = []
        self.constructor_host_seconds = 0.0
        self._selected_mismatches = None
        self._draft_prefix_matches = False
        if family == 'REFINED_BEPC':
            self.epcc = RefinedContextObserver(
                model, history_window=int(variant.get('text_window', 32)),
                pool_size=int(variant.get('visual_pool_size', 64)), audit=audit).install()
        elif family == 'MCONF':
            self.draft = PrefixCheckedDraftDistributions(model)
        self.stats = Counter(self.stats)
        self.stats.update(
            boundary_candidates=0, full_candidate_scores=0, parent_space_builds=0,
            bound_direct_accepts=0, bound_direct_rejects=0, unsafe_bound_decisions=0,
            mismatch_gate_visits=0, unique_mismatch_gate_requests=0,
            mismatch_gate_cache_hits=0, nonfinite_margin_rejections=0,
            draft_missing_or_prefix_mismatch=0, draft_probability_reads=0,
            ccve_candidate_scores=0, committed_mismatches=0,
            row_budget_binding=0, response_budget_binding=0,
            image_basis_initializations=0,
        )

    def tree(self, model, tokens, cache, positions, ids, retrieve):
        self.parent_cache = {}
        self.reference_parents = {}
        self.decision_cache = {}
        self.decision_reasons = {}
        if self.draft:
            prefix = getattr(self.draft, 'input_prefix', None)
            expected = torch.cat((ids, tokens[:, :1].to(ids.device)), dim=-1)
            self._draft_prefix_matches = prefix is not None and torch.equal(prefix.to(expected.device), expected)
        return super().tree(model, tokens, cache, positions, ids, retrieve)

    def _start_timing(self):
        if self.model.base_model.lm_head.weight.is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            return start, end
        return None

    def _end_timing(self, label, pair):
        if pair is not None:
            pair[1].record()
            self.timing_events.append((label, pair))

    def _ensure_spaces(self):
        if self.spaces is None:
            from refined_math import RefinedSpaces
            pair = self._start_timing()
            start = time.perf_counter()
            self.spaces = RefinedSpaces(
                self.epcc.visual_rows,
                rank_u=int(self.variant.get('rank_u', 8)),
                rank_p=int(self.variant.get('rank_p', 4)),
                solver=self.variant.get('solver', self.variant.get('text_solver', 'fast')))
            self.constructor_host_seconds += time.perf_counter() - start
            self._end_timing('image_basis', pair)
            self.stats['image_basis_initializations'] += 1

    def _parent(self, parent):
        if parent not in self.parent_cache:
            self._ensure_spaces()
            context = self.epcc.context(parent)
            pair = self._start_timing()
            value = self.spaces.parent(context['T'], context['h'])
            self._end_timing('parent_spaces', pair)
            self.parent_cache[parent] = (value, context)
            self.stats['parent_space_builds'] += 1
            if not value.get('valid', False):
                self.stats['unavailable_parent_spaces'] += 1
        return self.parent_cache[parent]

    def _bepc_score(self, parent, a, d, margin):
        from refined_math import score_components
        value, context = self._parent(parent)
        head = self.model.base_model.lm_head
        norm = self.model.base_model.model.norm
        db = 0.0 if head.bias is None else head.bias[a].float() - head.bias[d].float()
        pair = self._start_timing()
        result = score_components(
            value, head.weight[a], head.weight[d], norm.weight,
            norm.variance_epsilon, margin,
            bias_difference=db, gamma=float(self.variant.get('gamma', 0.5)),
            kappa=float(self.variant.get('kappa', 1.0)),
            eta=float(self.variant.get('eta', 1e-4)),
            residual_attenuation=bool(self.variant.get('residual_attenuation', True)))
        self._end_timing('candidate_readout', pair)
        mode = self.variant.get('correction_mode', 'both')
        if mode not in ('both', 'protection', 'rescue'):
            raise ValueError('correction_mode must be both, protection or rescue')
        if mode != 'both':
            active = result[mode]
            result['delta'] = torch.where(active, result['delta'], torch.zeros_like(result['delta']))
            result['score'] = margin - result['delta']
        if self.audit_enabled:
            from refined_math import RefinedSpaces
            if self.reference_spaces is None:
                self.reference_spaces = RefinedSpaces(
                    self.epcc.visual_rows, rank_u=int(self.variant.get('rank_u', 8)),
                    rank_p=int(self.variant.get('rank_p', 4)), solver='reference')
            if parent not in self.reference_parents:
                self.reference_parents[parent] = self.reference_spaces.parent(context['T'], context['h'])
            reference_state = self.reference_parents[parent]
            reference = score_components(
                reference_state, head.weight[a], head.weight[d], norm.weight,
                norm.variance_epsilon, margin, bias_difference=db,
                gamma=float(self.variant.get('gamma', 0.5)),
                kappa=float(self.variant.get('kappa', 1.0)),
                eta=float(self.variant.get('eta', 1e-4)),
                residual_attenuation=bool(self.variant.get('residual_attenuation', True)))
            if mode != 'both':
                reference['delta'] = torch.where(reference[mode], reference['delta'], torch.zeros_like(reference['delta']))
                reference['score'] = margin - reference['delta']
            # Compare projected states, never raw basis columns: signs and
            # rotations within an equivalent principal subspace are arbitrary.
            projection_errors = [
                torch.linalg.vector_norm(value[name] - reference_state[name]) /
                torch.linalg.vector_norm(reference_state['h']).clamp_min(1e-8)
                for name in ('h_U', 'h_P', 'h_R')]
            component_errors = [(result[name] - reference[name]).abs()
                                for name in ('m_U', 'm_P', 'm_R', 'score')]
            self.reference_audit_rows.append(torch.stack(projection_errors + component_errors).detach())
            tau = float(self.variant.get('tau', self.variant.get('threshold', 1.0)))
            if bool(result['score'] <= tau) != bool(reference['score'] <= tau):
                self.stats['reference_decision_disagreements'] += 1
                raise AssertionError('Refined Fast and Reference candidate decisions differ')
            self.stats['reference_candidate_audit_checks'] += 1
        self.stats['full_candidate_scores'] += 1
        fields = ('m_U', 'm_P', 'm_R', 'w_R', 'gate', 'delta', 'score',
                  'lens_margin', 'lens_closure_error', 'native_lens_error',
                  'protection', 'rescue', 'available')
        self.diagnostic_rows.append(torch.stack([
            torch.as_tensor(result[name], device=margin.device, dtype=torch.float32)
            for name in fields]).detach())
        if self.audit_enabled:
            correction = float((result['score'] - margin).abs())
            if correction > float(self.variant.get('gamma', 0.5)) * float(self.variant.get('kappa', 1)) + 1e-5:
                raise AssertionError('Refined BEPC score exceeded frozen bound')
        return result

    def _ccve(self, parent, a, d):
        key = ('CCVE', parent, a, d)
        if key not in self.score_cache:
            if not self.visual.prepared:
                self.visual.prepare()
            result = self.visual.fast_score_valid(parent, a, d)
            self.score_cache[key] = float(result['CCVE'])
            self.stats['ccve_candidate_scores'] += 1
        return self.score_cache[key]

    def _draft_probability(self, row, column, d):
        key = ('draft_probability', tuple(self.candidate_rows[row][:column]), d)
        if key in self.score_cache:
            return self.score_cache[key]
        path = key[1]
        q = self.draft.distributions.get(path) if self._draft_prefix_matches else None
        if q is None or not 0 <= d < len(q):
            self.stats['draft_missing_or_prefix_mismatch'] += 1
            probability = float('nan')
        else:
            probability = float(q.float().log_softmax(-1)[d].exp())
            self.stats['draft_probability_reads'] += 1
        self.score_cache[key] = probability
        return probability

    def allowed(self, row, column, logits, candidates):
        self.stats['mismatch_gate_visits'] += 1
        parent = self.retrieve_rows[row][column - 1]
        a, d = self.targets[row][column - 1], self.candidate_rows[row][column]
        key = (parent, a, d)
        if key in self.decision_cache:
            self.stats['mismatch_gate_cache_hits'] += 1
            return self.decision_cache[key]
        self.stats['unique_mismatch_gate_requests'] += 1
        margin = self.margin_tensor[row, column - 1]
        g = self.margin_rows[row][column - 1]
        tau = float(self.variant.get('tau', self.variant.get('threshold', 1.0)))
        reason = 'margin'
        if not math.isfinite(g):
            self.stats['nonfinite_margin_rejections'] += 1
            accept, reason = False, 'nonfinite_margin_reject'
        elif self.family == 'T2':
            accept = g <= tau
        elif self.family == 'MCONF':
            accept = g <= tau
            reason = 'margin_reject'
            if accept:
                probability = self._draft_probability(row, column, d)
                accept = math.isfinite(probability) and probability >= float(self.variant.get('q', self.variant.get('draft_probability_min', 0.1)))
                reason = 'draft_confidence_accept' if accept else 'draft_confidence_reject'
        elif self.family in ('ME', 'VLINE'):
            if self.family == 'ME':
                cap = float(self.variant.get('tau_M', self.variant.get('margin_cap', 4.0)))
                if g > cap:
                    accept, reason = False, 'margin_cap_reject'
                else:
                    e = self._ccve(parent, a, d)
                    accept = math.isfinite(e) and e <= float(self.variant.get('tau_E', self.variant.get('threshold', -0.165220245718956)))
                    reason = 'ccve_accept' if accept else 'ccve_reject'
            else:
                e = self._ccve(parent, a, d)
                score = g + float(self.variant.get('beta', 0.5)) * e
                accept = math.isfinite(score) and score <= tau
                reason = 'visual_linear_accept' if accept else 'visual_linear_reject'
        elif self.family == 'REFINED_BEPC':
            gamma, kappa = float(self.variant.get('gamma', 0.5)), float(self.variant.get('kappa', 1.0))
            accept = bound_decision(g, tau, gamma, kappa) if self.variant.get('prune', True) else None
            if accept is None:
                self.stats['boundary_candidates'] += 1
                result = self._bepc_score(parent, a, d, margin)
                accept = bool(result['score'] <= tau)
                reason = 'bepc_accept' if accept else 'bepc_reject'
            else:
                self.stats['bound_direct_accepts' if accept else 'bound_direct_rejects'] += 1
                reason = 'bound_accept' if accept else 'bound_reject'
                if self.audit_enabled:
                    full = self._bepc_score(parent, a, d, margin)
                    if bool(full['score'] <= tau) != accept:
                        self.stats['unsafe_bound_decisions'] += 1
                        raise AssertionError('Boundary shortcut changed refined BEPC decision')
        else:
            raise ValueError(self.family)
        self.decision_cache[key] = bool(accept)
        self.decision_reasons[key] = reason
        return bool(accept)

    def evaluate(self, logits, candidates, processor):
        assert processor is None
        self.candidate_rows = candidates.tolist()
        targets = logits[:, :-1].argmax(-1)
        self.targets = targets.tolist()
        self.retrieve_rows = self.retrieve.tolist()
        self.margin_tensor = native_margins(logits, candidates, targets)
        self.margin_rows = self.margin_tensor.tolist()
        best, best_length, best_mismatches = 0, -1, 0
        for row, sequence in enumerate(self.candidate_rows):
            length, mismatches = budgeted_scan(
                sequence, self.targets[row], eos=self.eos,
                committed_length=self.ids.shape[1], visible_boundary=self.visible_boundary,
                response_used=self.committed_mismatches,
                response_budget=self.response_budget, round_budget=self.round_budget,
                allowed=lambda column, row=row: self.allowed(row, column, logits, candidates),
                stats=self.stats)
            if length > best_length:
                best, best_length, best_mismatches = row, length, mismatches
        self._selected_mismatches = best_mismatches
        self.stats['accepted_mismatches'] += best_mismatches
        self.stats['accepted_draft'] += best_length
        self.acceptance_paths.append(dict(
            round=self.stats['rounds'] - 1, row=best, accepted_draft=best_length,
            mismatches=best_mismatches, response_mismatches_before=self.committed_mismatches,
            nodes=self.retrieve_rows[best][:best_length + 1],
            tokens=self.candidate_rows[best][:best_length + 1],
        ))
        return (torch.tensor(best, device=logits.device),
                torch.tensor(best_length, device=logits.device), logits[best, best_length])

    def update(self, *args, **kwargs):
        if self._selected_mismatches is None:
            raise RuntimeError('Budget update without selected path')
        result = super().update(*args, **kwargs)
        self.committed_mismatches += self._selected_mismatches
        self.stats['committed_mismatches'] = self.committed_mismatches
        if self.committed_mismatches > self.response_budget:
            raise AssertionError('Committed response mismatch budget exceeded')
        self._selected_mismatches = None
        return result

    def final_stats(self):
        rows = torch.stack(self.diagnostic_rows).cpu().tolist() if self.diagnostic_rows else []
        result = dict(self.stats)
        result.update(
            refined_controller=True, family=self.family,
            round_mismatch_budget=self.round_budget,
            response_mismatch_budget=self.response_budget,
            committed_response_mismatches=self.committed_mismatches,
            image_basis_lazy_initialization=True,
            image_basis_constructor_host_seconds=self.constructor_host_seconds,
            spaces=self.spaces.statistics() if self.spaces else {},
            observer_context_counters=dict(self.epcc.context_counters) if self.epcc else {},
            visual_pool_metadata=self.epcc.pool_meta if self.epcc else {},
            visual_observer_counters=dict(self.visual.counters) if self.visual else {},
            component_aggregation_outside_generation_timing=True,
            native_lens_abs_error_max=max((abs(r[9]) for r in rows), default=None),
            lens_closure_abs_error_max=max((abs(r[8]) for r in rows), default=None),
            scored_conflict_candidates=sum(bool(r[4]) for r in rows),
            scored_protection_candidates=sum(bool(r[10]) for r in rows),
            scored_rescue_candidates=sum(bool(r[11]) for r in rows),
            unavailable_scored_candidates=sum(not bool(r[12]) for r in rows),
            all_relaxed_methods_same_budget=True,
            extra_target_forwards=0,
            original_native_margin_preserved=True,
        )
        by_label = Counter()
        for label, pair in self.timing_events:
            by_label[label] += pair[0].elapsed_time(pair[1])
        result['component_cuda_ms'] = dict(by_label)
        audit_rows = torch.stack(self.reference_audit_rows).cpu().tolist() if self.reference_audit_rows else []
        result['reference_audit_projection_relative_error_max'] = max((max(r[:3]) for r in audit_rows), default=None)
        result['reference_audit_component_abs_error_max'] = max((max(r[3:6]) for r in audit_rows), default=None)
        result['reference_audit_score_abs_error_max'] = max((r[6] for r in audit_rows), default=None)
        result['reference_audit_extra_feature_work_excluded_from_main'] = bool(self.audit_enabled)
        return result


BEPCController = RefinedController
