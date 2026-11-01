"""Visual-first evaluation-order optimization of frozen refined BEPC.

All branch scans, budgets, native margins, KV/tree work and retained full
scores come from the read-only previous implementation. SPEED_BEPC is only a
constructor alias; residual_attenuation=False is an explicit new algorithm
arm. Production can omit diagnostic kernels/events, recorded in final_stats.
"""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import sys

import torch

import refined_controller as frozen
from refined_controller import RefinedController
from speed_math import visual_parent, visual_readout, visual_first_decision, score_components


class SpeedController(RefinedController):
    def __init__(self, runtime, model, variant, prompt_ids, image_grid, *,
                 collect=False, collect_variants=None, audit=False,
                 max_new_tokens=128, image_id=None, prompt_kind=None,
                 observer_factory=None):
        variant = dict(variant)
        variant['family'] = {'SPEED_BEPC': 'REFINED_BEPC'}.get(variant['family'], variant['family'])
        self.rich_diagnostics = bool(variant.get('rich_diagnostics', False) or audit)
        self.visual_first_prune = bool(variant.get('visual_first_prune', True))
        previous = frozen.RefinedContextObserver
        try:
            if variant['family'] in ('BEPC', 'BEPC_REFINED', 'REFINED_BEPC'):
                if observer_factory is None:
                    from speed_observer import SpeedContextObserver
                    observer_factory = SpeedContextObserver
                # Construction is request-local and serial; restore even if it
                # fails. Never monkeypatch any frozen source file or forward.
                frozen.RefinedContextObserver = observer_factory
            super().__init__(runtime, model, variant, prompt_ids, image_grid,
                             collect=collect, collect_variants=collect_variants,
                             audit=audit, max_new_tokens=max_new_tokens,
                             image_id=image_id, prompt_kind=prompt_kind)
        finally:
            frozen.RefinedContextObserver = previous
        self.visual_parent_cache = {}
        self.visual_bound_audit_rows = []
        self.stats.update(visual_parent_builds=0, visual_candidate_readouts=0,
                          visual_bound_direct_accepts=0, visual_bound_direct_rejects=0,
                          visual_bound_needs_text=0, visual_bound_audit_checks=0,
                          visual_bound_decision_disagreements=0,
                          visual_gate_inactive_margin_decisions=0,
                          visual_unavailable_margin_decisions=0)

    def tree(self, *args, **kwargs):
        self.visual_parent_cache = {}
        return super().tree(*args, **kwargs)

    def _start_timing(self):
        return super()._start_timing() if self.rich_diagnostics else None

    def _ensure_spaces(self):
        if self.spaces is None:
            # The optimized observer may defer pooling until the first actual
            # BEPC boundary. It must expose the same fixed pooled rows after.
            ensure = getattr(self.epcc, 'ensure_visual_rows', None)
            if callable(ensure):
                ensure()
            super()._ensure_spaces()

    def _visual_parent(self, parent):
        if parent not in self.visual_parent_cache:
            self._ensure_spaces()
            reader = getattr(self.epcc, 'parent_hidden', None)
            if callable(reader):
                h = reader(parent)
            else:
                # Compatibility with the old fully audited observer. This
                # intentionally does NOT call context()/construct text T.
                if not self.epcc._tree_ready or parent not in self.epcc.paths:
                    raise ValueError('Not a legal completed native tree parent')
                h = self.epcc.tree_raw[parent].float()
            self.visual_parent_cache[parent] = visual_parent(
                self.spaces.U, h, visual_valid=self.spaces.visual_info['valid'])
            self.stats['visual_parent_builds'] += 1
        return self.visual_parent_cache[parent]

    def _visual_bound(self, parent, a, d, g, tau):
        state = self._visual_parent(parent)
        head, norm = self.model.base_model.lm_head, self.model.base_model.model.norm
        result = visual_readout(state, head.weight[a], head.weight[d], norm.weight,
                                norm.variance_epsilon)
        m_U, available = result['host_summary'].tolist()
        self.stats['visual_candidate_readouts'] += 1
        return visual_first_decision(
            g, tau, m_U, available=bool(available),
            gamma=float(self.variant.get('gamma', .5)),
            kappa=float(self.variant.get('kappa', 1.)),
            eta=float(self.variant.get('eta', 1e-4)),
            correction_mode=self.variant.get('correction_mode', 'both'))

    def _bepc_score(self, parent, a, d, margin):
        if self.rich_diagnostics:
            # Full frozen numerical audit/diagnostics, including same-parent
            # exact-SVD checks, are preserved for every audit full evaluation.
            return super()._bepc_score(parent, a, d, margin)
        value, _ = self._parent(parent)
        head, norm = self.model.base_model.lm_head, self.model.base_model.model.norm
        bias = 0. if head.bias is None else head.bias[a].float() - head.bias[d].float()
        result = score_components(
            value, head.weight[a], head.weight[d], norm.weight, norm.variance_epsilon,
            margin, bias_difference=bias, gamma=float(self.variant.get('gamma', .5)),
            kappa=float(self.variant.get('kappa', 1.)), eta=float(self.variant.get('eta', 1e-4)),
            residual_attenuation=bool(self.variant.get('residual_attenuation', True)))
        mode = self.variant.get('correction_mode', 'both')
        if mode not in ('both', 'protection', 'rescue'):
            raise ValueError('Unknown correction mode')
        if mode != 'both':
            result['delta'] = torch.where(result[mode], result['delta'], torch.zeros_like(result['delta']))
            result['score'] = margin - result['delta']
        self.stats['full_candidate_scores'] += 1
        return result

    def _audit_shortcut(self, parent, a, d, margin, decision, bounds=None):
        full = self._bepc_score(parent, a, d, margin)
        correct = bool(full['score'] <= float(self.variant.get('tau', self.variant.get('threshold', 1.))))
        self.stats['visual_bound_audit_checks'] += int(bounds is not None)
        if correct != decision:
            self.stats['unsafe_bound_decisions'] += 1
            self.stats['visual_bound_decision_disagreements'] += int(bounds is not None)
            raise AssertionError('Visual-first or global necessary bound changed frozen full-score decision')
        if bounds is not None:
            score = float(full['score'])
            error = max(0., bounds['lower'] - score, score - bounds['upper'])
            self.visual_bound_audit_rows.append(dict(
                parent=int(parent), target=int(a), draft=int(d), decision=bool(decision),
                score=score, lower=bounds['lower'], upper=bounds['upper'],
                slack=bounds['slack'], interval_violation=error, reason=bounds['reason']))
            if error > bounds['slack']:
                raise AssertionError('Full BEPC score exceeded visual-first interval plus slack')

    def allowed(self, row, column, logits, candidates):
        if self.family != 'REFINED_BEPC':
            return super().allowed(row, column, logits, candidates)
        self.stats['mismatch_gate_visits'] += 1
        parent = self.retrieve_rows[row][column - 1]
        a, d = self.targets[row][column - 1], self.candidate_rows[row][column]
        key = (parent, a, d)
        if key in self.decision_cache:
            self.stats['mismatch_gate_cache_hits'] += 1
            return self.decision_cache[key]
        self.stats['unique_mismatch_gate_requests'] += 1
        margin, g = self.margin_tensor[row, column - 1], self.margin_rows[row][column - 1]
        tau = float(self.variant.get('tau', self.variant.get('threshold', 1.)))
        gamma, kappa = float(self.variant.get('gamma', .5)), float(self.variant.get('kappa', 1.))
        if not math.isfinite(g):
            self.stats['nonfinite_margin_rejections'] += 1
            accept, reason = False, 'nonfinite_margin_reject'
        else:
            accept = frozen.bound_decision(g, tau, gamma, kappa) if self.variant.get('prune', True) else None
            if accept is not None:
                reason = 'bound_accept' if accept else 'bound_reject'
                self.stats['bound_direct_accepts' if accept else 'bound_direct_rejects'] += 1
                if self.audit_enabled:
                    self._audit_shortcut(parent, a, d, margin, accept)
            else:
                self.stats['boundary_candidates'] += 1
                bounds = self._visual_bound(parent, a, d, g, tau) if self.visual_first_prune else None
                accept = None if bounds is None else bounds['decision']
                if accept is not None:
                    reason = bounds['reason']
                    self.stats['visual_bound_direct_accepts' if accept else 'visual_bound_direct_rejects'] += 1
                    if reason == 'visual_gate_inactive_margin':
                        self.stats['visual_gate_inactive_margin_decisions'] += 1
                    elif reason == 'visual_unavailable_margin':
                        self.stats['visual_unavailable_margin_decisions'] += 1
                    if self.audit_enabled:
                        self._audit_shortcut(parent, a, d, margin, accept, bounds)
                else:
                    self.stats['visual_bound_needs_text'] += int(bounds is not None)
                    full = self._bepc_score(parent, a, d, margin)
                    accept = bool(full['score'] <= tau)
                    reason = 'bepc_accept' if accept else 'bepc_reject'
        self.decision_cache[key], self.decision_reasons[key] = bool(accept), reason
        return bool(accept)

    def evaluate(self, logits, candidates, processor):
        """Frozen scan/selection, plus reuse of already available CPU metadata."""
        assert processor is None
        self.candidate_rows = candidates.tolist()
        note_candidates = getattr(self.epcc, 'note_candidates', None)
        if callable(note_candidates):
            note_candidates(self.candidate_rows)
        targets = logits[:, :-1].argmax(-1)
        self.targets = targets.tolist()
        self.retrieve_rows = self.retrieve.tolist()
        self.margin_tensor = frozen.native_margins(logits, candidates, targets)
        self.margin_rows = self.margin_tensor.tolist()
        best, best_length, best_mismatches = 0, -1, 0
        for row, sequence in enumerate(self.candidate_rows):
            length, mismatches = frozen.budgeted_scan(
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
        note_selection = getattr(self.epcc, 'note_selection', None)
        if callable(note_selection):
            note_selection(best, best_length)
        terminal = self.original.get('update_inference_inputs')
        note_terminal = getattr(terminal, 'note_selection', None)
        if callable(note_terminal):
            note_terminal(best, best_length, self.candidate_rows[best][:best_length + 1])
            self.stats['terminal_cpu_selection_hints'] += 1
        return (torch.tensor(best, device=logits.device),
                torch.tensor(best_length, device=logits.device), logits[best, best_length])

    def final_stats(self):
        result = super().final_stats()
        result.update(speed_controller=True, visual_first_prune=self.visual_first_prune,
                      residual_attenuation=bool(self.variant.get('residual_attenuation', True)),
                      rich_diagnostics=self.rich_diagnostics,
                      production_diagnostic_kernels_omitted=not self.rich_diagnostics,
                      visual_bound_slack=dict(absolute=1e-5, relative=1e-6),
                      visual_bound_audit_interval_violation_max=max(
                          (row['interval_violation'] for row in self.visual_bound_audit_rows), default=None),
                      visual_bound_audit_rows=list(self.visual_bound_audit_rows),
                      visual_parent_cache_scope='current native tree only',
                      retained_score_definition='frozen refined_math.score_components')
        if not self.rich_diagnostics:
            for field in ('native_lens_abs_error_max', 'lens_closure_abs_error_max',
                          'scored_conflict_candidates', 'scored_protection_candidates',
                          'scored_rescue_candidates', 'unavailable_scored_candidates'):
                result[field] = None
        return result


BEPCController = SpeedController
