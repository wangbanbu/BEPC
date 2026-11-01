"""Isolated BEPC execution-order optimization and bounded rescue ablation.

Nothing in the sealed prior controller is edited. The symmetric configuration
has exactly the prior candidate score. Once the response mismatch budget is
exhausted, a greedy-prefix scan replaces feature scoring and the passive hidden
observer is removed. This does not replace Target/Draft forwards or KV updates.

The sole algorithmic ablation is gamma_rescue != gamma_protection. It preserves
the same U/P construction, conflict gate, native margin, tau and 2/4 budgets.
No labels or subsequent continuations enter the score or any shortcut bound.
"""
from __future__ import annotations

import math

import torch

import refined_controller as frozen
from speed_controller import SpeedController
from speed_math import visual_readout, visual_first_decision


def _validate_scales(gamma_rescue, gamma_protection, kappa=1.):
    values = tuple(map(float, (gamma_rescue, gamma_protection, kappa)))
    if not all(math.isfinite(x) for x in values):
        raise ValueError('BEPC correction scales must be finite')
    if not (0 <= values[0] <= 1 and 0 <= values[1] <= 1 and values[2] > 0):
        raise ValueError('Require rescue/protection gamma in[0,1] and kappa>0')


def asymmetric_bound_decision(g, tau, gamma_rescue, gamma_protection, kappa):
    """Safe interval for score in[g-gamma_rescue*k,g+gamma_protection*k]."""
    _validate_scales(gamma_rescue, gamma_protection, kappa)
    if not math.isfinite(float(g)) or not math.isfinite(float(tau)):
        return None
    if gamma_rescue == gamma_protection:
        return frozen.bound_decision(g, tau, gamma_rescue, kappa)
    radius_r, radius_p = gamma_rescue * kappa, gamma_protection * kappa
    slack = 1e-5 + 1e-6 * (abs(g) + abs(tau) + max(radius_r, radius_p))
    if g < tau - radius_p - slack:
        return True
    if g > tau + radius_r + slack:
        return False
    return None


def asymmetric_visual_decision(g, tau, m_U, *, available=True,
                               gamma_rescue=.5, gamma_protection=.5,
                               kappa=1., eta=1e-4, correction_mode='both'):
    """Unknown P can only protect when m_U>0 or rescue when m_U<0."""
    _validate_scales(gamma_rescue, gamma_protection, kappa)
    gamma = gamma_protection if m_U > 0 else gamma_rescue
    return visual_first_decision(
        g, tau, m_U, available=available, gamma=gamma, kappa=kappa,
        eta=eta, correction_mode=correction_mode)


def adjust_rescue_result(result, margin, *, gamma_rescue, gamma_protection):
    """Change only positive, active rescue correction of a protection-scale readout.

    Equal scales return the original dictionary untouched, preserving every
    operation and comparison of the old symmetric score (including FP32 order).
    The base score is evaluated with gamma=gamma_protection>0.
    """
    _validate_scales(gamma_rescue, gamma_protection)
    if gamma_protection <= 0:
        raise ValueError('This isolated ablation requires positive protection scale')
    if gamma_rescue == gamma_protection:
        return result
    delta = torch.where(result['rescue'],
                        result['delta'] * (gamma_rescue / gamma_protection),
                        result['delta'])
    return dict(result, delta=delta, correction=delta, score=margin - delta)


def strict_exhausted_selection(candidate_rows, targets, *, eos,
                                committed_length, visible_boundary,
                                stats=None):
    """Exactly budgeted_scan(response_used=response_budget), first-longest.

    Greedy matches remain accepted even at EOS/visible boundaries, as in the
    frozen scanner; mismatches at those boundaries are rejected without counting
    a budget-binding event. Padding ends the branch. No margin is requested.
    """
    best, best_length = 0, -1
    for row, sequence in enumerate(candidate_rows):
        length = 0
        eos_seen = sequence[0] == eos
        for j, d in enumerate(sequence[1:]):
            a = targets[row][j]
            if d < 0:
                break
            if d != a:
                if (not eos_seen and committed_length + j + 1 < visible_boundary
                        and a != eos and d != eos and stats is not None):
                    stats['response_budget_binding'] += 1
                break
            length += 1
            if d == eos:
                eos_seen = True
        if length > best_length:
            best, best_length = row, length
    return best, best_length


class ExperimentalBEPC(SpeedController):
    """Copied-controller extension; used only when experimental_bepc=True."""
    def __init__(self, runtime, model, variant, prompt_ids, image_grid, **kwargs):
        variant = dict(variant)
        if not variant.get('experimental_bepc', False):
            raise ValueError('Explicit experimental_bepc=True is required')
        self.gamma_rescue = float(variant.get('gamma_rescue', .5))
        self.gamma_protection = float(variant.get('gamma_protection', .5))
        _validate_scales(self.gamma_rescue, self.gamma_protection,
                         float(variant.get('kappa', 1.)))
        if self.gamma_protection != .5:
            raise ValueError('This study freezes protection gamma=.5')
        if variant.get('residual_attenuation', False):
            raise ValueError('This study freezes residual_attenuation=False')
        if (float(variant.get('tau', 1.)) != 1.
                or int(variant.get('rank_u', 8)) != 8
                or int(variant.get('rank_p', 4)) != 4
                or int(variant.get('text_window', 32)) != 32
                or int(variant.get('visual_pool_size', 64)) != 64
                or variant.get('correction_mode', 'both') != 'both'):
            raise ValueError('The prior BEPC subspaces, tau and both-direction gate are frozen')
        variant['gamma'] = self.gamma_protection
        variant['residual_attenuation'] = False
        self._retired_observer_counters = None
        self._retired_pool_metadata = None
        super().__init__(runtime, model, variant, prompt_ids, image_grid, **kwargs)
        if self.family != 'REFINED_BEPC':
            self.restore()
            raise ValueError('ExperimentalBEPC requires the refined BEPC family')
        self.stats.update(exhausted_fast_rounds=0, observer_retirements=0,
                          skipped_native_margin_rounds=0)

    def _visual_bound(self, parent, a, d, g, tau):
        state = self._visual_parent(parent)
        head, norm = self.model.base_model.lm_head, self.model.base_model.model.norm
        result = visual_readout(state, head.weight[a], head.weight[d], norm.weight,
                                norm.variance_epsilon)
        m_U, available = result['host_summary'].tolist()
        self.stats['visual_candidate_readouts'] += 1
        return asymmetric_visual_decision(
            g, tau, m_U, available=bool(available),
            gamma_rescue=self.gamma_rescue, gamma_protection=self.gamma_protection,
            kappa=float(self.variant.get('kappa', 1.)),
            eta=float(self.variant.get('eta', 1e-4)),
            correction_mode=self.variant.get('correction_mode', 'both'))

    def _bepc_score(self, parent, a, d, margin):
        result = super()._bepc_score(parent, a, d, margin)
        adjusted = adjust_rescue_result(
            result, margin, gamma_rescue=self.gamma_rescue,
            gamma_protection=self.gamma_protection)
        if adjusted is not result and self.rich_diagnostics:
            # Parent audits still compare the unchanged U/P subspace construction.
            # Store the actual new correction and score in candidate diagnostics.
            self.diagnostic_rows[-1][5] = adjusted['delta'].detach()
            self.diagnostic_rows[-1][6] = adjusted['score'].detach()
        return adjusted

    def allowed(self, row, column, logits, candidates):
        # Symmetric scores use the old exact path verbatim.
        if self.gamma_rescue == self.gamma_protection:
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
        tau = float(self.variant.get('tau', 1.))
        if not math.isfinite(g):
            self.stats['nonfinite_margin_rejections'] += 1
            accept, reason = False, 'nonfinite_margin_reject'
        else:
            accept = (asymmetric_bound_decision(
                g, tau, self.gamma_rescue, self.gamma_protection,
                float(self.variant.get('kappa', 1.)))
                if self.variant.get('prune', True) else None)
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
        if self.committed_mismatches < self.response_budget:
            return super().evaluate(logits, candidates, processor)
        assert processor is None
        self.candidate_rows = candidates.tolist()
        self.targets = logits[:, :-1].argmax(-1).tolist()
        self.retrieve_rows = self.retrieve.tolist()
        best, best_length = strict_exhausted_selection(
            self.candidate_rows, self.targets, eos=self.eos,
            committed_length=self.ids.shape[1], visible_boundary=self.visible_boundary,
            stats=self.stats)
        self._selected_mismatches = 0
        self.stats['accepted_draft'] += best_length
        self.stats['exhausted_fast_rounds'] += 1
        self.stats['skipped_native_margin_rounds'] += 1
        self.acceptance_paths.append(dict(
            round=self.stats['rounds'] - 1, row=best, accepted_draft=best_length,
            mismatches=0, response_mismatches_before=self.committed_mismatches,
            nodes=self.retrieve_rows[best][:best_length + 1],
            tokens=self.candidate_rows[best][:best_length + 1]))
        terminal = self.original.get('update_inference_inputs')
        note_terminal = getattr(terminal, 'note_selection', None)
        if callable(note_terminal):
            note_terminal(best, best_length, self.candidate_rows[best][:best_length + 1])
            self.stats['terminal_cpu_selection_hints'] += 1
        return (torch.tensor(best, device=logits.device),
                torch.tensor(best_length, device=logits.device), logits[best, best_length])

    def update(self, *args, **kwargs):
        result = super().update(*args, **kwargs)
        if self.committed_mismatches >= self.response_budget and self.epcc is not None:
            # The final budget-consuming commit has completed normally. Only
            # later rounds retire the observer; it can never be needed again.
            self._retired_observer_counters = dict(self.epcc.context_counters)
            self._retired_pool_metadata = dict(self.epcc.pool_meta)
            self.epcc.restore()
            self.epcc = None
            self.parent_cache = {}
            self.visual_parent_cache = {}
            self.reference_parents = {}
            self.stats['observer_retirements'] += 1
        return result

    def final_stats(self):
        result = super().final_stats()
        if self._retired_observer_counters is not None:
            result['observer_context_counters'] = self._retired_observer_counters
            result['visual_pool_metadata'] = self._retired_pool_metadata
        result.update(
            experimental_bepc=True,
            gamma_rescue=self.gamma_rescue, gamma_protection=self.gamma_protection,
            budget_exhaustion_feature_retirement=True,
            asymmetric_correction=self.gamma_rescue != self.gamma_protection,
            retained_score_definition=(
                'frozen score_components unchanged' if self.gamma_rescue == self.gamma_protection
                else 'frozen score_components; active rescue delta scaled only'))
        return result
