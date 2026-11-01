"""Exact visual-first necessary bounds for the frozen refined BEPC score.

This module changes evaluation order, not the retained-candidate score. The
optional residual_attenuation=False arm is a separate algorithmic ablation.
All visual readouts use the previous experiment's FP32 RMSNorm convention.
No Target/Draft forward, labels or future continuation are read here.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import torch

from refined_math import score_components  # frozen retained-candidate mathematics

BOUND_ABS_SLACK = 1e-5
BOUND_REL_SLACK = 1e-6


def _parameters(gamma, kappa, eta, absolute_slack, relative_slack):
    values = (gamma, kappa, eta, absolute_slack, relative_slack)
    if not all(math.isfinite(float(x)) for x in values):
        raise ValueError('Bound parameters must be finite')
    if not 0 <= gamma <= 1 or kappa <= 0 or min(eta, absolute_slack, relative_slack) < 0:
        raise ValueError('Require gamma in[0,1], kappa>0 and nonnegative eta/slack')


@torch.no_grad()
def visual_parent(U, h, *, visual_valid=True):
    """Build h_U only; do not access a text history or construct P."""
    if U.ndim != 2 or h.numel() != U.shape[0] or U.device != h.device:
        raise ValueError('U must be[d,r] and h[d] on the same device')
    h = h.float().reshape(-1)
    U = U.float()
    h_U = U @ (U.T @ h)
    return dict(U=U, h=h, h_U=h_U,
                visual_valid=bool(visual_valid and U.shape[1] > 0))


@torch.no_grad()
def visual_readout(state, head_a, head_d, norm_weight, norm_eps):
    """Same m_U expression as old score_components, without text subspaces.

    The two-element output tensor lets the controller make one host readback:
    [m_U, finite/available]. Nonfinite or missing visual spaces imply the old
    exact Margin fallback, since no valid full state can be formed.
    """
    if not math.isfinite(float(norm_eps)) or norm_eps <= 0:
        raise ValueError('norm_eps must be positive and finite')
    h = state['h']
    if any(value.numel() != h.numel() for value in (head_a, head_d, norm_weight)):
        raise ValueError('Readout rows must match the parent hidden dimension')
    D = norm_weight.float().reshape(-1) * torch.rsqrt(h.square().mean() + norm_eps)
    direction = (head_a.float().reshape(-1) - head_d.float().reshape(-1)) * D
    m_U = direction @ state['h_U']
    finite = torch.isfinite(torch.cat((h, state['h_U'], direction))).all() & torch.isfinite(m_U)
    available = finite & torch.as_tensor(state['visual_valid'], device=h.device, dtype=torch.bool)
    return dict(m_U=m_U, available=available,
                host_summary=torch.stack((m_U, available.to(torch.float32))))


def visual_first_decision(g, tau, m_U, *, available=True, gamma=.5, kappa=1.,
                          eta=1e-4, correction_mode='both',
                          absolute_slack=BOUND_ABS_SLACK,
                          relative_slack=BOUND_REL_SLACK):
    """Return a conservative exact decision, or None to request complete P.

    For m_U>0 the only possible nonzero correction is protection:
       score in [g, g+B].
    For m_U<0 the only possible nonzero correction is rescue:
       score in [g-B, g].
    B=gamma*min(abs(m_U), kappa), since w_R is in[0,1] (also when disabled).
    Invalid P/nonconflict always gives delta=0 and is contained in both ranges.
    The slack retreats from both comparison boundaries; it never moves tau.
    """
    _parameters(gamma, kappa, eta, absolute_slack, relative_slack)
    if correction_mode not in ('both', 'protection', 'rescue'):
        raise ValueError('Unknown correction mode')
    g, tau, m_U = float(g), float(tau), float(m_U)
    if not math.isfinite(g) or not math.isfinite(tau):
        return dict(decision=None, reason='nonfinite_margin_or_tau', lower=None,
                    upper=None, bound=None, slack=None)
    if not available or not math.isfinite(m_U):
        return dict(decision=g <= tau, reason='visual_unavailable_margin',
                    lower=g, upper=g, bound=0., slack=0.)
    bound = gamma * min(abs(m_U), kappa)
    slack = absolute_slack + relative_slack * (abs(g) + abs(tau) + abs(m_U) + bound)
    inactive_mode = ((correction_mode == 'protection' and m_U <= 0) or
                     (correction_mode == 'rescue' and m_U >= 0))
    if gamma == 0 or m_U == 0 or abs(m_U) + slack < eta or inactive_mode:
        return dict(decision=g <= tau, reason='visual_gate_inactive_margin',
                    lower=g, upper=g, bound=bound, slack=slack)
    lower, upper = (g, g + bound) if m_U > 0 else (g - bound, g)
    if upper + slack <= tau:
        decision, reason = True, 'visual_bound_accept'
    elif lower - slack > tau:
        decision, reason = False, 'visual_bound_reject'
    else:
        decision, reason = None, 'visual_bound_needs_text'
    return dict(decision=decision, reason=reason, lower=lower, upper=upper,
                bound=bound, slack=slack)
