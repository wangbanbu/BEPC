"""Label-free same-layer BEPC decomposition and bounded margin correction.

Reference and Fast define the same normalized-row right-singular subspaces.
Reference uses FP32 thin SVD; Fast uses FP32 small-Gram eigh and reconstructs
right directions, with explicit full-SVD fallback on numerical ambiguity.
These are geometric readout components, not calibrated probabilities or a
causal separation of factual evidence from a language prior.
"""
from __future__ import annotations

import math
from collections import Counter

import torch
import torch.nn.functional as F

ABS_RANK_SQUARED = 1e-10
REL_RANK_SQUARED = 1e-5
READOUT_EPSILON = 1e-8
ORTHOGONAL_TOLERANCE = 2e-5
RITZ_RELATIVE_TOLERANCE = 5e-5
BOUNDARY_GAP_RELATIVE = 1e-5


def _pooled_shape(height, width, cap):
    """Nearest integer grid to aspect-preserving sqrt-area scaling.

    Minimize squared log-distance to the ideal scaled dimensions, subject to
    no upsampling and area <= cap. Ties prefer more vectors, then height/width.
    The finite search uses metadata only and does not inspect image values.
    """
    if height * width <= cap:
        return height, width
    scale = math.sqrt(cap / (height * width))
    ideal_h, ideal_w = height * scale, width * scale
    options = []
    for out_h in range(1, min(height, cap) + 1):
        for out_w in range(1, min(width, cap // out_h) + 1):
            distance = math.log(out_h / ideal_h) ** 2 + math.log(out_w / ideal_w) ** 2
            options.append((distance, -out_h * out_w, out_h, out_w))
    _, _, out_h, out_w = min(options)
    return out_h, out_w


@torch.no_grad()
def pool_visual_rows(V, image_grid_thw, spatial_merge_size=2, max_vectors=64):
    """Pool a single COCO t=1 image on its real merged row-major 2D grid.

    ``image_grid_thw`` is Qwen's *pre-spatial-merge* [t,h,w] metadata. Input
    rows must exactly match (h/merge)*(w/merge). Multi-image/video inputs and
    mismatched grids fail clearly instead of inventing a spatial arrangement.
    Pool FP32 before row normalization; the latter belongs to RefinedSpaces.
    """
    if V.ndim != 2:
        raise ValueError('V must be [visual_token_rows, hidden_dim]')
    if not isinstance(spatial_merge_size, int) or spatial_merge_size < 1:
        raise ValueError('spatial_merge_size must be a positive integer')
    if not isinstance(max_vectors, int) or max_vectors < 1:
        raise ValueError('max_vectors must be a positive integer')
    grid = torch.as_tensor(image_grid_thw).detach()
    if tuple(grid.shape) not in ((3,), (1, 3)):
        raise ValueError('Only one image_grid_thw [3] or [1,3] is supported')
    raw = grid.reshape(-1).cpu().tolist()
    if any(not math.isfinite(float(value)) or float(value) != int(value) for value in raw):
        raise ValueError('image_grid_thw must contain finite integer dimensions')
    temporal, height, width = map(int, raw)
    if temporal != 1 or height < 1 or width < 1:
        raise ValueError('This registered COCO pooling requires t=1 and positive h,w')
    if height % spatial_merge_size or width % spatial_merge_size:
        raise ValueError('Grid dimensions must be divisible by spatial_merge_size')
    height //= spatial_merge_size
    width //= spatial_merge_size
    if V.shape[0] != height * width:
        raise ValueError(f'Visual rows {V.shape[0]} != real merged grid {height}x{width}')
    out_h, out_w = _pooled_shape(height, width, max_vectors)
    value = V.float()
    pooled = value
    if (out_h, out_w) != (height, width):
        image = value.reshape(height, width, V.shape[1]).permute(2, 0, 1).unsqueeze(0)
        pooled = F.adaptive_avg_pool2d(image, (out_h, out_w))[0].permute(1, 2, 0).reshape(-1, V.shape[1])
    meta = dict(
        original_grid_t_h_w=[temporal, height * spatial_merge_size, width * spatial_merge_size],
        merged_grid_h_w=[height, width], pooled_grid_h_w=[out_h, out_w],
        input_rows=V.shape[0], pooled_rows=pooled.shape[0], max_vectors=max_vectors,
        spatial_merge_size=spatial_merge_size, temporal_frames=1,
        flattened_order='merged spatial row-major; height then width; hidden dimension last',
        pooling='adaptive_avg_pool2d_fp32' if pooled is not value else 'identity_fp32',
        shape_rule='min squared log-distance to sqrt-area aspect-preserving dimensions; area<=cap; no upsample; ties larger area, then smaller height,width',
        row_normalization='after pooling, in RefinedSpaces',
    )
    return pooled, meta


def _row_normalize(A):
    finite = bool(torch.isfinite(A).all())
    if not finite:
        return A[:0], dict(valid=False, input_rows=A.shape[0], rows=0,
                           zero_rows=0, reason='nonfinite_input')
    norms = torch.linalg.vector_norm(A, dim=1)
    nonzero = norms > 0
    selected = A[nonzero]
    normalized = selected / norms[nonzero, None]
    return normalized, dict(valid=True, input_rows=A.shape[0], rows=normalized.shape[0],
                           zero_rows=A.shape[0] - normalized.shape[0], reason=None)


def _basis_errors(B, against):
    if not B.shape[1]:
        return 0.0, 0.0
    identity = torch.eye(B.shape[1], device=B.device, dtype=B.dtype)
    orthogonal = float((B.T @ B - identity).abs().max())
    cross = float((against.T @ B).abs().max()) if against is not None and against.shape[1] else 0.0
    return orthogonal, cross


def _clean_basis(B, against):
    if not B.shape[1]:
        return B
    if against is not None and against.shape[1]:
        B = B - against @ (against.T @ B)
        B = B - against @ (against.T @ B)
    # Preserve the selected subspace; QR does not select extra near-null ranks.
    Q, R = torch.linalg.qr(B, mode='reduced')
    if not bool(torch.isfinite(Q).all()) or not bool(torch.isfinite(R).all()):
        raise FloatingPointError('nonfinite basis cleanup')
    if bool((R.diagonal().abs() < 1e-4).any()):
        raise FloatingPointError('selected directions collapsed during cleanup')
    return Q


class RefinedSpaces:
    """Fixed per-image U and per-legal-parent P, in identical FP32 coordinates.

    The caller owns ancestry, parent exclusion and the latest-32-row window.
    This module rejects >32 text rows to prevent a hidden algorithm change.
    Solver counters include U initialization and any explicit SVD fallback.
    """
    @torch.no_grad()
    def __init__(self, pooledV, rank_u=8, rank_p=4, solver='fast'):
        if pooledV.ndim != 2:
            raise ValueError('pooledV must be [M,d]')
        if pooledV.shape[0] > 64:
            raise ValueError('Registered refined visual pool is at most64 rows')
        if solver not in ('fast', 'reference'):
            raise ValueError('solver must be fast or reference')
        if not isinstance(rank_u, int) or not isinstance(rank_p, int) or min(rank_u, rank_p) < 1:
            raise ValueError('Ranks must be positive integers')
        self.solver = solver
        self.rank_u, self.rank_p = rank_u, rank_p
        self.hidden_dim = pooledV.shape[1]
        if self.hidden_dim < 1:
            raise ValueError('Hidden dimension must be positive')
        self.counters = Counter(visual_basis_builds=1, text_basis_builds=0,
                                gram_eigh_calls=0, full_svd_calls=0, fast_fallbacks=0,
                                full_svd_failures=0, invalid_inputs=0)
        self.fallback_reasons = Counter()
        self.V, self.visual_normalization = _row_normalize(pooledV.float())
        if self.visual_normalization['valid']:
            self.U, self.visual_info = self._basis(self.V, self.rank_u, kind='visual')
        else:
            self.counters['invalid_inputs'] += 1
            self.U = pooledV.new_empty((self.hidden_dim, 0), dtype=torch.float32)
            self.visual_info = dict(valid=False, reason='nonfinite_visual_rows', effective_rank=0,
                                    solver_requested=solver, solver_used='none', fallback=False)
        self.info = dict(visual=self.visual_info, normalization=self.visual_normalization,
                         rank_U=self.U.shape[1], solver=solver,
                         absolute_rank_squared=ABS_RANK_SQUARED,
                         relative_rank_squared=REL_RANK_SQUARED)

    def _reference(self, A, rank, against, info):
        self.counters['full_svd_calls'] += 1
        try:
            _, singular, right = torch.linalg.svd(A, full_matrices=False)
            squared = singular.square()
            top = float(squared[0]) if squared.numel() else 0.0
            cutoff = max(ABS_RANK_SQUARED, top * REL_RANK_SQUARED)
            count = min(rank, int((squared > cutoff).sum()))
            B = _clean_basis(right[:count].T.contiguous(), against)
            orthogonal, cross = _basis_errors(B, against)
            valid = bool(count) and math.isfinite(orthogonal + cross) and max(orthogonal, cross) <= ORTHOGONAL_TOLERANCE
            info.update(solver_used='full_svd', raw_selected_rank=count,
                        effective_rank=count if valid else 0, top_squared=top,
                        squared_cutoff=cutoff, orthogonality_error=orthogonal,
                        cross_orthogonality_error=cross, valid=valid,
                        reason=None if valid else 'no_effective_rank_or_basis_error')
            return B if valid else A.new_empty((A.shape[1], 0)), info
        except (RuntimeError, FloatingPointError) as error:
            if 'out of memory' in str(error).lower():
                raise
            # Failed exact computation is an explicit invalid-state Margin
            # fallback, never a fabricated basis or a silent changed algorithm.
            self.counters['full_svd_failures'] += 1
            info.update(solver_used='full_svd', effective_rank=0, valid=False,
                        reason='full_svd_numerical_failure:' + type(error).__name__)
            return A.new_empty((A.shape[1], 0)), info

    def _basis(self, A, rank, *, kind, against=None):
        info = dict(solver_requested=self.solver, solver_used='none', kind=kind,
                    requested_rank=rank, input_rows=A.shape[0], fallback=False,
                    fallback_reason=None, valid=False, effective_rank=0)
        if not A.shape[0]:
            info['reason'] = 'no_nonzero_rows'
            return A.new_empty((A.shape[1], 0)), info
        if self.solver == 'reference':
            return self._reference(A, rank, against, info)

        def fallback(reason):
            self.counters['fast_fallbacks'] += 1
            self.fallback_reasons[reason] += 1
            info.update(fallback=True, fallback_reason=reason)
            return self._reference(A, rank, against, info)

        self.counters['gram_eigh_calls'] += 1
        try:
            gram = A @ A.T
            values, vectors = torch.linalg.eigh(gram)
            values, vectors = values.flip(0), vectors.flip(1)
            if not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(vectors).all()):
                return fallback('nonfinite_gram_eigensystem')
            top = max(0.0, float(values[0]))
            cutoff = max(ABS_RANK_SQUARED, top * REL_RANK_SQUARED)
            if float(values[-1]) < -cutoff:
                return fallback('material_negative_gram_eigenvalue')
            # Near the numerical-rank cutoff, use the exact common SVD rule.
            cutoff_guard = max(1e-12, top * 5e-7)
            if bool(((values - cutoff).abs() <= cutoff_guard).any()):
                return fallback('eigenvalue_near_rank_cutoff')
            count = min(rank, int((values > cutoff).sum()))
            info.update(top_squared=top, squared_cutoff=cutoff, effective_rank=count)
            if not count:
                info.update(solver_used='gram_eigh', reason='no_effective_rank', valid=False)
                return A.new_empty((A.shape[1], 0)), info
            if count < values.numel() and float(values[count]) > cutoff:
                gap = float(values[count - 1] - values[count]) / max(top, 1e-30)
                info['selected_boundary_relative_gap'] = gap
                if gap <= BOUNDARY_GAP_RELATIVE:
                    return fallback('ambiguous_selected_rank_boundary')
            B = A.T @ vectors[:, :count]
            B = B / values[:count].sqrt().unsqueeze(0)
            B = _clean_basis(B, against)
            orthogonal, cross = _basis_errors(B, against)
            if not math.isfinite(orthogonal + cross) or max(orthogonal, cross) > ORTHOGONAL_TOLERANCE:
                return fallback('basis_orthogonality_error')
            AB = A @ B
            quotient = AB.T @ AB
            residual = A.T @ AB - B @ quotient
            relative_residual = float(torch.linalg.vector_norm(residual)) / max(top * math.sqrt(count), 1e-30)
            if not math.isfinite(relative_residual) or relative_residual > RITZ_RELATIVE_TOLERANCE:
                return fallback('large_right_subspace_residual')
            info.update(solver_used='gram_eigh', valid=True, reason=None,
                        orthogonality_error=orthogonal, cross_orthogonality_error=cross,
                        right_subspace_relative_residual=relative_residual)
            return B, info
        except (RuntimeError, FloatingPointError) as error:
            if 'out of memory' in str(error).lower():
                raise
            return fallback('eigh_or_cleanup_failure:' + type(error).__name__)

    @torch.no_grad()
    def parent(self, T, h):
        if T.ndim != 2 or T.shape[1] != self.hidden_dim or T.shape[0] > 32:
            raise ValueError('T must contain at most32 legal ancestor rows of hidden_dim')
        if tuple(h.shape) not in ((self.hidden_dim,), (1, self.hidden_dim)):
            raise ValueError('h must be one hidden state, [d] or [1,d]')
        if T.device != self.U.device or h.device != self.U.device:
            raise ValueError('U, legal text history and parent state must share a device')
        self.counters['text_basis_builds'] += 1
        h = h.float().reshape(-1)
        normalized, normalization = _row_normalize(T.float())
        if self.U.shape[1] and normalization['valid']:
            Z = normalized - (normalized @ self.U) @ self.U.T
            # A second identical projection only cleans finite-precision U leakage.
            Z = Z - (Z @ self.U) @ self.U.T
            P, text_info = self._basis(Z, self.rank_p, kind='text', against=self.U)
        else:
            self.counters['invalid_inputs'] += int(not normalization['valid'])
            P = h.new_empty((self.hidden_dim, 0))
            text_info = dict(valid=False, reason='invalid_visual_or_text_rows', effective_rank=0,
                             solver_requested=self.solver, solver_used='none', fallback=False)
        h_U = self.U @ (self.U.T @ h)
        h_P = P @ (P.T @ h)
        h_R = h - h_U - h_P
        valid = bool(self.U.shape[1] and P.shape[1] and self.visual_info['valid']
                     and text_info['valid'] and bool(torch.isfinite(h).all()))
        info = dict(visual=self.visual_info, text=text_info, normalization=normalization,
                    rank_U=self.U.shape[1], rank_P=P.shape[1], valid=valid,
                    parent_excluded_by_contract=True, history_rows=T.shape[0])
        return dict(U=self.U, P=P, h=h, h_U=h_U, h_P=h_P, h_R=h_R,
                    hU=h_U, hP=h_P, hR=h_R, valid=valid,
                    available=h.new_tensor(valid, dtype=torch.bool), info=info)

    def statistics(self):
        return dict(self.counters, fallback_reasons=dict(self.fallback_reasons))


@torch.no_grad()
def score_components(state, head_a, head_d, norm_weight, norm_eps, native_margin,
                     bias_difference=0, gamma=.5, kappa=1, eta=1e-4,
                     residual_attenuation=True):
    """Independent FP32 readouts; never repair native BF16/FP32 lens mismatch.

    No host Tensor readbacks. Invalid/nonfinite subspaces preserve the native
    Margin via delta=0. ``w_R`` is a coverage attenuation, not confidence.
    """
    if not all(math.isfinite(float(x)) for x in (norm_eps, gamma, kappa, eta)):
        raise ValueError('Score hyperparameters must be finite')
    if not 0 <= gamma <= 1 or kappa <= 0 or eta < 0 or norm_eps <= 0:
        raise ValueError('Require gamma in[0,1], positive kappa/norm_eps and nonnegative eta')
    h = state['h']
    if any(value.numel() != h.numel() for value in (head_a, head_d, norm_weight)):
        raise ValueError('Candidate head rows and final RMSNorm weight must match hidden_dim')
    D = norm_weight.float().reshape(-1) * torch.rsqrt(h.square().mean() + norm_eps)
    direction = (head_a.float().reshape(-1) - head_d.float().reshape(-1)) * D
    m_U = direction @ state['h_U']
    m_P = direction @ state['h_P']
    m_R = direction @ state['h_R']  # independent residual readout, not g-minus-parts
    bias = torch.as_tensor(bias_difference, device=h.device, dtype=torch.float32).reshape(())
    g = torch.as_tensor(native_margin, device=h.device, dtype=torch.float32).reshape(())
    lens = direction @ h + bias
    numerator = m_U.abs() + m_P.abs()
    denominator = numerator + m_R.abs() + bias.abs() + READOUT_EPSILON
    w_R = numerator / denominator if residual_attenuation else torch.ones_like(g)
    c_U = (state['U'].T @ direction).square().sum() / (direction.square().sum() + READOUT_EPSILON)
    finite = torch.isfinite(torch.stack((g, lens, m_U, m_P, m_R, w_R, c_U))).all()
    available = torch.as_tensor(state['valid'], device=h.device, dtype=torch.bool) & finite
    gate = available & (m_U * m_P < 0) & (torch.minimum(m_U.abs(), m_P.abs()) >= eta)
    magnitude = torch.minimum(torch.minimum(m_U.abs(), m_P.abs()), g.new_tensor(kappa))
    delta = torch.where(gate, gamma * w_R * m_P.sign() * magnitude, torch.zeros_like(g))
    score = g - delta
    closure = (m_U + m_P + m_R + bias - lens).abs()
    native_error = (g - lens).abs()
    return dict(
        g=g, native_margin=g, lens_margin=lens, g_lens=lens,
        m_U=m_U, m_P=m_P, m_R=m_R, bias_difference=bias,
        lens_closure_error=closure, readout_closure_error=closure,
        native_lens_error=native_error, native_readout_error=native_error,
        w_R=w_R, c_U=c_U, available=available, gate=gate, delta=delta,
        correction=delta, score=score, fallback_margin=~available,
        protection=gate & (m_U > 0) & (m_P < 0),
        rescue=gate & (m_U < 0) & (m_P > 0),
    )
