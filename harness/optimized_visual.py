"""Exact scalar-path engineering for the frozen final-block CCVE observer.

The native tree capture, mRoPE/KV assertions, full attention normalizer,
FP32 output projection and FP32 candidate readout are inherited unchanged.
Only the lifetime of FP32 K/V and head-row copies, and CPU-known scalar
index handling, differ. This observer adds no Target forward or approximation.
"""
from __future__ import annotations

from numbers import Integral

import torch

from visual_online import BatchVisualScores, grouped_visual_features


class OptimizedVisualScores(BatchVisualScores):
    """Frozen arithmetic with tree-local FP32 inputs and scalar CPU indexing.

    ``fast_score_valid`` requires CPU integer indices already obtained from the
    legal native retrieve tree. It still checks their bounds. ``scores`` keeps
    the original padded-array API for validation, rather than silently changing
    the numerical parent batch shape used by that API.
    """

    def __init__(self, *args, **kwargs):
        # The inherited constructor still copies W_O and RMSNorm weights once
        # per request. Their cost remains inside the request timing boundary.
        super().__init__(*args, **kwargs)
        self._K32 = self._V32 = self._node_indices = None
        self._computed_parents = set()
        self._head_rows = {}
        self._head_bytes = 0
        self.counters.update(
            scalar_score_requests=0,
            parent_feature_cache_hits=0,
            fp32_k_materializations=0,
            fp32_v_materializations=0,
            fp32_kv_buffer_bytes_peak=0,
            fp32_head_buffer_bytes_peak=0,
        )

    def _clear_tree_buffers(self):
        # Clear before a new native forward, avoiding simultaneous old/new
        # FP32 contexts. No cache is shared across committed prefix changes.
        self._K32 = self._V32 = self._node_indices = None
        self._computed_parents = set()
        self._head_rows = {}
        self._head_bytes = 0

    def enable(self, active=True):
        if active and not self.closed:
            self._clear_tree_buffers()
        return super().enable(active)

    @torch.no_grad()
    def prepare(self):
        # Keep every native capture, cache-length, mRoPE, mask and finite-value
        # check from the reference. Do not invent a flattened causal prefix.
        super().prepare()
        self._clear_tree_buffers()
        self._K32 = self.K.float()
        self._V32 = self.V.float() if self.with_ccve else None
        self._node_indices = torch.arange(self.tree_nodes, device=self.K.device)
        extra_bytes = 0
        if self._K32.data_ptr() != self.K.data_ptr():
            self.counters['fp32_k_materializations'] += 1
            extra_bytes += self._K32.numel() * self._K32.element_size()
        if self.with_ccve and self._V32.data_ptr() != self.V.data_ptr():
            self.counters['fp32_v_materializations'] += 1
            extra_bytes += self._V32.numel() * self._V32.element_size()
        self.counters['fp32_kv_buffer_bytes_peak'] = max(
            self.counters['fp32_kv_buffer_bytes_peak'], extra_bytes)

    def _require_ready(self):
        if not self.prepared or self.closed or self.active:
            raise RuntimeError('prepare the completed native tree before scores')

    def _validate_scalar_indices(self, parent, a, d):
        if not all(isinstance(x, Integral) for x in (parent, a, d)):
            raise TypeError('Fast scalar indices must be CPU integers, not CUDA tensors')
        parent, a, d = int(parent), int(a), int(d)
        if not 0 <= parent < self.tree_nodes:
            raise ValueError('Parent index lies outside the captured tree')
        vocab = self.base.lm_head.weight.shape[0]
        if not 0 <= a < vocab or not 0 <= d < vocab:
            raise ValueError('Candidate token lies outside the Target output vocabulary')
        return parent, a, d

    @torch.no_grad()
    def _ensure_parent(self, parent):
        if parent in self._computed_parents:
            self.counters['parent_feature_cache_hits'] += 1
            return
        # Advanced indexing and parent batch size one match the frozen online
        # scalar caller. Batching additional parents could change GPU GEMM
        # rounding and is deliberately not introduced here.
        index = self._node_indices[parent:parent + 1]
        query = self.q[:, index].transpose(0, 1)
        mass, visual = grouped_visual_features(
            query, self._K32, self._V32, self.mask[index],
            self.visual_keys, self.Wo)
        self._mass_cache[index] = mass
        if self.with_ccve:
            self._visual_cache[index] = visual
            raw = self.state['raw_x'][0, index].float()
            self._D_cache[index] = self.norm_weight * torch.rsqrt(
                raw.square().mean(-1, keepdim=True) + self.norm_eps)
            self.counters['visual_branch_parent_rows'] += 1
        self._computed[index] = True
        self._computed_parents.add(parent)
        self.counters['attention_parent_rows'] += 1

    @torch.no_grad()
    def _head_row(self, token):
        if token not in self._head_rows:
            # Keep [1, hidden] layout, subtraction and sum(-1) shape identical
            # to the reference's one-candidate advanced-indexing expression.
            head = self.base.lm_head.weight.detach()
            self._head_rows[token] = head[token:token + 1].float()
            if self._head_rows[token].data_ptr() != head[token:token + 1].data_ptr():
                self._head_bytes += self._head_rows[token].numel() * 4
            self.counters['fp32_head_buffer_bytes_peak'] = max(
                self.counters['fp32_head_buffer_bytes_peak'], self._head_bytes)
        return self._head_rows[token]

    @torch.no_grad()
    def fast_score_valid(self, parent, a, d):
        """Return one legal parent/candidate score without GPU index readback."""
        self._require_ready()
        parent, a, d = self._validate_scalar_indices(parent, a, d)
        self._ensure_parent(parent)
        mass = self._mass_cache[parent]
        ccve = None
        if self.with_ccve:
            direction = self._head_row(a) - self._head_row(d)
            ccve = (direction * self._D_cache[parent:parent + 1]
                    * self._visual_cache[parent:parent + 1]).sum(-1)[0]
        if self.check_cache:
            if not bool(torch.isfinite(mass)) or (
                    ccve is not None and not bool(torch.isfinite(ccve))):
                raise AssertionError('Nonfinite score on a valid candidate')
        self.counters['score_calls'] += 1
        self.counters['scalar_score_requests'] += 1
        return dict(visual_attention_mass=mass, CCVE=ccve, valid=True)

    single_score = fast_score_valid

    @torch.no_grad()
    def reference_score_valid(self, parent, a, d):
        """Audit-only frozen scalar recomputation, independent of fast caches.

        Re-promotes native BF16 K/V and output-head rows and recomputes the
        attention branch, D, and candidate reduction. No fast counter/cache is
        changed; this method must not enter measured production requests.
        """
        self._require_ready()
        parent, a, d = self._validate_scalar_indices(parent, a, d)
        index = torch.tensor([parent], device=self.K.device)
        mass, visual = grouped_visual_features(
            self.q[:, index].transpose(0, 1), self.K, self.V,
            self.mask[index], self.visual_keys, self.Wo)
        ccve = None
        if self.with_ccve:
            raw = self.state['raw_x'][0, index].float()
            D = self.norm_weight * torch.rsqrt(
                raw.square().mean(-1, keepdim=True) + self.norm_eps)
            head = self.base.lm_head.weight.detach()
            aa = torch.tensor([a], device=head.device)
            dd = torch.tensor([d], device=head.device)
            direction = head[aa].float() - head[dd].float()
            ccve = (direction * D * visual).sum(-1)[0]
        return dict(visual_attention_mass=mass[0], CCVE=ccve, valid=True)

    @torch.no_grad()
    def scores(self, parents, a, d):
        # Preserve frozen vector/padding semantics and arithmetic as an audit
        # path. Its GPU->CPU cache bookkeeping is intentionally absent from the
        # hot scalar API; main decoding calls fast_score_valid instead.
        result = super().scores(parents, a, d)
        legal = (parents >= 0) & (a >= 0) & (d >= 0)
        self._computed_parents.update(parents[legal].reshape(-1).tolist())
        return result

    def close(self):
        self._clear_tree_buffers()
        return super().close()


# Both descriptive names are exported for controller/validation callers.
OptimizedBatchVisualScores = OptimizedVisualScores
