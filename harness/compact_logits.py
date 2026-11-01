"""Compact, inference-only T=0 view of native ViSpec verification logits.

Native Target forward, mRoPE positions and KV mutation are unchanged. The
[R,L,V] leaf-path expansion is replaced with [N,V] + [R,L] node indices. Only
small token/value gathers and an explicitly selected complete vocabulary row
are materialized. Unsupported Tensor operations fail instead of silently
expanding. ``materialize_reference`` is an audit-only operation.
"""
from __future__ import annotations

import torch


class _IndexedLogits:
    def __init__(self, owner, nodes):
        self.owner = owner
        self.nodes = nodes

    @property
    def device(self):
        return self.owner.device

    @property
    def dtype(self):
        return self.owner.dtype

    @property
    def shape(self):
        return torch.Size((*self.nodes.shape, self.owner.vocab_size))

    @property
    def ndim(self):
        return len(self.shape)

    def size(self, dim=None):
        return self.shape if dim is None else self.shape[dim]

    def argmax(self, dim=None, keepdim=False):
        if dim not in (-1, self.ndim - 1):
            raise NotImplementedError("Compact logits support only vocabulary argmax")
        result = self.owner.node_argmax()[self.nodes]
        return result.unsqueeze(-1) if keepdim else result

    def gather(self, dim, index):
        if dim not in (-1, self.ndim - 1):
            raise NotImplementedError("Compact logits support only vocabulary gather")
        if index.ndim != self.ndim or tuple(index.shape[:-1]) != tuple(self.nodes.shape):
            raise ValueError("Vocabulary gather requires the complete selected node shape")
        return self.owner.native[self.nodes.unsqueeze(-1), index]

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if func is torch.argmax:
            return args[0].argmax(*args[1:], **kwargs)
        raise NotImplementedError(f"Unsupported compact-logits torch operation: {func}")

    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        if any(item is Ellipsis or item is None for item in key):
            raise NotImplementedError("Compact logits do not support ellipsis/newaxis")
        if len(key) <= self.nodes.ndim:
            selected = self.nodes[key]
            if selected.ndim == 0:
                # Explicit selected full vocabulary row, as required by native
                # posterior return. Indexing is a view/gather of exactly one row.
                return self.owner.native[selected]
            return _IndexedLogits(self.owner, selected)
        if len(key) == self.nodes.ndim + 1:
            node_key, vocab_key = key[:-1], key[-1]
            selected = self.nodes[node_key]
            if selected.ndim == 0:
                return self.owner.native[selected, vocab_key]
            raise NotImplementedError("Use vocabulary gather for non-scalar node selections")
        raise IndexError("Too many compact-logits indices")


class CompactTreeLogits(_IndexedLogits):
    # A root resolves ownership through this property, not an owned reference.
    # Temporary indexed views still strongly own their root for valid lifetimes.
    # In particular no root -> root cycle delays native GPU-logit destruction.
    @property
    def owner(self):
        return self

    def __init__(self, native, retrieve_indices):
        if native.ndim != 2 or retrieve_indices.ndim != 2:
            raise ValueError("Expected native [N,V] logits and [R,L] retrieve indices")
        if retrieve_indices.dtype != torch.long:
            raise TypeError("retrieve_indices must be torch.long")
        self.native = native
        # Upstream permits CPU retrieval indices indexing CUDA logits. Mirror
        # this tiny tree index once on the native device so later cached-argmax
        # indexing and native pairwise gathers share legal devices. Preserve
        # the original retrieval object and the logits without copies/casts.
        self.retrieve_indices = retrieve_indices
        self.retrieve_index_device_copies = int(native.device != retrieve_indices.device)
        self.native_retrieve_indices = (
            retrieve_indices.to(device=native.device)
            if self.retrieve_index_device_copies else retrieve_indices
        )
        self.vocab_size = native.shape[-1]
        self._node_argmax = None
        self.argmax_reductions = 0
        self.reference_materializations = 0
        # Calling _IndexedLogits.__init__(self, self, ...) would retain a
        # self-cycle. Set only nodes; inherited owner reads use the property.
        self.nodes = self.native_retrieve_indices

    @property
    def device(self):
        return self.native.device

    @property
    def dtype(self):
        return self.native.dtype

    def node_argmax(self):
        if self._node_argmax is None:
            self._node_argmax = self.native.argmax(-1)
            self.argmax_reductions += 1
        return self._node_argmax

    def native_candidate_margins(self, candidates, targets=None):
        """Frozen FP32 a-minus-d subtraction, with padding gathered at token 0."""
        nodes = self.native_retrieve_indices[:, :-1]
        if targets is None:
            targets = self.node_argmax()[nodes]
        if candidates.shape != self.retrieve_indices.shape or targets.shape != nodes.shape:
            raise ValueError("Candidates/targets do not match the retrieval tree")
        target = self.native[nodes, targets]
        draft = self.native[nodes, candidates[:, 1:].clamp_min(0)]
        return target.float() - draft.float()

    def materialize_reference(self):
        """Audit only: exactly the upstream expansion, including -1 padding."""
        self.reference_materializations += 1
        return self.native[self.retrieve_indices]

    def audit_statistics(self):
        """Shape-only accounting; no GPU synchronization or actual expansion."""
        rows = self.retrieve_indices.numel()
        native_rows = self.native.shape[0]
        posterior_rows = self.retrieve_indices.shape[0] * max(0, self.retrieve_indices.shape[1] - 1)
        itemsize = self.native.element_size()
        return dict(
            native_rows=native_rows, expanded_rows=rows,
            retrieve_index_device_copies=self.retrieve_index_device_copies,
            retrieve_index_copy_bytes=(
                self.retrieve_indices.numel() * self.retrieve_indices.element_size()
                * self.retrieve_index_device_copies),
            nominal_expanded_bytes=rows * self.vocab_size * itemsize,
            retained_native_bytes=self.native.numel() * itemsize,
            avoided_expansion_allocation_bytes=rows * self.vocab_size * itemsize,
            baseline_posterior_argmax_rows=posterior_rows,
            compact_posterior_argmax_rows=native_rows,
            nominal_argmax_rows_saved=posterior_rows - native_rows,
            native_argmax_reductions=self.argmax_reductions,
            audit_reference_materializations=self.reference_materializations,
        )


def native_candidate_margins(logits, candidates, targets):
    """Drop-in common helper, accepting compact views or reference tensors."""
    if isinstance(logits, CompactTreeLogits):
        return logits.native_candidate_margins(candidates, targets)
    rows = logits[:, :-1]
    a = rows.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    d = rows.gather(-1, candidates[:, 1:].clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return a.float() - d.float()


def tree_decoding(model, tree_candidates, past_key_values, tree_position_ids,
                  input_ids, retrieve_indices):
    """Upstream forward copied verbatim, changing only returned-logit plumbing."""
    position_ids = tree_position_ids + input_ids.shape[1]
    if model.base_model.config.architectures[0] == "Qwen2_5_VLForConditionalGeneration":
        position_ids = position_ids.unsqueeze(0) + model.base_model.rope_deltas
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    outputs, tree_logits, hidden_state = model(
        tree_candidates, output_orig=True, past_key_values=past_key_values,
        position_ids=position_ids,
    )
    logits = CompactTreeLogits(tree_logits[0], retrieve_indices)
    return logits, hidden_state, outputs
