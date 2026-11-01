"""Decision-time EPCC on real native forwards; no runtime monkey patching.

Lifecycle::

    obs = OnlineEPCC(model).install()
    obs.begin_prefill(input_ids, image_grid_thw)
    result = native_initialize_tree(...)       # existing Target prefill
    obs.finish_prefill()
    obs.begin_tree(tokens, input_ids, retrieve, tree_mask, positions)
    result = native_tree_decoding(...)         # existing Target tree forward
    obs.finish_tree()
    epcc = obs.score_rows(row, column, target_top1_id, draft_id)
    result = native_update_inference_inputs(...)
    obs.commit(best, accepted_length, old_retrieve, old_candidates, result[0])

The raw-state commit uses the *old* tree/retrieve/candidates, exactly like the
native KV gather. Relaxed accepted paths become the actual next prefix; there
is no comparison to a historical Strict prefix. Arm/finish calls and commits
belong inside actual decode timing. No extra forward, random control, archive,
debug Gram matrix or virtual edit is performed. Missing rank returns NaN and
must cause the caller to fall back to its frozen exact-acceptance rule.

Math is the E5 frozen FP32 tensor-only benchmark: weighted centered full thin
SVD, U rank 8 / P rank 1, cosine temperature 1.2, U thin QR and legal P rank
reduction. U depends on the current parent h; only its *raw* visual rows cache
across trees. Each current parent shares one U/P across candidate directions.
"""
from __future__ import annotations

import torch


def _basis(X, weights, rank):
    X = X.float()
    n, d = X.shape
    if n < 2 or rank <= 0 or torch.equal(X, X[:1].expand_as(X)):
        return X.new_empty((d, 0))
    if weights is None:
        A = X - X.mean(0, keepdim=True)
    else:
        weights = weights / (weights.sum() + 1e-12)
        A = (X - (weights[:, None] * X).sum(0, keepdim=True))
        A = A * weights.clamp_min(0).sqrt()[:, None]
    if float(A.abs().max()) < 1e-10:
        return X.new_empty((d, 0))
    _, s, vh = torch.linalg.svd(A, full_matrices=False)
    threshold = max(1e-10, 1e-6 * float(s[0]))
    k = min(rank, n - 1, d, int((s > threshold).sum()))
    return vh[:k].T.contiguous()


def build_spaces(V, T, h):
    """Return frozen E5 U/P; empty effective spaces are never padded."""
    V, T, h = V.float(), T.float(), h.float()
    if not len(V):
        return h.new_empty((len(h), 0)), h.new_empty((len(h), 0))
    cos = (V @ h) / ((V.norm(dim=1) + 1e-6) * (h.norm() + 1e-6))
    U = _basis(V, (cos / 1.2).softmax(0), 8)
    if U.shape[1]:
        U, _ = torch.linalg.qr(U, mode='reduced')
    P = _basis(T - (T @ U) @ U.T, None, 1)
    if P.shape[1]:
        for _ in range(2):
            P = P - U @ (U.T @ P)
        original = P
        P, R = torch.linalg.qr(P, mode='reduced')
        keep = R.diag().abs().gt(1e-6)
        if not bool(keep.all()):
            P = original[:, keep]
            if P.shape[1]:
                P, _ = torch.linalg.qr(P, mode='reduced')
    return U, P


class OnlineEPCC:
    """Batch-one, one-image raw-state observer, final decoder block pre-RMS.

    ``context(parent)`` exposes only legal V/T/h and history positions, without
    CPU copies. ``status(parent)`` and ``components(parent,a,d)`` are optional
    cheap interfaces; the latter returns FP32 [M_U, M_P, EPCC]. No threshold or
    acceptance direction is selected here. Call ``restore`` when disabled.
    """

    def __init__(self, model, *, special_ids=None, capacity=4096, layer_index=-1):
        self.model = model
        self.base = getattr(model, 'base_model', model)
        self.decoder = self.base.model
        self.block = self.decoder.layers[layer_index]
        self.norm = self.decoder.norm
        self.head = self.base.lm_head.weight
        self.image_token_id = int(self.base.config.image_token_id)
        if special_ids is None:
            special_ids = model.tokenizer.all_special_ids
        self.special_ids = frozenset(int(x) for x in special_ids) | {self.image_token_id}
        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError('capacity must be positive')
        self._handle = None
        self._mode = None
        self._captured = None
        self.cache = None
        self.cache_length = 0
        self.committed_ids = None
        self.visual_positions = None
        self.tree_raw = None
        self.contexts = {}
        self._tree_ready = False

    def install(self):
        """Install only a passive forward hook; repeated installs are harmless."""
        if self._handle is None:
            self._handle = self.block.register_forward_hook(self._hook)
        return self

    def restore(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._mode = None
        self._captured = None

    def _hook(self, module, args, output):
        if self._mode is not None:
            if self._captured is not None:
                raise RuntimeError('More than one Target block call in armed capture')
            raw = output[0] if isinstance(output, (tuple, list)) else output
            if raw.ndim != 3 or raw.shape[0] != 1:
                raise ValueError('EPCC supports batch-one raw states [1,N,D]')
            self._captured = raw[0].detach()
        # Never return an output replacement.

    @staticmethod
    def _ids(value, name):
        if value.ndim == 2 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 1:
            raise ValueError(f'{name} must be [N] or [1,N]')
        return value.detach()

    def _arm(self, mode):
        if self._handle is None:
            raise RuntimeError('Call install before arming capture')
        if self._mode is not None:
            raise RuntimeError('Previous capture has not been finished')
        self._mode, self._captured = mode, None

    def begin_prefill(self, input_ids, image_grid_thw=None):
        self._arm('prefill')
        self._prefill_ids = self._ids(input_ids, 'input_ids').clone()
        self._grid = image_grid_thw
        self.cache_length = 0
        self.committed_ids = None
        self.tree_raw = None
        self.contexts = {}
        self._tree_ready = False

    @torch.no_grad()
    def finish_prefill(self):
        if self._mode != 'prefill' or self._captured is None:
            raise RuntimeError('No armed prefill forward was captured')
        raw = self._captured
        n = len(self._prefill_ids)
        if len(raw) != n or n > self.capacity:
            raise ValueError('Prefill state length or cache capacity mismatch')
        ids = self._prefill_ids.to(raw.device)
        positions = torch.where(ids == self.image_token_id)[0]
        if len(positions) == 0:
            raise ValueError('No expanded real image-token positions')
        if self._grid is not None:
            # Qwen2.5-VL spatial merge size 2: t*h*w / 4 visual rows.
            grid = self._grid.reshape(-1, 3)
            expected = int((grid.prod(dim=1) // 4).sum())
            if len(positions) != expected:
                raise ValueError('Expanded image tokens do not match merged grid')
        shape = (self.capacity, raw.shape[-1])
        if (self.cache is None or tuple(self.cache.shape) != shape or
                self.cache.dtype != raw.dtype or self.cache.device != raw.device):
            self.cache = raw.new_empty(shape)
        self.cache[:n].copy_(raw)
        self.cache_length = n
        self.committed_ids = ids.clone()
        self.visual_positions = positions
        self._mode, self._captured = None, None
        return self

    def begin_tree(self, tree_tokens, input_ids, retrieve_indices, tree_mask,
                   tree_position_ids=None):
        if self.committed_ids is None:
            raise RuntimeError('Finish prefill before capturing a tree')
        ids = self._ids(input_ids, 'input_ids')
        if len(ids) != self.cache_length or not torch.equal(
                ids.to(self.committed_ids.device), self.committed_ids):
            raise ValueError('Tree prefix differs from actual committed raw cache')
        tokens = self._ids(tree_tokens, 'tree_tokens').to(self.cache.device)
        n = len(tokens)
        if retrieve_indices.ndim != 2:
            raise ValueError('retrieve_indices must be [rows,depth]')
        mask = tree_mask.detach().reshape(n, n)
        if not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError('Expected native binary ancestor tree_mask')
        mask_cpu = mask.bool().cpu()
        rows = retrieve_indices.detach().cpu().tolist()
        paths = {}
        for row in rows:
            path, padded = [], False
            for node in row:
                if node == -1:
                    padded = True
                    continue
                if padded or node < 0 or node >= n or node in path:
                    raise ValueError('Malformed retrieve path or nonterminal padding')
                path.append(node)
                if path[0] != 0:
                    raise ValueError('Native tree paths must start at root node zero')
                old = paths.setdefault(node, tuple(path))
                if old != tuple(path):
                    raise ValueError('Node occurs under inconsistent parent paths')
        if set(paths) != set(range(n)):
            raise ValueError('retrieve paths do not cover the native tree nodes')
        for node, path in paths.items():
            if set(torch.where(mask_cpu[node])[0].tolist()) != set(path):
                raise ValueError('Native mask and retrieve ancestry disagree')
        if tree_position_ids is not None:
            positions = self._ids(tree_position_ids, 'tree_position_ids').cpu()
            expected = torch.tensor([len(paths[i]) - 1 for i in range(n)])
            if not torch.equal(positions, expected):
                raise ValueError('Native tree depth/position and ancestry disagree')
        self._arm('tree')
        self.tokens = tokens.clone()
        self.retrieve = retrieve_indices.detach().clone()
        self.paths = paths
        self.tree_raw = None
        self.contexts = {}
        self._tree_ready = False

    @torch.no_grad()
    def finish_tree(self):
        if self._mode != 'tree' or self._captured is None:
            raise RuntimeError('No armed tree forward was captured')
        if len(self._captured) != len(self.tokens):
            raise ValueError('Raw output does not match native tree node count')
        self.tree_raw = self._captured
        self._mode, self._captured = None, None
        self._tree_ready = True
        return self

    @torch.no_grad()
    def context(self, parent):
        """Gather legal history strictly before current parent, latest 512 text."""
        if not self._tree_ready:
            raise RuntimeError('Finish tree before requesting a context')
        parent = int(parent)
        if parent not in self.paths:
            raise ValueError('Invalid parent node')
        path = self.paths[parent]
        before = list(path[:-1])
        history_ids = torch.cat((self.committed_ids, self.tokens[before]))
        history_states = torch.cat((self.cache[:self.cache_length],
                                    self.tree_raw[before]), dim=0)
        ordinary = torch.ones_like(history_ids, dtype=torch.bool)
        for token in self.special_ids:
            ordinary &= history_ids != token
        keep = torch.where(ordinary)[0][-512:]
        return dict(V=self.cache[self.visual_positions].float(),
                    T=history_states[keep].float(), h=self.tree_raw[parent],
                    path=path, history_ids=history_ids[keep],
                    history_positions=keep)

    @torch.no_grad()
    def _parent(self, parent):
        parent = int(parent)
        if not self._tree_ready:
            raise RuntimeError('Finish tree before scoring')
        if parent not in self.contexts:
            c = self.context(parent)
            U, P = build_spaces(c['V'], c['T'], c['h'])
            x = c['h'].float()
            available = U.shape[1] > 0 and P.shape[1] > 0
            D = self.norm.weight.float() * torch.rsqrt(
                x.square().mean() + self.norm.variance_epsilon)
            self.contexts[parent] = dict(
                U=U, P=P, h=x, u=U @ (U.T @ x), p=P @ (P.T @ x), D=D,
                rank_U=U.shape[1], rank_P=P.shape[1], available=available,
                reason=None if available else 'zero_effective_U_or_P_rank_no_padding')
        return self.contexts[parent]

    def status(self, parent):
        c = self._parent(parent)
        return {k: c[k] for k in ('rank_U', 'rank_P', 'available', 'reason')}

    @torch.no_grad()
    def components(self, parent, a, d):
        """FP32 [M_U,M_P,EPCC] for the supplied fixed target/Draft direction."""
        c = self._parent(parent)
        if not c['available']:
            return c['h'].new_full((3,), float('nan'))
        wdD = (self.head[a].float() - self.head[d].float()) * c['D']
        mu, mp = wdD @ c['u'], wdD @ c['p']
        return torch.stack((mu, mp, torch.minimum(mu.relu(), (-mp).relu())))

    def score(self, parent_node, a, d):
        return self.components(parent_node, a, d)[2]

    def score_rows(self, row, column, a, d):
        if column < 1 or column >= self.retrieve.shape[1]:
            raise ValueError('column must identify a non-root candidate')
        parent = int(self.retrieve[row, column - 1])
        node = int(self.retrieve[row, column])
        if node < 0 or parent < 0 or self.paths[node][-2] != parent:
            raise ValueError('Candidate has no matching native tree parent')
        if int(self.tokens[node]) != int(d):
            raise ValueError('Draft token does not match retrieve candidate node')
        return self.score(parent, a, d)

    @torch.no_grad()
    def commit(self, best_candidate, accept_length, retrieve_indices, candidates,
               new_input_ids=None):
        """Append native-selected raw nodes; call once per successful KV commit."""
        if not self._tree_ready:
            raise RuntimeError('No uncommitted captured tree')
        if not torch.equal(retrieve_indices.to(self.retrieve.device), self.retrieve):
            raise ValueError('Commit retrieve differs from the captured native tree')
        best, count = int(best_candidate), int(accept_length) + 1
        if best < 0 or best >= len(self.retrieve) or count < 1 or count > self.retrieve.shape[1]:
            raise ValueError('Invalid native best row or accepted length')
        selected = self.retrieve[best, :count].to(self.tree_raw.device)
        if bool((selected < 0).any()):
            raise ValueError('Cannot commit padded native nodes')
        selected_tokens = self.tokens[selected]
        if not torch.equal(candidates[best, :count].to(selected_tokens.device), selected_tokens):
            raise ValueError('Native candidate tokens differ from selected tree nodes')
        ids = torch.cat((self.committed_ids, selected_tokens))
        if new_input_ids is not None and not torch.equal(
                self._ids(new_input_ids, 'new_input_ids').to(ids.device), ids):
            raise ValueError('Native updated prefix differs from selected raw-state path')
        stop = self.cache_length + count
        if stop > self.capacity:
            raise ValueError('Raw-state cache capacity exceeded')
        self.cache[self.cache_length:stop].copy_(self.tree_raw[selected])
        self.cache_length = stop
        self.committed_ids = ids
        self.contexts = {}
        self._tree_ready = False
        return self
