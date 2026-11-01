"""Passive same-layer observer with a genuinely bounded ordinary-text cache.

Only pooled image rows and the latest W ordinary committed text rows survive a
native round. The current parent, siblings, descendants and special tokens are
excluded from its legal text context. No original forward/KV is modified.
"""
from __future__ import annotations

from collections import Counter
import torch


class RefinedContextObserver:
    def __init__(self, model, *, history_window=32, pool_size=64, audit=False):
        self.model = model
        self.base = getattr(model, 'base_model', model)
        self.block = self.base.model.layers[-1]
        self.image_token_id = int(self.base.config.image_token_id)
        self.special_ids = frozenset(map(int, model.tokenizer.all_special_ids)) | {self.image_token_id}
        self.history_window, self.pool_size = int(history_window), int(pool_size)
        if self.history_window != 32 or self.pool_size != 64:
            raise ValueError('Refined main definition fixes text W32 and image pool64')
        self.audit = bool(audit)
        self._handle = None
        self._mode = self._captured = None
        self._tree_ready = False
        self.committed_ids = None
        self.history_states = self.visual_rows = self.tree_raw = None
        self.history_ids, self.history_positions = [], []
        self.paths, self.contexts = {}, {}
        self.context_counters = Counter()
        self.pool_meta = {}

    def install(self):
        if self._handle is None:
            self._handle = self.block.register_forward_hook(self._hook)
        return self

    def _hook(self, module, args, output):
        if self._mode is not None:
            if self._captured is not None:
                raise RuntimeError('More than one Target final-block call while armed')
            raw = output[0] if isinstance(output, (tuple, list)) else output
            if raw.ndim != 3 or raw.shape[0] != 1:
                raise ValueError('Expected batch-one final-block prefinal-RMS raw state')
            self._captured = raw[0].detach()

    @staticmethod
    def _ids(value):
        if value.ndim == 2 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 1:
            raise ValueError('IDs must be [N] or [1,N]')
        return value.detach()

    def _arm(self, mode):
        if self._handle is None or self._mode is not None:
            raise RuntimeError('Observer not installed or previous capture unfinished')
        self._mode, self._captured = mode, None

    def begin_prefill(self, input_ids, image_grid_thw=None):
        self._arm('prefill')
        self._prefill_ids = self._ids(input_ids).clone()
        self.grid = image_grid_thw
        self._tree_ready = False

    @torch.no_grad()
    def finish_prefill(self):
        if self._mode != 'prefill' or self._captured is None:
            raise RuntimeError('No completed native prefill capture')
        from refined_math import pool_visual_rows
        raw = self._captured
        ids = self._prefill_ids.to(raw.device)
        ids_cpu = ids.tolist()
        if len(ids_cpu) != len(raw):
            raise ValueError('Prefill raw/ID length mismatch')
        visual_positions = [i for i, token in enumerate(ids_cpu) if token == self.image_token_id]
        if not visual_positions:
            raise ValueError('No real expanded image-token rows')
        visual = raw[visual_positions].float()
        self.visual_rows, self.pool_meta = pool_visual_rows(visual, self.grid, max_vectors=self.pool_size)
        keep = [i for i, token in enumerate(ids_cpu) if token not in self.special_ids][-self.history_window:]
        self.history_states = raw[keep].float()
        self.history_ids = [ids_cpu[i] for i in keep]
        self.history_positions = keep
        self.committed_ids = ids.clone()
        self.context_counters['visual_pool_builds'] += 1
        self.context_counters['visual_source_rows'] += len(visual_positions)
        self.context_counters['pooled_visual_rows'] += len(self.visual_rows)
        self.context_counters['committed_history_updates'] += 1
        self._mode = self._captured = self._prefill_ids = None
        return self

    def begin_tree(self, tree_tokens, input_ids, retrieve_indices, tree_mask,
                   tree_position_ids=None):
        if self.committed_ids is None:
            raise RuntimeError('Prefill not finished')
        ids = self._ids(input_ids)
        if len(ids) != len(self.committed_ids) or not torch.equal(ids.to(self.committed_ids.device), self.committed_ids):
            raise ValueError('Tree prefix differs from actually committed prefix')
        self.tokens = self._ids(tree_tokens).clone()
        self.token_rows = self.tokens.tolist()
        self.retrieve = retrieve_indices.detach()
        self.retrieve_rows = self.retrieve.tolist()
        n = len(self.tokens)
        paths = {}
        for row in self.retrieve_rows:
            path, padded = [], False
            for node in row:
                if node == -1:
                    padded = True
                    continue
                if padded or node < 0 or node >= n or node in path:
                    raise ValueError('Malformed native retrieve row')
                path.append(node)
                if path[0] != 0 or paths.setdefault(node, tuple(path)) != tuple(path):
                    raise ValueError('Inconsistent native ancestor prefix')
        if set(paths) != set(range(n)):
            raise ValueError('Retrieve rows do not cover native nodes')
        # Exact ancestor validation remains in the runtime path. Tree geometry
        # is tiny; no fabricated ordinary causal mask is substituted.
        mask = tree_mask.detach().reshape(n, n).bool().cpu()
        for node, path in paths.items():
            if set(torch.where(mask[node])[0].tolist()) != set(path):
                raise ValueError('Native tree mask differs from retrieve ancestry')
        if tree_position_ids is not None:
            positions = self._ids(tree_position_ids).cpu().tolist()
            if positions != [len(paths[i]) - 1 for i in range(n)]:
                raise ValueError('Native tree positions differ from legal depth')
        self.paths = paths
        self.contexts = {}
        self.tree_raw = None
        self._tree_ready = False
        self._arm('tree')

    def finish_tree(self):
        if self._mode != 'tree' or self._captured is None:
            raise RuntimeError('No completed native tree capture')
        if len(self._captured) != len(self.tokens):
            raise ValueError('Native tree raw shape mismatch')
        self.tree_raw = self._captured
        self._mode = self._captured = None
        self._tree_ready = True
        return self

    @torch.no_grad()
    def context(self, parent):
        parent = int(parent)
        if not self._tree_ready or parent not in self.paths:
            raise ValueError('Not a legal completed native tree parent')
        if parent not in self.contexts:
            path = self.paths[parent]
            # Crucial: path[:-1], never include the current parent itself.
            ancestors = [i for i in path[:-1] if self.token_rows[i] not in self.special_ids]
            ancestor_states = self.tree_raw[ancestors].float()
            T = torch.cat((self.history_states, ancestor_states), dim=0)[-self.history_window:]
            history_ids = (self.history_ids + [self.token_rows[i] for i in ancestors])[-self.history_window:]
            history_positions = (self.history_positions + [len(self.committed_ids) + path.index(i) for i in ancestors])[-self.history_window:]
            self.contexts[parent] = dict(
                V=self.visual_rows, T=T, h=self.tree_raw[parent].float(),
                path=path, history_ids=history_ids, history_positions=history_positions,
                parent_absolute_position=len(self.committed_ids) + len(path) - 1)
            self.context_counters['legal_parent_history_gathers'] += 1
            self.context_counters['max_text_rows'] = max(self.context_counters['max_text_rows'], len(T))
        return self.contexts[parent]

    @torch.no_grad()
    def commit(self, best_candidate, accept_length, retrieve_indices, candidates,
               new_input_ids=None):
        if not self._tree_ready:
            raise RuntimeError('No native tree available for commit')
        best, count = int(best_candidate), int(accept_length) + 1
        if retrieve_indices.tolist() != self.retrieve_rows or not 0 <= best < len(self.retrieve_rows):
            raise ValueError('Commit tree differs from captured native tree')
        selected = self.retrieve_rows[best][:count]
        if len(selected) != count or any(i < 0 for i in selected):
            raise ValueError('Cannot commit padding')
        selected_ids = [self.token_rows[i] for i in selected]
        if candidates[best, :count].tolist() != selected_ids:
            raise ValueError('Selected native token/tree alignment mismatch')
        before = len(self.committed_ids)
        native_ids = self.tokens[selected].to(self.committed_ids.device)
        ids = torch.cat((self.committed_ids, native_ids))
        if new_input_ids is not None and not torch.equal(self._ids(new_input_ids).to(ids.device), ids):
            raise ValueError('Native updated prefix differs from selected tree nodes')
        ordinary = [i for i in selected if self.token_rows[i] not in self.special_ids]
        new_states = self.tree_raw[ordinary].float()
        # Advanced indexing and cat produce dedicated compact storage. No
        # view retains the discarded full prefill or another branch's states.
        self.history_states = torch.cat((self.history_states, new_states), dim=0)[-self.history_window:].clone()
        additions = [(self.token_rows[node], before + k) for k, node in enumerate(selected)
                     if self.token_rows[node] not in self.special_ids]
        self.history_ids = (self.history_ids + [x[0] for x in additions])[-self.history_window:]
        self.history_positions = (self.history_positions + [x[1] for x in additions])[-self.history_window:]
        self.committed_ids = ids
        self.contexts = {}
        self.tree_raw = None
        self._tree_ready = False
        self.context_counters['committed_history_updates'] += 1
        self.context_counters['history_storage_bytes_peak'] = max(
            self.context_counters['history_storage_bytes_peak'], self.history_states.numel() * self.history_states.element_size())
        return self

    def restore(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._mode = self._captured = self.tree_raw = self.history_states = self.visual_rows = None
        self.contexts = {}
