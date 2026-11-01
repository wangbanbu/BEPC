"""Lean legal-prefix observer; production avoids redundant GPU assertions.

API: SpeedContextObserver(...same constructor as RefinedContextObserver).
parent_hidden(parent) reads legal h only, without T/context construction.
note_candidates(candidate_rows) reuses controller CPU candidate data, optional.
note_selection(best:int,length:int) reuses CPU selected path at commit, optional.
Audit=True retains independent native mask/position/prefix/token checks.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch

import refined_observer as _old


def legal_paths(retrieve_rows, node_count):
    """Exact CPU ancestry; shared nodes must have exactly the same full prefix."""
    paths = {}
    for row in retrieve_rows:
        path, padded = [], False
        for node in row:
            if node == -1:
                padded = True
                continue
            if padded or node < 0 or node >= node_count or node in path:
                raise ValueError('Malformed native retrieve row')
            path.append(node)
            if path[0] != 0 or paths.setdefault(node, tuple(path)) != tuple(path):
                raise ValueError('Inconsistent native ancestor prefix')
    if set(paths) != set(range(node_count)):
        raise ValueError('Retrieve rows do not cover native nodes')
    return paths


class SpeedContextObserver(_old.RefinedContextObserver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._selection_hint = None
        self._candidate_rows_hint = None
        self.token_rows = None
        self.context_counters.update(
            production_gpu_prefix_equal_skips=0, production_gpu_mask_checks_skipped=0,
            production_gpu_position_copies_skipped=0, reused_cpu_candidate_trees=0,
            reused_cpu_commit_selections=0, lazy_tree_token_readbacks=0,
            parent_hidden_reads_without_history=0)

    def begin_tree(self, tree_tokens, input_ids, retrieve_indices, tree_mask,
                   tree_position_ids=None):
        if self.committed_ids is None:
            raise RuntimeError('Prefill not finished')
        ids = self._ids(input_ids)
        if len(ids) != len(self.committed_ids):
            raise ValueError('Committed prefix length differs from native input')
        if self.audit:
            # Frozen code validates the actual native prefix, masks and
            # positions independently; all audit work remains in parity only.
            super().begin_tree(tree_tokens, input_ids, retrieve_indices,
                               tree_mask, tree_position_ids)
        else:
            self.tokens = self._ids(tree_tokens)
            self.retrieve = retrieve_indices.detach()
            self.retrieve_rows = self.retrieve.tolist()
            self.paths = legal_paths(self.retrieve_rows, len(self.tokens))
            self.token_rows = None  # populated from existing CPU candidates
            self.contexts = {}
            self.tree_raw = None
            self._tree_ready = False
            self._arm('tree')
            self.context_counters['production_gpu_prefix_equal_skips'] += 1
            self.context_counters['production_gpu_mask_checks_skipped'] += 1
            self.context_counters['production_gpu_position_copies_skipped'] += int(tree_position_ids is not None)
        self._selection_hint = self._candidate_rows_hint = None

    def ensure_visual_rows(self):
        """Pool remains identical and timed in prefill; return its cached rows."""
        if self.visual_rows is None:
            raise RuntimeError("Finish native prefill before requesting visual rows")
        return self.visual_rows

    def note_candidates(self, candidate_rows):
        """Use already-copied CPU rows, retaining full tree-token consistency."""
        if not self._tree_ready or len(candidate_rows) != len(self.retrieve_rows):
            raise ValueError('Candidate rows do not match completed native tree')
        tokens = [None] * len(self.tokens)
        for nodes, row in zip(self.retrieve_rows, candidate_rows):
            if len(nodes) != len(row):
                raise ValueError('Candidate/retrieve width mismatch')
            for node, token in zip(nodes, row):
                if node < 0:
                    if token >= 0:
                        raise ValueError('Nonpadding candidate at padding node')
                    continue
                if token < 0 or (tokens[node] is not None and tokens[node] != token):
                    raise ValueError('One native node maps to inconsistent candidate tokens')
                tokens[node] = token
        if any(token is None for token in tokens):
            raise ValueError('Candidates do not cover native nodes')
        if self.audit and tokens != self.tokens.tolist():
            raise ValueError('CPU candidates differ from actual native tree tokens')
        self.token_rows = tokens
        self._candidate_rows_hint = candidate_rows
        self.context_counters['reused_cpu_candidate_trees'] += 1

    def note_selection(self, best, length):
        self._selection_hint = (int(best), int(length))

    def _ensure_cpu_tokens(self):
        if self.token_rows is None:
            self.token_rows = self.tokens.tolist()
            self.context_counters['lazy_tree_token_readbacks'] += 1

    @torch.no_grad()
    def parent_hidden(self, parent):
        parent = int(parent)
        if not self._tree_ready or parent not in self.paths:
            raise ValueError('Not a legal native completed parent')
        self.context_counters['parent_hidden_reads_without_history'] += 1
        return self.tree_raw[parent].float()

    def context(self, parent):
        self._ensure_cpu_tokens()
        return super().context(parent)

    @torch.no_grad()
    def commit(self, best_candidate, accept_length, retrieve_indices, candidates,
               new_input_ids=None):
        if self.audit:
            return super().commit(best_candidate, accept_length, retrieve_indices,
                                  candidates, new_input_ids)
        if not self._tree_ready:
            raise RuntimeError('No native tree available for commit')
        if self._selection_hint is None:
            best, length = int(best_candidate), int(accept_length)
        else:
            best, length = self._selection_hint
            self.context_counters['reused_cpu_commit_selections'] += 1
        count = length + 1
        if not 0 <= best < len(self.retrieve_rows):
            raise ValueError('Invalid selected native row')
        selected = self.retrieve_rows[best][:count]
        if len(selected) != count or any(node < 0 for node in selected):
            raise ValueError('Cannot commit native padding')
        self._ensure_cpu_tokens()
        before = len(self.committed_ids)
        ordinary = [node for node in selected if self.token_rows[node] not in self.special_ids]
        states = self.tree_raw[ordinary].float()
        self.history_states = torch.cat((self.history_states, states), dim=0)[-self.history_window:].clone()
        additions = [(self.token_rows[node], before + j) for j, node in enumerate(selected)
                     if self.token_rows[node] not in self.special_ids]
        self.history_ids = (self.history_ids + [x[0] for x in additions])[-self.history_window:]
        self.history_positions = (self.history_positions + [x[1] for x in additions])[-self.history_window:]
        if new_input_ids is None:
            self.committed_ids = torch.cat((self.committed_ids, self.tokens[selected].to(self.committed_ids.device)))
        else:
            updated = self._ids(new_input_ids)
            if len(updated) != before + count:
                raise ValueError('Native updated prefix length differs from selected path')
            # The native update has already constructed this exact tensor. Reuse
            # it instead of concatenating the same full prompt on GPU again.
            self.committed_ids = updated
        self.contexts = {}
        self.tree_raw = None
        self._tree_ready = False
        self._selection_hint = self._candidate_rows_hint = None
        self.context_counters['committed_history_updates'] += 1
        self.context_counters['history_storage_bytes_peak'] = max(
            self.context_counters['history_storage_bytes_peak'],
            self.history_states.numel() * self.history_states.element_size())
        return self


RefinedContextObserver = SpeedContextObserver
