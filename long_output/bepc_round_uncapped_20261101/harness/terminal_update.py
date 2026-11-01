"""Skip native Draft construction only when native after-update stopping is certain.

TerminalUpdate(original_update, prompt_length=..., max_new_tokens=...,
               eos_token_id=..., enabled=True, audit=False)
implements the exact native update_inference_inputs signature. Root installs
this process-local callable before constructing a controller. Optional
note_selection(best:int,length:int,selected_tokens:CPU sequence) avoids the
extra production GPU terminal-predicate readback for controller-owned arms.
On guaranteed stop the original function still commits identical Target KV,
input IDs, current_length and greedy next token. Its otherwise unused next
Draft-tree outputs are None, making any accidental next iteration fail loudly.
"""
from __future__ import annotations
from collections import Counter
import torch


class TerminalUpdate:
    def __init__(self, original, *, prompt_length, max_new_tokens, eos_token_id,
                 enabled=True, audit=False):
        self.original = original
        self.prompt_length = int(prompt_length)
        self.max_new_tokens = int(max_new_tokens)
        self.eos_token_id = int(eos_token_id)
        self.enabled, self.audit = bool(enabled), bool(audit)
        self.stats = Counter(update_calls=0, terminal_draft_skips=0,
                             eos_terminal_skips=0, budget_terminal_skips=0,
                             cpu_selection_hints=0, gpu_terminal_predicate_reads=0,
                             stochastic_fallbacks=0)
        self._selection_hint = None

    def note_selection(self, best, length, selected_tokens):
        best, length = int(best), int(length)
        tokens = tuple(map(int, selected_tokens))
        if length < 0 or len(tokens) != length + 1:
            raise ValueError('Terminal selection must include root and accepted Draft tokens')
        self._selection_hint = (best, length, tokens)

    def _terminal(self, input_ids, candidates, best, length):
        hint, self._selection_hint = self._selection_hint, None
        if hint is not None:
            _, length_cpu, tokens = hint
            # Optional audit checks independently read the actual chosen block.
            if self.audit:
                if int(best) != hint[0] or int(length) != length_cpu or candidates[int(best), :length_cpu + 1].tolist() != list(tokens):
                    raise AssertionError('Terminal CPU hint differs from actual native selection')
            self.stats['cpu_selection_hints'] += 1
            eos = self.eos_token_id in tokens
            budget = input_ids.shape[-1] + length_cpu + 1 - self.prompt_length > self.max_new_tokens
            return eos, budget
        # Strict-native posterior has no existing CPU selection metadata. One
        # scalar readback evaluates both stop predicates without reading all
        # candidate rows or treating an off-path EOS as an actual terminal.
        row = candidates[best]
        positions = torch.arange(candidates.shape[-1], device=candidates.device)
        selected = positions <= length
        eos = ((row == self.eos_token_id) & selected).any()
        budget = input_ids.shape[-1] + length + 1 - self.prompt_length > self.max_new_tokens
        bits = eos.to(torch.int32) + 2 * torch.as_tensor(budget, device=eos.device, dtype=torch.int32)
        code = int(bits)
        self.stats['gpu_terminal_predicate_reads'] += 1
        return bool(code & 1), bool(code & 2)

    def __call__(self, input_ids, candidates, best_candidate, accept_length,
                 retrieve_indices, logits_processor, new_token,
                 past_key_values_data_list, current_length_data, model,
                 hidden_state_new, sample_p):
        self.stats['update_calls'] += 1
        args = (input_ids, candidates, best_candidate, accept_length,
                retrieve_indices, logits_processor, new_token,
                past_key_values_data_list, current_length_data, model,
                hidden_state_new, sample_p)
        if not self.enabled or logits_processor is not None:
            self._selection_hint = None
            self.stats['stochastic_fallbacks'] += int(logits_processor is not None)
            return self.original(*args)
        eos, budget = self._terminal(input_ids, candidates, best_candidate, accept_length)
        if not eos and not budget:
            return self.original(*args)
        self.stats['terminal_draft_skips'] += 1
        self.stats['eos_terminal_skips'] += int(eos)
        self.stats['budget_terminal_skips'] += int(budget)
        layer = model.spec_layer
        original_tree = layer.topK_genrate
        called = 0
        def unused_terminal_tree(*tree_args, **tree_kwargs):
            nonlocal called
            called += 1
            return None, None, None, None
        layer.topK_genrate = unused_terminal_tree
        try:
            result = self.original(*args)
        finally:
            layer.topK_genrate = original_tree
        if called != 1:
            raise AssertionError('Expected one otherwise-unused terminal native Draft construction')
        return result

    def statistics(self):
        return dict(self.stats, native_stop_rule='selected EOS or generated_count>max_new_tokens, after full KV commit',
                    next_tree_outputs_unused=True, additional_target_forwards=0,
                    original_update_KV_and_sample_p_preserved=True)
