"""LVSpec visual-relative verification on unchanged native ViSpec trees.

Paper: https://arxiv.org/html/2604.05650 sections3.1--3.3.
Released reference: zju-jiyicheng/LVSpec commit
84dde16ea16b39402552eb4eb95d26f9c5101d15.

This is a declared chain-to-tree port: rank only positions in one legal native
root-to-leaf path, using the released parent-hidden/logit alignment. No global
sibling pooling, invented probability gate, BEPC/ME2/4 budget, or extra forward.
"""
from __future__ import annotations
from collections import Counter
import math

import torch
import torch.nn.functional as F


PAPER_DEFAULTS = dict(loose_ratio=.7, similarity_top_n=10, draft_window=10,
                      pst_enabled=True, pst_include_root=True,
                      visual_reference='image_tokens', stable_rank_ties=True,
                      disable_relaxation=False)
DEVELOPMENT_RATIO_GRID = (0., .3, .5, .7)


def legal_branch_paths(retrieve_rows, node_count):
    """Reject unrelated-prefix node reuse, nonterminal padding and cycles."""
    paths = {}
    for row in retrieve_rows:
        path, padded = [], False
        for node in row:
            if node == -1:
                padded = True
                continue
            if padded or node < 0 or node >= node_count or node in path:
                raise ValueError('Malformed native retrieve path')
            path.append(node)
            if path[0] != 0 or paths.setdefault(node, tuple(path)) != tuple(path):
                raise ValueError('Native node reused under different ancestor prefixes')
    if set(paths) != set(range(node_count)):
        raise ValueError('Native retrieve paths must cover all real tree nodes')
    return paths


@torch.no_grad()
def visual_relevance(visual_hidden, text_hidden, top_n=10):
    """FP32 Top-N cosine mean; hidden states must share final-norm coordinates."""
    if visual_hidden.ndim != 2 or text_hidden.ndim != 2 or visual_hidden.shape[1] != text_hidden.shape[1]:
        raise ValueError('Expected visual[V,d] and text[N,d] in the same space')
    if not isinstance(top_n, int) or top_n < 1 or not visual_hidden.shape[0]:
        raise ValueError('Need positive Top-N and at least one real visual token')
    if visual_hidden.device != text_hidden.device:
        raise ValueError('Visual and text hidden states must share device')
    if not bool(torch.isfinite(visual_hidden).all()) or not bool(torch.isfinite(text_hidden).all()):
        raise ValueError('Nonfinite LVSpec hidden state')
    visual = F.normalize(visual_hidden.float(), dim=-1)
    text = F.normalize(text_hidden.float(), dim=-1)
    similarities = text @ visual.T
    return similarities.topk(min(top_n, len(visual)), dim=-1).values.mean(-1)


def branch_low_relevance_mask(scores, loose_ratio=.7, draft_window=10, stable_ties=True):
    """Rank positions on ONE legal branch, never all nodes of a verification tree.

    A chain of10 positions and ratio.7 agrees with released absolute k=7.
    Short ViSpec branches use floor(ratio*actual_window), a declared adaptation.
    ratio0 alone does not disable PST and therefore is not a Strict fallback.
    """
    if scores.ndim != 1 or not math.isfinite(float(loose_ratio)) or not 0 <= loose_ratio <= 1:
        raise ValueError('Require 1D branch scores and ratio in[0,1]')
    if not isinstance(draft_window, int) or draft_window < 1:
        raise ValueError('draft_window must be a positive integer')
    width = min(len(scores), draft_window)
    count = math.floor(float(loose_ratio)*width)
    allowed = torch.zeros(len(scores), device=scores.device, dtype=torch.bool)
    if count:
        indices = torch.argsort(scores[:width], stable=bool(stable_ties))[:count]
        allowed[indices] = True
    return allowed


def scan_branch(tokens, targets, rank_allowed, *, eos, committed_length,
                visible_boundary, pst_enabled=True, pst_include_root=True,
                draft_window=10, disable_relaxation=False):
    """Native longest-prefix logic with LVSpec rank/PST alternatives.

    Original LVSpec has no response/round mismatch cap. Common stop safety:
    EOS mismatches and relaxation beyond max_new_tokens are never accepted.
    Exact tokens retain native priority even after EOS/budget, as in ViSpec.
    """
    if len(targets) != len(tokens)-1 or len(rank_allowed) != len(targets):
        raise ValueError('Branch tokens/targets/relative mask misaligned')
    search = tokens[:draft_window+1] if pst_include_root else tokens[1:draft_window+1]
    search = {int(token) for token in search if token >= 0}
    length = mismatches = 0
    eos_seen = tokens[0] == eos
    reasons = Counter()
    for j, d in enumerate(tokens[1:]):
        a = targets[j]
        if d < 0:
            break
        if d == a:
            accepted, reason = True, 'exact'
        elif disable_relaxation:
            accepted, reason = False, 'strict_fallback'
        elif eos_seen or a == eos or d == eos or committed_length+j+1 >= visible_boundary:
            accepted, reason = False, 'common_stop_guard'
        else:
            low = bool(rank_allowed[j])
            pst = bool(pst_enabled and j < draft_window and a in search)
            accepted = low or pst
            reason = 'visual_and_pst' if low and pst else 'visual' if low else 'pst' if pst else 'strict_visual_anchor'
        reasons[reason] += 1
        if not accepted:
            break
        length += 1
        mismatches += d != a
        eos_seen = eos_seen or d == eos
    return length, mismatches, reasons


class LVSpecController:
    FUNCTIONS = ('initialize_tree', 'tree_decoding', 'evaluate_posterior', 'update_inference_inputs')

    def __init__(self, runtime, model, variant, prompt_ids, image_grid, *,
                 collect=False, audit=False, max_new_tokens=128,
                 image_id=None, prompt_kind=None, **unused):
        if collect:
            raise ValueError('LVSpec benchmark does not collect future-continuation probes')
        self.runtime, self.model, self.variant = runtime, model, dict(variant)
        self.parameters = dict(PAPER_DEFAULTS, **{key:value for key,value in variant.items() if key in PAPER_DEFAULTS})
        ratio = float(self.parameters['loose_ratio'])
        if not math.isfinite(ratio) or not 0 <= ratio <= 1:
            raise ValueError('loose_ratio must lie in[0,1]')
        if self.parameters['similarity_top_n'] != 10 or self.parameters['draft_window'] != 10:
            raise ValueError('This benchmark freezes paper N=10 and maximum chain windowK=10')
        if self.parameters['visual_reference'] != 'image_tokens':
            raise ValueError('Main benchmark uses paper-defined real image-token reference only')
        self.audit, self.image_id, self.prompt_kind = bool(audit), image_id, prompt_kind
        self.eos = int(model.tokenizer.eos_token_id)
        self.image_token_id = int(model.base_model.config.image_token_id)
        self.prompt_ids = prompt_ids
        self.visible_boundary = prompt_ids.shape[-1] + int(max_new_tokens)
        self.original = {name:getattr(runtime, name) for name in self.FUNCTIONS}
        self.norm = model.base_model.model.norm
        self.handle = self.norm.register_forward_hook(self._capture)
        self.capture_mode = self.captured = self.visual = self.tree_hidden = None
        self.committed_ids = None
        self.acceptance_paths = []
        self.stats = Counter(rounds=0, accepted_draft=0, accepted_mismatches=0,
            committed_mismatches=0, native_parent_hidden_rows=0, similarity_computations=0,
            visual_hidden_rows=0, branch_ranking_sets=0, ranked_positions=0,
            selected_exact_tokens=0, selected_visual_relaxations=0, selected_pst_relaxations=0,
            selected_visual_and_pst_relaxations=0, selected_common_stop_guards=0)
        self.branch_widths = Counter()
        self.selected_mismatches = None
        runtime.initialize_tree = self.prefill
        runtime.tree_decoding = self.tree
        runtime.evaluate_posterior = self.evaluate
        runtime.update_inference_inputs = self.update

    def _capture(self, module, args, output):
        if self.capture_mode is None:
            return
        if self.captured is not None or output.ndim != 3 or output.shape[0] != 1:
            raise ValueError('Expected exactly one native Target final-norm output')
        self.captured = output[0].detach()

    def _arm(self, mode):
        if self.capture_mode is not None or self.captured is not None:
            raise RuntimeError('Previous LVSpec native capture not consumed')
        self.capture_mode = mode

    def prefill(self, *args, **kwargs):
        ids = args[0] if args else kwargs['input_ids']
        disabled = self.parameters['disable_relaxation']
        if not disabled:
            self._arm('prefill')
        result = self.original['initialize_tree'](*args, **kwargs)
        if not disabled:
            if self.captured is None or len(self.captured) != ids.shape[-1]:
                raise ValueError('Prefill final-norm states do not match expanded prompt')
            image_rows = ids[0].to(self.captured.device) == self.image_token_id
            if not bool(image_rows.any()):
                raise ValueError('No real image-token rows for LVSpec visual reference')
            self.visual = self.captured[image_rows].float()
            self.stats['visual_hidden_rows'] = len(self.visual)
        self.committed_ids = ids.detach().clone()
        self.capture_mode = self.captured = None
        return result

    def tree(self, model, tokens, cache, positions, ids, retrieve):
        self.tokens, self.ids, self.retrieve = tokens, ids, retrieve
        self.retrieve_rows = retrieve.tolist()
        self.paths = legal_branch_paths(self.retrieve_rows, tokens.numel())
        if self.audit:
            if self.committed_ids is None or not torch.equal(ids, self.committed_ids.to(ids.device)):
                raise ValueError('Native tree prefix differs from committed prefix')
            mask = model.base_model.model.tree_mask.detach().reshape(tokens.numel(), tokens.numel()).bool().cpu()
            for node, path in self.paths.items():
                if set(torch.where(mask[node])[0].tolist()) != set(path):
                    raise ValueError('Native tree attention differs from legal ancestor path')
            if positions.reshape(-1).cpu().tolist() != [len(self.paths[i])-1 for i in range(tokens.numel())]:
                raise ValueError('Native tree position does not match ancestry depth')
        disabled = self.parameters['disable_relaxation']
        if not disabled:
            self._arm('tree')
        result = self.original['tree_decoding'](model, tokens, cache, positions, ids, retrieve)
        if not disabled:
            if self.captured is None or len(self.captured) != tokens.numel():
                raise ValueError('Native tree final-norm states do not align with real nodes')
            self.tree_hidden = self.captured
            self.stats['native_parent_hidden_rows'] += len(self.tree_hidden)
        self.capture_mode = self.captured = None
        self.stats['rounds'] += 1
        return result

    def evaluate(self, logits, candidates, processor):
        if processor is not None:
            raise ValueError('LVSpec port is T=0 only; no probabilistic rejection rule is claimed')
        token_rows = candidates.tolist()
        targets = logits[:, :-1].argmax(-1).tolist()
        if len(token_rows) != len(self.retrieve_rows):
            raise ValueError('Candidate rows do not match native retrieve paths')
        disabled = self.parameters['disable_relaxation']
        scores = None
        if not disabled:
            if self.visual is None or self.tree_hidden is None:
                raise RuntimeError('LVSpec visual/native tree states not available')
            scores = visual_relevance(self.visual, self.tree_hidden, self.parameters['similarity_top_n'])
            self.stats['similarity_computations'] += 1
        best, best_length, best_mismatches, best_reasons = 0, -1, 0, Counter()
        native_tokens = self.tokens.reshape(-1).tolist() if self.audit else None
        for row, tokens in enumerate(token_rows):
            nodes = self.retrieve_rows[row]
            real = next((i for i,node in enumerate(nodes) if node < 0), len(nodes))
            if real < 1 or len(tokens) != len(nodes):
                raise ValueError('Malformed candidate/retrieve row')
            if self.audit:
                for node, token in zip(nodes[:real], tokens[:real]):
                    if token != native_tokens[node]:
                        raise ValueError('Candidate token differs from its real tree node')
            width = max(0, real-1)
            if disabled:
                allowed = [False]*len(targets[row])
            else:
                # Exact released alignment: hidden at parent nodes[j] and
                # logits[row,j] both correspond to candidate token[j+1].
                parent_scores = scores[nodes[:width]]
                local = branch_low_relevance_mask(parent_scores, self.parameters['loose_ratio'],
                    self.parameters['draft_window'], self.parameters['stable_rank_ties']).tolist()
                allowed = local + [False]*(len(targets[row])-len(local))
                self.stats['branch_ranking_sets'] += 1
                self.stats['ranked_positions'] += min(width, self.parameters['draft_window'])
                self.branch_widths[width] += 1
            length, mismatches, reasons = scan_branch(tokens, targets[row], allowed,
                eos=self.eos, committed_length=self.ids.shape[-1], visible_boundary=self.visible_boundary,
                pst_enabled=self.parameters['pst_enabled'], pst_include_root=self.parameters['pst_include_root'],
                draft_window=self.parameters['draft_window'], disable_relaxation=disabled)
            if length > best_length:
                best, best_length, best_mismatches, best_reasons = row, length, mismatches, reasons
        self.selected_mismatches = best_mismatches
        self.stats['accepted_draft'] += best_length
        self.stats['accepted_mismatches'] += best_mismatches
        for key, value in best_reasons.items():
            self.stats['selected_scan_reason_'+key] += value
        self.stats['selected_exact_tokens'] += best_reasons['exact']
        self.stats['selected_visual_relaxations'] += best_reasons['visual']
        self.stats['selected_pst_relaxations'] += best_reasons['pst']
        self.stats['selected_visual_and_pst_relaxations'] += best_reasons['visual_and_pst']
        self.stats['selected_common_stop_guards'] += best_reasons['common_stop_guard']
        self.acceptance_paths.append(dict(round=self.stats['rounds']-1, row=best,
            accepted_draft=best_length, mismatches=best_mismatches,
            response_mismatches_before=self.stats['committed_mismatches'],
            nodes=self.retrieve_rows[best][:best_length+1], tokens=token_rows[best][:best_length+1]))
        terminal = self.original['update_inference_inputs']
        note = getattr(terminal, 'note_selection', None)
        if callable(note):
            note(best, best_length, token_rows[best][:best_length+1])
        return (torch.tensor(best, device=logits.device),
                torch.tensor(best_length, device=logits.device), logits[best, best_length])

    def update(self, *args, **kwargs):
        if self.selected_mismatches is None:
            raise RuntimeError('Native update without selected LVSpec path')
        result = self.original['update_inference_inputs'](*args, **kwargs)
        self.stats['committed_mismatches'] += self.selected_mismatches
        self.selected_mismatches = None
        self.committed_ids = result[0].detach()
        self.tree_hidden = None
        return result

    def final_stats(self):
        return dict(self.stats, family='LVSpec', lvspec_parameters=dict(self.parameters),
            effective_branch_width_histogram={str(k):v for k,v in sorted(self.branch_widths.items())},
            round_mismatch_budget=None, response_mismatch_budget=None,
            committed_response_mismatches=self.stats['committed_mismatches'],
            original_lvspec_mismatch_budget='unlimited', additional_target_forwards=0,
            extra_target_forwards=0, hidden_coordinate='post_target_final_RMSNorm_pre_lm_head',
            hidden_alignment='native parent node j predicting candidate token j+1; released-code alignment',
            local_rank_set='one legal actual root-to-leaf path only; floor(ratio*min(actual_positions,10))',
            pst_scope='same actual branch, root included per released code, maximum10Draft+root',
            introduced_probability_or_entropy_gate=False,
            retained_visual_reference='only actual image_token_id rows of native Target prefill',
            common_stop_safety='exact-only EOS and no relaxed mismatch beyond visible token budget')

    def restore(self):
        for name, original in self.original.items():
            setattr(self.runtime, name, original)
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self.capture_mode = self.captured = self.visual = self.tree_hidden = None


LVSPECController = LVSpecController
