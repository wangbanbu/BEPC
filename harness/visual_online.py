"""Read-only final-block visual scores for actual ViSpec tree decoding.

Uses the native Q/K projections, actual mRoPE embeddings, populated KV buffers,
and the native additive tree mask. It never replaces SDPA, asks for attention
weights, wraps runtime functions, or reads future generated Captions.

Controller use:
    visual = BatchVisualScores(base_model, prompt_ids, with_ccve=True)
    visual.enable(True)
    result = native_tree_forward(...)
    visual.enable(False)
    visual.prepare()
    scores = visual.scores(parent_nodes, native_top1_ids, draft_ids)
    # apply scores only where scores['valid'] and the controller's valid mask
    visual.close()
"""
from __future__ import annotations

import math
import torch


@torch.no_grad()
def grouped_visual_features(q, k, v, mask, visual_keys, wo=None):
    """FP32 E4 full-softmax visual partition for unique parent rows.

    q: [parents, query_heads, head_dim]; k/v: [kv_heads, keys, head_dim].
    mask: [parents, keys], additive (or bool True=allowed, as native SDPA).
    Returns head-mean visual mass and optional W_O-projected visual branch.
    The visual partition is never separately renormalized.
    """
    if q.ndim != 3 or k.ndim != 3 or mask.ndim != 2:
        raise ValueError('Expected q[P,H,D], k[KV,S,D], mask[P,S]')
    parents, heads, dim = q.shape
    kv_heads, keys, kdim = k.shape
    if heads % kv_heads or dim != kdim or mask.shape != (parents, keys):
        raise ValueError('GQA head, mask, or key dimensions do not align')
    if visual_keys.shape != (keys,) or visual_keys.dtype != torch.bool:
        raise ValueError('visual_keys must be a boolean key-position vector')
    groups = heads // kv_heads
    query = q.float().reshape(parents, kv_heads, groups, dim)
    score = torch.einsum('phgd,hsd->phgs', query, k.float()) / math.sqrt(dim)
    if mask.dtype == torch.bool:
        additive = torch.zeros_like(mask, dtype=torch.float32).masked_fill(~mask, -torch.inf)
    else:
        additive = mask.float()
    alpha = torch.softmax(score + additive[:, None, None, :], dim=-1)
    visual_alpha = alpha * visual_keys[None, None, None, :]
    mass = visual_alpha.sum(-1).mean(dim=(1, 2))
    if wo is None:
        # Deliberately does not promote/materialize V or W_O in the mass arm.
        return mass, None
    if v is None or v.shape != k.shape or wo.ndim != 2 or wo.shape[1] != heads * dim:
        raise ValueError('V or W_O does not align with grouped query heads')
    visual_heads = torch.einsum('phgs,hsd->phgd', visual_alpha, v.float())
    visual_output = visual_heads.reshape(parents, heads * dim) @ wo.float().T
    return mass, visual_output


class BatchVisualScores:
    """One request's observer; parent outputs are cached only within one tree.

    Padding is returned as finite zero placeholders with valid=False; callers
    must apply the explicit validity mask. This never makes padding an event.
    check_cache=True adds native rotated-K/V and finite-value assertions for
    validation. It does not switch the native attention backend.
    """

    def __init__(self, base_model, prompt_ids, *, with_ccve=False,
                 check_cache=False, rotary_fn=None):
        self.base = base_model
        self.block = base_model.model.layers[-1]
        self.attn = self.block.self_attn
        self.with_ccve = bool(with_ccve)
        self.check_cache = bool(check_cache)
        self.rotary_fn = rotary_fn
        ids = prompt_ids.detach()
        if ids.ndim == 2 and ids.shape[0] != 1:
            raise ValueError('Only native batch=1 is supported')
        if ids.ndim not in (1, 2):
            raise ValueError('Expected one expanded prompt token sequence')
        self.prompt_ids = ids.reshape(-1).clone()
        self.visual_positions = torch.where(self.prompt_ids == base_model.config.image_token_id)[0]
        if self.visual_positions.numel() == 0:
            raise ValueError('Expanded prompt has no image-token positions')
        self.Wo = None
        self.norm_weight = None
        self.norm_eps = None
        if self.with_ccve:
            if self.attn.o_proj.bias is not None or base_model.lm_head.bias is not None:
                raise ValueError('E4 CCVE definition requires bias-free output projections')
            self.Wo = self.attn.o_proj.weight.detach().to(dtype=torch.float32, copy=True)
            self.norm_weight = base_model.model.norm.weight.detach().to(dtype=torch.float32, copy=True)
            self.norm_eps = base_model.model.norm.variance_epsilon
        self.active = False
        self.closed = False
        self.prepared = False
        self.state = {}
        self.handles = []
        self.counters = dict(prepared_trees=0, attention_parent_rows=0,
                             visual_branch_parent_rows=0, score_calls=0)
        self.handles.append(self.attn.register_forward_pre_hook(self._attn_pre, with_kwargs=True))
        for name in ['q_proj', 'k_proj'] + (['v_proj'] if self.check_cache else []):
            self.handles.append(getattr(self.attn, name).register_forward_hook(self._projection(name)))
        if self.with_ccve:
            self.handles.append(self.block.register_forward_hook(self._block_post))

    def enable(self, active=True):
        if self.closed:
            raise RuntimeError('Observer has been closed')
        self.active = bool(active)
        if self.active:
            self.state = {}
            self.prepared = False

    def _attn_pre(self, module, args, kwargs):
        if not self.active:
            return
        if kwargs.get('output_attentions', False):
            raise ValueError('Observer requires native output_attentions=False')
        cache = kwargs.get('past_key_value')
        embeddings = kwargs.get('position_embeddings')
        if cache is None or embeddings is None:
            raise ValueError('Native KV cache and actual mRoPE embeddings are required')
        self.state.update(mask=kwargs.get('attention_mask'), cache=cache,
                          prefix=int(cache[0].current_length),
                          cos=embeddings[0], sin=embeddings[1])

    def _projection(self, name):
        def capture(module, args, output):
            if self.active:
                self.state[name] = output.detach()
        return capture

    def _block_post(self, module, args, output):
        if self.active:
            self.state['raw_x'] = output[0].detach()

    @torch.no_grad()
    def prepare(self):
        """Prepare the just-completed native tree before any posterior KV commit."""
        if self.closed or self.active:
            raise RuntimeError('Disable observer after the completed native tree before prepare')
        required = {'q_proj', 'k_proj', 'cache', 'prefix', 'cos', 'sin', 'mask'}
        if self.with_ccve:
            required.add('raw_x')
        if not required.issubset(self.state):
            raise RuntimeError('No complete active native tree capture')
        s = self.state
        if s['q_proj'].shape[0] != 1:
            raise ValueError('Only native batch=1 is supported')
        n = s['q_proj'].shape[1]
        heads, kv_heads, dim = self.attn.num_heads, self.attn.num_key_value_heads, self.attn.head_dim
        q = s['q_proj'].view(1, n, heads, dim).transpose(1, 2)
        k = s['k_proj'].view(1, n, kv_heads, dim).transpose(1, 2)
        rotary = self.rotary_fn
        if rotary is None:
            from vispec.model.modeling_qwen2_5_vl_kv import apply_multimodal_rotary_pos_emb
            rotary = apply_multimodal_rotary_pos_emb
        q, rotated_k = rotary(q, k, s['cos'], s['sin'], self.attn.rope_scaling['mrope_section'])
        K, V = s['cache']
        length = int(K.current_length)
        if length != int(V.current_length) or length != s['prefix'] + n:
            raise ValueError('KV was committed/changed after the observed native tree')
        if s['prefix'] < len(self.prompt_ids):
            raise ValueError('KV prefix cannot precede the original expanded prompt')
        if self.check_cache:
            if not torch.equal(K.data[:, :, s['prefix']:length], rotated_k):
                raise AssertionError('Actual native rotated K does not match captured mRoPE')
            native_v = s['v_proj'].view(1, n, kv_heads, dim).transpose(1, 2)
            if not torch.equal(V.data[:, :, s['prefix']:length], native_v):
                raise AssertionError('Actual native V does not match captured projection')
        mask = s['mask']
        if mask is None:
            # Native SDPA q_len=1 without mask has is_causal=False. A multi-node
            # tree requires its actual supplied mask; never invent a flat path.
            if n != 1:
                raise ValueError('Actual additive mask is required for multi-node tree attention')
            mask = torch.zeros((1, 1, 1, length), device=q.device, dtype=torch.float32)
        if mask.ndim != 4 or mask.shape[:2] != (1, 1) or mask.shape[2] != n or mask.shape[3] < length:
            raise ValueError('Expected native batch1/shared-head tree mask [1,1,n,>=KV]')
        self.q = q[0]
        self.K = K.data[0, :, :length].detach()
        self.V = V.data[0, :, :length].detach() if self.with_ccve else None
        self.mask = mask[0, 0, :, :length]
        self.visual_keys = torch.zeros(length, dtype=torch.bool, device=self.K.device)
        self.visual_keys[self.visual_positions.to(self.K.device)] = True
        self.tree_nodes = n
        self.key_length = length
        self._computed = torch.zeros(n, dtype=torch.bool, device=self.K.device)
        self._mass_cache = torch.zeros(n, dtype=torch.float32, device=self.K.device)
        self._visual_cache = self._D_cache = None
        if self.with_ccve:
            self._visual_cache = torch.empty((n, self.Wo.shape[0]), dtype=torch.float32, device=self.K.device)
            self._D_cache = torch.empty_like(self._visual_cache)
        self.prepared = True
        self.counters['prepared_trees'] += 1

    @torch.no_grad()
    def scores(self, parents, a, d):
        """Scores aligned to supplied parent/native-a/d arrays, including siblings."""
        if not self.prepared or self.closed or self.active:
            raise RuntimeError('prepare the completed native tree before scores')
        if parents.shape != a.shape or a.shape != d.shape:
            raise ValueError('parent, Target top1 and Draft arrays must have identical shape')
        shape = parents.shape
        p, aa, dd = parents.reshape(-1).long(), a.reshape(-1).long(), d.reshape(-1).long()
        valid = p.ge(0) & aa.ge(0) & dd.ge(0)
        if bool((p[valid] >= self.tree_nodes).any()):
            raise ValueError('Parent index lies outside the captured tree')
        vocab = self.base.lm_head.weight.shape[0]
        if bool(((aa[valid] >= vocab) | (dd[valid] >= vocab)).any()):
            raise ValueError('Candidate token lies outside the Target output vocabulary')
        unique = torch.unique(p[valid], sorted=True)
        new = unique[~self._computed[unique]]
        if new.numel():
            query = self.q[:, new].transpose(0, 1)
            mass, visual = grouped_visual_features(query, self.K, self.V, self.mask[new],
                                                   self.visual_keys, self.Wo)
            self._mass_cache[new] = mass
            if self.with_ccve:
                self._visual_cache[new] = visual
                raw = self.state['raw_x'][0, new].float()
                self._D_cache[new] = self.norm_weight * torch.rsqrt(raw.square().mean(-1, keepdim=True) + self.norm_eps)
                self.counters['visual_branch_parent_rows'] += new.numel()
            self._computed[new] = True
            self.counters['attention_parent_rows'] += new.numel()
        masses = torch.zeros_like(p, dtype=torch.float32)
        masses[valid] = self._mass_cache[p[valid]]
        ccve = None
        if self.with_ccve:
            ccve = torch.zeros_like(masses)
            head = self.base.lm_head.weight.detach()
            direction = head[aa[valid]].float() - head[dd[valid]].float()
            ccve[valid] = (direction * self._D_cache[p[valid]] * self._visual_cache[p[valid]]).sum(-1)
        if self.check_cache:
            if not bool(torch.isfinite(masses[valid]).all()) or (ccve is not None and not bool(torch.isfinite(ccve[valid]).all())):
                raise AssertionError('Nonfinite score on a valid candidate')
        self.counters['score_calls'] += 1
        return dict(visual_attention_mass=masses.reshape(shape),
                    CCVE=ccve.reshape(shape) if ccve is not None else None,
                    valid=valid.reshape(shape))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.active = False
        self.closed = True
        self.prepared = False
        self.state = {}
        self.Wo = self.norm_weight = None
        for name in ['q', 'K', 'V', 'mask', 'visual_keys', '_computed', '_mass_cache',
                     '_visual_cache', '_D_cache']:
            if hasattr(self, name):
                setattr(self, name, None)
