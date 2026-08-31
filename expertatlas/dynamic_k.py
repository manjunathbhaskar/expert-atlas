"""Dynamic top-k: run only as many experts as the router's own mass demands.

This is the per-token, adaptive version of the static keep-set in
`expertatlas/offloading.py`. Instead of deciding *before* a forward pass which
experters are allowed, it lets the full router run for a token, looks at the
softmax probabilities it produces, and stops loading/executing FFN experts as
soon as a cumulative probability mass is reached.

Why this is a different question
--------------------------------
`expertatlas/offloading.py` asks: "if we only ever keep 8 of 64 experts, can
we still get the model to work?"  That is a **structural** pruning question.

This module asks: "if we let the router decide *per token* how many of its
chosen 8 experts are actually needed, can we save compute while preserving the
output?"  That is a **dynamic, uncertainty-adaptive** question.

The two can be combined: first prune the candidate pool to a kept set, then
apply dynamic-k inside that kept set. Only the first is implemented in
`offloading.py`; this module implements the second on the full or a pre-pruned
gate.

Honest limits
-------------
* The gate is still evaluated over the full (or kept) candidate pool, so the
  gate projection is not saved. The savings are in the FFN matmuls only.
* The `threshold` is a hyperparameter. This module does not tune it on the
  evaluated prompt; the caller should pick it on a held-out split.
* With `relative=False` (the default) the threshold is compared against the
  cumulative **raw** top-k weights. On models with `norm_topk_prob=False`
  (OLMoE), those weights sum to the top-k share of the full softmax — measured
  ~0.42 on OLMoE-1B-7B-0924 — so absolute thresholds like 0.9 are unreachable
  and truncation never fires. `relative=True` normalises by the token's total
  top-k mass first, so the threshold means "this fraction of the mass the
  router actually gave its top-k" and is reachable on any model.
* A token with diffuse router mass (all 8 experts ~0.125) will keep all 8. A
  token with one dominant expert may keep 1. The average `k` is the number to
  report.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicTopKExperts(nn.Module):
    """FFN experts that are called with a variable number of selected experts
    per token, determined by the caller's `top_k_index` and `top_k_weights`."""

    def __init__(self, *, num_experts: int, hidden_size: int, intermediate_size: int,
                 act_fn, gate_up_proj: torch.Tensor, down_proj: torch.Tensor):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_size
        self.intermediate_dim = intermediate_size
        self.gate_up_proj = gate_up_proj
        self.down_proj = down_proj
        self.act_fn = act_fn

    def forward(self, hidden_states, top_k_index, top_k_weights):
        # top_k_index may have a different number of columns per token, but
        # because torch one_hot needs a fixed num_classes and the matmul is
        # sparse, we loop over tokens. This is slower than the batched original
        # for full top-k, but for small k it is the same or better.
        final = torch.zeros_like(hidden_states)
        for t in range(hidden_states.shape[0]):
            idx = top_k_index[t]
            w = top_k_weights[t]
            for e, wt in zip(idx, w):
                if wt == 0:
                    continue
                g, u = F.linear(hidden_states[t:t+1], self.gate_up_proj[e]).chunk(2, dim=-1)
                h = self.act_fn(g) * u
                h = F.linear(h, self.down_proj[e])
                final[t] += h.to(final.dtype).squeeze(0) * wt
        return final


class DynamicTopKMoeBlock(nn.Module):
    """Wrap an existing offloaded or full MLP to truncate top-k by cumulative
    probability mass per token.

    The wrapper reuses the underlying `gate` and `experts` parameters. It does
    not own a separate set of weights; it only changes how many experts are
    actually computed.
    """

    def __init__(self, *, gate, experts, mass_threshold: float = 0.95,
                 relative: bool = False):
        super().__init__()
        self.gate = gate
        self.experts = experts
        self.mass_threshold = mass_threshold
        self.relative = relative

    def _truncate(self, topk_weights, topk_indices):
        """Per token, keep the smallest prefix of sorted top-k that exceeds
        `mass_threshold`. Returns truncated tensors and the mean kept count."""
        # topk_weights are not guaranteed sorted; sort by weight descending.
        w_sorted, sort_idx = torch.sort(topk_weights, dim=-1, descending=True)
        cum = torch.cumsum(w_sorted, dim=-1)
        if self.relative:
            cum = cum / cum[..., -1:].clamp_min(torch.finfo(cum.dtype).tiny)
        crossed = (cum >= self.mass_threshold).long()
        # If threshold is never reached (e.g. threshold=1.0 and float error),
        # keep the full set, not just the top-1.
        any_crossed = crossed.sum(dim=-1) > 0
        keep = torch.where(any_crossed, crossed.argmax(dim=-1) + 1,
                           torch.tensor(topk_weights.shape[-1], dtype=torch.long, device=topk_weights.device))
        max_keep = topk_weights.shape[-1]
        keep = keep.clamp(1, max_keep)

        new_idx, new_w, kept = [], [], []
        for t in range(topk_weights.shape[0]):
            k = int(keep[t].item())
            kept.append(k)
            order = sort_idx[t, :k]
            new_idx.append(topk_indices[t, order])
            new_w.append(w_sorted[t, :k])

        # Pad to a fixed size so the one-hot / gather operations can be
        # expressed as dense tensors (padding rows use zero weight).
        pad = max(kept)
        out_idx = torch.zeros(topk_weights.shape[0], pad, dtype=topk_indices.dtype, device=topk_indices.device)
        out_w = torch.zeros(topk_weights.shape[0], pad, dtype=topk_weights.dtype, device=topk_weights.device)
        for t, k in enumerate(kept):
            out_idx[t, :k] = new_idx[t]
            out_w[t, :k] = new_w[t]
        return out_idx, out_w, kept

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hs = hidden_states.view(-1, hidden_dim)
        _logits, topk_weights, topk_indices = self.gate(hs)
        topk_indices, topk_weights, kept = self._truncate(topk_weights, topk_indices)
        out = self.experts(hs, topk_indices, topk_weights)
        self._last_kept = kept
        return out.reshape(batch_size, sequence_length, hidden_dim)


def patch_dynamic_k(model, mass_threshold: float = 0.95, layers=None,
                    relative: bool = False):
    """Wrap each layer's `mlp` with `DynamicTopKMoeBlock`.

    The original `mlp` must expose `.gate` and `.experts`. If it is already an
    `OffloadedSparseMoeBlock`, it will be wrapped; if it is the full original,
    it will also be wrapped (and the gate still has 64 experts, but the FFN
    only runs the truncated set).

    Returns a list of wrapped layers.
    """
    wrapped = []
    for l in (layers if layers is not None else range(len(model.model.layers))):
        mlp = model.model.layers[l].mlp
        new_block = DynamicTopKMoeBlock(gate=mlp.gate, experts=mlp.experts,
                                        mass_threshold=mass_threshold, relative=relative)
        # stash original for restore
        new_block._original_mlp = mlp
        model.model.layers[l].mlp = new_block
        wrapped.append(l)
    return wrapped


def restore_dynamic_k(model):
    """Remove `DynamicTopKMoeBlock` wrappers and put the original mlps back."""
    restored = []
    for l in range(len(model.model.layers)):
        mlp = model.model.layers[l].mlp
        if hasattr(mlp, "_original_mlp"):
            model.model.layers[l].mlp = mlp._original_mlp
            restored.append(l)
    return restored


class DynamicKMoe:
    """Context manager for dynamic-k FFN truncation."""

    def __init__(self, model, mass_threshold: float = 0.95, layers=None,
                 relative: bool = False):
        self.model = model
        self.mass_threshold = mass_threshold
        self.layers = layers
        self.relative = relative
        self.wrapped: list[int] = []

    def __enter__(self):
        self.wrapped = patch_dynamic_k(self.model, self.mass_threshold, self.layers,
                                       relative=self.relative)
        return self

    def __exit__(self, *exc):
        restore_dynamic_k(self.model)
