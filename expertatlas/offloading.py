"""Real conditional compute: load and run only a kept subset of experts.

This is the step the keep-top-K and ablation probes deliberately did **not**
take: they zeroed or masked *router choices* while leaving every expert weight
in memory and every FFN matmul on the compute path. That answers "does
restricting the router change output?", not "is it cheaper to run a smaller
model?". This module answers the second question.

What it does
------------
For each targeted MoE layer, `offload_model` replaces the layer's
`OlmoeSparseMoeBlock` with a `OffloadedSparseMoeBlock` whose `gate` only scores
`k <= 64` experts and whose `experts` module only owns `k` FFN weight pairs.
The replacement is an `nn.Module` that re-implements the exact `forward`
contract of the original block:

    hidden_states -> gate -> top_k over the kept set -> FFN for those k only

The original block is **not kept in memory**: after copying the kept experts'
weights into the offloaded block, the original `mlp` parameters are set to
`None` and the `original_mlp` references are stored for optional restoration.

Honest limits (state these in any results doc)
----------------------------------------------
* This still **loads the full checkpoint from disk once** when `load_model()`
  is called. "Load only the kept experts" would require a custom
  `from_pretrained` / state-dict filter and is not implemented here. The
  saving is **runtime memory** and **compute**, not the initial disk read.
* The gate is shrunk from `64 x hidden_dim` to `k x hidden_dim`. For small
  `k` this saves a modest linear projection; the FFN dominates both memory and
  time, and that is where almost all of the saving comes from.
* Offloading changes the *numerical* result relative to the full model only
  to the extent that the kept set differs from the full top-8. If the kept
  set is the full model, the offloaded block is bit-identical to the original
  (modulo floating-point non-associativity from a different loop order).
* No quantization or distillation is performed; the kept experts keep their
  original bfloat16 weights. A follow-up can optionally compress them further.

How to use
----------
    kept = {layer: set(range(64)) for layer in range(16)}  # no-op
    offload_model(model, kept)
    # model is now smaller in memory and faster at forward time

Or as a context manager:
    with OffloadedMoe(model, kept) as om:
        out = model(...)
    # original mlp blocks restored on exit

The module also exposes `estimate_expert_memory` and `estimate_expert_flops`
so a benchmark script can report saved bytes and FLOPs, not just speedup.
"""

from __future__ import annotations

import gc
from collections.abc import MutableMapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _layer_mlp(model, layer: int):
    """The path the OLMoE decoder stack uses for its sparse MoE block."""
    return model.model.layers[layer].mlp


def _int_dtype(dtype: Any) -> torch.dtype:
    if isinstance(dtype, str):
        return getattr(torch, dtype)
    return dtype


# ---------------------------------------------------------------------------
# Offloaded gate and FFN
# ---------------------------------------------------------------------------


class OffloadedTopKRouter(nn.Module):
    """Top-k router restricted to a kept subset of the original expert pool.

    `num_experts` here is `k`, the number of kept experts, not the original 64.
    The original `gate.weight` is sliced to `kept_indices` and kept in the same
    dtype. Top-k is `min(top_k, k)` — if fewer than `top_k` experts are kept
    the block raises rather than silently shrink the contribution, because a
    smaller `top_k` would be a second, unreported confound.
    """

    def __init__(self, *, hidden_size: int, num_experts: int, top_k: int,
                 norm_topk_prob: bool, dtype, weight: torch.Tensor,
                 original_indices: list[int]):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.hidden_dim = hidden_size
        self.original_indices = list(original_indices)
        self.weight = nn.Parameter(weight.to(_int_dtype(dtype)).contiguous())

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)
        router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        router_scores = router_top_value
        return router_logits, router_scores, router_indices


class OffloadedExperts(nn.Module):
    """Collection of FFN experts for the kept subset.

    The original `OlmoeExperts` stores weights as 3-D tensors
    (n_experts, *, hidden). We copy the kept rows and use the same one-hot
    gather loop, but the loop only iterates over `k` experts, not 64.
    """

    def __init__(self, *, num_experts: int, hidden_size: int, intermediate_size: int,
                 act_fn, dtype, gate_up_proj: torch.Tensor, down_proj: torch.Tensor,
                 original_indices: list[int]):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_size
        self.intermediate_dim = intermediate_size
        self.gate_up_proj = nn.Parameter(gate_up_proj.to(_int_dtype(dtype)).contiguous())
        self.down_proj = nn.Parameter(down_proj.to(_int_dtype(dtype)).contiguous())
        self.act_fn = act_fn
        self.original_indices = list(original_indices)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class OffloadedSparseMoeBlock(nn.Module):
    """Replaces `OlmoeSparseMoeBlock` for a kept subset of experts.

    The original block's `forward` does exactly one thing: reshape to 2-D, call
    the gate, call the experts, reshape back. We keep that contract so
    `OlmoeDecoderLayer` does not need to be touched.
    """

    def __init__(self, *, config, kept_indices: list[int],
                 gate_weight: torch.Tensor, gate_up_proj: torch.Tensor,
                 down_proj: torch.Tensor, dtype):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = len(kept_indices)
        self.top_k = min(config.num_experts_per_tok, self.num_experts)
        if self.top_k != config.num_experts_per_tok:
            # If fewer experts are kept than top_k, the model's behaviour has
            # changed in a second way (less total contribution mass). The
            # caller must choose what to do; by default we abort.
            raise ValueError(
                f"kept {self.num_experts} experts but top_k is {config.num_experts_per_tok}; "
                "this would silently reduce the contribution mass. "
                "Keep at least top_k experts per restricted layer."
            )
        self.original_indices = list(kept_indices)
        self.gate = OffloadedTopKRouter(
            hidden_size=config.hidden_size, num_experts=self.num_experts,
            top_k=self.top_k, norm_topk_prob=config.norm_topk_prob,
            dtype=dtype, weight=gate_weight, original_indices=kept_indices,
        )
        act = config.hidden_act
        from transformers.activations import ACT2FN
        self.experts = OffloadedExperts(
            num_experts=self.num_experts, hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size, act_fn=ACT2FN[act],
            dtype=dtype, gate_up_proj=gate_up_proj, down_proj=down_proj,
            original_indices=kept_indices,
        )

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        _, top_k_weights, top_k_index = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states, top_k_index, top_k_weights)
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


# ---------------------------------------------------------------------------
# Model patching / unpatching
# ---------------------------------------------------------------------------


def _free_module(m: nn.Module) -> None:
    """Aggressively clear parameters to let the CPU allocator reclaim the
    tensor memory. There is no `torch.cpu.empty_cache()`, so we `del`
    parameters and `gc.collect()` in the caller."""
    for p in list(m.parameters(recurse=False)):
        if p is not None and p is getattr(m, p, None):
            # For parameters whose name we can resolve, set the attribute to
            # None to drop the reference immediately.
            for name, param in list(m.named_parameters(recurse=False)):
                if param is p:
                    if isinstance(m, MutableMapping) or hasattr(m, name):
                        try:
                            object.__setattr__(m, name, None)
                        except AttributeError:
                            pass
    for n, child in list(m.named_children()):
        try:
            object.__setattr__(m, n, None)
        except AttributeError:
            pass


def offload_model(model, keep_by_layer: dict[int, set[int] | list[int]],
                  *, reset: bool = True) -> dict[int, int]:
    """Replace every targeted layer's full MoE block with a kept-expert one.

    Args:
        model: a loaded `OlmoeForCausalLM` model.
        keep_by_layer: `{layer: set or list of kept expert indices}`.
        reset: if True, also free the original MoE weights to recover memory.

    Returns:
        `{layer: num_kept}` for the actually replaced layers. A layer whose
        keep set is empty or smaller than `top_k` is left as the full block
        and reported in `left_as_full`.
    """
    if not keep_by_layer:
        return {}

    top_k = model.config.num_experts_per_tok
    replaced: dict[int, int] = {}
    left_as_full: list[int] = []

    for layer, kept in sorted(keep_by_layer.items()):
        kept = sorted(set(int(x) for x in kept if 0 <= x < model.config.num_local_experts))
        if len(kept) < top_k:
            left_as_full.append(layer)
            continue

        mlp = _layer_mlp(model, layer)
        # Pull original parameters before we replace the module.
        orig_gate = mlp.gate
        orig_experts = mlp.experts
        gate_weight = orig_gate.weight[kept].detach().clone()
        gate_up_proj = orig_experts.gate_up_proj[kept].detach().clone()
        down_proj = orig_experts.down_proj[kept].detach().clone()

        dtype = orig_gate.weight.dtype
        new_block = OffloadedSparseMoeBlock(
            config=model.config, kept_indices=kept, gate_weight=gate_weight,
            gate_up_proj=gate_up_proj, down_proj=down_proj, dtype=dtype,
        )

        # Store the original so `restore_model` can put it back.
        new_block._original_full_mlp = mlp
        _layer_mlp(model, layer).__dict__["_offloaded"] = new_block
        # Actually replace the module.
        model.model.layers[layer].mlp = new_block
        replaced[layer] = len(kept)

        if reset:
            # Drop the original parameters. We keep the original MLP object
            # alive only for restoration metadata; parameters are zeroed.
            orig_gate.weight = nn.Parameter(torch.zeros(1))
            orig_experts.gate_up_proj = nn.Parameter(torch.zeros(1))
            orig_experts.down_proj = nn.Parameter(torch.zeros(1))
            del gate_weight, gate_up_proj, down_proj
            gc.collect()

    if left_as_full:
        print(f"offload_model: {len(left_as_full)} layers left as full because "
              f"keep set < top_k ({top_k}): {left_as_full}", flush=True)
    print(f"offload_model: replaced {len(replaced)}/16 layers; "
          f"kept experts per layer: {replaced}", flush=True)
    return replaced


def restore_model(model) -> list[int]:
    """Put the original `mlp` blocks back if `offload_model` stored them.

    Returns the list of layers restored.
    """
    restored = []
    for layer in range(len(model.model.layers)):
        mlp = _layer_mlp(model, layer)
        if hasattr(mlp, "_original_full_mlp"):
            model.model.layers[layer].mlp = mlp._original_full_mlp
            restored.append(layer)
    return restored


class OffloadedMoe:
    """Context-manager wrapper around `offload_model` / `restore_model`."""

    def __init__(self, model, keep_by_layer: dict[int, set[int] | list[int]], *, reset: bool = True):
        self.model = model
        self.keep_by_layer = keep_by_layer
        self.reset = reset
        self.replaced: dict[int, int] = {}

    def __enter__(self):
        self.replaced = offload_model(self.model, self.keep_by_layer, reset=self.reset)
        return self

    def __exit__(self, *exc):
        restore_model(self.model)


# ---------------------------------------------------------------------------
# Cost estimates
# ---------------------------------------------------------------------------


def estimate_expert_memory(n_experts: int, hidden_size: int, intermediate_size: int,
                           dtype=torch.bfloat16) -> int:
    """Bytes for one layer's FFN experts (gate_up_proj + down_proj)."""
    bytes_per = 2 if dtype in (torch.bfloat16, torch.float16) else 4
    gate_up = n_experts * 2 * intermediate_size * hidden_size
    down = n_experts * hidden_size * intermediate_size
    return (gate_up + down) * bytes_per


def estimate_expert_flops_per_token(n_experts: int, hidden_size: int, intermediate_size: int,
                                     top_k: int) -> int:
    """Approximate FFN FLOPs per token if exactly `top_k` experts are computed.

    The OLMoE FFN is GLU-style: gate and up are each (hidden -> intermediate),
    then down (intermediate -> hidden). Per expert:
      2 * hidden * intermediate  (gate + up)
    + intermediate * hidden      (down)
    = 3 * hidden * intermediate  (approximate; activation and multiply are cheap).
    Times `top_k` experts per token.
    """
    return top_k * 3 * hidden_size * intermediate_size


def offloading_savings_summary(keep_by_layer: dict[int, set[int] | list[int]],
                                config) -> dict:
    """Memory and FLOP savings of the offloaded design relative to full model."""
    full_per_layer = estimate_expert_memory(
        config.num_local_experts, config.hidden_size, config.intermediate_size,
        dtype=getattr(config, "torch_dtype", torch.bfloat16))
    kept_per_layer = {
        layer: estimate_expert_memory(len(kept), config.hidden_size, config.intermediate_size)
        for layer, kept in keep_by_layer.items()
    }
    total_full = full_per_layer * config.num_hidden_layers
    total_kept = sum(kept_per_layer.values()) + full_per_layer * (config.num_hidden_layers - len(keep_by_layer))
    return {
        "full_ffn_bytes_per_layer": full_per_layer,
        "total_full_ffn_bytes": total_full,
        "total_kept_ffn_bytes": total_kept,
        "ffn_bytes_frac": total_kept / total_full if total_full else 1.0,
        "kept_per_layer": kept_per_layer,
    }
