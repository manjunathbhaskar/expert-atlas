"""Offloading: replace the full MoE block with a kept-expert block.

These tests use a hand-built fake OLMoE-like stack, not the real 13GB model,
to verify the offloaded block's numerical behaviour and the model-patching
helpers. The real benchmark is `scripts/run_offloading_baseline.py`.
"""

import pytest
import torch
import torch.nn as nn
from transformers.activations import ACT2FN

from expertatlas.offloading import (
    OffloadedExperts,
    OffloadedSparseMoeBlock,
    OffloadedTopKRouter,
    estimate_expert_flops_per_token,
    estimate_expert_memory,
    offload_model,
    offloading_savings_summary,
    restore_model,
)

N_LAYERS = 2
N_EXPERTS = 8
TOP_K = 2
HIDDEN = 8
INTERMEDIATE = 16


class _FakeCfg:
    num_local_experts = N_EXPERTS
    num_experts_per_tok = TOP_K
    hidden_size = HIDDEN
    intermediate_size = INTERMEDIATE
    hidden_act = "silu"
    norm_topk_prob = False
    num_hidden_layers = N_LAYERS


class _FakeFullExperts(nn.Module):
    """Exact clone of OlmoeExperts forward, tiny."""

    def __init__(self, seed=0):
        super().__init__()
        self.num_experts = N_EXPERTS
        self.hidden_dim = HIDDEN
        self.intermediate_dim = INTERMEDIATE
        torch.manual_seed(seed)
        self.gate_up_proj = nn.Parameter(torch.randn(N_EXPERTS, 2 * INTERMEDIATE, HIDDEN))
        self.down_proj = nn.Parameter(torch.randn(N_EXPERTS, HIDDEN, INTERMEDIATE))
        self.act_fn = ACT2FN["silu"]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            mask = mask.permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for e in hit:
            e = e[0]
            if e == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[e])
            x = hidden_states[token_idx]
            gate, up = nn.functional.linear(x, self.gate_up_proj[e]).chunk(2, dim=-1)
            h = self.act_fn(gate) * up
            h = nn.functional.linear(h, self.down_proj[e])
            h = h * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, h.to(final.dtype))
        return final


class _FakeFullGate(nn.Module):
    """Exact clone of OlmoeTopKRouter forward, tiny."""

    def __init__(self, seed=1):
        super().__init__()
        self.top_k = TOP_K
        self.num_experts = N_EXPERTS
        self.norm_topk_prob = False
        self.hidden_dim = HIDDEN
        torch.manual_seed(seed)
        self.weight = nn.Parameter(torch.randn(N_EXPERTS, HIDDEN))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, HIDDEN)
        logits = nn.functional.linear(hidden_states, self.weight)
        probs = torch.nn.functional.softmax(logits, dtype=torch.float, dim=-1)
        topv, topi = torch.topk(probs, self.top_k, dim=-1)
        topv = topv.to(logits.dtype)
        return logits, topv, topi


class _FakeMlp(nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.gate = _FakeFullGate(seed)
        self.experts = _FakeFullExperts(seed)

    def forward(self, hidden_states):
        b, s, h = hidden_states.shape
        hs = hidden_states.view(-1, h)
        _, w, idx = self.gate(hs)
        out = self.experts(hs, idx, w)
        return out.reshape(b, s, h)


def _fake_model():
    model = nn.Module()
    inner = nn.Module()
    layers = nn.ModuleList()
    for i in range(N_LAYERS):
        layer = nn.Module()
        layer.mlp = _FakeMlp(i)
        layers.append(layer)
    inner.layers = layers
    model.model = inner
    model.config = _FakeCfg()
    return model


def test_offloaded_full_kept_matches_original():
    """Keeping all experts should be numerically identical to the original block."""
    model = _fake_model()
    x = torch.randn(1, 7, HIDDEN)
    orig = [m(x).detach().clone() for m in (model.model.layers[i].mlp for i in range(N_LAYERS))]

    # Offload keeping the whole set; restore afterwards.
    with torch.no_grad():
        offload_model(model, {i: set(range(N_EXPERTS)) for i in range(N_LAYERS)}, reset=False)
    off = [model.model.layers[i].mlp(x).detach().clone() for i in range(N_LAYERS)]
    restore_model(model)

    for a, b in zip(orig, off):
        assert torch.allclose(a, b, atol=1e-4, rtol=1e-3)


def test_offloaded_partial_kept_only_uses_kept_experts():
    """Restricting to a subset must not involve the removed experts' weights."""
    model = _fake_model()
    x = torch.randn(1, 5, HIDDEN)

    kept = {0: {0, 1, 2, 3}}
    with torch.no_grad():
        offload_model(model, kept, reset=False)
    out = model.model.layers[0].mlp(x)
    # Inspect the offloaded block BEFORE restoring, or the original full MLP
    # overwrites it.
    block = model.model.layers[0].mlp
    assert block.gate.num_experts == 4
    assert block.gate.top_k == 2
    assert block.experts.gate_up_proj.shape == (4, 2 * INTERMEDIATE, HIDDEN)
    restore_model(model)

    # If we surgically set a kept expert's weights to zero, the offloaded
    # output for a token whose top-2 are both in {0,1} should be zero after
    # zeroing those two experts (and no other experts can save it because the
    # kept set excludes 4..7).
    keep = [0, 1]
    with torch.no_grad():
        offload_model(model, {0: set(keep)}, reset=False)
    model.model.layers[0].mlp.experts.gate_up_proj.data *= 0
    model.model.layers[0].mlp.experts.down_proj.data *= 0
    out_zeroed = model.model.layers[0].mlp(x)
    assert torch.allclose(out_zeroed, torch.zeros_like(out_zeroed), atol=1e-6)
    restore_model(model)


def test_offload_too_few_experts_left_as_full():
    """A keep set below top_k is a behavioural change; the module refuses to
    silently shrink top_k and leaves the layer as full."""
    model = _fake_model()
    with torch.no_grad():
        replaced = offload_model(model, {0: {0, 1}}, reset=False)  # top_k=2, keep 2 = OK
        left = offload_model(model, {1: {0}}, reset=False)         # top_k=2, keep 1 < top_k
    assert replaced[0] == 2
    assert 1 not in left
    # Layer 1 was left as full; no new block, so the mlp is still the original
    assert not hasattr(model.model.layers[1].mlp, "_original_full_mlp")


def test_offload_estimates_are_monotone_in_kept_experts():
    for k in (2, 4, 8):
        mem = estimate_expert_memory(k, HIDDEN, INTERMEDIATE)
        flops = estimate_expert_flops_per_token(k, HIDDEN, INTERMEDIATE, TOP_K)
        assert mem > 0 and flops > 0
    assert estimate_expert_memory(8, HIDDEN, INTERMEDIATE) > estimate_expert_memory(2, HIDDEN, INTERMEDIATE)


def test_offloading_savings_summary():
    keep = {0: {0, 1, 2}, 1: {4, 5}}
    s = offloading_savings_summary(keep, _FakeCfg())
    assert s["total_full_ffn_bytes"] > s["total_kept_ffn_bytes"]
    assert 0 < s["ffn_bytes_frac"] < 1
    assert s["kept_per_layer"][0] > s["kept_per_layer"][1]
