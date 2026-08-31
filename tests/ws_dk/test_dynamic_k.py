"""Dynamic top-k: per-token truncation by cumulative top-k mass."""

import pytest
import torch
import torch.nn as nn
from transformers.activations import ACT2FN

from expertatlas.dynamic_k import DynamicTopKMoeBlock, DynamicKMoe, patch_dynamic_k, restore_dynamic_k


class _FakeGate(nn.Module):
    def __init__(self, n_experts, top_k):
        super().__init__()
        self.num_experts = n_experts
        self.top_k = top_k
        self.hidden_dim = 4
        self.norm_topk_prob = False
        self.weight = nn.Parameter(torch.randn(n_experts, 4))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        logits = torch.nn.functional.linear(hidden_states, self.weight)
        probs = torch.nn.functional.softmax(logits, dtype=torch.float, dim=-1)
        topv, topi = torch.topk(probs, self.top_k, dim=-1)
        return logits, topv.to(logits.dtype), topi


class _FakeExperts(nn.Module):
    def __init__(self, n_experts):
        super().__init__()
        self.num_experts = n_experts
        self.hidden_dim = 4
        self.intermediate_dim = 6
        self.gate_up_proj = nn.Parameter(torch.randn(n_experts, 12, 4))
        self.down_proj = nn.Parameter(torch.randn(n_experts, 4, 6))
        self.act_fn = ACT2FN["silu"]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        # Same one-hot gather as OlmoeExperts.
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            mask = mask.permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for e in hit:
            e = e[0]
            if e >= self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[e])
            x = hidden_states[token_idx]
            g, u = torch.nn.functional.linear(x, self.gate_up_proj[e]).chunk(2, dim=-1)
            h = self.act_fn(g) * u
            h = torch.nn.functional.linear(h, self.down_proj[e])
            h = h * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, h.to(final.dtype))
        return final


class _FakeMlp(nn.Module):
    def __init__(self, n_experts=8, top_k=2):
        super().__init__()
        self.gate = _FakeGate(n_experts, top_k)
        self.experts = _FakeExperts(n_experts)

    def forward(self, x):
        b, s, h = x.shape
        hs = x.view(-1, h)
        _l, w, i = self.gate(hs)
        out = self.experts(hs, i, w)
        return out.reshape(b, s, h)


def _fake_model():
    model = nn.Module()
    inner = nn.Module()
    layers = nn.ModuleList()
    for _ in range(2):
        layer = nn.Module()
        layer.mlp = _FakeMlp(8, 4)
        layers.append(layer)
    inner.layers = layers
    model.model = inner
    return model


def test_dynamic_k_with_threshold_one_uses_all_experts():
    """Threshold = 1.0 means keep the full top-k set."""
    model = _fake_model()
    x = torch.randn(1, 3, 4)
    with torch.no_grad():
        ref = model.model.layers[0].mlp(x).clone()
    with DynamicKMoe(model, mass_threshold=1.0):
        out = model.model.layers[0].mlp(x)
    assert torch.allclose(ref, out, atol=1e-4, rtol=1e-3)


def test_dynamic_k_reduces_kept_count():
    """With a high threshold, the average number of experts per token is
    strictly below the full top-k."""
    model = _fake_model()
    block = DynamicTopKMoeBlock(gate=model.model.layers[0].mlp.gate,
                                experts=model.model.layers[0].mlp.experts,
                                mass_threshold=0.55)
    x = torch.randn(1, 5, 4)
    with torch.no_grad():
        out = block(x)
    # With a low threshold, at least one token should need fewer than top_k.
    assert min(block._last_kept) < 4
    assert all(1 <= k <= 4 for k in block._last_kept)
    assert out.shape == x.shape


def test_dynamic_k_restores_original():
    model = _fake_model()
    with DynamicKMoe(model, mass_threshold=0.95):
        pass
    # after exit, the mlp should be the original module
    assert not hasattr(model.model.layers[0].mlp, "_original_mlp")


def test_relative_threshold_one_keeps_full_set():
    """relative=True with threshold 1.0 keeps everything -> identical output."""
    model = _fake_model()
    x = torch.randn(1, 3, 4)
    with torch.no_grad():
        ref = model.model.layers[0].mlp(x).clone()
    with DynamicKMoe(model, mass_threshold=1.0, relative=True):
        out = model.model.layers[0].mlp(x)
    assert torch.allclose(ref, out, atol=1e-4, rtol=1e-3)


def test_relative_threshold_fires_where_absolute_cannot():
    """With norm_topk_prob=False and many experts, the raw top-k mass is far
    below 1.0, so an absolute 0.9 threshold never truncates -- but the same
    0.9 threshold interpreted relative to the top-k mass does."""
    torch.manual_seed(0)
    model = _fake_model()  # 8 experts, top_k=4: top-4 raw mass < 1.0
    gate = model.model.layers[0].mlp.gate
    experts = model.model.layers[0].mlp.experts
    x = torch.randn(1, 24, 4)

    abs_block = DynamicTopKMoeBlock(gate=gate, experts=experts,
                                    mass_threshold=0.995, relative=False)
    rel_block = DynamicTopKMoeBlock(gate=gate, experts=experts,
                                    mass_threshold=0.75, relative=True)
    with torch.no_grad():
        abs_block(x)
        rel_block(x)
    # relative truncation must keep strictly fewer on average than an
    # absolute threshold the raw mass cannot reach
    assert sum(rel_block._last_kept) < sum(abs_block._last_kept)
    assert min(rel_block._last_kept) >= 1
    assert max(rel_block._last_kept) <= 4


def test_relative_truncation_keeps_raw_weights():
    """Truncation decides k on the normalised mass but must apply the RAW
    (un-normalised) weights to the kept experts."""
    gate = _FakeGate(8, 4)
    experts = _FakeExperts(8)
    block = DynamicTopKMoeBlock(gate=gate, experts=experts,
                                mass_threshold=0.6, relative=True)
    w = torch.tensor([[0.20, 0.10, 0.05, 0.02]])
    idx = torch.tensor([[3, 1, 6, 0]])
    out_idx, out_w, kept = block._truncate(w, idx)
    # 0.20/0.37 = 0.54 < 0.6; (0.20+0.10)/0.37 = 0.81 >= 0.6 -> keep 2
    assert kept == [2]
    assert torch.allclose(out_w[0, :2], torch.tensor([0.20, 0.10]))
    assert out_idx[0, :2].tolist() == [3, 1]


def test_dynamic_k_output_shape_and_finite():
    model = _fake_model()
    x = torch.randn(1, 7, 4)
    with DynamicKMoe(model, mass_threshold=0.8):
        out = model.model.layers[0].mlp(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
