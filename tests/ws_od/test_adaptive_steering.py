"""Unit tests for entropy-triggered adaptive steering (tiny fake gate)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from expertatlas.adaptive_steering import AdaptiveEntropySteering, entropy_bits


class FakeGate(nn.Module):
    def __init__(self, num_experts=8, hidden_dim=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.norm_topk_prob = False
        self.weight = nn.Parameter(torch.randn(num_experts, hidden_dim))

    def forward(self, hidden_states):
        hs = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(hs, self.weight)
        probs = torch.softmax(logits, dtype=torch.float, dim=-1)
        v, i = torch.topk(probs, self.top_k, dim=-1)
        return logits, v.to(logits.dtype), i


class FakeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = FakeGate()


class FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = FakeMLP()


class FakeModel(nn.Module):
    def __init__(self, n_layers=2):
        super().__init__()
        inner = nn.Module()
        inner.layers = nn.ModuleList([FakeLayer() for _ in range(n_layers)])
        self.model = inner


def test_entropy_bits_uniform():
    logits = torch.zeros(3, 8)
    ent = entropy_bits(logits)
    assert torch.allclose(ent, torch.full((3,), 3.0), atol=1e-5)  # log2(8)


def test_no_trigger_below_threshold_means_identity():
    torch.manual_seed(0)
    m = FakeModel()
    x = torch.randn(5, 4)
    base = m.model.layers[0].mlp.gate(x)
    with AdaptiveEntropySteering(m, {0: {1, 2}}, delta=5.0,
                                 tau_by_layer={0: 100.0}) as st:
        out = m.model.layers[0].mlp.gate(x)
    assert st.trigger_counts[0] == 0 and st.token_counts[0] == 5
    for a, b in zip(base, out):
        assert torch.equal(a, b)


def test_trigger_boosts_targets_at_fired_tokens_only():
    torch.manual_seed(0)
    m = FakeModel()
    gate = m.model.layers[0].mlp.gate
    x = torch.randn(6, 4)
    raw = F.linear(x, gate.weight)
    ent = entropy_bits(raw)
    tau = float(ent.median())
    fire = ent > tau
    with AdaptiveEntropySteering(m, {0: {3}}, delta=2.5,
                                 tau_by_layer={0: tau}) as st:
        biased, _v, _i = gate(x)
    assert st.trigger_counts[0] == int(fire.sum()) > 0
    expected = raw.clone()
    expected[fire, 3] += 2.5
    assert torch.allclose(biased, expected, atol=1e-5)


def test_selection_actually_changes_under_large_boost():
    torch.manual_seed(1)
    m = FakeModel()
    gate = m.model.layers[0].mlp.gate
    x = torch.randn(4, 4)
    _l, _v, base_idx = gate(x)
    with AdaptiveEntropySteering(m, {0: {7}}, delta=100.0,
                                 tau_by_layer={0: -1.0}):
        _l2, _v2, idx = gate(x)
    assert (idx == 7).any(dim=-1).all()
    assert not torch.equal(base_idx, idx)


def test_remove_restores_original_forward():
    m = FakeModel()
    gate = m.model.layers[1].mlp.gate
    orig = gate.forward
    st = AdaptiveEntropySteering(m, {1: {0}}, delta=1.0, tau_by_layer={1: 0.0})
    assert gate.forward.__func__ is not orig.__func__ if hasattr(gate.forward, "__func__") else True
    st.remove()
    assert gate.forward == orig  # bound-method equality: same func, same instance


def test_missing_threshold_raises():
    m = FakeModel()
    with pytest.raises(ValueError, match="without an entropy threshold"):
        AdaptiveEntropySteering(m, {0: {1}}, delta=1.0, tau_by_layer={})
