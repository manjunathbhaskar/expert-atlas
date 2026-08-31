"""Unit tests for residual-stream anchor injection (tiny fake decoder stack)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from expertatlas.anchoring import AnchorInjector, AnchorSpec


class FakeDecoderLayer(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.seen: list[torch.Tensor] = []

    def forward(self, hidden_states, **kwargs):
        self.seen.append(hidden_states.detach().clone())
        return (hidden_states,)


class FakeModel(nn.Module):
    def __init__(self, n_layers=3, d=4):
        super().__init__()
        inner = nn.Module()
        inner.layers = nn.ModuleList([FakeDecoderLayer(d) for _ in range(n_layers)])
        self.model = inner

    def forward(self, hs):
        for layer in self.model.layers:
            (hs,) = layer(hs)
        return hs


def _spec(layer=1, s=2, e=4, d=4, alpha=0.5):
    return AnchorSpec(layer=layer, pos_start=s, pos_end=e,
                      vector=torch.ones(d), alpha=alpha)


def test_injects_only_at_target_layer_and_positions():
    torch.manual_seed(0)
    m = FakeModel()
    x = torch.randn(1, 6, 4)
    m(x)  # baseline pass
    with AnchorInjector(m, [_spec()]) as inj:
        m(x)
    assert inj.n_fired == 1
    base0, treat0 = m.model.layers[0].seen
    assert torch.equal(base0, treat0)  # layer 0 untouched
    base1, treat1 = m.model.layers[1].seen
    assert torch.equal(base1[:, :2], treat1[:, :2])
    assert torch.equal(base1[:, 4:], treat1[:, 4:])
    assert not torch.equal(base1[:, 2:4], treat1[:, 2:4])


def test_norm_matched_scaling():
    torch.manual_seed(1)
    m = FakeModel()
    x = torch.randn(1, 6, 4)
    alpha = 0.5
    with AnchorInjector(m, [_spec(alpha=alpha)]):
        m(x)
    base = x[:, 2:4, :]
    treat = m.model.layers[1].seen[0][:, 2:4, :]
    delta = (treat - base)[0, 0]
    expected = alpha * base.norm(dim=-1).mean()
    assert torch.allclose(delta.norm(), expected, atol=1e-5)
    # same delta at every position in the span
    assert torch.allclose(delta, (treat - base)[0, 1], atol=1e-6)


def test_skips_short_sequences():
    torch.manual_seed(2)
    m = FakeModel()
    x = torch.randn(1, 3, 4)  # shorter than pos_end=4
    with AnchorInjector(m, [_spec()]) as inj:
        out = m(x)
    assert inj.n_fired == 0
    assert torch.equal(out, x)


def test_hooks_removed_on_exit():
    torch.manual_seed(3)
    m = FakeModel()
    x = torch.randn(1, 6, 4)
    with AnchorInjector(m, [_spec()]):
        pass
    m(x)
    assert torch.equal(m.model.layers[1].seen[-1], x)


def test_empty_span_rejected():
    with pytest.raises(ValueError):
        AnchorInjector(FakeModel(),
                       [AnchorSpec(layer=0, pos_start=3, pos_end=3,
                                   vector=torch.ones(4), alpha=0.5)])
