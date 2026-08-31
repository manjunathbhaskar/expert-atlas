"""Unit tests for the attention transport capture/boost interfaces."""

import pytest
import torch
from transformers import OlmoeConfig, OlmoeForCausalLM

from expertatlas.attention_transport import (
    HeadBoost,
    LastRowAttentionCapture,
    heads_to_by_layer,
)


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = OlmoeConfig(
        vocab_size=64, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        num_experts=4, num_experts_per_tok=2, max_position_embeddings=64,
        attn_implementation="eager",
    )
    m = OlmoeForCausalLM(cfg).eval()
    ids = torch.randint(0, 64, (1, 12))
    return m, ids


def test_capture_rows_shape_and_simplex(tiny):
    m, ids = tiny
    with LastRowAttentionCapture(m) as cap, torch.no_grad():
        m(input_ids=ids)
    assert sorted(cap.rows) == [0, 1]
    for r in cap.rows.values():
        assert r.shape == (4, 12)
        assert torch.allclose(r.sum(dim=-1), torch.ones(4), atol=1e-4)
    mass = cap.needle_mass((3, 6))
    assert mass.shape == (2, 4)
    assert (mass >= 0).all() and (mass <= 1.0001).all()


def test_capture_does_not_change_logits(tiny):
    m, ids = tiny
    with torch.no_grad():
        base = m(input_ids=ids).logits
    with LastRowAttentionCapture(m), torch.no_grad():
        cap_out = m(input_ids=ids).logits
    assert torch.allclose(base, cap_out, atol=1e-5)
    with torch.no_grad():
        after = m(input_ids=ids).logits
    assert torch.allclose(base, after, atol=1e-5)  # impl restored


def test_boost_moves_mass_onto_span_only_at_selected_heads(tiny):
    m, ids = tiny
    span = (3, 6)
    with LastRowAttentionCapture(m) as cap, torch.no_grad():
        m(input_ids=ids)
    before = cap.needle_mass(span)

    heads = heads_to_by_layer([(0, 1), (1, 2)])
    with HeadBoost(m, heads, span, beta=6.0) as hb:
        with LastRowAttentionCapture(m) as cap2, torch.no_grad():
            m(input_ids=ids)
    assert hb.n_fired == 2
    after = cap2.needle_mass(span)
    assert after[0, 1] > before[0, 1]
    assert after[1, 2] > before[1, 2]
    # layer-0 unselected heads see identical inputs -> unchanged mass
    for h in (0, 2, 3):
        assert torch.allclose(after[0, h], before[0, h], atol=1e-4)


def test_boost_zero_beta_is_noop(tiny):
    m, ids = tiny
    with torch.no_grad():
        base = m(input_ids=ids).logits
    with HeadBoost(m, {0: [0, 1], 1: [3]}, (2, 5), beta=0.0), torch.no_grad():
        out = m(input_ids=ids).logits
    assert torch.allclose(base, out, atol=1e-5)


def test_boost_changes_final_logits(tiny):
    m, ids = tiny
    with torch.no_grad():
        base = m(input_ids=ids).logits[0, -1]
    with HeadBoost(m, {0: [0], 1: [1]}, (3, 6), beta=8.0), torch.no_grad():
        out = m(input_ids=ids).logits[0, -1]
    assert not torch.allclose(base, out, atol=1e-4)
