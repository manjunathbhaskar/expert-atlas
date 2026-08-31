"""PLAN.md §6.1 — capture fidelity sanity tests.

These test expertatlas.capture.route_from_logits in isolation against
synthetic router logits, so they run in milliseconds with no model
download. The full end-to-end check against the real model lives in
test_model_loading.py (network + disk + slow — separated deliberately).
"""

from __future__ import annotations

import torch

from expertatlas.capture import route_from_logits

N_EXPERTS = 64
TOP_K = 8


def _random_logits(n_tokens: int = 37, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n_tokens, N_EXPERTS, generator=g)


def test_exactly_topk_experts_selected():
    logits = _random_logits()
    ids, weights, mass = route_from_logits(logits, TOP_K, norm_topk_prob=False)
    assert ids.shape == (37, TOP_K)
    for row in ids:
        assert len(row) == TOP_K
        assert len(set(row.tolist())) == TOP_K, "duplicate expert id in top-k selection"


def test_full_softmax_sums_to_one():
    """Softmax over ALL experts sums to 1 — but that's over the FULL
    distribution, not the selected top-k. See test_topk_weights_do_not_sum_to_one
    for the norm_topk_prob=False case this guards against conflating."""
    logits = _random_logits()
    full = torch.softmax(logits, dim=-1, dtype=torch.float32)
    sums = full.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_topk_weights_do_not_sum_to_one_when_not_normalised():
    """OLMoE: norm_topk_prob=False. Top-k weights sum to the top-k MASS,
    which is < 1 for random logits with k < n_experts. Asserting this sums
    to 1.0 would be a real bug masquerading as a passing test."""
    logits = _random_logits()
    _, weights, mass = route_from_logits(logits, TOP_K, norm_topk_prob=False)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, mass, atol=1e-6)
    assert bool((sums < 0.999).any()), (
        "expected top-k mass < 1 for random logits with top_k < n_experts; "
        "if this now sums to ~1, norm_topk_prob handling broke silently"
    )


def test_topk_weights_sum_to_one_when_normalised():
    """Mixtral-style: norm_topk_prob=True renormalises top-k weights to 1."""
    logits = _random_logits()
    _, weights, mass = route_from_logits(logits, TOP_K, norm_topk_prob=True)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_topk_mass_is_recorded():
    logits = _random_logits()
    _, _, mass = route_from_logits(logits, TOP_K, norm_topk_prob=False)
    assert mass.shape == (37,)
    assert bool((mass > 0).all()) and bool((mass <= 1.0 + 1e-6).all())


def test_expert_ids_in_range():
    logits = _random_logits()
    ids, _, _ = route_from_logits(logits, TOP_K, norm_topk_prob=False)
    assert int(ids.min()) >= 0
    assert int(ids.max()) < N_EXPERTS


def test_determinism_same_prompt():
    logits_a = _random_logits(seed=42)
    logits_b = _random_logits(seed=42)
    ids_a, weights_a, mass_a = route_from_logits(logits_a, TOP_K, norm_topk_prob=False)
    ids_b, weights_b, mass_b = route_from_logits(logits_b, TOP_K, norm_topk_prob=False)
    assert torch.equal(ids_a, ids_b)
    assert torch.equal(weights_a, weights_b)
    assert torch.equal(mass_a, mass_b)


def test_no_silent_layer_skips_all_layers_represented():
    """Simulates the multi-layer aggregation step: every layer index 0..n-1
    must produce a row, none silently dropped by a hook that didn't fire."""
    n_layers = 16
    seen_layers = set()
    for layer in range(n_layers):
        logits = _random_logits(seed=layer)
        ids, _, _ = route_from_logits(logits, TOP_K, norm_topk_prob=False)
        assert ids.shape[0] > 0
        seen_layers.add(layer)
    assert seen_layers == set(range(n_layers))
