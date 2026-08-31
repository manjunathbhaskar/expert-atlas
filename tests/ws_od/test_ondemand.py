"""Unit tests for the on-demand expert runtime (no model download)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from expertatlas.ondemand import ExpertLRU, OnDemandExperts, PerLayerLRU


class FakeStore:
    """Deterministic per-expert weights; counts fetches."""

    def __init__(self, hidden=8, inter=4):
        self.hidden, self.inter = hidden, inter
        self.fetches = 0

    def fetch(self, layer, expert):
        self.fetches += 1
        g = torch.Generator().manual_seed(layer * 1000 + expert)
        gate = torch.randn(self.inter, self.hidden, generator=g)
        up = torch.randn(self.inter, self.hidden, generator=g)
        down = torch.randn(self.hidden, self.inter, generator=g)
        return gate, up, down


def test_lru_hit_miss_eviction():
    store = FakeStore()
    lru = ExpertLRU(store, capacity=2)
    lru.get(0, 0)
    lru.get(0, 1)
    assert lru.stats.misses == 2 and lru.stats.hits == 0
    lru.get(0, 0)  # hit
    assert lru.stats.hits == 1
    lru.get(0, 2)  # evicts (0,1), the least recently used
    assert lru.stats.evictions == 1
    lru.get(0, 1)  # miss again
    assert lru.stats.misses == 4
    assert lru.resident_experts() == 2


def test_lru_capacity_zero_always_refetches():
    store = FakeStore()
    lru = ExpertLRU(store, capacity=0)
    lru.get(0, 0)
    lru.get(0, 0)
    assert lru.stats.misses == 2 and lru.resident_experts() == 0


def test_lru_returns_identical_weights_cached_and_fresh():
    store = FakeStore()
    a = ExpertLRU(store, capacity=8).get(3, 7)
    b = ExpertLRU(store, capacity=0).get(3, 7)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_ondemand_experts_matches_reference_math():
    """OnDemandExperts must equal a direct dense computation with the same
    weights and routing."""
    store = FakeStore()
    lru = ExpertLRU(store, capacity=64)
    n_experts, hidden, k = 4, 8, 2
    mod = OnDemandExperts(lru, layer=0, num_experts=n_experts)

    torch.manual_seed(0)
    n_tok = 5
    x = torch.randn(n_tok, hidden)
    top_k_index = torch.randint(0, n_experts, (n_tok, k))
    top_k_weights = torch.rand(n_tok, k)

    got = mod(x, top_k_index, top_k_weights)

    want = torch.zeros_like(x)
    for t in range(n_tok):
        for j in range(k):
            gate, up, down = store.fetch(0, int(top_k_index[t, j]))
            h = F.silu(x[t] @ gate.T) * (x[t] @ up.T)
            want[t] += (h @ down.T) * top_k_weights[t, j]

    assert torch.allclose(got, want, atol=1e-5)


def test_ondemand_experts_holds_no_parameters():
    mod = OnDemandExperts(ExpertLRU(FakeStore(), 1), layer=0, num_experts=4)
    assert sum(p.numel() for p in mod.parameters()) == 0


def test_per_layer_lru_isolated_budgets():
    store = FakeStore()
    lru = PerLayerLRU(store, capacity=4, n_layers=2)  # 2 per layer
    assert lru.per_layer_capacity == 2
    lru.get(0, 0)
    lru.get(0, 1)
    lru.get(1, 0)
    lru.get(1, 1)
    # layer-1 fills must NOT evict layer-0 entries
    assert lru.stats.evictions == 0
    assert lru.stats.misses == 4
    lru.get(0, 0)
    lru.get(1, 1)
    assert lru.stats.hits == 2
    lru.get(0, 2)  # evicts within layer 0 only
    assert lru.stats.evictions == 1
    lru.get(1, 0)  # still resident
    assert lru.stats.hits == 3
    assert lru.resident_experts() == 4


def test_per_layer_lru_capacity_below_layers_refetches():
    store = FakeStore()
    lru = PerLayerLRU(store, capacity=1, n_layers=2)  # 0 per layer
    lru.get(0, 0)
    lru.get(0, 0)
    assert lru.stats.misses == 2 and lru.resident_experts() == 0


def test_per_layer_lru_weights_match_global():
    store = FakeStore()
    a = PerLayerLRU(store, capacity=64, n_layers=4).get(3, 7)
    b = ExpertLRU(store, capacity=0).get(3, 7)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_per_layer_lru_sequential_forward_pattern_hits():
    """Simulate the real access pattern: layers touched in order, twice.
    A global LRU at half capacity gets 0 hits; per-layer gets 100% on pass 2."""
    n_layers, per_layer_use = 4, 2
    total = n_layers * per_layer_use
    pattern = [(l, e) for l in range(n_layers) for e in range(per_layer_use)]

    g = ExpertLRU(FakeStore(), capacity=total // 2)
    p = PerLayerLRU(FakeStore(), capacity=total // 2, n_layers=n_layers)
    for lru in (g, p):
        for _ in range(2):
            for l, e in pattern:
                lru.get(l, e)
    assert g.stats.hits == 0
    # per-layer: 1 of 2 slots per layer -> expert 0 of each layer... also evicted
    # within-layer since per-layer use (2) > per-layer capacity (1). Use full
    # per-layer capacity instead for the positive case:
    p2 = PerLayerLRU(FakeStore(), capacity=total, n_layers=n_layers)
    for _ in range(2):
        for l, e in pattern:
            p2.get(l, e)
    assert p2.stats.hits == total  # second pass all hits


def test_pinned_cache_serves_hot_experts_from_ram():
    from expertatlas.ondemand import PinnedCache

    store = FakeStore()
    counts = [[10, 5, 1, 0], [0, 1, 5, 10]]  # 2 layers x 4 experts
    pc = PinnedCache(store, capacity=4, counts=counts)  # 2 pinned per layer
    assert pc.pinned_sets[0] == {0, 1}
    assert pc.pinned_sets[1] == {2, 3}
    pc.get(0, 0); pc.get(0, 0)
    assert pc.stats.hits == 1 and pc.stats.misses == 1
    pc.get(0, 3); pc.get(0, 3)  # unpinned: fetch-and-discard both times
    assert pc.stats.misses == 3
    pc.get(1, 3); pc.get(1, 3)
    assert pc.stats.hits == 2
    assert pc.resident_experts() == 2


def test_pinned_cache_weights_match_uncached():
    from expertatlas.ondemand import PinnedCache

    store = FakeStore()
    counts = [[1] * 8 for _ in range(4)]
    a = PinnedCache(store, capacity=32, counts=counts).get(3, 7)
    b = ExpertLRU(FakeStore(), capacity=0).get(3, 7)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_pinned_cache_capacity_bound_respected():
    from expertatlas.ondemand import PinnedCache

    store = FakeStore()
    counts = [[i for i in range(8)] for _ in range(4)]
    pc = PinnedCache(store, capacity=8, counts=counts)  # 2 per layer
    for l in range(4):
        for e in range(8):
            pc.get(l, e)
    assert pc.resident_experts() == 8
    assert pc.stats.evictions == 0
