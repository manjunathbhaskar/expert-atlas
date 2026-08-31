"""WS-C tests for the token-budget control and co-activation nulls."""

import numpy as np
import pytest

from expertatlas.aggregate import aggregate_counts, marginal_counts, subsample_cells
from expertatlas.coactivation import (
    coactivation_matrix, degree_preserving_null, detect_communities,
    pointwise_mutual_information,
)

N_LAYERS, N_EXPERTS, TOP_K = 2, 8, 3


def make_prompts():
    """Two cells that differ only in topic — and in length, 20 tokens vs 5.
    This reproduces the measured 1.73x topic/length confound in miniature."""
    return {
        0: {"topic": "python", "lang": "en", "register": "formal", "format": "prose"},
        1: {"topic": "history", "lang": "en", "register": "formal", "format": "prose"},
    }


def make_rows(lengths=(20, 5), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for pid, n_tok in enumerate(lengths):
        for pos in range(n_tok):
            for layer in range(N_LAYERS):
                rows.append({
                    "prompt_id": pid,
                    "token_pos": pos,
                    "layer": layer,
                    "expert_ids": rng.choice(N_EXPERTS, TOP_K, replace=False),
                    "gate_weights": np.full(TOP_K, 1 / TOP_K),
                })
    return rows


def test_subsample_equalises_token_budget():
    """The WS-B interface request. Without this, the long-prompt cell contributes
    4x the routing observations and length masquerades as topic affinity."""
    kept, budget = subsample_cells(make_rows(), make_prompts(), seed=0)
    assert budget == 5
    per_cell = {}
    for r in kept:
        t = make_prompts()[r["prompt_id"]]["topic"]
        per_cell.setdefault(t, set()).add((r["prompt_id"], r["token_pos"]))
    assert {k: len(v) for k, v in per_cell.items()} == {"python": 5, "history": 5}


def test_subsample_keeps_all_layers_of_a_kept_token():
    """A token must never be partially observed — all layers or none."""
    kept, _ = subsample_cells(make_rows(), make_prompts(), seed=0)
    layers = {}
    for r in kept:
        layers.setdefault((r["prompt_id"], r["token_pos"]), set()).add(r["layer"])
    assert all(v == set(range(N_LAYERS)) for v in layers.values())


def test_subsample_is_deterministic():
    a, _ = subsample_cells(make_rows(), make_prompts(), seed=42)
    b, _ = subsample_cells(make_rows(), make_prompts(), seed=42)
    key = lambda rs: sorted((r["prompt_id"], r["token_pos"], r["layer"]) for r in rs)
    assert key(a) == key(b)


def test_aggregate_shape_and_budget_recorded():
    cm = aggregate_counts(make_rows(), make_prompts(), N_LAYERS, N_EXPERTS)
    assert cm.counts.shape == (N_LAYERS * N_EXPERTS, 2)
    assert cm.tokens_per_cell == 5
    assert cm.expert_uids[0] == "L00E00"
    assert cm.domain_labels == ["history", "python"]


def test_aggregate_total_matches_kept_selections():
    cm = aggregate_counts(make_rows(), make_prompts(), N_LAYERS, N_EXPERTS)
    # 2 cells x 5 tokens x 2 layers x top_k selections
    assert cm.counts.sum() == pytest.approx(2 * 5 * N_LAYERS * TOP_K)


def test_marginal_counts_covers_every_factor():
    """The factorial payoff: separate lift per factor, so a 'code expert' can be
    distinguished from a 'syntax expert' and an 'English expert'."""
    m = marginal_counts(make_rows(), make_prompts(), N_LAYERS, N_EXPERTS)
    assert set(m) == {"topic", "lang", "register", "format"}
    assert m["lang"].counts.shape[1] == 1  # only 'en' in this fixture


# --- co-activation -----------------------------------------------------------

def test_coactivation_is_symmetric_with_zero_diagonal():
    m = coactivation_matrix(make_rows(), N_LAYERS, N_EXPERTS)
    np.testing.assert_allclose(m, m.T)
    assert np.allclose(np.diag(m), 0.0)


def test_within_layer_only_never_links_across_layers():
    m = coactivation_matrix(make_rows(), N_LAYERS, N_EXPERTS, within_layer_only=True)
    cross = m[:N_EXPERTS, N_EXPERTS:]
    assert cross.sum() == 0.0


def _simulate_cofiring(skew: float, n_draws: int = 120_000, seed: int = 0):
    """Structureless co-firing: pairs drawn independently by popularity only.
    Any PMI signal here is base-rate leakage, by construction."""
    rng = np.random.default_rng(seed)
    pop = np.ones(8)
    pop[0] = skew
    p = pop / pop.sum()
    co = np.zeros((8, 8))
    for _ in range(n_draws):
        i, j = rng.choice(8, 2, replace=False, p=p)
        co[i, j] += 1
        co[j, i] += 1
    return co


def test_pmi_separates_real_structure_from_popularity_at_realistic_skew():
    """A popular expert co-fires with everyone without being *related* to anyone.
    PMI must rank a genuine association above that.

    Tested at 2x skew, the documented validity limit. MoE load balancing is an
    explicit training objective holding skew near 1x, so this is the operating
    regime; `test_pmi_degrades_under_extreme_skew` pins what happens outside it.
    """
    co = _simulate_cofiring(skew=2.0)
    baseline = np.abs(pointwise_mutual_information(co)).max()
    assert baseline < 0.5, f"base-rate leakage too high: {baseline:.3f}"

    planted = co.copy()
    planted[5, 6] = planted[6, 5] = co[5, 6] * 5
    pmi = pointwise_mutual_information(planted)
    assert pmi[5, 6] > 3 * baseline, "planted association not separable from noise"
    assert pmi[5, 6] > pmi[0, 1], "popularity outranked a real association"


def test_pmi_degrades_under_extreme_skew_and_warns():
    """Honest limit: at 10x skew, base-rate leakage (~0.8) approaches genuine
    signal (~0.97) and communities become unreliable. The code must warn rather
    than silently return numbers that look fine."""
    co = _simulate_cofiring(skew=10.0)
    with pytest.warns(RuntimeWarning, match="skew"):
        leakage = np.abs(pointwise_mutual_information(co)).max()
    assert leakage > 0.5, "expected measurable leakage at 10x skew"


def test_usage_skew_matches_construction():
    from expertatlas.coactivation import usage_skew
    assert usage_skew(_simulate_cofiring(skew=1.0)) == pytest.approx(1.0, abs=0.15)
    assert usage_skew(_simulate_cofiring(skew=4.0)) > 2.0


def test_planted_communities_beat_degree_preserving_null():
    """H4. Modularity is high for almost any graph, so the observed value is
    only meaningful against a degree-preserving rewired null."""
    size = 12
    w = np.zeros((size, size))
    for block in (range(0, 6), range(6, 12)):
        for i in block:
            for j in block:
                if i != j:
                    w[i, j] = 10.0
    w[0, 6] = w[6, 0] = 1.0  # single weak bridge

    labels, mod = detect_communities(w, seed=0)
    assert len(set(labels)) == 2
    null_mean, null_std = degree_preserving_null(w, n_trials=20, seed=0)
    assert mod > null_mean + 2 * max(null_std, 1e-6)
