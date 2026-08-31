"""Expert co-activation graph and community detection (WS-C).

The complementary axis to topic affinity: affinity asks *what* an expert
responds to, co-activation asks *which experts fire together*. Topic affinity x
co-activation communities is what PLAN.md §2 calls functional regions.

The null model matters more than the method here. Modularity is high for almost
any graph, including random ones, so an uncorrected Louvain result will always
look like it found structure. We compare against a **degree-preserving** rewired
null — an Erdos-Renyi null would inflate modularity and manufacture communities
that are not there.
"""

from __future__ import annotations

import numpy as np

def coactivation_matrix(
    rows: list[dict],
    n_layers: int,
    n_experts: int,
    within_layer_only: bool = True,
) -> np.ndarray:
    """Symmetric co-firing counts over the global expert index (layer*n+idx).

    Args:
        within_layer_only: if True, only count pairs selected for the same token
            in the same layer. Cross-layer co-activation is dominated by trivial
            sequential structure (every layer fires for every token), so the
            within-layer signal is the informative one.
    """
    size = n_layers * n_experts
    m = np.zeros((size, size), dtype=np.float64)

    if within_layer_only:
        for r in rows:
            base = int(r["layer"]) * n_experts
            ids = [base + int(e) for e in r["expert_ids"]]
            for i, a in enumerate(ids):
                for b in ids[i + 1:]:
                    m[a, b] += 1.0
                    m[b, a] += 1.0
    else:
        by_token: dict[tuple, list[int]] = {}
        for r in rows:
            key = (r["prompt_id"], r["token_pos"])
            base = int(r["layer"]) * n_experts
            by_token.setdefault(key, []).extend(base + int(e) for e in r["expert_ids"])
        for ids in by_token.values():
            for i, a in enumerate(ids):
                for b in ids[i + 1:]:
                    m[a, b] += 1.0
                    m[b, a] += 1.0
    return m


def usage_skew(co: np.ndarray) -> float:
    """max/min expert co-firing degree. 1.0 = perfectly balanced.

    Validity gate for `pointwise_mutual_information` — see its docstring.
    MoE load balancing is designed to hold this near 1, so the assumption is
    usually safe, but it must be checked rather than assumed.
    """
    d = np.asarray(co, dtype=np.float64).sum(axis=1)
    d = d[d > 0]
    return float(d.max() / d.min()) if d.size else 1.0


PMI_SKEW_LIMIT = 2.0


def pointwise_mutual_information(co: np.ndarray, warn: bool = True) -> np.ndarray:
    """PMI-weighted edges.

    Raw co-firing counts are dominated by base rate: two frequently-used experts
    co-fire often without being related. PMI divides that out, and is the
    co-activation analogue of using lift instead of heat for affinity.

    Measured limitation (tests/ws_c/test_aggregate.py characterises this):
    because an expert can never co-fire with *itself*, PMI does not fully remove
    base rate when usage is heavily skewed. Simulated, structureless data gives::

        skew  1.0 -> max|PMI| 0.23      skew  4.0 -> 0.54
        skew  2.0 -> max|PMI| 0.36      skew 10.0 -> 0.80

    against a planted association of ~1.42. So separation is clear up to about
    2x skew and collapses by 10x. MoE load balancing is an explicit training
    objective holding skew near 1, which is why PMI is appropriate here — but
    the assumption is checked, not assumed.
    """
    import warnings
    co = np.asarray(co, dtype=np.float64)
    total = co.sum()
    if total == 0:
        return np.zeros_like(co)

    skew = usage_skew(co)
    if warn and skew > PMI_SKEW_LIMIT:
        warnings.warn(
            f"expert usage skew {skew:.1f}x exceeds {PMI_SKEW_LIMIT}x — PMI retains "
            "base-rate contamination in this regime and communities may be artefacts. "
            "Check load balancing before trusting co-activation results.",
            RuntimeWarning, stacklevel=2,
        )
    p_joint = co / total
    p_marg = co.sum(axis=1) / total
    denom = np.outer(p_marg, p_marg)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.where((p_joint > 0) & (denom > 0), np.log2(p_joint / denom), 0.0)
    return np.clip(pmi, 0.0, None)  # positive PMI only


def detect_communities(weights: np.ndarray, seed: int = 0) -> tuple[np.ndarray, float]:
    """Louvain communities. Returns (labels, modularity)."""
    import networkx as nx

    w = np.asarray(weights, dtype=np.float64)
    g = nx.from_numpy_array(np.triu(w, k=1))
    g.remove_edges_from([(u, v) for u, v, d in g.edges(data=True) if d["weight"] <= 0])

    try:
        communities = nx.community.louvain_communities(g, seed=seed, weight="weight")
    except Exception:
        return np.zeros(w.shape[0], dtype=int), 0.0

    labels = np.zeros(w.shape[0], dtype=int)
    for cid, members in enumerate(communities):
        for node in members:
            labels[node] = cid
    mod = nx.community.modularity(g, communities, weight="weight") if g.number_of_edges() else 0.0
    return labels, float(mod)


def degree_preserving_null(
    weights: np.ndarray, n_trials: int = 50, seed: int = 0
) -> tuple[float, float]:
    """Modularity of degree-preserving rewired graphs — (mean, std).

    H4 (PLAN.md §1) is only supported if observed modularity exceeds this by a
    clear margin. Reporting raw modularity alone would be meaningless.
    """
    import networkx as nx

    w = np.asarray(weights, dtype=np.float64)
    g = nx.from_numpy_array(np.triu(w, k=1))
    g.remove_edges_from([(u, v) for u, v, d in g.edges(data=True) if d["weight"] <= 0])
    if g.number_of_edges() < 4:
        return 0.0, 0.0

    mods = []
    for t in range(n_trials):
        h = g.copy()
        try:
            nx.double_edge_swap(h, nswap=max(1, h.number_of_edges()),
                                max_tries=h.number_of_edges() * 20, seed=seed + t)
        except (nx.NetworkXError, nx.NetworkXAlgorithmError):
            continue
        comms = nx.community.louvain_communities(h, seed=seed + t, weight="weight")
        mods.append(nx.community.modularity(h, comms, weight="weight"))

    if not mods:
        return 0.0, 0.0
    return float(np.mean(mods)), float(np.std(mods))
