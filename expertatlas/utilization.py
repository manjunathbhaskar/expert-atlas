"""Expert utilization / routing-collapse diagnostics (Workstream 3).

The standard MoE routing-collapse diagnostic: with `n_experts` and top-`k`
routing, a perfectly balanced router gives every expert an expected share of
`k / n_experts`. Real routers deviate, and the *shape* of that deviation is a
known failure mode — a few "hot" experts absorb most of the load while "cold"
experts idle.

Why this matters for the two other workstreams
----------------------------------------------
`docs/FINDINGS.md` established that 557/1024 experts carry a real (>=2x lift),
FDR-significant topic affinity. If those specialised experts are *also* the
hot ones, then specialisation concentrates load onto a small load-bearing
subset — which is a direct, cheap explanation for why feeding more
domains/context into the model raises collapse risk: the extra load keeps
landing on the same few.

That is a testable cross-reference, not an assumption, and this module exists
to test it rather than assert it.

Important measurement note
--------------------------
`docs/FINDINGS.md` measured `usage_skew = 227x` on the 480-prompt run and
correctly refused to trust PMI-based communities as a result. Skew that large
is plausibly genuine at inference time (load balancing is a *training*-time
objective over the full training distribution; nothing enforces balance on a
narrow evaluation sample). This module therefore reports the skew alongside
every statistic so downstream consumers can gate on it, exactly as
`coactivation.py` does.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class UtilizationStats:
    """Per-expert load statistics over a set of routing observations."""
    counts: np.ndarray          # (n_layers*n_experts,) raw selection counts
    share: np.ndarray           # counts / total selections
    expected_share: float       # k / n_experts, the balanced-router baseline
    load_ratio: np.ndarray      # share / expected_share; 1.0 == perfectly balanced
    variance: float             # variance of share (routing-collapse diagnostic)
    coefficient_of_variation: float
    gini: float
    skew: float                 # max/min load_ratio over non-dead experts
    n_hot: int
    n_cold: int
    n_dead: int
    uids: list[str]

    def to_json(self) -> dict:
        d = asdict(self)
        for key in ("counts", "share", "load_ratio"):
            d[key] = [float(v) for v in d[key]]
        return d


def gini_coefficient(x: np.ndarray) -> float:
    """Gini of a non-negative load vector. 0 = perfectly even, 1 = one expert takes all."""
    v = np.sort(np.asarray(x, dtype=np.float64).ravel())
    n = v.size
    total = v.sum()
    if n == 0 or total <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * v).sum()) / (n * total) - (n + 1.0) / n)


def compute_utilization(
    counts: np.ndarray,
    n_experts_per_layer: int,
    top_k: int,
    uids: list[str] | None = None,
    hot_threshold: float = 2.0,
    cold_threshold: float = 0.5,
) -> UtilizationStats:
    """Load statistics from raw per-expert selection counts.

    Args:
        counts: (n_layers*n_experts,) total selections per expert.
        hot_threshold: load_ratio above which an expert counts as "hot"
            (default 2.0 = carries twice its fair share). This is the
            capacity-factor convention used in MoE serving.
        cold_threshold: load_ratio below which an expert counts as "cold".
    """
    c = np.asarray(counts, dtype=np.float64).ravel()
    total = c.sum()
    if total <= 0:
        raise ValueError("no routing observations — counts sum to zero")

    share = c / total
    # Shares are taken over ALL layers' experts at once. Every layer routes every
    # token to exactly top_k experts, so total selections = n_tokens*n_layers*top_k
    # and a balanced router gives each of the n_layers*n_experts experts a share of
    # 1/(n_layers*n_experts). top_k cancels out — it does not enter the baseline.
    expected_share = 1.0 / c.size
    load_ratio = share / expected_share

    alive = load_ratio[c > 0]
    skew = float(alive.max() / alive.min()) if alive.size else 1.0

    return UtilizationStats(
        counts=c,
        share=share,
        expected_share=float(expected_share),
        load_ratio=load_ratio,
        variance=float(share.var()),
        coefficient_of_variation=float(share.std() / share.mean()) if share.mean() else 0.0,
        gini=gini_coefficient(c),
        skew=skew,
        n_hot=int((load_ratio >= hot_threshold).sum()),
        n_cold=int(((load_ratio <= cold_threshold) & (c > 0)).sum()),
        n_dead=int((c == 0).sum()),
        uids=list(uids) if uids is not None else [f"E{i:04d}" for i in range(c.size)],
    )


def hot_specialist_overlap(
    stats: UtilizationStats,
    specialist_uids: set[str],
    hot_threshold: float = 2.0,
    n_permutations: int = 10_000,
    seed: int = 0,
) -> dict:
    """Are lift-significant specialists over-represented among hot experts?

    This is the load-bearing question of Workstream 3. It is tested against a
    permutation null (random expert sets of the same size), not asserted from a
    raw overlap count — a large overlap is expected by chance alone when 54% of
    experts are specialists, and reporting the raw count would be exactly the
    "significance without a null" error `docs/TRANSFER.md` §11 documents.

    Returns observed overlap, null mean/std, enrichment ratio, and an empirical
    two-sided p-value.
    """
    rng = np.random.default_rng(seed)
    uids = np.array(stats.uids)
    is_hot = stats.load_ratio >= hot_threshold
    hot_uids = set(uids[is_hot].tolist())

    n_hot = len(hot_uids)
    n_spec = len(specialist_uids & set(uids.tolist()))
    observed = len(hot_uids & specialist_uids)

    if n_hot == 0 or n_spec == 0:
        return {
            "observed_overlap": observed, "n_hot": n_hot, "n_specialists": n_spec,
            "null_mean": 0.0, "null_std": 0.0, "enrichment": float("nan"),
            "p_value": 1.0, "verdict": "untestable (no hot experts or no specialists)",
        }

    # Null: draw n_spec experts uniformly at random, count overlap with the hot set.
    idx = np.arange(uids.size)
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        draw = set(uids[rng.choice(idx, size=n_spec, replace=False)].tolist())
        null[i] = len(draw & hot_uids)

    null_mean, null_std = float(null.mean()), float(null.std())
    p = float((np.abs(null - null_mean) >= abs(observed - null_mean)).mean())
    enrichment = observed / null_mean if null_mean > 0 else float("nan")

    if p >= 0.05:
        verdict = "no enrichment — specialists are not disproportionately hot"
    elif enrichment > 1.0:
        verdict = "ENRICHED — specialisation concentrates load onto hot experts"
    else:
        verdict = "DEPLETED — specialists are disproportionately cold"

    return {
        "observed_overlap": int(observed),
        "n_hot": int(n_hot),
        "n_specialists": int(n_spec),
        "null_mean": null_mean,
        "null_std": null_std,
        "enrichment": float(enrichment),
        "p_value": p,
        "n_permutations": n_permutations,
        "verdict": verdict,
    }
