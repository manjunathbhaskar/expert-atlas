"""The single most important test in this workstream.

PLAN.md §6.2 calls the planted-structure test "the single most valuable test in
the project" because it proves the instrument works before anything it says
about a real model is trusted. This file is the context-rot analogue.

It plants a **length-blind null router** — one that selects experts uniformly at
random, with routing behaviour that does not depend on input length in any way —
and asserts two things:

  1. the trap metric produces a strong, confident, entirely fake length trend
     on that null router, and
  2. every metric actually used in `docs/CONTEXT_ROT.md` produces no trend on
     the same data.

If (1) ever fails, the trap is not real and the elaborate defence in
`context_metrics.py` is unnecessary. If (2) ever fails, the workstream is
manufacturing its own headline and every number in CONTEXT_ROT.md is void.
"""

from __future__ import annotations

import numpy as np
import pytest

from expertatlas.context_metrics import (
    MIN_COHENS_D,
    apply_fdr,
    distinct_experts_touched,
    expected_distinct_under_null,
    length_trend,
    selection_share,
    set_hit_rate,
)

N_TOTAL = 16 * 64
TOP_K = 8
BUCKETS = [128, 256, 512, 1024, 2048, 3072, 3840]


def null_router_draws(n_tokens: int, rng: np.random.Generator) -> np.ndarray:
    """A router whose behaviour is completely independent of input length.

    Uniform top-k selection per (token, layer). Any metric that trends with
    length on THIS is measuring token count, not routing.
    """
    out = np.empty((n_tokens * 16, TOP_K), dtype=np.int64)
    for layer in range(16):
        base = layer * 64
        for t in range(n_tokens):
            out[layer * n_tokens + t] = base + rng.choice(64, size=TOP_K, replace=False)
    return out


@pytest.fixture(scope="module")
def null_dataset():
    """16 prompts per bucket from the length-blind null router."""
    rng = np.random.default_rng(0)
    rows = []
    for b in BUCKETS:
        for _ in range(16):
            n_tokens = int(b * 0.97)
            draws = null_router_draws(n_tokens, rng)
            rows.append({"bucket": b, "n_tokens": n_tokens, "draws": draws})
    return rows


def test_trap_metric_manufactures_a_fake_trend():
    """distinct_experts_touched trends hard with length on a length-BLIND router.

    This is the artefact the whole module docstring warns about, demonstrated
    rather than asserted. A near-perfect monotone trend here comes from coupon
    collecting alone.

    Deliberately does NOT reuse the shared `null_dataset` fixture (128-3840
    tokens, matching the real experiment's buckets): at those sizes
    distinct_experts_touched saturates at 1024/1024 for every single bucket
    (verified directly -- even the shortest, ~124 tokens, already sees all
    1024 layer-expert pairs), so there is zero variance for a trend to be
    computed on and length_trend correctly returns NaN rather than a fake
    trend. That is length_trend behaving correctly on degenerate input, not
    evidence the trap is harmless -- it just means the trap needs small
    enough n to be visible, which is exactly what a coupon-collector effect
    predicts. This test uses its own much smaller token counts, where the
    unsaturated growth the docstring describes is actually demonstrable.
    """
    rng = np.random.default_rng(0)
    small_buckets = [1, 2, 4, 8, 16, 32, 64]
    rows = [{"bucket": b, "n_tokens": b, "draws": null_router_draws(b, rng)}
            for b in small_buckets for _ in range(16)]

    lengths = np.array([r["n_tokens"] for r in rows], dtype=float)
    buckets = np.array([r["bucket"] for r in rows])
    vals = np.array([distinct_experts_touched(r["draws"]) for r in rows], float)

    t = length_trend("trap", lengths, vals, buckets=buckets, n_permutations=300)
    assert t.spearman_rho > 0.9, (
        "the trap is supposed to produce a strong fake trend; if it does not, "
        "the defence in context_metrics.py may be unnecessary — re-derive it"
    )
    assert t.long_mean > t.short_mean


def test_trap_growth_is_explained_by_the_coupon_collector_null(null_dataset):
    """The fake growth matches the closed-form null, i.e. it is pure arithmetic."""
    for r in null_dataset:
        obs = distinct_experts_touched(r["draws"])
        exp = expected_distinct_under_null(r["draws"].size, N_TOTAL)
        assert abs(obs - exp) / N_TOTAL < 0.05, (
            f"observed {obs} vs coupon-collector expectation {exp:.1f} — if these "
            "diverge, expected_distinct_under_null is wrong and the report's "
            "'it's all arithmetic' claim is unsupported"
        )


def test_rate_metrics_are_flat_on_the_null_router(null_dataset):
    """Every metric CONTEXT_ROT.md actually uses shows no length trend here."""
    rng = np.random.default_rng(7)
    ref = np.zeros(N_TOTAL, dtype=bool)
    ref[rng.choice(N_TOTAL, size=200, replace=False)] = True

    lengths = np.array([r["n_tokens"] for r in null_dataset], dtype=float)
    buckets = np.array([r["bucket"] for r in null_dataset])
    vals = np.array([set_hit_rate(r["draws"], ref) for r in null_dataset])

    t = length_trend("set_hit_rate", lengths, vals, buckets=buckets, n_permutations=1000)
    t = apply_fdr([t])[0]
    assert t.verdict != "TREND", (
        f"a rate metric trended on a length-blind null router (rho={t.spearman_rho}, "
        f"d={t.cohens_d}) — the length normalisation is broken and every routing "
        "number in CONTEXT_ROT.md would be an artefact"
    )


def test_selection_share_is_a_distribution_not_a_count(null_dataset):
    """Shares sum to 1 at every length, so they cannot grow with token count."""
    shares = [selection_share(r["draws"], N_TOTAL) for r in null_dataset]
    for s in shares:
        assert np.isclose(s.sum(), 1.0)
    short = np.mean([s for s, r in zip(shares, null_dataset) if r["bucket"] == BUCKETS[0]], axis=0)
    long = np.mean([s for s, r in zip(shares, null_dataset) if r["bucket"] == BUCKETS[-1]], axis=0)
    # Same expected distribution regardless of length.
    assert np.abs(short - long).max() < 0.002


def test_planted_real_trend_is_recovered(null_dataset):
    """Positive control: a genuine, length-dependent routing change IS detected.

    Without this, `test_rate_metrics_are_flat_on_the_null_router` could pass
    simply because the trend test never fires.
    """
    rng = np.random.default_rng(11)
    ref = np.zeros(N_TOTAL, dtype=bool)
    ref[:200] = True

    lengths, buckets, vals = [], [], []
    for r in null_dataset:
        # Plant: at long context the router genuinely drifts off the reference
        # set. Implemented by re-drawing a fraction of draws away from it.
        # clamp at 0: the null_dataset fixture shrinks n_tokens to 0.97x the
        # nominal bucket, so the shortest bucket's n_tokens falls slightly BELOW
        # BUCKETS[0] itself, giving a negative log-ratio for that one row --
        # a numerical edge artifact, not an intended negative drift.
        frac_off = max(0.0, 0.30 * np.log2(r["n_tokens"] / BUCKETS[0]) / np.log2(BUCKETS[-1] / BUCKETS[0]))
        d = r["draws"].copy()
        n = d.size
        k = int(frac_off * n)
        if k:
            flat = d.ravel()
            pick = rng.choice(n, size=k, replace=False)
            flat[pick] = rng.integers(200, N_TOTAL, size=k)
            d = flat.reshape(r["draws"].shape)
        lengths.append(r["n_tokens"])
        buckets.append(r["bucket"])
        vals.append(set_hit_rate(d, ref))

    t = length_trend("planted", np.array(lengths, float), np.array(vals),
                     buckets=np.array(buckets), n_permutations=1000)
    t = apply_fdr([t])[0]
    assert t.verdict == "TREND", f"planted real trend not recovered: {t.to_dict()}"
    assert t.delta < 0


def test_significant_but_trivial_is_not_reported_as_a_trend():
    """The exact failure mode docs/FINDINGS.md records: significance without
    effect size. A tiny effect measured on many samples must NOT be a TREND."""
    rng = np.random.default_rng(3)
    n_per = 400
    lengths, buckets, vals = [], [], []
    for b in BUCKETS:
        for _ in range(n_per):
            lengths.append(b)
            buckets.append(b)
            # A real but negligible slope, swamped by noise: large n makes it
            # statistically detectable while remaining practically meaningless.
            vals.append(0.5 + 0.0004 * np.log2(b) + rng.normal(0, 0.05))

    t = length_trend("tiny", np.array(lengths, float), np.array(vals),
                     buckets=np.array(buckets), n_permutations=500)
    t = apply_fdr([t])[0]
    assert abs(t.cohens_d) < MIN_COHENS_D
    assert t.verdict != "TREND", (
        "a trivial effect was promoted to a TREND — this is precisely the "
        "70%-significant / 0.79x-median-lift mistake docs/TRANSFER.md §11 records"
    )


def test_fdr_controls_false_positives_across_the_family():
    """On pure noise, the whole trend family yields ~no FDR-significant trends."""
    rng = np.random.default_rng(5)
    lengths = np.repeat(BUCKETS, 16).astype(float)
    trends = []
    for i in range(40):
        vals = rng.normal(size=lengths.size)
        trends.append(length_trend(f"noise{i}", lengths, vals,
                                   buckets=lengths.astype(int), n_permutations=400,
                                   seed=i))
    trends = apply_fdr(trends, q=0.05)
    n_trend = sum(1 for t in trends if t.verdict == "TREND")
    assert n_trend == 0, f"{n_trend}/40 pure-noise metrics reported as TREND"
