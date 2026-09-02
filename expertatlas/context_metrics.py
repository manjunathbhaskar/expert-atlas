"""Routing metrics as a function of INPUT LENGTH (Workstream 1: context rot).

Everything else in this repo treats length as a nuisance variable and destroys
it: `aggregate.py::subsample_cells` equalises the token budget per factorial
cell precisely so that an expert responding to sequence position cannot
masquerade as a topic specialist (see `probes/README.md`, "Measured confound").

**Here length IS the independent variable, so that control is unavailable.**
That inverts the risk, and this module exists mostly to manage the inversion.

===========================================================================
THE NORMALISATION TRAP  (read this before adding any metric here)
===========================================================================

Almost every natural-sounding routing statistic is *mechanically* monotone in
token count, with no routing change whatsoever:

  * "distinct experts touched" — a 3840-token prompt makes 30x more top-8 draws
    than a 128-token prompt. Even with a frozen, length-independent router, the
    coupon-collector expectation rises toward the 1024-expert ceiling. Plotting
    it against length produces a beautiful, completely fake curve.
  * "total co-activation edge weight", "number of experts above threshold",
    "sum of anything over tokens" — same failure, same cause.
  * raw counts feeding a lift computation — long buckets contribute more
    observations, so the *global* marginal P(expert) that lift divides by is
    itself dominated by the long buckets, and short buckets acquire apparent
    "specialisation" that is pure sample-size asymmetry.

`distinct_experts_touched` below is implemented **deliberately, as a negative
control**. It is the trap, kept in the codebase and plotted in
`docs/CONTEXT_ROT.md` next to its normalised counterpart, so the report can
show what the artefact looks like rather than merely claim to have avoided it.
It is never used as evidence.

---------------------------------------------------------------------------
The two defences actually used, and the normalisation for every metric
---------------------------------------------------------------------------

**Defence 1 — a content-identical measurement window (primary).**
`probes/probe_set_context.py` guarantees that, within a replicate, the trailing
question block and the needle sentence are *byte-identical* across every length
bucket and condition. All primary metrics are computed on those windows only.
The token multiset being measured is therefore literally the same string in
every cell; only the amount of preceding context differs. This is stronger than
any post-hoc normalisation because there is nothing left to normalise: equal
token counts hold by construction, not by subsampling.

**Defence 2 — strictly per-token or per-draw quantities.**
No metric here is a sum over tokens. Each is a mean over tokens, or a rate over
top-k draws, so its expectation is length-invariant under a null router.

Per-metric normalisation, stated explicitly (brief requirement):

| metric | normalisation | length-invariant under a null router? |
|---|---|---|
| `mean_router_entropy` | mean over window tokens x layers of H(full 64-way softmax), bits | yes — a mean, on an identical token multiset |
| `mean_topk_mass` | mean over window tokens x layers of sum of selected gate weights | yes — same |
| `expert_share_matrix` | column-normalised to a probability distribution per bucket; window sizes equal by construction | yes — a distribution, not a count |
| `needle_affinity_rate` | fraction of top-k draws in the window landing in a fixed reference expert set, in [0,1] | yes — a rate over draws |
| `hot_load_share` | fraction of top-k draws landing in the WS-3 hot set, in [0,1] | yes — a rate over draws |
| `coactivation_*` | built from equal-size windows only; PMI is already base-rate divided | yes, but see the skew gate |
| `distinct_experts_touched` | **NONE — this is the trap, a negative control** | **no** |

The lift/specialisation metric additionally re-derives its base rate *within*
the length sweep (`expert_share_matrix` normalises each bucket column before
any comparison), so a bucket contributing more raw draws cannot shift the
denominator other buckets are judged against.

---------------------------------------------------------------------------
Significance discipline
---------------------------------------------------------------------------
`docs/FINDINGS.md` records this project reporting 70% of cells "significant"
with a median lift of 0.79x — significance manufactured by sample size. Every
trend here therefore carries BOTH bars:

  * a permutation null over shuffled length-bucket labels, then Benjamini-
    Hochberg FDR across the whole family of trend tests (`stats.py::bh_fdr`),
  * AND a practical effect-size floor (Cohen's d between the shortest and
    longest bucket), reported alongside the raw delta in the metric's own units.

A trend that clears only one bar is reported as not clearing the bar.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

# Practical-significance floor for a length trend, in Cohen's d between the
# shortest and longest bucket. 0.8 is the conventional "large effect" bound.
# Chosen a priori, not tuned after seeing results.
MIN_COHENS_D = 0.8

# Minimum |Spearman rho| between the metric and log2(length) for the trend to
# be called monotone at all. A trend can be significant and non-monotone
# (e.g. a single outlying bucket), which is not what "context rot" claims.
MIN_TREND_RHO = 0.5

# coactivation.py's own documented PMI validity limit. Copied, not imported,
# only so this module states the number it gates on; the import is used below.
PMI_SKEW_LIMIT = 2.0


# ---------------------------------------------------------------------------
# Per-token router quantities
# ---------------------------------------------------------------------------


def router_entropy_bits(router_logits: np.ndarray) -> np.ndarray:
    """Entropy of the router's FULL n_experts-way softmax, in bits, per token.

    Args:
        router_logits: (n_tokens, n_experts) raw router logits for one layer.

    Returns:
        (n_tokens,) entropy in bits. Range [0, log2(n_experts)]; for OLMoE's
        64 experts the uniform-router ceiling is exactly 6.0 bits.

    This is the decisiveness measure the brief asks for. It is computed over
    all 64 experts, NOT over the selected 8 -- the top-8 distribution is
    conditioned on the selection and would understate indecision by
    construction. `topk_mass` (already recorded by capture.py) is the
    complementary measure: how much probability the router actually committed
    to the experts it chose.
    """
    x = np.asarray(router_logits, dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    ex = np.exp(x)
    p = ex / ex.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.sum(np.where(p > 0, p * np.log2(p), 0.0), axis=-1)
    return h


def token_span_from_chars(offsets: list[tuple[int, int]], char_span) -> tuple[int, int]:
    """Map a character span to a half-open token span via offset mapping.

    Used to locate the byte-identical question / needle windows inside a
    tokenised prompt. Returns (start_token, end_token); end is exclusive.
    Raises if the span is empty -- a silently empty measurement window would
    make every downstream metric NaN in a way that is easy to miss.
    """
    c0, c1 = int(char_span[0]), int(char_span[1])
    idx = [i for i, (a, b) in enumerate(offsets) if b > c0 and a < c1]
    if not idx:
        raise ValueError(f"empty token span for char span {(c0, c1)}")
    return idx[0], idx[-1] + 1


# ---------------------------------------------------------------------------
# THE TRAP -- implemented on purpose, never used as evidence
# ---------------------------------------------------------------------------


def distinct_experts_touched(expert_ids_per_draw: np.ndarray) -> int:
    """NEGATIVE CONTROL. Count of distinct experts selected at least once.

    **This metric is invalid for comparing across length buckets and is only
    ever reported as a demonstration of the artefact it produces.** It grows
    monotonically with token count under a null router that does not respond to
    length at all, because more tokens means more draws (coupon collector).
    See this module's docstring. Do not promote it to a finding.
    """
    return int(np.unique(np.asarray(expert_ids_per_draw).ravel()).size)


def expected_distinct_under_null(n_draws: int, n_experts: int) -> float:
    """Coupon-collector expectation for `distinct_experts_touched` under a
    uniform null router: n * (1 - (1 - 1/n)^draws).

    Quantifies exactly how much of any observed growth is mechanical. If the
    observed curve tracks this, there is no routing effect to report.
    """
    n = float(n_experts)
    return n * (1.0 - (1.0 - 1.0 / n) ** float(n_draws))


# ---------------------------------------------------------------------------
# Rate-based (length-invariant) routing metrics
# ---------------------------------------------------------------------------


def selection_share(expert_ids_per_draw: np.ndarray, n_experts_total: int) -> np.ndarray:
    """Per-expert share of top-k draws, summing to 1.

    The length-normalised replacement for raw selection counts. Because it is a
    distribution over experts rather than a count, its expectation does not move
    with the number of tokens in the window.
    """
    flat = np.asarray(expert_ids_per_draw).ravel().astype(np.int64)
    counts = np.bincount(flat, minlength=n_experts_total).astype(np.float64)
    total = counts.sum()
    return counts / total if total > 0 else counts


def set_hit_rate(expert_ids_per_draw: np.ndarray, member_mask: np.ndarray) -> float:
    """Fraction of top-k draws landing inside a fixed reference expert set.

    In [0, 1] and independent of window size, so it is comparable across
    buckets. Used for:
      * needle-relevant specialisation (reference set = experts with high lift
        for needle tokens measured at the SHORTEST bucket), and
      * hot/cold load concentration (reference set = WS-3's hot experts).
    """
    flat = np.asarray(expert_ids_per_draw).ravel().astype(np.int64)
    if flat.size == 0:
        return float("nan")
    return float(member_mask[flat].mean())


# ---------------------------------------------------------------------------
# Trend testing: both bars
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrendResult:
    """One metric's dependence on input length, with both bars applied."""
    metric: str
    n_prompts: int
    buckets: list[int]
    bucket_means: list[float]
    short_mean: float
    long_mean: float
    delta: float                 # long - short, in the metric's own units
    pct_change: float
    spearman_rho: float          # metric vs log2(prompt length), per prompt
    perm_p: float                # permutation null over shuffled bucket labels
    fdr_significant: bool | None  # filled in by bh_fdr across the family
    cohens_d: float              # shortest vs longest bucket
    passes_effect_size: bool
    passes_monotonicity: bool

    @property
    def verdict(self) -> str:
        if self.fdr_significant is None:
            return "PENDING-FDR"
        both = self.fdr_significant and self.passes_effect_size and self.passes_monotonicity
        if both:
            return "TREND"
        if self.fdr_significant and not self.passes_effect_size:
            return "SIGNIFICANT-BUT-TRIVIAL"
        if self.fdr_significant and not self.passes_monotonicity:
            return "SIGNIFICANT-BUT-NON-MONOTONE"
        return "FLAT"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict
        return d


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    na, nb = a.size, b.size
    sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if sp2 <= 0:
        return 0.0
    return float((b.mean() - a.mean()) / np.sqrt(sp2))


def length_trend(
    metric: str,
    lengths: np.ndarray,
    values: np.ndarray,
    buckets: np.ndarray | None = None,
    n_permutations: int = 2000,
    seed: int = 0,
) -> TrendResult:
    """Test whether `values` trends with input length, both bars applied.

    Args:
        lengths: per-prompt token count (the independent variable).
        values: per-prompt metric value.
        buckets: per-prompt declared bucket label, used only for grouping in
            the reported per-bucket means. Passed explicitly rather than
            inferred from `lengths`, because prompts end on a sentence boundary
            and so land a few tokens short of their nominal bucket.

    The null is a permutation of the length labels across prompts -- the same
    principle as `stats.py::shuffle_labels` (which redistributes counts under
    "labels carry no information"), applied to a continuous per-prompt statistic
    rather than a count matrix. `perm_p` is the two-sided empirical probability
    that a shuffled assignment produces |rho| at least as large.

    A parametric p-value is deliberately not used: prompts within a replicate
    share a haystack stream and are not independent, which would make an
    asymptotic p-value anticonservative. The permutation preserves the observed
    value distribution exactly.
    """
    from scipy.stats import spearmanr

    lengths = np.asarray(lengths, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    buckets = (np.asarray(buckets) if buckets is not None
               else np.asarray(lengths, dtype=np.int64))
    ok = np.isfinite(lengths) & np.isfinite(values)
    lengths, values, buckets = lengths[ok], values[ok], buckets[ok]
    if lengths.size < 4 or np.ptp(values) == 0:
        return TrendResult(metric, int(lengths.size), [], [], float("nan"), float("nan"),
                           float("nan"), float("nan"), float("nan"), 1.0, None,
                           float("nan"), False, False)

    logl = np.log2(lengths)
    rho = float(spearmanr(logl, values).statistic)

    rng = np.random.default_rng(seed)
    idx = np.arange(values.size)
    at_least = 0
    for _ in range(n_permutations):
        rng.shuffle(idx)
        r = spearmanr(logl, values[idx]).statistic
        if np.isfinite(r) and abs(r) >= abs(rho):
            at_least += 1
    perm_p = (at_least + 1.0) / (n_permutations + 1.0)

    groups: dict[int, list[float]] = {}
    for i in range(values.size):
        groups.setdefault(int(buckets[i]), []).append(float(values[i]))
    gkeys = sorted(groups)
    bucket_means = [float(np.mean(groups[k])) for k in gkeys]

    short_vals = np.array(groups[gkeys[0]])
    long_vals = np.array(groups[gkeys[-1]])
    d = _cohens_d(short_vals, long_vals)
    short_mean, long_mean = float(short_vals.mean()), float(long_vals.mean())
    delta = long_mean - short_mean
    pct = (delta / abs(short_mean) * 100.0) if short_mean != 0 else float("nan")

    return TrendResult(
        metric=metric,
        n_prompts=int(values.size),
        buckets=[int(k) for k in gkeys],
        bucket_means=bucket_means,
        short_mean=short_mean,
        long_mean=long_mean,
        delta=float(delta),
        pct_change=float(pct),
        spearman_rho=rho,
        perm_p=float(perm_p),
        fdr_significant=None,
        cohens_d=float(d),
        passes_effect_size=bool(np.isfinite(d) and abs(d) >= MIN_COHENS_D),
        passes_monotonicity=bool(abs(rho) >= MIN_TREND_RHO),
    )


def apply_fdr(trends: list[TrendResult], q: float = 0.05) -> list[TrendResult]:
    """BH-FDR across the whole family of trend tests (`stats.py::bh_fdr`).

    Every trend reported in `docs/CONTEXT_ROT.md` goes through this together --
    correcting each metric in isolation would reintroduce exactly the
    multiple-comparison inflation the project exists to avoid.
    """
    from expertatlas.stats import bh_fdr

    if not trends:
        return []
    p = np.array([t.perm_p for t in trends], dtype=np.float64).reshape(-1, 1)
    mask = bh_fdr(p, q=q).ravel()
    out = []
    for t, m in zip(trends, mask):
        out.append(TrendResult(**{**asdict(t), "fdr_significant": bool(m)}))
    return out


# ---------------------------------------------------------------------------
# Community stability across buckets
# ---------------------------------------------------------------------------


def adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    """ARI between two clusterings. 1.0 = identical, ~0 = chance agreement.

    Implemented here rather than pulled from sklearn so this module's numeric
    core depends only on numpy -- the same reason `stats.py` computes chi2 in
    closed form.
    """
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.size != b.size or a.size == 0:
        return float("nan")
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    cont = np.zeros((ai.max() + 1, bi.max() + 1), dtype=np.float64)
    np.add.at(cont, (ai, bi), 1.0)

    def comb2(x):
        return x * (x - 1.0) / 2.0

    sum_ij = comb2(cont).sum()
    sum_a = comb2(cont.sum(axis=1)).sum()
    sum_b = comb2(cont.sum(axis=0)).sum()
    n = comb2(float(a.size))
    expected = sum_a * sum_b / n if n > 0 else 0.0
    maxi = 0.5 * (sum_a + sum_b)
    denom = maxi - expected
    return float((sum_ij - expected) / denom) if denom != 0 else 1.0


@dataclass(frozen=True)
class CommunityStability:
    """Co-activation community structure at one length bucket.

    `reliable` is False whenever `usage_skew` exceeds coactivation.py's own
    documented 2.0x PMI validity limit. `docs/FINDINGS.md` measured 227x on the
    main run and reported H4 as UNRELIABLE rather than as evidence; the same
    gate is applied here and the same wording is used, against this
    workstream's interest.
    """
    bucket: int
    usage_skew: float
    modularity: float
    n_communities: int
    ari_vs_shortest: float
    reliable: bool

    def to_dict(self) -> dict:
        return asdict(self)


def community_structure(
    co_matrix: np.ndarray,
    bucket: int,
    reference_labels: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[CommunityStability, np.ndarray]:
    """Louvain communities + skew validity gate for one bucket's co-activation.

    The co-activation matrix must be built from an EQUAL-SIZE token window --
    co-firing counts scale with token count, so unequal windows would make
    modularity a length artefact rather than a measurement.
    """
    from expertatlas.coactivation import (
        detect_communities,
        pointwise_mutual_information,
        usage_skew,
    )

    skew = usage_skew(co_matrix)
    pmi = pointwise_mutual_information(co_matrix, warn=False)
    labels, modularity = detect_communities(pmi, seed=seed)
    ari = (adjusted_rand_index(reference_labels, labels)
           if reference_labels is not None else 1.0)
    return CommunityStability(
        bucket=int(bucket),
        usage_skew=float(skew),
        modularity=float(modularity),
        n_communities=int(len(set(int(x) for x in labels))),
        ari_vs_shortest=float(ari),
        reliable=bool(skew <= PMI_SKEW_LIMIT),
    ), labels


# ---------------------------------------------------------------------------
# ASCII figure (matplotlib is not installed in this venv)
# ---------------------------------------------------------------------------


def ascii_overlay(
    buckets: list[int],
    series: dict[str, list[float]],
    height: int = 14,
    width: int = 62,
) -> str:
    """Overlay several length-indexed series on one min-max normalised axis.

    Each series is scaled to its own [0,1] range, so the plot shows SHAPE
    agreement -- whether accuracy, entropy and specialisation move together --
    not magnitudes, which are in incomparable units. Absolute values are always
    printed in the accompanying table; this figure is never the only report of
    a number.
    """
    marks = "*o#+x@"
    keys = list(series)
    grid = [[" "] * width for _ in range(height)]

    xs = np.log2(np.asarray(buckets, dtype=np.float64))
    xspan = xs.max() - xs.min()
    cols = [int(round((x - xs.min()) / xspan * (width - 1))) if xspan > 0 else 0 for x in xs]

    for si, k in enumerate(keys):
        v = np.asarray(series[k], dtype=np.float64)
        finite = v[np.isfinite(v)]
        if finite.size == 0:
            continue
        lo, hi = finite.min(), finite.max()
        norm = (v - lo) / (hi - lo) if hi > lo else np.full_like(v, 0.5)
        for c, y in zip(cols, norm):
            if not np.isfinite(y):
                continue
            r = height - 1 - int(round(y * (height - 1)))
            grid[r][c] = marks[si % len(marks)]

    lines = ["    1.0 |" + "".join(grid[0])]
    for r in range(1, height - 1):
        lines.append("        |" + "".join(grid[r]))
    lines.append("    0.0 |" + "".join(grid[-1]))
    lines.append("        +" + "-" * width)
    labels = [""] * width
    axis = [" "] * width
    for c, b in zip(cols, buckets):
        s = str(b)
        start = min(max(0, c - len(s) // 2), width - len(s))
        for i, ch in enumerate(s):
            axis[start + i] = ch
    lines.append("         " + "".join(axis) + "   (input tokens, log2 axis)")
    lines.append("")
    for si, k in enumerate(keys):
        v = [x for x in series[k] if np.isfinite(x)]
        rng_s = f"[{min(v):.4g} .. {max(v):.4g}]" if v else "[no data]"
        lines.append(f"   {marks[si % len(marks)]}  {k}  {rng_s}")
    return "\n".join(lines)
