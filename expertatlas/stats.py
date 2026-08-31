"""Lift, significance, and FDR correction (PLAN.md §5 WS-C, minimally
pulled into Phase 0 because the §6.1 null-model sanity tests need it as
the Phase 0 gate). Full co-activation / split-half / effect-size work is
still Phase 2 scope — this module covers exactly what Checkpoint 0 needs
and no more.

Everything here operates on plain count matrices (n_experts x n_domains)
rather than raw traces, so it's model-agnostic and fast to test.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests


def compute_lift(
    expert_domain_counts: np.ndarray,  # (n_experts, n_domains)
    laplace: float = 1.0,
) -> np.ndarray:
    """log2( P(expert | domain) / P(expert) ), Laplace-smoothed.

    Positive lift = expert fires more on this domain than its base rate;
    0 = exactly base rate; negative = fires less. This is the primary
    metric of the whole project (PLAN.md §0) — raw counts/heat are not.
    """
    counts = np.asarray(expert_domain_counts, dtype=np.float64)
    n_experts, n_domains = counts.shape

    domain_totals = counts.sum(axis=0)  # (n_domains,)
    expert_totals = counts.sum(axis=1)  # (n_experts,)
    grand_total = counts.sum()

    p_e_given_d = (counts + laplace) / (domain_totals[None, :] + laplace * n_experts)
    p_e = (expert_totals + laplace) / (grand_total + laplace * n_experts)

    lift = np.log2(p_e_given_d / p_e[:, None])
    return lift


def chi2_pvalues(expert_domain_counts: np.ndarray) -> np.ndarray:
    """Per (expert, domain) chi-squared test of independence against a
    2x2 contingency table: {this expert, other experts} x {this domain,
    other domains}. Returns a (n_experts, n_domains) p-value matrix.
    """
    counts = np.asarray(expert_domain_counts, dtype=np.float64)
    n_experts, n_domains = counts.shape
    domain_totals = counts.sum(axis=0)
    expert_totals = counts.sum(axis=1)
    grand_total = counts.sum()

    pvalues = np.ones((n_experts, n_domains))
    for e in range(n_experts):
        for d in range(n_domains):
            a = counts[e, d]
            b = expert_totals[e] - a
            c = domain_totals[d] - a
            dd = grand_total - a - b - c
            table = np.array([[a, b], [c, dd]])
            if table.min() < 0 or grand_total == 0:
                pvalues[e, d] = 1.0
                continue
            try:
                _, p, _, _ = chi2_contingency(table, correction=True)
            except ValueError:
                # degenerate table (a zero row/col) -> not testable, not significant
                p = 1.0
            pvalues[e, d] = p
    return pvalues


def bh_fdr(pvalues: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction, flattened across the whole
    matrix (the correct unit of multiple comparison — PLAN.md §0: 1024
    experts x D domains is tens of thousands of simultaneous tests).
    Returns a boolean mask, same shape as pvalues, True = significant.
    """
    flat = np.asarray(pvalues).flatten()
    reject, _, _, _ = multipletests(flat, alpha=q, method="fdr_bh")
    return reject.reshape(pvalues.shape)


def shuffle_labels(
    expert_domain_counts: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Null model: redistribute each domain's total count across experts
    according to the GLOBAL expert marginal (i.e. what you'd see if domain
    labels carried no information about routing). Preserves expert_totals'
    shape of randomness while destroying any real expert<->domain association.
    """
    counts = np.asarray(expert_domain_counts, dtype=np.float64)
    n_experts, n_domains = counts.shape
    expert_totals = counts.sum(axis=1)
    p_expert = expert_totals / expert_totals.sum()
    domain_totals = counts.sum(axis=0).astype(int)

    shuffled = np.zeros_like(counts)
    for d in range(n_domains):
        draws = rng.multinomial(domain_totals[d], p_expert)
        shuffled[:, d] = draws
    return shuffled


# ---------------------------------------------------------------------------
# WS-C additions (Phase 2). Everything above is the Phase 0 subset.
# ---------------------------------------------------------------------------


def chi2_pvalues_fast(expert_domain_counts: np.ndarray) -> np.ndarray:
    """Vectorised equivalent of `chi2_pvalues`, with Yates correction.

    The loop version issues n_experts * n_domains scipy calls — 1024 x 240 =
    245,760 at real scale, which takes minutes. This computes the same 2x2
    chi-squared statistic in closed form across the whole matrix at once.
    `test_chi2_fast_matches_loop` asserts they agree.
    """
    from scipy.stats import chi2 as chi2_dist

    counts = np.asarray(expert_domain_counts, dtype=np.float64)
    a = counts
    row = counts.sum(axis=1, keepdims=True)      # expert totals
    col = counts.sum(axis=0, keepdims=True)      # domain totals
    n = counts.sum()
    if n == 0:
        return np.ones_like(counts)

    b = row - a
    c = col - a
    d = n - a - b - c

    # Yates-corrected 2x2 chi-squared.
    num = n * np.clip(np.abs(a * d - b * c) - n / 2.0, 0.0, None) ** 2
    den = row * col * (n - row) * (n - col)

    with np.errstate(divide="ignore", invalid="ignore"):
        stat = np.where(den > 0, num / den, 0.0)
    p = chi2_dist.sf(stat, df=1)
    # Degenerate tables (empty row or column) are not testable.
    return np.where(den > 0, p, 1.0)


def cramers_v(expert_domain_counts: np.ndarray) -> np.ndarray:
    """Effect size per (expert, domain). For a 2x2 table V = sqrt(chi2 / n).

    Reported alongside every p-value: with hundreds of thousands of tokens,
    trivially small associations become 'significant'. Significance without
    effect size is how you publish a map of noise that survived FDR.
    """
    counts = np.asarray(expert_domain_counts, dtype=np.float64)
    a = counts
    row = counts.sum(axis=1, keepdims=True)
    col = counts.sum(axis=0, keepdims=True)
    n = counts.sum()
    if n == 0:
        return np.zeros_like(counts)
    b, c = row - a, col - a
    d = n - a - b - c
    den = row * col * (n - row) * (n - col)
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = np.where(den > 0, n * (a * d - b * c) ** 2 / den, 0.0)
    return np.sqrt(np.clip(stat / n, 0.0, 1.0))


def specialisation_score(lift: np.ndarray) -> np.ndarray:
    """Per-expert specialisation in [0, 1]. 0 = generalist, 1 = single-domain.

    Normalised entropy of the expert's positive-lift profile. Only positive
    lift contributes: an expert that merely *avoids* one domain is not a
    specialist in the remaining ones.
    """
    pos = np.clip(np.asarray(lift, dtype=np.float64), 0.0, None)
    total = pos.sum(axis=1, keepdims=True)
    n_domains = pos.shape[1]
    if n_domains < 2:
        return np.zeros(pos.shape[0])

    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(total > 0, pos / total, 0.0)
        ent = -np.sum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    norm = ent / np.log(n_domains)
    # Flat profile -> entropy 1 -> specialisation 0.
    return np.where(total.ravel() > 0, 1.0 - norm, 0.0)


def split_half_replication(
    lift_a: np.ndarray, lift_b: np.ndarray
) -> tuple[float, np.ndarray]:
    """H6 — the project gate (PLAN.md §1).

    Spearman correlation between lift vectors computed on disjoint prompt
    halves. If affinity does not replicate across held-out prompts we are
    measuring noise, and no downstream result means anything.

    Returns (pooled_rho, per_expert_rho).
    """
    from scipy.stats import spearmanr

    a = np.asarray(lift_a, dtype=np.float64)
    b = np.asarray(lift_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")

    pooled = spearmanr(a.ravel(), b.ravel()).statistic

    per_expert = np.full(a.shape[0], np.nan)
    for i in range(a.shape[0]):
        if np.ptp(a[i]) > 0 and np.ptp(b[i]) > 0:
            per_expert[i] = spearmanr(a[i], b[i]).statistic
    return float(pooled), per_expert


def null_lift_distribution(
    expert_domain_counts: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Permutation null for lift. Every reported lift must be read against this.

    Returns per-cell mean/std under the null plus the two-sided empirical
    p-value, which does not assume the chi-squared approximation holds in
    sparse cells (it often does not for rare experts).
    """
    counts = np.asarray(expert_domain_counts, dtype=np.float64)
    rng = np.random.default_rng(seed)
    observed = compute_lift(counts)

    acc = np.zeros_like(observed)
    acc_sq = np.zeros_like(observed)
    at_least_as_extreme = np.zeros_like(observed)

    for _ in range(n_permutations):
        null_lift = compute_lift(shuffle_labels(counts, rng))
        acc += null_lift
        acc_sq += null_lift ** 2
        at_least_as_extreme += (np.abs(null_lift) >= np.abs(observed)).astype(float)

    mean = acc / n_permutations
    var = np.clip(acc_sq / n_permutations - mean ** 2, 0.0, None)
    return {
        "observed": observed,
        "null_mean": mean,
        "null_std": np.sqrt(var),
        "empirical_p": (at_least_as_extreme + 1.0) / (n_permutations + 1.0),
    }
