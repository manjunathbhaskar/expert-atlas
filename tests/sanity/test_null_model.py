"""PLAN.md §6.1 — null model sanity tests.

These calibrate the whole statistical pipeline (PLAN.md: "If this fails,
every downstream number is wrong"). Run on pure noise: no planted
structure at all, generated straight from the null model itself.
"""

from __future__ import annotations

import numpy as np

from expertatlas.stats import bh_fdr, chi2_pvalues, compute_lift, shuffle_labels

N_EXPERTS = 64
N_DOMAINS = 6
# Large enough that multinomial sampling noise in the null doesn't itself
# read as spurious lift -- mean|lift| under pure noise scales ~1/sqrt(N),
# so this needs to be big enough for the test to distinguish "no signal"
# from "not enough data to tell". See test_shuffled_labels_give_zero_lift.
N_TOKENS_PER_DOMAIN = 20_000


def _null_counts(seed: int) -> np.ndarray:
    """Counts with NO real expert<->domain association: every domain's
    tokens are drawn from the same uniform expert distribution."""
    rng = np.random.default_rng(seed)
    p_expert = np.full(N_EXPERTS, 1.0 / N_EXPERTS)
    counts = np.zeros((N_EXPERTS, N_DOMAINS))
    for d in range(N_DOMAINS):
        counts[:, d] = rng.multinomial(N_TOKENS_PER_DOMAIN, p_expert)
    return counts


def test_shuffled_labels_give_zero_lift():
    counts = _null_counts(seed=1)
    lift = compute_lift(counts)
    mean_abs_lift = np.abs(lift).mean()
    assert mean_abs_lift < 0.08, f"mean |lift| under pure null too high: {mean_abs_lift:.4f}"


def test_shuffle_labels_preserves_domain_totals():
    counts = _null_counts(seed=2)
    rng = np.random.default_rng(99)
    shuffled = shuffle_labels(counts, rng)
    assert np.allclose(shuffled.sum(axis=0), counts.sum(axis=0))


def test_fdr_controls_false_positives():
    """On pure noise, BH-FDR at q=0.05 should flag roughly <=5% of cells,
    not the ~5% raw false-positive rate you'd get from uncorrected p<0.05
    across thousands of tests."""
    counts = _null_counts(seed=3)
    pvalues = chi2_pvalues(counts)

    uncorrected_rate = float((pvalues < 0.05).mean())
    significant = bh_fdr(pvalues, q=0.05)
    fdr_rate = float(significant.mean())

    # BH-FDR must not flag MORE than the uncorrected rate would on pure
    # noise, and in practice should flag close to zero once corrected.
    assert fdr_rate <= uncorrected_rate + 1e-9
    assert fdr_rate <= 0.05, f"BH-FDR flagged {fdr_rate:.1%} of cells on pure noise"


def test_lift_and_pvalues_shapes_match():
    counts = _null_counts(seed=4)
    lift = compute_lift(counts)
    pvalues = chi2_pvalues(counts)
    assert lift.shape == counts.shape
    assert pvalues.shape == counts.shape
