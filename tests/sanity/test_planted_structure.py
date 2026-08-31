"""PLAN.md §6.2 — the single most valuable test in the project.

Plant a real, known specialist in synthetic count data and verify the
stats pipeline (compute_lift + chi2 + BH-FDR) recovers it, with
approximately correct effect size, and doesn't manufacture a pile of
other false positives out of the noise around it.

This must pass before any result from the real model is trusted.
"""

from __future__ import annotations

import math

import numpy as np

from expertatlas.stats import bh_fdr, chi2_pvalues, compute_lift

N_EXPERTS = 64
# Deliberately closer to the real project's scale (240 factorial cells,
# PLAN.md §7) than an arbitrary small number. With few domains, the
# marginal P(expert) used in the lift denominator is itself contaminated
# by the boosted domain's contribution (it's a non-trivial share of that
# expert's total activity), which attenuates the observed lift below the
# true fold-change -- a real property of the lift formula, not a bug. At
# N_DOMAINS=6 that attenuation is large enough to fail a 0.3 tolerance
# around log2(5); at this scale it's small, matching production conditions.
N_DOMAINS = 150
N_TOKENS_PER_DOMAIN = 2500
PLANTED_EXPERT = 17
PLANTED_DOMAIN = 0
PLANTED_FOLD = 5.0  # expert fires 5x base rate on the planted domain


def _planted_counts(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base_p = np.full(N_EXPERTS, 1.0 / N_EXPERTS)

    counts = np.zeros((N_EXPERTS, N_DOMAINS))
    for d in range(N_DOMAINS):
        p = base_p.copy()
        if d == PLANTED_DOMAIN:
            p[PLANTED_EXPERT] *= PLANTED_FOLD
            p /= p.sum()
        counts[:, d] = rng.multinomial(N_TOKENS_PER_DOMAIN, p)
    return counts


def test_recovers_planted_specialist():
    counts = _planted_counts(seed=123)
    lift = compute_lift(counts)
    pvalues = chi2_pvalues(counts)
    significant = bh_fdr(pvalues, q=0.05)

    expected_lift = math.log2(PLANTED_FOLD)
    observed_lift = lift[PLANTED_EXPERT, PLANTED_DOMAIN]

    assert significant[PLANTED_EXPERT, PLANTED_DOMAIN], (
        "planted specialist was NOT flagged significant after FDR correction "
        "-- the statistical pipeline is not sensitive enough to trust on real data"
    )
    assert abs(observed_lift - expected_lift) < 0.3, (
        f"planted lift {observed_lift:.3f} too far from expected {expected_lift:.3f}"
    )

    n_other_significant = int(significant.sum()) - 1  # minus the planted cell itself
    max_expected_false_positives = math.ceil(0.05 * N_EXPERTS * N_DOMAINS)
    assert n_other_significant <= max_expected_false_positives, (
        f"{n_other_significant} other cells flagged significant on data with only "
        f"one planted effect -- exceeds the {max_expected_false_positives} expected "
        "under q=0.05 FDR control, suggests the correction isn't working"
    )


def test_planted_specialist_is_the_max_lift_expert_for_its_domain():
    counts = _planted_counts(seed=124)
    lift = compute_lift(counts)
    top_expert_for_domain = int(np.argmax(lift[:, PLANTED_DOMAIN]))
    assert top_expert_for_domain == PLANTED_EXPERT
