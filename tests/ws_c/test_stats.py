"""WS-C tests. The planted-structure and base-rate tests are the ones that matter:
they prove the instrument reads correctly before it is pointed at a real model.
"""

import numpy as np
import pytest

from expertatlas.stats import (
    bh_fdr, chi2_pvalues, chi2_pvalues_fast, compute_lift, cramers_v,
    null_lift_distribution, shuffle_labels, specialisation_score,
    split_half_replication,
)

N_EXPERTS, N_DOMAINS = 40, 5


def uniform_counts(n=N_EXPERTS, d=N_DOMAINS, per_cell=200.0):
    return np.full((n, d), per_cell)


def planted_counts(expert=7, domain=2, factor=5.0, per_cell=200.0):
    """One genuine specialist; everything else exactly at base rate."""
    c = uniform_counts(per_cell=per_cell)
    c[expert, domain] *= factor
    return c


# --- The control that guards against publishing heat as affinity -------------

def test_lift_is_zero_under_uniform_counts():
    lift = compute_lift(uniform_counts())
    assert np.allclose(lift, 0.0, atol=0.02)


def test_lift_is_base_rate_corrected():
    """An expert used 10x more overall, but proportionally across every domain,
    must have lift ~0 everywhere.

    This is the single most important test in the project. Failing it means the
    pipeline is reporting routing *heat* as domain affinity — the exact error
    PLAN.md §0 exists to prevent, and the one that would make a confident,
    completely wrong atlas.
    """
    c = uniform_counts()
    c[3, :] *= 10.0
    lift = compute_lift(c)
    assert np.abs(lift[3]).max() < 0.05, f"heat leaked into lift: {lift[3]}"


def test_planted_specialist_recovered():
    """Planted 5x enrichment is recovered — but NOT as log2(5).

    Multiplying one cell by 5 also inflates that expert's own marginal and its
    domain's total, so part of the enrichment is absorbed into the base rate
    lift divides by. Here the true value is 1.365, not 2.322.

    This is not a defect: it is lift behaving correctly. Anyone "fixing" the
    implementation to return log2(5) would be reintroducing the base-rate
    contamination the metric exists to remove. The expected value is therefore
    derived from the definition rather than hardcoded.
    """
    counts = planted_counts(factor=5.0)
    expected = np.log2(
        (counts[7, 2] / counts[:, 2].sum()) / (counts[7].sum() / counts.sum())
    )
    assert expected == pytest.approx(1.365, abs=0.01)  # pins the analysis, not the code
    assert compute_lift(counts)[7, 2] == pytest.approx(expected, abs=0.05)


def test_planted_specialist_is_the_largest_lift():
    """Ranking is what the atlas actually displays, so it must be right even
    where absolute magnitude is attenuated by the marginal shift above."""
    lift = compute_lift(planted_counts(factor=5.0))
    assert np.unravel_index(np.argmax(lift), lift.shape) == (7, 2)


def test_planted_specialist_is_significant_and_others_are_not():
    counts = planted_counts(factor=5.0)
    sig = bh_fdr(chi2_pvalues_fast(counts), q=0.05)
    assert sig[7, 2], "planted specialist not detected"
    n_false = sig.sum() - sig[7, 2]
    assert n_false <= 0.05 * counts.size, f"too many false positives: {n_false}"


# --- Null-model calibration: if these fail, every number downstream is wrong --

def test_shuffled_labels_give_near_zero_lift():
    rng = np.random.default_rng(0)
    lift = compute_lift(shuffle_labels(planted_counts(), rng))
    assert np.abs(lift).mean() < 0.2


def test_fdr_controls_false_positives_on_pure_noise():
    rng = np.random.default_rng(1)
    counts = rng.multinomial(200, np.full(N_EXPERTS, 1 / N_EXPERTS),
                             size=N_DOMAINS).T.astype(float)
    sig = bh_fdr(chi2_pvalues_fast(counts), q=0.05)
    assert sig.mean() <= 0.05, f"FDR not controlled: {sig.mean():.3f} flagged"


def test_permutation_null_flags_planted_only():
    res = null_lift_distribution(planted_counts(factor=6.0), n_permutations=200, seed=0)
    assert res["empirical_p"][7, 2] < 0.02
    assert np.abs(res["null_mean"]).max() < 0.25


# --- Vectorisation must not change the answer --------------------------------

def test_chi2_fast_matches_loop():
    counts = planted_counts(factor=3.0)
    np.testing.assert_allclose(
        chi2_pvalues_fast(counts), chi2_pvalues(counts), rtol=1e-6, atol=1e-9
    )


# --- Effect size, specialisation, H6 -----------------------------------------

def test_effect_size_small_even_when_significant():
    """Large n makes trivial associations significant. Cramer's V must stay small,
    which is why it is reported next to every p-value."""
    c = uniform_counts(per_cell=1e6)
    c[5, 1] *= 1.01
    v = cramers_v(c)
    assert v[5, 1] < 0.05


def test_specialisation_generalist_vs_specialist():
    flat = np.zeros((2, N_DOMAINS))
    assert specialisation_score(flat)[0] == pytest.approx(0.0, abs=1e-6)

    spec = np.zeros((1, N_DOMAINS))
    spec[0, 2] = 3.0
    assert specialisation_score(spec)[0] > 0.95


def test_split_half_replicates_signal_not_noise():
    """H6 — the project gate. Planted signal must replicate; noise must not."""
    rng = np.random.default_rng(3)
    lift_a = compute_lift(planted_counts(factor=8.0))
    lift_b = compute_lift(planted_counts(factor=8.0))
    rho_signal, _ = split_half_replication(lift_a, lift_b)
    assert rho_signal > 0.5

    noise_a = rng.normal(size=(N_EXPERTS, N_DOMAINS))
    noise_b = rng.normal(size=(N_EXPERTS, N_DOMAINS))
    rho_noise, _ = split_half_replication(noise_a, noise_b)
    assert abs(rho_noise) < 0.2


def test_split_half_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        split_half_replication(np.zeros((4, 3)), np.zeros((4, 2)))
