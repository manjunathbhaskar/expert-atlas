"""Unit tests for expertatlas/context_metrics.py."""

from __future__ import annotations

import numpy as np
import pytest

from expertatlas.context_metrics import (
    PMI_SKEW_LIMIT,
    adjusted_rand_index,
    ascii_overlay,
    community_structure,
    length_trend,
    router_entropy_bits,
    set_hit_rate,
    token_span_from_chars,
)


class TestRouterEntropy:
    def test_uniform_router_hits_the_ceiling(self):
        """64 equal logits -> exactly log2(64) = 6 bits, the max indecision."""
        h = router_entropy_bits(np.zeros((5, 64)))
        assert np.allclose(h, 6.0)

    def test_confident_router_is_near_zero(self):
        logits = np.full((3, 64), -50.0)
        logits[:, 7] = 50.0
        assert np.all(router_entropy_bits(logits) < 1e-6)

    def test_entropy_is_over_all_experts_not_just_topk(self):
        """A router spreading mass over 32 experts must read as MORE indecisive
        than one concentrating on 8, even though both select 8. Measuring
        entropy over the selected top-k only would collapse this distinction."""
        broad = np.full((1, 64), -20.0)
        broad[0, :32] = 0.0
        narrow = np.full((1, 64), -20.0)
        narrow[0, :8] = 0.0
        assert router_entropy_bits(broad)[0] > router_entropy_bits(narrow)[0]

    def test_bounded(self):
        rng = np.random.default_rng(0)
        h = router_entropy_bits(rng.normal(size=(50, 64)) * 5)
        assert np.all(h >= 0) and np.all(h <= 6.0 + 1e-9)

    def test_shift_invariant(self):
        """Softmax is shift-invariant; entropy must be too (guards the
        max-subtraction in the implementation)."""
        rng = np.random.default_rng(1)
        x = rng.normal(size=(4, 64))
        assert np.allclose(router_entropy_bits(x), router_entropy_bits(x + 1000.0))


class TestTokenSpans:
    def test_maps_char_span_to_token_span(self):
        offsets = [(0, 5), (5, 11), (11, 12), (12, 16), (16, 20)]
        assert token_span_from_chars(offsets, (5, 12)) == (1, 3)

    def test_partial_overlap_is_included(self):
        offsets = [(0, 5), (5, 11), (11, 20)]
        assert token_span_from_chars(offsets, (8, 13)) == (1, 3)

    def test_empty_span_raises_rather_than_silently_nan(self):
        with pytest.raises(ValueError):
            token_span_from_chars([(0, 5), (5, 10)], (20, 25))


class TestSetHitRate:
    def test_rate_is_in_unit_interval_and_size_independent(self):
        rng = np.random.default_rng(2)
        mask = np.zeros(1024, dtype=bool)
        mask[:256] = True  # exactly a quarter
        small = rng.integers(0, 1024, size=(50, 8))
        big = rng.integers(0, 1024, size=(5000, 8))
        assert 0.0 <= set_hit_rate(small, mask) <= 1.0
        assert abs(set_hit_rate(big, mask) - 0.25) < 0.02

    def test_empty_returns_nan_not_zero(self):
        """0.0 would silently read as 'no affinity'; NaN propagates honestly."""
        assert np.isnan(set_hit_rate(np.empty((0, 8), dtype=int),
                                     np.zeros(1024, dtype=bool)))


class TestAdjustedRandIndex:
    def test_identical_partitions(self):
        a = np.array([0, 0, 1, 1, 2, 2])
        assert adjusted_rand_index(a, a) == pytest.approx(1.0)

    def test_relabelled_partition_is_still_identical(self):
        a = np.array([0, 0, 1, 1, 2, 2])
        b = np.array([5, 5, 9, 9, 1, 1])
        assert adjusted_rand_index(a, b) == pytest.approx(1.0)

    def test_random_partitions_are_near_zero(self):
        rng = np.random.default_rng(4)
        vals = [adjusted_rand_index(rng.integers(0, 5, 400), rng.integers(0, 5, 400))
                for _ in range(20)]
        assert abs(np.mean(vals)) < 0.05


class TestCommunityStability:
    def _co(self, skew: float, n: int = 64, seed: int = 0):
        rng = np.random.default_rng(seed)
        deg = np.linspace(1.0, skew, n)
        m = np.outer(deg, deg) * 100.0 + rng.random((n, n))
        m = (m + m.T) / 2.0
        np.fill_diagonal(m, 0.0)
        return m

    def test_flags_unreliable_above_the_documented_skew_limit(self):
        s, _ = community_structure(self._co(skew=50.0), bucket=1024)
        assert s.usage_skew > PMI_SKEW_LIMIT
        assert s.reliable is False, (
            "high-skew co-activation must be flagged UNRELIABLE — docs/FINDINGS.md "
            "made exactly this call for H4 at 227x skew"
        )

    def test_balanced_usage_is_reliable(self):
        s, _ = community_structure(self._co(skew=1.05), bucket=128)
        assert s.usage_skew <= PMI_SKEW_LIMIT
        assert s.reliable is True

    def test_ari_against_reference_is_one_for_identical_input(self):
        co = self._co(skew=1.02, seed=3)
        s1, labels = community_structure(co, bucket=128)
        s2, _ = community_structure(co, bucket=256, reference_labels=labels)
        assert s2.ari_vs_shortest == pytest.approx(1.0)


class TestLengthTrend:
    def test_perfect_monotone_trend_is_detected(self):
        lengths = np.repeat([128, 256, 512, 1024, 2048, 3072, 3840], 8).astype(float)
        vals = -np.log2(lengths) + np.random.default_rng(0).normal(0, 0.01, lengths.size)
        t = length_trend("m", lengths, vals, buckets=lengths.astype(int),
                         n_permutations=300)
        assert t.spearman_rho < -0.9
        assert t.perm_p < 0.01
        assert abs(t.cohens_d) > 5

    def test_degenerate_input_does_not_crash(self):
        lengths = np.array([128.0, 256.0, 512.0, 1024.0])
        t = length_trend("const", lengths, np.ones(4), buckets=lengths.astype(int),
                         n_permutations=50)
        assert t.perm_p == 1.0
        assert t.verdict in ("PENDING-FDR", "FLAT")

    def test_verdict_requires_all_three_conditions(self):
        lengths = np.repeat([128, 3840], 30).astype(float)
        rng = np.random.default_rng(9)
        vals = np.where(lengths > 1000, 5.0, 0.0) + rng.normal(0, 0.1, lengths.size)
        t = length_trend("big", lengths, vals, buckets=lengths.astype(int),
                         n_permutations=300)
        assert t.passes_effect_size and t.passes_monotonicity
        assert t.fdr_significant is None and t.verdict == "PENDING-FDR", (
            "a trend must not claim a verdict before FDR is applied across the family"
        )


class TestAsciiOverlay:
    def test_renders_and_labels_every_series(self):
        buckets = [128, 256, 512, 1024, 2048, 3072, 3840]
        out = ascii_overlay(buckets, {
            "accuracy": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            "entropy": [5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7],
        })
        assert "accuracy" in out and "entropy" in out
        assert "128" in out and "3840" in out

    def test_tolerates_nan_and_flat_series(self):
        out = ascii_overlay([128, 3840], {
            "flat": [1.0, 1.0],
            "missing": [float("nan"), float("nan")],
        })
        assert "no data" in out
