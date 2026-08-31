"""WS-3 tests: utilization statistics and the hot/specialist enrichment null."""

import json
from pathlib import Path

import numpy as np
import pytest

from expertatlas.utilization import (
    compute_utilization, gini_coefficient, hot_specialist_overlap,
)

ROOT = Path(__file__).resolve().parents[2]
N_LAYERS, N_EXPERTS, TOP_K = 4, 8, 2
SIZE = N_LAYERS * N_EXPERTS


def uids(n=SIZE):
    return [f"L{i // N_EXPERTS:02d}E{i % N_EXPERTS:02d}" for i in range(n)]


# --- baseline correctness ----------------------------------------------------

def test_balanced_router_gives_load_ratio_one():
    """The baseline must be 1/(n_layers*n_experts) — top_k cancels out.
    Getting this wrong would rescale every hot/cold call by a factor of top_k."""
    s = compute_utilization(np.full(SIZE, 100.0), N_EXPERTS, TOP_K, uids=uids())
    np.testing.assert_allclose(s.load_ratio, 1.0)
    assert s.expected_share == pytest.approx(1.0 / SIZE)
    assert s.n_hot == 0 and s.n_cold == 0 and s.n_dead == 0
    assert s.gini == pytest.approx(0.0, abs=1e-9)


def test_hot_cold_dead_classification():
    c = np.full(SIZE, 100.0)
    c[0] = 250.0    # >2x fair share once renormalised
    c[1] = 10.0     # cold
    c[2] = 0.0      # dead
    s = compute_utilization(c, N_EXPERTS, TOP_K, uids=uids())
    assert s.uids[0] in [u for u, r in zip(s.uids, s.load_ratio) if r >= 2.0]
    assert s.n_dead == 1
    assert s.n_cold >= 1
    # dead experts must not be counted as cold — they are a different failure
    assert s.load_ratio[2] == 0.0


def test_skew_ignores_dead_experts():
    """max/min over *alive* experts; including zeros would give inf."""
    c = np.full(SIZE, 100.0)
    c[0] = 0.0
    s = compute_utilization(c, N_EXPERTS, TOP_K, uids=uids())
    assert np.isfinite(s.skew)
    assert s.skew == pytest.approx(1.0)


def test_empty_counts_raise():
    with pytest.raises(ValueError):
        compute_utilization(np.zeros(SIZE), N_EXPERTS, TOP_K)


def test_gini_bounds():
    assert gini_coefficient(np.ones(50)) == pytest.approx(0.0, abs=1e-9)
    concentrated = np.zeros(50); concentrated[0] = 1.0
    assert gini_coefficient(concentrated) > 0.9


# --- the enrichment null -----------------------------------------------------

def test_enrichment_null_detects_planted_hot_specialists():
    c = np.full(SIZE, 100.0)
    c[:6] = 400.0                       # make the first six hot
    s = compute_utilization(c, N_EXPERTS, TOP_K, uids=uids())
    specialists = set(s.uids[:6])       # specialists ARE the hot ones
    r = hot_specialist_overlap(s, specialists, n_permutations=2000, seed=0)
    assert r["enrichment"] > 1.0
    assert r["p_value"] < 0.05
    assert "ENRICHED" in r["verdict"]


def test_enrichment_null_detects_depletion():
    """The direction actually measured on the real run — specialists avoid hot."""
    c = np.full(SIZE, 100.0)
    c[:6] = 400.0
    s = compute_utilization(c, N_EXPERTS, TOP_K, uids=uids())
    specialists = set(s.uids[6:])       # specialists are everything BUT the hot set
    r = hot_specialist_overlap(s, specialists, n_permutations=2000, seed=0)
    assert r["enrichment"] < 1.0
    assert "DEPLETED" in r["verdict"]


def test_enrichment_reports_no_effect_when_random():
    """Guards the failure mode of reporting a raw overlap count as a finding:
    with many specialists, a large raw overlap is expected by chance."""
    rng = np.random.default_rng(0)
    c = rng.uniform(50, 150, SIZE)
    s = compute_utilization(c, N_EXPERTS, TOP_K, uids=uids())
    specialists = set(rng.choice(s.uids, size=SIZE // 2, replace=False).tolist())
    r = hot_specialist_overlap(s, specialists, n_permutations=2000, seed=1)
    assert r["p_value"] > 0.01, "random specialist set flagged as enriched"


def test_enrichment_untestable_when_no_hot_experts():
    s = compute_utilization(np.full(SIZE, 100.0), N_EXPERTS, TOP_K, uids=uids())
    r = hot_specialist_overlap(s, set(s.uids[:4]), n_permutations=100)
    assert "untestable" in r["verdict"]
    assert r["p_value"] == 1.0


# --- the real run ------------------------------------------------------------

@pytest.mark.skipif(not (ROOT / "data" / "utilization.json").exists(),
                    reason="run scripts/run_utilization.py first")
def test_real_run_result_is_depletion_not_enrichment():
    """Pins the actual finding so it cannot silently flip without notice.

    The brief that commissioned this analysis hypothesised ENRICHMENT
    (specialisation concentrating load). The measurement found the opposite.
    """
    d = json.loads((ROOT / "data" / "utilization.json").read_text())
    o = d["hot_specialist_overlap"]
    assert o["enrichment"] < 1.0
    assert o["p_value"] < 0.01
    assert "DEPLETED" in o["verdict"]


@pytest.mark.skipif(not (ROOT / "data" / "utilization.json").exists(),
                    reason="run scripts/run_utilization.py first")
def test_real_run_has_no_dead_experts():
    """docs/FINDINGS.md rests on the 227x skew being genuine rather than a
    counting bug; a dead expert would undermine that reading."""
    d = json.loads((ROOT / "data" / "utilization.json").read_text())
    assert d["utilization"]["n_dead"] == 0


@pytest.mark.skipif(not (ROOT / "data" / "utilization.json").exists(),
                    reason="run scripts/run_utilization.py first")
def test_skew_exceeds_pmi_validity_limit_and_is_recorded():
    """coactivation.py's PMI limit is 2.0x. The real run is far outside it, so
    anything PMI-based must stay untrusted — this test exists so that gate is
    never quietly dropped."""
    d = json.loads((ROOT / "data" / "utilization.json").read_text())
    assert d["utilization"]["skew"] > 2.0
