"""WS2: the matched-LOAD null, which separates 'which experts' from 'how much network'.

The utilization analysis measured that H1 specialists are disproportionately cold, and that
per-domain load removed spans 5.3x. A size-matched null holds |S| fixed but lets
total load float to ~|S| by construction, so it cannot distinguish "sql's experts
matter" from "sql's ablation deleted twice as much routed traffic".
"""

import json
from pathlib import Path

import numpy as np
import pytest

from expertatlas.interference import load_removed, matched_load_null_sets

ROOT = Path(__file__).resolve().parents[2]
N = 1024


@pytest.fixture(scope="module")
def load_vec():
    """Skewed like the real thing (WS3 measured 227x skew, Gini 0.43)."""
    rng = np.random.default_rng(0)
    v = rng.lognormal(mean=0.0, sigma=1.0, size=N)
    return v / v.mean()          # mean load_ratio == 1.0, as the real vector is


def test_returns_sets_of_exact_target_size(load_vec):
    rng = np.random.default_rng(1)
    target = set(range(100))
    sets, _ = matched_load_null_sets(load_vec, target, 10, rng)
    assert all(len(s) == 100 for s in sets)


def test_matches_target_load_within_tolerance(load_vec):
    rng = np.random.default_rng(2)
    hot = set(np.argsort(load_vec)[-100:].tolist())   # a deliberately hot set
    target_load = load_removed(load_vec, hot)
    sets, diag = matched_load_null_sets(load_vec, hot, 20, rng, tolerance=0.05)
    for s in sets:
        assert abs(load_removed(load_vec, s) - target_load) <= 0.05 * target_load
    assert diag["achieved_relative_error_max"] <= 0.05


def test_beats_size_matched_null_on_load_bias(load_vec):
    """The load-bearing claim: a size-matched null is biased when the target set
    is not average-load; a load-matched null is not."""
    rng = np.random.default_rng(3)
    hot = set(np.argsort(load_vec)[-100:].tolist())
    target_load = load_removed(load_vec, hot)

    size_matched = [load_removed(load_vec, set(rng.choice(N, 100, replace=False).tolist()))
                    for _ in range(100)]
    load_matched, _ = matched_load_null_sets(load_vec, hot, 100, rng, tolerance=0.05)
    lm = [load_removed(load_vec, s) for s in load_matched]

    size_bias = abs(np.mean(size_matched) - target_load)
    load_bias = abs(np.mean(lm) - target_load)
    assert load_bias < size_bias / 5, (
        f"matched-load null not materially better: bias {load_bias:.1f} vs {size_bias:.1f}"
    )


def test_sets_are_not_all_identical(load_vec):
    """Matching load must not collapse the null onto one set — then it would have
    no variance and every observation would look extreme."""
    rng = np.random.default_rng(4)
    hot = set(np.argsort(load_vec)[-100:].tolist())
    sets, _ = matched_load_null_sets(load_vec, hot, 30, rng, tolerance=0.05)
    assert len({frozenset(s) for s in sets}) >= 25


def test_infeasible_target_is_flagged_not_silently_wrong(load_vec):
    """A target hotter than any 100 experts can reach must WARN, not return a
    quietly-biased null. Silent failure here would invalidate a published claim."""
    rng = np.random.default_rng(5)
    impossible = set(np.argsort(load_vec)[-100:].tolist())
    inflated = load_vec.copy()
    # Make the target unreachable by shrinking everything except the target set.
    mask = np.ones(N, bool)
    mask[list(impossible)] = False
    inflated[mask] *= 0.01
    _sets, diag = matched_load_null_sets(inflated, impossible, 5, rng, tolerance=0.001,
                                         max_tries_per_set=50)
    assert diag["n_returned"] < 5
    assert "warning" in diag


def test_deterministic_given_seed(load_vec):
    target = set(range(100))
    a, _ = matched_load_null_sets(load_vec, target, 5, np.random.default_rng(7))
    b, _ = matched_load_null_sets(load_vec, target, 5, np.random.default_rng(7))
    assert [sorted(s) for s in a] == [sorted(s) for s in b]


def test_rejects_oversized_request(load_vec):
    with pytest.raises(ValueError):
        matched_load_null_sets(load_vec, set(range(N + 5)), 2, np.random.default_rng(8))


# --- against the real vectors ------------------------------------------------

@pytest.mark.skipif(not (ROOT / "data" / "utilization.json").exists()
                    or not (ROOT / "data" / "interference_precompute.json").exists(),
                    reason="needs the real run artefacts")
def test_all_six_real_domains_are_feasible():
    """If any real domain's load were unreachable, WS2's null would be invalid
    for that domain and the regression could not include it."""
    u = json.loads((ROOT / "data" / "utilization.json").read_text())
    d = json.loads((ROOT / "data" / "interference_precompute.json").read_text())
    load = np.array(u["utilization"]["load_ratio"])
    idx = {x: i for i, x in enumerate(u["utilization"]["uids"])}
    rng = np.random.default_rng(0)

    for dom, uids in d["expert_set_uids"].items():
        S = {idx[x] for x in uids}
        sets, diag = matched_load_null_sets(load, S, 10, rng, tolerance=0.02)
        assert diag["feasible"], f"{dom}: {diag.get('warning')}"
        assert diag["n_returned"] == 10, f"{dom} underpowered: {diag}"


@pytest.mark.skipif(not (ROOT / "data" / "interference_precompute.json").exists(),
                    reason="needs the real run artefacts")
def test_real_load_spread_justifies_this_null():
    """Pins WS3's finding: the spread is large enough that a size-matched null
    is genuinely inadequate. If this ever drops near 1.0x, this whole apparatus
    is unnecessary and should be removed rather than kept out of habit."""
    d = json.loads((ROOT / "data" / "interference_precompute.json").read_text())
    lr = d["load_removed"]
    assert max(lr.values()) / min(lr.values()) > 3.0
