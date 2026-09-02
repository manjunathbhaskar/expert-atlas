"""Unit tests for the distance-only analysis and the two-stage chain detector."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.run_distance_only import spearman_perm
from scripts.run_multihop_chain import chain_detect


def test_spearman_perm_detects_perfect_monotone_decline():
    rng = np.random.default_rng(0)
    x = np.arange(40, dtype=np.float64)
    y = -x + 0.001 * rng.standard_normal(40)
    rho, p = spearman_perm(x.copy(), y, np.random.default_rng(1))
    assert rho < -0.99
    assert p < 0.05


def test_spearman_perm_null_on_unrelated_data():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(40)
    y = rng.standard_normal(40)
    rho, p = spearman_perm(x.copy(), y, np.random.default_rng(3))
    assert abs(rho) < 0.5
    assert p > 0.05


class FakeTok:
    """Whitespace tokenizer over a fixed word list."""

    def __init__(self):
        self.w2i: dict[str, int] = {}
        self.i2w: dict[int, str] = {}

    def _id(self, w):
        if w not in self.w2i:
            i = len(self.w2i)
            self.w2i[w] = i
            self.i2w[i] = w
        return self.w2i[w]

    def __call__(self, text):
        return {"input_ids": [self._id(w) for w in text.split()]}

    def decode(self, ids):
        return " ".join(self.i2w[i] for i in ids)


def test_chain_detect_hops_from_bridge_to_fact():
    tok = FakeTok()
    filler = " ".join(f"filler{i % 6} words here" for i in range(20))
    bridge = "Zurich office designated Site Kestrel"
    fact = "codeword for Site Kestrel is silver"
    text = f"{filler} {bridge} {filler} {fact} {filler} QQQ what about Zurich office"
    ids = tok(text)["input_ids"]
    q_start = ids.index(tok.w2i["QQQ"])
    rec = {"question_token_span": [q_start, len(ids)]}

    a_span, b_span = chain_detect(tok, text, rec, w_a=5, w_b=6)

    b_lo, b_hi = ids.index(tok.w2i["designated"]) - 2, ids.index(tok.w2i["designated"]) + 3
    assert a_span[0] <= b_lo + 2 and a_span[1] >= b_lo + 1  # stage A near the bridge
    fact_start = len(ids) - 1 - ids[::-1].index(tok.w2i["codeword"])
    assert b_span[0] <= fact_start < b_span[1] or (
        b_span[0] < fact_start + 6 and b_span[1] > fact_start)  # stage B covers the fact
    # stage B must have left the stage-A window
    assert not (b_span[0] >= a_span[0] and b_span[1] <= a_span[1])
