"""Unit tests for the lexical span-discovery detector."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.run_span_discovery import lexical_signal, detect_forward, MAX_DF
from scripts.run_spanfree_boost import detect, overlaps


class FakeTok:
    """Token id -> string via a fixed vocabulary."""

    def __init__(self, vocab):
        self.vocab = vocab

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)


def test_rare_question_token_scores_at_its_context_position():
    vocab = {0: " the", 1: " Lisbon", 2: " filler", 3: "\n", 4: " office"}
    tok = FakeTok(vocab)
    ctx = [2, 2, 1, 2, 2, 0, 0, 0]
    q = [1, 4]  # " Lisbon office" — office absent from context
    sig = lexical_signal(ctx, q, tok)
    assert sig[2] == 1.0  # df(" Lisbon") == 1
    assert sig.sum() == 1.0


def test_whitespace_and_short_tokens_carry_no_signal():
    vocab = {0: "\n", 1: " is", 2: " Lisbon", 3: " filler"}
    tok = FakeTok(vocab)
    ctx = [0, 0, 1, 3, 2, 3]
    q = [0, 1, 2]
    sig = lexical_signal(ctx, q, tok)
    assert sig[0] == 0.0 and sig[1] == 0.0  # "\n" excluded (non-alpha)
    assert sig[2] == 0.0  # " is" excluded (<3 alpha chars)
    assert sig[4] > 0.0


def test_high_df_tokens_filtered():
    vocab = {0: " common", 1: " rare"}
    tok = FakeTok(vocab)
    ctx = [0] * MAX_DF + [1]
    sig = lexical_signal(ctx, [0, 1], tok)
    assert sig[:MAX_DF].sum() == 0.0
    assert sig[MAX_DF] == 1.0


def test_idf_weighting_prefers_rarer_token():
    vocab = {0: " duplicated", 1: " unique", 2: " pad"}
    tok = FakeTok(vocab)
    ctx = [0, 2, 0, 2, 1, 2]
    sig = lexical_signal(ctx, [0, 1], tok)
    assert sig[4] == 1.0
    assert sig[0] == 0.5 == sig[2]


def test_detect_forward_ties_break_to_latest_window():
    # single spike: every window containing it ties; v2 must start AT the
    # spike so coverage extends forward over the asserted fact
    sig = np.zeros(200)
    sig[100] = 1.0
    span = detect_forward(sig, 24, q_start=180)
    assert span == (100, 124)


def test_detect_forward_respects_q_start():
    sig = np.zeros(200)
    sig[190] = 5.0  # inside question region — ignored
    sig[50] = 1.0
    span = detect_forward(sig, 8, q_start=180)
    assert span == (50, 58)


def test_detect_windows_pick_peak_and_respect_q_start():
    sig = np.zeros(100)
    sig[50:54] = 1.0
    sig[90] = 10.0  # inside the question region — must be ignored
    span = detect(sig, 8, q_start=80)
    assert overlaps(span, (50, 54))
    assert span[1] <= 80
