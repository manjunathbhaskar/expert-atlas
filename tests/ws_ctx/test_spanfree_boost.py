"""Unit tests for the span-free detection helpers (scripts/run_spanfree_boost)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.run_spanfree_boost import detect, overlaps


def test_detect_finds_peak_window():
    sig = np.zeros(200)
    sig[50:62] = 1.0
    start, end = detect(sig, 12, q_start=180)
    assert end - start == 12
    assert overlaps((start, end), (50, 62))


def test_detect_excludes_position_zero_and_question():
    sig = np.zeros(200)
    sig[0] = 100.0        # attention-sink artifact must not win
    sig[190:] = 100.0     # question region must not be a candidate
    sig[30:38] = 1.0
    start, end = detect(sig, 8, q_start=180)
    assert start >= 1 and end <= 180
    assert overlaps((start, end), (30, 38))


def test_overlaps_basic():
    assert overlaps((5, 10), (9, 12))
    assert not overlaps((5, 10), (10, 12))
    assert not overlaps((0, 3), (5, 8))
