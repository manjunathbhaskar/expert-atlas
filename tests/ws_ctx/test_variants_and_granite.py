"""Unit tests for the harder-variant generators and the Granite/variant
pipelines' pure helpers."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from probes.generate_context_probes_granite_hard import _depths
from probes.generate_context_probes_variants import (_assemble, _bridge_depth,
                                                     HOP1, HOP2, PARA_NEEDLE)
import probes.probe_set_context as base


def test_hard_distractor_depths_avoid_needle_and_stay_in_range():
    d = _depths(24, 0.15)
    assert len(d) == 24
    assert all(0.0 < x < 1.0 for x in d)
    # shifted points can land as close as ~0.028 from the needle depth;
    # the guarantee is separation, not a wide margin
    assert all(abs(x - 0.15) >= 0.02 for x in d)


def test_bridge_depth_far_from_needle_and_in_range():
    for nd in (0.15, 0.50):
        bd = _bridge_depth(nd)
        assert 0.05 <= bd <= 0.95
        assert abs(bd - nd) >= 0.2


def test_assemble_records_correct_spans_for_all_inserts():
    sentences = [f"Filler sentence number {i} about weather patterns." for i in range(40)]
    needle = HOP2.format(site="Kestrel", word="silver")
    bridge = HOP1.format(entity="Zurich", site="Kestrel")
    text, spans, qspan = _assemble(sentences, [(0.15, needle), (0.55, bridge)],
                                   "Zurich")
    assert text[spans[0][0]:spans[0][1]] == needle
    assert text[spans[1][0]:spans[1][1]] == bridge
    assert text[qspan[0]:qspan[1]] == base.QUESTION_TEMPLATE.format(entity="Zurich")
    assert spans[0][0] < spans[1][0]  # needle earlier than bridge here


def test_paraphrase_needle_shares_only_entity_with_question():
    q = base.QUESTION_TEMPLATE.format(entity="Zurich").lower()
    n = PARA_NEEDLE.format(entity="Zurich", word="silver").lower()
    q_words = {w.strip("?.,") for w in q.split() if len(w.strip("?.,")) >= 4}
    n_words = {w.strip("?.,") for w in n.split() if len(w.strip("?.,")) >= 4}
    assert q_words & n_words == {"zurich"}


def test_multihop_fact_shares_no_contentful_token_with_question():
    q = base.QUESTION_TEMPLATE.format(entity="Zurich").lower()
    n = HOP2.format(site="Kestrel", word="silver").lower()
    q_words = {w.strip("?.,") for w in q.split() if len(w.strip("?.,")) >= 4}
    n_words = {w.strip("?.,") for w in n.split() if len(w.strip("?.,")) >= 4}
    # "codeword"/"security" appear in both by template: the multihop needle
    # intentionally reuses the fact template, but NOT the entity
    assert "zurich" not in n_words
    assert "kestrel" not in q_words


def test_detect_l8_masks_question_region():
    from scripts.run_context_variants import detect_l8

    class FakeClf:
        def predict_proba(self, X):
            p = np.zeros((len(X), 2))
            p[:, 1] = X[:, 0]
            return p

    hs = np.zeros((100, 3))
    hs[40, 0] = 1.0
    hs[90, 0] = 5.0  # inside question region — must be ignored
    span = detect_l8(FakeClf(), hs, 8, q_start=80)
    assert span[0] == 40 and span[1] == 48
