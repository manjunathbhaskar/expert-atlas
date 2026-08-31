"""Probe-set invariants for the context sweep.

The design's whole validity rests on a few structural guarantees. If any of
these break, length stops being the only independent variable and every routing
number in docs/CONTEXT_ROT.md becomes uninterpretable — so they are tested,
not assumed.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest
import yaml

from probes.probe_set_context import (
    CANDIDATE_WORDS,
    DISTRACTOR_LEVELS,
    HAYSTACK_DOMAINS,
    LENGTH_BUCKETS,
    NEEDLE_TEMPLATE,
    QUESTION_TEMPLATE,
    _generate_haystack_stream,
    assert_single_token,
    build_probe_set,
)

PROBE_SET = Path(__file__).parent.parent.parent / "probes" / "probe_set_context.yaml"


@pytest.fixture(scope="module")
def ps():
    if not PROBE_SET.exists():
        pytest.skip("probe_set_context.yaml not generated")
    return yaml.safe_load(PROBE_SET.read_text())


class TestDesign:
    def test_at_least_six_length_buckets(self, ps):
        assert len(ps["length_buckets"]) >= 6

    def test_buckets_span_short_to_near_context_limit(self, ps):
        b = sorted(ps["length_buckets"])
        assert b[0] <= 256
        # OLMoE max_position_embeddings = 4096; stay clear but get close.
        assert 3500 <= b[-1] < 4096

    def test_both_chroma_conditions_are_present(self, ps):
        assert set(ps["distractor_levels"]) == set(DISTRACTOR_LEVELS)
        assert set(ps["haystack_domains"]) == set(HAYSTACK_DOMAINS)

    def test_design_is_fully_crossed(self, ps):
        cells = defaultdict(int)
        for p in ps["prompts"]:
            cells[(p["bucket"], p["haystack"], p["n_distractors"])] += 1
        expected = len(LENGTH_BUCKETS) * len(HAYSTACK_DOMAINS) * len(DISTRACTOR_LEVELS)
        assert len(cells) == expected
        assert len(set(cells.values())) == 1, f"unbalanced cells: {dict(cells)}"

    def test_prompts_land_inside_their_bucket(self, ps):
        for p in ps["prompts"]:
            assert p["n_tokens"] <= p["bucket"]
            # within 12% of target, else the "length is the IV" claim is loose
            assert p["n_tokens"] >= p["bucket"] * 0.88, p


class TestTheControlThatMakesItWork:
    """Within a replicate, only the haystack amount may vary."""

    def test_needle_is_byte_identical_across_all_buckets_and_conditions(self, ps):
        by_rep = defaultdict(set)
        for p in ps["prompts"]:
            needle = NEEDLE_TEMPLATE.format(entity=p["entity"], word=p["answer_word"])
            a, b = p["needle_char_span"]
            by_rep[p["replicate"]].add(p["text"][a:b])
            assert p["text"][a:b] == needle
        for rep, s in by_rep.items():
            assert len(s) == 1, f"replicate {rep} needle text varies across cells: {s}"

    def test_question_block_is_byte_identical_across_all_buckets(self, ps):
        by_rep = defaultdict(set)
        for p in ps["prompts"]:
            a, b = p["question_char_span"]
            by_rep[p["replicate"]].add(p["text"][a:b])
            assert p["text"][a:b] == QUESTION_TEMPLATE.format(entity=p["entity"])
        for rep, s in by_rep.items():
            assert len(s) == 1, f"replicate {rep} question text varies: {s}"

    def test_question_block_is_the_prompt_suffix(self, ps):
        """The measurement window must be the trailing tokens, so it is the
        part of the prompt the model reads LAST at every length."""
        for p in ps["prompts"]:
            assert p["question_char_span"][1] == len(p["text"])

    def test_haystack_is_a_prefix_across_buckets(self, ps):
        """Shorter buckets must use a prefix of the same sentence stream, so
        haystack topic mix cannot covary with length.

        Checked via substring containment on the whitespace-normalised body,
        not exact split-token set equality. `_sentences()`'s naive
        ". "-splitting is sensitive to exactly where the needle sat before
        removal (its insertion depth is a fraction of a length that differs
        per bucket, so it can land next to a different sentence boundary in
        each bucket), which produced false positives here even though the
        underlying stream content was verified, directly and manually, to
        nest correctly (every raw sentence from the short bucket's body is
        a literal substring of the long bucket's body). Containment on
        normalised text is the property that actually matters -- whether
        the short haystack's CONTENT is present in the long one -- and does
        not depend on getting sentence-boundary tokenisation exactly right.
        """
        groups = defaultdict(dict)
        for p in ps["prompts"]:
            groups[(p["replicate"], p["haystack"], p["n_distractors"])][p["bucket"]] = p

        def normalised_body(p) -> str:
            a, b = p["needle_char_span"]
            body = p["text"][: p["question_char_span"][0]]
            needle = p["text"][a:b]
            return " ".join(body.replace(needle, " ").split())

        for key, by_bucket in groups.items():
            ordered = [by_bucket[b] for b in sorted(by_bucket)]
            for short, long in zip(ordered, ordered[1:]):
                for s in _sentences(short):
                    assert s in normalised_body(long), (
                        f"{key}: sentence from bucket {short['bucket']} not found in "
                        f"bucket {long['bucket']}'s body — topic mix drifts with length: {s!r}"
                    )

    def test_no_repeated_haystack_sentences(self, ps):
        """Verbatim repetition growing with length would confound the IV."""
        for p in ps["prompts"]:
            sents = _sentences(p)
            assert len(sents) == len(set(sents)), (
                f"prompt {p['prompt_id']} repeats haystack text; long buckets would "
                "then differ from short ones in repetition as well as length"
            )


def _sentences(p) -> list[str]:
    a, b = p["needle_char_span"]
    body = p["text"][: p["question_char_span"][0]]
    needle = p["text"][a:b]
    # Removing the needle leaves a double-space artifact at whatever sentence
    # boundary it was inserted into, and that boundary's position varies by
    # bucket (insertion depth is a fraction of a length that differs per
    # bucket) -- so a plain ". "-split mis-parses around it differently per
    # bucket, even though the underlying sentence stream is a genuine prefix.
    # Collapsing whitespace before splitting removes the artifact rather than
    # the split logic silently depending on exactly where the needle sat.
    cleaned = " ".join(body.replace(needle, " ").split())
    parts = [s.strip() for s in cleaned.split(". ")]
    return [s for s in parts if len(s) > 20]


class TestAnswerScoring:
    def test_candidate_pool_size_sets_chance_accuracy(self, ps):
        assert ps["chance_accuracy"] == pytest.approx(1.0 / len(CANDIDATE_WORDS))
        assert len(set(ps["candidate_words"])) == len(ps["candidate_words"])

    def test_answer_word_is_always_in_the_candidate_pool(self, ps):
        for p in ps["prompts"]:
            assert p["answer_word"] in ps["candidate_words"]

    def test_distractors_use_wrong_answers_from_the_same_pool(self, ps):
        """Distractors must compete in scoring, not merely add tokens."""
        for p in ps["prompts"]:
            if p["n_distractors"] == 0:
                continue
            body = p["text"][: p["question_char_span"][0]]
            hits = [w for w in ps["candidate_words"]
                    if w != p["answer_word"] and f"is {w}." in body]
            assert hits, f"prompt {p['prompt_id']} has distractors that never compete"

    def test_correct_answer_appears_exactly_once_in_the_haystack(self, ps):
        for p in ps["prompts"]:
            body = p["text"][: p["question_char_span"][0]]
            assert body.count(f"is {p['answer_word']}.") == 1

    def test_assert_single_token_rejects_multi_token_words(self):
        class FakeTok:
            def __call__(self, s, add_special_tokens=False):
                return {"input_ids": [1, 2] if "juniper" in s else [1]}

        assert_single_token(FakeTok(), words=("silver",))
        with pytest.raises(RuntimeError, match="not single tokens"):
            assert_single_token(FakeTok(), words=("juniper",))


class TestGenerator:
    def test_haystack_stream_is_deterministic(self):
        import random
        a = _generate_haystack_stream("similar", random.Random(1), 100)
        b = _generate_haystack_stream("similar", random.Random(1), 100)
        assert a == b

    def test_haystack_domains_are_lexically_disjoint(self):
        """The similar/dissimilar contrast must actually be a contrast."""
        import random
        a = " ".join(_generate_haystack_stream("similar", random.Random(2), 200)).lower()
        b = " ".join(_generate_haystack_stream("dissimilar", random.Random(2), 200)).lower()
        assert "glacier" not in a and "office" not in b

    def test_build_is_reproducible(self):
        class FakeTok:
            # Fixed single-token response -- this test is about determinism
            # of build_probe_set given a fixed seed, not about real
            # tokenization, and assert_single_token only checks len==1.
            # A length-based heuristic (len(s)//4) previously used here
            # coincidentally returned 2 "tokens" for the two 8-character
            # candidate words ("thunder", "lantern"), tripping the real
            # single-token invariant check for the wrong reason.
            def __call__(self, s, add_special_tokens=False, **kw):
                return {"input_ids": [0]}

        a = build_probe_set(FakeTok(), replicates=1, seed=0)
        b = build_probe_set(FakeTok(), replicates=1, seed=0)
        assert [p["sha1"] for p in a["prompts"]] == [p["sha1"] for p in b["prompts"]]
