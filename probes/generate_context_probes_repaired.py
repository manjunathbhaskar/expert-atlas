"""Generate probes/probe_set_context_repaired.yaml -- a deconfounded variant
for the residual-stream needle probe.

Why this exists: in probe_set_context_hard.yaml the entity<->answer_word
pairing is 1:1 (replicate r always uses ENTITIES[r] with CANDIDATE_WORDS[r]).
The question block names the entity, so a linear probe trained and tested on
that set can decode the answer word from the ENTITY MENTION alone -- verified:
a probe on the question-window mean at LAYER 0 (raw embeddings, before any
attention) scores 100%. Final-position decodability is therefore
uninterpretable as evidence of retrieval.

This set breaks the shortcut with two disjoint pairing systems:

  * pairing set A (replicates 0-7):  entity_i  ->  word_i
  * pairing set B (replicates 8-15): entity_i  ->  word_{(i+3) mod 8}

A probe trained on set A and evaluated on set B (and vice versa) cannot score
above chance via entity identity -- the entity shortcut now predicts a
SPECIFIC WRONG word, which is separately measurable as a "shortcut rate".
Any correct decoding on the held-out pairing set must come from the needle's
actual content.

Everything else follows the hard variant (8 distractors in the non-trivial
arm, similar/dissimilar haystacks, needle at 50% depth, byte-identical
needle/question across buckets within a replicate). Buckets are restricted to
(256, 1024, 3840) to keep the 192-prompt budget: 16 replicates x 2 haystacks
x 2 distractor levels x 3 buckets.

Usage:
    python probes/generate_context_probes_repaired.py
"""

from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from transformers import AutoTokenizer

import probes.probe_set_context as base

OUT = Path(__file__).parent / "probe_set_context_repaired.yaml"
SEED = 2  # distinct from base (0) and hard (1)
LENGTH_BUCKETS = (256, 1024, 3840)
DISTRACTOR_LEVELS = (0, 8)
DISTRACTOR_DEPTHS = (0.10, 0.22, 0.30, 0.42, 0.58, 0.70, 0.78, 0.90)
PAIR_SHIFT_B = 3


def pairings() -> list[tuple[int, str, str, str]]:
    """(replicate, pairing_set, entity, answer_word) for all 16 replicates."""
    out = []
    for i in range(8):
        out.append((i, "A", base.ENTITIES[i], base.CANDIDATE_WORDS[i]))
    for i in range(8):
        out.append((8 + i, "B", base.ENTITIES[i],
                    base.CANDIDATE_WORDS[(i + PAIR_SHIFT_B) % 8]))
    return out


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(base.MODEL_ID)
    base.assert_single_token(tokenizer)

    def n_tok(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    prompts = []
    pid = 0
    for rep, pset, entity, answer_word in pairings():
        rng = random.Random(SEED * 100003 + rep)
        needle = base.NEEDLE_TEMPLATE.format(entity=entity, word=answer_word)
        other_entities = [e for e in base.ENTITIES if e != entity]
        other_words = [w for w in base.CANDIDATE_WORDS if w != answer_word]
        distractor_pool = [
            base.NEEDLE_TEMPLATE.format(
                entity=other_entities[i % len(other_entities)],
                word=other_words[i % len(other_words)])
            for i in range(max(DISTRACTOR_LEVELS))
        ]
        for haystack in base.HAYSTACK_DOMAINS:
            stream = base._generate_haystack_stream(
                haystack, random.Random(rng.randrange(1 << 30)), n_sentences=420)
            for n_dist in DISTRACTOR_LEVELS:
                distractors = distractor_pool[:n_dist]
                for bucket in LENGTH_BUCKETS:
                    # binary-search the haystack prefix, as in build_probe_set
                    saved = base.DISTRACTOR_DEPTHS
                    base.DISTRACTOR_DEPTHS = DISTRACTOR_DEPTHS
                    try:
                        lo, hi, best = 0, len(stream), None
                        while lo <= hi:
                            mid = (lo + hi) // 2
                            text, nspan, qspan = base._assemble(
                                stream[:mid], needle, distractors, entity)
                            t = n_tok(text)
                            if t <= bucket:
                                best = (text, nspan, qspan, t)
                                lo = mid + 1
                            else:
                                hi = mid - 1
                    finally:
                        base.DISTRACTOR_DEPTHS = saved
                    if best is None:
                        raise RuntimeError(f"bucket {bucket} too small")
                    text, nspan, qspan, t = best
                    prompts.append({
                        "prompt_id": pid, "text": text, "bucket": bucket,
                        "n_tokens": t, "haystack": haystack,
                        "n_distractors": n_dist, "replicate": rep,
                        "pairing_set": pset,
                        "entity": entity, "answer_word": answer_word,
                        "needle_char_span": list(nspan),
                        "question_char_span": list(qspan),
                        "sha1": hashlib.sha1(text.encode()).hexdigest()[:12],
                    })
                    pid += 1

    ps = {
        "probe_set_id": "probe_set_context_repaired",
        "version": "1.0",
        "design": ("needle-in-haystack with two disjoint entity<->answer "
                   "pairing sets, so a linear probe cannot use entity "
                   "identity across the A/B split"),
        "tokenizer_model_id": base.MODEL_ID,
        "length_buckets": list(LENGTH_BUCKETS),
        "distractor_levels": list(DISTRACTOR_LEVELS),
        "haystack_domains": list(base.HAYSTACK_DOMAINS),
        "needle_depth": base.NEEDLE_DEPTH,
        "distractor_depths": list(DISTRACTOR_DEPTHS),
        "candidate_words": list(base.CANDIDATE_WORDS),
        "chance_accuracy": 1.0 / len(base.CANDIDATE_WORDS),
        "pair_shift_B": PAIR_SHIFT_B,
        "n_replicates": 16,
        "n_prompts": len(prompts),
        "seed": SEED,
        "prompts": prompts,
    }
    OUT.write_text(yaml.safe_dump(ps, sort_keys=False, allow_unicode=True))
    print(f"wrote {OUT}: {len(prompts)} prompts")
    for b in LENGTH_BUCKETS:
        got = [p["n_tokens"] for p in prompts if p["bucket"] == b]
        print(f"  bucket {b:>5}: n={len(got)} min={min(got)} max={max(got)}")


if __name__ == "__main__":
    main()
