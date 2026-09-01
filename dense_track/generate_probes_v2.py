"""REGISTERED v2 dense-track substrate (see REGISTRATION_V2.md).

Confirmatory replication substrate: same-entity confusable-attribute
distractor design (which induced failures in the v1 exploratory arm),
regenerated fresh with a new seed and 24 replicates, registered before any
v2 measurement. Distractors use the SAME entity as the needle with a
confusable attribute word, e.g.

    needle:      "The security codeword for the Zurich office is silver."
    distractor:  "The visitor codeword for the Zurich office is copper."

The question asks for the SECURITY codeword, so the task stays uniquely
answerable, but every distractor now collides with the entity the question
names. 8 distractors, attribute words drawn round-robin from a fixed pool,
wrong answers from the forced-choice candidate pool as before.

24 replicates x 2 haystacks x 3 buckets (256, 1024, 1900) = 144 prompts.

Usage:
    .venv-dense/bin/python dense_track/generate_probes_v2.py
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
from dense_track.common import MODEL_ID

OUT = Path(__file__).parent / "probe_set_dense_v2.yaml"
SEED = 23
LENGTH_BUCKETS = (256, 1024, 1900)
N_DISTRACTORS = 8
DISTRACTOR_DEPTHS = (0.10, 0.22, 0.30, 0.42, 0.58, 0.70, 0.78, 0.90)

CONFUSABLE_ATTRIBUTES = (
    "visitor", "loading", "maintenance", "evening",
    "backup", "delivery", "archive", "weekend",
)
DISTRACTOR_TEMPLATE = "The {attr} codeword for the {entity} office is {word}."


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base.assert_single_token(tokenizer)

    def n_tok(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    prompts = []
    pid = 0
    for rep in range(24):
        rng = random.Random(SEED * 100003 + rep)
        entity = base.ENTITIES[rep % len(base.ENTITIES)]
        answer_word = base.CANDIDATE_WORDS[
            (rep + rep // len(base.ENTITIES)) % len(base.CANDIDATE_WORDS)]
        needle = base.NEEDLE_TEMPLATE.format(entity=entity, word=answer_word)
        other_words = [w for w in base.CANDIDATE_WORDS if w != answer_word]
        distractors = [
            DISTRACTOR_TEMPLATE.format(
                attr=CONFUSABLE_ATTRIBUTES[i % len(CONFUSABLE_ATTRIBUTES)],
                entity=entity,
                word=other_words[i % len(other_words)])
            for i in range(N_DISTRACTORS)
        ]
        for haystack in base.HAYSTACK_DOMAINS:
            stream = base._generate_haystack_stream(
                haystack, random.Random(rng.randrange(1 << 30)), n_sentences=420)
            for bucket in LENGTH_BUCKETS:
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
                    "n_distractors": N_DISTRACTORS, "replicate": rep,
                    "entity": entity, "answer_word": answer_word,
                    "needle_char_span": list(nspan),
                    "question_char_span": list(qspan),
                    "sha1": hashlib.sha1(text.encode()).hexdigest()[:12],
                })
                pid += 1

    ps = {
        "probe_set_id": "probe_set_dense_v2",
        "version": "1.0",
        "design": ("REGISTERED v2 substrate: same-entity confusable-attribute "
                   "distractors; full pipeline re-run per REGISTRATION_V2.md"),
        "tokenizer_model_id": MODEL_ID,
        "length_buckets": list(LENGTH_BUCKETS),
        "n_distractors": N_DISTRACTORS,
        "haystack_domains": list(base.HAYSTACK_DOMAINS),
        "needle_depth": base.NEEDLE_DEPTH,
        "distractor_depths": list(DISTRACTOR_DEPTHS),
        "candidate_words": list(base.CANDIDATE_WORDS),
        "chance_accuracy": 1.0 / len(base.CANDIDATE_WORDS),
        "n_replicates": 24,
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
