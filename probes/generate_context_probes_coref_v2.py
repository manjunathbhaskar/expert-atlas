"""Generate probes/probe_set_context_coref_v2.yaml -- coreference with
RANDOMIZED per-pair antecedent distance.

Registered design: docs/COREF_V2_REGISTRATION.md. Identical to the v1
coref substrate except that the anchor-to-referent distance d in {1,2,3}
is drawn uniformly, per pair, with its own RNG draw -- independently for
the true pair and for each of the 4 distractor pairs in the same prompt.
The d-1 sentences between anchor and referent are pulled from the normal
haystack-domain sentence generator (a dedicated filler stream), so no
fixed offset is correct by construction.

Usage:
    .venv-dense/bin/python probes/generate_context_probes_coref_v2.py
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
from probes.generate_context_probes_depth import _distractor_depths
from probes.generate_context_probes_variants import _assemble

OUT = Path(__file__).parent / "probe_set_context_coref_v2.yaml"
REPLICATES = 8
SEED = 13
DEPTHS = (0.15, 0.50)
BUCKETS = (384, 3840)
DEV_BUCKET = 1024
N_DISTRACTORS = 4
DISTANCES = (1, 2, 3)

ANCHOR = "The {entity} office is run by a single site director."
COREF_NEEDLE = ("Her personal passphrase, {word}, stays locked in the "
                "drawer.")


def _pair(anchor: str, needle: str, d: int, fillers: list[str]) -> str:
    """Anchor + (d-1) filler sentences + referent, as one insert block."""
    return " ".join([anchor, *[fillers.pop() for _ in range(d - 1)], needle])


def build(tokenizer) -> dict:
    def n_tok(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    base.assert_single_token(tokenizer)
    prompts = []
    pid = 0
    for rep in range(REPLICATES):
        rng = random.Random(SEED * 100003 + rep)
        entity = base.ENTITIES[rep % len(base.ENTITIES)]
        answer_word = base.CANDIDATE_WORDS[rep % len(base.CANDIDATE_WORDS)]
        other_e = [e for e in base.ENTITIES if e != entity]
        other_w = [w for w in base.CANDIDATE_WORDS if w != answer_word]

        for depth in DEPTHS:
            d_depths = _distractor_depths(depth)
            for haystack in base.HAYSTACK_DOMAINS:
                stream = base._generate_haystack_stream(
                    haystack, random.Random(rng.randrange(1 << 30)), 420)
                for bucket in BUCKETS + ((DEV_BUCKET,)
                                             if depth == 0.15 else ()):
                    # One independent uniform draw per pair, per emitted
                    # prompt (true first, then each distractor).
                    true_d = rng.choice(DISTANCES)
                    dist_ds = [rng.choice(DISTANCES)
                               for _ in range(N_DISTRACTORS)]
                    fillers = base._generate_haystack_stream(
                        haystack, random.Random(rng.randrange(1 << 30)), 40)
                    pair = _pair(ANCHOR.format(entity=entity),
                                 COREF_NEEDLE.format(word=answer_word),
                                 true_d, fillers)
                    distractor_pool = [
                        _pair(ANCHOR.format(entity=other_e[i % len(other_e)]),
                              COREF_NEEDLE.format(
                                  word=other_w[i % len(other_w)]),
                              dist_ds[i], fillers)
                        for i in range(N_DISTRACTORS)]
                    inserts = [(depth, pair)]
                    for dd, d in zip(d_depths, distractor_pool):
                        inserts.append((dd, d))
                    lo, hi, best = 0, len(stream), None
                    while lo <= hi:
                        mid = (lo + hi) // 2
                        text, spans, qspan = _assemble(
                            stream[:mid], inserts, entity)
                        t = n_tok(text)
                        if t <= bucket:
                            best = (text, spans, qspan, t)
                            lo = mid + 1
                        else:
                            hi = mid - 1
                    if best is None:
                        raise RuntimeError(f"bucket {bucket} too small")
                    text, spans, qspan, t = best
                    pair_s, pair_e = spans[0]
                    needle_sent = COREF_NEEDLE.format(word=answer_word)
                    n_s = text.index(needle_sent, pair_s)
                    anchor_sent = ANCHOR.format(entity=entity)
                    prompts.append({
                        "prompt_id": pid, "text": text,
                        "variant": "coref_v2",
                        "bucket": bucket, "n_tokens": t,
                        "haystack": haystack,
                        "n_distractors": N_DISTRACTORS,
                        "replicate": rep, "entity": entity,
                        "answer_word": answer_word,
                        "needle_depth": depth,
                        "coref_distance": true_d,
                        "distractor_distances": list(dist_ds),
                        "dev": bucket == DEV_BUCKET,
                        "needle_char_span": [n_s, n_s + len(needle_sent)],
                        "anchor_char_span": [pair_s,
                                             pair_s + len(anchor_sent)],
                        "question_char_span": list(qspan),
                        "sha1": hashlib.sha1(text.encode()).hexdigest()[:12],
                    })
                    pid += 1

    return {
        "probe_set_id": "probe_set_context_coref_v2",
        "tokenizer_model_id": base.MODEL_ID,
        "variants": ["coref_v2"],
        "length_buckets": list(BUCKETS), "dev_bucket": DEV_BUCKET,
        "needle_depths": list(DEPTHS),
        "coref_distances": list(DISTANCES),
        "candidate_words": list(base.CANDIDATE_WORDS),
        "chance_accuracy": 1.0 / len(base.CANDIDATE_WORDS),
        "n_replicates": REPLICATES, "seed": SEED,
        "n_prompts": len(prompts),
        "notes": (
            "Coreference v2: anchor-to-referent distance drawn uniformly "
            "from {1,2,3} per pair (own RNG draw, true pair and each "
            "distractor independently); d-1 haystack-domain filler "
            "sentences between anchor and referent. Removes the v1 "
            "next-sentence tautology. Registered before any evaluation."
        ),
        "prompts": prompts,
    }


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(base.MODEL_ID)
    ps = build(tokenizer)
    OUT.write_text(yaml.safe_dump(ps, sort_keys=False, allow_unicode=True))
    print(f"wrote {OUT}: {ps['n_prompts']} prompts")
    from collections import Counter
    print("  true-pair distance counts:",
          dict(Counter(p["coref_distance"] for p in ps["prompts"])))
    for depth in DEPTHS:
        for b in BUCKETS + (DEV_BUCKET,):
            got = [p["n_tokens"] for p in ps["prompts"]
                   if p["needle_depth"] == depth and p["bucket"] == b]
            if got:
                print(f"  depth {depth:.2f} bucket {b:>5}: n={len(got)} "
                      f"min={min(got)} max={max(got)}")


if __name__ == "__main__":
    main()
