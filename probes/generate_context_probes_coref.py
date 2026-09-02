"""Generate probes/probe_set_context_coref.yaml -- coreference variant where
the answer sentence shares ZERO contentful tokens with the question.

The needle is a two-sentence coreference pair:

  anchor : "The {entity} office is run by a single site director."
  needle : "Her personal passphrase, {word}, stays locked in the drawer."

The question is the standard one ("What is the security codeword for the
{entity} office?"). The needle sentence shares no contentful token with the
question by construction -- not "security", not "codeword", not "office",
not the entity. The only route to it is resolving "Her" back to the
adjacent anchor sentence (which carries the entity token). This is the
open coreference case flagged in the manuscript conclusion: does it behave
like paraphrase (one semantic hop) or like multi-hop (a chained walk)?

Distractors are matched anchor+coref pairs for other entities/words, so
competition happens in the same two-sentence form as the needle. Registered
expectation (before any evaluation): the single-stage lexical/IDF detector
finds the anchor at best and misses the answer sentence, because the answer
sentence has zero question-token overlap.

Design mirrors the variants set: depths 0.15/0.50, buckets 256 and 3840,
4 distractor pairs (two-sentence pairs are ~2x longer than single-sentence distractors, so 4 pairs match the 256 bucket), 8 replicates, similar+dissimilar haystacks
=> 2 depths x 2 buckets x 2 haystacks x 8 = 64 prompts, plus a 16-prompt
DEV arm (1024 tokens, depth 0.15) for detector calibration only.

Usage:
    HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 \
      .venv/bin/python probes/generate_context_probes_coref.py
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

OUT = Path(__file__).parent / "probe_set_context_coref.yaml"
REPLICATES = 8
SEED = 11
DEPTHS = (0.15, 0.50)
BUCKETS = (256, 3840)
DEV_BUCKET = 1024
N_DISTRACTORS = 4

ANCHOR = "The {entity} office is run by a single site director."
COREF_NEEDLE = ("Her personal passphrase, {word}, stays locked in the "
                "drawer.")


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

        pair = (ANCHOR.format(entity=entity) + " "
                + COREF_NEEDLE.format(word=answer_word))
        distractor_pool = [
            ANCHOR.format(entity=other_e[i % len(other_e)]) + " "
            + COREF_NEEDLE.format(word=other_w[i % len(other_w)])
            for i in range(N_DISTRACTORS)]

        for depth in DEPTHS:
            d_depths = _distractor_depths(depth)
            for haystack in base.HAYSTACK_DOMAINS:
                stream = base._generate_haystack_stream(
                    haystack, random.Random(rng.randrange(1 << 30)), 420)
                for bucket in BUCKETS + ((DEV_BUCKET,)
                                             if depth == 0.15 else ()):
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
                        "prompt_id": pid, "text": text, "variant": "coref",
                        "bucket": bucket, "n_tokens": t,
                        "haystack": haystack,
                        "n_distractors": N_DISTRACTORS,
                        "replicate": rep, "entity": entity,
                        "answer_word": answer_word,
                        "needle_depth": depth,
                        "dev": bucket == DEV_BUCKET,
                        "needle_char_span": [n_s, n_s + len(needle_sent)],
                        "anchor_char_span": [pair_s,
                                             pair_s + len(anchor_sent)],
                        "question_char_span": list(qspan),
                        "sha1": hashlib.sha1(text.encode()).hexdigest()[:12],
                    })
                    pid += 1

    return {
        "probe_set_id": "probe_set_context_coref",
        "tokenizer_model_id": base.MODEL_ID,
        "variants": ["coref"],
        "length_buckets": list(BUCKETS), "dev_bucket": DEV_BUCKET,
        "needle_depths": list(DEPTHS),
        "candidate_words": list(base.CANDIDATE_WORDS),
        "chance_accuracy": 1.0 / len(base.CANDIDATE_WORDS),
        "n_replicates": REPLICATES, "seed": SEED,
        "n_prompts": len(prompts),
        "notes": (
            "Coreference variant: the answer sentence shares zero "
            "contentful tokens with the question; only a pronoun link to "
            "the adjacent anchor sentence (which carries the entity) "
            "identifies it. Registered before any evaluation."
        ),
        "prompts": prompts,
    }


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(base.MODEL_ID)
    ps = build(tokenizer)
    OUT.write_text(yaml.safe_dump(ps, sort_keys=False, allow_unicode=True))
    print(f"wrote {OUT}: {ps['n_prompts']} prompts")
    for depth in DEPTHS:
        for b in BUCKETS + (DEV_BUCKET,):
            got = [p["n_tokens"] for p in ps["prompts"]
                   if p["needle_depth"] == depth and p["bucket"] == b]
            if got:
                print(f"  coref depth {depth:.2f} bucket {b:>5}: "
                      f"n={len(got)} min={min(got)} max={max(got)}")


if __name__ == "__main__":
    main()
