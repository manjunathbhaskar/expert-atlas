"""Generate dense_track/probe_set_dense.yaml -- the OLMoE context-rot
substrate rebuilt for Pythia-2.8B's tokenizer and 2048-token context limit.

Design is held as close as possible to probe_set_context_repaired.yaml
(the substrate every OLMoE attention-transport result runs on): same needle
and question templates, same haystack generators, same 8 forced-choice
candidate words (verified single-token under the GPT-NeoX tokenizer), same
2 haystack x 2 distractor-level (0 vs 8) crossing, needle at 50% depth,
byte-identical needle/question across buckets within a replicate.

The one forced difference: Pythia-2.8B has max_position_embeddings=2048, so
the length buckets are (256, 1024, 1900) instead of (256, 1024, 3840). The
long bucket sits at 93% of the model's positional range, matching the OLMoE
long bucket's 94% of its 4096 range.

16 replicates x 2 haystacks x 2 distractor levels x 3 buckets = 192 prompts,
matching the OLMoE repaired substrate's budget exactly.

Usage:
    .venv-dense/bin/python dense_track/generate_probes.py
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

OUT = Path(__file__).parent / "probe_set_dense.yaml"
SEED = 7  # distinct from base (0), hard (1), repaired (2)
LENGTH_BUCKETS = (256, 1024, 1900)
DISTRACTOR_LEVELS = (0, 8)
DISTRACTOR_DEPTHS = (0.10, 0.22, 0.30, 0.42, 0.58, 0.70, 0.78, 0.90)


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base.assert_single_token(tokenizer)

    def n_tok(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    prompts = []
    pid = 0
    for rep in range(16):
        rng = random.Random(SEED * 100003 + rep)
        entity = base.ENTITIES[rep % len(base.ENTITIES)]
        answer_word = base.CANDIDATE_WORDS[rep % len(base.CANDIDATE_WORDS)]
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
                        "entity": entity, "answer_word": answer_word,
                        "needle_char_span": list(nspan),
                        "question_char_span": list(qspan),
                        "sha1": hashlib.sha1(text.encode()).hexdigest()[:12],
                    })
                    pid += 1

    ps = {
        "probe_set_id": "probe_set_dense",
        "version": "1.0",
        "design": ("OLMoE context-rot substrate rebuilt for Pythia-2.8B: "
                   "needle-in-haystack, task fixed, length varied, buckets "
                   "capped by the model's 2048-token positional range"),
        "tokenizer_model_id": MODEL_ID,
        "length_buckets": list(LENGTH_BUCKETS),
        "distractor_levels": list(DISTRACTOR_LEVELS),
        "haystack_domains": list(base.HAYSTACK_DOMAINS),
        "needle_depth": base.NEEDLE_DEPTH,
        "distractor_depths": list(DISTRACTOR_DEPTHS),
        "candidate_words": list(base.CANDIDATE_WORDS),
        "chance_accuracy": 1.0 / len(base.CANDIDATE_WORDS),
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
