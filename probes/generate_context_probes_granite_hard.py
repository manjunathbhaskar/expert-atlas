"""Generate probes/probe_set_context_granite_hard.yaml -- the registered
ESCALATION after the Granite depth set failed to produce failing prompts.

Measured constraint this responds to (data/granite_transport/records.jsonl):
Granite-3.0-3B-A800M is at 1.000 forced-choice accuracy in the 3840 bucket at
every needle depth (0.15/0.50/0.85), so the depth factor that breaks OLMoE
does not break Granite, and the model's 4096-token limit rules out going
longer. The one fair intensification left inside the same task semantics is
DISTRACTOR LOAD: 24 distinct competing distractors (same template as the
needle, all with wrong entities AND wrong answer words drawn from the
forced-choice pool -- unambiguous by construction, no conflicting fact about
the needle's own entity).

  depth      : 0.15 (hardest for OLMoE)
  bucket     : 3840
  distractors: 24 distinct (entity, word) pairs, depths spread 0.05-0.95
  haystack   : similar, dissimilar
  replicates : 8            => 16 prompts

If Granite still does not fail on this set, that is the finding: the
head-level mechanism replication is not testable on Granite at <= 4096 tokens
because Granite does not exhibit context rot there, and it is documented as a
negative, not escalated further into ambiguity.

Usage:
    HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 \
      .venv/bin/python probes/generate_context_probes_granite_hard.py
"""

from __future__ import annotations

import hashlib
import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from transformers import AutoTokenizer

import probes.probe_set_context as base
from probes.generate_context_probes_granite import CANDIDATE_WORDS, GRANITE_ID
from probes.generate_context_probes_variants import _assemble

OUT = Path(__file__).parent / "probe_set_context_granite_hard.yaml"
REPLICATES = 8
SEED = 8
NEEDLE_DEPTH = 0.15
BUCKET = 3840
N_DISTRACTORS = 24


def _depths(n: int, needle: float) -> list[float]:
    raw = [0.05 + 0.90 * i / (n - 1) for i in range(n)]
    return [round(d + 0.05 if abs(d - needle) < 0.03 and d < needle
                  else d - 0.05 if abs(d - needle) < 0.03 else d, 3)
            for d in raw]


def main() -> None:
    base.CANDIDATE_WORDS = CANDIDATE_WORDS
    base.assert_single_token.__defaults__ = (CANDIDATE_WORDS,)
    tokenizer = AutoTokenizer.from_pretrained(GRANITE_ID)
    base.assert_single_token(tokenizer)

    def n_tok(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    prompts = []
    pid = 0
    for rep in range(REPLICATES):
        rng = random.Random(SEED * 100003 + rep)
        entity = base.ENTITIES[rep % len(base.ENTITIES)]
        answer_word = CANDIDATE_WORDS[rep % len(CANDIDATE_WORDS)]
        needle = base.NEEDLE_TEMPLATE.format(entity=entity, word=answer_word)
        other_e = [e for e in base.ENTITIES if e != entity]
        other_w = [w for w in CANDIDATE_WORDS if w != answer_word]
        pairs = list(itertools.product(other_e, other_w))
        rng.shuffle(pairs)
        distractors = [base.NEEDLE_TEMPLATE.format(entity=e, word=w)
                       for e, w in pairs[:N_DISTRACTORS]]
        d_depths = _depths(N_DISTRACTORS, NEEDLE_DEPTH)

        for haystack in base.HAYSTACK_DOMAINS:
            stream = base._generate_haystack_stream(
                haystack, random.Random(rng.randrange(1 << 30)), 420)
            inserts = [(NEEDLE_DEPTH, needle)] + list(zip(d_depths, distractors))
            lo, hi, best = 0, len(stream), None
            while lo <= hi:
                mid = (lo + hi) // 2
                text, spans, qspan = _assemble(stream[:mid], inserts, entity)
                t = n_tok(text)
                if t <= BUCKET:
                    best = (text, spans, qspan, t)
                    lo = mid + 1
                else:
                    hi = mid - 1
            text, spans, qspan, t = best
            prompts.append({
                "prompt_id": pid, "text": text, "bucket": BUCKET,
                "n_tokens": t, "haystack": haystack,
                "n_distractors": N_DISTRACTORS, "replicate": rep,
                "entity": entity, "answer_word": answer_word,
                "needle_depth": NEEDLE_DEPTH,
                "needle_char_span": list(spans[0]),
                "question_char_span": list(qspan),
                "sha1": hashlib.sha1(text.encode()).hexdigest()[:12],
            })
            pid += 1

    ps = {
        "probe_set_id": "probe_set_context_granite_hard",
        "tokenizer_model_id": GRANITE_ID,
        "candidate_words": list(CANDIDATE_WORDS),
        "chance_accuracy": 1.0 / len(CANDIDATE_WORDS),
        "needle_depth": NEEDLE_DEPTH, "bucket": BUCKET,
        "n_distractors": N_DISTRACTORS,
        "n_replicates": REPLICATES, "seed": SEED,
        "n_prompts": len(prompts),
        "notes": "Registered escalation: 24-distractor load at depth 0.15 / "
                 "3840 tokens after Granite stayed at ceiling on the depth "
                 "set. See module docstring.",
        "prompts": prompts,
    }
    OUT.write_text(yaml.safe_dump(ps, sort_keys=False, allow_unicode=True))
    got = [p["n_tokens"] for p in prompts]
    print(f"wrote {OUT}: {len(prompts)} prompts "
          f"min={min(got)} max={max(got)}")


if __name__ == "__main__":
    main()
