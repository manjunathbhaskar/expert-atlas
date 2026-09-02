"""Generate probes/probe_set_context.yaml — needle-in-haystack probes for the
context-rot workstream (WS1).

Modelled on Chroma's "Context Rot: How Increasing Input Tokens Impacts LLM
Performance": task difficulty is held FIXED while input length is the only
varied quantity, with distractor and needle-haystack-similarity conditions.

probe_set_v1.yaml is left completely untouched so the original topic results
stay comparable.

Design decisions and their reasons:

- **Length buckets** stop at 3584 tokens. OLMoE-1B-7B-0924 has
  max_position_embeddings=4096 (checked from the config, not assumed), and
  the question + answer must fit after the haystack, so the top bucket
  leaves headroom. Going past the trained context would confound "context
  rot" with "position embeddings outside training range", which is a
  different phenomenon and would make the result uninterpretable.

- **The needle is identical across every bucket and condition.** Only the
  amount of surrounding filler changes. This is the whole point: if
  accuracy or routing changes, it cannot be because the task got harder.

- **The question is identical across every bucket too**, and always placed
  at the very end. Needle *position* is held at the midpoint rather than
  varied, because varying both position and length at once would confound
  two effects; position is a documented follow-up, not part of this sweep.

- **Three conditions**, mirroring Chroma's manipulations:
    baseline   — needle in topically-unrelated filler
    distractor — needle plus 4 plausible-but-wrong near-miss statements
    similar    — needle in filler drawn from the SAME topic as the needle,
                 so lexical overlap between needle and haystack is high

- **Filler is deterministic and in-repo**: sentences are assembled from a
  fixed pool with a seeded RNG, so the probe set is byte-reproducible and
  contains no machine-generated prose that a human hasn't been able to
  inspect. `probe_set_v1.yaml` already carries a known
  `translation_reviewed: false` risk (TRANSFER.md §4.1); this generator
  deliberately avoids adding a second unreviewed-text risk by using a
  small, readable, hand-written sentence pool instead of LLM-written text.

Usage:
    python probes/generate_context_probes.py
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import yaml

OUT = Path(__file__).parent / "probe_set_context.yaml"

# Token budget per bucket (approximate — realised token counts are measured
# at capture time and recorded, these are targets for filler assembly).
LENGTH_BUCKETS = [256, 512, 768, 1024, 1536, 2048, 3072, 3584]
CONDITIONS = ["baseline", "distractor", "similar"]
REPLICATES = 3  # independent filler samples per (bucket, condition)

# The needle: one specific, unambiguous, checkable fact. Deliberately uses a
# rare token sequence so retrieval can't succeed via prior knowledge.
NEEDLE = "The maintenance code for the Kestrel relay station is 7-ZULU-419."
QUESTION = "\n\nQuestion: What is the maintenance code for the Kestrel relay station?\nAnswer:"
EXPECTED_ANSWER = "7-ZULU-419"

# Near-miss distractors: same surface form, wrong values. These make the task
# require actual retrieval rather than pattern-matching on "code-shaped text".
DISTRACTORS = [
    "The maintenance code for the Osprey relay station is 3-VICTOR-820.",
    "The shutdown code for the Kestrel relay station is 5-TANGO-113.",
    "The maintenance code for the Kestrel weather mast is 9-ECHO-274.",
    "The calibration code for the Kestrel relay station is 2-SIERRA-566.",
]

# Unrelated filler (baseline condition) — mundane, topically distant prose.
FILLER_UNRELATED = [
    "The harbour master recorded the tide levels every morning before dawn.",
    "Rain collected in the gutters along the north side of the warehouse.",
    "She kept a ledger of every delivery that arrived after midnight.",
    "The bakery on Fenn Street opened at five and closed by early afternoon.",
    "Wooden crates were stacked three high against the loading bay wall.",
    "A stray cat had taken to sleeping under the bench near the ferry stop.",
    "The clock in the station hall ran four minutes fast all winter.",
    "Gulls circled the fish market whenever the boats came in.",
    "He repainted the railings every spring, always the same shade of green.",
    "The library kept its older maps in a cabinet behind the reading desk.",
    "Frost formed on the greenhouse panes most nights in early November.",
    "Deliveries of coal arrived by cart twice a week until the road was paved.",
]

# Same-domain filler (similar condition) — technical/station vocabulary that
# lexically overlaps the needle without containing the answer.
FILLER_SIMILAR = [
    "The relay station logs signal strength at fifteen minute intervals.",
    "Maintenance windows for the relay network are scheduled each quarter.",
    "Station technicians record equipment codes in the operations binder.",
    "The Kestrel site runs on backup power during scheduled maintenance.",
    "Relay hardware is inspected before the seasonal weather advisories.",
    "Access to the station requires a current maintenance authorisation.",
    "The operations manual lists procedures for each relay installation.",
    "Signal routing between stations is verified after every maintenance cycle.",
    "Technicians report station faults through the central maintenance desk.",
    "Each relay station keeps a printed record of its service history.",
    "The maintenance schedule rotates across all stations in the network.",
    "Station equipment is labelled according to the standard code format.",
]


def approx_tokens(text: str) -> int:
    """Rough token estimate for filler assembly only. Real token counts are
    measured with the actual tokenizer at capture time and recorded in the
    trace; this is just to hit bucket targets while building the file."""
    return int(len(text.split()) * 1.35)


def build_haystack(target_tokens: int, condition: str, rng: random.Random) -> str:
    pool = FILLER_SIMILAR if condition == "similar" else FILLER_UNRELATED
    sentences: list[str] = []
    running = approx_tokens(NEEDLE) + approx_tokens(QUESTION)

    if condition == "distractor":
        running += sum(approx_tokens(d) for d in DISTRACTORS)

    while running < target_tokens:
        s = rng.choice(pool)
        sentences.append(s)
        running += approx_tokens(s)

    # Needle at the midpoint: position held constant so that LENGTH is the
    # only independent variable (see module docstring).
    mid = len(sentences) // 2
    body = sentences[:mid]

    if condition == "distractor":
        # Distractors spread through the haystack, not clustered next to the
        # needle — clustering would make them trivially easy to discount.
        spread = DISTRACTORS[:]
        rng.shuffle(spread)
        step = max(1, len(body) // (len(spread) + 1))
        for i, d in enumerate(spread):
            insert_at = min(len(body), (i + 1) * step)
            body.insert(insert_at, d)

    body.append(NEEDLE)
    body.extend(sentences[mid:])
    return " ".join(body)


def main() -> None:
    prompts = []
    pid = 0
    for bucket in LENGTH_BUCKETS:
        for condition in CONDITIONS:
            for rep in range(REPLICATES):
                # Seed per cell so the file is byte-reproducible.
                seed = int(hashlib.sha1(f"{bucket}-{condition}-{rep}".encode()).hexdigest()[:8], 16)
                rng = random.Random(seed)
                haystack = build_haystack(bucket, condition, rng)
                text = haystack + QUESTION
                prompts.append({
                    "prompt_id": pid,
                    "text": text,
                    "length_bucket": bucket,
                    "condition": condition,
                    "replicate": rep,
                    "approx_tokens": approx_tokens(text),
                    "expected_answer": EXPECTED_ANSWER,
                    "split": "A" if rep % 2 == 0 else "B",
                })
                pid += 1

    doc = {
        "probe_set_id": "probe_set_context",
        "version": "1.0",
        "purpose": (
            "Needle-in-haystack sweep for the context-rot workstream. Task difficulty is "
            "held FIXED (identical needle, identical question, needle always at haystack "
            "midpoint); input LENGTH is the only independent variable. Modelled on Chroma's "
            "'Context Rot: How Increasing Input Tokens Impacts LLM Performance'."
        ),
        "needle": NEEDLE,
        "question": QUESTION.strip(),
        "expected_answer": EXPECTED_ANSWER,
        "length_buckets": LENGTH_BUCKETS,
        "conditions": {
            "baseline": "needle in topically-unrelated filler",
            "distractor": "needle plus 4 plausible near-miss statements spread through the haystack",
            "similar": "needle in same-domain filler with high lexical overlap",
        },
        "n_prompts": len(prompts),
        "max_position_embeddings": 4096,
        "notes": (
            "Top bucket is 3584, below the model's 4096 limit, so the question and answer fit "
            "without exceeding trained context -- going past it would confound context rot with "
            "out-of-range position embeddings. Needle POSITION is held constant at the midpoint; "
            "varying position as well as length would confound two effects and is a documented "
            "follow-up, not part of this sweep."
        ),
        "prompts": prompts,
    }

    OUT.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100))
    print(f"wrote {OUT} — {len(prompts)} prompts "
          f"({len(LENGTH_BUCKETS)} buckets x {len(CONDITIONS)} conditions x {REPLICATES} replicates)")
    by_bucket = {}
    for p in prompts:
        by_bucket.setdefault(p["length_bucket"], []).append(p["approx_tokens"])
    for b in LENGTH_BUCKETS:
        vals = by_bucket[b]
        print(f"  bucket {b:5d}: approx tokens {min(vals)}-{max(vals)}")


if __name__ == "__main__":
    main()
