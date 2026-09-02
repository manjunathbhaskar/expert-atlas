"""Generate probes/probe_set_context_distance.yaml -- ZERO-distractor set that
varies needle-to-readout DISTANCE, for the distance-only trigger test.

Motivation. Every failure this project has explained was distractor-driven:
the OLMoE hard set (docs/CONTEXT_ROT_HARD.md) rots only in the 8-distractor
arm (the 0-distractor arm is 1.000 at every length), and Granite required a
24-distractor escalation to fail at all. Published work (positional-distance
degradation with clean haystacks) reports degradation with NO distractors.
This set asks whether the identified retrieval heads' needle attention also
declines under pure distance -- no distractors, only the gap between the
needle and the readout position varies.

Design (registered here, before any capture):

  distractors : 0 only              (the whole point)
  depths      : 0.15, 0.50, 0.85    (varies distance at fixed length)
  buckets     : 256, 1024, 2048, 3840  (varies distance at fixed depth)
  haystack    : similar, dissimilar
  replicates  : 8

3 x 4 x 2 x 8 = 192 prompts. Distance per prompt = n_tokens minus the
needle's end token index (computed at capture time from the token span).

Distractor depths are irrelevant (0 distractors) but NEEDLE_DEPTH is varied
exactly as in generate_context_probes_depth.py.

Usage:
    HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 \
      .venv/bin/python probes/generate_context_probes_distance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from transformers import AutoTokenizer

import probes.probe_set_context as base

OUT = Path(__file__).parent / "probe_set_context_distance.yaml"
REPLICATES = 8
SEED = 9
DEPTHS = (0.15, 0.50, 0.85)
BUCKETS = (256, 1024, 2048, 3840)


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(base.MODEL_ID)
    base.DISTRACTOR_LEVELS = (0,)
    base.LENGTH_BUCKETS = BUCKETS

    merged = None
    for di, depth in enumerate(DEPTHS):
        base.NEEDLE_DEPTH = depth
        ps = base.build_probe_set(tokenizer, replicates=REPLICATES,
                                  seed=SEED * 1000 + di)
        for p in ps["prompts"]:
            p["prompt_id"] = di * 10000 + p["prompt_id"]
            p["needle_depth"] = depth
        if merged is None:
            merged = ps
            merged["prompts"] = list(ps["prompts"])
        else:
            merged["prompts"].extend(ps["prompts"])

    merged["probe_set_id"] = "probe_set_context_distance"
    merged["needle_depths"] = list(DEPTHS)
    merged["needle_depth"] = None  # varies per prompt; see needle_depths
    merged["n_prompts"] = len(merged["prompts"])
    merged["notes"] = (
        "Zero-distractor set varying needle-to-readout distance (depth x "
        "length), registered in probes/generate_context_probes_distance.py "
        "before capture. For the distance-only retrieval-head test."
    )
    OUT.write_text(yaml.safe_dump(merged, sort_keys=False, allow_unicode=True))

    print(f"wrote {OUT}: {merged['n_prompts']} prompts")
    for depth in DEPTHS:
        for b in BUCKETS:
            got = [p for p in merged["prompts"]
                   if p["needle_depth"] == depth and p["bucket"] == b]
            print(f"  depth {depth:.2f} bucket {b:>5}: n={len(got)}")


if __name__ == "__main__":
    main()
