"""Generate probes/probe_set_context_granite_depth.yaml -- needle depth as a
planned factor, for Granite-3.0-3B-A800M.

Granite does not context-rot on the 50%-depth substrate at any tested length
(1.000 accuracy at >=1024, docs/MECHANISM_GRANITE.md), so the head-level
mechanism replication needs a harder substrate with real failing prompts.
OLMoE's depth sweep (docs/CONTEXT_DEPTH.md) showed early needles are the
hardest condition; this set applies the same pre-declared depth factor to the
Granite design:

  depths     : 0.15, 0.50, 0.85   (registered here, before any capture)
  buckets    : 256, 3840
  distractors: 8 only
  haystack   : similar, dissimilar
  replicates : 8

3 x 2 x 2 x 8 = 96 prompts, plus a 16-prompt DEV arm (depth 0.15, bucket
1024, prompt_id offset 90000) used ONLY for beta / detector-width calibration
-- it never enters any evaluation. Candidate pool = the Granite-verified pool
(single-token under both tokenizers). Distractor depths recomputed per needle
depth, as in the OLMoE depth generator.

Usage:
    HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 \
      .venv/bin/python probes/generate_context_probes_granite_depth.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from transformers import AutoTokenizer

import probes.probe_set_context as base
from probes.generate_context_probes_depth import _distractor_depths
from probes.generate_context_probes_granite import CANDIDATE_WORDS, GRANITE_ID

OUT = Path(__file__).parent / "probe_set_context_granite_depth.yaml"
REPLICATES = 8
SEED = 6
DEPTHS = (0.15, 0.50, 0.85)


def main() -> None:
    base.CANDIDATE_WORDS = CANDIDATE_WORDS
    base.assert_single_token.__defaults__ = (CANDIDATE_WORDS,)
    base.DISTRACTOR_LEVELS = (8,)
    base.LENGTH_BUCKETS = (256, 3840)

    tokenizer = AutoTokenizer.from_pretrained(GRANITE_ID)
    merged = None
    for di, depth in enumerate(DEPTHS):
        base.NEEDLE_DEPTH = depth
        base.DISTRACTOR_DEPTHS = _distractor_depths(depth)
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

    # DEV arm: depth 0.15 at 1024 tokens, calibration only
    base.NEEDLE_DEPTH = 0.15
    base.DISTRACTOR_DEPTHS = _distractor_depths(0.15)
    base.LENGTH_BUCKETS = (1024,)
    dev = base.build_probe_set(tokenizer, replicates=REPLICATES,
                               seed=SEED * 1000 + 9)
    for p in dev["prompts"]:
        p["prompt_id"] = 90000 + p["prompt_id"]
        p["needle_depth"] = 0.15
        p["dev"] = True
    merged["prompts"].extend(dev["prompts"])

    merged["probe_set_id"] = "probe_set_context_granite_depth"
    merged["tokenizer_model_id"] = GRANITE_ID
    merged["candidate_words"] = list(CANDIDATE_WORDS)
    merged["needle_depths"] = list(DEPTHS)
    merged["needle_depth"] = None
    merged["n_prompts"] = len(merged["prompts"])
    merged["notes"] = (
        "Granite-tokenised depth-factor set (0.15/0.50/0.85), buckets 256 and "
        "3840, 8 distractors; registered before capture in "
        "probes/generate_context_probes_granite_depth.py."
    )
    OUT.write_text(yaml.safe_dump(merged, sort_keys=False, allow_unicode=True))

    print(f"wrote {OUT}: {merged['n_prompts']} prompts")
    got = [p["n_tokens"] for p in merged["prompts"] if p.get("dev")]
    print(f"  dev arm (depth 0.15 bucket 1024): n={len(got)} "
          f"min={min(got)} max={max(got)}")
    for depth in DEPTHS:
        for b in (256, 3840):
            got = [p["n_tokens"] for p in merged["prompts"]
                   if p["needle_depth"] == depth and p["bucket"] == b]
            print(f"  depth {depth:.2f} bucket {b:>5}: n={len(got)} "
                  f"min={min(got)} max={max(got)}")


if __name__ == "__main__":
    main()
