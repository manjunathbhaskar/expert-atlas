"""Generate probes/probe_set_context_depth.yaml -- needle DEPTH as a planned
factor, for OLMoE.

Needle depth was held fixed at 50% for the entire project and flagged as a
scope limit from the start (`probes/probe_set_context.py` NEEDLE_DEPTH
comment, `docs/CONTEXT_ROT.md`). This set varies it as a PRE-DECLARED factor:

  depths   : 0.15, 0.50, 0.85     (early / middle / late -- registered here,
                                    before any capture, not chosen post hoc)
  buckets  : 256, 3840             (the two endpoints where the accuracy gap
                                    is largest in the hard run: 93.8% -> 68.8%)
  distractors: 8 only              (the arm where context rot appears)
  haystack : similar, dissimilar
  replicates: 8

3 x 2 x 1 x 2 x 8 = 96 prompts. The 0.50/8-distractor cells reproduce the
hard design's corresponding cells (different seed, so different haystack
content -- a consistency check, not a byte-identical rerun).

Distractor depths are recomputed per needle depth so distractors straddle the
needle without colliding (the fixed hard-set list assumes a 50% needle).

Usage:
    HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 \
      .venv/bin/python probes/generate_context_probes_depth.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from transformers import AutoTokenizer

import probes.probe_set_context as base

OUT = Path(__file__).parent / "probe_set_context_depth.yaml"
REPLICATES = 8
SEED = 4
DEPTHS = (0.15, 0.50, 0.85)


def _distractor_depths(needle_depth: float) -> tuple[float, ...]:
    """8 depths spread over (0.05, 0.95), each >= 0.04 away from the needle."""
    raw = [0.05 + 0.90 * i / 7 for i in range(8)]
    out = []
    for d in raw:
        if abs(d - needle_depth) < 0.04:
            d = d + 0.05 if d < needle_depth else d - 0.05
        out.append(round(d, 3))
    return tuple(out)


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(base.MODEL_ID)
    base.DISTRACTOR_LEVELS = (8,)
    base.LENGTH_BUCKETS = (256, 3840)

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

    merged["probe_set_id"] = "probe_set_context_depth"
    merged["needle_depths"] = list(DEPTHS)
    merged["needle_depth"] = None  # varies per prompt; see needle_depths
    merged["n_prompts"] = len(merged["prompts"])
    merged["notes"] = (
        "Needle depth varied as a planned factor (0.15/0.50/0.85), registered "
        "in probes/generate_context_probes_depth.py before capture. Buckets "
        "256 and 3840 only, 8 distractors only."
    )
    OUT.write_text(yaml.safe_dump(merged, sort_keys=False, allow_unicode=True))

    print(f"wrote {OUT}: {merged['n_prompts']} prompts")
    for depth in DEPTHS:
        for b in (256, 3840):
            got = [p for p in merged["prompts"]
                   if p["needle_depth"] == depth and p["bucket"] == b]
            print(f"  depth {depth:.2f} bucket {b:>5}: n={len(got)}")


if __name__ == "__main__":
    main()
