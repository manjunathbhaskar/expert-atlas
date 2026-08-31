"""Generate probes/probe_set_context_granite.yaml -- the hard context-rot
design re-tokenised for Granite-3.0-3B-A800M.

The OLMoE candidate pool is 7/8 multi-token under the Granite tokenizer, which
would silently break the single-logit forced-choice metric. This variant swaps
in a candidate pool verified single-token under BOTH tokenizers (so the same
set could later be run on OLMoE for a like-for-like comparison), and measures
bucket lengths in Granite tokens.

Everything else matches the hard variant: 8 distractors in the non-trivial
arm, 8 replicates, needle at 50% depth, buckets 256-3840.

Usage:
    HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 \
      .venv/bin/python probes/generate_context_probes_granite.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from transformers import AutoTokenizer

import probes.probe_set_context as base

OUT = Path(__file__).parent / "probe_set_context_granite.yaml"
GRANITE_ID = "ibm-granite/granite-3.0-3b-a800m-base"
REPLICATES = 8
SEED = 3

#: verified exactly one token with a leading space under BOTH the Granite and
#: the OLMoE tokenizer (see git history of this file for the candidate scan).
CANDIDATE_WORDS = (
    "silver", "gold", "stone", "river",
    "cloud", "forest", "shadow", "mirror",
)


def main() -> None:
    base.CANDIDATE_WORDS = CANDIDATE_WORDS
    # assert_single_token's `words=CANDIDATE_WORDS` default bound at def time.
    base.assert_single_token.__defaults__ = (CANDIDATE_WORDS,)
    base.DISTRACTOR_LEVELS = (0, 8)
    base.DISTRACTOR_DEPTHS = (0.10, 0.22, 0.30, 0.42, 0.58, 0.70, 0.78, 0.90)
    base.LENGTH_BUCKETS = tuple(b for b in base.LENGTH_BUCKETS if b != 128)

    tokenizer = AutoTokenizer.from_pretrained(GRANITE_ID)
    ps = base.build_probe_set(tokenizer, replicates=REPLICATES, seed=SEED)
    ps["probe_set_id"] = "probe_set_context_granite"
    ps["tokenizer_model_id"] = GRANITE_ID
    ps["candidate_words"] = list(CANDIDATE_WORDS)
    ps["notes"] = (
        "Hard context-rot design re-tokenised for Granite; candidate pool "
        "verified single-token under both Granite and OLMoE tokenizers."
    )
    OUT.write_text(yaml.safe_dump(ps, sort_keys=False, allow_unicode=True))

    tot = sum(p["n_tokens"] for p in ps["prompts"])
    print(f"wrote {OUT}: {ps['n_prompts']} prompts, {tot} tokens total")
    for b in base.LENGTH_BUCKETS:
        got = [p["n_tokens"] for p in ps["prompts"] if p["bucket"] == b]
        print(f"  bucket {b:>5}: n={len(got)} min={min(got)} max={max(got)}")


if __name__ == "__main__":
    main()
