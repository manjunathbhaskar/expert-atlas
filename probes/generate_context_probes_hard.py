"""Generate probes/probe_set_context_hard.yaml -- a harder, higher-power
variant of the WS1 context-rot sweep.

Does NOT touch probes/probe_set_context.py's module-level defaults, so the
already-published, reproducible null result (docs/CONTEXT_ROT.md, based on
probes/probe_set_context.yaml, 4 replicates, 0/4 distractors) stays exactly
reproducible via `python probes/probe_set_context.py --replicates 4`.

What's harder, and why:
  - DISTRACTOR_LEVELS (0, 8) instead of (0, 4): doubles the number of
    plausible near-miss answers competing with the correct one in the
    non-trivial condition, directly increasing task difficulty rather than
    changing what's being measured.
  - DISTRACTOR_DEPTHS extended from 4 to 8 positions to actually place all
    8 distractors (the original 4-position list would silently only place
    4 of a larger set via zip() truncation -- checked before relying on it).
  - replicates=8 (vs. 4): doubles per-cell n, the axis the original run's
    own wall-clock accounting explicitly identified as "the one axis where
    less costs only statistical power" -- i.e. the correct lever to pull
    for more power without changing what's being tested.

Everything else (7 length buckets 128-3840, needle depth fixed at 50%,
similar/dissimilar haystack axis, the whole assembly/scoring pipeline) is
identical to the base design, so results stay comparable to Chroma's named
comparisons the same way the original design is.

Usage:
    python probes/generate_context_probes_hard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from transformers import AutoTokenizer

import probes.probe_set_context as base

OUT = Path(__file__).parent / "probe_set_context_hard.yaml"
REPLICATES = 8
SEED = 1  # different from the base file's seed=0, so content is visibly distinct


def main() -> None:
    # Monkeypatch the module's globals rather than editing probe_set_context.py,
    # so this file's own default generation behaviour (used by the already-
    # published result) is completely unaffected.
    base.DISTRACTOR_LEVELS = (0, 8)
    base.DISTRACTOR_DEPTHS = (0.10, 0.22, 0.30, 0.42, 0.58, 0.70, 0.78, 0.90)
    # 128 tokens cannot fit the needle + 8 distractors + question (checked --
    # build_probe_set raises rather than silently truncating). Dropped for
    # this variant; 256 remains as the short end, still satisfying "short to
    # near context limit" (the test suite's own bar is <=256).
    base.LENGTH_BUCKETS = tuple(b for b in base.LENGTH_BUCKETS if b != 128)

    tokenizer = AutoTokenizer.from_pretrained(base.MODEL_ID)
    ps = base.build_probe_set(tokenizer, replicates=REPLICATES, seed=SEED)
    ps["probe_set_id"] = "probe_set_context_hard"
    ps["notes"] = (
        "Harder variant of probe_set_context: 8 distractors (vs 4) in the "
        "non-trivial condition, 8 replicates (vs 4). See "
        "probes/generate_context_probes_hard.py for the exact deltas."
    )

    OUT.write_text(yaml.safe_dump(ps, sort_keys=False, allow_unicode=True))

    tot = sum(p["n_tokens"] for p in ps["prompts"])
    print(f"wrote {OUT}: {ps['n_prompts']} prompts, {tot} tokens total")
    for b in base.LENGTH_BUCKETS:
        got = [p["n_tokens"] for p in ps["prompts"] if p["bucket"] == b]
        print(f"  bucket {b:>5}: n={len(got)} actual tokens min={min(got)} max={max(got)}")


if __name__ == "__main__":
    main()
