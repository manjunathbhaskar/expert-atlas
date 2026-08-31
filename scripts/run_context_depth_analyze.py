"""Needle-depth analysis: does WHERE the needle sits change context rot?

Depths 0.15 / 0.50 / 0.85 were registered in
probes/generate_context_probes_depth.py before capture; buckets 256 and 3840
are the endpoints of the hard-set accuracy gap. This script joins the sweep's
accuracy records (data/context_traces_depth/accuracy.jsonl) to the probe set's
needle_depth field and tests, at each bucket, whether depth affects
forced-choice answer probability (one-way permutation test on the between-depth
spread of means, 2000 label shuffles).

Usage:
    .venv/bin/python scripts/run_context_depth_analyze.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

TRACES = REPO_ROOT / "data" / "context_traces_depth"
PROBE_SET = REPO_ROOT / "probes" / "probe_set_context_depth.yaml"
OUT_JSON = REPO_ROOT / "data" / "context_depth.json"
OUT_MD = REPO_ROOT / "docs" / "CONTEXT_DEPTH.md"
N_PERM = 2000
SEED = 0


def perm_anova(groups: list[np.ndarray], rng: np.random.Generator) -> float:
    """Permutation p for spread of group means (statistic: variance of means)."""
    obs = float(np.var([g.mean() for g in groups]))
    pooled = np.concatenate(groups)
    sizes = [len(g) for g in groups]
    count = 0
    for _ in range(N_PERM):
        rng.shuffle(pooled)
        i, means = 0, []
        for s in sizes:
            means.append(pooled[i:i + s].mean())
            i += s
        if np.var(means) >= obs:
            count += 1
    return count / N_PERM


def main() -> None:
    ps = yaml.safe_load(PROBE_SET.read_text())
    depth_by_pid = {p["prompt_id"]: p["needle_depth"] for p in ps["prompts"]}
    recs = [json.loads(l) for l in
            (TRACES / "accuracy.jsonl").read_text().splitlines()]
    for r in recs:
        r["depth"] = depth_by_pid[r["prompt_id"]]

    rng = np.random.default_rng(SEED)
    results = {"n_prompts": len(recs), "cells": {}, "tests": {}}
    depths = sorted({r["depth"] for r in recs})
    buckets = sorted({r["bucket"] for r in recs})
    for b in buckets:
        groups = []
        for d in depths:
            sub = [r for r in recs if r["bucket"] == b and r["depth"] == d]
            acc = float(np.mean([r["forced_choice_correct"] for r in sub]))
            prob = np.array([r["forced_choice_prob"] for r in sub])
            results["cells"][f"bucket{b}_depth{d}"] = {
                "n": len(sub), "fc_acc": acc, "mean_prob": float(prob.mean()),
            }
            groups.append(prob)
        results["tests"][f"bucket{b}_depth_effect_perm_p"] = perm_anova(groups, rng)

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
