"""Keep-top-K permutation null (handoff Task 3).

`docs/KEEP_TOPK_FAIR_PROBE.md` showed that restricting the router to the
lift-ranked + hot-core keep set costs a lot of loss. But it never asked the
control question: is the lift-ranked selection at least *better than chance*,
i.e. better than a random keep set of exactly the same size? If it is not,
the atlas's lift ranking carries no usable signal for conditional compute,
which is itself a reportable (negative) result.

For each keep fraction this script measures held-out teacher-forced loss for:

- `selected` — top-N%-by-lift for the target domain, union the hot core
  (identical policy and RestrictedGate mechanism to probe_keep_topk_fair.py);
- `random_i` — N seeded uniform random keep sets of EXACTLY the same total
  size, run through the identical mechanism.

Honesty note on the word "null": the project standard for a permutation null
is >=200 permutations. Each random draw here costs a full forward-pass eval
of the 7B model, so this run uses a handful of draws (default 8, per the
handoff's 5-10 guidance) and reports the comparison as DIRECTIONAL, with the
min/max of the random distribution shown rather than a p-value dressed up as
precise.

Usage:
    HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 .venv/bin/python \
        scripts/run_keep_topk_null.py --keep-frac 0.10 0.20 0.30 --n-random 8
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from expertatlas.capture import load_model  # noqa: E402
from scripts.probe_keep_topk_fair import (  # noqa: E402
    N_EXPERTS_PER_LAYER,
    N_LAYERS,
    RestrictedGate,
    by_layer,
    load_hot_core,
    load_lift_ranked,
    load_split_b_prompts,
    mean_nll,
)

ATLAS_PATH = REPO_ROOT / "data" / "atlas.json"
UTIL_PATH = REPO_ROOT / "data" / "utilization.json"
MODEL_ID = "allenai/OLMoE-1B-7B-0924"
OUT_MD = REPO_ROOT / "docs" / "KEEP_TOPK_NULL.md"
OUT_JSON = REPO_ROOT / "data" / "keep_topk_null.json"


def _eval_keep_set(loaded, keep_set, target, control):
    keep_layers = by_layer(keep_set)
    for l in range(N_LAYERS):
        keep_layers.setdefault(l, set())
    with RestrictedGate(loaded.model, keep_layers) as rg:
        n_skipped = len(rg.skipped_layers)
        loss_t = mean_nll(loaded, target)
        loss_c = mean_nll(loaded, control)
    return loss_t, loss_c, n_skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="medicine")
    ap.add_argument("--control", default="cooking")
    ap.add_argument("--keep-frac", type=float, nargs="+", default=[0.10, 0.20, 0.30])
    ap.add_argument("--n-random", type=int, default=8)
    ap.add_argument("--n-prompts", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-md", type=str, default=str(OUT_MD))
    ap.add_argument("--out-json", type=str, default=str(OUT_JSON))
    args = ap.parse_args()

    if not ATLAS_PATH.exists() or not UTIL_PATH.exists():
        raise SystemExit("atlas.json / utilization.json not found -- run Phase 3 analysis first")

    ranked = load_lift_ranked(args.domain)
    hot_core = load_hot_core()
    target = load_split_b_prompts(args.domain, args.n_prompts)
    control = load_split_b_prompts(args.control, args.n_prompts)
    print(f"hot core: {len(hot_core)}; target prompts: {len(target)}, control: {len(control)}", flush=True)

    print(f"loading {MODEL_ID} ...", flush=True)
    loaded = load_model(MODEL_ID, device="cpu", dtype="bfloat16")
    loaded.model.eval()

    n_total = N_LAYERS * N_EXPERTS_PER_LAYER
    all_slots = [(l, i) for l in range(N_LAYERS) for i in range(N_EXPERTS_PER_LAYER)]
    rng = random.Random(args.seed)

    print("condition=baseline ...", flush=True)
    base_t = mean_nll(loaded, target)
    base_c = mean_nll(loaded, control)
    print(f"  loss_on_target={base_t:.4f}  loss_on_control={base_c:.4f}", flush=True)

    results = {"baseline": {"loss_on_target": base_t, "loss_on_control": base_c},
               "fractions": []}

    for frac in args.keep_frac:
        n_top = int(round(frac * n_total))
        top_by_lift = {(l, i) for l, i, _ in ranked[:n_top]}
        selected = top_by_lift | hot_core
        size = len(selected)

        print(f"frac={frac:.0%}: selected set size {size} ({size/n_total:.1%}) ...", flush=True)
        sel_t, sel_c, sel_skip = _eval_keep_set(loaded, selected, target, control)
        print(f"  selected: loss_t={sel_t:.4f} loss_c={sel_c:.4f} skipped_layers={sel_skip}", flush=True)

        rand_rows = []
        for r_i in range(args.n_random):
            rand_set = set(rng.sample(all_slots, size))
            r_t, r_c, r_skip = _eval_keep_set(loaded, rand_set, target, control)
            print(f"  random_{r_i}: loss_t={r_t:.4f} loss_c={r_c:.4f} skipped_layers={r_skip}", flush=True)
            rand_rows.append({"loss_on_target": r_t, "loss_on_control": r_c,
                              "skipped_layers": r_skip})

        rand_t = [r["loss_on_target"] for r in rand_rows]
        n_leq = sum(1 for v in rand_t if v <= sel_t)
        results["fractions"].append({
            "keep_frac": frac,
            "n_kept": size,
            "selected": {"loss_on_target": sel_t, "loss_on_control": sel_c,
                         "skipped_layers": sel_skip},
            "random": rand_rows,
            "random_target_mean": float(np.mean(rand_t)),
            "random_target_min": float(np.min(rand_t)),
            "random_target_max": float(np.max(rand_t)),
            "n_random_leq_selected_target": n_leq,
        })

    Path(args.out_json).write_text(json.dumps(results, indent=2))
    _write_report(results, args)
    print(f"wrote {args.out_md} and {args.out_json}", flush=True)


def _write_report(results, args):
    base = results["baseline"]
    lines = [
        "# Keep-top-K: random-keep-set control (directional null)",
        "",
        "## Limitations (read first)",
        "",
        f"- Only {args.n_random} random draws per fraction (each draw is a full 7B "
        "forward-pass eval), far below the project's >=200-permutation standard. "
        "This is a DIRECTIONAL control, not a calibrated p-value.",
        "- One model, one seed, one domain pair "
        f"(`{args.domain}` vs `{args.control}`), {args.n_prompts} held-out split-B "
        "prompts per domain. ",
        "- Random sets are uniform over all 1,024 expert slots; layers left with "
        "fewer than top_k kept experts are unrestricted (same policy as the fair "
        "probe) and per-condition skipped-layer counts are in the JSON.",
        "",
        f"Baseline loss: target {base['loss_on_target']:.4f}, "
        f"control {base['loss_on_control']:.4f} nats.",
        "",
        "## Results",
        "",
        "| keep frac | n kept | selected loss (target) | random mean | random min..max | # random <= selected |",
        "|---|---|---|---|---|---|",
    ]
    for row in results["fractions"]:
        sel = row["selected"]
        lines.append(
            f"| {row['keep_frac']:.0%} | {row['n_kept']}/1024 | "
            f"{sel['loss_on_target']:.4f} | {row['random_target_mean']:.4f} | "
            f"{row['random_target_min']:.4f}..{row['random_target_max']:.4f} | "
            f"{row['n_random_leq_selected_target']}/{len(row['random'])} |"
        )
    lines += [
        "",
        "## Reading this",
        "",
        "- If the selected (lift + hot-core) set beats every random draw, the atlas "
        "ranking carries real signal for conditional compute even though the absolute "
        "loss cost is high.",
        "- If random draws match or beat it, the lift ranking adds nothing usable at "
        "these fractions -- a negative result to report as such.",
    ]
    Path(args.out_md).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
