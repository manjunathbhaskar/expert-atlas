"""Real conditional compute: keep only a subset of experts and load them.

Companion to `docs/KEEP_TOPK_FAIR_PROBE.md` and `expertatlas/steering.py`.
Those scripts answer "does *restricting the router choice* preserve output?"
This one answers the next question: "does *removing the non-kept expert
weights* from the forward pass preserve output, and is it faster?"

It replaces each targeted layer's `OlmoeSparseMoeBlock` with an
`expertatlas.offloading.OffloadedSparseMoeBlock` that only owns and computes
the kept experts. The router still operates, but it scores `k` experts
instead of 64, and the FFN loop only touches `k` weight pairs.

Conditions tested (all start from the same split-B held-out prompts as the
keep-top-K probes):

- `baseline` — the full 1,024-expert model.
- `keep_topk_global` — keep the top N% by lift for the target domain *plus*
  the hot core, chosen globally (same policy as
  `docs/KEEP_TOPK_FAIR_PROBE.md`), then actually unload the non-kept FFN
  weights (not just mask the router).
- `keep_topk_quota` — same, but the top N% is chosen *per layer* so every
  layer keeps the same fraction. This is the selection-unit test flagged as
  open in `docs/KEEP_TOPK_FAIR_PROBE.md`.

Metric: mean per-token teacher-forced cross-entropy (nats) on held-out
split-B prompts, plus wall-clock time and estimated FFN FLOPs/bytes.

Memory note
-----------
`expertatlas/offloading.py` can free the original full weights (`reset=True`)
when offloading. This script uses `reset=False` and `restore_model()` between
conditions so one model load can run all conditions, but the *reported* memory
savings are **analytical** (computed from the kept-set sizes), not measured
RSS. A separate process-per-condition run would be needed to measure real RSS.
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from expertatlas.capture import load_model  # noqa: E402
from expertatlas.offloading import (  # noqa: E402
    OffloadedMoe,
    estimate_expert_flops_per_token,
    estimate_expert_memory,
    offloading_savings_summary,
)

ATLAS_PATH = REPO_ROOT / "data" / "atlas.json"
UTIL_PATH = REPO_ROOT / "data" / "utilization.json"
PROBE_SET_PATH = REPO_ROOT / "probes" / "probe_set_v1.yaml"
OUT_MD = REPO_ROOT / "docs" / "OFFLOADING.md"
OUT_JSON = REPO_ROOT / "data" / "offloading.json"

MODEL_ID = "allenai/OLMoE-1B-7B-0924"
N_LAYERS, N_EXPERTS, TOP_K = 16, 64, 8
HOT_THRESHOLD = 2.0


def _mean_nll(loaded, prompts: list[str]) -> float:
    losses = []
    for text in prompts:
        inputs = loaded.tokenizer(text, return_tensors="pt").to(str(loaded.model.device))
        input_ids = inputs["input_ids"]
        if input_ids.shape[1] < 2:
            continue
        with torch.no_grad():
            # output_router_logits=False: OffloadedTopKRouter is not an
            # OlmoeTopKRouter, so the recorder collects nothing and the aux
            # load-balancing loss would crash on an empty tuple.
            out = loaded.model(**inputs, labels=input_ids, output_router_logits=False)
        losses.append(float(out.loss))
    return float(np.mean(losses)) if losses else float("nan")


def _load_prompts(domain: str, n: int) -> list[str]:
    ps = yaml.safe_load(PROBE_SET_PATH.read_text())
    return [p["text"] for p in ps["prompts"]
            if p["topic"] == domain and p.get("split") == "B"][:n]


def _hot_core() -> set[tuple[int, int]]:
    util = json.loads(UTIL_PATH.read_text())
    load_ratio = util["utilization"]["load_ratio"]
    uids = util["utilization"]["uids"]
    order = {u: i for i, u in enumerate(uids)}
    hot = set()
    for i, lr in enumerate(load_ratio):
        if lr >= HOT_THRESHOLD:
            layer, idx = divmod(i, N_EXPERTS)
            hot.add((layer, idx))
    return hot


def _lift_ranked(domain: str) -> list[tuple[int, int, float]]:
    atlas = json.loads(ATLAS_PATH.read_text())
    out = []
    for e in atlas["experts"]:
        lift = e.get("lift", {}).get(domain)
        if lift is not None:
            out.append((e["layer"], e["idx"], lift))
    out.sort(key=lambda x: x[2], reverse=True)
    return out


def _global_keep_set(domain: str, frac: float, hot_core: set) -> dict[int, set[int]]:
    ranked = _lift_ranked(domain)
    n_total = N_LAYERS * N_EXPERTS
    n_top = int(round(frac * n_total))
    top_by_lift = {(l, i) for l, i, _ in ranked[:n_top]}
    keep = top_by_lift | hot_core
    by_layer: dict[int, set[int]] = {l: set() for l in range(N_LAYERS)}
    for l, i in keep:
        by_layer[l].add(i)
    return by_layer


def _per_layer_quota_keep_set(domain: str, frac: float, hot_core: set) -> dict[int, set[int]]:
    """Top frac% within each layer, guaranteed equal fraction per layer, plus
    the hot core. If adding the hot core makes a layer exceed N_EXPERTS, the
    hot-core experts for that layer are still kept (they are the generalist
    safety net), but this is reported as a cap."""
    ranked = _lift_ranked(domain)
    by_layer_full: dict[int, list[tuple[int, float]]] = {l: [] for l in range(N_LAYERS)}
    for l, i, lift in ranked:
        by_layer_full[l].append((i, lift))
    n_per_layer = max(1, int(round(frac * N_EXPERTS)))
    by_layer: dict[int, set[int]] = {l: set() for l in range(N_LAYERS)}
    for l in range(N_LAYERS):
        top = sorted(by_layer_full[l], key=lambda x: x[1], reverse=True)[:n_per_layer]
        by_layer[l] = {i for i, _ in top}
    # Add hot core, merging but capping at N_EXPERTS to keep top_k valid.
    for l, i in hot_core:
        by_layer[l].add(i)
        if len(by_layer[l]) > N_EXPERTS:
            by_layer[l] = set(sorted(by_layer[l])[:N_EXPERTS])
    return by_layer


def _one_condition(loaded, label: str, keep_by_layer: dict, target: list[str], control: list[str]) -> dict:
    t0 = time.time()
    with OffloadedMoe(loaded.model, keep_by_layer, reset=False):
        loss_t = _mean_nll(loaded, target)
        loss_c = _mean_nll(loaded, control)
    wall = time.time() - t0

    config = loaded.model.config
    est = offloading_savings_summary(keep_by_layer, config)
    # Total kept across all layers (not just restricted ones).
    total_kept = sum(len(s) for s in keep_by_layer.values())
    total_full = N_LAYERS * N_EXPERTS
    return {
        "condition": label,
        "n_kept": total_kept,
        "kept_frac": total_kept / total_full,
        "loss_on_target": round(loss_t, 4),
        "loss_on_control": round(loss_c, 4),
        "delta_target": round(loss_t - loss_t, 4),
        "delta_control": round(loss_c - loss_c, 4),
        "wall_seconds": round(wall, 1),
        "est_ffn_bytes_frac": round(est["ffn_bytes_frac"], 3),
        "est_ffn_flops_frac": round(
            sum(estimate_expert_flops_per_token(len(keep_by_layer.get(l, set())),
                                                 config.hidden_size, config.intermediate_size, TOP_K)
                for l in range(N_LAYERS))
            / sum(estimate_expert_flops_per_token(N_EXPERTS, config.hidden_size,
                                                  config.intermediate_size, TOP_K)
                  for l in range(N_LAYERS)),
            3,
        ),
    }


def _peak_rss_mb() -> float:
    # macOS ru_maxrss is in bytes; Linux is in kilobytes. Normalize to MB.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024.0 * 1024.0) if rss > 1e9 else rss / 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="medicine")
    ap.add_argument("--control", default="cooking")
    ap.add_argument("--frac", type=float, nargs="+", default=[0.20, 0.30])
    ap.add_argument("--n-prompts", type=int, default=6)
    ap.add_argument("--no-baseline", action="store_true", help="skip the full baseline (faster for repeated tests)")
    args = ap.parse_args()

    target = _load_prompts(args.domain, args.n_prompts)
    control = _load_prompts(args.control, args.n_prompts)
    if not target or not control:
        raise SystemExit(f"not enough split-B prompts for {args.domain}/{args.control}")

    hot_core = _hot_core()
    print(f"hot core: {len(hot_core)} experts (load_ratio >= {HOT_THRESHOLD})", flush=True)

    print(f"loading {MODEL_ID} ...", flush=True)
    t0 = time.time()
    loaded = load_model(MODEL_ID, device="cpu", dtype="bfloat16")
    print(f"loaded in {time.time() - t0:.1f}s (peak RSS { _peak_rss_mb():.0f} MB)", flush=True)
    loaded.model.eval()

    results = []
    if not args.no_baseline:
        print("condition=baseline ...", flush=True)
        b_t0 = time.time()
        base_t = _mean_nll(loaded, target)
        base_c = _mean_nll(loaded, control)
        results.append({
            "condition": "baseline",
            "n_kept": N_LAYERS * N_EXPERTS,
            "kept_frac": 1.0,
            "loss_on_target": round(base_t, 4),
            "loss_on_control": round(base_c, 4),
            "delta_target": 0.0,
            "delta_control": 0.0,
            "wall_seconds": round(time.time() - b_t0, 1),
            "est_ffn_bytes_frac": 1.0,
            "est_ffn_flops_frac": 1.0,
        })

    for frac in args.frac:
        for kind, fn in (("global", _global_keep_set), ("per_layer_quota", _per_layer_quota_keep_set)):
            label = f"keep_{int(100*frac):02d}_{kind}"
            print(f"condition={label} ...", flush=True)
            keep = fn(args.domain, frac, hot_core)
            r = _one_condition(loaded, label, keep, target, control)
            # Fill in deltas against the baseline (if no baseline, stay 0).
            if results:
                r["delta_target"] = round(r["loss_on_target"] - results[0]["loss_on_target"], 4)
                r["delta_control"] = round(r["loss_on_control"] - results[0]["loss_on_control"], 4)
            results.append(r)
            print(f"  {r}", flush=True)

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    _write_report(results, args.domain, args.control, hot_core)
    print(f"wrote {OUT_MD} and {OUT_JSON}", flush=True)


def _write_report(results, domain, control, hot_core):
    base = results[0] if results and results[0]["condition"] == "baseline" else None
    lines = [
        "# Offloading baseline: real conditional compute",
        "",
        f"Target domain: `{domain}`. Control domain: `{control}`. "
        "Metric: mean per-token teacher-forced cross-entropy (nats), held-out (split=B) prompts. "
        f"Hot core: {len(hot_core)} experts.",
        "",
        "This is the next step after `docs/KEEP_TOPK_FAIR_PROBE.md`: that document "
        "showed that *masking the router* to a kept set is still not enough to "
        "preserve accuracy. This document tests whether the same keep-sets, when "
        "realised as an actually smaller FFN (non-kept experts are not loaded and "
        "not computed), perform the same, better, or worse.",
        "",
        "## Method caveats",
        "",
        "- The full 13GB checkpoint is still loaded once from disk. The savings here "
        "are the **runtime FFN parameters** that the offloaded block keeps in memory "
        "and the **FFN matmuls** it executes, not the initial disk read. A state-dict "
        "filter that only deserialises the kept experts would be the genuine "
        "'do not load the whole model' step; that is not implemented yet.",
        "- Offloading is applied to **all 16 layers** with the same keep fraction. "
        "A layer-wise or token-adaptive policy is possible but not tested.",
        "- Wall-clock times include the tokenisation and teacher-forced forward pass; "
        "they are rough and should not be over-interpreted for a single run.",
        "",
        "## Results",
        "",
        "| condition | n kept | kept frac | loss on target | delta | loss on control | delta | wall (s) | FFN bytes frac | FFN FLOPs frac |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['condition']} | {r['n_kept']}/{N_LAYERS*N_EXPERTS} | {r['kept_frac']:.1%} | "
            f"{r['loss_on_target']:.4f} | {r['delta_target']:+.4f} | "
            f"{r['loss_on_control']:.4f} | {r['delta_control']:+.4f} | "
            f"{r['wall_seconds']:.1f} | {r['est_ffn_bytes_frac']:.3f} | {r['est_ffn_flops_frac']:.3f} |"
        )
    lines += [
        "",
        "## Reading this",
        "",
        "- If the `per_layer_quota` condition is closer to baseline than `global`, "
        "that supports the 'selection unit, not mechanism' hypothesis from "
        "`docs/KEEP_TOPK_FAIR_PROBE.md`.",
        "- If even the offloaded versions lose accuracy badly, the keep-top-K idea "
        "is likely limited by *which* experts are needed, not by whether the router "
        "was free to choose among them.",
        "- Wall-clock speedup should track `FFN FLOPs frac` if the offloaded block "
        "is the bottleneck; if it does not, measurement noise or non-FFN overheads "
        "dominate.",
        "",
        "## Honest limits",
        "",
        f"- n={len(results)} conditions, one model, one seed, one domain pair, "
        "split-B prompts only. Directional.",
        "- No measured RSS; the byte fraction is an upper-bound estimate from the "
        "kept-set sizes, not an observed memory footprint.",
        "- No random-draw null for the keep-sets. A 'top-by-lift beats random' claim "
        "is not supported by this run alone.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
