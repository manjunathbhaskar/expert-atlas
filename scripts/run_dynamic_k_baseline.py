"""Dynamic top-k benchmark: adaptively compute fewer FFN experts per token.

This is the per-token, adaptive counterpart to `scripts/run_offloading_baseline.py`.
`OffloadedMoe` keeps the same fixed subset for every token; `DynamicKMoe`
keeps a different `k_t` per token, determined by the router's own softmax mass.

The current implementation measures output quality (teacher-forced NLL) and the
**average number of FFN experts actually executed per token**. Wall-clock time
is reported but is NOT expected to improve with this Python-loop prototype; the
FFN matmuls are still small on CPU and the per-token Python overhead dominates.
The value is in the FLOP count and the proof that variable `k` does not by
itself collapse output.

Usage (after the causal run frees the model):
    HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 .venv/bin/python \
        scripts/run_dynamic_k_baseline.py --mass-threshold 0.90 0.95 0.99
"""

from __future__ import annotations

import argparse
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
from expertatlas.dynamic_k import DynamicKMoe  # noqa: E402

PROBE_SET_PATH = REPO_ROOT / "probes" / "probe_set_v1.yaml"
OUT_MD = REPO_ROOT / "docs" / "DYNAMIC_K.md"
OUT_JSON = REPO_ROOT / "data" / "dynamic_k.json"
OUT_MD_REL = REPO_ROOT / "docs" / "DYNAMIC_K_RELATIVE.md"
OUT_JSON_REL = REPO_ROOT / "data" / "dynamic_k_relative.json"

MODEL_ID = "allenai/OLMoE-1B-7B-0924"


def _mean_nll_and_kept(loaded, prompts: list[str]) -> tuple[float, float]:
    losses = []
    kept_counts = []
    for text in prompts:
        inputs = loaded.tokenizer(text, return_tensors="pt").to(str(loaded.model.device))
        input_ids = inputs["input_ids"]
        if input_ids.shape[1] < 2:
            continue
        with torch.no_grad():
            mlp = loaded.model.model.layers[0].mlp  # sample layer 0
            out = loaded.model(**inputs, labels=input_ids)
            if hasattr(mlp, "_last_kept"):
                kept_counts.extend(mlp._last_kept)
        losses.append(float(out.loss))
    mean_kept = float(np.mean(kept_counts)) if kept_counts else float("nan")
    return float(np.mean(losses)) if losses else float("nan"), mean_kept


def _load_prompts(domain: str, n: int) -> list[str]:
    ps = yaml.safe_load(PROBE_SET_PATH.read_text())
    return [p["text"] for p in ps["prompts"]
            if p["topic"] == domain and p.get("split") == "B"][:n]


def _peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024.0 * 1024.0) if rss > 1e9 else rss / 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="medicine")
    ap.add_argument("--control", default="cooking")
    ap.add_argument("--mass-threshold", type=float, nargs="+", default=[0.90, 0.95, 0.99])
    ap.add_argument("--n-prompts", type=int, default=6)
    ap.add_argument("--relative", action="store_true",
                    help="interpret thresholds relative to each token's total "
                         "top-k mass (reachable on norm_topk_prob=False models); "
                         "writes DYNAMIC_K_RELATIVE.md / dynamic_k_relative.json")
    args = ap.parse_args()
    out_md = OUT_MD_REL if args.relative else OUT_MD
    out_json = OUT_JSON_REL if args.relative else OUT_JSON

    target = _load_prompts(args.domain, args.n_prompts)
    control = _load_prompts(args.control, args.n_prompts)
    if not target or not control:
        raise SystemExit(f"not enough split-B prompts for {args.domain}/{args.control}")

    print(f"loading {MODEL_ID} ...", flush=True)
    t0 = time.time()
    loaded = load_model(MODEL_ID, device="cpu", dtype="bfloat16")
    print(f"loaded in {time.time() - t0:.1f}s (peak RSS {_peak_rss_mb():.0f} MB)", flush=True)
    loaded.model.eval()

    results = []
    print("condition=baseline (full top-8) ...", flush=True)
    b_t0 = time.time()
    loss_t, _ = _mean_nll_and_kept(loaded, target)
    loss_c, _ = _mean_nll_and_kept(loaded, control)
    results.append({
        "condition": "baseline",
        "mass_threshold": 1.0,
        "mean_kept": 8.0,
        "loss_on_target": round(loss_t, 4),
        "loss_on_control": round(loss_c, 4),
        "wall_seconds": round(time.time() - b_t0, 1),
    })

    for th in args.mass_threshold:
        print(f"condition=dynamic_k_{th} ...", flush=True)
        t0 = time.time()
        with DynamicKMoe(loaded.model, mass_threshold=th, relative=args.relative):
            loss_t, kept_t = _mean_nll_and_kept(loaded, target)
            loss_c, kept_c = _mean_nll_and_kept(loaded, control)
        wall = time.time() - t0
        results.append({
            "condition": f"dynamic_k_{th}",
            "mass_threshold": th,
            "mean_kept_target": round(kept_t, 2),
            "mean_kept_control": round(kept_c, 2),
            "loss_on_target": round(loss_t, 4),
            "loss_on_control": round(loss_c, 4),
            "delta_target": round(loss_t - results[0]["loss_on_target"], 4),
            "delta_control": round(loss_c - results[0]["loss_on_control"], 4),
            "wall_seconds": round(wall, 1),
        })

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    _write_report(results, args.domain, args.control, relative=args.relative, out_md=out_md)
    print(f"wrote {out_md} and {out_json}", flush=True)


def _write_report(results, domain, control, relative=False, out_md=OUT_MD):
    base = results[0]
    lines = [
        "# Dynamic top-k: per-token adaptive FFN truncation"
        + (" (relative-mass thresholds)" if relative else ""),
        "",
        f"Target domain: `{domain}`. Control domain: `{control}`. "
        "Metric: mean per-token teacher-forced cross-entropy (nats), held-out (split=B) prompts.",
        "",
        "This is the per-token adaptive counterpart to `docs/OFFLOADING.md`. "
        "`OffloadedMoe` keeps a fixed subset of experts for every token. "
        "`DynamicKMoe` keeps a variable number `k_t` per token, determined by the "
        "router's own softmax mass: the smallest prefix of the top-8 probabilities "
        "that exceeds the `mass_threshold`.",
        "",
        "## Method caveats",
        "",
        "- The gate is still evaluated over the full candidate pool (or the fixed "
        "kept set, if used with `OffloadedMoe`). The savings are in FFN matmuls, not "
        "in the router projection.",
        "- The current implementation uses a per-token Python loop inside the FFN. "
        "Wall-clock time is therefore dominated by Python overhead and is **not "
        "expected to improve** over the original in this prototype. The reported "
        "`mean_kept` is the FFN-compute saving; a fused CUDA/Metal kernel would be "
        "needed to realise it as wall-clock speed.",
        "- The threshold is a hyperparameter. It was not tuned on the evaluated prompts.",
        ("- Thresholds are RELATIVE: a threshold of 0.9 keeps the smallest prefix "
         "carrying 90% of the mass the router gave its top-8, per token. Absolute "
         "thresholds never fire on OLMoE (`norm_topk_prob=False`; top-8 absolute "
         "mass is ~0.42 on average) \u2014 see `docs/DYNAMIC_K.md`."
         if relative else
         "- Thresholds are ABSOLUTE cumulative top-k weight mass. On OLMoE "
         "(`norm_topk_prob=False`) the top-8 weights sum to ~0.42 on average, so "
         "absolute thresholds >= 0.9 never fire."),
        "",
        "## Results",
        "",
        "| condition | mass threshold | mean kept | loss on target | delta | loss on control | delta | wall (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if r["condition"] == "baseline":
            lines.append(
                f"| {r['condition']} | 1.0 | {r['mean_kept']:.2f} | "
                f"{r['loss_on_target']:.4f} | — | {r['loss_on_control']:.4f} | — | {r['wall_seconds']:.1f} |"
            )
        else:
            lines.append(
                f"| {r['condition']} | {r['mass_threshold']} | "
                f"{r.get('mean_kept_target', 8.0):.2f} | {r['loss_on_target']:.4f} | "
                f"{r['delta_target']:+.4f} | {r['loss_on_control']:.4f} | "
                f"{r['delta_control']:+.4f} | {r['wall_seconds']:.1f} |"
            )
    lines += [
        "",
        "## Reading this",
        "",
        "- If `mean_kept` drops substantially (e.g. 8 -> 4) and loss stays close to "
        "baseline, the router is often concentrated and most of the top-8 mass is "
        "carried by a small prefix. That is evidence for an *adaptive* sparsity policy.",
        "- If `mean_kept` stays near 8 even at 0.99, the router is usually diffuse "
        "and dynamic-k saves little.",
        "- Wall-clock is not the metric here: this is a quality + FLOP-count "
        "experiment. A production implementation would fuse the variable-k loop.",
        "",
        "## Honest limits",
        "",
        f"- n={len(results)-1} thresholds, one model, one seed, one domain pair, "
        "split-B prompts only. Directional.",
        "- No wall-clock speed claim is made for this Python prototype.",
        "- No random or held-out threshold selection; thresholds are reported as given.",
    ]
    out_md.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
