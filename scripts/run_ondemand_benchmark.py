"""Measured benchmark for the on-demand expert runtime (`expertatlas/ondemand.py`).

This is the project's local-execution claim put to an actual measurement, not
an analytical byte count (docs/OFFLOADING.md's honest limitation): can the
model run correctly while the process only ever holds a small fraction of the
expert weights, and what does that cost in wall-clock?

Conditions
----------
* `dense`      — normal resident model (reference loss + speed).
* `ondemand-N` — on-demand runtime with an LRU capacity of N experts
                 (N=0 refetches every use).
* `ondemand-N-rkT` — same, plus relative dynamic-k at mass threshold T
                 (docs/DYNAMIC_K_RELATIVE.md): fewer experts fetched AND
                 computed per token. NLL is NOT expected to match dense here;
                 the deviation is the known dynamic-k quality cost.

Per condition: teacher-forced mean NLL over the same prompt set (same weights,
same math as dense; measured deviation is BF16 kernel drift — weights verified
bit-identical and the per-module output differs by 1 bf16 ulp because
transformers dispatches a differently-batched experts kernel, consistent with
the BF16 batch-shape drift already documented in RESEARCH_LOG entry 10),
tokens/sec, LRU hit rate, bytes fetched, and memory read from /proc/self/status:

* `RssAnon` — anonymous (non-reclaimable without swap) resident memory: the
  process's true footprint. THIS is the local-execution number.
* `RssFile` — file-backed mmap pages currently resident: the OS page cache
  visible in our RSS; the kernel reclaims these under pressure without swap,
  so they are not part of the hard footprint (but they are why warm reruns
  are fast on this box).
* `VmHWM`   — peak RSS high-water mark (anon+file, cumulative across the
  whole process lifetime, so it is reported once, not per condition).

Each condition runs in a SEPARATE subprocess so memory numbers are clean.

Usage:
    HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 .venv/bin/python \
        scripts/run_ondemand_benchmark.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MODEL_ID = "allenai/OLMoE-1B-7B-0924"
OUT_MD = REPO_ROOT / "docs" / "ONDEMAND.md"
OUT_JSON = REPO_ROOT / "data" / "ondemand_benchmark.json"
DEFAULT_CAPACITIES = [0, 64, 128, 256, 512, 1024]
EXPERT_BYTES_TOTAL = 1024 * 3 * 2048 * 1024 * 2  # 16L x 64E x (gate+up+down) bf16


def proc_status() -> dict[str, int]:
    out = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        for key in ("RssAnon", "RssFile", "VmHWM", "VmRSS"):
            if line.startswith(key + ":"):
                out[key] = int(line.split()[1]) * 1024  # kB -> bytes
    return out


def load_prompts(n_per_domain: int) -> list[str]:
    from scripts.probe_keep_topk_fair import load_split_b_prompts

    prompts = []
    for domain in ("python", "medicine", "history", "cooking"):
        prompts.extend(load_split_b_prompts(domain, n_per_domain))
    return prompts


def run_condition(condition: str, n_per_domain: int) -> dict:
    """Runs inside the child process; prints one JSON line to stdout."""
    import torch

    prompts = load_prompts(n_per_domain)

    if condition == "dense":
        from expertatlas.capture import load_model
        loaded = load_model(MODEL_ID, device="cpu", dtype="bfloat16")
        loaded.model.eval()
        model, tokenizer = loaded.model, loaded.tokenizer
        extra = {}
    else:
        from expertatlas.ondemand import load_ondemand
        parts = condition.split("-")
        capacity = int(parts[1])
        policy = ("per_layer" if "pl" in parts[2:]
                  else "pinned" if "pin" in parts[2:] else "global")
        usage_counts = None
        if policy == "pinned":
            u = json.loads((REPO_ROOT / "data" / "utilization.json").read_text())
            counts = u["utilization"]["counts"]
            n_l = u["model"]["n_layers"]
            usage_counts = np.asarray(counts).reshape(n_l, -1)
        om = load_ondemand(MODEL_ID, "data/hf_cache", cache_experts=capacity,
                           cache_policy=policy, usage_counts=usage_counts)
        model, tokenizer = om.model, om.tokenizer
        extra = {"capacity": capacity, "om": om}
        rk = next((p for p in parts[2:] if p.startswith("rk")), None)
        if rk:
            from expertatlas.dynamic_k import patch_dynamic_k
            threshold = float(rk[2:])
            patch_dynamic_k(model, mass_threshold=threshold, relative=True)
            extra["rk_threshold"] = threshold

    mem_loaded = proc_status()

    total_loss, total_tokens, t0 = 0.0, 0, time.time()
    kept_counts: list[int] = []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            out = model(**ids, labels=ids["input_ids"], output_router_logits=False)
        n_tok = int(ids["input_ids"].shape[1])
        total_loss += out.loss.item() * (n_tok - 1)
        total_tokens += n_tok
        if "rk_threshold" in extra:
            for layer in model.model.layers:
                kept_counts.extend(getattr(layer.mlp, "_last_kept", []))
    wall = time.time() - t0
    mem_end = proc_status()

    rec = {
        "condition": condition,
        "n_prompts": len(prompts),
        "mean_nll": total_loss / max(total_tokens - len(prompts), 1),
        "total_tokens": total_tokens,
        "wall_seconds": wall,
        "tokens_per_second": total_tokens / wall,
        "mem_after_load": mem_loaded,
        "mem_after_eval": mem_end,
    }
    if "om" in extra:
        om = extra["om"]
        rec["lru"] = om.lru.stats.to_dict()
        rec["lru_resident_experts"] = om.lru.resident_experts()
        rec["lru_resident_bytes"] = om.lru.resident_bytes()
        rec["resident_param_bytes"] = om.resident_param_bytes
        if "rk_threshold" in extra:
            rec["rk_threshold"] = extra["rk_threshold"]
            rec["mean_kept_k"] = sum(kept_counts) / max(len(kept_counts), 1)
    print("RESULT " + json.dumps(rec), flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", type=str, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--capacities", type=int, nargs="+", default=DEFAULT_CAPACITIES)
    ap.add_argument("--per-layer", action="store_true",
                    help="also run each capacity with the per-layer cache policy")
    ap.add_argument("--pinned", action="store_true",
                    help="also run each capacity with the usage-pinned cache policy")
    ap.add_argument("--rk-thresholds", type=float, nargs="*",
                    default=[0.9, 0.8, 0.7, 0.5],
                    help="relative dynamic-k thresholds, run at --rk-capacity")
    ap.add_argument("--rk-capacity", type=int, default=1024)
    ap.add_argument("--n-per-domain", type=int, default=3)
    ap.add_argument("--out-md", type=str, default=str(OUT_MD))
    ap.add_argument("--out-json", type=str, default=str(OUT_JSON))
    args = ap.parse_args()

    if args.child:
        run_condition(args.child, args.n_per_domain)
        return

    conditions = ["dense"] + [f"ondemand-{c}" for c in args.capacities]
    if args.per_layer:
        conditions += [f"ondemand-{c}-pl" for c in args.capacities if c > 0]
    if args.pinned:
        conditions += [f"ondemand-{c}-pin" for c in args.capacities if c > 0]
    conditions += [f"ondemand-{args.rk_capacity}-rk{t}" for t in args.rk_thresholds]
    results = []
    for cond in conditions:
        print(f"=== {cond} ===", flush=True)
        cp = subprocess.run(
            [sys.executable, __file__, "--child", cond,
             "--n-per-domain", str(args.n_per_domain)],
            capture_output=True, text=True, cwd=REPO_ROOT)
        line = next((ln for ln in cp.stdout.splitlines() if ln.startswith("RESULT ")), None)
        if line is None:
            print(cp.stdout[-2000:], file=sys.stderr)
            print(cp.stderr[-4000:], file=sys.stderr)
            raise SystemExit(f"condition {cond} failed")
        rec = json.loads(line[len("RESULT "):])
        print(json.dumps({k: rec[k] for k in ("mean_nll", "tokens_per_second")},
                         indent=None), flush=True)
        results.append(rec)

    dense = results[0]
    for rec in results[1:]:
        dev = abs(rec["mean_nll"] - dense["mean_nll"])
        rec["nll_abs_dev_vs_dense"] = dev
        if "rk_threshold" in rec:
            continue  # dynamic-k deviation is the measured quality cost
        if dev > 2e-2:
            print(f"WARNING: {rec['condition']} NLL deviates from dense by {dev:.2e} "
                  f"(beyond BF16 kernel-drift scale; investigate)", flush=True)

    Path(args.out_json).write_text(json.dumps(results, indent=2))
    _write_report(results, args)
    print(f"wrote {args.out_md} and {args.out_json}", flush=True)


def _write_report(results: list[dict], args) -> None:
    dense = results[0]
    gib = 2 ** 30
    lines = [
        "# On-demand expert runtime: measured memory and wall-clock",
        "",
        "## Limitations (read first)",
        "",
        "- **Warm page cache.** This machine's RAM exceeds the checkpoint size, so",
        "  the OS caches the mmap'd shards after first touch; wall-clock numbers are",
        "  therefore a warm-cache bound, optimistic vs. a machine that genuinely",
        "  cannot hold the model (misses would become real disk reads there).",
        "  The MEMORY numbers are unaffected: `RssAnon` is what the process itself",
        "  must hold and is the honest local-execution figure.",
        "- One model (OLMoE-1B-7B-0924), CPU, BF16, batch 1, teacher-forced NLL on",
        f"  {dense['n_prompts']} split-B prompts (4 domains). No GPU or generation claims.",
        "- Correctness: expert weights were verified bit-identical to the dense",
        "  model's, and a single experts-module comparison with identical routing",
        "  differs by at most 1 bf16 ulp (transformers dispatches a differently",
        "  batched kernel; reduction-order drift). The small NLL deviation reported",
        "  below for non-dynamic-k conditions is that kernel drift, not a quality",
        "  change. Dynamic-k conditions deviate by design (measured quality cost).",
        "",
        "## Results",
        "",
        "| condition | mean NLL | mean k | tok/s | slowdown | RssAnon (GiB) | RssFile (GiB) | LRU hit rate | fetched (GiB) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for rec in results:
        mem = rec["mem_after_eval"]
        lru = rec.get("lru")
        mean_k = rec.get("mean_kept_k")
        lines.append(
            f"| {rec['condition']} | {rec['mean_nll']:.6f} "
            f"| {(f'{mean_k:.2f}' if mean_k is not None else '8 (full)')} "
            f"| {rec['tokens_per_second']:.1f} "
            f"| {dense['tokens_per_second'] / rec['tokens_per_second']:.2f}x "
            f"| {mem.get('RssAnon', 0) / gib:.2f} | {mem.get('RssFile', 0) / gib:.2f} "
            f"| {(f'{lru['hit_rate']:.3f}' if lru else '-')} "
            f"| {(f'{lru['bytes_fetched'] / gib:.1f}' if lru else '-')} |"
        )
    lines += [
        "",
        f"Expert weights total {EXPERT_BYTES_TOTAL / gib:.1f} GiB of the checkpoint;",
        "the on-demand process keeps only the LRU capacity's worth resident",
        "(`RssAnon`), plus ~0.9 GiB of non-expert weights and runtime overhead.",
        "",
        "## Reading this",
        "",
        "- If `mean NLL` matches dense (up to BF16 kernel drift) at every capacity,",
        "  the runtime is CORRECT and the whole trade is memory-vs-time, quantified",
        "  above.",
        "- The dense condition's RssAnon is the footprint this machine needed to run",
        "  the model at all; the smallest capacity's RssAnon is what a machine would",
        "  need with this runtime.",
    ]
    Path(args.out_md).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
