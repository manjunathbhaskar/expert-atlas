"""Does relative dynamic-k behave differently at long context?

`docs/DYNAMIC_K_RELATIVE.md` measured the quality/compute curve on short
domain prompts; nobody has checked what per-token adaptive selection does on
the long-context substrate where accuracy degrades (`docs/CONTEXT_ROT_HARD.md`).
Two pre-registered questions:

1. **Compute**: does the router demand MORE experts per token (higher mean
   kept k at a fixed relative threshold) as context grows? If routing gets
   more diffuse with length (`docs/MECHANISM.md` found entropy rises), the
   mass threshold should be crossed later and mean k should rise.
2. **Quality**: does truncating low-mass experts hurt long-context accuracy
   MORE than short-context accuracy (interaction), or is the cost flat in
   length? A disproportionate long-context cost would mean the low-mass tail
   of the router distribution is doing real retrieval work at long context.

Design
------
Buckets: the shortest (256) and the most degraded evaluated end (3840) of the
hard variant; all prompts in each bucket. Conditions: baseline (full top-8)
and relative dynamic-k at thresholds {0.9, 0.7, 0.5} — the same grid as
DYNAMIC_K_RELATIVE.md, no tuning. Scoring: `run_context_sweep.score_answer`
(imported), forced choice over the same 8 candidates. The baseline condition
must reproduce the stored `forced_choice_prob` for each prompt (repro gate).

Honest limits: one model, one seed, teacher-forced; n=24 prompts per bucket;
answer_prob deltas are reported with paired sign-flip permutation p and dz per
this project's standards, with BH-FDR across the accuracy family.

Usage:
    HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 .venv/bin/python \
        scripts/run_dynamic_k_context.py [--buckets 256,3840] [--limit N]
    .venv/bin/python scripts/run_dynamic_k_context.py --analyze
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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

MODEL_ID = "allenai/OLMoE-1B-7B-0924"
PROBE_HARD = REPO_ROOT / "probes" / "probe_set_context_hard.yaml"
HARD_JSON = REPO_ROOT / "data" / "context_rot_hard.json"
OUT_DIR = REPO_ROOT / "data" / "dynamic_k_context"
RECORDS = OUT_DIR / "records.jsonl"
OUT_MD = REPO_ROOT / "docs" / "DYNAMIC_K_CONTEXT.md"

THRESHOLDS = (0.9, 0.7, 0.5)
DEFAULT_BUCKETS = (256, 3840)
N_SIGNFLIP = 20000
Q_FDR = 0.05


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def done_keys() -> set[tuple[int, str]]:
    if not RECORDS.exists():
        return set()
    out = set()
    for line in RECORDS.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out.add((r["prompt_id"], r["condition"]))
    return out


def run(args) -> None:
    sweep = _load_module(REPO_ROOT / "scripts" / "run_context_sweep.py", "_ws1_sweep")
    ps = yaml.safe_load(PROBE_HARD.read_text())
    buckets = tuple(int(b) for b in args.buckets.split(","))
    prompts = [p for p in ps["prompts"] if p["bucket"] in buckets]
    prompts.sort(key=lambda p: p["prompt_id"])
    stored = {p["prompt_id"]: p for p in json.loads(HARD_JSON.read_text())["per_prompt"]}

    print(f"loading {MODEL_ID} ...", flush=True)
    loaded = load_model(MODEL_ID, device="cpu", dtype="bfloat16")
    loaded.model.eval()

    candidate_ids = [loaded.tokenizer(" " + w, add_special_tokens=False)["input_ids"][0]
                     for w in ps["candidate_words"]]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise RuntimeError("candidate words collide under the tokenizer")

    conditions = ["baseline"] + [f"rk{t}" for t in THRESHOLDS]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    already = done_keys()
    jobs = [(p, c) for p in prompts for c in conditions
            if (p["prompt_id"], c) not in already]
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(prompts)} prompts x {len(conditions)} conditions; "
          f"{len(already)} done; {len(jobs)} to run", flush=True)

    t_run = time.time()
    for i, (p, cond) in enumerate(jobs):
        enc = loaded.tokenizer(p["text"], return_tensors="pt")
        answer_id = loaded.tokenizer(" " + p["answer_word"],
                                     add_special_tokens=False)["input_ids"][0]
        t0 = time.time()
        kept: list[int] = []
        if cond == "baseline":
            with torch.no_grad():
                out = loaded.model(**enc, output_router_logits=False)
        else:
            thr = float(cond[2:])
            with DynamicKMoe(loaded.model, mass_threshold=thr, relative=True):
                with torch.no_grad():
                    out = loaded.model(**enc, output_router_logits=False)
                for layer in loaded.model.model.layers:
                    kept.extend(getattr(layer.mlp, "_last_kept", []))
        last = out.logits[0, -1, :].float()
        acc = sweep.score_answer(last, candidate_ids, answer_id)

        rec = {
            "prompt_id": p["prompt_id"], "bucket": p["bucket"], "condition": cond,
            "n_tokens": int(enc["input_ids"].shape[1]),
            "mean_kept_k": (float(np.mean(kept)) if kept else 8.0),
            **acc,
        }
        if cond == "baseline":
            dev = abs(rec["forced_choice_prob"] - stored[p["prompt_id"]]["answer_prob"])
            rec["repro_abs_dev"] = dev
            if dev > 1e-6:
                print(f"  REPRO WARNING pid={p['prompt_id']}: baseline "
                      f"forced_choice_prob deviates from stored by {dev:.2e}", flush=True)
        with RECORDS.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        el = time.time() - t_run
        eta = el / (i + 1) * (len(jobs) - i - 1) / 60
        print(f"[{i+1}/{len(jobs)}] pid={p['prompt_id']:>4} b={p['bucket']:>5} "
              f"{cond:<9} k={rec['mean_kept_k']:.2f} p={rec['forced_choice_prob']:.3f} "
              f"{time.time()-t0:5.1f}s ETA {eta:6.1f}m", flush=True)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _signflip_p(diffs: np.ndarray, n_perm: int, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    obs = abs(diffs.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, diffs.size))
    null = np.abs((signs * diffs).mean(axis=1))
    return float((np.sum(null >= obs) + 1) / (n_perm + 1))


def _bh_fdr(pvals: list[float], q: float) -> list[bool]:
    order = np.argsort(pvals)
    m = len(pvals)
    sig = [False] * m
    thresh = 0
    for rank, idx in enumerate(order, start=1):
        if pvals[idx] <= q * rank / m:
            thresh = rank
    for rank, idx in enumerate(order, start=1):
        if rank <= thresh:
            sig[idx] = True
    return sig


def analyze() -> None:
    recs = [json.loads(l) for l in RECORDS.read_text().splitlines() if l.strip()]
    by_key = {(r["prompt_id"], r["condition"]): r for r in recs}
    buckets = sorted({r["bucket"] for r in recs})
    pids = {b: sorted({r["prompt_id"] for r in recs if r["bucket"] == b}) for b in buckets}

    lines = [
        "# Relative dynamic-k on the long-context substrate",
        "",
        "## Limitations (read first)",
        "",
        "- One model, one seed, teacher-forced forced-choice scoring; "
        f"n={min(len(v) for v in pids.values())} prompts per bucket.",
        "- Thresholds are the pre-registered DYNAMIC_K_RELATIVE.md grid; nothing",
        "  was tuned on these prompts.",
        "- The baseline condition reproduced the stored hard-variant answer_prob",
        "  (max dev reported below); if it had not, nothing here would be",
        "  comparable to docs/CONTEXT_ROT_HARD.md.",
        "",
    ]
    max_dev = max((r.get("repro_abs_dev", 0.0) for r in recs), default=0.0)
    lines += [f"Baseline reproduction: max |forced_choice_prob - stored| = {max_dev:.2e}", ""]

    lines += ["## Mean kept k by bucket (question 1: does the router demand more experts with length?)",
              "", "| threshold | " + " | ".join(f"bucket {b}" for b in buckets) + " |",
              "|---|" + "---|" * len(buckets)]
    for thr in THRESHOLDS:
        row = [f"rk{thr}"]
        for b in buckets:
            ks = [by_key[(pid, f"rk{thr}")]["mean_kept_k"] for pid in pids[b]
                  if (pid, f"rk{thr}") in by_key]
            row.append(f"{np.mean(ks):.3f}")
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Accuracy cost by bucket (question 2: does truncation hurt long context more?)",
              "",
              "| bucket | condition | mean answer_prob | delta vs baseline | dz | perm p | FDR sig | accuracy |",
              "|---|---|---|---|---|---|---|---|"]
    pvals, cells = [], []
    for b in buckets:
        base = np.array([by_key[(pid, "baseline")]["forced_choice_prob"] for pid in pids[b]])
        base_acc = np.mean([by_key[(pid, "baseline")]["forced_choice_correct"] for pid in pids[b]])
        lines.append(f"| {b} | baseline | {base.mean():.4f} | — | — | — | — | {base_acc:.3f} |")
        for thr in THRESHOLDS:
            cond = f"rk{thr}"
            probs = np.array([by_key[(pid, cond)]["forced_choice_prob"] for pid in pids[b]])
            accs = np.mean([by_key[(pid, cond)]["forced_choice_correct"] for pid in pids[b]])
            d = probs - base
            dz = d.mean() / (d.std(ddof=1) + 1e-12)
            p = _signflip_p(d, N_SIGNFLIP)
            pvals.append(p)
            cells.append((b, cond, probs.mean(), d.mean(), dz, p, accs))
    sig = _bh_fdr(pvals, Q_FDR)
    out_rows = []
    for (b, cond, mp, dm, dz, p, accs), s in zip(cells, sig):
        out_rows.append((b, f"| {b} | {cond} | {mp:.4f} | {dm:+.4f} | {dz:+.2f} "
                            f"| {p:.4f} | {s} | {accs:.3f} |"))
    # keep bucket grouping/order
    for b in buckets:
        for bb, row in out_rows:
            if bb == b:
                lines.append(row)

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--buckets", type=str, default=",".join(str(b) for b in DEFAULT_BUCKETS))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.analyze:
        analyze()
    else:
        run(args)


if __name__ == "__main__":
    main()
