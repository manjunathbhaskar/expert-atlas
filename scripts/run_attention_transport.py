"""Attention transport: retrieval-head identification + collapse test (WS1).

Question — the repaired probe (docs/PROBE_REPAIRED.md) showed the needle's
content survives at its source position but arrives degraded at the readout
on model-wrong long prompts, and the anchor test (docs/ANCHOR_CAUSAL.md)
showed injecting content at the readout is content-independent, i.e. the
missing ingredient is not "the right content" but plausibly "the transport".
This script measures the transport directly: attention from the final
(readout) position onto the needle's tokens, per layer and head.

Registered design (declared before any long-bucket data is inspected):

  Stage 1 — identification (SHORT prompts, model-correct only):
    * repaired probe set, 256-token bucket, forced_choice_correct prompts;
    * for each prompt, capture the final query row's post-softmax attention
      at every (layer, head); score = total mass on needle_token_span;
    * rank the 256 head cells by mean needle mass; the top K=16 (6.25%) are
      the candidate "retrieval heads". K is fixed in advance.

  Stage 2 — collapse test (LONG prompts, 3840 bucket, right vs wrong):
    * per prompt, per head: needle mass from the final position;
    * primary statistic: per-prompt mean needle mass over the K identified
      heads (M_top) and over the other 240 heads (M_rest);
    * group test on M_top (right vs wrong): label-shuffle permutation
      (2000 perms) + Cohen d.  NULL: no difference between groups.
    * SPECIFICITY test: obs = (right-wrong drop in identified heads) minus
      (right-wrong drop in remaining heads); null from 2000 random K-head
      subsets (head-identity null).  NULL: the drop is diffuse — a random
      set of K heads shows the same drop as the identified set.
    * per-head right-vs-wrong permutation p with BH-FDR + practical floor
      (drop >= half the head's short-prompt mass) to count how many heads
      individually collapse — "concentrated vs diffuse" is reported however
      it comes out.

Output: data/attention_transport.json + per-prompt masses in
data/attention_transport/.

Usage:
    HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 \
      .venv/bin/python scripts/run_attention_transport.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from expertatlas.attention_transport import LastRowAttentionCapture
from expertatlas.capture import load_model
from expertatlas.stats import bh_fdr

MODEL_ID = "allenai/OLMoE-1B-7B-0924"
REPO_ROOT = Path(__file__).parent.parent
PROBE_SET = REPO_ROOT / "probes" / "probe_set_context_repaired.yaml"
RECORDS = REPO_ROOT / "data" / "context_probe_repaired" / "records.jsonl"
OUT_DIR = REPO_ROOT / "data" / "attention_transport"
OUT_JSON = REPO_ROOT / "data" / "attention_transport.json"

ID_BUCKET = 256
EVAL_BUCKET = 3840
TOP_K = 16
N_PERM = 2000
SEED = 0


def load_records() -> list[dict]:
    return [json.loads(l) for l in RECORDS.read_text().splitlines()]


def capture_masses(loaded, prompts: dict[int, dict], recs: list[dict],
                   tag: str) -> dict[int, np.ndarray]:
    """Per prompt: (n_layers, n_heads) needle attention mass from last pos."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok = loaded.tokenizer
    out: dict[int, np.ndarray] = {}
    t0 = time.time()
    for i, r in enumerate(recs):
        pid = r["prompt_id"]
        f = OUT_DIR / f"mass_{tag}_{pid:06d}.npz"
        if f.exists():
            out[pid] = np.load(f)["mass"]
            continue
        ids = tok(prompts[pid]["text"], return_tensors="pt")
        with LastRowAttentionCapture(loaded.model) as cap, torch.no_grad():
            loaded.model(**ids, logits_to_keep=1)
        span = tuple(r["needle_token_span"])
        mass = cap.needle_mass(span).numpy()
        np.savez_compressed(f, mass=mass)
        out[pid] = mass
        el = time.time() - t0
        print(f"[{tag} {i + 1}/{len(recs)}] pid={pid} n={r['n_tokens']} "
              f"ETA {(el / (i + 1)) * (len(recs) - i - 1) / 60:.1f} min",
              flush=True)
    return out


def perm_diff_p(a: np.ndarray, b: np.ndarray, rng) -> tuple[float, float]:
    """Two-sided label-shuffle permutation p for mean(a)-mean(b), plus d."""
    obs = a.mean() - b.mean()
    pooled = np.concatenate([a, b])
    n = len(a)
    cnt = 0
    for _ in range(N_PERM):
        rng.shuffle(pooled)
        if abs(pooled[:n].mean() - pooled[n:].mean()) >= abs(obs):
            cnt += 1
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                 / (len(a) + len(b) - 2))
    d = float(obs / sp) if sp > 0 else float("nan")
    return cnt / N_PERM, d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(OUT_JSON))
    args = ap.parse_args()

    ps = yaml.safe_load(PROBE_SET.read_text())
    prompts = {p["prompt_id"]: p for p in ps["prompts"]}
    recs = load_records()

    loaded = load_model(MODEL_ID, device="cpu", dtype="bfloat16")
    loaded.model.eval()

    # ---- Stage 1: identification on short, model-correct prompts ----
    id_recs = [r for r in recs
               if r["bucket"] == ID_BUCKET and r["forced_choice_correct"]]
    print(f"stage 1: {len(id_recs)} short correct prompts", flush=True)
    id_mass = capture_masses(loaded, prompts, id_recs, "short")
    id_stack = np.stack([id_mass[r["prompt_id"]] for r in id_recs])
    mean_mass = id_stack.mean(axis=0)                    # (L, H)
    n_layers, n_heads = mean_mass.shape
    chance = float(np.mean([
        (r["needle_token_span"][1] - r["needle_token_span"][0]) / r["n_tokens"]
        for r in id_recs]))
    flat = mean_mass.ravel()
    order = np.argsort(flat)[::-1]
    top_cells = [(int(c // n_heads), int(c % n_heads)) for c in order[:TOP_K]]
    top_mask = np.zeros(n_layers * n_heads, dtype=bool)
    top_mask[order[:TOP_K]] = True
    print("top heads (layer, head, mean mass, x chance):", flush=True)
    for l, h in top_cells:
        print(f"  L{l} H{h}: {mean_mass[l, h]:.4f} "
              f"({mean_mass[l, h] / chance:.1f}x)", flush=True)

    # ---- Stage 2: collapse test on the long bucket ----
    ev_recs = [r for r in recs if r["bucket"] == EVAL_BUCKET]
    ev_mass = capture_masses(loaded, prompts, ev_recs, "long")
    right = [r for r in ev_recs if r["forced_choice_correct"]]
    wrong = [r for r in ev_recs if not r["forced_choice_correct"]]
    R = np.stack([ev_mass[r["prompt_id"]].ravel() for r in right])  # (nR, 256)
    W = np.stack([ev_mass[r["prompt_id"]].ravel() for r in wrong])
    long_chance = float(np.mean([
        (r["needle_token_span"][1] - r["needle_token_span"][0]) / r["n_tokens"]
        for r in ev_recs]))

    rng = np.random.default_rng(SEED)
    m_top_r, m_top_w = R[:, top_mask].mean(axis=1), W[:, top_mask].mean(axis=1)
    m_rest_r, m_rest_w = R[:, ~top_mask].mean(axis=1), W[:, ~top_mask].mean(axis=1)
    p_top, d_top = perm_diff_p(m_top_r.copy(), m_top_w.copy(), rng)
    p_rest, d_rest = perm_diff_p(m_rest_r.copy(), m_rest_w.copy(), rng)

    # specificity: identified-set drop vs random K-head-set drop
    obs_spec = ((m_top_r.mean() - m_top_w.mean())
                - (m_rest_r.mean() - m_rest_w.mean()))
    null_spec = []
    for _ in range(N_PERM):
        mask = np.zeros(n_layers * n_heads, dtype=bool)
        mask[rng.choice(n_layers * n_heads, TOP_K, replace=False)] = True
        null_spec.append((R[:, mask].mean() - W[:, mask].mean())
                         - (R[:, ~mask].mean() - W[:, ~mask].mean()))
    null_spec = np.array(null_spec)
    p_spec = float((null_spec >= obs_spec).mean())

    # per-head drop with FDR + practical floor
    per_head = []
    rng2 = np.random.default_rng(SEED + 1)
    for c in range(n_layers * n_heads):
        p, d = perm_diff_p(R[:, c].copy(), W[:, c].copy(), rng2)
        per_head.append({"layer": int(c // n_heads), "head": int(c % n_heads),
                         "short_mean": float(flat[c]),
                         "right_mean": float(R[:, c].mean()),
                         "wrong_mean": float(W[:, c].mean()),
                         "p": p, "d": d,
                         "identified": bool(top_mask[c])})
    rej = bh_fdr(np.array([h["p"] for h in per_head]), q=0.05)
    for h, rj in zip(per_head, rej):
        drop = h["right_mean"] - h["wrong_mean"]
        h["fdr_sig"] = bool(rj)
        h["practical"] = bool(drop >= 0.5 * h["short_mean"]
                              and h["short_mean"] > 0)
        h["collapses"] = bool(rj and h["practical"] and drop > 0)

    n_collapse = sum(h["collapses"] for h in per_head)
    n_collapse_ident = sum(h["collapses"] for h in per_head if h["identified"])
    summary = {
        "n_short_correct": len(id_recs), "n_long_right": len(right),
        "n_long_wrong": len(wrong), "top_k": TOP_K,
        "chance_mass_short": chance, "chance_mass_long": long_chance,
        "top_cells": top_cells,
        "top_cells_mean_mass": [float(mean_mass[l, h]) for l, h in top_cells],
        "m_top": {"right": float(m_top_r.mean()), "wrong": float(m_top_w.mean()),
                  "perm_p": p_top, "d": d_top},
        "m_rest": {"right": float(m_rest_r.mean()), "wrong": float(m_rest_w.mean()),
                   "perm_p": p_rest, "d": d_rest},
        "specificity": {"obs": float(obs_spec),
                        "null_mean": float(null_spec.mean()),
                        "null_p95": float(np.percentile(null_spec, 95)),
                        "perm_p": p_spec},
        "n_heads_collapsing": n_collapse,
        "n_identified_collapsing": n_collapse_ident,
    }
    out = {"summary": summary, "per_head": per_head,
           "design": {"id_bucket": ID_BUCKET, "eval_bucket": EVAL_BUCKET,
                      "top_k": TOP_K, "n_perm": N_PERM, "seed": SEED}}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
