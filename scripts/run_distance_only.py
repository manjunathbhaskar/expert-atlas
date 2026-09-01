"""Distance-only trigger test: do the identified retrieval heads collapse
under pure positional distance, with ZERO distractors?

Context. Every failure this project has causally explained was
distractor-driven: the OLMoE hard set rots only in its 8-distractor arm (the
0-distractor arm is 1.000 at every length, docs/CONTEXT_ROT_HARD.md), and
Granite needed a 24-distractor escalation to fail at all. Published
positional-distance work reports degradation with clean haystacks. This
script tests whether the SAME 16 retrieval heads identified in
docs/ATTENTION_TRANSPORT.md (frozen -- loaded from
data/attention_transport.json, never re-identified here) lose needle
attention when only distance grows.

Registered design (declared before any capture from this probe set):

  Substrate: probes/probe_set_context_distance.yaml -- 0 distractors,
  depths {0.15, 0.50, 0.85} x buckets {256, 1024, 2048, 3840} x 2 haystacks
  x 8 replicates = 192 prompts. distance := n_tokens - needle_token_end.

  Per prompt, one teacher-forced forward pass captures (a) forced-choice
  correctness/prob from the final logits and (b) the final query row's
  post-softmax attention, giving M_top (mean needle mass over the 16 frozen
  heads) and M_rest (other 240 heads).

  Primary test (fixed length, distance via depth): within the 3840 bucket
  (n=48), Spearman rho of M_top vs distance, permutation p (2000 shuffles of
  the distance labels). Chance mass is constant within the bucket, so no
  span/length normalisation is needed. NULL: no monotonic decline.

  Secondary test (collapse index vs matched references): the 3840
  8-distractor run (data/attention_transport.json) gives matched-length
  reference levels M_right and M_wrong for the same 16 heads. For each 3840
  cell here, c := (M_right_ref - M_top_obs) / (M_right_ref - M_wrong_ref).
  c ~ 0 -> heads at the healthy level despite max distance (distractor-gated
  mechanism); c ~ 1 -> heads at the collapsed level under distance alone
  (shared mechanism). Interpretation bands registered: c < 0.2
  distractor-gated, c > 0.5 substantial distance-driven collapse,
  in between partial (reported as-is).

  Specificity: the same Spearman rho for M_rest -- a decline that is just as
  strong outside the identified heads is a global length artefact, not a
  retrieval-head effect.

  Accuracy: forced-choice accuracy per cell is reported; the prior
  0-distractor arm was 1.000 everywhere at depth 0.50, so any failures here
  (esp. depth 0.15) are themselves a finding and are listed.

  Effect-size floor: registered practical floor for "the heads collapse under
  distance alone" is BOTH perm p < 0.05 on the primary rho AND c >= 0.5 at
  the max-distance cell (3840 / depth 0.15). Anything less is reported at
  face value without the headline claim.

Output: data/distance_only.json + per-prompt masses in data/distance_only/.

Usage:
    HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 \
      .venv/bin/python scripts/run_distance_only.py
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
from expertatlas.context_metrics import token_span_from_chars

MODEL_ID = "allenai/OLMoE-1B-7B-0924"
REPO_ROOT = Path(__file__).parent.parent
PROBE_SET = REPO_ROOT / "probes" / "probe_set_context_distance.yaml"
HEADS_JSON = REPO_ROOT / "data" / "attention_transport.json"
OUT_DIR = REPO_ROOT / "data" / "distance_only"
OUT_JSON = REPO_ROOT / "data" / "distance_only.json"

N_PERM = 2000
SEED = 0
PRIMARY_BUCKET = 3840


def spearman_perm(x: np.ndarray, y: np.ndarray, rng) -> tuple[float, float]:
    """Spearman rho of y vs x with a label-shuffle permutation p (two-sided)."""

    def _rho(a, b):
        ra = np.argsort(np.argsort(a)).astype(np.float64)
        rb = np.argsort(np.argsort(b)).astype(np.float64)
        ra -= ra.mean()
        rb -= rb.mean()
        den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
        return float((ra * rb).sum() / den) if den > 0 else float("nan")

    obs = _rho(x, y)
    xs = x.copy()
    cnt = 0
    for _ in range(N_PERM):
        rng.shuffle(xs)
        if abs(_rho(xs, y)) >= abs(obs):
            cnt += 1
    return obs, cnt / N_PERM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(OUT_JSON))
    args = ap.parse_args()

    ps = yaml.safe_load(PROBE_SET.read_text())
    prompts = ps["prompts"]
    ref = json.loads(HEADS_JSON.read_text())["summary"]
    top_cells = [tuple(c) for c in ref["top_cells"]]
    ref_right = ref["m_top"]["right"]
    ref_wrong = ref["m_top"]["wrong"]

    loaded = load_model(MODEL_ID, device="cpu", dtype="bfloat16")
    loaded.model.eval()
    tok = loaded.tokenizer
    candidate_ids = [tok(" " + w, add_special_tokens=False)["input_ids"][0]
                     for w in ps["candidate_words"]]

    n_layers = loaded.model.config.num_hidden_layers
    n_heads = loaded.model.config.num_attention_heads
    top_mask = np.zeros(n_layers * n_heads, dtype=bool)
    for l, h in top_cells:
        top_mask[l * n_heads + h] = True

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t0 = time.time()
    for i, p in enumerate(prompts):
        pid = p["prompt_id"]
        f = OUT_DIR / f"row_{pid:06d}.json"
        if f.exists():
            rows.append(json.loads(f.read_text()))
            continue
        enc = tok(p["text"], return_tensors="pt", return_offsets_mapping=True)
        offsets = [tuple(x) for x in enc["offset_mapping"][0].tolist()]
        inputs = {k: v for k, v in enc.items() if k != "offset_mapping"}
        n_tokens = int(inputs["input_ids"].shape[1])
        n_span = token_span_from_chars(offsets, p["needle_char_span"])

        with LastRowAttentionCapture(loaded.model) as cap, torch.no_grad():
            out = loaded.model(**inputs, logits_to_keep=1)
        mass = cap.needle_mass(n_span).numpy().ravel()

        lg = out.logits[0, -1].detach().float().cpu().numpy().astype(np.float64)
        cand = lg[np.array(candidate_ids)]
        ans_pos = ps["candidate_words"].index(p["answer_word"])
        pr = np.exp(cand - cand.max())
        pr = pr / pr.sum()

        row = {
            "prompt_id": pid, "bucket": int(p["bucket"]),
            "needle_depth": float(p["needle_depth"]),
            "haystack": p["haystack"], "replicate": int(p["replicate"]),
            "n_tokens": n_tokens,
            "needle_token_span": list(n_span),
            "distance": n_tokens - n_span[1],
            "chance_mass": (n_span[1] - n_span[0]) / n_tokens,
            "forced_choice_correct": bool(int(np.argmax(cand)) == ans_pos),
            "forced_choice_prob": float(pr[ans_pos]),
            "m_top": float(mass[top_mask].mean()),
            "m_rest": float(mass[~top_mask].mean()),
        }
        f.write_text(json.dumps(row))
        rows.append(row)
        el = time.time() - t0
        done = sum(1 for r in rows if r is not None)
        print(f"[{i + 1}/{len(prompts)}] pid={pid} n={n_tokens} "
              f"fc={int(row['forced_choice_correct'])} m_top={row['m_top']:.3f} "
              f"ETA {(el / max(done, 1)) * (len(prompts) - i - 1) / 60:.1f} min",
              flush=True)

    # ---- analysis ----
    rng = np.random.default_rng(SEED)

    cells = {}
    for r in rows:
        key = (r["needle_depth"], r["bucket"])
        cells.setdefault(key, []).append(r)
    cell_stats = []
    for (depth, bucket), rs in sorted(cells.items()):
        m = np.array([r["m_top"] for r in rs])
        c_idx = (ref_right - m.mean()) / (ref_right - ref_wrong)
        cell_stats.append({
            "needle_depth": depth, "bucket": bucket, "n": len(rs),
            "accuracy": float(np.mean([r["forced_choice_correct"] for r in rs])),
            "mean_prob": float(np.mean([r["forced_choice_prob"] for r in rs])),
            "mean_distance": float(np.mean([r["distance"] for r in rs])),
            "m_top_mean": float(m.mean()), "m_top_sd": float(m.std(ddof=1)),
            "m_rest_mean": float(np.mean([r["m_rest"] for r in rs])),
            "chance_mass": float(np.mean([r["chance_mass"] for r in rs])),
            "collapse_index_vs_3840refs": float(c_idx) if bucket == PRIMARY_BUCKET else None,
        })

    prim = [r for r in rows if r["bucket"] == PRIMARY_BUCKET]
    dist = np.array([r["distance"] for r in prim], dtype=np.float64)
    mtop = np.array([r["m_top"] for r in prim], dtype=np.float64)
    mrest = np.array([r["m_rest"] for r in prim], dtype=np.float64)
    rho_top, p_top = spearman_perm(dist.copy(), mtop, rng)
    rho_rest, p_rest = spearman_perm(dist.copy(), mrest, rng)

    max_cell = next(c for c in cell_stats
                    if c["bucket"] == PRIMARY_BUCKET and c["needle_depth"] == 0.15)
    c_max = max_cell["collapse_index_vs_3840refs"]

    verdict = "distance_collapses_heads" if (p_top < 0.05 and rho_top < 0
                                             and c_max >= 0.5) else (
        "distractor_gated" if (c_max < 0.2) else "partial")

    summary = {
        "n_prompts": len(rows),
        "frozen_heads": [list(c) for c in top_cells],
        "reference_m_top": {"right_8distractor_3840": ref_right,
                            "wrong_8distractor_3840": ref_wrong},
        "primary_bucket": PRIMARY_BUCKET,
        "primary_rho_m_top_vs_distance": {"rho": rho_top, "perm_p": p_top,
                                          "n": len(prim)},
        "specificity_rho_m_rest_vs_distance": {"rho": rho_rest, "perm_p": p_rest},
        "collapse_index_max_distance_cell": c_max,
        "registered_verdict": verdict,
        "overall_accuracy": float(np.mean([r["forced_choice_correct"] for r in rows])),
        "n_failures": int(sum(not r["forced_choice_correct"] for r in rows)),
    }
    out = {"summary": summary, "cells": cell_stats,
           "design": {"n_perm": N_PERM, "seed": SEED,
                      "heads_source": str(HEADS_JSON.name)}}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    for c in cell_stats:
        print(c, flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
