"""Causal test: boost identified retrieval heads' attention onto the needle.

Fourth intervention family. The three failed families touched the router
(fixed boost, entropy-triggered boost) or the content (residual anchor at
the readout). None touched the thing that moves information between
positions — the attention weights. The anchor result was content-
independent (true = wrong = random), which points at transport, not
content. This test boosts pre-softmax attention scores from the readout
window (question span through the final position) onto the needle's tokens,
at exactly the head cells stage 1 identified as carrying needle content on
short correct prompts (data/attention_transport.json).

Registered design (before evaluation):

  * Heads: the K=16 identified cells from run_attention_transport.py.
  * Boost: HeadBoost, additive pre-softmax bias `beta` at those cells,
    queries = question_token_span start through end, keys = needle span.
  * beta calibrated on the 1024-token DEV bucket (first 16 prompt_ids),
    identified-heads condition only, betas {1.0, 2.0, 4.0}; highest dev
    forced-choice accuracy wins. The 3840 bucket plays no calibration part.
  * Conditions on the 3840 EVAL bucket (all 64 prompts):
      - heads  — the identified K cells;
      - random — K head cells drawn uniformly per prompt (fixed seed),
                 same beta, same query/key spans (matched control);
    baseline comes from the capture run's records.jsonl.
  * Metrics: per-prompt forced-choice correctness + answer probability.
    Paired sign-flip permutation (2000) on delta answer-prob for
    heads-vs-baseline and heads-vs-random; Cohen dz. Causal bar:
    p<0.05 AND |dz|>=0.8 AND beats the random control AND helps the
    model-wrong subset more than the model-right subset.

Output: data/attention_boost_causal.json

Usage:
    HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 \
      .venv/bin/python scripts/run_attention_boost_causal.py
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

from expertatlas.attention_transport import HeadBoost, heads_to_by_layer
from expertatlas.capture import load_model
from scripts.run_context_probe_capture import score_answer

MODEL_ID = "allenai/OLMoE-1B-7B-0924"
REPO_ROOT = Path(__file__).parent.parent
PROBE_SET = REPO_ROOT / "probes" / "probe_set_context_repaired.yaml"
RECORDS = REPO_ROOT / "data" / "context_probe_repaired" / "records.jsonl"
TRANSPORT = REPO_ROOT / "data" / "attention_transport.json"
OUT_JSON = REPO_ROOT / "data" / "attention_boost_causal.json"

BETAS = (1.0, 2.0, 4.0)
DEV_BUCKET, EVAL_BUCKET = 1024, 3840
N_DEV = 16
N_PERM = 2000
SEED = 0


def run_prompt(loaded, prompt, rec, candidate_ids, cells, beta) -> dict:
    tok = loaded.tokenizer
    ids = tok(prompt["text"], return_tensors="pt")
    answer_id = tok(" " + prompt["answer_word"],
                    add_special_tokens=False)["input_ids"][0]
    span = tuple(rec["needle_token_span"])
    q_start = rec["question_token_span"][0]
    with HeadBoost(loaded.model, heads_to_by_layer(cells), span, beta,
                   query_start=q_start) as hb, torch.no_grad():
        out = loaded.model(**ids, logits_to_keep=1)
    assert hb.n_fired > 0
    return score_answer(out.logits[0, -1, :].float(), candidate_ids, answer_id)


def paired_stats(a: np.ndarray, b: np.ndarray, rng) -> dict:
    d = a - b
    dz = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("nan")
    obs = d.mean()
    flips = rng.choice([-1.0, 1.0], size=(N_PERM, len(d)))
    null = (flips * d).mean(axis=1)
    return {"mean_delta": float(obs), "dz": dz,
            "perm_p": float((np.abs(null) >= abs(obs)).mean()), "n": int(len(d))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(OUT_JSON))
    args = ap.parse_args()

    transport = json.loads(TRANSPORT.read_text())
    top_cells = [tuple(c) for c in transport["summary"]["top_cells"]]
    n_layers = 16
    n_heads = 16
    print(f"identified cells: {top_cells}", flush=True)

    ps = yaml.safe_load(PROBE_SET.read_text())
    prompts = {p["prompt_id"]: p for p in ps["prompts"]}
    recs = {r["prompt_id"]: r for r in
            (json.loads(l) for l in RECORDS.read_text().splitlines())}

    loaded = load_model(MODEL_ID, device="cpu", dtype="bfloat16")
    loaded.model.eval()
    tok = loaded.tokenizer
    candidate_ids = [tok(" " + w, add_special_tokens=False)["input_ids"][0]
                     for w in ps["candidate_words"]]

    # ---- beta calibration on the DEV bucket, identified heads only ----
    dev = sorted(r for r in recs if recs[r]["bucket"] == DEV_BUCKET)[:N_DEV]
    calib = {}
    for beta in BETAS:
        accs, probs = [], []
        for pid in dev:
            res = run_prompt(loaded, prompts[pid], recs[pid], candidate_ids,
                             top_cells, beta)
            accs.append(res["forced_choice_correct"])
            probs.append(res["forced_choice_prob"])
        calib[beta] = {"acc": float(np.mean(accs)),
                       "mean_prob": float(np.mean(probs))}
        print(f"calib beta={beta}: {calib[beta]}", flush=True)
    base_dev_acc = float(np.mean(
        [recs[pid]["forced_choice_correct"] for pid in dev]))
    print(f"dev baseline acc={base_dev_acc:.3f}", flush=True)
    beta_star = max(BETAS, key=lambda b: (calib[b]["acc"], calib[b]["mean_prob"]))
    print(f"beta*={beta_star}", flush=True)

    # ---- evaluation on the 3840 bucket ----
    rng = np.random.default_rng(SEED)
    ev = sorted(pid for pid in recs if recs[pid]["bucket"] == EVAL_BUCKET)
    rows = []
    t0 = time.time()
    for i, pid in enumerate(ev):
        r = recs[pid]
        rand_flat = rng.choice(n_layers * n_heads, len(top_cells),
                               replace=False)
        rand_cells = [(int(c // n_heads), int(c % n_heads)) for c in rand_flat]
        row = {"prompt_id": pid,
               "baseline": {"forced_choice_correct": r["forced_choice_correct"],
                            "forced_choice_prob": r["forced_choice_prob"]},
               "heads": run_prompt(loaded, prompts[pid], r, candidate_ids,
                                   top_cells, beta_star),
               "random": run_prompt(loaded, prompts[pid], r, candidate_ids,
                                    rand_cells, beta_star)}
        rows.append(row)
        el = time.time() - t0
        print(f"[{i + 1}/{len(ev)}] pid={pid} "
              f"base={row['baseline']['forced_choice_prob']:.3f} "
              f"heads={row['heads']['forced_choice_prob']:.3f} "
              f"rand={row['random']['forced_choice_prob']:.3f} "
              f"ETA {(el / (i + 1)) * (len(ev) - i - 1) / 60:.1f} min",
              flush=True)

    conds = ("baseline", "heads", "random")
    probs = {c: np.array([row[c]["forced_choice_prob"] for row in rows])
             for c in conds}
    accs = {c: np.array([row[c]["forced_choice_correct"] for row in rows])
            for c in conds}
    stat_rng = np.random.default_rng(SEED + 1)
    wrong_mask = ~accs["baseline"].astype(bool)
    summary = {
        "beta_star": beta_star, "calibration": calib,
        "dev_baseline_acc": base_dev_acc,
        "acc": {c: float(accs[c].mean()) for c in conds},
        "mean_prob": {c: float(probs[c].mean()) for c in conds},
        "heads_vs_baseline": paired_stats(probs["heads"], probs["baseline"],
                                          stat_rng),
        "heads_vs_random": paired_stats(probs["heads"], probs["random"],
                                        stat_rng),
        "model_wrong_subset": {
            "n": int(wrong_mask.sum()),
            "acc": {c: float(accs[c][wrong_mask].mean()) for c in conds},
            "mean_prob": {c: float(probs[c][wrong_mask].mean()) for c in conds},
            "heads_vs_baseline": paired_stats(probs["heads"][wrong_mask],
                                              probs["baseline"][wrong_mask],
                                              stat_rng),
        },
        "model_right_subset": {
            "n": int((~wrong_mask).sum()),
            "heads_vs_baseline": paired_stats(probs["heads"][~wrong_mask],
                                              probs["baseline"][~wrong_mask],
                                              stat_rng),
        },
    }
    out = {"summary": summary, "rows": rows,
           "design": {"top_cells": top_cells, "betas": BETAS,
                      "dev_bucket": DEV_BUCKET, "eval_bucket": EVAL_BUCKET,
                      "n_perm": N_PERM, "seed": SEED}}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
