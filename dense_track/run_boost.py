"""Dense track stage 3: oracle-span causal attention boost with matched
random-head and wrong-span controls.

Registered design (dense_track/REGISTRATION.md item 5): beta calibrated on
the 1024 DEV bucket (first 16 prompt_ids), grid {1, 2, 4}; evaluation on the
1900 bucket (all 64 prompts); conditions per prompt:
  heads  — identified 16 cells, keys = true needle span;
  random — 16 random cells (fixed seed, drawn per prompt), same beta/spans;
  wrong  — identified cells, keys = width-16 span with no needle overlap.
Baseline comes from the sweep's records.

Output: dense_track/data/boost.json

Usage:
    .venv-dense/bin/python dense_track/run_boost.py
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

from dense_track.common import (
    DATA_DIR, N_HEADS, N_LAYERS, PROBE_SET, RECORDS, HeadBoost,
    heads_to_by_layer, load_dense_model, overlaps, paired_stats, score_answer,
)

BETAS = (1.0, 2.0, 4.0)
DEV_BUCKET, EVAL_BUCKET = 1024, 1900
N_DEV = 16
WRONG_WIDTH = 16
SEED = 0


def run_prompt(model, tok, prompt, rec, candidate_ids, cells, beta,
               key_span=None) -> dict:
    ids = tok(prompt["text"], return_tensors="pt")
    answer_id = tok(" " + prompt["answer_word"],
                    add_special_tokens=False)["input_ids"][0]
    span = key_span if key_span is not None else tuple(rec["needle_token_span"])
    q_start = rec["question_token_span"][0]
    with HeadBoost(model, heads_to_by_layer(cells), span, beta,
                   query_start=q_start) as hb, torch.no_grad():
        out = model(**ids, logits_to_keep=1)
    assert hb.n_fired > 0
    return score_answer(out.logits[0, -1, :].float(), candidate_ids, answer_id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-set", type=str, default=str(PROBE_SET))
    ap.add_argument("--records", type=str, default=str(RECORDS))
    ap.add_argument("--transport", type=str,
                    default=str(DATA_DIR / "transport.json"))
    ap.add_argument("--out", type=str, default=str(DATA_DIR / "boost.json"))
    args = ap.parse_args()
    transport = json.loads(Path(args.transport).read_text())
    top_cells = [tuple(c) for c in transport["top_cells"]]
    print(f"identified cells: {top_cells}", flush=True)

    ps = yaml.safe_load(Path(args.probe_set).read_text())
    prompts = {p["prompt_id"]: p for p in ps["prompts"]}
    recs = {r["prompt_id"]: r for r in
            (json.loads(l) for l in
             Path(args.records).read_text().splitlines())}

    model, tok = load_dense_model()
    candidate_ids = [tok(" " + w, add_special_tokens=False)["input_ids"][0]
                     for w in ps["candidate_words"]]

    # ---- beta calibration on the DEV bucket ----
    dev = sorted(r for r in recs if recs[r]["bucket"] == DEV_BUCKET)[:N_DEV]
    calib = {}
    for beta in BETAS:
        accs, probs = [], []
        for pid in dev:
            res = run_prompt(model, tok, prompts[pid], recs[pid],
                             candidate_ids, top_cells, beta)
            accs.append(res["forced_choice_correct"])
            probs.append(res["forced_choice_prob"])
        calib[beta] = {"acc": float(np.mean(accs)),
                       "mean_prob": float(np.mean(probs))}
        print(f"calib beta={beta}: {calib[beta]}", flush=True)
    base_dev_acc = float(np.mean(
        [recs[pid]["forced_choice_correct"] for pid in dev]))
    beta_star = max(BETAS, key=lambda b: (calib[b]["acc"], calib[b]["mean_prob"]))
    print(f"dev baseline acc={base_dev_acc:.3f}, beta*={beta_star}", flush=True)

    # ---- evaluation on the long bucket ----
    rng = np.random.default_rng(SEED)
    ev = sorted(pid for pid in recs if recs[pid]["bucket"] == EVAL_BUCKET)
    rows = []
    t0 = time.time()
    for i, pid in enumerate(ev):
        r = recs[pid]
        true_span = tuple(r["needle_token_span"])
        rand_flat = rng.choice(N_LAYERS * N_HEADS, len(top_cells), replace=False)
        rand_cells = [(int(c // N_HEADS), int(c % N_HEADS)) for c in rand_flat]
        while True:
            ws = int(rng.integers(1, r["question_token_span"][0] - WRONG_WIDTH))
            wrong_span = (ws, ws + WRONG_WIDTH)
            if not overlaps(wrong_span, true_span):
                break
        row = {"prompt_id": pid,
               "baseline": {"forced_choice_correct": r["forced_choice_correct"],
                            "forced_choice_prob": r["forced_choice_prob"]},
               "heads": run_prompt(model, tok, prompts[pid], r, candidate_ids,
                                   top_cells, beta_star),
               "random": run_prompt(model, tok, prompts[pid], r, candidate_ids,
                                    rand_cells, beta_star),
               "wrong": run_prompt(model, tok, prompts[pid], r, candidate_ids,
                                   top_cells, beta_star, key_span=wrong_span)}
        rows.append(row)
        el = time.time() - t0
        print(f"[{i + 1}/{len(ev)}] pid={pid} "
              f"base={row['baseline']['forced_choice_prob']:.3f} "
              f"heads={row['heads']['forced_choice_prob']:.3f} "
              f"rand={row['random']['forced_choice_prob']:.3f} "
              f"wrong={row['wrong']['forced_choice_prob']:.3f} "
              f"ETA {(el / (i + 1)) * (len(ev) - i - 1) / 60:.1f} min",
              flush=True)

    conds = ("baseline", "heads", "random", "wrong")
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
        "heads_vs_wrongspan": paired_stats(probs["heads"], probs["wrong"],
                                           stat_rng),
        "model_wrong_subset": {
            "n": int(wrong_mask.sum()),
            "acc": {c: float(accs[c][wrong_mask].mean()) for c in conds}
            if wrong_mask.any() else {},
            "mean_prob": {c: float(probs[c][wrong_mask].mean()) for c in conds}
            if wrong_mask.any() else {},
            "heads_vs_baseline": paired_stats(probs["heads"][wrong_mask],
                                              probs["baseline"][wrong_mask],
                                              stat_rng)
            if wrong_mask.sum() >= 2 else None,
            "heads_vs_random": paired_stats(probs["heads"][wrong_mask],
                                            probs["random"][wrong_mask],
                                            stat_rng)
            if wrong_mask.sum() >= 2 else None,
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
                      "wrong_width": WRONG_WIDTH, "seed": SEED}}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
