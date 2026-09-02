"""EXPLORATORY dense-track boost on the harder confusable-distractor set.

Everything is frozen from the registered pipeline: the 16 identified head
cells (dense_track/data/transport.json) and beta* from the registered
calibration (dense_track/data/boost.json). No re-identification and no
re-calibration on this set; results are exploratory by declaration
(see generate_probes_hard.py docstring).

Evaluates all 1900-token hard prompts under the same four conditions
(baseline / heads / random / wrong-span) and writes
dense_track/data/boost_hard.json.

Usage:
    .venv-dense/bin/python dense_track/run_boost_hard.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from dense_track.common import (
    DATA_DIR, N_HEADS, N_LAYERS, HeadBoost, heads_to_by_layer,
    load_dense_model, overlaps, paired_stats, score_answer,
)
from dense_track.run_boost import WRONG_WIDTH, run_prompt

PROBE_SET_HARD = Path(__file__).parent / "probe_set_dense_hard.yaml"
RECORDS_HARD = DATA_DIR / "records_hard.jsonl"
EVAL_BUCKET = 1900
SEED = 0


def main() -> None:
    transport = json.loads((DATA_DIR / "transport.json").read_text())
    top_cells = [tuple(c) for c in transport["top_cells"]]
    beta_star = json.loads((DATA_DIR / "boost.json").read_text())[
        "summary"]["beta_star"]
    print(f"frozen cells: {top_cells}\nfrozen beta*={beta_star}", flush=True)

    ps = yaml.safe_load(PROBE_SET_HARD.read_text())
    prompts = {p["prompt_id"]: p for p in ps["prompts"]}
    recs = {r["prompt_id"]: r for r in
            (json.loads(l) for l in RECORDS_HARD.read_text().splitlines())}

    model, tok = load_dense_model()
    candidate_ids = [tok(" " + w, add_special_tokens=False)["input_ids"][0]
                     for w in ps["candidate_words"]]

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
        "beta_star": beta_star,
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
                                              stat_rng)
            if (~wrong_mask).sum() >= 2 else None,
        },
    }
    out = {"summary": summary, "rows": rows,
           "design": {"top_cells": top_cells, "beta_star": beta_star,
                      "eval_bucket": EVAL_BUCKET,
                      "wrong_width": WRONG_WIDTH, "seed": SEED,
                      "exploratory": True}}
    (DATA_DIR / "boost_hard.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
