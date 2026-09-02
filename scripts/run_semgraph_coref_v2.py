"""Coref v2, Experiment 1: the FROZEN v1 detector on randomized distances.

Registered design: docs/COREF_V2_REGISTRATION.md (committed with the
substrate before any evaluation). The v1-calibrated walk (h=2, alpha=0,
gamma=1, adjacency edge) runs EXACTLY as frozen -- no recalibration --
on the v2 substrate where the anchor-to-referent distance is drawn
uniformly from {1,2,3} per pair. Registered prediction: it degrades
toward the semantic-only floor for distance > 1.

Stages (resumable):
  B  baseline records for the v2 eval arm;
  D  eval on the 3840 bucket: baseline / oracle / wrong / lexical /
     semantic / graph(frozen); paired sign-flip stats + per-distance
     decomposition.

Usage:
    .venv-dense/bin/python scripts/run_semgraph_coref_v2.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from expertatlas.capture import load_model
from expertatlas.context_metrics import token_span_from_chars

from run_semgraph import (
    ALPHAS,
    BETA,
    CELLS_JSON,
    EVAL_BUCKET,
    HOPS,
    MODEL_ID,
    N_PERM,
    SEED,
    WRONG_WIDTH,
    Embedder,
    detect_semantic,
    overlaps,
    paired_stats,
    prompt_features,
    run_boost,
    stage_b,
)
from run_semgraph_coref import GAMMAS, detect_graph_adj, detect_lexical

REPO_ROOT = Path(__file__).parent.parent
COREF_V2_SET = REPO_ROOT / "probes" / "probe_set_context_coref_v2.yaml"
OUT_DIR = REPO_ROOT / "data" / "semgraph"
RECORDS = OUT_DIR / "records_coref_v2.jsonl"
OUT_JSON = REPO_ROOT / "data" / "semgraph_coref_v2.json"

# v1-calibrated parameters, FROZEN (docs/COREF_REGISTRATION.md result).
FROZEN_H = 2
FROZEN_ALPHA = 0.0
FROZEN_GAMMA = 1.0


def stage_d(loaded, emb, ps, prompts, recs, top_cells) -> None:
    tok = loaded.tokenizer
    cand = [tok(" " + w, add_special_tokens=False)["input_ids"][0]
            for w in ps["candidate_words"]]
    rng = np.random.default_rng(SEED)
    ev = sorted((r for r in recs.values()
                 if not r["dev"] and r["bucket"] == EVAL_BUCKET),
                key=lambda r: r["prompt_id"])
    rows, t0 = [], time.time()
    for i, r in enumerate(ev):
        pid = r["prompt_id"]
        pr = prompts[pid]
        true = tuple(r["needle_token_span"])
        anchor = tuple(r["anchor_token_span"])
        q_start = r["question_token_span"][0]
        ctx_ids, q_ids, spans, s_embs, q_emb = prompt_features(tok, emb, pr, r)
        sem_j = detect_semantic(q_emb, s_embs)
        g_j, path = detect_graph_adj(tok, ctx_ids, q_ids, spans, s_embs,
                                     q_emb, FROZEN_H, FROZEN_ALPHA,
                                     FROZEN_GAMMA)
        lex_span = detect_lexical(ctx_ids, q_ids, tok, q_start)
        while True:
            ws = int(rng.integers(1, q_start - WRONG_WIDTH))
            wrong_span = (ws, ws + WRONG_WIDTH)
            if not overlaps(wrong_span, true):
                break
        row = {"prompt_id": pid, "needle_depth": r["needle_depth"],
               "coref_distance": pr["coref_distance"],
               "baseline": {
                   "forced_choice_correct": r["forced_choice_correct"],
                   "forced_choice_prob": r["forced_choice_prob"]},
               "oracle": run_boost(loaded, pr, r, cand, top_cells, true),
               "wrong": run_boost(loaded, pr, r, cand, top_cells,
                                  wrong_span)}
        row["wrong"]["span"] = list(wrong_span)
        for det, span in (("lexical", lex_span),
                          ("semantic", spans[sem_j]),
                          ("graph", spans[g_j])):
            row[det] = run_boost(loaded, pr, r, cand, top_cells, span)
            row[det]["span"] = list(span)
            row[det]["hit"] = overlaps(span, true)
            row[det]["anchor_hit"] = overlaps(span, anchor)
        row["graph"]["path"] = path
        row["graph"]["path_anchor_hit"] = any(
            overlaps(spans[j], anchor) for j in path)
        rows.append(row)
        print(f"[coref_v2 {i + 1}/{len(ev)}] pid={pid} "
              f"d={pr['coref_distance']} "
              f"base={row['baseline']['forced_choice_prob']:.3f} "
              f"orc={row['oracle']['forced_choice_prob']:.3f} "
              f"wrg={row['wrong']['forced_choice_prob']:.3f} "
              f"lex={row['lexical']['forced_choice_prob']:.3f}"
              f"{'*' if row['lexical']['hit'] else ' '} "
              f"sem={row['semantic']['forced_choice_prob']:.3f}"
              f"{'*' if row['semantic']['hit'] else ' '} "
              f"grf={row['graph']['forced_choice_prob']:.3f}"
              f"{'*' if row['graph']['hit'] else ' '} "
              f"ETA {((time.time() - t0) / (i + 1)) * (len(ev) - i - 1) / 60:.1f} min",
              flush=True)

    conds = ("baseline", "oracle", "wrong", "lexical", "semantic", "graph")
    probs = {c: np.array([row[c]["forced_choice_prob"] for row in rows])
             for c in conds}
    accs = {c: np.array([row[c]["forced_choice_correct"] for row in rows])
            for c in conds}
    stat_rng = np.random.default_rng(SEED + 1)
    fail = ~accs["baseline"].astype(bool)
    oracle_eff = probs["oracle"].mean() - probs["baseline"].mean()
    dets = ("lexical", "semantic", "graph")
    dist = np.array([row["coref_distance"] for row in rows])
    summary = {
        "frozen_params": {"h": FROZEN_H, "alpha": FROZEN_ALPHA,
                          "gamma": FROZEN_GAMMA},
        "acc": {c: float(accs[c].mean()) for c in conds},
        "mean_prob": {c: float(probs[c].mean()) for c in conds},
        "hit_rate": {d: float(np.mean([row[d]["hit"] for row in rows]))
                     for d in dets},
        "anchor_hit_rate": {d: float(np.mean([row[d]["anchor_hit"]
                                              for row in rows]))
                            for d in dets},
        "graph_path_anchor_hit_rate": float(np.mean(
            [row["graph"]["path_anchor_hit"] for row in rows])),
        "failing_n": int(fail.sum()),
        "n_by_distance": {int(k): int((dist == k).sum())
                          for k in sorted(set(dist))},
        "hit_rate_by_distance": {
            d: {int(k): float(np.mean([row[d]["hit"] for row in rows
                                       if row["coref_distance"] == k]))
                for k in sorted(set(dist))}
            for d in dets},
        "acc_by_distance": {
            c: {int(k): float(accs[c][dist == k].mean())
                for k in sorted(set(dist))}
            for c in conds},
    }
    for det in dets:
        summary[f"{det}_vs_baseline"] = paired_stats(
            probs[det], probs["baseline"], stat_rng)
        summary[f"{det}_vs_wrong"] = paired_stats(
            probs[det], probs["wrong"], stat_rng)
        summary[f"{det}_pct_of_oracle"] = float(
            (probs[det].mean() - probs["baseline"].mean()) / oracle_eff) \
            if oracle_eff else float("nan")
        if fail.sum() >= 4:
            summary[f"{det}_vs_wrong_failing"] = paired_stats(
                probs[det][fail], probs["wrong"][fail], stat_rng)
            summary[f"{det}_failing_acc"] = float(accs[det][fail].mean())
    if fail.sum() >= 4:
        summary["failing_acc"] = {c: float(accs[c][fail].mean())
                                  for c in conds}
    print(json.dumps(summary, indent=2), flush=True)
    OUT_JSON.write_text(json.dumps({
        "summary": summary, "rows": rows,
        "design": {"top_cells": top_cells, "beta": BETA,
                   "wrong_width": WRONG_WIDTH, "eval_bucket": EVAL_BUCKET,
                   "n_perm": N_PERM, "seed": SEED,
                   "hops_grid": HOPS, "alphas_grid": ALPHAS,
                   "gammas_grid": GAMMAS}}, indent=2))
    print(f"wrote {OUT_JSON}", flush=True)


def main() -> None:
    ps = yaml.safe_load(COREF_V2_SET.read_text())
    prompts = {p["prompt_id"]: p for p in ps["prompts"]}
    loaded = load_model(MODEL_ID, device="cpu", dtype="bfloat16")
    loaded.model.eval()
    top_cells = [tuple(c)
                 for c in json.loads(CELLS_JSON.read_text())["top_cells"]]

    import run_semgraph as sg
    sg.RECORDS = RECORDS  # separate record file for the v2 substrate
    recs = stage_b(loaded, ps, prompts)
    tok = loaded.tokenizer
    for r in recs.values():
        if "anchor_token_span" not in r:
            p = prompts[r["prompt_id"]]
            enc = tok(p["text"], return_offsets_mapping=True)
            r["anchor_token_span"] = list(token_span_from_chars(
                enc["offset_mapping"], p["anchor_char_span"]))
    emb = Embedder()
    stage_d(loaded, emb, ps, prompts, recs, top_cells)


if __name__ == "__main__":
    main()
