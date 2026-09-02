"""Coref v2, Experiment 2: distance-tolerant decaying discourse edge.

Registered design: docs/COREF_V2_EXP2_REGISTRATION.md (committed after
Exp 1's result was recorded and before this evaluation). The hard
adjacency term is replaced by prox(j) = decay^(j-(prev+1)) for j > prev
(0 otherwise); calibration of h x alpha x gamma x decay on the 16-prompt
dev arm only; evaluation on the 3840 bucket with the frozen repair
pipeline.

Usage:
    .venv-dense/bin/python scripts/run_semgraph_coref_v2_exp2.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from expertatlas.capture import load_model
from expertatlas.context_metrics import token_span_from_chars

from run_semgraph import (
    BETA,
    CELLS_JSON,
    EVAL_BUCKET,
    MODEL_ID,
    N_PERM,
    SEED,
    WRONG_WIDTH,
    Embedder,
    _link_tokens,
    detect_semantic,
    overlaps,
    paired_stats,
    prompt_features,
    run_boost,
    stage_b,
)
from run_semgraph_coref import detect_lexical

REPO_ROOT = Path(__file__).parent.parent
COREF_V2_SET = REPO_ROOT / "probes" / "probe_set_context_coref_v2.yaml"
OUT_DIR = REPO_ROOT / "data" / "semgraph"
RECORDS = OUT_DIR / "records_coref_v2.jsonl"
OUT_JSON = REPO_ROOT / "data" / "semgraph_coref_v2_exp2.json"

HOPS = (1, 2, 3)
ALPHAS = (0.0, 0.5, 1.0)
GAMMAS = (0.5, 1.0)
DECAYS = (0.3, 0.5, 0.7)


def detect_graph_prox(tok, ctx_ids, q_ids, sent_spans, sent_embs, q_emb,
                      h: int, alpha: float, gamma: float,
                      decay: float) -> tuple[int, list[int]]:
    """Greedy walk with lexical + alpha*semantic + gamma*prox edges.

    prox(j) = decay^(j - (prev+1)) for j > prev, else 0; the v1 hard
    adjacency is the decay -> 0 limit.
    """
    df = Counter(ctx_ids)
    q_set = set(q_ids)
    sent_tok = [set(ctx_ids[s:e]) for s, e in sent_spans]
    cur_link = _link_tokens(q_ids, df, tok, exclude=set())
    cur_emb = q_emb
    prev = None
    visited: list[int] = []
    for _hop in range(h):
        best, best_score = None, -np.inf
        for j in range(len(sent_spans)):
            if j in visited:
                continue
            lex = sum(1.0 / df[t] for t in cur_link & sent_tok[j])
            prox = (decay ** (j - (prev + 1))
                    if prev is not None and j > prev else 0.0)
            score = lex + alpha * float(sent_embs[j] @ cur_emb) + gamma * prox
            if score >= best_score:
                best, best_score = j, score
        if best is None:
            break
        visited.append(best)
        prev = best
        s, e = sent_spans[best]
        cur_link = _link_tokens(list(ctx_ids[s:e]), df, tok, exclude=q_set)
        cur_emb = sent_embs[best]
    return visited[-1] if visited else 0, visited


def stage_c(loaded, emb, prompts, recs) -> tuple[int, float, float, float]:
    """Dev-only calibration of (h, alpha, gamma, decay) by answer hit."""
    tok = loaded.tokenizer
    dev = sorted((r for r in recs.values() if r["dev"]),
                 key=lambda r: r["prompt_id"])
    hits: dict[tuple, int] = {}
    for r in dev:
        pr = prompts[r["prompt_id"]]
        true = tuple(r["needle_token_span"])
        ctx_ids, q_ids, spans, s_embs, q_emb = prompt_features(tok, emb, pr, r)
        for h in HOPS:
            for a in ALPHAS:
                for g in GAMMAS:
                    for dc in DECAYS:
                        j, _ = detect_graph_prox(
                            tok, ctx_ids, q_ids, spans, s_embs, q_emb,
                            h, a, g, dc)
                        hits[(h, a, g, dc)] = hits.get((h, a, g, dc), 0) \
                            + int(overlaps(spans[j], true))
    for key in sorted(hits):
        print(f"[C] h={key[0]} alpha={key[1]} gamma={key[2]} "
              f"decay={key[3]}: dev hit {hits[key]}/{len(dev)}", flush=True)
    best = max(sorted(hits), key=lambda k: hits[k])
    print(f"[C] selected h={best[0]} alpha={best[1]} gamma={best[2]} "
          f"decay={best[3]} (dev hit {hits[best]}/{len(dev)})", flush=True)
    return best


def stage_d(loaded, emb, ps, prompts, recs, top_cells, params) -> None:
    h, alpha, gamma, decay = params
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
        g_j, path = detect_graph_prox(tok, ctx_ids, q_ids, spans, s_embs,
                                      q_emb, h, alpha, gamma, decay)
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
        print(f"[exp2 {i + 1}/{len(ev)}] pid={pid} "
              f"d={pr['coref_distance']} "
              f"base={row['baseline']['forced_choice_prob']:.3f} "
              f"orc={row['oracle']['forced_choice_prob']:.3f} "
              f"wrg={row['wrong']['forced_choice_prob']:.3f} "
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
        "selected_params": {"h": params[0], "alpha": params[1],
                            "gamma": params[2], "decay": params[3]},
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
                   "gammas_grid": GAMMAS, "decays_grid": DECAYS}},
        indent=2))
    print(f"wrote {OUT_JSON}", flush=True)


def main() -> None:
    ps = yaml.safe_load(COREF_V2_SET.read_text())
    prompts = {p["prompt_id"]: p for p in ps["prompts"]}
    loaded = load_model(MODEL_ID, device="cpu", dtype="bfloat16")
    loaded.model.eval()
    top_cells = [tuple(c)
                 for c in json.loads(CELLS_JSON.read_text())["top_cells"]]

    import run_semgraph as sg
    sg.RECORDS = RECORDS  # reuse Exp 1's baseline records (same substrate)
    recs = stage_b(loaded, ps, prompts)
    tok = loaded.tokenizer
    for r in recs.values():
        if "anchor_token_span" not in r:
            p = prompts[r["prompt_id"]]
            enc = tok(p["text"], return_offsets_mapping=True)
            r["anchor_token_span"] = list(token_span_from_chars(
                enc["offset_mapping"], p["anchor_char_span"]))
    emb = Embedder()
    params = stage_c(loaded, emb, prompts, recs)
    stage_d(loaded, emb, ps, prompts, recs, top_cells, params)


if __name__ == "__main__":
    main()
