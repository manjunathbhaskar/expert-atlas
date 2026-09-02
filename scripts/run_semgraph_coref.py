"""Coreference span discovery: graph walk with discourse-adjacency edges.

Registered design: docs/COREF_REGISTRATION.md (committed with the substrate
before any evaluation). Reuses the SEMGRAPH framework and the same frozen
16-head boost; the ONE registered extension is a discourse-adjacency edge
term (gamma) in the graph walk, encoding that pronouns overwhelmingly
resolve to the immediately preceding sentence.

Stages (resumable):
  B  baseline records for the coref dev + eval arms;
  C  dev-only calibration over hops x alpha x gamma;
  D  eval on the 3840 bucket: baseline / oracle / wrong / lexical /
     semantic / graph; paired sign-flip stats.

Usage:
    .venv-dense/bin/python scripts/run_semgraph_coref.py
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
    ALPHAS,
    BETA,
    CELLS_JSON,
    EVAL_BUCKET,
    HOPS,
    MAX_DF,
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

REPO_ROOT = Path(__file__).parent.parent
COREF_SET = REPO_ROOT / "probes" / "probe_set_context_coref.yaml"
OUT_DIR = REPO_ROOT / "data" / "semgraph"
RECORDS = OUT_DIR / "records_coref.jsonl"
OUT_JSON = REPO_ROOT / "data" / "semgraph_coref.json"

GAMMAS = (0.0, 0.5, 1.0)


def detect_graph_adj(tok, ctx_ids, q_ids, sent_spans, sent_embs, q_emb,
                     h: int, alpha: float,
                     gamma: float) -> tuple[int, list[int]]:
    """Greedy walk with lexical + alpha*semantic + gamma*adjacency edges.

    adj(j) = 1 iff j immediately follows the previously selected sentence
    (0 on the first hop, which starts from the question node).
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
            adj = 1.0 if prev is not None and j == prev + 1 else 0.0
            score = lex + alpha * float(sent_embs[j] @ cur_emb) + gamma * adj
            if score >= best_score:  # >= keeps the LATEST on ties
                best, best_score = j, score
        if best is None:
            break
        visited.append(best)
        prev = best
        s, e = sent_spans[best]
        cur_link = _link_tokens(list(ctx_ids[s:e]), df, tok, exclude=q_set)
        cur_emb = sent_embs[best]
    return visited[-1] if visited else 0, visited


def detect_lexical(ctx_ids, q_ids, tok, q_start: int,
                   width: int = WRONG_WIDTH) -> tuple[int, int]:
    """Single-stage IDF detector (registered comparator; expected 0% hit)."""
    df = Counter(ctx_ids)
    q_set = {t for t in set(q_ids)
             if df[t] and df[t] < MAX_DF
             and sum(c.isalpha() for c in tok.decode([t])) >= 3}
    sig = np.zeros(len(ctx_ids))
    for i, t in enumerate(ctx_ids):
        if t in q_set:
            sig[i] = 1.0 / df[t]
    s = sig[1:q_start].astype(np.float64)
    if len(s) <= width:
        return 1, q_start
    win = np.convolve(s, np.ones(width), mode="valid")
    start = int(len(win) - 1 - np.argmax(win[::-1])) + 1
    return start, start + width


def stage_c(tok, emb, prompts, recs) -> dict:
    dev = sorted((r for r in recs.values() if r["dev"]),
                 key=lambda r: r["prompt_id"])
    feats = {r["prompt_id"]:
             prompt_features(tok, emb, prompts[r["prompt_id"]], r)
             for r in dev}
    sem_hits, grid = [], {}
    for r in dev:
        ctx_ids, q_ids, spans, s_embs, q_emb = feats[r["prompt_id"]]
        true = tuple(r["needle_token_span"])
        sem_hits.append(overlaps(spans[detect_semantic(q_emb, s_embs)], true))
        for h in HOPS:
            for a in ALPHAS:
                for g in GAMMAS:
                    j, _ = detect_graph_adj(tok, ctx_ids, q_ids, spans,
                                            s_embs, q_emb, h, a, g)
                    grid.setdefault((h, a, g), []).append(
                        overlaps(spans[j], true))
    rates = {k: float(np.mean(v)) for k, v in grid.items()}
    h_star, a_star, g_star = max(rates, key=lambda k: (rates[k], *k))
    calib = {"h": h_star, "alpha": a_star, "gamma": g_star,
             "dev_grid": {f"h{h}/a{a}/g{g}": v
                          for (h, a, g), v in sorted(rates.items())},
             "graph_dev_hit": rates[(h_star, a_star, g_star)],
             "semantic_dev_hit": float(np.mean(sem_hits))}
    print(f"calib coref: h*={h_star} alpha*={a_star} gamma*={g_star} "
          f"graph dev hit={calib['graph_dev_hit']:.3f} "
          f"semantic dev hit={calib['semantic_dev_hit']:.3f}", flush=True)
    print(calib["dev_grid"], flush=True)
    return calib


def stage_d(loaded, emb, ps, prompts, recs, top_cells, calib) -> None:
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
                                     q_emb, calib["h"], calib["alpha"],
                                     calib["gamma"])
        lex_span = detect_lexical(ctx_ids, q_ids, tok, q_start)
        while True:
            ws = int(rng.integers(1, q_start - WRONG_WIDTH))
            wrong_span = (ws, ws + WRONG_WIDTH)
            if not overlaps(wrong_span, true):
                break
        row = {"prompt_id": pid, "needle_depth": r["needle_depth"],
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
        print(f"[coref {i + 1}/{len(ev)}] pid={pid} "
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
    summary = {
        "calibration": calib,
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
    ps = yaml.safe_load(COREF_SET.read_text())
    prompts = {p["prompt_id"]: p for p in ps["prompts"]}
    loaded = load_model(MODEL_ID, device="cpu", dtype="bfloat16")
    loaded.model.eval()
    top_cells = [tuple(c)
                 for c in json.loads(CELLS_JSON.read_text())["top_cells"]]

    import run_semgraph as sg
    sg.RECORDS = RECORDS  # separate record file for the coref substrate
    recs = stage_b(loaded, ps, prompts)
    # anchor spans for failure decomposition
    tok = loaded.tokenizer
    for r in recs.values():
        if "anchor_token_span" not in r:
            p = prompts[r["prompt_id"]]
            enc = tok(p["text"], return_offsets_mapping=True)
            r["anchor_token_span"] = list(token_span_from_chars(
                enc["offset_mapping"], p["anchor_char_span"]))
    emb = Embedder()
    calib = stage_c(tok, emb, prompts, recs)
    stage_d(loaded, emb, ps, prompts, recs, top_cells, calib)


if __name__ == "__main__":
    main()
