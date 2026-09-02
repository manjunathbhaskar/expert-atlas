"""Semantic + graph-walk span detectors with the frozen head boost (SEMGRAPH).

Registered design: docs/SEMGRAPH_REGISTRATION.md (committed before any
evaluation-bucket measurement). This script only changes the span LOCATOR;
the repair (16 identified retrieval heads, beta=4.0) stays frozen.

Stages (resumable; records are append-only):
  A  re-derive the 16 retrieval-head cells (repaired set, 256 bucket,
     model-correct prompts, top-16 by mean final-row needle mass);
  B  baseline records for the variants dev + eval arms (token spans +
     forced-choice score);
  C  dev-only calibration of the graph walk (h, alpha) per variant;
  D  eval: baseline / oracle / wrong-span / semantic-boost / graph-boost
     on the 3840 bucket, both variants; paired sign-flip stats.

Usage:
    .venv-dense/bin/python scripts/run_semgraph.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from expertatlas.attention_transport import (
    HeadBoost,
    LastRowAttentionCapture,
    heads_to_by_layer,
)
from expertatlas.capture import load_model
from expertatlas.context_metrics import token_span_from_chars

from run_context_probe_capture import score_answer

MAX_DF = 8  # same stopword guard as run_span_discovery.MAX_DF

MODEL_ID = "allenai/OLMoE-1B-7B-0924"
EMBED_ID = "sentence-transformers/all-MiniLM-L6-v2"
REPO_ROOT = Path(__file__).parent.parent
REPAIRED_SET = REPO_ROOT / "probes" / "probe_set_context_repaired.yaml"
VARIANTS_SET = REPO_ROOT / "probes" / "probe_set_context_variants.yaml"
OUT_DIR = REPO_ROOT / "data" / "semgraph"
CELLS_JSON = OUT_DIR / "top_cells.json"
RECORDS = OUT_DIR / "records.jsonl"
OUT_JSON = REPO_ROOT / "data" / "semgraph.json"

BETA = 4.0
TOP_K = 16
ID_BUCKET = 256
EVAL_BUCKET = 3840
WRONG_WIDTH = 16
N_PERM = 2000
SEED = 0
HOPS = (1, 2, 3)
ALPHAS = (0.0, 0.5, 1.0)
VARIANTS = ("paraphrase", "multihop")

_SENT_RE = re.compile(r"[^.]*\.(?:\s+|$)")


# ------------------------------------------------------------- sentences
def sentences(text: str, q_char_start: int) -> list[tuple[int, int, str]]:
    """(char_start, char_end, text) of sentences fully before the question."""
    out = []
    for m in _SENT_RE.finditer(text):
        s, e = m.start(), m.end()
        body = text[s:e].strip()
        if not body:
            continue
        if e > q_char_start:
            break
        out.append((s, s + len(text[s:e].rstrip()), body))
    return out


def sent_token_spans(offsets, sents) -> list[tuple[int, int]]:
    return [tuple(token_span_from_chars(offsets, (s, e))) for s, e, _ in sents]


# -------------------------------------------------------------- detectors
def detect_semantic(q_emb: np.ndarray, sent_embs: np.ndarray) -> int:
    """Argmax cosine; ties to the LATEST sentence."""
    sims = sent_embs @ q_emb
    return int(len(sims) - 1 - np.argmax(sims[::-1]))


def _link_tokens(ids: list[int], df: Counter, tok,
                 exclude: set[int]) -> set[int]:
    return {t for t in set(ids)
            if t not in exclude and df[t] < MAX_DF
            and sum(c.isalpha() for c in tok.decode([t])) >= 3}


def detect_graph(tok, ctx_ids, q_ids, sent_spans, sent_embs, q_emb,
                 h: int, alpha: float) -> tuple[int, list[int]]:
    """Greedy walk from the question node; returns (final idx, path)."""
    df = Counter(ctx_ids)
    q_set = set(q_ids)
    sent_tok = [set(ctx_ids[s:e]) for s, e in sent_spans]
    cur_link = _link_tokens(q_ids, df, tok, exclude=set())
    cur_emb = q_emb
    visited: list[int] = []
    for hop in range(h):
        best, best_score = None, -np.inf
        for j in range(len(sent_spans)):
            if j in visited:
                continue
            lex = sum(1.0 / df[t] for t in cur_link & sent_tok[j])
            score = lex + alpha * float(sent_embs[j] @ cur_emb)
            if score >= best_score:  # >= keeps the LATEST on ties
                best, best_score = j, score
        if best is None:
            break
        visited.append(best)
        s, e = sent_spans[best]
        cur_link = _link_tokens(list(ctx_ids[s:e]), df, tok, exclude=q_set)
        cur_emb = sent_embs[best]
    return visited[-1] if visited else 0, visited


def overlaps(a, b) -> bool:
    return a[0] < b[1] and b[0] < a[1]


# ------------------------------------------------------------------ boost
def run_boost(loaded, prompt, rec, cand, cells, span) -> dict:
    tok = loaded.tokenizer
    ids = tok(prompt["text"], return_tensors="pt")
    answer_id = tok(" " + prompt["answer_word"],
                    add_special_tokens=False)["input_ids"][0]
    with HeadBoost(loaded.model, heads_to_by_layer(cells), span, BETA,
                   query_start=rec["question_token_span"][0]) as hb, \
            torch.no_grad():
        out = loaded.model(**ids, logits_to_keep=1)
    assert hb.n_fired > 0
    return score_answer(out.logits[0, -1, :].float(), cand, answer_id)


def paired_stats(a: np.ndarray, b: np.ndarray, rng) -> dict:
    d = a - b
    dz = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("nan")
    flips = rng.choice([-1.0, 1.0], size=(N_PERM, len(d)))
    null = (flips * d).mean(axis=1)
    return {"mean_delta": float(d.mean()), "dz": dz,
            "perm_p": float((np.abs(null) >= abs(d.mean())).mean()),
            "n": int(len(d))}


# ---------------------------------------------------------------- stage A
def stage_a(loaded) -> list[tuple[int, int]]:
    if CELLS_JSON.exists():
        return [tuple(c) for c in json.loads(CELLS_JSON.read_text())["top_cells"]]
    ps = yaml.safe_load(REPAIRED_SET.read_text())
    prompts = [p for p in ps["prompts"] if p["bucket"] == ID_BUCKET]
    tok = loaded.tokenizer
    cand = [tok(" " + w, add_special_tokens=False)["input_ids"][0]
            for w in ps["candidate_words"]]
    masses, t0 = [], time.time()
    for i, p in enumerate(prompts):
        enc = tok(p["text"], return_tensors="pt", return_offsets_mapping=True)
        offsets = [tuple(x) for x in enc["offset_mapping"][0].tolist()]
        inputs = {k: v for k, v in enc.items() if k != "offset_mapping"}
        with LastRowAttentionCapture(loaded.model) as cap, torch.no_grad():
            out = loaded.model(**inputs, logits_to_keep=1)
        answer_id = tok(" " + p["answer_word"],
                        add_special_tokens=False)["input_ids"][0]
        acc = score_answer(out.logits[0, -1, :].float(), cand, answer_id)
        if not acc["forced_choice_correct"]:
            continue
        span = tuple(token_span_from_chars(offsets, p["needle_char_span"]))
        masses.append(cap.needle_mass(span).numpy())
        print(f"[A {i + 1}/{len(prompts)}] pid={p['prompt_id']} "
              f"kept={len(masses)} "
              f"ETA {((time.time() - t0) / (i + 1)) * (len(prompts) - i - 1) / 60:.1f} min",
              flush=True)
    mean_mass = np.stack(masses).mean(axis=0)
    n_layers, n_heads = mean_mass.shape
    order = np.argsort(mean_mass.ravel())[::-1]
    top_cells = [(int(c // n_heads), int(c % n_heads)) for c in order[:TOP_K]]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CELLS_JSON.write_text(json.dumps({
        "top_cells": top_cells,
        "top_cells_mean_mass": [float(mean_mass[l, h]) for l, h in top_cells],
        "n_short_correct": len(masses)}, indent=2))
    for l, h in top_cells:
        print(f"  L{l} H{h}: {mean_mass[l, h]:.4f}", flush=True)
    return top_cells


# ---------------------------------------------------------------- stage B
def stage_b(loaded, ps, prompts) -> dict[int, dict]:
    recs = {}
    if RECORDS.exists():
        recs = {r["prompt_id"]: r for r in
                (json.loads(l) for l in RECORDS.read_text().splitlines())}
    todo = [p for p in sorted(prompts.values(), key=lambda p: p["prompt_id"])
            if p["prompt_id"] not in recs
            and (p.get("dev") or p["bucket"] == EVAL_BUCKET)]
    tok = loaded.tokenizer
    cand = [tok(" " + w, add_special_tokens=False)["input_ids"][0]
            for w in ps["candidate_words"]]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, p in enumerate(todo):
        enc = tok(p["text"], return_tensors="pt", return_offsets_mapping=True)
        offsets = [tuple(x) for x in enc["offset_mapping"][0].tolist()]
        inputs = {k: v for k, v in enc.items() if k != "offset_mapping"}
        with torch.no_grad():
            out = loaded.model(**inputs, logits_to_keep=1)
        answer_id = tok(" " + p["answer_word"],
                        add_special_tokens=False)["input_ids"][0]
        acc = score_answer(out.logits[0, -1, :].float(), cand, answer_id)
        acc.update({
            "prompt_id": p["prompt_id"], "variant": p["variant"],
            "bucket": p["bucket"], "needle_depth": p["needle_depth"],
            "dev": bool(p.get("dev")),
            "n_tokens": int(inputs["input_ids"].shape[1]),
            "question_token_span": list(token_span_from_chars(
                offsets, p["question_char_span"])),
            "needle_token_span": list(token_span_from_chars(
                offsets, p["needle_char_span"])),
        })
        if "bridge_char_span" in p:
            acc["bridge_token_span"] = list(token_span_from_chars(
                offsets, p["bridge_char_span"]))
        with RECORDS.open("a") as fh:
            fh.write(json.dumps(acc) + "\n")
        recs[p["prompt_id"]] = acc
        print(f"[B {i + 1}/{len(todo)}] pid={p['prompt_id']:>4} "
              f"{p['variant']:>10} bucket={p['bucket']:>4} "
              f"fc={int(acc['forced_choice_correct'])} "
              f"p={acc['forced_choice_prob']:.3f} "
              f"ETA {((time.time() - t0) / (i + 1)) * (len(todo) - i - 1) / 60:.1f} min",
              flush=True)
    return recs


# ----------------------------------------------------------- embed helper
class Embedder:
    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(EMBED_ID, device="cpu")

    def __call__(self, texts: list[str]) -> np.ndarray:
        return self.m.encode(texts, normalize_embeddings=True,
                             show_progress_bar=False)


def prompt_features(tok, emb, p, rec):
    enc = tok(p["text"], return_offsets_mapping=True)
    ctx_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    qs, qe = rec["question_token_span"]
    q_ids = ctx_ids[qs:qe]
    sents = sentences(p["text"], p["question_char_span"][0])
    spans = sent_token_spans(offsets, sents)
    q_text = p["text"][p["question_char_span"][0]:p["question_char_span"][1]]
    embs = emb([s[2] for s in sents] + [q_text])
    return ctx_ids, q_ids, spans, embs[:-1], embs[-1]


# ---------------------------------------------------------------- stage C
def stage_c(tok, emb, prompts, recs) -> dict:
    calib = {}
    for variant in VARIANTS:
        dev = sorted((r for r in recs.values()
                      if r["dev"] and r["variant"] == variant),
                     key=lambda r: r["prompt_id"])
        feats = {r["prompt_id"]:
                 prompt_features(tok, emb, prompts[r["prompt_id"]], r)
                 for r in dev}
        sem_hits = []
        grid = {}
        for r in dev:
            ctx_ids, q_ids, spans, s_embs, q_emb = feats[r["prompt_id"]]
            true = tuple(r["needle_token_span"])
            sem_hits.append(overlaps(spans[detect_semantic(q_emb, s_embs)],
                                     true))
            for h in HOPS:
                for a in ALPHAS:
                    j, _ = detect_graph(tok, ctx_ids, q_ids, spans, s_embs,
                                        q_emb, h, a)
                    grid.setdefault((h, a), []).append(
                        overlaps(spans[j], true))
        rates = {k: float(np.mean(v)) for k, v in grid.items()}
        h_star, a_star = max(rates, key=lambda k: (rates[k], k[0], k[1]))
        calib[variant] = {
            "h": h_star, "alpha": a_star,
            "dev_grid": {f"h{h}/a{a}": v for (h, a), v in sorted(rates.items())},
            "graph_dev_hit": rates[(h_star, a_star)],
            "semantic_dev_hit": float(np.mean(sem_hits))}
        print(f"calib {variant}: h*={h_star} alpha*={a_star} "
              f"graph dev hit={rates[(h_star, a_star)]:.3f} "
              f"semantic dev hit={np.mean(sem_hits):.3f}", flush=True)
        print(calib[variant]["dev_grid"], flush=True)
    return calib


# ---------------------------------------------------------------- stage D
def stage_d(loaded, emb, ps, prompts, recs, top_cells, calib) -> None:
    tok = loaded.tokenizer
    cand = [tok(" " + w, add_special_tokens=False)["input_ids"][0]
            for w in ps["candidate_words"]]
    rng = np.random.default_rng(SEED)
    results = {}
    for variant in VARIANTS:
        ev = sorted((r for r in recs.values()
                     if not r["dev"] and r["variant"] == variant
                     and r["bucket"] == EVAL_BUCKET),
                    key=lambda r: r["prompt_id"])
        rows, t0 = [], time.time()
        for i, r in enumerate(ev):
            pid = r["prompt_id"]
            pr = prompts[pid]
            true = tuple(r["needle_token_span"])
            q_start = r["question_token_span"][0]
            ctx_ids, q_ids, spans, s_embs, q_emb = prompt_features(
                tok, emb, pr, r)
            sem_j = detect_semantic(q_emb, s_embs)
            g_j, path = detect_graph(tok, ctx_ids, q_ids, spans, s_embs,
                                     q_emb, calib[variant]["h"],
                                     calib[variant]["alpha"])
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
            for det, j in (("semantic", sem_j), ("graph", g_j)):
                span = spans[j]
                row[det] = run_boost(loaded, pr, r, cand, top_cells, span)
                row[det]["span"] = list(span)
                row[det]["hit"] = overlaps(span, true)
                if "bridge_token_span" in r:
                    row[det]["bridge_hit"] = overlaps(
                        span, tuple(r["bridge_token_span"]))
            if "bridge_token_span" in r and path:
                row["graph"]["path_bridge_hit"] = any(
                    overlaps(spans[j], tuple(r["bridge_token_span"]))
                    for j in path)
            row["graph"]["path"] = path
            rows.append(row)
            print(f"[{variant} {i + 1}/{len(ev)}] pid={pid} "
                  f"base={row['baseline']['forced_choice_prob']:.3f} "
                  f"orc={row['oracle']['forced_choice_prob']:.3f} "
                  f"wrg={row['wrong']['forced_choice_prob']:.3f} "
                  f"sem={row['semantic']['forced_choice_prob']:.3f}"
                  f"{'*' if row['semantic']['hit'] else ' '} "
                  f"grf={row['graph']['forced_choice_prob']:.3f}"
                  f"{'*' if row['graph']['hit'] else ' '} "
                  f"ETA {((time.time() - t0) / (i + 1)) * (len(ev) - i - 1) / 60:.1f} min",
                  flush=True)

        conds = ("baseline", "oracle", "wrong", "semantic", "graph")
        probs = {c: np.array([row[c]["forced_choice_prob"] for row in rows])
                 for c in conds}
        accs = {c: np.array([row[c]["forced_choice_correct"] for row in rows])
                for c in conds}
        stat_rng = np.random.default_rng(SEED + 1)
        fail = ~accs["baseline"].astype(bool)
        oracle_eff = probs["oracle"].mean() - probs["baseline"].mean()
        summary = {
            "calibration": calib[variant],
            "acc": {c: float(accs[c].mean()) for c in conds},
            "mean_prob": {c: float(probs[c].mean()) for c in conds},
            "hit_rate": {d: float(np.mean([row[d]["hit"] for row in rows]))
                         for d in ("semantic", "graph")},
            "failing_n": int(fail.sum()),
        }
        if any("bridge_hit" in row["semantic"] for row in rows):
            summary["bridge_hit_rate"] = {
                d: float(np.mean([row[d].get("bridge_hit", False)
                                  for row in rows]))
                for d in ("semantic", "graph")}
            summary["graph_path_bridge_hit_rate"] = float(np.mean(
                [row["graph"].get("path_bridge_hit", False) for row in rows]))
        for det in ("semantic", "graph"):
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
        results[variant] = {"summary": summary, "rows": rows}
        print(json.dumps(summary, indent=2), flush=True)

    out = {"results": results,
           "design": {"top_cells": top_cells, "beta": BETA,
                      "wrong_width": WRONG_WIDTH, "eval_bucket": EVAL_BUCKET,
                      "n_perm": N_PERM, "seed": SEED, "embedder": EMBED_ID,
                      "hops_grid": HOPS, "alphas_grid": ALPHAS}}
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_JSON}", flush=True)


def main() -> None:
    ps = yaml.safe_load(VARIANTS_SET.read_text())
    prompts = {p["prompt_id"]: p for p in ps["prompts"]}
    loaded = load_model(MODEL_ID, device="cpu", dtype="bfloat16")
    loaded.model.eval()
    top_cells = stage_a(loaded)
    recs = stage_b(loaded, ps, prompts)
    emb = Embedder()
    calib = stage_c(loaded.tokenizer, emb, prompts, recs)
    stage_d(loaded, emb, ps, prompts, recs, top_cells, calib)


if __name__ == "__main__":
    main()
