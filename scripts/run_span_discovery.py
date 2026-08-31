"""Span discovery without the failing pathway: idf-weighted question overlap.

docs/SPANFREE_BOOST.md measured the circularity wall: the only label-free
signal tried so far that works at all is the 16 retrieval heads' own
attention, which collapses exactly on the prompts that need repair. This
experiment tests the simplest signal that CANNOT share that cause, because
it uses no model forward pass at all for detection: the needle is, by the
task's construction, the context span that mentions the question's rare
tokens (the entity), so an idf-weighted lexical overlap between the
question and each context window should peak on it regardless of what the
model's attention does.

Pre-registered design (declared before evaluation):

  Detector ("lexical"): for each context position i in [1, question_start),
  score(i) = sum over question token ids t present at i of 1/count(t in
  context), with tokens appearing >= MAX_DF times in the context ignored
  (stopword guard). Candidate span = argmax sliding-window sum. Window
  width chosen on the 1024 DEV bucket (first 16 prompt_ids) by span hit
  rate, grid {8, 12, 16, 24}. No model signal of any kind is used for
  detection — criterion 4 of the primary objective is satisfied by
  construction.

  Boost: frozen from the oracle test — same 16 identified head cells,
  beta*=4.0, queries = question span through final position, keys = the
  DETECTED span. Width ties on dev hit rate break toward the LARGEST
  width: the answer word sits at the needle's end and carries no lexical
  signal (it is absent from the question), so a narrow window anchored on
  the entity tokens can truncate the token the boost most needs to cover.
  This tie-break was fixed before any eval-bucket result was inspected.

  AMENDMENT (v2, made after v1's eval run — logged honestly): v1 used
  sf.detect, whose np.argmax breaks window ties toward the EARLIEST
  window. With a sparse lexical signal (often a single entity-token
  spike) every window containing the spike ties, so v1 systematically
  chose the window ENDING at the entity token — excluding the fact
  asserted after it (the answer word), which is what the boost needs to
  cover. v1 hit the needle 100% of the time by the overlap criterion yet
  recovered only 1.3% of the oracle effect (its rows are preserved in
  data/span_discovery_v1.json as a logged negative). v2 breaks ties
  toward the LATEST window, extending coverage forward from the peak:
  the idf signal marks the mention of the question's topic, and the
  asserted fact completes the sentence after it. No other change.

  Evaluation: the held-out 3840 bucket (all 64 prompts). Baseline, oracle
  ceiling, and the strength-matched wrong-span control are REUSED from
  data/spanfree_boost.json (identical prompts, seed, and design), so the
  only new forwards are the detector-boost condition. The depth-0.15 set
  (n=16, the substrate where the attention detector collapsed to 37.5%)
  is evaluated the same way against data/spanfree_depth.json.

  Nulls, stated first: (1) the detector's boost does no better than
  baseline; (2) it does no better than the wrong-span control (generic
  perturbation, not span discovery). Registered success bar (from the
  ownership directive): >=12/14 repairs on the failing set (or >=85% of
  the oracle delta answer-prob), AND beats the wrong-span control with
  paired sign-flip permutation (2000) p<0.05 AND |dz|>=0.8.

Output: data/span_discovery.json

Usage:
    HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 \
      .venv/bin/python scripts/run_span_discovery.py
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

from expertatlas.capture import load_model
import scripts.run_spanfree_boost as sf

REPO_ROOT = Path(__file__).parent.parent
SPANFREE = REPO_ROOT / "data" / "spanfree_boost.json"
SPANFREE_DEPTH = REPO_ROOT / "data" / "spanfree_depth.json"
DEPTH_PROBE = REPO_ROOT / "probes" / "probe_set_context_depth.yaml"
DEPTH_ACC = REPO_ROOT / "data" / "context_traces_depth" / "accuracy.jsonl"
OUT_JSON = REPO_ROOT / "data" / "span_discovery.json"

WIDTHS = (8, 12, 16, 24)
MAX_DF = 8          # question tokens appearing >= this often in context carry no signal
DEV_BUCKET = 1024
N_DEV = 16
N_PERM = 2000
SEED = 0


def lexical_signal(ctx_ids: list[int], q_ids: list[int], tok) -> np.ndarray:
    """idf-weighted per-position overlap between context and question tokens.

    Only contentful question tokens (>=3 alphabetic characters when decoded)
    carry signal; whitespace and punctuation tokens are excluded.
    """
    df = Counter(ctx_ids)
    q_set = {t for t in set(q_ids)
             if df[t] and df[t] < MAX_DF
             and sum(c.isalpha() for c in tok.decode([t])) >= 3}
    sig = np.zeros(len(ctx_ids))
    for i, t in enumerate(ctx_ids):
        if t in q_set:
            sig[i] = 1.0 / df[t]
    return sig


def detect_forward(sig: np.ndarray, width: int, q_start: int) -> tuple[int, int]:
    """Argmax sliding window over [1, q_start), ties broken to LATEST start."""
    s = sig[1:q_start].astype(np.float64)
    if len(s) <= width:
        return 1, q_start
    win = np.convolve(s, np.ones(width), mode="valid")
    start = int(len(win) - 1 - np.argmax(win[::-1])) + 1
    return start, start + width


def detect_lexical(tok, prompt_text: str, rec: dict, width: int) -> tuple[int, int]:
    ids = tok(prompt_text)["input_ids"]
    q_start, q_end = rec["question_token_span"]
    sig = lexical_signal(ids, ids[q_start:q_end], tok)
    return detect_forward(sig, width, q_start)


def eval_set(loaded, prompts, recs, prior_rows, top_cells, width, name,
             candidate_ids) -> list[dict]:
    tok = loaded.tokenizer
    rows = []
    t0 = time.time()
    for i, prior in enumerate(prior_rows):
        pid = prior["prompt_id"]
        r, pr = recs[pid], prompts[pid]
        true_span = tuple(r["needle_token_span"])
        span = detect_lexical(tok, pr["text"], r, width)
        res = sf.run_boost(loaded, pr, r, candidate_ids, top_cells, span)
        res["span"] = list(span)
        res["hit"] = sf.overlaps(span, true_span)
        rows.append({"prompt_id": pid, "baseline": prior["baseline"],
                     "oracle": prior["oracle"], "wrong": prior["wrong"],
                     "lexical": res})
        el = time.time() - t0
        print(f"[{name} {i + 1}/{len(prior_rows)}] pid={pid} "
              f"base={prior['baseline']['forced_choice_prob']:.3f} "
              f"lex={res['forced_choice_prob']:.3f}"
              f"{'*' if res['hit'] else ' '} "
              f"ETA {(el / (i + 1)) * (len(prior_rows) - i - 1) / 60:.1f} min",
              flush=True)
    return rows


def summarize(rows: list[dict], rng) -> dict:
    conds = ("baseline", "oracle", "wrong", "lexical")
    probs = {c: np.array([r[c]["forced_choice_prob"] for r in rows])
             for c in conds}
    accs = {c: np.array([r[c]["forced_choice_correct"] for r in rows])
            for c in conds}
    fail = ~accs["baseline"].astype(bool)
    oracle_eff = probs["oracle"].mean() - probs["baseline"].mean()
    out = {
        "n": len(rows),
        "acc": {c: float(accs[c].mean()) for c in conds},
        "mean_prob": {c: float(probs[c].mean()) for c in conds},
        "span_hit_rate": float(np.mean([r["lexical"]["hit"] for r in rows])),
        "lexical_vs_baseline": sf.paired_stats(probs["lexical"],
                                               probs["baseline"], rng),
        "lexical_vs_wrong": sf.paired_stats(probs["lexical"], probs["wrong"],
                                            rng),
        "pct_of_oracle": float((probs["lexical"].mean()
                                - probs["baseline"].mean()) / oracle_eff)
        if oracle_eff else float("nan"),
        "failing_subset": {
            "n": int(fail.sum()),
            "acc": {c: float(accs[c][fail].mean()) for c in conds},
            "mean_prob": {c: float(probs[c][fail].mean()) for c in conds},
            "lexical_vs_wrong": sf.paired_stats(probs["lexical"][fail],
                                                probs["wrong"][fail], rng),
        },
    }
    return out


def main() -> None:
    prior = json.loads(SPANFREE.read_text())
    prior_depth = json.loads(SPANFREE_DEPTH.read_text())
    top_cells = [tuple(c) for c in prior["design"]["top_cells"]]

    ps = yaml.safe_load(sf.PROBE_SET.read_text())
    prompts = {p["prompt_id"]: p for p in ps["prompts"]}
    recs = {r["prompt_id"]: r for r in
            (json.loads(l) for l in sf.RECORDS.read_text().splitlines())}

    loaded = load_model(sf.MODEL_ID, device="cpu", dtype="bfloat16")
    loaded.model.eval()
    tok = loaded.tokenizer
    candidate_ids = [tok(" " + w, add_special_tokens=False)["input_ids"][0]
                     for w in ps["candidate_words"]]

    # ---- width calibration on DEV bucket (span hit rate only) ----
    dev = sorted(p for p in recs if recs[p]["bucket"] == DEV_BUCKET)[:N_DEV]
    rates = {}
    for w in WIDTHS:
        hits = [sf.overlaps(detect_lexical(tok, prompts[p]["text"], recs[p], w),
                            tuple(recs[p]["needle_token_span"])) for p in dev]
        rates[w] = float(np.mean(hits))
    width = max(WIDTHS, key=lambda w: (rates[w], w))
    print(f"dev hit rates {rates} -> width*={width}", flush=True)

    rng = np.random.default_rng(SEED + 1)
    rows_main = eval_set(loaded, prompts, recs, prior["rows"], top_cells,
                         width, "3840", candidate_ids)
    summary_main = summarize(rows_main, rng)
    print(json.dumps(summary_main, indent=2), flush=True)

    # ---- depth-0.15 set (hardest substrate) ----
    dps = yaml.safe_load(DEPTH_PROBE.read_text())
    dprompts = {p["prompt_id"]: p for p in dps["prompts"]}
    drecs = {r["prompt_id"]: r for r in
             (json.loads(l) for l in DEPTH_ACC.read_text().splitlines())}
    dcand = [tok(" " + w, add_special_tokens=False)["input_ids"][0]
             for w in dps["candidate_words"]]
    rows_depth = eval_set(loaded, dprompts, drecs, prior_depth["rows"],
                          top_cells, width, "depth", dcand)
    summary_depth = summarize(rows_depth, rng)
    print(json.dumps(summary_depth, indent=2), flush=True)

    OUT_JSON.write_text(json.dumps({
        "summary": {"main_3840": summary_main, "depth_015": summary_depth,
                    "dev_hit_rates": rates, "width_star": width},
        "rows_main": rows_main, "rows_depth": rows_depth,
        "design": {"max_df": MAX_DF, "widths": WIDTHS, "beta": sf.BETA,
                   "n_perm": N_PERM, "seed": SEED,
                   "top_cells": top_cells}}, indent=2))
    print(f"wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
