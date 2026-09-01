"""Dense track stage 4 (conditional on the oracle boost working): the
existing lexical idf-overlap span detector + boost.

Detector reproduced from scripts/run_span_discovery.py (v2 semantics:
sliding-window argmax with ties broken toward the LATEST window). Width
chosen on the 1024 DEV bucket by span hit rate, grid {8, 12, 16, 24},
ties toward the largest width. Boost frozen from the oracle test (same
cells, same beta*). Baseline / oracle / wrong-span rows reused from
dense_track/data/boost.json; only the detector condition adds forwards.

Output: dense_track/data/spanfree.json

Usage:
    .venv-dense/bin/python dense_track/run_spanfree.py
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

from dense_track.common import (
    DATA_DIR, PROBE_SET, RECORDS, load_dense_model, overlaps, paired_stats,
)
from dense_track.run_boost import run_prompt

WIDTHS = (8, 12, 16, 24)
MAX_DF = 8
DEV_BUCKET = 1024
N_DEV = 16
SEED = 0


def lexical_signal(ctx_ids, q_ids, tok) -> np.ndarray:
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
    s = sig[1:q_start].astype(np.float64)
    if len(s) <= width:
        return 1, q_start
    win = np.convolve(s, np.ones(width), mode="valid")
    start = int(len(win) - 1 - np.argmax(win[::-1])) + 1
    return start, start + width


def detect_lexical(tok, text: str, rec: dict, width: int) -> tuple[int, int]:
    ids = tok(text)["input_ids"]
    q_start, q_end = rec["question_token_span"]
    sig = lexical_signal(ids, ids[q_start:q_end], tok)
    return detect_forward(sig, width, q_start)


def main() -> None:
    boost = json.loads((DATA_DIR / "boost.json").read_text())
    top_cells = [tuple(c) for c in boost["design"]["top_cells"]]
    beta_star = boost["summary"]["beta_star"]

    ps = yaml.safe_load(PROBE_SET.read_text())
    prompts = {p["prompt_id"]: p for p in ps["prompts"]}
    recs = {r["prompt_id"]: r for r in
            (json.loads(l) for l in RECORDS.read_text().splitlines())}

    model, tok = load_dense_model()
    candidate_ids = [tok(" " + w, add_special_tokens=False)["input_ids"][0]
                     for w in ps["candidate_words"]]

    # ---- width selection on the DEV bucket by span hit rate ----
    dev = sorted(r for r in recs if recs[r]["bucket"] == DEV_BUCKET)[:N_DEV]
    hit_rates = {}
    for w in WIDTHS:
        hits = [overlaps(detect_lexical(tok, prompts[pid]["text"], recs[pid], w),
                         tuple(recs[pid]["needle_token_span"]))
                for pid in dev]
        hit_rates[w] = float(np.mean(hits))
        print(f"width {w}: dev hit rate {hit_rates[w]:.3f}", flush=True)
    best = max(hit_rates.values())
    width = max(w for w in WIDTHS if hit_rates[w] == best)  # ties -> largest
    print(f"width*={width}", flush=True)

    # ---- evaluation: detector-boost condition on the long bucket ----
    rows = []
    t0 = time.time()
    prior_rows = boost["rows"]
    for i, prior in enumerate(prior_rows):
        pid = prior["prompt_id"]
        r, pr = recs[pid], prompts[pid]
        true_span = tuple(r["needle_token_span"])
        span = detect_lexical(tok, pr["text"], r, width)
        res = run_prompt(model, tok, pr, r, candidate_ids, top_cells,
                         beta_star, key_span=span)
        res["span"] = list(span)
        res["hit"] = overlaps(span, true_span)
        rows.append({"prompt_id": pid, "baseline": prior["baseline"],
                     "oracle": prior["heads"], "wrong": prior["wrong"],
                     "lexical": res})
        el = time.time() - t0
        print(f"[{i + 1}/{len(prior_rows)}] pid={pid} "
              f"base={prior['baseline']['forced_choice_prob']:.3f} "
              f"lex={res['forced_choice_prob']:.3f}"
              f"{'*' if res['hit'] else ' '} "
              f"ETA {(el / (i + 1)) * (len(prior_rows) - i - 1) / 60:.1f} min",
              flush=True)

    conds = ("baseline", "oracle", "wrong", "lexical")
    probs = {c: np.array([r[c]["forced_choice_prob"] for r in rows])
             for c in conds}
    accs = {c: np.array([r[c]["forced_choice_correct"] for r in rows])
            for c in conds}
    fail = ~accs["baseline"].astype(bool)
    rng = np.random.default_rng(SEED + 2)
    oracle_eff = probs["oracle"].mean() - probs["baseline"].mean()
    summary = {
        "width_star": width, "dev_hit_rates": hit_rates,
        "n": len(rows),
        "acc": {c: float(accs[c].mean()) for c in conds},
        "mean_prob": {c: float(probs[c].mean()) for c in conds},
        "span_hit_rate": float(np.mean([r["lexical"]["hit"] for r in rows])),
        "lexical_vs_baseline": paired_stats(probs["lexical"],
                                            probs["baseline"], rng),
        "lexical_vs_wrong": paired_stats(probs["lexical"], probs["wrong"], rng),
        "pct_of_oracle": float((probs["lexical"].mean()
                                - probs["baseline"].mean()) / oracle_eff)
        if oracle_eff else float("nan"),
        "failing_subset": {
            "n": int(fail.sum()),
            "acc": {c: float(accs[c][fail].mean()) for c in conds}
            if fail.any() else {},
            "mean_prob": {c: float(probs[c][fail].mean()) for c in conds}
            if fail.any() else {},
            "lexical_vs_wrong": paired_stats(probs["lexical"][fail],
                                             probs["wrong"][fail], rng)
            if fail.sum() >= 2 else None,
        },
    }
    out = {"summary": summary, "rows": rows}
    (DATA_DIR / "spanfree.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
