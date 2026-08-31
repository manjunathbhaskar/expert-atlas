"""Does the attention-boost repair extend to the depth sweep's early-needle failures?

docs/CONTEXT_DEPTH.md measured the worst baseline in the project: 0.375
accuracy at 3840 tokens with the needle at depth 0.15. This run applies the
already-frozen repair to those 16 prompts — no new calibration of any kind:
same 16 identified head cells, same beta*=4.0, same dev-chosen heads-detector
width (8) from docs/SPANFREE_BOOST.md. Conditions: oracle-span boost
(ceiling), heads-detector span-free boost, wrong-span control (width 16,
away from the needle), baseline from the depth capture's accuracy.jsonl.

Nulls first: (1) the oracle boost does not transfer off the substrate the
heads were identified on; (2) the span-free variant does no better than the
wrong-span control. Paired sign-flip permutation (2000), |dz|>=0.8 floor.

Output: data/spanfree_depth.json

Usage:
    HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 \
      .venv/bin/python scripts/run_spanfree_depth.py
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

from expertatlas.attention_transport import LastRowAttentionCapture
from expertatlas.capture import load_model
import scripts.run_spanfree_boost as sf

REPO_ROOT = Path(__file__).parent.parent
PROBE_SET = REPO_ROOT / "probes" / "probe_set_context_depth.yaml"
ACCURACY = REPO_ROOT / "data" / "context_traces_depth" / "accuracy.jsonl"
OUT_JSON = REPO_ROOT / "data" / "spanfree_depth.json"

EVAL_BUCKET = 3840
DEPTH = 0.15
WIDTH = 8          # frozen from SPANFREE_BOOST dev calibration
N_PERM = 2000
SEED = 0


def detect_heads(loaded, prompt, rec, top_cells) -> tuple[int, int]:
    tok = loaded.tokenizer
    ids = tok(prompt["text"], return_tensors="pt")
    with LastRowAttentionCapture(loaded.model) as cap, torch.no_grad():
        loaded.model(**ids, logits_to_keep=1)
    n = ids["input_ids"].shape[1]
    sig = np.zeros(n)
    for l, h in top_cells:
        sig += cap.rows[l][h].numpy()
    return sf.detect(sig, WIDTH, rec["question_token_span"][0])


def main() -> None:
    transport = json.loads(sf.TRANSPORT.read_text())
    top_cells = [tuple(c) for c in transport["summary"]["top_cells"]]

    ps = yaml.safe_load(PROBE_SET.read_text())
    prompts = {p["prompt_id"]: p for p in ps["prompts"]}
    depth = {p["prompt_id"]: p["needle_depth"] for p in ps["prompts"]}
    recs = {r["prompt_id"]: r for r in
            (json.loads(l) for l in ACCURACY.read_text().splitlines())}

    loaded = load_model(sf.MODEL_ID, device="cpu", dtype="bfloat16")
    loaded.model.eval()
    tok = loaded.tokenizer
    candidate_ids = [tok(" " + w, add_special_tokens=False)["input_ids"][0]
                     for w in ps["candidate_words"]]

    ev = sorted(p for p in recs
                if recs[p]["bucket"] == EVAL_BUCKET and depth[p] == DEPTH)
    print(f"{len(ev)} early-needle prompts at {EVAL_BUCKET}", flush=True)
    rng = np.random.default_rng(SEED)
    rows = []
    t0 = time.time()
    for i, pid in enumerate(ev):
        r, pr = recs[pid], prompts[pid]
        true_span = tuple(r["needle_token_span"])
        q_start = r["question_token_span"][0]
        row = {"prompt_id": pid,
               "baseline": {"forced_choice_correct": r["forced_choice_correct"],
                            "forced_choice_prob": r["forced_choice_prob"]}}
        row["oracle"] = sf.run_boost(loaded, pr, r, candidate_ids, top_cells,
                                     true_span)
        span = detect_heads(loaded, pr, r, top_cells)
        row["heads"] = sf.run_boost(loaded, pr, r, candidate_ids, top_cells,
                                    span)
        row["heads"]["span"] = list(span)
        row["heads"]["hit"] = sf.overlaps(span, true_span)
        while True:
            ws = int(rng.integers(1, q_start - sf.WRONG_WIDTH))
            wrong_span = (ws, ws + sf.WRONG_WIDTH)
            if not sf.overlaps(wrong_span, true_span):
                break
        row["wrong"] = sf.run_boost(loaded, pr, r, candidate_ids, top_cells,
                                    wrong_span)
        rows.append(row)
        el = time.time() - t0
        print(f"[{i + 1}/{len(ev)}] pid={pid} "
              f"base={row['baseline']['forced_choice_prob']:.3f} "
              f"oracle={row['oracle']['forced_choice_prob']:.3f} "
              f"heads={row['heads']['forced_choice_prob']:.3f}"
              f"{'*' if row['heads']['hit'] else ' '} "
              f"wrong={row['wrong']['forced_choice_prob']:.3f} "
              f"ETA {(el / (i + 1)) * (len(ev) - i - 1) / 60:.1f} min",
              flush=True)

    conds = ("baseline", "oracle", "heads", "wrong")
    probs = {c: np.array([row[c]["forced_choice_prob"] for row in rows])
             for c in conds}
    accs = {c: np.array([row[c]["forced_choice_correct"] for row in rows])
            for c in conds}
    stat_rng = np.random.default_rng(SEED + 1)
    summary = {
        "n": len(rows), "depth": DEPTH, "bucket": EVAL_BUCKET,
        "acc": {c: float(accs[c].mean()) for c in conds},
        "mean_prob": {c: float(probs[c].mean()) for c in conds},
        "heads_span_hit_rate": float(np.mean([r["heads"]["hit"]
                                              for r in rows])),
        "oracle_vs_baseline": sf.paired_stats(probs["oracle"],
                                              probs["baseline"], stat_rng),
        "oracle_vs_wrong": sf.paired_stats(probs["oracle"], probs["wrong"],
                                           stat_rng),
        "heads_vs_baseline": sf.paired_stats(probs["heads"],
                                             probs["baseline"], stat_rng),
        "heads_vs_wrong": sf.paired_stats(probs["heads"], probs["wrong"],
                                          stat_rng),
    }
    OUT_JSON.write_text(json.dumps({"summary": summary, "rows": rows,
                                    "design": {"beta": sf.BETA,
                                               "width": WIDTH,
                                               "top_cells": top_cells,
                                               "n_perm": N_PERM,
                                               "seed": SEED}}, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
