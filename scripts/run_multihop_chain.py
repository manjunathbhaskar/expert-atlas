"""Two-stage lexical chain detector for the multi-hop variant.

docs/CONTEXT_VARIANTS.md measured exactly where the zero-forward-pass lexical
detector fails: on the multihop variant it locks onto the bridge sentence
("The {entity} office is designated Site {site}.") 56.2% of the time and
never finds the fact sentence ("The security codeword for Site {site} is
{word}.") -- 0% needle hit, 0% of the oracle effect. The registered fallback
(L8 residual probe) recovers 61.4% of oracle but needs a labeled dev set.

This experiment tests the obvious training-free fix: CHAIN the lexical
detector. The bridge is findable from the question (shared entity token); the
fact is findable from the bridge (shared site token). Two lexical passes, one
hop each, still zero model forward passes for detection.

Registered design (declared before any evaluation on the 3840 bucket):

  Stage A (bridge): idf-weighted overlap between the question and the
  context (run_span_discovery.lexical_signal), argmax window, width wA.

  Stage B (fact): chained query = the token ids inside the stage-A window
  that (a) decode to >= 3 alphabetic characters, (b) appear < MAX_DF times
  in the context, and (c) do NOT appear in the question (question tokens
  would re-find the bridge). Signal = idf-weighted overlap of the context
  with this chained query, with positions inside the stage-A window zeroed
  (a hop must leave its source). Argmax window, width wB, ties to LATEST
  (same registered tie-break and rationale as span discovery v2: the answer
  word completes the sentence after the lexical anchor).

  Calibration: (wA, wB) on the multihop DEV arm only (16 prompts, 1024
  tokens), grid {8,12,16,24}^2, by needle (hop2) hit rate; ties break to
  largest wB then largest wA. No eval-bucket data is inspected first.

  Evaluation: the 32 multihop 3840-bucket prompts. Baseline, oracle and
  wrong-span conditions are REUSED from data/context_variants.json
  (identical prompts, seed, frozen boost settings: same 16 head cells,
  beta=4.0); the only new forwards are the chain-boost condition.

  NULLs, stated first: (1) the chain boost does no better than baseline;
  (2) no better than the wrong-span control. Registered bar (same as span
  discovery): beats wrong-span with paired sign-flip permutation (2000)
  p < 0.05 AND |dz| >= 0.8. Hit rate, repair rate and fraction of the
  oracle delta are reported however they come out, alongside the stored
  l8probe reference (61.4% of oracle) for the training-free-vs-trained
  comparison.

  Registered failure risk: stage B inherits stage A's misses (bridge hit
  was only 56.2% at 3840 with width 8) -- per-prompt chaining outcomes are
  recorded so failures decompose into "missed bridge" vs "missed hop".

Output: data/multihop_chain.json

Usage:
    HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 \
      .venv/bin/python scripts/run_multihop_chain.py
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

from run_span_discovery import MAX_DF, detect_forward, lexical_signal
from run_spanfree_boost import overlaps, paired_stats, run_boost

MODEL_ID = "allenai/OLMoE-1B-7B-0924"
REPO_ROOT = Path(__file__).parent.parent
PROBE_SET = REPO_ROOT / "probes" / "probe_set_context_variants.yaml"
VARIANTS_JSON = REPO_ROOT / "data" / "context_variants.json"
RECORDS = REPO_ROOT / "data" / "context_variants" / "records.jsonl"
TRANSPORT = REPO_ROOT / "data" / "attention_transport.json"
OUT_JSON = REPO_ROOT / "data" / "multihop_chain.json"

WIDTHS = (8, 12, 16, 24)
N_PERM = 2000
SEED = 0


def chain_detect(tok, text: str, rec: dict, w_a: int, w_b: int):
    """Two lexical hops: question -> bridge window -> fact window."""
    ids = tok(text)["input_ids"]
    q_start, q_end = rec["question_token_span"]
    q_ids = ids[q_start:q_end]

    sig_a = lexical_signal(ids, q_ids, tok)
    a_span = detect_forward(sig_a, w_a, q_start)

    df = Counter(ids)
    q_set = set(q_ids)
    chained = [t for t in set(ids[a_span[0]:a_span[1]])
               if t not in q_set and df[t] < MAX_DF
               and sum(c.isalpha() for c in tok.decode([t])) >= 3]
    sig_b = lexical_signal(ids, chained, tok)
    sig_b[a_span[0]:a_span[1]] = 0.0
    b_span = detect_forward(sig_b, w_b, q_start)
    return a_span, b_span


def main() -> None:
    ps = yaml.safe_load(PROBE_SET.read_text())
    prompts = {p["prompt_id"]: p for p in ps["prompts"]}
    recs = {r["prompt_id"]: r for r in
            (json.loads(l) for l in RECORDS.read_text().splitlines())}
    prior = json.loads(VARIANTS_JSON.read_text())["results"]["multihop"]
    top_cells = [tuple(c) for c in
                 json.loads(TRANSPORT.read_text())["summary"]["top_cells"]]

    loaded = load_model(MODEL_ID, device="cpu", dtype="bfloat16")
    loaded.model.eval()
    tok = loaded.tokenizer
    cand = [tok(" " + w, add_special_tokens=False)["input_ids"][0]
            for w in ps["candidate_words"]]

    # ---- calibration on the multihop dev arm ----
    dev = sorted((r for r in recs.values()
                  if r["dev"] and r["variant"] == "multihop"),
                 key=lambda r: r["prompt_id"])
    grid = {}
    for w_a in WIDTHS:
        for w_b in WIDTHS:
            hits = []
            for r in dev:
                _, b_span = chain_detect(tok, prompts[r["prompt_id"]]["text"],
                                         r, w_a, w_b)
                hits.append(overlaps(b_span, tuple(r["needle_token_span"])))
            grid[(w_a, w_b)] = float(np.mean(hits))
    (w_a, w_b) = max(grid, key=lambda k: (grid[k], k[1], k[0]))
    print(f"calib: wA={w_a} wB={w_b} dev needle hit={grid[(w_a, w_b)]:.3f}",
          flush=True)
    print({f"{a}/{b}": v for (a, b), v in sorted(grid.items())}, flush=True)

    # ---- evaluation on the 3840 bucket ----
    rows = []
    t0 = time.time()
    ev = prior["rows"]
    for i, pv in enumerate(ev):
        pid = pv["prompt_id"]
        r, pr = recs[pid], prompts[pid]
        true_span = tuple(r["needle_token_span"])
        a_span, b_span = chain_detect(tok, pr["text"], r, w_a, w_b)
        res = run_boost(loaded, pr, r, cand, top_cells, b_span)
        res["span"] = list(b_span)
        res["hit"] = overlaps(b_span, true_span)
        bridge = tuple(r["bridge_token_span"])
        row = {"prompt_id": pid, "needle_depth": pv["needle_depth"],
               "baseline": pv["baseline"], "oracle": pv["oracle"],
               "wrong": pv["wrong"],
               "chain": res,
               "stageA_span": list(a_span),
               "stageA_bridge_hit": overlaps(a_span, bridge),
               "chain_bridge_hit": overlaps(b_span, bridge)}
        rows.append(row)
        el = time.time() - t0
        print(f"[{i + 1}/{len(ev)}] pid={pid} "
              f"base={pv['baseline']['forced_choice_prob']:.3f} "
              f"chain={res['forced_choice_prob']:.3f}"
              f"{'*' if res['hit'] else ' '} "
              f"A{'+' if row['stageA_bridge_hit'] else '-'} "
              f"ETA {(el / (i + 1)) * (len(ev) - i - 1) / 60:.1f} min",
              flush=True)

    conds = ("baseline", "oracle", "wrong", "chain")
    probs = {c: np.array([row[c]["forced_choice_prob"] for row in rows])
             for c in conds}
    accs = {c: np.array([row[c]["forced_choice_correct"] for row in rows])
            for c in conds}
    rng = np.random.default_rng(SEED + 1)
    fail = ~accs["baseline"].astype(bool)
    oracle_eff = probs["oracle"].mean() - probs["baseline"].mean()

    summary = {
        "calibration": {"w_a": w_a, "w_b": w_b, "dev_grid":
                        {f"{a}/{b}": v for (a, b), v in sorted(grid.items())},
                        "dev_needle_hit": grid[(w_a, w_b)]},
        "acc": {c: float(accs[c].mean()) for c in conds},
        "mean_prob": {c: float(probs[c].mean()) for c in conds},
        "hit_rate": float(np.mean([row["chain"]["hit"] for row in rows])),
        "stageA_bridge_hit_rate": float(
            np.mean([row["stageA_bridge_hit"] for row in rows])),
        "hit_given_bridge_hit": float(np.mean(
            [row["chain"]["hit"] for row in rows if row["stageA_bridge_hit"]]
        )) if any(row["stageA_bridge_hit"] for row in rows) else None,
        "failing_n": int(fail.sum()),
        "chain_vs_baseline": paired_stats(probs["chain"], probs["baseline"], rng),
        "chain_vs_wrong": paired_stats(probs["chain"], probs["wrong"], rng),
        "chain_pct_of_oracle": float(
            (probs["chain"].mean() - probs["baseline"].mean()) / oracle_eff)
        if oracle_eff else float("nan"),
        "l8probe_pct_of_oracle_reference": prior["summary"]["l8probe_pct_of_oracle"],
    }
    if fail.sum() >= 4:
        summary["chain_vs_wrong_failing"] = paired_stats(
            probs["chain"][fail], probs["wrong"][fail], rng)
        summary["chain_failing_acc"] = float(accs["chain"][fail].mean())

    out = {"summary": summary, "rows": rows,
           "design": {"widths": WIDTHS, "n_perm": N_PERM, "seed": SEED,
                      "boost_source": "frozen (top_cells + beta from "
                                      "attention_transport / spanfree runs)"}}
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
