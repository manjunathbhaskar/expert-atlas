"""Span-free variant of the attention-boost repair (WS1).

docs/ATTENTION_BOOST_CAUSAL.md proved the mechanism with an oracle: boosting
the 16 identified retrieval heads' attention onto the needle's LABELED token
span repaired 14/14 failing 3840-token prompts. A real question carries no
such label. This script derives the boost target from signals the model
produces on its own and measures how much of the oracle effect survives.

Registered design (declared before any evaluation):

  Detectors — all computed from ONE unlabeled forward pass per prompt;
  candidate windows range over positions [1, question_start) (position 0 is
  excluded as the known attention-sink artifact; the question's own location
  is part of the query, not a label):
    * heads   — total final-position attention of the 16 identified
                retrieval heads per position; candidate span = argmax
                sliding-window sum.
    * experts — per-token fraction of the 128 routing draws (16 layers x
                top-8) landing in the needle-affine expert set (defined on
                the HARD probe set's short bucket — an independent
                substrate); argmax sliding-window mean.
    * resid   — cosine similarity between the mean layer-8 hidden state
                over the question window and each position's layer-8
                hidden state; argmax sliding-window mean.
  Window width per detector chosen on the DEV bucket (1024, first 16
  prompt_ids) by span hit rate (window overlaps true needle span), grid
  {8, 12, 16, 24}. The 3840 bucket plays no part in any choice.

  Boost: identical to the oracle test — same 16 heads, same beta*=4.0,
  queries = question span through final position, keys = the DETECTED span.

  Conditions on the 3840 EVAL bucket (all 64 prompts):
    * one condition per detector;
    * wrong-span control — width-16 span drawn uniformly in
      [1, question_start-16), rejected if it overlaps the true needle span
      (the control must be guaranteed wrong), same heads/beta (fixed seed);
    * baseline — capture run's records.jsonl;
    * oracle ceiling — data/attention_boost_causal.json's "heads" rows.

  Nulls stated first: (1) a detector's boost does no better than baseline;
  (2) — the decisive one — it does no better than the wrong-span control,
  i.e. the method is generic perturbation, not span discovery. Bar per
  contrast: paired sign-flip permutation p<0.05 AND |dz|>=0.8; a detector
  only counts as FINDING the span if it beats the wrong-span control.

Output: data/spanfree_boost.json

Usage:
    HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 \
      .venv/bin/python scripts/run_spanfree_boost.py
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

from expertatlas.attention_transport import (
    HeadBoost,
    LastRowAttentionCapture,
    heads_to_by_layer,
)
from expertatlas.capture import load_model
from scripts.run_context_probe_capture import score_answer
import scripts.run_context_analyze as rca

MODEL_ID = "allenai/OLMoE-1B-7B-0924"
REPO_ROOT = Path(__file__).parent.parent
PROBE_SET = REPO_ROOT / "probes" / "probe_set_context_repaired.yaml"
RECORDS = REPO_ROOT / "data" / "context_probe_repaired" / "records.jsonl"
TRANSPORT = REPO_ROOT / "data" / "attention_transport.json"
ORACLE = REPO_ROOT / "data" / "attention_boost_causal.json"
OUT_JSON = REPO_ROOT / "data" / "spanfree_boost.json"
SIGNAL_DIR = REPO_ROOT / "data" / "spanfree_signals"

BETA = 4.0                      # fixed by the oracle test's dev calibration
WIDTHS = (8, 12, 16, 24)
WRONG_WIDTH = 16
DEV_BUCKET, EVAL_BUCKET = 1024, 3840
N_DEV = 16
N_PERM = 2000
SEED = 0
HS_LAYER = 8                    # same layer the probe/anchor work used
DETECTORS = ("heads", "experts", "resid")


def affine_mask() -> np.ndarray:
    """Needle-affine expert mask, defined on the HARD set's short bucket."""
    rca.TRACES = REPO_ROOT / "data" / "context_traces_hard"
    rca.PROBE_SET_PATH = REPO_ROOT / "probes" / "probe_set_context_hard.yaml"
    recs_list, _ = rca.load_prompt_features()
    recs = {r["prompt_id"]: r for r in recs_list}
    feats = [f for f in (rca.per_prompt_routing(r) for r in
                         sorted(recs_list, key=lambda x: x["prompt_id"]))
             if f is not None]
    short = min(recs[f["prompt_id"]]["bucket"] for f in feats)
    mask, _, _ = rca.needle_affine_set(feats, recs, short)
    return mask.reshape(rca.N_LAYERS, rca.N_EXPERTS)


def capture_signals(loaded, prompt, rec, top_cells) -> dict[str, np.ndarray]:
    """One unlabeled forward; returns the three per-position signals."""
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    f = SIGNAL_DIR / f"sig_{rec['prompt_id']:06d}.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k] for k in DETECTORS}
    tok = loaded.tokenizer
    ids = tok(prompt["text"], return_tensors="pt")
    with LastRowAttentionCapture(loaded.model) as cap, torch.no_grad():
        out = loaded.model(**ids, logits_to_keep=1,
                           output_router_logits=True,
                           output_hidden_states=True)
    n = ids["input_ids"].shape[1]
    q_start = rec["question_token_span"][0]

    heads_sig = np.zeros(n)
    for l, h in top_cells:
        heads_sig += cap.rows[l][h].numpy()

    am = _AFFINE
    k = 8
    exp_sig = np.zeros(n)
    for l, rl in enumerate(out.router_logits):
        topk = torch.topk(rl.float(), k, dim=-1).indices.numpy()  # (n, 8)
        exp_sig += am[l][topk].sum(axis=1)
    exp_sig /= 16 * k

    hs = out.hidden_states[HS_LAYER][0].float()
    q_vec = hs[q_start:].mean(dim=0)
    resid_sig = torch.nn.functional.cosine_similarity(
        hs, q_vec.unsqueeze(0), dim=-1).numpy()

    sig = {"heads": heads_sig, "experts": exp_sig, "resid": resid_sig}
    np.savez_compressed(f, **sig)
    return sig


def detect(sig: np.ndarray, width: int, q_start: int) -> tuple[int, int]:
    """Argmax sliding-window over positions [1, q_start)."""
    s = sig[1:q_start].astype(np.float64)
    if len(s) <= width:
        return 1, q_start
    win = np.convolve(s, np.ones(width), mode="valid")
    start = int(np.argmax(win)) + 1
    return start, start + width


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def run_boost(loaded, prompt, rec, candidate_ids, cells, span) -> dict:
    tok = loaded.tokenizer
    ids = tok(prompt["text"], return_tensors="pt")
    answer_id = tok(" " + prompt["answer_word"],
                    add_special_tokens=False)["input_ids"][0]
    with HeadBoost(loaded.model, heads_to_by_layer(cells), span, BETA,
                   query_start=rec["question_token_span"][0]) as hb, \
            torch.no_grad():
        out = loaded.model(**ids, logits_to_keep=1)
    assert hb.n_fired > 0
    return score_answer(out.logits[0, -1, :].float(), candidate_ids, answer_id)


def paired_stats(a: np.ndarray, b: np.ndarray, rng) -> dict:
    d = a - b
    dz = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("nan")
    flips = rng.choice([-1.0, 1.0], size=(N_PERM, len(d)))
    null = (flips * d).mean(axis=1)
    return {"mean_delta": float(d.mean()), "dz": dz,
            "perm_p": float((np.abs(null) >= abs(d.mean())).mean()),
            "n": int(len(d))}


_AFFINE: np.ndarray | None = None


def main() -> None:
    global _AFFINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(OUT_JSON))
    args = ap.parse_args()

    _AFFINE = affine_mask()
    print(f"affine experts: {int(_AFFINE.sum())}/1024", flush=True)

    transport = json.loads(TRANSPORT.read_text())
    top_cells = [tuple(c) for c in transport["summary"]["top_cells"]]
    oracle = json.loads(ORACLE.read_text())
    oracle_rows = {r["prompt_id"]: r for r in oracle["rows"]}

    ps = yaml.safe_load(PROBE_SET.read_text())
    prompts = {p["prompt_id"]: p for p in ps["prompts"]}
    recs = {r["prompt_id"]: r for r in
            (json.loads(l) for l in RECORDS.read_text().splitlines())}

    loaded = load_model(MODEL_ID, device="cpu", dtype="bfloat16")
    loaded.model.eval()
    tok = loaded.tokenizer
    candidate_ids = [tok(" " + w, add_special_tokens=False)["input_ids"][0]
                     for w in ps["candidate_words"]]

    # ---- width calibration on DEV bucket (span hit rate only) ----
    dev = sorted(p for p in recs if recs[p]["bucket"] == DEV_BUCKET)[:N_DEV]
    calib: dict[str, dict] = {d: {} for d in DETECTORS}
    for i, pid in enumerate(dev):
        sig = capture_signals(loaded, prompts[pid], recs[pid], top_cells)
        r = recs[pid]
        true_span = tuple(r["needle_token_span"])
        q_start = r["question_token_span"][0]
        for det in DETECTORS:
            for w in WIDTHS:
                span = detect(sig[det], w, q_start)
                calib[det].setdefault(w, []).append(overlaps(span, true_span))
        print(f"[dev {i + 1}/{len(dev)}] pid={pid}", flush=True)
    width_star, dev_hit = {}, {}
    for det in DETECTORS:
        rates = {w: float(np.mean(v)) for w, v in calib[det].items()}
        width_star[det] = max(WIDTHS, key=lambda w: rates[w])
        dev_hit[det] = rates
        print(f"{det}: dev hit rates {rates} -> width*={width_star[det]}",
              flush=True)

    # ---- evaluation on the 3840 bucket ----
    rng = np.random.default_rng(SEED)
    ev = sorted(p for p in recs if recs[p]["bucket"] == EVAL_BUCKET)
    rows = []
    t0 = time.time()
    for i, pid in enumerate(ev):
        r, pr = recs[pid], prompts[pid]
        true_span = tuple(r["needle_token_span"])
        q_start = r["question_token_span"][0]
        sig = capture_signals(loaded, pr, r, top_cells)
        row = {"prompt_id": pid,
               "baseline": {"forced_choice_correct": r["forced_choice_correct"],
                            "forced_choice_prob": r["forced_choice_prob"]},
               "oracle": oracle_rows[pid]["heads"]}
        for det in DETECTORS:
            span = detect(sig[det], width_star[det], q_start)
            row[det] = run_boost(loaded, pr, r, candidate_ids, top_cells, span)
            row[det]["span"] = list(span)
            row[det]["hit"] = overlaps(span, true_span)
        while True:
            ws = int(rng.integers(1, q_start - WRONG_WIDTH))
            wrong_span = (ws, ws + WRONG_WIDTH)
            if not overlaps(wrong_span, true_span):
                break
        row["wrong"] = run_boost(loaded, pr, r, candidate_ids, top_cells,
                                 wrong_span)
        row["wrong"]["span"] = list(wrong_span)
        rows.append(row)
        el = time.time() - t0
        print(f"[{i + 1}/{len(ev)}] pid={pid} "
              f"base={row['baseline']['forced_choice_prob']:.3f} "
              + " ".join(f"{d}={row[d]['forced_choice_prob']:.3f}"
                         f"{'*' if row[d]['hit'] else ' '}"
                         for d in DETECTORS)
              + f" wrong={row['wrong']['forced_choice_prob']:.3f} "
              f"ETA {(el / (i + 1)) * (len(ev) - i - 1) / 60:.1f} min",
              flush=True)

    conds = ("baseline", "oracle", "wrong", *DETECTORS)
    probs = {c: np.array([row[c]["forced_choice_prob"] for row in rows])
             for c in conds}
    accs = {c: np.array([row[c]["forced_choice_correct"] for row in rows])
            for c in conds}
    wrong_mask = ~accs["baseline"].astype(bool)
    stat_rng = np.random.default_rng(SEED + 1)
    oracle_eff = probs["oracle"].mean() - probs["baseline"].mean()
    summary: dict = {
        "width_star": width_star, "dev_hit_rates": dev_hit,
        "acc": {c: float(accs[c].mean()) for c in conds},
        "mean_prob": {c: float(probs[c].mean()) for c in conds},
        "eval_span_hit_rate": {
            d: float(np.mean([row[d]["hit"] for row in rows]))
            for d in DETECTORS},
        "model_wrong_subset": {
            "n": int(wrong_mask.sum()),
            "acc": {c: float(accs[c][wrong_mask].mean()) for c in conds},
            "mean_prob": {c: float(probs[c][wrong_mask].mean())
                          for c in conds}},
    }
    for det in DETECTORS:
        summary[f"{det}_vs_baseline"] = paired_stats(
            probs[det], probs["baseline"], stat_rng)
        summary[f"{det}_vs_wrong"] = paired_stats(
            probs[det], probs["wrong"], stat_rng)
        summary[f"{det}_vs_wrong_model_wrong"] = paired_stats(
            probs[det][wrong_mask], probs["wrong"][wrong_mask], stat_rng)
        summary[f"{det}_pct_of_oracle"] = float(
            (probs[det].mean() - probs["baseline"].mean()) / oracle_eff
        ) if oracle_eff else float("nan")
    out = {"summary": summary, "rows": rows,
           "design": {"beta": BETA, "widths": WIDTHS,
                      "wrong_width": WRONG_WIDTH, "dev_bucket": DEV_BUCKET,
                      "eval_bucket": EVAL_BUCKET, "hs_layer": HS_LAYER,
                      "n_perm": N_PERM, "seed": SEED,
                      "top_cells": top_cells}}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
