"""Harder OLMoE probe variants: paraphrase + multi-hop (WS1).

The lexical span detector met the registered bar on a substrate where the
question nearly quotes the needle (docs/SPAN_DISCOVERY_SOLVED.md, scope limit
stated there). This run measures exactly where that advantage ends, on the
two pre-declared harder variants in probe_set_context_variants.yaml:

  paraphrase -- needle wording shares ONLY the entity token with the question;
  multihop   -- the answer sentence (hop2) shares NO contentful token with
                the question; a bridge sentence (hop1) links entity -> site.

Registered design (declared before any evaluation):

  Stage 0 -- baseline forced-choice accuracy on all prompts (256/3840 eval
    buckets + 1024 dev arms). Reported per variant x depth x bucket.

  Detectors, both label-free at evaluation time:
    * lexical -- run_span_discovery.detect_lexical, width calibrated per
      variant on that variant's DEV arm only (hit rate; widths 8/12/16/24).
    * l8probe -- the registered fallback from the span-discovery
      preregistration: logistic probe on the layer-8 residual stream,
      trained on the DEV arm only (positions inside the needle span = 1,
      elsewhere = 0, balanced by subsampling), applied per position on eval
      prompts; sliding-window argmax (width calibrated on dev hit rate).
      This signal is upstream of the failing attention pathway (the fact is
      ~99.5% decodable at L8 at the source, docs/PROBE_REPAIRED.md), so it
      does not inherit the measured circularity of the attention detector.

  Stage 1 -- boost evaluation on the 3840 bucket (64 prompts: 2 variants x
    2 depths x 2 haystacks x 8 replicates), frozen OLMoE settings (the 16
    identified cells, beta=4.0 from the oracle test's dev calibration).
    Conditions: baseline, oracle span (the answer sentence), wrong span
    (non-overlapping random 16-token span), lexical span, l8probe span.
    Registered expectation (stated in the variants generator): the lexical
    detector degrades on multihop by finding hop1 instead of hop2.
    NULLs, stated first: each detector's boost does no better than baseline;
    no better than the wrong-span control. Bar: paired sign-flip permutation
    (2000) p < 0.05 AND |dz| >= 0.8 vs wrong-span. Hit rates, repair rates,
    and the fraction of the oracle delta recovered are reported per variant
    however they come out.

Usage (stages resume; never run in parallel with other model jobs):
    HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 \
      .venv/bin/python scripts/run_context_variants.py --stage 0
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
sys.path.insert(0, str(Path(__file__).parent))

from expertatlas.capture import load_model
from expertatlas.context_metrics import token_span_from_chars

from run_context_probe_capture import score_answer
from run_span_discovery import detect_lexical, detect_forward
from run_spanfree_boost import overlaps, paired_stats, run_boost

MODEL_ID = "allenai/OLMoE-1B-7B-0924"
REPO_ROOT = Path(__file__).parent.parent
PROBE_SET = REPO_ROOT / "probes" / "probe_set_context_variants.yaml"
TRANSPORT = REPO_ROOT / "data" / "attention_transport.json"
OUT_DIR = REPO_ROOT / "data" / "context_variants"
RECORDS = OUT_DIR / "records.jsonl"
OUT_JSON = REPO_ROOT / "data" / "context_variants.json"

EVAL_BUCKET = 3840
WIDTHS = (8, 12, 16, 24)
WRONG_WIDTH = 16
HS_LAYER = 8
N_PERM = 2000
SEED = 0
VARIANTS = ("paraphrase", "multihop")


def load_probe():
    ps = yaml.safe_load(PROBE_SET.read_text())
    return ps, {p["prompt_id"]: p for p in ps["prompts"]}


def load_records() -> dict[int, dict]:
    if not RECORDS.exists():
        return {}
    return {r["prompt_id"]: r for r in
            (json.loads(l) for l in RECORDS.read_text().splitlines())}


# ---------------------------------------------------------------- stage 0
def stage0(loaded, ps, prompts) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    recs = load_records()
    todo = [p for p in sorted(prompts.values(), key=lambda p: p["prompt_id"])
            if p["prompt_id"] not in recs]
    tok = loaded.tokenizer
    cand = [tok(" " + w, add_special_tokens=False)["input_ids"][0]
            for w in ps["candidate_words"]]
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
            "haystack": p["haystack"], "replicate": p["replicate"],
            "entity": p["entity"], "answer_word": p["answer_word"],
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
        el = time.time() - t0
        print(f"[0 {i + 1}/{len(todo)}] pid={p['prompt_id']:>4} "
              f"{p['variant']:>10} bucket={p['bucket']:>4} "
              f"depth={p['needle_depth']:.2f} "
              f"fc={int(acc['forced_choice_correct'])} "
              f"p={acc['forced_choice_prob']:.3f} "
              f"ETA {(el / (i + 1)) * (len(todo) - i - 1) / 60:.1f} min",
              flush=True)

    recs = load_records()
    print("\naccuracy by (variant, depth, bucket):", flush=True)
    for v in VARIANTS:
        for depth in (0.15, 0.50):
            for b in (256, EVAL_BUCKET):
                g = [r for r in recs.values()
                     if not r["dev"] and r["variant"] == v
                     and r["needle_depth"] == depth and r["bucket"] == b]
                if g:
                    print(f"  {v:>10} depth {depth:.2f} bucket {b:>4}: acc="
                          f"{np.mean([r['forced_choice_correct'] for r in g]):.3f}"
                          f" (n={len(g)})", flush=True)


# ------------------------------------------------------------- l8 probe
def l8_states(loaded, text: str) -> np.ndarray:
    tok = loaded.tokenizer
    ids = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = loaded.model(**ids, logits_to_keep=1, output_hidden_states=True)
    return out.hidden_states[HS_LAYER][0].float().numpy()


def fit_l8_probe(loaded, prompts, dev_recs: list[dict], rng):
    """Logistic probe: needle-span L8 states vs subsampled non-needle states."""
    from sklearn.linear_model import LogisticRegression

    X, y = [], []
    for r in dev_recs:
        hs = l8_states(loaded, prompts[r["prompt_id"]]["text"])
        ns, ne = r["needle_token_span"]
        pos = hs[ns:ne]
        neg_idx = [i for i in range(1, r["question_token_span"][0])
                   if not (ns <= i < ne)]
        neg_idx = rng.choice(neg_idx, size=min(len(neg_idx), 4 * len(pos)),
                             replace=False)
        X.append(pos); y.append(np.ones(len(pos)))
        X.append(hs[neg_idx]); y.append(np.zeros(len(neg_idx)))
    X = np.concatenate(X); y = np.concatenate(y)
    clf = LogisticRegression(max_iter=2000, C=0.1).fit(X, y)
    print(f"l8 probe train acc={clf.score(X, y):.3f} n={len(y)}", flush=True)
    return clf


def detect_l8(clf, hs: np.ndarray, width: int, q_start: int) -> tuple[int, int]:
    sig = clf.predict_proba(hs)[:, 1]
    sig[q_start:] = 0.0
    return detect_forward(sig, width, q_start)


# ---------------------------------------------------------------- stage 1
def calibrate(loaded, prompts, dev_recs, clf, dev_hs) -> dict:
    tok = loaded.tokenizer
    out = {}
    for det in ("lexical", "l8probe"):
        hits = {}
        for w in WIDTHS:
            h = []
            for r in dev_recs:
                true_span = tuple(r["needle_token_span"])
                if det == "lexical":
                    span = detect_lexical(tok, prompts[r["prompt_id"]]["text"],
                                          r, w)
                else:
                    span = detect_l8(clf, dev_hs[r["prompt_id"]], w,
                                     r["question_token_span"][0])
                h.append(overlaps(span, true_span))
            hits[w] = float(np.mean(h))
        w_star = max(WIDTHS, key=lambda w: hits[w])
        out[det] = {"width": w_star, "dev_hits": hits}
        print(f"calib {det}: width*={w_star} hits={hits}", flush=True)
    return out


def stage1(loaded, ps, prompts) -> None:
    recs = load_records()
    tok = loaded.tokenizer
    cand = [tok(" " + w, add_special_tokens=False)["input_ids"][0]
            for w in ps["candidate_words"]]
    top_cells = [tuple(c) for c in
                 json.loads(TRANSPORT.read_text())["summary"]["top_cells"]]
    rng = np.random.default_rng(SEED)

    results = {}
    for variant in VARIANTS:
        dev = sorted((r for r in recs.values()
                      if r["dev"] and r["variant"] == variant),
                     key=lambda r: r["prompt_id"])
        print(f"\n=== {variant}: {len(dev)} dev prompts ===", flush=True)
        dev_hs = {r["prompt_id"]: l8_states(loaded, prompts[r["prompt_id"]]["text"])
                  for r in dev}
        clf = fit_l8_probe(loaded, prompts, dev, rng)
        calib = calibrate(loaded, prompts, dev, clf, dev_hs)

        ev = sorted((r for r in recs.values()
                     if not r["dev"] and r["variant"] == variant
                     and r["bucket"] == EVAL_BUCKET),
                    key=lambda r: r["prompt_id"])
        rows = []
        t0 = time.time()
        for i, r in enumerate(ev):
            pid = r["prompt_id"]
            pr = prompts[pid]
            true_span = tuple(r["needle_token_span"])
            q_start = r["question_token_span"][0]
            while True:
                ws = int(rng.integers(1, q_start - WRONG_WIDTH))
                wrong_span = (ws, ws + WRONG_WIDTH)
                if not overlaps(wrong_span, true_span):
                    break
            lex_span = detect_lexical(tok, pr["text"], r,
                                      calib["lexical"]["width"])
            hs = l8_states(loaded, pr["text"])
            l8_span = detect_l8(clf, hs, calib["l8probe"]["width"], q_start)
            row = {"prompt_id": pid, "needle_depth": r["needle_depth"],
                   "baseline": {
                       "forced_choice_correct": r["forced_choice_correct"],
                       "forced_choice_prob": r["forced_choice_prob"]},
                   "oracle": run_boost(loaded, pr, r, cand, top_cells,
                                       true_span),
                   "wrong": run_boost(loaded, pr, r, cand, top_cells,
                                      wrong_span),
                   "lexical": run_boost(loaded, pr, r, cand, top_cells,
                                        lex_span),
                   "l8probe": run_boost(loaded, pr, r, cand, top_cells,
                                        l8_span)}
            row["wrong"]["span"] = list(wrong_span)
            for det, span in (("lexical", lex_span), ("l8probe", l8_span)):
                row[det]["span"] = list(span)
                row[det]["hit"] = overlaps(span, true_span)
                if "bridge_token_span" in r:
                    row[det]["bridge_hit"] = overlaps(
                        span, tuple(r["bridge_token_span"]))
            rows.append(row)
            el = time.time() - t0
            print(f"[{variant} {i + 1}/{len(ev)}] pid={pid} "
                  f"base={row['baseline']['forced_choice_prob']:.3f} "
                  f"orc={row['oracle']['forced_choice_prob']:.3f} "
                  f"wrg={row['wrong']['forced_choice_prob']:.3f} "
                  f"lex={row['lexical']['forced_choice_prob']:.3f}"
                  f"{'*' if row['lexical']['hit'] else ' '} "
                  f"l8={row['l8probe']['forced_choice_prob']:.3f}"
                  f"{'*' if row['l8probe']['hit'] else ' '} "
                  f"ETA {(el / (i + 1)) * (len(ev) - i - 1) / 60:.1f} min",
                  flush=True)

        conds = ("baseline", "oracle", "wrong", "lexical", "l8probe")
        probs = {c: np.array([row[c]["forced_choice_prob"] for row in rows])
                 for c in conds}
        accs = {c: np.array([row[c]["forced_choice_correct"] for row in rows])
                for c in conds}
        stat_rng = np.random.default_rng(SEED + 1)
        fail = ~accs["baseline"].astype(bool)
        oracle_eff = probs["oracle"].mean() - probs["baseline"].mean()
        summary = {
            "calibration": calib,
            "acc": {c: float(accs[c].mean()) for c in conds},
            "mean_prob": {c: float(probs[c].mean()) for c in conds},
            "hit_rate": {det: float(np.mean([row[det]["hit"] for row in rows]))
                         for det in ("lexical", "l8probe")},
            "failing_n": int(fail.sum()),
        }
        for det in ("lexical", "l8probe"):
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
            summary["failing_mean_prob"] = {c: float(probs[c][fail].mean())
                                            for c in conds}
        if any("bridge_hit" in row["lexical"] for row in rows):
            summary["bridge_hit_rate"] = {
                det: float(np.mean([row[det].get("bridge_hit", False)
                                    for row in rows]))
                for det in ("lexical", "l8probe")}
        results[variant] = {"summary": summary, "rows": rows}
        print(json.dumps(summary, indent=2), flush=True)

    out = {"results": results,
           "design": {"top_cells": top_cells, "widths": WIDTHS,
                      "wrong_width": WRONG_WIDTH, "hs_layer": HS_LAYER,
                      "eval_bucket": EVAL_BUCKET, "n_perm": N_PERM,
                      "seed": SEED}}
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_JSON}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=str, required=True, help="0 | 1 | all")
    args = ap.parse_args()
    ps, prompts = load_probe()
    loaded = load_model(MODEL_ID, device="cpu", dtype="bfloat16")
    loaded.model.eval()
    if args.stage in ("0", "all"):
        stage0(loaded, ps, prompts)
    if args.stage in ("1", "all"):
        stage1(loaded, ps, prompts)


if __name__ == "__main__":
    main()
