"""Granite head-level replication of the attention-transport mechanism (WS1).

The full causal chain (fact survives at source -> specific retrieval heads
collapse -> boosting them repairs the failures) has only ever been shown on
OLMoE. Granite did not context-rot on the 50%-depth substrate at any tested
length (docs/MECHANISM_GRANITE.md), so this uses the harder depth-factor set
(probes/probe_set_context_granite_depth.yaml: depths 0.15/0.50/0.85, buckets
256/3840) to obtain real failing prompts, then runs the same registered
three-stage pipeline plus the boost repair.

Registered design (declared before any capture):

  Stage 0 -- baseline accuracy on all 96 eval prompts + 16 dev prompts
    (forced-choice over the Granite-verified candidate pool). GATE: stage 2+
    require >= 4 model-wrong prompts in the 3840 bucket; if Granite still
    does not fail, that is the (negative) finding and the run stops there.

  Stage 1 -- identification (256-token bucket, model-correct only, all
    depths pooled): last-row post-softmax attention mass on the needle span
    per (layer, head); top K=16 cells by mean mass are the candidate
    retrieval heads (same fixed K as OLMoE; Granite has 32x24=768 cells, so
    K is 2.1% of cells vs 6.25% on OLMoE -- reported, not tuned).

  Stage 2 -- collapse test (3840 bucket, right vs wrong):
    NULL 1: identified-head needle mass does not differ between groups
    (label-shuffle permutation, 2000 perms, + Cohen d).
    NULL 2 (specificity): the right-wrong drop in the identified set does
    not exceed that of random K-cell sets (2000 head-identity draws).
    Per-head permutation p with BH-FDR + practical floor (drop >= half the
    head's short-prompt mass). Concentrated vs diffuse reported either way.

  Stage 3 -- boost repair (3840 bucket, all prompts):
    beta calibrated on the 16-prompt DEV arm only (depth 0.15, 1024 tokens,
    prompt_id >= 90000), identified heads + oracle span, betas {1,2,4,8},
    highest (acc, mean_prob) wins; lexical-detector width calibrated on the
    same dev arm by span hit rate (widths 8/12/16/24, latest-tie detector
    from run_span_discovery). Conditions on eval:
      baseline (stage 0), heads+oracle span, random-K-cells+oracle span
      (strength-matched control), heads+wrong span (non-overlapping random
      16-token span), heads+lexical-detected span.
    NULLs, stated first: oracle boost does not beat baseline; does not beat
    the random-head control; lexical boost does not beat the wrong-span
    control. Registered bar: paired sign-flip permutation (2000) p < 0.05
    AND |dz| >= 0.8 on the failing subset, plus repair counts and the
    fraction of the oracle delta recovered, reported however they come out.

Usage (stages resume; run sequentially, never in parallel with OLMoE jobs):
    HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 \
      .venv/bin/python scripts/run_granite_transport.py --stage 0
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

from expertatlas.attention_transport import (HeadBoost, LastRowAttentionCapture,
                                             heads_to_by_layer)
from expertatlas.context_metrics import token_span_from_chars
from expertatlas.stats import bh_fdr

from run_attention_transport import perm_diff_p
from run_context_probe_capture import score_answer
from run_second_model_check import load_granite
from run_span_discovery import detect_lexical
from run_spanfree_boost import overlaps, paired_stats

REPO_ROOT = Path(__file__).parent.parent
PROBE_SET = REPO_ROOT / "probes" / "probe_set_context_granite_depth.yaml"
OUT_DIR = REPO_ROOT / "data" / "granite_transport"
RECORDS = OUT_DIR / "records.jsonl"
OUT_TRANSPORT = REPO_ROOT / "data" / "granite_transport.json"
OUT_BOOST = REPO_ROOT / "data" / "granite_boost.json"

ID_BUCKET = 256
EVAL_BUCKET = 3840
TOP_K = 16
BETAS = (1.0, 2.0, 4.0, 8.0)
WIDTHS = (8, 12, 16, 24)
WRONG_WIDTH = 16
MIN_WRONG = 4
N_PERM = 2000
SEED = 0


def load_probe():
    ps = yaml.safe_load(PROBE_SET.read_text())
    return ps, {p["prompt_id"]: p for p in ps["prompts"]}


def load_records() -> dict[int, dict]:
    if not RECORDS.exists():
        return {}
    return {r["prompt_id"]: r for r in
            (json.loads(l) for l in RECORDS.read_text().splitlines())}


HARD_OFFSET = 500_000
HARD_PROBE = REPO_ROOT / "probes" / "probe_set_context_granite_hard.yaml"
HARD_RECORDS = REPO_ROOT / "data" / "granite_hard" / "records.jsonl"


def load_hard_merged():
    """Merge the depth set (identification + dev calibration) with the
    24-distractor escalation set (evaluation), offsetting hard prompt ids.

    The depth set's own 3840 bucket stayed at ceiling (0 model-wrong), so the
    eval bucket for stages 2-3 is the hard set only; identification (256
    bucket, correct prompts) and dev-arm calibration come from the depth set.
    """
    ps, prompts = load_probe()
    recs = load_records()
    hard_ps = yaml.safe_load(HARD_PROBE.read_text())
    hard_recs = {r["prompt_id"]: r for r in
                 (json.loads(l) for l in HARD_RECORDS.read_text().splitlines())}
    # drop the depth set's (all-correct) 3840 bucket from eval consideration
    recs = {pid: r for pid, r in recs.items()
            if r["dev"] or r["bucket"] != EVAL_BUCKET}
    for p in hard_ps["prompts"]:
        pid = p["prompt_id"] + HARD_OFFSET
        q = dict(p, prompt_id=pid)
        prompts[pid] = q
        r = dict(hard_recs[p["prompt_id"]], prompt_id=pid)
        recs[pid] = r
    return ps, prompts, recs


def get_model():
    loaded = load_granite()
    loaded.model.eval()
    return loaded


def candidate_ids_of(loaded, ps):
    tok = loaded.tokenizer
    ids = [tok(" " + w, add_special_tokens=False)["input_ids"][0]
           for w in ps["candidate_words"]]
    assert len(set(ids)) == len(ids)
    return ids


# ---------------------------------------------------------------- stage 0
def stage0(loaded, ps, prompts) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    recs = load_records()
    todo = [p for p in sorted(prompts.values(), key=lambda p: p["prompt_id"])
            if p["prompt_id"] not in recs]
    cand = candidate_ids_of(loaded, ps)
    tok = loaded.tokenizer
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
            "prompt_id": p["prompt_id"], "bucket": p["bucket"],
            "needle_depth": p.get("needle_depth", 0.5), "dev": bool(p.get("dev")),
            "n_tokens": int(inputs["input_ids"].shape[1]),
            "haystack": p["haystack"], "replicate": p["replicate"],
            "entity": p["entity"], "answer_word": p["answer_word"],
            "question_token_span": list(token_span_from_chars(
                offsets, p["question_char_span"])),
            "needle_token_span": list(token_span_from_chars(
                offsets, p["needle_char_span"])),
        })
        with RECORDS.open("a") as fh:
            fh.write(json.dumps(acc) + "\n")
        el = time.time() - t0
        print(f"[0 {i + 1}/{len(todo)}] pid={p['prompt_id']:>5} "
              f"bucket={p['bucket']:>4} depth={p['needle_depth']:.2f} "
              f"fc={int(acc['forced_choice_correct'])} "
              f"p={acc['forced_choice_prob']:.3f} "
              f"ETA {(el / (i + 1)) * (len(todo) - i - 1) / 60:.1f} min",
              flush=True)

    recs = load_records()
    print("\naccuracy by (depth, bucket):", flush=True)
    for depth in (0.15, 0.50, 0.85):
        for b in (ID_BUCKET, EVAL_BUCKET):
            g = [r for r in recs.values()
                 if not r["dev"] and r["needle_depth"] == depth
                 and r["bucket"] == b]
            if g:
                print(f"  depth {depth:.2f} bucket {b:>4}: "
                      f"acc={np.mean([r['forced_choice_correct'] for r in g]):.3f} "
                      f"(n={len(g)})", flush=True)
    n_wrong = sum(1 for r in recs.values()
                  if not r["dev"] and r["bucket"] == EVAL_BUCKET
                  and not r["forced_choice_correct"])
    print(f"model-wrong at {EVAL_BUCKET}: {n_wrong} "
          f"(gate: >= {MIN_WRONG})", flush=True)


# ---------------------------------------------------------------- capture
def capture_masses(loaded, prompts, recs: list[dict], tag: str) -> dict[int, np.ndarray]:
    tok = loaded.tokenizer
    out: dict[int, np.ndarray] = {}
    t0 = time.time()
    for i, r in enumerate(recs):
        pid = r["prompt_id"]
        f = OUT_DIR / f"mass_{tag}_{pid:06d}.npz"
        if f.exists():
            out[pid] = np.load(f)["mass"]
            continue
        ids = tok(prompts[pid]["text"], return_tensors="pt")
        with LastRowAttentionCapture(loaded.model) as cap, torch.no_grad():
            loaded.model(**ids, logits_to_keep=1)
        mass = cap.needle_mass(tuple(r["needle_token_span"])).numpy()
        np.savez_compressed(f, mass=mass)
        out[pid] = mass
        el = time.time() - t0
        print(f"[{tag} {i + 1}/{len(recs)}] pid={pid} n={r['n_tokens']} "
              f"ETA {(el / (i + 1)) * (len(recs) - i - 1) / 60:.1f} min",
              flush=True)
    return out


# ------------------------------------------------------------ stages 1+2
def stage12(loaded, prompts, recs=None) -> None:
    if recs is None:
        recs = load_records()
    ev_all = [r for r in recs.values() if not r["dev"]]

    id_recs = sorted((r for r in ev_all if r["bucket"] == ID_BUCKET
                      and r["forced_choice_correct"]),
                     key=lambda r: r["prompt_id"])
    print(f"stage 1: {len(id_recs)} short correct prompts", flush=True)
    id_mass = capture_masses(loaded, prompts, id_recs, "short")
    id_stack = np.stack([id_mass[r["prompt_id"]] for r in id_recs])
    mean_mass = id_stack.mean(axis=0)
    n_layers, n_heads = mean_mass.shape
    chance = float(np.mean([
        (r["needle_token_span"][1] - r["needle_token_span"][0]) / r["n_tokens"]
        for r in id_recs]))
    flat = mean_mass.ravel()
    order = np.argsort(flat)[::-1]
    top_cells = [(int(c // n_heads), int(c % n_heads)) for c in order[:TOP_K]]
    top_mask = np.zeros(n_layers * n_heads, dtype=bool)
    top_mask[order[:TOP_K]] = True
    print("top heads (layer, head, mean mass, x chance):", flush=True)
    for l, h in top_cells:
        print(f"  L{l} H{h}: {mean_mass[l, h]:.4f} "
              f"({mean_mass[l, h] / chance:.1f}x)", flush=True)

    ev_recs = sorted((r for r in ev_all if r["bucket"] == EVAL_BUCKET),
                     key=lambda r: r["prompt_id"])
    right = [r for r in ev_recs if r["forced_choice_correct"]]
    wrong = [r for r in ev_recs if not r["forced_choice_correct"]]
    if len(wrong) < MIN_WRONG:
        out = {"summary": {"n_long_wrong": len(wrong),
                           "gate_min_wrong": MIN_WRONG,
                           "verdict": "GATE FAILED -- too few failing prompts "
                                      "at 3840; collapse test not run"},
               "top_cells": top_cells}
        OUT_TRANSPORT.write_text(json.dumps(out, indent=2))
        print(out["summary"]["verdict"], flush=True)
        return
    ev_mass = capture_masses(loaded, prompts, ev_recs, "long")
    R = np.stack([ev_mass[r["prompt_id"]].ravel() for r in right])
    W = np.stack([ev_mass[r["prompt_id"]].ravel() for r in wrong])
    long_chance = float(np.mean([
        (r["needle_token_span"][1] - r["needle_token_span"][0]) / r["n_tokens"]
        for r in ev_recs]))

    rng = np.random.default_rng(SEED)
    m_top_r, m_top_w = R[:, top_mask].mean(axis=1), W[:, top_mask].mean(axis=1)
    m_rest_r, m_rest_w = R[:, ~top_mask].mean(axis=1), W[:, ~top_mask].mean(axis=1)
    p_top, d_top = perm_diff_p(m_top_r.copy(), m_top_w.copy(), rng)
    p_rest, d_rest = perm_diff_p(m_rest_r.copy(), m_rest_w.copy(), rng)

    obs_spec = ((m_top_r.mean() - m_top_w.mean())
                - (m_rest_r.mean() - m_rest_w.mean()))
    null_spec = []
    for _ in range(N_PERM):
        mask = np.zeros(n_layers * n_heads, dtype=bool)
        mask[rng.choice(n_layers * n_heads, TOP_K, replace=False)] = True
        null_spec.append((R[:, mask].mean() - W[:, mask].mean())
                         - (R[:, ~mask].mean() - W[:, ~mask].mean()))
    null_spec = np.array(null_spec)
    p_spec = float((null_spec >= obs_spec).mean())

    per_head = []
    rng2 = np.random.default_rng(SEED + 1)
    for c in range(n_layers * n_heads):
        p, d = perm_diff_p(R[:, c].copy(), W[:, c].copy(), rng2)
        per_head.append({"layer": int(c // n_heads), "head": int(c % n_heads),
                         "short_mean": float(flat[c]),
                         "right_mean": float(R[:, c].mean()),
                         "wrong_mean": float(W[:, c].mean()),
                         "p": p, "d": d, "identified": bool(top_mask[c])})
    rej = bh_fdr(np.array([h["p"] for h in per_head]), q=0.05)
    for h, rj in zip(per_head, rej):
        drop = h["right_mean"] - h["wrong_mean"]
        h["fdr_sig"] = bool(rj)
        h["practical"] = bool(drop >= 0.5 * h["short_mean"] and h["short_mean"] > 0)
        h["collapses"] = bool(rj and h["practical"] and drop > 0)

    summary = {
        "n_short_correct": len(id_recs), "n_long_right": len(right),
        "n_long_wrong": len(wrong), "top_k": TOP_K,
        "n_layers": n_layers, "n_heads": n_heads,
        "chance_mass_short": chance, "chance_mass_long": long_chance,
        "top_cells": top_cells,
        "top_cells_mean_mass": [float(mean_mass[l, h]) for l, h in top_cells],
        "m_top": {"right": float(m_top_r.mean()), "wrong": float(m_top_w.mean()),
                  "perm_p": p_top, "d": d_top},
        "m_rest": {"right": float(m_rest_r.mean()), "wrong": float(m_rest_w.mean()),
                   "perm_p": p_rest, "d": d_rest},
        "specificity": {"obs": float(obs_spec),
                        "null_mean": float(null_spec.mean()),
                        "null_p95": float(np.percentile(null_spec, 95)),
                        "perm_p": p_spec},
        "n_heads_collapsing": sum(h["collapses"] for h in per_head),
        "n_identified_collapsing": sum(h["collapses"] for h in per_head
                                       if h["identified"]),
    }
    out = {"summary": summary, "per_head": per_head,
           "design": {"id_bucket": ID_BUCKET, "eval_bucket": EVAL_BUCKET,
                      "top_k": TOP_K, "n_perm": N_PERM, "seed": SEED}}
    OUT_TRANSPORT.write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {OUT_TRANSPORT}", flush=True)


# ---------------------------------------------------------------- stage 3
def run_boost(loaded, prompt, rec, cand, cells, span, beta) -> dict:
    tok = loaded.tokenizer
    ids = tok(prompt["text"], return_tensors="pt")
    answer_id = tok(" " + prompt["answer_word"],
                    add_special_tokens=False)["input_ids"][0]
    with HeadBoost(loaded.model, heads_to_by_layer(cells), span, beta,
                   query_start=rec["question_token_span"][0]) as hb, \
            torch.no_grad():
        out = loaded.model(**ids, logits_to_keep=1)
    assert hb.n_fired > 0
    return score_answer(out.logits[0, -1, :].float(), cand, answer_id)


def stage3(loaded, ps, prompts, recs=None) -> None:
    if recs is None:
        recs = load_records()
    transport = json.loads(OUT_TRANSPORT.read_text())
    top_cells = [tuple(c) for c in
                 transport.get("top_cells")
                 or transport["summary"]["top_cells"]]
    n_layers = transport["summary"]["n_layers"]
    n_heads = transport["summary"]["n_heads"]
    cand = candidate_ids_of(loaded, ps)
    tok = loaded.tokenizer

    # ---- calibration on the DEV arm only ----
    dev = sorted((r for r in recs.values() if r["dev"]),
                 key=lambda r: r["prompt_id"])
    calib = {}
    for beta in BETAS:
        accs, probs = [], []
        for r in dev:
            span = tuple(r["needle_token_span"])
            res = run_boost(loaded, prompts[r["prompt_id"]], r, cand,
                            top_cells, span, beta)
            accs.append(res["forced_choice_correct"])
            probs.append(res["forced_choice_prob"])
        calib[beta] = {"acc": float(np.mean(accs)),
                       "mean_prob": float(np.mean(probs))}
        print(f"calib beta={beta}: {calib[beta]}", flush=True)
    dev_base = float(np.mean([r["forced_choice_correct"] for r in dev]))
    beta_star = max(BETAS, key=lambda b: (calib[b]["acc"], calib[b]["mean_prob"]))
    print(f"dev baseline acc={dev_base:.3f}  beta*={beta_star}", flush=True)

    width_hits = {}
    for w in WIDTHS:
        hits = []
        for r in dev:
            span = detect_lexical(tok, prompts[r["prompt_id"]]["text"], r, w)
            hits.append(overlaps(span, tuple(r["needle_token_span"])))
        width_hits[w] = float(np.mean(hits))
        print(f"calib width={w}: hit={width_hits[w]:.3f}", flush=True)
    width_star = max(WIDTHS, key=lambda w: width_hits[w])
    print(f"width*={width_star}", flush=True)

    # ---- evaluation on the 3840 bucket, all depths ----
    ev = sorted((r for r in recs.values()
                 if not r["dev"] and r["bucket"] == EVAL_BUCKET),
                key=lambda r: r["prompt_id"])
    rng = np.random.default_rng(SEED)
    rows = []
    t0 = time.time()
    for i, r in enumerate(ev):
        pid = r["prompt_id"]
        pr = prompts[pid]
        true_span = tuple(r["needle_token_span"])
        q_start = r["question_token_span"][0]
        rand_flat = rng.choice(n_layers * n_heads, TOP_K, replace=False)
        rand_cells = [(int(c // n_heads), int(c % n_heads)) for c in rand_flat]
        while True:
            ws = int(rng.integers(1, q_start - WRONG_WIDTH))
            wrong_span = (ws, ws + WRONG_WIDTH)
            if not overlaps(wrong_span, true_span):
                break
        lex_span = detect_lexical(tok, pr["text"], r, width_star)
        row = {"prompt_id": pid, "needle_depth": r["needle_depth"],
               "baseline": {"forced_choice_correct": r["forced_choice_correct"],
                            "forced_choice_prob": r["forced_choice_prob"]},
               "oracle": run_boost(loaded, pr, r, cand, top_cells,
                                   true_span, beta_star),
               "random": run_boost(loaded, pr, r, cand, rand_cells,
                                   true_span, beta_star),
               "wrong": run_boost(loaded, pr, r, cand, top_cells,
                                  wrong_span, beta_star),
               "lexical": run_boost(loaded, pr, r, cand, top_cells,
                                    lex_span, beta_star)}
        row["wrong"]["span"] = list(wrong_span)
        row["lexical"]["span"] = list(lex_span)
        row["lexical"]["hit"] = overlaps(lex_span, true_span)
        rows.append(row)
        el = time.time() - t0
        print(f"[3 {i + 1}/{len(ev)}] pid={pid} "
              f"base={row['baseline']['forced_choice_prob']:.3f} "
              f"orc={row['oracle']['forced_choice_prob']:.3f} "
              f"rnd={row['random']['forced_choice_prob']:.3f} "
              f"wrg={row['wrong']['forced_choice_prob']:.3f} "
              f"lex={row['lexical']['forced_choice_prob']:.3f}"
              f"{'*' if row['lexical']['hit'] else ' '} "
              f"ETA {(el / (i + 1)) * (len(ev) - i - 1) / 60:.1f} min",
              flush=True)

    conds = ("baseline", "oracle", "random", "wrong", "lexical")
    probs = {c: np.array([row[c]["forced_choice_prob"] for row in rows])
             for c in conds}
    accs = {c: np.array([row[c]["forced_choice_correct"] for row in rows])
            for c in conds}
    stat_rng = np.random.default_rng(SEED + 1)
    fail = ~accs["baseline"].astype(bool)
    oracle_eff = probs["oracle"].mean() - probs["baseline"].mean()
    summary = {
        "beta_star": beta_star, "calibration": calib,
        "dev_baseline_acc": dev_base,
        "width_star": width_star, "width_hits": width_hits,
        "acc": {c: float(accs[c].mean()) for c in conds},
        "mean_prob": {c: float(probs[c].mean()) for c in conds},
        "lexical_span_hit_rate": float(np.mean([row["lexical"]["hit"]
                                                for row in rows])),
        "oracle_vs_baseline": paired_stats(probs["oracle"], probs["baseline"],
                                           stat_rng),
        "oracle_vs_random": paired_stats(probs["oracle"], probs["random"],
                                         stat_rng),
        "oracle_vs_wrong": paired_stats(probs["oracle"], probs["wrong"],
                                        stat_rng),
        "lexical_vs_baseline": paired_stats(probs["lexical"], probs["baseline"],
                                            stat_rng),
        "lexical_vs_wrong": paired_stats(probs["lexical"], probs["wrong"],
                                         stat_rng),
        "lexical_pct_of_oracle": float(
            (probs["lexical"].mean() - probs["baseline"].mean()) / oracle_eff)
        if oracle_eff else float("nan"),
        "failing_subset": {
            "n": int(fail.sum()),
            "acc": {c: float(accs[c][fail].mean()) for c in conds},
            "mean_prob": {c: float(probs[c][fail].mean()) for c in conds},
            "oracle_vs_random": paired_stats(probs["oracle"][fail],
                                             probs["random"][fail], stat_rng),
            "oracle_vs_wrong": paired_stats(probs["oracle"][fail],
                                            probs["wrong"][fail], stat_rng),
            "lexical_vs_wrong": paired_stats(probs["lexical"][fail],
                                             probs["wrong"][fail], stat_rng),
        },
    }
    out = {"summary": summary, "rows": rows,
           "design": {"top_cells": top_cells, "betas": BETAS,
                      "widths": WIDTHS, "wrong_width": WRONG_WIDTH,
                      "eval_bucket": EVAL_BUCKET, "n_perm": N_PERM,
                      "seed": SEED}}
    OUT_BOOST.write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {OUT_BOOST}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=str, required=True,
                    help="0 | 12 | 3 | all")
    ap.add_argument("--probe-set", type=str, default=None,
                    help="override probe set (with --out-dir, e.g. the "
                         "escalation set)")
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--hard", action="store_true",
                    help="stages 12/3 on the merged set: identification + "
                         "dev calibration from the depth set, evaluation on "
                         "the 24-distractor escalation set")
    args = ap.parse_args()

    global PROBE_SET, OUT_DIR, RECORDS
    if args.probe_set:
        PROBE_SET = Path(args.probe_set)
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)
        RECORDS = OUT_DIR / "records.jsonl"

    recs = None
    if args.hard:
        ps, prompts, recs = load_hard_merged()
        OUT_DIR = REPO_ROOT / "data" / "granite_hard"
    else:
        ps, prompts = load_probe()
    loaded = get_model()
    print(f"loaded {loaded.model_id}", flush=True)
    if args.stage in ("0", "all"):
        stage0(loaded, ps, prompts)
    if args.stage in ("12", "all"):
        stage12(loaded, prompts, recs)
    if args.stage in ("3", "all"):
        stage3(loaded, ps, prompts, recs)


if __name__ == "__main__":
    main()
