"""Causal test of residual-stream anchoring at the readout position (WS1).

Motivation — the deconfounded probe result (docs/PROBE_REPAIRED.md): the
needle's content is fully decodable at its source position (needle_last L8,
acc 0.995 at all lengths, 1.000 on model-wrong prompts) but arrives degraded
at the final position exactly on model-wrong prompts (final L16: 0.714 wrong
vs 0.960 right). The deficient link is mid-stack transport into the readout
position. This script tests the direct repair: ADD a clean encoding of the
needle's content into the residual stream at the FINAL position, at the same
depth the source representation lives (entering decoder layer 8), and measure
whether long-context forced-choice accuracy recovers.

Registered design (before any evaluation):

  * Anchor vectors: per answer word, the CENTROID of `needle_last` layer-8
    hidden states over the SHORT (256-token) bucket's prompts with that
    answer word (8 prompts each; both pairing sets). Short bucket = clean
    encodings; evaluation bucket (3840) never contributes to its own anchor.
  * Injection: `AnchorInjector` (norm-matched), input of decoder layer 8,
    final token position only.
  * Alpha calibrated on the 1024-token DEV bucket (first 16 prompt_ids),
    true-anchor condition only, alphas {0.25, 0.5, 1.0}; the alpha with the
    highest dev forced-choice accuracy is used for evaluation. The 3840
    bucket plays no part in calibration.
  * Conditions on the 3840 EVAL bucket (all 64 prompts):
      - true    — centroid of the prompt's own answer word;
      - wrong   — centroid of word (index+3) mod 8 (a specific other answer);
      - random  — fixed-seed Gaussian direction (per prompt), same alpha.
    Baseline (no injection) comes from the capture run's records.jsonl
    (same weights, same prompts, deterministic forward).
  * Metrics: per-prompt forced-choice correctness and answer probability.
    Paired sign-flip permutation (2000 flips) on delta answer-prob for
    true-vs-baseline, true-vs-random and true-vs-wrong; Cohen dz; the
    project's causal bar is p<0.05 AND |dz|>=0.8 AND treatment beats BOTH
    matched controls. The n=14 model-wrong subset is reported separately
    (descriptive; underpowered).

Output: data/anchor_causal.json

Usage:
    HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 \
      .venv/bin/python scripts/run_anchor_causal.py
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

from expertatlas.anchoring import AnchorInjector, AnchorSpec
from expertatlas.capture import load_model
from scripts.run_context_probe_capture import score_answer

MODEL_ID = "allenai/OLMoE-1B-7B-0924"
REPO_ROOT = Path(__file__).parent.parent
PROBE_SET = REPO_ROOT / "probes" / "probe_set_context_repaired.yaml"
HIDDEN_DIR = REPO_ROOT / "data" / "context_probe_repaired"
OUT_JSON = REPO_ROOT / "data" / "anchor_causal.json"

ANCHOR_LAYER_HS = 8      # hidden-state index the centroid is read from
INJECT_LAYER = 8         # forward_pre of layers[8] sees exactly hs[8]
ALPHAS = (0.25, 0.5, 1.0)
DEV_BUCKET, EVAL_BUCKET, SRC_BUCKET = 1024, 3840, 256
N_DEV = 16
WRONG_SHIFT = 3
N_PERM = 2000
SEED = 0


def load_records() -> list[dict]:
    recs = [json.loads(l) for l in
            (HIDDEN_DIR / "records.jsonl").read_text().splitlines()]
    recs = {r["prompt_id"]: r for r in recs}
    return [recs[k] for k in sorted(recs)]


def build_centroids(recs: list[dict], words: list[str]) -> dict[str, np.ndarray]:
    cent: dict[str, list[np.ndarray]] = {w: [] for w in words}
    for r in recs:
        if r["bucket"] != SRC_BUCKET:
            continue
        arr = np.load(HIDDEN_DIR / f"hidden_{r['prompt_id']:06d}.npz")
        cent[r["answer_word"]].append(arr[f"needle_last_{ANCHOR_LAYER_HS}"])
    return {w: np.stack(v).mean(axis=0) for w, v in cent.items()}


def run_prompt(loaded, prompt: dict, candidate_ids, vector: np.ndarray | None,
               alpha: float) -> dict:
    tok = loaded.tokenizer
    ids = tok(prompt["text"], return_tensors="pt")
    n = int(ids["input_ids"].shape[1])
    answer_id = tok(" " + prompt["answer_word"], add_special_tokens=False)["input_ids"][0]

    def fwd():
        with torch.no_grad():
            return loaded.model(**ids, logits_to_keep=1)

    if vector is None:
        out = fwd()
    else:
        spec = AnchorSpec(layer=INJECT_LAYER, pos_start=n - 1, pos_end=n,
                          vector=torch.from_numpy(vector.copy()), alpha=alpha)
        with AnchorInjector(loaded.model, [spec]) as inj:
            out = fwd()
        assert inj.n_fired == 1
    return score_answer(out.logits[0, -1, :].float(), candidate_ids, answer_id)


def paired_stats(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict:
    d = a - b
    dz = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("nan")
    obs = d.mean()
    flips = rng.choice([-1.0, 1.0], size=(N_PERM, len(d)))
    null = (flips * d).mean(axis=1)
    p = float((np.abs(null) >= abs(obs)).mean())
    return {"mean_delta": float(obs), "dz": dz, "perm_p": p, "n": int(len(d))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(OUT_JSON))
    args = ap.parse_args()

    ps = yaml.safe_load(PROBE_SET.read_text())
    prompts = {p["prompt_id"]: p for p in ps["prompts"]}
    words = sorted({p["answer_word"] for p in ps["prompts"]})
    recs = load_records()
    centroids = build_centroids(recs, words)
    print(f"centroids built for {len(centroids)} words", flush=True)

    loaded = load_model(MODEL_ID, device="cpu", dtype="bfloat16")
    loaded.model.eval()
    tok = loaded.tokenizer
    candidate_ids = [tok(" " + w, add_special_tokens=False)["input_ids"][0]
                     for w in ps["candidate_words"]]

    # ---- alpha calibration on the DEV bucket, true anchor only ----
    dev = [r for r in recs if r["bucket"] == DEV_BUCKET][:N_DEV]
    calib = {}
    for alpha in ALPHAS:
        accs, probs = [], []
        for r in dev:
            p = prompts[r["prompt_id"]]
            res = run_prompt(loaded, p, candidate_ids,
                             centroids[r["answer_word"]], alpha)
            accs.append(res["forced_choice_correct"])
            probs.append(res["forced_choice_prob"])
        calib[alpha] = {"acc": float(np.mean(accs)), "mean_prob": float(np.mean(probs))}
        print(f"calib alpha={alpha}: acc={calib[alpha]['acc']:.3f} "
              f"prob={calib[alpha]['mean_prob']:.3f}", flush=True)
    base_dev_acc = float(np.mean([r["forced_choice_correct"] for r in dev]))
    print(f"dev baseline acc={base_dev_acc:.3f}", flush=True)
    alpha_star = max(ALPHAS, key=lambda a: (calib[a]["acc"], calib[a]["mean_prob"]))
    print(f"alpha*={alpha_star}", flush=True)

    # ---- evaluation on the 3840 bucket ----
    rng = np.random.default_rng(SEED)
    ev = [r for r in recs if r["bucket"] == EVAL_BUCKET]
    rows = []
    t0 = time.time()
    for i, r in enumerate(ev):
        p = prompts[r["prompt_id"]]
        w = r["answer_word"]
        wrong_w = words[(words.index(w) + WRONG_SHIFT) % len(words)]
        rand_v = rng.standard_normal(centroids[w].shape[0]).astype(np.float32)
        row = {"prompt_id": r["prompt_id"],
               "baseline": {"forced_choice_correct": r["forced_choice_correct"],
                            "forced_choice_prob": r["forced_choice_prob"]}}
        for cond, vec in (("true", centroids[w]), ("wrong", centroids[wrong_w]),
                          ("random", rand_v)):
            row[cond] = run_prompt(loaded, p, candidate_ids, vec, alpha_star)
        rows.append(row)
        el = time.time() - t0
        print(f"[{i + 1}/{len(ev)}] pid={r['prompt_id']} "
              f"base={row['baseline']['forced_choice_prob']:.3f} "
              f"true={row['true']['forced_choice_prob']:.3f} "
              f"wrong={row['wrong']['forced_choice_prob']:.3f} "
              f"rand={row['random']['forced_choice_prob']:.3f} "
              f"ETA {(el / (i + 1)) * (len(ev) - i - 1) / 60:.1f} min", flush=True)

    conds = ("baseline", "true", "wrong", "random")
    probs = {c: np.array([row[c]["forced_choice_prob"] for row in rows]) for c in conds}
    accs = {c: np.array([row[c]["forced_choice_correct"] for row in rows]) for c in conds}
    stat_rng = np.random.default_rng(SEED + 1)
    summary = {
        "alpha_star": alpha_star, "calibration": calib,
        "dev_baseline_acc": base_dev_acc,
        "acc": {c: float(accs[c].mean()) for c in conds},
        "mean_prob": {c: float(probs[c].mean()) for c in conds},
        "true_vs_baseline": paired_stats(probs["true"], probs["baseline"], stat_rng),
        "true_vs_random": paired_stats(probs["true"], probs["random"], stat_rng),
        "true_vs_wrong": paired_stats(probs["true"], probs["wrong"], stat_rng),
    }
    wrong_mask = ~accs["baseline"].astype(bool)
    summary["model_wrong_subset"] = {
        "n": int(wrong_mask.sum()),
        "acc": {c: float(accs[c][wrong_mask].mean()) for c in conds},
        "mean_prob": {c: float(probs[c][wrong_mask].mean()) for c in conds},
    }
    out = {"summary": summary, "rows": rows,
           "design": {"inject_layer": INJECT_LAYER, "anchor_layer_hs": ANCHOR_LAYER_HS,
                      "alphas": ALPHAS, "dev_bucket": DEV_BUCKET,
                      "eval_bucket": EVAL_BUCKET, "src_bucket": SRC_BUCKET,
                      "wrong_shift": WRONG_SHIFT, "n_perm": N_PERM, "seed": SEED}}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
