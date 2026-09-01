"""Dense track stages 1-2: retrieval-head identification + collapse test.

Pure analysis over the sweep's outputs (no model forwards).

Stage 1 (registered): rank all 1024 (layer, head) cells by mean final-position
needle attention mass over SHORT-bucket (256) model-CORRECT prompts; take the
top K=16.

Stage 2 (registered): on the LONG bucket (1900), compare the identified cells'
mean needle mass between model-right and model-wrong prompts (Welch t +
Cohen d + label permutation), and test specificity with 2000 random 16-cell
subsets of the remaining cells.

Output: dense_track/data/transport.json

Usage:
    .venv-dense/bin/python dense_track/run_transport.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from dense_track.common import DATA_DIR, N_HEADS, N_LAYERS, PROBE_SET, RECORDS

SHORT_BUCKET, LONG_BUCKET = 256, 1900
TOP_K = 16
N_PERM = 2000
SEED = 0


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                 / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def main() -> None:
    ps = yaml.safe_load(PROBE_SET.read_text())
    recs = [json.loads(l) for l in RECORDS.read_text().splitlines()]
    masses = np.load(DATA_DIR / "needle_mass.npz")
    needle_len = {r["prompt_id"]: r["needle_token_span"][1]
                  - r["needle_token_span"][0] for r in recs}

    # ---- stage 1: identification on short correct prompts ----
    id_recs = [r for r in recs
               if r["bucket"] == SHORT_BUCKET and r["forced_choice_correct"]]
    print(f"stage 1: {len(id_recs)} short correct prompts", flush=True)
    stack = np.stack([masses[str(r["prompt_id"])] for r in id_recs])
    mean_mass = stack.mean(axis=0)
    order = np.argsort(mean_mass.ravel())[::-1]
    top_cells = [(int(c // N_HEADS), int(c % N_HEADS)) for c in order[:TOP_K]]
    mean_needle_tokens = float(np.mean(
        [needle_len[r["prompt_id"]] for r in id_recs]))
    chance = {r["prompt_id"]: needle_len[r["prompt_id"]] / r["n_tokens"]
              for r in recs}
    mean_chance_short = float(np.mean([chance[r["prompt_id"]] for r in id_recs]))
    print(f"top {TOP_K} cells (chance mass ~{mean_chance_short:.4f}):")
    for l, h in top_cells:
        print(f"  L{l:>2} H{h:>2}: mass={mean_mass[l, h]:.3f} "
              f"({mean_mass[l, h] / mean_chance_short:.1f}x chance)")

    # ---- stage 2: collapse on the long bucket ----
    long_recs = [r for r in recs if r["bucket"] == LONG_BUCKET]
    right = [r for r in long_recs if r["forced_choice_correct"]]
    wrong = [r for r in long_recs if not r["forced_choice_correct"]]
    print(f"\nstage 2: long bucket n_right={len(right)} n_wrong={len(wrong)}",
          flush=True)

    mask = np.zeros((N_LAYERS, N_HEADS), dtype=bool)
    for l, h in top_cells:
        mask[l, h] = True

    def per_prompt(rs, m):
        return np.array([masses[str(r["prompt_id"])][m].mean() for r in rs])

    R_id, W_id = per_prompt(right, mask), per_prompt(wrong, mask)
    R_ot, W_ot = per_prompt(right, ~mask), per_prompt(wrong, ~mask)
    mean_chance_long = float(np.mean([chance[r["prompt_id"]] for r in long_recs]))

    rng = np.random.default_rng(SEED)
    # label permutation for the identified-cell contrast
    allv = np.concatenate([R_id, W_id])
    nr = len(R_id)
    obs = R_id.mean() - W_id.mean()
    null = []
    for _ in range(N_PERM):
        perm = rng.permutation(allv)
        null.append(perm[:nr].mean() - perm[nr:].mean())
    p_lab = float((np.abs(null) >= abs(obs)).mean())

    # specificity: random 16-cell subsets of the OTHER cells
    R_all = np.stack([masses[str(r["prompt_id"])] for r in right])
    W_all = np.stack([masses[str(r["prompt_id"])] for r in wrong])
    other_flat = np.flatnonzero(~mask.ravel())
    Rf, Wf = R_all.reshape(len(right), -1), W_all.reshape(len(wrong), -1)
    obs_spec = (R_id.mean() - W_id.mean()) - (R_ot.mean() - W_ot.mean())
    null_spec = []
    for _ in range(N_PERM):
        pick = rng.choice(other_flat, TOP_K, replace=False)
        m = np.zeros(N_LAYERS * N_HEADS, dtype=bool)
        m[pick] = True
        null_spec.append((Rf[:, m].mean() - Wf[:, m].mean())
                         - (Rf[:, ~m].mean() - Wf[:, ~m].mean()))
    null_spec = np.array(null_spec)
    p_spec = float((null_spec >= obs_spec).mean())

    # per-cell collapse criterion: right-wrong drop > 50% of right mass
    per_cell_drop_id = int(sum(
        (R_all[:, l, h].mean() - W_all[:, l, h].mean())
        > 0.5 * R_all[:, l, h].mean()
        for l, h in top_cells))
    other_cells = [(int(c // N_HEADS), int(c % N_HEADS)) for c in other_flat]
    per_cell_drop_other = int(sum(
        (R_all[:, l, h].mean() - W_all[:, l, h].mean())
        > 0.5 * R_all[:, l, h].mean() and R_all[:, l, h].mean() > 0
        for l, h in other_cells))

    summary = {
        "model": ps["tokenizer_model_id"],
        "top_cells": top_cells,
        "top_cells_mean_mass": [float(mean_mass[l, h]) for l, h in top_cells],
        "chance_mass_short": mean_chance_short,
        "chance_mass_long": mean_chance_long,
        "mean_needle_tokens": mean_needle_tokens,
        "n_short_correct": len(id_recs),
        "long_bucket": {"n_right": len(right), "n_wrong": len(wrong)},
        "identified": {"right_mean": float(R_id.mean()),
                       "wrong_mean": float(W_id.mean()),
                       "cohen_d": cohen_d(R_id, W_id),
                       "perm_p": p_lab},
        "other": {"right_mean": float(R_ot.mean()),
                  "wrong_mean": float(W_ot.mean()),
                  "cohen_d": cohen_d(R_ot, W_ot)},
        "specificity": {"obs_excess_drop": float(obs_spec),
                        "null_mean": float(null_spec.mean()),
                        "null_p95": float(np.percentile(null_spec, 95)),
                        "p": p_spec},
        "per_cell_collapse": {"identified": per_cell_drop_id,
                              "identified_total": TOP_K,
                              "other": per_cell_drop_other,
                              "other_total": len(other_cells)},
    }
    (DATA_DIR / "transport.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
