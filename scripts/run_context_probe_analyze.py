"""Residual-stream needle probe: analysis (WS1, upstream test).

Question (docs/CONTEXT_ROT_STORY.md §4): when the router loses the
needle-affine specialist pathway on long prompts and the model answers wrong,
is the needle's content still linearly recoverable from the hidden states?
The answer localises the failure:

  * probe succeeds at the needle site AND at the final position, model wrong
      -> information present but unused: readout/routing remains a viable target;
  * probe succeeds at the needle site but fails at the final position
      -> transport failure: attention stops carrying it forward;
  * probe fails even at the needle site (late layers)
      -> the representation itself degrades at source.

Design, registered before evaluation:

  * Features: float32 hidden states captured by run_context_probe_capture.py
    (17 layers x 4 positions), one vector per prompt. Labels: the 8 answer
    words. Chance = 0.125. Labels are balanced (24 per class).
  * NOTE the replicate confound: replicate r always carries answer word r, so
    a replicate split cannot be used. Splits are by LENGTH BUCKET and by
    (bucket, condition) cell; every class appears on both sides of every split.
  * Primary split ("does the code persist to long contexts"): train on buckets
    {256, 512, 1024} (96 prompts), evaluate on {2048, 3072, 3840} (96 prompts).
    Multinomial logistic regression, standardised features (scaler fit on train
    only), C chosen by 3-fold CV within the training set only. Evaluation
    prompts and their labels never touch fitting or calibration.
  * Control split ("rule out covariate shift with length"): within long buckets
    only, leave-one-(bucket, condition)-cell-out (12 folds of 8 prompts).
  * Primary cells, chosen before any evaluation: (`needle_last`, layer 8) for
    "written at the source, mid-stack" and (`final`, layer 16) for "available
    at the readout". All other (layer, position) cells are reported in full as
    exploratory.
  * Null: 200 refits with training labels shuffled (primary cells only), giving
    a calibrated null distribution of test accuracy. Effect floor: the probe is
    only called informative if accuracy also exceeds 2x chance (0.25).
  * Crux conditioning (long-bucket prompts only): probe correctness vs model
    forced-choice correctness, and vs the prompt's `needle_affinity_rate` from
    data/context_rot_hard.json (median split within long buckets).

Output: data/context_probe/analysis.json and a printed report.

Usage:
    .venv/bin/python scripts/run_context_probe_analyze.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

REPO_ROOT = Path(__file__).parent.parent
IN_DIR = REPO_ROOT / "data" / "context_probe"
HARD_JSON = REPO_ROOT / "data" / "context_rot_hard.json"
POSITIONS = ("needle_last", "needle_mean", "q_mean", "final")
N_LAYERS = 17
SHORT = {256, 512, 1024}
LONG = {2048, 3072, 3840}
PRIMARY_CELLS = (("needle_last", 8), ("final", 16))
N_PERM = 200
SEED = 0


def load_data():
    recs = [json.loads(l) for l in (IN_DIR / "records.jsonl").read_text().splitlines()]
    recs = {r["prompt_id"]: r for r in recs}
    recs = [recs[k] for k in sorted(recs)]
    words = sorted({r["answer_word"] for r in recs})
    y = np.array([words.index(r["answer_word"]) for r in recs])
    X = {}  # (pos, layer) -> (n, H)
    mats = {(p, li): [] for p in POSITIONS for li in range(N_LAYERS)}
    for r in recs:
        z = np.load(IN_DIR / f"hidden_{r['prompt_id']:06d}.npz")
        for p in POSITIONS:
            for li in range(N_LAYERS):
                mats[(p, li)].append(z[f"{p}_{li}"])
    for k, v in mats.items():
        X[k] = np.stack(v)
    return recs, X, y, words


def fit_eval(Xtr, ytr, Xte, yte, seed=SEED, tune=True):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    base = LogisticRegression(max_iter=2000, random_state=seed)
    pipe = make_pipeline(StandardScaler(), base)
    if tune:
        gs = GridSearchCV(
            pipe, {"logisticregression__C": [0.01, 0.1, 1.0]},
            cv=StratifiedKFold(3, shuffle=True, random_state=seed), n_jobs=4)
        gs.fit(Xtr, ytr)
        model = gs.best_estimator_
    else:
        pipe.fit(Xtr, ytr)
        model = pipe
    pred = model.predict(Xte)
    scores = model.decision_function(Xte)
    # margin of true class over best other
    margins = []
    for i, yy in enumerate(yte):
        row = scores[i]
        others = np.delete(row, yy)
        margins.append(float(row[yy] - others.max()))
    return pred, np.array(margins), float((pred == yte).mean())


def main() -> None:
    recs, X, y, words = load_data()
    buckets = np.array([r["bucket"] for r in recs])
    tr = np.array([b in SHORT for b in buckets])
    te = np.array([b in LONG for b in buckets])
    model_correct = np.array([bool(r["forced_choice_correct"]) for r in recs])

    hard = {p["prompt_id"]: p for p in json.load(HARD_JSON.open())["per_prompt"]}
    naff = np.array([hard[r["prompt_id"]]["needle_affinity_rate"] for r in recs])

    rng = np.random.default_rng(SEED)
    out = {"words": words, "grid": {}, "primary": {}, "within_long_cv": {},
           "crux": {}}

    # ---- full exploratory grid: train short -> test long ----
    preds = {}
    margins = {}
    for p in POSITIONS:
        for li in range(N_LAYERS):
            pred, marg, acc = fit_eval(X[(p, li)][tr], y[tr], X[(p, li)][te], y[te])
            preds[(p, li)] = pred
            margins[(p, li)] = marg
            out["grid"][f"{p}_{li}"] = acc
            print(f"grid {p:12s} layer {li:2d}: acc={acc:.3f}", flush=True)

    # ---- permutation null at primary cells ----
    for p, li in PRIMARY_CELLS:
        obs = out["grid"][f"{p}_{li}"]
        null = []
        for _ in range(N_PERM):
            ysh = rng.permutation(y[tr])
            _, _, a = fit_eval(X[(p, li)][tr], ysh, X[(p, li)][te], y[te], tune=False)
            null.append(a)
        null = np.array(null)
        pval = float((np.sum(null >= obs) + 1) / (N_PERM + 1))
        out["primary"][f"{p}_{li}"] = {
            "acc": obs, "perm_p": pval,
            "null_mean": float(null.mean()), "null_max": float(null.max()),
            "clears_2x_chance": bool(obs >= 0.25),
        }
        print(f"PRIMARY {p} L{li}: acc={obs:.3f} perm_p={pval:.4f} "
              f"null={null.mean():.3f}(max {null.max():.3f})", flush=True)

    # ---- within-long CV control ----
    cells = sorted({(r["bucket"], r["haystack"], r["n_distractors"])
                    for r in recs if r["bucket"] in LONG})
    idx_long = np.where(te)[0]
    for p, li in PRIMARY_CELLS:
        correct = 0
        for cell in cells:
            fold = np.array([
                (recs[i]["bucket"], recs[i]["haystack"], recs[i]["n_distractors"]) == cell
                for i in idx_long])
            tr_i = idx_long[~fold]
            te_i = idx_long[fold]
            pred, _, _ = fit_eval(X[(p, li)][tr_i], y[tr_i], X[(p, li)][te_i], y[te_i],
                                  tune=False)
            correct += int((pred == y[te_i]).sum())
        acc = correct / len(idx_long)
        out["within_long_cv"][f"{p}_{li}"] = acc
        print(f"within-long CV {p} L{li}: acc={acc:.3f}", flush=True)

    # ---- crux conditioning on long-bucket prompts ----
    idx = np.where(te)[0]
    med = float(np.median(naff[idx]))
    for p, li in PRIMARY_CELLS:
        pr = preds[(p, li)]
        # preds indexed over test set in order of np.where(te)
        probe_ok = pr == y[te]
        rows = {}
        for name, mask in [
            ("model_wrong", ~model_correct[idx]),
            ("model_right", model_correct[idx]),
            ("model_wrong_low_affinity", (~model_correct[idx]) & (naff[idx] < med)),
            ("model_right_high_affinity", model_correct[idx] & (naff[idx] >= med)),
        ]:
            n = int(mask.sum())
            rows[name] = {"n": n,
                          "probe_acc": float(probe_ok[mask].mean()) if n else None}
        out["crux"][f"{p}_{li}"] = rows
        print(f"crux {p} L{li}: " + json.dumps(rows), flush=True)

    # margin vs answer_prob / affinity on long prompts (Spearman)
    from scipy.stats import spearmanr
    aprob = np.array([hard[r["prompt_id"]]["answer_prob"] for r in recs])
    for p, li in PRIMARY_CELLS:
        m = margins[(p, li)]
        r1 = spearmanr(m, aprob[idx])
        r2 = spearmanr(m, naff[idx])
        out["crux"][f"{p}_{li}"]["margin_vs_answer_prob"] = {
            "rho": float(r1.statistic), "p": float(r1.pvalue)}
        out["crux"][f"{p}_{li}"]["margin_vs_needle_affinity"] = {
            "rho": float(r2.statistic), "p": float(r2.pvalue)}
        print(f"margin {p} L{li}: vs answer_prob rho={r1.statistic:.3f} p={r1.pvalue:.2g}; "
              f"vs affinity rho={r2.statistic:.3f} p={r2.pvalue:.2g}", flush=True)

    (IN_DIR / "analysis.json").write_text(json.dumps(out, indent=2))
    print("wrote", IN_DIR / "analysis.json", flush=True)


if __name__ == "__main__":
    main()
