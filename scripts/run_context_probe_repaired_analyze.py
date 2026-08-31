"""Deconfounded residual-stream needle probe: analysis.

Companion to probes/generate_context_probes_repaired.py. The hard-variant
probe run found that a linear probe reads the answer word off the ENTITY
MENTION (q_mean layer-0 accuracy 100%: raw embeddings, before any attention,
at a window that contains the entity but not the answer). Because
entity<->answer is 1:1 in that set, every position that can see the entity is
confounded. This analysis uses the re-paired set, where:

  * pairing set A (replicates 0-7):  entity_i -> word_i
  * pairing set B (replicates 8-15): entity_i -> word_{(i+3) mod 8}

Design, registered before evaluation:
  * Train on one pairing set, evaluate on the other, both directions, mean
    reported. Across the split the entity shortcut predicts a SPECIFIC WRONG
    word, so it scores ~0, and its usage is separately measurable as
    `shortcut_rate` (fraction of test predictions equal to the training
    pairing's word for the test prompt's entity).
  * Multinomial logistic regression, standardised features (train-fit only),
    fixed C=1.0 (no tuning: the training sets are small and the hard-variant
    grid showed the result is not C-sensitive where it is strong).
  * Accuracy reported per length bucket (256 / 1024 / 3840) per (position,
    layer) cell. Chance = 0.125.
  * Primary cells, chosen in advance: (`needle_last`, layer 8) = "written at
    the source", (`final`, layer 16) = "available at the readout".
    Null: 200 refits with shuffled training labels. Floor: 2x chance (0.25).
  * Per-prompt crux at the primary cells, long bucket only: probe correctness
    vs the model's own forced-choice correctness.

Output: data/context_probe_repaired/analysis.json

Usage:
    .venv/bin/python scripts/run_context_probe_repaired_analyze.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

REPO_ROOT = Path(__file__).parent.parent
IN_DIR = REPO_ROOT / "data" / "context_probe_repaired"
POSITIONS = ("needle_last", "needle_mean", "q_mean", "final")
N_LAYERS = 17
BUCKETS = (256, 1024, 3840)
PRIMARY_CELLS = (("needle_last", 8), ("final", 16))
PAIR_SHIFT_B = 3
N_PERM = 200
SEED = 0


def fit_eval(Xtr, ytr, Xte, seed=SEED):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=2000, random_state=seed))
    pipe.fit(Xtr, ytr)
    return pipe.predict(Xte)


def main() -> None:
    recs = [json.loads(l) for l in (IN_DIR / "records.jsonl").read_text().splitlines()]
    recs = {r["prompt_id"]: r for r in recs}
    recs = [recs[k] for k in sorted(recs)]
    n = len(recs)
    words = sorted({r["answer_word"] for r in recs})
    entities = sorted({r["entity"] for r in recs})
    y = np.array([words.index(r["answer_word"]) for r in recs])
    is_A = np.array([r["pairing_set"] == "A" for r in recs])
    buckets = np.array([r["bucket"] for r in recs])
    model_correct = np.array([bool(r["forced_choice_correct"]) for r in recs])

    # per-prompt: what the ENTITY SHORTCUT from the *other* pairing set predicts
    # (train on A -> shortcut for a B prompt with entity e_i is word_i; and
    #  train on B -> shortcut for an A prompt with entity e_i is word_{(i+3)%8})
    ent_idx = np.array([entities.index(r["entity"]) for r in recs])

    X = {}
    mats = {(p, li): [] for p in POSITIONS for li in range(N_LAYERS)}
    for r in recs:
        z = np.load(IN_DIR / f"hidden_{r['prompt_id']:06d}.npz")
        for p in POSITIONS:
            for li in range(N_LAYERS):
                mats[(p, li)].append(z[f"{p}_{li}"])
    for k, v in mats.items():
        X[k] = np.stack(v)

    out = {"words": words, "model_accuracy_by_bucket": {}, "grid": {},
           "primary": {}, "crux": {}}
    for b in BUCKETS:
        m = buckets == b
        out["model_accuracy_by_bucket"][str(b)] = float(model_correct[m].mean())
    print("model acc by bucket:", out["model_accuracy_by_bucket"], flush=True)

    def entity_words_under_train_pairing(train_is_A: bool):
        # entities sorted alphabetically != base order; map via each prompt's
        # own pairing: build entity -> word map from the TRAINING prompts.
        src = [r for r, a in zip(recs, is_A) if a == train_is_A]
        return {r["entity"]: r["answer_word"] for r in src}

    rng = np.random.default_rng(SEED)
    preds_store = {}
    for p in POSITIONS:
        for li in range(N_LAYERS):
            cell = f"{p}_{li}"
            accs = {str(b): [] for b in BUCKETS}
            shortcut = {str(b): [] for b in BUCKETS}
            all_pred = np.full(n, -1)
            for train_A in (True, False):
                tr = is_A == train_A
                te = ~tr
                pred = fit_eval(X[(p, li)][tr], y[tr], X[(p, li)][te])
                all_pred[te] = pred
                emap = entity_words_under_train_pairing(train_A)
                te_idx = np.where(te)[0]
                for j, i_glob in enumerate(te_idx):
                    b = str(buckets[i_glob])
                    accs[b].append(int(pred[j] == y[i_glob]))
                    shortcut[b].append(
                        int(words[pred[j]] == emap[recs[i_glob]["entity"]]))
            preds_store[(p, li)] = all_pred
            out["grid"][cell] = {
                "acc_by_bucket": {b: float(np.mean(v)) for b, v in accs.items()},
                "shortcut_rate_by_bucket": {b: float(np.mean(v))
                                            for b, v in shortcut.items()},
                "acc_overall": float((all_pred == y).mean()),
            }
            g = out["grid"][cell]
            print(f"{p:12s} L{li:2d}: " +
                  " ".join(f"{b}={g['acc_by_bucket'][str(b)]:.3f}" for b in BUCKETS) +
                  "  shortcut@3840=" f"{g['shortcut_rate_by_bucket']['3840']:.3f}",
                  flush=True)

    # permutation nulls at primary cells (both directions, shuffled train labels)
    for p, li in PRIMARY_CELLS:
        obs = out["grid"][f"{p}_{li}"]["acc_overall"]
        null = []
        for _ in range(N_PERM):
            accs = []
            for train_A in (True, False):
                tr = is_A == train_A
                te = ~tr
                ysh = rng.permutation(y[tr])
                pred = fit_eval(X[(p, li)][tr], ysh, X[(p, li)][te])
                accs.append(float((pred == y[te]).mean()))
            null.append(float(np.mean(accs)))
        null = np.array(null)
        pval = float((np.sum(null >= obs) + 1) / (N_PERM + 1))
        out["primary"][f"{p}_{li}"] = {
            "acc": obs, "perm_p": pval, "null_mean": float(null.mean()),
            "null_max": float(null.max()), "clears_2x_chance": bool(obs >= 0.25),
        }
        print(f"PRIMARY {p} L{li}: acc={obs:.3f} p={pval:.4f} "
              f"null={null.mean():.3f} (max {null.max():.3f})", flush=True)

    # crux: long bucket, probe vs model correctness
    for p, li in PRIMARY_CELLS:
        pr = preds_store[(p, li)]
        m = buckets == 3840
        probe_ok = (pr == y) & m
        rows = {}
        for name, mask in [("model_wrong", m & ~model_correct),
                           ("model_right", m & model_correct)]:
            k = int(mask.sum())
            rows[name] = {"n": k,
                          "probe_acc": float((pr[mask] == y[mask]).mean()) if k else None}
        out["crux"][f"{p}_{li}"] = rows
        print(f"crux {p} L{li}:", json.dumps(rows), flush=True)

    (IN_DIR / "analysis.json").write_text(json.dumps(out, indent=2))
    print("wrote", IN_DIR / "analysis.json", flush=True)


if __name__ == "__main__":
    main()
