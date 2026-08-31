"""Generalist-vs-specialist pathway analysis of context degradation (WS-1/WS-3)
-> `docs/CONTEXT_PATHWAY.md`.

The open question left by `docs/CONTEXT_ROT_HARD.md` and `docs/MECHANISM.md`:
when routing degrades with length, does that degradation concentrate in the
GENERALIST pathway (WS-3's hot, high-load experts) or in the SPECIALIST
pathway (the needle-affine experts, which WS-3 showed are disproportionately
cold)?

Design: partition the 1,024 global experts into four DISJOINT sets --

  * affine          : needle-affine experts, defined at the shortest bucket
                      only (same construction as `run_context_analyze.py`)
  * hot_nonaffine   : WS-3 hot experts (load_ratio >= 2.0) not in `affine`
  * cold_nonaffine  : WS-3 cold experts not in `affine`
  * mid_nonaffine   : everything else

-- then, per prompt and per measurement window (needle window and question
window, both content-identical across buckets by construction), the share of
top-k draws landing in each set. Two families of tests:

  1. LENGTH TRENDS: each share vs log2(length), permutation null + BH-FDR
     across the whole family + effect-size floor (`context_metrics`).
  2. ACCURACY PREDICTION: partial Spearman of each share vs `answer_prob`
     controlling for log2(length) via rank-residual regression (the same
     procedure `docs/MECHANISM.md` used), permutation p on the partial rho.

This script only READS the trace directory; it writes its own doc/json and
never touches `docs/CONTEXT_ROT_HARD.md` or `data/context_rot_hard.json`.

Usage:
    python scripts/run_context_pathway.py \
        --traces data/context_traces_hard \
        --probe-set probes/probe_set_context_hard.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import run_context_analyze as rca  # reuse loading + reference-set code exactly

from expertatlas.context_metrics import apply_fdr, length_trend, set_hit_rate

REPO_ROOT = Path(__file__).parent.parent
OUT_MD = REPO_ROOT / "docs" / "CONTEXT_PATHWAY.md"
OUT_JSON = REPO_ROOT / "data" / "context_pathway.json"
N_PERM = 2000


def rank_residuals(x: np.ndarray, ctrl: np.ndarray) -> np.ndarray:
    """Residuals of rank(x) regressed on rank(ctrl) -- the partial-Spearman
    building block used in docs/MECHANISM.md."""
    from scipy.stats import rankdata

    rx = rankdata(x).astype(np.float64)
    rc = rankdata(ctrl).astype(np.float64)
    A = np.stack([rc, np.ones_like(rc)], axis=1)
    coef, *_ = np.linalg.lstsq(A, rx, rcond=None)
    return rx - A @ coef


def partial_spearman(x: np.ndarray, y: np.ndarray, ctrl: np.ndarray,
                     n_perm: int = N_PERM, seed: int = 0) -> tuple[float, float]:
    """Partial Spearman rho of x vs y controlling ctrl, with a permutation p
    (shuffle x's residuals; two-sided)."""
    rx = rank_residuals(x, ctrl)
    ry = rank_residuals(y, ctrl)
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom == 0:
        return float("nan"), 1.0
    rho = float((rx * ry).sum() / denom)
    rng = np.random.default_rng(seed)
    hits = 0
    rxp = rx.copy()
    for _ in range(n_perm):
        rng.shuffle(rxp)
        r = (rxp * ry).sum() / denom
        if abs(r) >= abs(rho):
            hits += 1
    return rho, (hits + 1.0) / (n_perm + 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=str, default="data/context_traces_hard")
    ap.add_argument("--probe-set", type=str,
                    default="probes/probe_set_context_hard.yaml")
    ap.add_argument("--out-md", type=str, default=str(OUT_MD))
    ap.add_argument("--out-json", type=str, default=str(OUT_JSON))
    args = ap.parse_args()

    rca.TRACES = REPO_ROOT / args.traces
    rca.PROBE_SET_PATH = REPO_ROOT / args.probe_set

    recs_list, ps = rca.load_prompt_features()
    recs = {r["prompt_id"]: r for r in recs_list}
    feats = []
    for r in sorted(recs_list, key=lambda x: x["prompt_id"]):
        f = rca.per_prompt_routing(r)
        if f is not None:
            feats.append(f)
    print(f"{len(feats)} prompts with traces", flush=True)
    present = sorted({recs[f["prompt_id"]]["bucket"] for f in feats})
    short_bucket = present[0]

    affine_mask, _, _ = rca.needle_affine_set(feats, recs, short_bucket)
    util = rca.load_utilization()
    if util is None:
        raise SystemExit("data/utilization.json required -- run WS-3 first")
    uids = util["utilization"]["uids"]
    load = np.asarray(util["utilization"]["load_ratio"], dtype=np.float64)
    order = {u: i for i, u in enumerate(uids)}
    idx = np.array([order[f"L{l:02d}E{e:02d}"]
                    for l in range(rca.N_LAYERS) for e in range(rca.N_EXPERTS)])
    load = load[idx]
    hot_mask = load >= 2.0
    thresh = np.sort(load)[util["utilization"]["n_cold"] - 1]
    cold_mask = load <= thresh

    sets = {
        "affine": affine_mask,
        "hot_nonaffine": hot_mask & ~affine_mask,
        "cold_nonaffine": cold_mask & ~affine_mask,
        "mid_nonaffine": ~affine_mask & ~hot_mask & ~cold_mask,
    }
    composition = {
        "n_affine": int(affine_mask.sum()),
        "n_affine_hot": int((affine_mask & hot_mask).sum()),
        "n_affine_cold": int((affine_mask & cold_mask).sum()),
        "n_hot": int(hot_mask.sum()),
        "n_cold": int(cold_mask.sum()),
        "n_total": int(rca.N_TOTAL),
    }
    print("set composition:", composition, flush=True)
    for k, m in sets.items():
        print(f"  {k}: {int(m.sum())} experts", flush=True)

    # per-prompt shares
    rows = []
    for f in feats:
        r = recs[f["prompt_id"]]
        d = {
            "prompt_id": f["prompt_id"],
            "bucket": r["bucket"],
            "n_tokens": r["n_tokens"],
            "answer_prob": float(r["forced_choice_prob"]),
        }
        for name, mask in sets.items():
            d[f"{name}_share_needle"] = set_hit_rate(f["_draws_needle"], mask)
            d[f"{name}_share_q"] = set_hit_rate(f["_draws_q"], mask)
        rows.append(d)

    L = np.array([d["n_tokens"] for d in rows], dtype=np.float64)
    B = np.array([d["bucket"] for d in rows], dtype=np.int64)
    prob = np.array([d["answer_prob"] for d in rows], dtype=np.float64)
    logl = np.log2(L)

    metric_names = [f"{name}_share_{w}" for name in sets for w in ("needle", "q")]

    # family 1: length trends, one FDR family
    trends = [length_trend(m, L, np.array([d[m] for d in rows]), buckets=B)
              for m in metric_names]
    trends = apply_fdr(trends)

    # family 2: partial correlation with answer_prob controlling length
    partials = {}
    for m in metric_names:
        v = np.array([d[m] for d in rows])
        rho, p = partial_spearman(v, prob, logl)
        from scipy.stats import spearmanr
        raw = float(spearmanr(v, prob).statistic)
        partials[m] = {"raw_rho": raw, "partial_rho": rho, "perm_p": p}
        print(f"  {m:28s} raw={raw:+.3f} partial={rho:+.3f} p={p:.4f}", flush=True)

    out = {
        "traces": args.traces,
        "composition": composition,
        "trends": [t.to_dict() for t in trends],
        "partials": partials,
        "rows": rows,
    }
    Path(args.out_json).write_text(json.dumps(out, indent=1))

    # ---- report -------------------------------------------------------------
    md = []
    md.append("# Which pathway degrades with context length: generalist or specialist?\n")
    md.append("## Limits (read first)\n")
    md.append(
        "1. **One model (OLMoE-1B-7B-0924), one seed, one task design.** Same scope\n"
        "   limits as `docs/CONTEXT_ROT_HARD.md`.\n"
        "2. **The substrate verdict carries over**: forced-choice accuracy declines\n"
        "   but below the preregistered effect-size bar, so every routing trend here\n"
        "   is descriptive of routing itself; the accuracy link is the per-prompt\n"
        "   partial correlation, which is correlational, not causal.\n"
        "3. **`load_ratio` and lift come from the same count matrix** (WS-3's stated\n"
        "   caveat): hot/cold vs specialist are partly definitionally opposed.\n"
        "4. **The affine set is defined at the shortest bucket** on this run's own\n"
        "   traces (192 prompts), so it inherits that pipeline's assumptions.\n"
        "5. BF16 CPU regeneration of the traces drifts slightly from the committed\n"
        "   run (see RESEARCH_LOG entry 8); all numbers here are from one internally\n"
        "   consistent regeneration.\n")
    md.append("## Expert-set partition (disjoint)\n")
    md.append("| set | # experts | of which hot | of which cold |\n|---|---|---|---|")
    md.append(f"| affine (specialist pathway) | {composition['n_affine']} | "
              f"{composition['n_affine_hot']} | {composition['n_affine_cold']} |")
    for k in ("hot_nonaffine", "cold_nonaffine", "mid_nonaffine"):
        md.append(f"| {k} | {int(sets[k].sum())} | - | - |")
    md.append("")
    md.append("## Length trends (permutation null, BH-FDR across this family, "
              "effect-size floor)\n")
    md.append("| share | window | short | long | delta | rho | perm p | FDR | d | verdict |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for t in trends:
        name, window = t.metric.rsplit("_share_", 1)
        md.append(f"| {name} | {'needle' if window == 'needle' else 'question'} "
                  f"| {t.short_mean:.4f} | {t.long_mean:.4f} | {t.delta:+.4f} "
                  f"| {t.spearman_rho:+.3f} | {t.perm_p:.4f} "
                  f"| {'yes' if t.fdr_significant else 'no'} | {t.cohens_d:+.2f} "
                  f"| **{t.verdict}** |")
    md.append("")
    md.append("## Does each pathway's share predict accuracy, independent of length?\n")
    md.append("Partial Spearman vs `answer_prob`, controlling log2(length) by\n"
              "rank-residual regression (same procedure as `docs/MECHANISM.md`),\n"
              f"permutation p ({N_PERM} shuffles, two-sided).\n")
    md.append("| share | window | raw rho | partial rho | perm p |")
    md.append("|---|---|---|---|---|")
    for m in metric_names:
        name, window = m.rsplit("_share_", 1)
        pr = partials[m]
        md.append(f"| {name} | {'needle' if window == 'needle' else 'question'} "
                  f"| {pr['raw_rho']:+.3f} | {pr['partial_rho']:+.3f} "
                  f"| {pr['perm_p']:.4f} |")
    md.append("")
    md.append("## Reading the two tables together\n")
    md.append(
        "The length trends say where routing mass MOVES as context grows; the\n"
        "partial correlations say which of those movements tracks correctness.\n"
        "A pathway is implicated in degradation only if it shows both: a length\n"
        "trend AND a length-independent association with `answer_prob`. A length\n"
        "trend alone (like `entropy_all` in `docs/MECHANISM.md`) is movement that\n"
        "explains nothing; a partial correlation alone is prompt-to-prompt\n"
        "variation that length does not touch. Numbers above are the evidence;\n"
        "this section only states which cells clear both requirements.\n")
    tmap = {t.metric: t for t in trends}
    md.append("| share | window | length trend? | predicts accuracy (partial)? | implicated |")
    md.append("|---|---|---|---|---|")
    for m in metric_names:
        name, window = m.rsplit("_share_", 1)
        t = tmap[m]
        trend_ok = bool(t.fdr_significant and t.passes_effect_size)
        pred_ok = bool(partials[m]["perm_p"] < 0.05 and abs(partials[m]["partial_rho"]) >= 0.3)
        md.append(f"| {name} | {'needle' if window == 'needle' else 'question'} "
                  f"| {'yes' if trend_ok else 'no'} | {'yes' if pred_ok else 'no'} "
                  f"| {'**YES**' if trend_ok and pred_ok else 'no'} |")
    md.append("\n(`predicts accuracy` bar: perm p < 0.05 AND |partial rho| >= 0.3 -- the\n"
              "same practical-effect floor spirit as everywhere else in this repo.)\n")
    Path(args.out_md).write_text("\n".join(md) + "\n")
    print(f"wrote {args.out_md} and {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
