"""Does the context-rot MECHANISM replicate on Granite-3.0-3B-A800M?
-> `docs/MECHANISM_GRANITE.md`.

Three pre-declared questions, mirroring the OLMoE chain exactly
(`docs/CONTEXT_ROT_HARD.md` -> `docs/MECHANISM.md` -> `docs/CONTEXT_PATHWAY.md`):

  Q1. Does forced-choice accuracy decline with context length? (substrate)
  Q2. Does the needle-affine specialist share of needle-window routing decline
      with length? (length trend: permutation null + BH-FDR + |d| >= 0.8)
  Q3. Does the needle-affine share predict answer probability INDEPENDENT of
      length? (partial Spearman controlling log2(length), permutation p,
      |partial rho| >= 0.3 practical floor -- same bar as CONTEXT_PATHWAY.md)

The mechanism claim replicates only if Q2 and Q3 both pass. WS-3 utilization
(hot/cold) does not exist for Granite, so the four-way pathway partition is
not reproduced here; the affine set (the causally-relevant one on OLMoE) is.

Reads `data/context_traces_granite/`; reuses `run_context_analyze`'s loading,
per-prompt routing, and needle-affine-set construction with Granite dimensions
patched in (32 layers x 40 experts, top-8), and `run_context_pathway`'s
partial-Spearman implementation, so no statistical code is duplicated.

Usage:
    .venv/bin/python scripts/run_context_mechanism_granite.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import run_context_analyze as rca
from run_context_pathway import partial_spearman

from expertatlas.context_metrics import apply_fdr, length_trend, set_hit_rate

REPO_ROOT = Path(__file__).parent.parent
OUT_MD = REPO_ROOT / "docs" / "MECHANISM_GRANITE.md"
OUT_JSON = REPO_ROOT / "data" / "mechanism_granite.json"

N_LAYERS, N_EXPERTS = 32, 40


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=str, default="data/context_traces_granite")
    ap.add_argument("--probe-set", type=str,
                    default="probes/probe_set_context_granite.yaml")
    args = ap.parse_args()

    rca.TRACES = REPO_ROOT / args.traces
    rca.PROBE_SET_PATH = REPO_ROOT / args.probe_set
    rca.N_LAYERS, rca.N_EXPERTS = N_LAYERS, N_EXPERTS
    rca.N_TOTAL = N_LAYERS * N_EXPERTS

    recs_list, ps = rca.load_prompt_features()
    recs = {r["prompt_id"]: r for r in recs_list}
    feats = []
    for r in sorted(recs_list, key=lambda x: x["prompt_id"]):
        f = rca.per_prompt_routing(r)
        if f is not None:
            feats.append(f)
    print(f"{len(feats)} prompts with traces", flush=True)
    buckets = sorted({recs[f["prompt_id"]]["bucket"] for f in feats})
    short_bucket = buckets[0]

    # Q1: substrate
    acc_by_bucket = {}
    for b in buckets:
        rs = [recs[f["prompt_id"]] for f in feats if recs[f["prompt_id"]]["bucket"] == b]
        acc_by_bucket[b] = {
            "n": len(rs),
            "fc_acc": float(np.mean([r["forced_choice_correct"] for r in rs])),
            "fc_prob": float(np.mean([r["forced_choice_prob"] for r in rs])),
        }
        print(f"bucket {b:>5}: acc={acc_by_bucket[b]['fc_acc']:.3f} "
              f"prob={acc_by_bucket[b]['fc_prob']:.3f} n={len(rs)}", flush=True)

    # affine set at shortest bucket only (identical construction to OLMoE)
    affine_mask, _, counts = rca.needle_affine_set(feats, recs, short_bucket)
    n_affine = int(affine_mask.sum())
    print(f"needle-affine experts (bucket {short_bucket}): {n_affine} "
          f"of {rca.N_TOTAL}", flush=True)

    rows = []
    for f in feats:
        r = recs[f["prompt_id"]]
        rows.append({
            "prompt_id": f["prompt_id"],
            "bucket": r["bucket"],
            "n_tokens": r["n_tokens"],
            "fc_correct": bool(r["forced_choice_correct"]),
            "answer_prob": float(r["forced_choice_prob"]),
            "affine_share_needle": set_hit_rate(f["_draws_needle"], affine_mask),
            "affine_share_q": set_hit_rate(f["_draws_q"], affine_mask),
            "entropy_all": f["entropy_all"],
        })

    L = np.array([d["n_tokens"] for d in rows], dtype=np.float64)
    B = np.array([d["bucket"] for d in rows], dtype=np.int64)
    prob = np.array([d["answer_prob"] for d in rows], dtype=np.float64)
    logl = np.log2(L)

    metric_names = ["affine_share_needle", "affine_share_q", "entropy_all"]

    # Q2: length trends (one FDR family)
    trends = apply_fdr([
        length_trend(m, L, np.array([d[m] for d in rows]), buckets=B)
        for m in metric_names
    ])
    for t in trends:
        print(f"trend {t.metric:22s} short={t.short_mean:.4f} long={t.long_mean:.4f} "
              f"rho={t.spearman_rho:+.3f} p={t.perm_p:.4f} d={t.cohens_d:+.2f} "
              f"fdr={'yes' if t.fdr_significant else 'no'} {t.verdict}", flush=True)

    # Q3: partial correlations with answer_prob controlling length
    partials = {}
    for m in metric_names:
        v = np.array([d[m] for d in rows])
        rho, p = partial_spearman(v, prob, logl)
        from scipy.stats import spearmanr
        raw = float(spearmanr(v, prob).statistic)
        partials[m] = {"raw_rho": raw, "partial_rho": rho, "perm_p": p}
        print(f"partial {m:22s} raw={raw:+.3f} partial={rho:+.3f} p={p:.4f}", flush=True)

    tmap = {t.metric: t for t in trends}
    q2 = tmap["affine_share_needle"]
    q2_pass = bool(q2.fdr_significant and q2.passes_effect_size and q2.spearman_rho < 0)
    pr = partials["affine_share_needle"]
    q3_pass = bool(pr["perm_p"] < 0.05 and pr["partial_rho"] >= 0.3)
    verdict = "REPLICATES" if (q2_pass and q3_pass) else "DOES NOT REPLICATE"

    out = {
        "model_id": "ibm-granite/granite-3.0-3b-a800m-base",
        "traces": args.traces,
        "n_prompts": len(rows),
        "n_affine": n_affine,
        "n_total": rca.N_TOTAL,
        "short_bucket": short_bucket,
        "accuracy_by_bucket": {str(k): v for k, v in acc_by_bucket.items()},
        "trends": [t.to_dict() for t in trends],
        "partials": partials,
        "q2_pass": q2_pass,
        "q3_pass": q3_pass,
        "verdict": verdict,
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(out, indent=1))

    md = []
    md.append("# Does the context-rot mechanism replicate on a second model? "
              "(Granite-3.0-3B-A800M)\n")
    md.append("## Limits (read first)\n")
    md.append(
        "1. **Different candidate pool.** 7/8 of the OLMoE forced-choice words are\n"
        "   multi-token under the Granite tokenizer, so this run uses a pool verified\n"
        "   single-token under BOTH tokenizers (see\n"
        "   `probes/generate_context_probes_granite.py`). Same design, same seed\n"
        "   structure, different surface content -- a conceptual replication, not a\n"
        "   byte-identical one.\n"
        "2. **No WS-3 utilization run exists for Granite**, so the hot/cold pathway\n"
        "   partition from `docs/CONTEXT_PATHWAY.md` is not reproduced; only the\n"
        "   needle-affine specialist set (the causally-relevant pathway on OLMoE) is.\n"
        "3. One seed, one task design, CPU BF16 -- same scope limits as the OLMoE run.\n"
        "4. Correlational throughout. No intervention was run on Granite.\n")
    md.append(f"## Q1 -- substrate: accuracy vs length ({len(rows)} prompts)\n")
    md.append("| bucket | n | forced-choice acc | mean answer prob |")
    md.append("|---|---|---|---|")
    for b in buckets:
        a = acc_by_bucket[b]
        md.append(f"| {b} | {a['n']} | {a['fc_acc']:.3f} | {a['fc_prob']:.3f} |")
    md.append("")
    md.append(f"## Q2 -- length trends (needle-affine set: {n_affine} of "
              f"{rca.N_TOTAL} experts, defined at bucket {short_bucket} only)\n")
    md.append("| metric | short | long | delta | rho | perm p | FDR | d | verdict |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for t in trends:
        md.append(f"| {t.metric} | {t.short_mean:.4f} | {t.long_mean:.4f} "
                  f"| {t.delta:+.4f} | {t.spearman_rho:+.3f} | {t.perm_p:.4f} "
                  f"| {'yes' if t.fdr_significant else 'no'} | {t.cohens_d:+.2f} "
                  f"| **{t.verdict}** |")
    md.append("")
    md.append("## Q3 -- does each metric predict answer probability, independent "
              "of length?\n")
    md.append("Partial Spearman controlling log2(length), permutation p "
              "(2000 shuffles, two-sided), practical floor |partial rho| >= 0.3.\n")
    md.append("| metric | raw rho | partial rho | perm p |")
    md.append("|---|---|---|---|")
    for m in metric_names:
        p = partials[m]
        md.append(f"| {m} | {p['raw_rho']:+.3f} | {p['partial_rho']:+.3f} "
                  f"| {p['perm_p']:.4f} |")
    md.append("")
    md.append(f"## Verdict: **{verdict}**\n")
    md.append(f"- Q2 (specialist share declines with length, FDR + |d|>=0.8): "
              f"{'PASS' if q2_pass else 'FAIL'}")
    md.append(f"- Q3 (specialist share predicts accuracy independent of length, "
              f"perm p<0.05 + partial rho>=+0.3): {'PASS' if q3_pass else 'FAIL'}")
    md.append("\nOLMoE reference values (docs/CONTEXT_PATHWAY.md): specialist share "
              "0.0417 -> 0.0335 (d=-1.39, FDR yes); partial rho +0.651, p=0.0005.\n")
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"verdict: {verdict}; wrote {OUT_MD} and {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
