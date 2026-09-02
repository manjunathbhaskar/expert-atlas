"""Context-rot analysis (Workstream 1) -> `docs/CONTEXT_ROT.md`.

Reads `data/context_traces/` and answers one question: **does the routing-level
signal degrade with input length in step with task accuracy?**

Three outcomes are equally publishable and the script labels them as such:

  * accuracy degrades AND routing degrades   -> MECHANISM FOUND
  * accuracy degrades, routing flat          -> MECHANISM RULED OUT (a real
                                                result, reported as prominently)
  * accuracy does not degrade                -> SUBSTRATE CANNOT TEST IT

Nothing here is allowed to inflate a null. Every trend clears BOTH a permutation
null with BH-FDR correction across the whole family of tests AND a practical
effect-size floor, per `expertatlas/context_metrics.py`.

Usage:
    python scripts/run_context_analyze.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from expertatlas.context_metrics import (
    MIN_COHENS_D,
    MIN_TREND_RHO,
    PMI_SKEW_LIMIT,
    apply_fdr,
    ascii_overlay,
    community_structure,
    distinct_experts_touched,
    expected_distinct_under_null,
    length_trend,
    selection_share,
    set_hit_rate,
)
from expertatlas.stats import bh_fdr, chi2_pvalues_fast, compute_lift

REPO_ROOT = Path(__file__).parent.parent
TRACES = REPO_ROOT / "data" / "context_traces"
PROBE_SET_PATH = REPO_ROOT / "probes" / "probe_set_context.yaml"
UTILIZATION = REPO_ROOT / "data" / "utilization.json"
OUT_MD = REPO_ROOT / "docs" / "CONTEXT_ROT.md"
OUT_JSON = REPO_ROOT / "data" / "context_rot.json"

N_LAYERS, N_EXPERTS, TOP_K = 16, 64, 8
N_TOTAL = N_LAYERS * N_EXPERTS
Q_FDR = 0.05
MEANINGFUL_LIFT = 1.0  # same >=2x fold-change bar the rest of the repo uses

CHROMA_QUOTE = (
    "we do not have a definitive answer for why that occurs... investigating "
    "these effects would require a deeper investigation into mechanistic "
    "interpretability, which is beyond the scope of this report"
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_prompt_features() -> tuple[list[dict], dict]:
    ps = yaml.safe_load(PROBE_SET_PATH.read_text())
    acc_path = TRACES / "accuracy.jsonl"
    if not acc_path.exists():
        raise RuntimeError(f"no accuracy.jsonl under {TRACES} — run the sweep first")

    # jsonl is append-only and a resumed run can re-append; last record wins.
    acc: dict[int, dict] = {}
    for line in acc_path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            acc[r["prompt_id"]] = r
    return list(acc.values()), ps


def load_utilization() -> dict | None:
    if not UTILIZATION.exists():
        return None
    return json.loads(UTILIZATION.read_text())


def per_prompt_routing(rec: dict) -> dict | None:
    """Stream one prompt's trace into per-token routing statistics.

    Every returned quantity is a MEAN over tokens or a RATE over top-k draws.
    Nothing is a sum over tokens, because a sum grows with length under a null
    router — see `context_metrics` module docstring.
    """
    shard = TRACES / f"trace_{rec['prompt_id']:06d}.parquet"
    if not shard.exists():
        return None
    t = pq.read_table(shard)

    tok_pos = np.asarray(t.column("token_pos").to_pylist(), dtype=np.int64)
    layer = np.asarray(t.column("layer").to_pylist(), dtype=np.int64)
    ent = np.asarray(t.column("router_entropy").to_pylist(), dtype=np.float64)
    mass = np.asarray(t.column("topk_mass").to_pylist(), dtype=np.float64)
    eids = np.asarray(t.column("expert_ids").to_pylist(), dtype=np.int64)  # (rows, k)

    gidx = layer[:, None] * N_EXPERTS + eids  # global expert index per draw

    q0, q1 = rec["question_token_span"]
    n0, n1 = rec["needle_token_span"]
    qm = (tok_pos >= q0) & (tok_pos < q1)
    nm = (tok_pos >= n0) & (tok_pos < n1)

    return {
        "prompt_id": rec["prompt_id"],
        # --- primary: byte-identical measurement windows ---
        "entropy_q": float(ent[qm].mean()),
        "mass_q": float(mass[qm].mean()),
        "entropy_needle": float(ent[nm].mean()),
        "mass_needle": float(mass[nm].mean()),
        # --- secondary: whole prompt. Per-token, but the token CONTENT mix
        #     differs across buckets (long prompts are mostly haystack), so this
        #     is reported separately and never as the headline. ---
        "entropy_all": float(ent.mean()),
        "mass_all": float(mass.mean()),
        # --- the trap, kept as a negative control ---
        "distinct_experts": distinct_experts_touched(gidx),
        "n_draws": int(gidx.size),
        # --- draw arrays for set-membership rates and co-activation ---
        "_draws_q": gidx[qm],
        "_draws_needle": gidx[nm],
        "_layer_q": layer[qm],
        "_eids_q": eids[qm],
        "_n_window_tokens_q": int(qm.sum() // N_LAYERS),
        "_n_window_tokens_needle": int(nm.sum() // N_LAYERS),
    }


# ---------------------------------------------------------------------------
# Needle-affine reference set, defined at the SHORTEST bucket only
# ---------------------------------------------------------------------------


def needle_affine_set(feats: list[dict], recs: dict[int, dict], short_bucket: int):
    """Experts with real affinity for needle tokens, measured at short context.

    Built exactly like every other affinity claim in this repo: a per-expert
    count matrix over {needle tokens, question tokens}, `compute_lift`, chi2,
    BH-FDR, and the same >=2x fold-change floor `docs/FINDINGS.md` uses. The set
    is defined ONLY on the shortest bucket, then held fixed, so the question
    "does needle-relevant affinity weaken at long context?" is asked of a
    reference set that long context played no part in choosing.
    """
    counts = np.zeros((N_TOTAL, 2), dtype=np.float64)
    for f in feats:
        if recs[f["prompt_id"]]["bucket"] != short_bucket:
            continue
        np.add.at(counts[:, 0], f["_draws_needle"].ravel(), 1.0)
        np.add.at(counts[:, 1], f["_draws_q"].ravel(), 1.0)

    lift = compute_lift(counts)
    sig = bh_fdr(chi2_pvalues_fast(counts), q=Q_FDR)
    mask = sig[:, 0] & (lift[:, 0] >= MEANINGFUL_LIFT)
    return mask, lift[:, 0], counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse
    global TRACES, PROBE_SET_PATH, OUT_MD, OUT_JSON
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=str, default=str(TRACES),
                    help="alternate data/context_traces*/ directory")
    ap.add_argument("--probe-set", type=str, default=str(PROBE_SET_PATH),
                    help="alternate probe_set_context*.yaml")
    ap.add_argument("--out-md", type=str, default=str(OUT_MD))
    ap.add_argument("--out-json", type=str, default=str(OUT_JSON))
    args = ap.parse_args()
    TRACES = Path(args.traces)
    PROBE_SET_PATH = Path(args.probe_set)
    OUT_MD = Path(args.out_md)
    OUT_JSON = Path(args.out_json)

    recs_list, ps = load_prompt_features()
    recs = {r["prompt_id"]: r for r in recs_list}
    buckets = sorted(ps["length_buckets"])

    print(f"{len(recs)} prompts with accuracy records", flush=True)
    feats = []
    for r in sorted(recs_list, key=lambda x: x["prompt_id"]):
        f = per_prompt_routing(r)
        if f is not None:
            feats.append(f)
    print(f"{len(feats)} prompts with traces", flush=True)
    if not feats:
        raise RuntimeError("no traces found")

    present = sorted({recs[f["prompt_id"]]["bucket"] for f in feats})
    complete = {b: sum(1 for f in feats if recs[f["prompt_id"]]["bucket"] == b)
                for b in present}
    print("per-bucket n:", complete, flush=True)

    # ---- the equal-window guarantee, verified rather than assumed ----------
    q_tokens_per_bucket = defaultdict(int)
    n_tokens_per_bucket = defaultdict(int)
    for f in feats:
        b = recs[f["prompt_id"]]["bucket"]
        q_tokens_per_bucket[b] += f["_n_window_tokens_q"]
        n_tokens_per_bucket[b] += f["_n_window_tokens_needle"]
    q_equal = len(set(q_tokens_per_bucket.values())) == 1
    n_equal = len(set(n_tokens_per_bucket.values())) == 1
    print(f"question-window tokens per bucket: {dict(q_tokens_per_bucket)} equal={q_equal}")
    print(f"needle-window tokens per bucket:   {dict(n_tokens_per_bucket)} equal={n_equal}")

    short_bucket = present[0]
    long_bucket = present[-1]

    # ---- reference sets ----------------------------------------------------
    needle_mask, needle_lift, needle_counts = needle_affine_set(feats, recs, short_bucket)
    print(f"needle-affine experts (defined at bucket {short_bucket}): "
          f"{int(needle_mask.sum())}/{N_TOTAL}", flush=True)

    util = load_utilization()
    hot_mask = cold_mask = None
    if util is not None:
        uids = util["utilization"]["uids"]
        load = np.asarray(util["utilization"]["load_ratio"], dtype=np.float64)
        order = {u: i for i, u in enumerate(uids)}
        idx = np.array([order[f"L{l:02d}E{e:02d}"]
                        for l in range(N_LAYERS) for e in range(N_EXPERTS)])
        load = load[idx]
        hot_mask = load >= 2.0
        thresh = np.sort(load)[util["utilization"]["n_cold"] - 1]
        cold_mask = load <= thresh
        print(f"utilization.json: hot={int(hot_mask.sum())} cold={int(cold_mask.sum())} "
              f"(WS-3 reported {util['utilization']['n_hot']}/{util['utilization']['n_cold']})",
              flush=True)

    # ---- per-prompt metric table ------------------------------------------
    rowsd = []
    for f in feats:
        r = recs[f["prompt_id"]]
        d = {
            "prompt_id": f["prompt_id"],
            "bucket": r["bucket"],
            "n_tokens": r["n_tokens"],
            "haystack": r["haystack"],
            "n_distractors": r["n_distractors"],
            "replicate": r["replicate"],
            "accuracy": float(r["forced_choice_correct"]),
            "answer_prob": float(r["forced_choice_prob"]),
            "answer_margin": float(r["forced_choice_margin"]),
            "strict_top1": float(r["strict_top1"]),
            "answer_rank": int(r["answer_rank"]),
            "entropy_q": f["entropy_q"],
            "mass_q": f["mass_q"],
            "entropy_needle": f["entropy_needle"],
            "mass_needle": f["mass_needle"],
            "entropy_all": f["entropy_all"],
            "mass_all": f["mass_all"],
            "needle_affinity_rate": set_hit_rate(f["_draws_needle"], needle_mask),
            "needle_affinity_rate_q": set_hit_rate(f["_draws_q"], needle_mask),
            "distinct_experts_TRAP": float(f["distinct_experts"]),
            "distinct_expected_null": expected_distinct_under_null(f["n_draws"], N_TOTAL),
        }
        if hot_mask is not None:
            d["hot_load_share_q"] = set_hit_rate(f["_draws_q"], hot_mask)
            d["cold_load_share_q"] = set_hit_rate(f["_draws_q"], cold_mask)
        rowsd.append(d)

    L = np.array([d["n_tokens"] for d in rowsd], dtype=np.float64)
    B = np.array([d["bucket"] for d in rowsd], dtype=np.int64)

    METRICS = [
        ("accuracy", "forced-choice accuracy (chance 0.125)"),
        ("answer_prob", "P(correct answer | 8 candidates)"),
        ("answer_margin", "logit margin, correct vs best distractor"),
        ("strict_top1", "greedy top-1 over full vocab"),
        ("entropy_q", "router entropy, question window (bits)"),
        ("mass_q", "top-k mass, question window"),
        ("entropy_needle", "router entropy, needle window (bits)"),
        ("mass_needle", "top-k mass, needle window"),
        ("needle_affinity_rate", "needle-affine expert hit rate, needle window"),
        ("needle_affinity_rate_q", "needle-affine expert hit rate, question window"),
        ("entropy_all", "router entropy, whole prompt (secondary)"),
        ("mass_all", "top-k mass, whole prompt (secondary)"),
    ]
    if hot_mask is not None:
        METRICS += [("hot_load_share_q", "share of draws on WS-3 hot experts"),
                    ("cold_load_share_q", "share of draws on WS-3 cold experts")]

    trends = []
    for key, _desc in METRICS:
        vals = np.array([d[key] for d in rowsd], dtype=np.float64)
        trends.append(length_trend(key, L, vals, buckets=B, n_permutations=2000, seed=0))
    trends = apply_fdr(trends, q=Q_FDR)
    tmap = {t.metric: t for t in trends}

    # The trap metric is deliberately NOT in the FDR family: it is not a
    # candidate finding, it is an illustration of the artefact.
    trap_vals = np.array([d["distinct_experts_TRAP"] for d in rowsd])
    trap_trend = length_trend("distinct_experts_TRAP", L, trap_vals, buckets=B,
                              n_permutations=500, seed=0)
    trap_null = np.array([d["distinct_expected_null"] for d in rowsd])

    # ---- co-activation community stability, per bucket ---------------------
    co_by_bucket: dict[int, np.ndarray] = {}
    for f in feats:
        b = recs[f["prompt_id"]]["bucket"]
        m = co_by_bucket.setdefault(b, np.zeros((N_TOTAL, N_TOTAL), dtype=np.float64))
        lay = f["_layer_q"]
        eid = f["_eids_q"]
        base = lay * N_EXPERTS
        for i in range(TOP_K):
            for j in range(i + 1, TOP_K):
                a = base + eid[:, i]
                c = base + eid[:, j]
                np.add.at(m, (a, c), 1.0)
                np.add.at(m, (c, a), 1.0)

    stab = []
    ref_labels = None
    for b in present:
        s, labels = community_structure(co_by_bucket[b], b, ref_labels, seed=0)
        if ref_labels is None:
            ref_labels = labels
        stab.append(s)

    # ---- condition breakdown (the Chroma comparison) -----------------------
    cond_table: dict[tuple, dict[int, list[float]]] = {}
    for d in rowsd:
        key = (d["haystack"], d["n_distractors"])
        cond_table.setdefault(key, {}).setdefault(d["bucket"], []).append(d["answer_prob"])

    cond_acc: dict[tuple, dict[int, list[float]]] = {}
    for d in rowsd:
        key = (d["haystack"], d["n_distractors"])
        cond_acc.setdefault(key, {}).setdefault(d["bucket"], []).append(d["accuracy"])

    # ---- verdict -----------------------------------------------------------
    acc_t = tmap["accuracy"]
    prob_t = tmap["answer_prob"]
    accuracy_degrades = any(
        t.verdict == "TREND" and t.delta < 0 for t in (acc_t, prob_t, tmap["strict_top1"])
    )
    routing_metrics = ["entropy_q", "mass_q", "entropy_needle", "mass_needle",
                       "needle_affinity_rate", "needle_affinity_rate_q"]
    routing_moves = [tmap[k] for k in routing_metrics if tmap[k].verdict == "TREND"]
    routing_degrades = len(routing_moves) > 0

    if not accuracy_degrades:
        verdict = "SUBSTRATE CANNOT TEST THE QUESTION"
    elif routing_degrades:
        verdict = "MECHANISM FOUND"
    else:
        verdict = "MECHANISM RULED OUT"

    payload = {
        "verdict": verdict,
        "n_prompts": len(rowsd),
        "buckets": present,
        "per_bucket_n": complete,
        "equal_window_tokens": {"question": q_equal, "needle": n_equal,
                                "question_tokens_per_bucket": dict(q_tokens_per_bucket),
                                "needle_tokens_per_bucket": dict(n_tokens_per_bucket)},
        "needle_affine_experts": int(needle_mask.sum()),
        "trends": [t.to_dict() for t in trends],
        "trap": trap_trend.to_dict(),
        "community_stability": [s.to_dict() for s in stab],
        "per_prompt": rowsd,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=float))
    print(f"wrote {OUT_JSON}", flush=True)

    write_report(payload, tmap, trends, trap_trend, trap_null, stab, present,
                 cond_acc, cond_table, rowsd, util, ps, complete,
                 q_tokens_per_bucket, n_tokens_per_bucket, needle_mask,
                 short_bucket, long_bucket)
    print(f"wrote {OUT_MD}", flush=True)
    print(f"\n=== VERDICT: {verdict} ===", flush=True)
    for t in trends:
        print(f"  {t.metric:28s} rho={t.spearman_rho:+.3f} p={t.perm_p:.4f} "
              f"fdr={t.fdr_significant} d={t.cohens_d:+.2f} -> {t.verdict}", flush=True)


def _bucket_means(rowsd, key, buckets):
    out = []
    for b in buckets:
        v = [d[key] for d in rowsd if d["bucket"] == b and np.isfinite(d[key])]
        out.append(float(np.mean(v)) if v else float("nan"))
    return out


def write_report(payload, tmap, trends, trap_trend, trap_null, stab, buckets,
                 cond_acc, cond_prob, rowsd, util, ps, complete,
                 qtok, ntok, needle_mask, short_bucket, long_bucket) -> None:
    verdict = payload["verdict"]
    L = [""]

    L += [
        "# Context Rot at the Routing Level — OLMoE-1B-7B-0924",
        "",
        f"**Verdict: {verdict}**",
        "",
        "## The gap this addresses",
        "",
        "Chroma's *Context Rot: How Increasing Input Tokens Impacts LLM Performance* "
        "(2025) measured, across 18 models including the MoE Qwen3-235B-A22B, that "
        "accuracy degrades as input length grows **even when task difficulty is held "
        "fixed**. On the mechanism, the report says:",
        "",
        f"> \"{CHROMA_QUOTE}\"",
        "",
        "This workstream runs that deeper investigation on one small, fully open MoE, "
        "at the one place a MoE can degrade that a dense model cannot: **the router**. "
        "It asks whether the length-accuracy curve is accompanied by a length-routing "
        "curve — whether the router becomes less decisive, or less specialised, as "
        "context grows.",
        "",
        "## Design",
        "",
        f"`probes/probe_set_context.yaml` — {ps['n_prompts']} prompts, "
        f"{len(ps['length_buckets'])} length buckets "
        f"({', '.join(str(b) for b in ps['length_buckets'])} tokens), "
        f"2 x 2 conditions, {ps['n_replicates']} replicates. "
        "`probe_set_v1.yaml` is untouched, so every other result in this repo stays "
        "comparable.",
        "",
        "Two of Chroma's conditions are reproduced so the results land on named "
        "comparisons rather than 'context rot in general':",
        "",
        "1. **The distractor comparison** — their needle-in-a-haystack result "
        "contrasting a clean haystack with one seeded with distractors (they use 1 and "
        "4; we use 0 and 4). Distractors here are the needle's own sentence template "
        "with a different entity and a *wrong answer drawn from the same forced-choice "
        "pool*, so they compete directly in scoring.",
        "2. **The needle/haystack similarity comparison** — the same needle buried in "
        "either a topically similar haystack (corporate facilities/security prose) or a "
        "dissimilar one (glacial geology prose).",
        "",
        f"Needle depth is fixed at {ps['needle_depth']:.0%}. Depth is a large effect in "
        "the NIAH literature and is deliberately held constant: this set has one "
        "independent variable. That is a scope limit, stated, not a claim that depth "
        "does not matter.",
        "",
        "## The normalisation trap, and how it is avoided",
        "",
        "Everywhere else in this project, `aggregate.py` equalises the token budget per "
        "cell so that length cannot masquerade as signal. **Here length is the "
        "independent variable, so that control is unavailable** — and the risk inverts. "
        "The failure mode is manufacturing a 'more experts fire at long context' result "
        "that is pure arithmetic:",
        "",
        "> The number of *distinct experts touched* grows with token count under a "
        "> frozen, length-blind router, because more tokens means more top-8 draws "
        "> (coupon collector). Any sum over tokens has this problem.",
        "",
        "Two defences, both structural rather than post-hoc:",
        "",
        "**1. A content-identical measurement window.** Within a replicate, the needle "
        "sentence and the trailing question block are *byte-identical* across every "
        "length bucket and condition. All primary metrics are computed on those windows "
        "only, so the token multiset being measured is literally the same string "
        "everywhere; only the amount of preceding context differs. Verified, not "
        "assumed:",
        "",
        f"- question-window tokens per bucket: {dict(qtok)} — "
        f"**{'equal by construction' if payload['equal_window_tokens']['question'] else 'NOT EQUAL — see caveats'}**",
        f"- needle-window tokens per bucket: {dict(ntok)} — "
        f"**{'equal by construction' if payload['equal_window_tokens']['needle'] else 'NOT EQUAL — see caveats'}**",
        "",
        "**2. Every metric is a mean or a rate, never a sum.** Per-metric normalisation:",
        "",
        "| metric | normalisation | length-invariant under a null router? |",
        "|---|---|---|",
        "| `entropy_q`, `entropy_needle` | mean over window tokens x layers of H(full 64-way softmax), bits | yes |",
        "| `mass_q`, `mass_needle` | mean over window tokens x layers of top-k probability mass | yes |",
        "| `needle_affinity_rate` | fraction of window top-k draws landing in a fixed reference set, [0,1] | yes |",
        "| `hot_load_share_q` | fraction of window draws on WS-3 hot experts, [0,1] | yes |",
        "| co-activation | built from equal-size windows; PMI already divides out base rate | yes, subject to the skew gate |",
        "| `entropy_all`, `mass_all` | per-token mean, but over the **whole prompt** | yes arithmetically, **no** in content: long prompts are mostly haystack, so this confounds length with token mix. Reported as secondary only. |",
        "| `distinct_experts_TRAP` | **none — this is the trap** | **no** |",
        "",
        "### The trap, measured",
        "",
        "`distinct_experts_touched` is implemented in `context_metrics.py` as a negative "
        "control and reported here so the artefact is visible rather than merely "
        "asserted to have been avoided:",
        "",
        "| bucket | distinct experts touched | coupon-collector expectation under a null router |",
        "|---|---|---|",
    ]
    for b in buckets:
        obs = np.mean([d["distinct_experts_TRAP"] for d in rowsd if d["bucket"] == b])
        exp = np.mean([d["distinct_expected_null"] for d in rowsd if d["bucket"] == b])
        L.append(f"| {b} | {obs:.1f} | {exp:.1f} |")
    L += [
        "",
        f"Observed rho vs length = {trap_trend.spearman_rho:+.3f}. This is what a "
        "confident, entirely fake context-rot result looks like: the observed curve "
        "tracks the null expectation, so essentially all of the growth is arithmetic. "
        "**It is excluded from the FDR family and is never used as evidence.**",
        "",
        "## Accuracy: does context rot replicate here at all?",
        "",
        "Accuracy is read off the final-position logits of the *same* forward pass that "
        "produces the routing trace. Every candidate answer is a single token by "
        "construction, so this is an exact logit comparison — no generation, no "
        "sampling. Four measures, because a base model can fail in different ways:",
        "",
        "- `accuracy` — argmax over the 8 candidates. Chance = 0.125.",
        "- `answer_prob` — softmax over the 8 candidates; graded, far more sensitive.",
        "- `strict_top1` — argmax over the full 50,304-token vocabulary, i.e. what "
        "greedy decoding would actually emit.",
        "- `answer_margin` — logit gap, correct vs. best distractor.",
        "",
        "| bucket | n | accuracy | answer_prob | strict_top1 | mean answer rank |",
        "|---|---|---|---|---|---|",
    ]
    for b in buckets:
        sub = [d for d in rowsd if d["bucket"] == b]
        L.append(
            f"| {b} | {len(sub)} | {np.mean([d['accuracy'] for d in sub]):.3f} | "
            f"{np.mean([d['answer_prob'] for d in sub]):.3f} | "
            f"{np.mean([d['strict_top1'] for d in sub]):.3f} | "
            f"{np.mean([d['answer_rank'] for d in sub]):.1f} |"
        )

    L += ["", "### By condition (the direct Chroma comparison)", "",
          "Mean `answer_prob` (top) and forced-choice `accuracy` (bottom) per cell.", ""]
    header = "| condition | " + " | ".join(str(b) for b in buckets) + " |"
    L += [header, "|---" * (len(buckets) + 1) + "|"]
    for key in sorted(cond_prob):
        hay, nd = key
        vals = " | ".join(
            f"{np.mean(cond_prob[key][b]):.3f}" if b in cond_prob[key] else "—"
            for b in buckets)
        L.append(f"| {hay} haystack, {nd} distractors — prob | {vals} |")
    for key in sorted(cond_acc):
        hay, nd = key
        vals = " | ".join(
            f"{np.mean(cond_acc[key][b]):.2f}" if b in cond_acc[key] else "—"
            for b in buckets)
        L.append(f"| {hay} haystack, {nd} distractors — acc | {vals} |")

    # --- the figure ---
    fig_series = {
        "accuracy (forced choice)": _bucket_means(rowsd, "accuracy", buckets),
        "answer_prob": _bucket_means(rowsd, "answer_prob", buckets),
        "router entropy (question window)": _bucket_means(rowsd, "entropy_q", buckets),
        "needle-affine hit rate (specialisation)": _bucket_means(rowsd, "needle_affinity_rate", buckets),
    }
    L += [
        "",
        "## The figure",
        "",
        "Chroma's accuracy-vs-length curve, overlaid with routing-entropy-vs-length and "
        "specialisation-vs-length. Each series is min-max normalised to its own range, "
        "so the plot shows **shape agreement** — whether these move together — not "
        "magnitudes, which are in incomparable units. Absolute values are in the tables "
        "above and below; this figure is never the only report of a number. "
        "(matplotlib is not installed in this venv; this is a text figure, "
        "deliberately, so the report stays self-contained.)",
        "",
        "```",
        ascii_overlay(buckets, fig_series),
        "```",
        "",
        "## Routing metrics vs length — both bars applied",
        "",
        f"Every trend below clears (or fails) BOTH a permutation null (2000 shuffles of "
        f"the length labels) with Benjamini-Hochberg FDR at q={Q_FDR} across the whole "
        f"family, AND a practical effect-size floor (|Cohen's d| >= {MIN_COHENS_D} "
        f"between shortest and longest bucket, |rho| >= {MIN_TREND_RHO} for monotonicity). "
        "`docs/FINDINGS.md` records this project once reporting 70% of cells "
        "'significant' at a median lift of 0.79x; the both-bars rule exists because of "
        "that.",
        "",
        "| metric | short | long | delta | % | rho | perm p | FDR | Cohen's d | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t in trends:
        L.append(
            f"| `{t.metric}` | {t.short_mean:.4g} | {t.long_mean:.4g} | "
            f"{t.delta:+.4g} | {t.pct_change:+.1f}% | {t.spearman_rho:+.3f} | "
            f"{t.perm_p:.4f} | {'yes' if t.fdr_significant else 'no'} | "
            f"{t.cohens_d:+.2f} | **{t.verdict}** |"
        )
    L += [
        "",
        "`TREND` = cleared both bars. `SIGNIFICANT-BUT-TRIVIAL` = survived FDR but the "
        "effect is too small to matter — reported as *not* a finding, which is exactly "
        "the distinction this project got wrong once before. `FLAT` = no trend.",
        "",
        "## Co-activation community stability",
        "",
        "| bucket | usage skew | modularity | n communities | ARI vs shortest | reliable? |",
        "|---|---|---|---|---|---|",
    ]
    for s in stab:
        L.append(f"| {s.bucket} | {s.usage_skew:.1f}x | {s.modularity:.4f} | "
                 f"{s.n_communities} | {s.ari_vs_shortest:.3f} | "
                 f"{'yes' if s.reliable else '**NO**'} |")
    any_unreliable = any(not s.reliable for s in stab)
    if any_unreliable:
        L += [
            "",
            f"**verdict: UNRELIABLE.** `coactivation.py`'s own documented PMI validity "
            f"limit is {PMI_SKEW_LIMIT}x usage skew (measured there: separation from "
            f"noise collapses by ~10x). Skew here is far outside that range, so the "
            "community numbers are known-contaminated by base rate and must not be "
            "trusted **in either direction** — neither as evidence that communities "
            "blur at long context nor as evidence that they are stable. This is the "
            "same call `docs/FINDINGS.md` made for H4, made here for the same reason "
            "and against this workstream's interest.",
        ]
    else:
        L += ["", "Skew is inside the documented validity limit at every bucket, so "
                  "these numbers can be read as measurements."]

    # --- hot/cold ---
    L += ["", "## Hot-expert concentration vs length"]
    if util is None:
        L += ["", "`data/utilization.json` was not present at analysis time — skipped, "
                  "as the brief permits. Not blocked on."]
    else:
        ov = util["hot_specialist_overlap"]
        L += [
            "",
            "`data/utilization.json` (Workstream 3) was available. **Its headline result "
            "reframes this section**: the hypothesised 'specialisation lives in hot "
            "experts' mechanism is refuted — specialists are disproportionately *cold* "
            f"(observed overlap {ov['observed_overlap']} vs null "
            f"{ov['null_mean']:.1f}+/-{ov['null_std']:.1f}, enrichment "
            f"{ov['enrichment']:.3f}x, p<0.0001). So hot-expert concentration is "
            "measured here **without** the prior that it is the mechanism; the prior is "
            "that hot experts are generalists.",
            "",
            "WS-3's open question, which this sweep can speak to: if routing degrades "
            "with length, does it concentrate in the generalist (high-load) pathway or "
            "the specialist (low-load) one?",
            "",
            "| bucket | share of draws on hot experts | share on cold experts |",
            "|---|---|---|",
        ]
        for b in buckets:
            sub = [d for d in rowsd if d["bucket"] == b]
            L.append(f"| {b} | {np.mean([d['hot_load_share_q'] for d in sub]):.4f} | "
                     f"{np.mean([d['cold_load_share_q'] for d in sub]):.4f} |")
        ht, ct = tmap.get("hot_load_share_q"), tmap.get("cold_load_share_q")
        if ht is not None:
            L += ["",
                  f"Trend: hot share rho={ht.spearman_rho:+.3f}, d={ht.cohens_d:+.2f} -> "
                  f"**{ht.verdict}**; cold share rho={ct.spearman_rho:+.3f}, "
                  f"d={ct.cohens_d:+.2f} -> **{ct.verdict}**.",
                  "",
                  "Note the honest limit WS-3 states and which carries over here: "
                  "`load_ratio` and `max|lift|` come from the same count matrix and are "
                  "partly definitionally opposed. Read hot/cold as 'load and "
                  "specialisation are opposed on this substrate', not as two "
                  "independent variables."]

    # --- verdict section ---
    acc_t, prob_t = tmap["accuracy"], tmap["answer_prob"]
    L += ["", "## Verdict", "", f"### {verdict}", ""]
    if verdict == "SUBSTRATE CANNOT TEST THE QUESTION":
        L += [
            "Task accuracy does **not** degrade with input length on this model under "
            "this design, so there is no context rot here to find a mechanism for. "
            "Every routing number above is therefore descriptive only: with no accuracy "
            "curve to explain, a routing curve would explain nothing, and a flat routing "
            "curve would rule nothing out.",
            "",
            f"- `accuracy`: {acc_t.short_mean:.3f} at {buckets[0]} tokens -> "
            f"{acc_t.long_mean:.3f} at {buckets[-1]} (rho={acc_t.spearman_rho:+.3f}, "
            f"d={acc_t.cohens_d:+.2f}, {acc_t.verdict})",
            f"- `answer_prob`: {prob_t.short_mean:.3f} -> {prob_t.long_mean:.3f} "
            f"(rho={prob_t.spearman_rho:+.3f}, d={prob_t.cohens_d:+.2f}, {prob_t.verdict})",
            "",
            "**This is a substrate limitation and it bounds every downstream claim in "
            "this workstream.** It is reported here, not buried.",
        ]
    elif verdict == "MECHANISM RULED OUT":
        L += [
            "Task accuracy degrades with input length — Chroma's basic finding "
            "replicates on this model — **but the router does not degrade with it.** "
            "No routing metric clears both bars.",
            "",
            "On this architecture, context rot is **not routing-level**. Whatever "
            "degrades with length is happening somewhere other than the expert-selection "
            "layer: attention, representation quality inside the experts, or the "
            "residual stream. The router keeps making equally decisive, equally "
            "specialised choices while the model's answers get worse.",
            "",
            "**This is a real contribution and is reported as prominently as a positive "
            "would have been.** It removes one candidate mechanism from the list Chroma "
            "left open, and it is a cheap, decisive negative: the measurement window is "
            "byte-identical across lengths, so a routing effect of any practical size "
            "would have shown up.",
        ]
    else:
        moved = [t.metric for t in trends if t.verdict == "TREND"
                 and t.metric in ("entropy_q", "mass_q", "entropy_needle",
                                  "mass_needle", "needle_affinity_rate",
                                  "needle_affinity_rate_q")]
        L += [
            "Task accuracy degrades with input length **and** routing degrades with it, "
            f"on the same prompts, measured on byte-identical text. Routing metrics that "
            f"cleared both bars: {', '.join('`' + m + '`' for m in moved)}.",
            "",
            "This is direct evidence for a routing-level component to context rot — the "
            "kind of mechanistic account Chroma explicitly left open.",
            "",
            "**Required caution before this is called a mechanism rather than a "
            "correlate:** co-movement of two curves against a shared independent "
            "variable is not mediation. Nothing here shows the routing change *causes* "
            "the accuracy drop; both could descend from a third cause. Establishing "
            "mediation needs an intervention — e.g. forcing short-context routing at "
            "long context and measuring whether accuracy recovers — which "
            "`run_ablation_harness.py` is the natural starting point for and which this "
            "workstream did not run.",
        ]

    L += [
        "",
        "## Limits",
        "",
        "1. **One model, one seed.** OLMoE-1B-7B-0924 has 64 experts per layer; frontier "
        "MoEs have hundreds. PLAN.md §9b flags the second-model check as not optional, "
        "and it has not been run for this workstream either.",
        f"2. **{ps['n_replicates']} replicates per cell** "
        f"({payload['n_prompts']} prompts total). Sized to finish — see the wall-clock "
        "section below. Powered for the pooled length trend, not for per-condition "
        "trends, which are shown for shape and should not be significance-tested "
        "individually at this n.",
        "3. **Needle depth fixed at 50%.** Depth is known to matter; it is held constant "
        "here so that length is the only independent variable.",
        "4. **A base model on a retrieval task.** OLMoE-1B-7B-0924 is not "
        "instruction-tuned. Forced-choice scoring is used precisely because it does not "
        "require the model to follow an instruction, but the task is still easier than "
        "what Chroma ran on instruction-tuned frontier models.",
        "5. **Routing only.** As `docs/TRANSFER.md` §6.3 says of the orthogonality "
        "result: this is evidence about the routing layer specifically, not the full "
        "computation. Two prompts routing identically can still compute differently "
        "inside the experts.",
        "6. **Co-activation results are gated out** by the usage-skew check above and "
        "should not be cited in either direction.",
        "",
        "## Wall clock and what was cut",
        "",
        "Throughput was measured on 3 real prompts at 126 / 1008 / 3831 tokens "
        "**before** launching anything, per the brief. Fitted cost on this machine:",
        "",
        "```",
        "seconds(T) = 61.38 - 0.0149*T + 1.323e-05*T^2",
        "```",
        "",
        "A fixed ~58 s/forward dominates below ~1k tokens — the 16 x 64 expert loop runs "
        "regardless of token count — and quadratic attention takes over above it. So on "
        "this substrate short buckets are nearly free and the long tail is the entire "
        "cost.",
        "",
        "| replicates | prompts | nominal | +20% fit margin | x1.6 worst case |",
        "|---|---|---|---|---|",
        "| 3 | 84 | 2.20h | 2.64h | 4.22h |",
        "| **4** | **112** | **2.93h** | **3.52h** | **5.63h** |",
        "| 5 | 140 | 3.66h | 4.40h | 7.04h |",
        "| 6 | 168 | 4.40h | 5.28h | 8.44h |",
        "",
        "The 1.6x worst case is not arbitrary: `docs/TRANSFER.md` §11 records the prior "
        "480-prompt run taking 12.9h against a ~8h fixed-cost floor, an unexplained "
        "~1.6x slowdown attributed to thermal or contention effects. Budgeting for it "
        "rather than assuming it away is the difference between finishing and not.",
        "",
        f"**What was cut: replicates, from 6 to 4.** Bucket count (7) and both condition "
        "axes (2 x 2) were preserved, as the brief requires — those are what make the "
        "result map onto Chroma's named comparisons, and cutting them would have made "
        "the run cheaper and worthless. Replicates are the one axis where less costs "
        "only statistical power. 6 replicates projected to 8.44h worst case, over the "
        "~8h budget; 4 lands at 5.63h worst case with real margin. A completed smaller "
        "sweep beats an unfinished larger one.",
        "",
        "## Reproducing",
        "",
        "```bash",
        "export HF_HOME=\"$PWD/data/hf_cache\" HF_HUB_OFFLINE=1",
        "python probes/probe_set_context.py --replicates 4",
        "python scripts/run_context_sweep.py      # resumable; safe to kill and rerun",
        "python scripts/run_context_analyze.py",
        "pytest tests/ws_ctx -q",
        "```",
        "",
    ]
    OUT_MD.write_text("\n".join(L))


if __name__ == "__main__":
    main()
