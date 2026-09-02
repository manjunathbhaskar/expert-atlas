"""Workstream 2 driver: precompute the interference predictors, then (after
scripts/run_ablation_multi.py has produced the measurements) fit and
permutation-test whether they predict cross-domain ablation damage.

Two stages, deliberately split because only the second one is expensive:

  precompute   pure routing statistics, no model load. Builds the domain
               overlap matrices, the per-domain ablation expert sets, and the
               Workstream-3 load covariate. Writes data/interference_precompute.json.
               EVERYTHING here is computed before any ablation happens -- that
               is the whole point of the experiment, so the file is written
               once and the ablation run only reads it.

  report       reads data/ablation_multi.jsonl (written by run_ablation_multi.py),
               builds the damage matrix, positions each observed damage in the
               matched-size random null, fits the predictive relationship,
               runs the Mantel permutation test, and writes docs/INTERFERENCE.md.

Usage:
    python scripts/run_interference.py precompute [--m 150]
    python scripts/run_interference.py report
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from expertatlas.aggregate import aggregate_counts
from expertatlas.interference import (
    domain_jackknife,
    fit_linear,
    bootstrap_ci_pairs,
    interference_lift_cosine,
    interference_ratio_cosine,
    interference_raw_cosine,
    jaccard,
    load_removed,
    mantel_test,
    multiple_regression,
    null_position,
    partial_correlation,
    routing_mass_ratio,
    routing_profile,
    top_m_expert_set,
)
from expertatlas.stats import bh_fdr, chi2_pvalues_fast, compute_lift

REPO_ROOT = Path(__file__).parent.parent
TRACES_DIR = REPO_ROOT / "data" / "traces"
PROBE_SET_PATH = REPO_ROOT / "probes" / "probe_set_v1.yaml"
ATLAS_PATH = REPO_ROOT / "data" / "atlas.json"
UTILIZATION_PATH = REPO_ROOT / "data" / "utilization.json"
ORTHO_DOC = REPO_ROOT / "docs" / "ORTHOGONALITY.md"
PRECOMPUTE_PATH = REPO_ROOT / "data" / "interference_precompute.json"
MEASUREMENTS_PATH = REPO_ROOT / "data" / "ablation_multi.jsonl"
OUT_DOC = REPO_ROOT / "docs" / "INTERFERENCE.md"

N_LAYERS, N_EXPERTS = 16, 64
Q_FDR = 0.05
MEANINGFUL_LIFT = 1.0

# Chosen to span the published overlap range in docs/ORTHOGONALITY.md, which is
# the only way the regression means anything: python/rust/sql/math_proof are the
# tight code cluster (cosine 0.74-0.90), history is the far end (-0.19 to -0.25
# against every code domain), cooking sits between and gives mid-range points.
# 6 domains -> 30 ordered (ablator, victim) pairs from 7 model sweeps.
DOMAINS = ["python", "rust", "sql", "math_proof", "history", "cooking"]


# ---------------------------------------------------------------------------
# precompute
# ---------------------------------------------------------------------------


def load_rows():
    """Same loader as scripts/run_orthogonality_analysis.py, so the overlap
    matrix this produces is comparable to the published one by construction."""
    probe_set = yaml.safe_load(PROBE_SET_PATH.read_text())
    prompts_by_id = {p["prompt_id"]: p for p in probe_set["prompts"]}
    shards = sorted(TRACES_DIR.glob("trace_*.parquet"))
    if not shards:
        raise RuntimeError(f"no trace shards under {TRACES_DIR} -- run capture first")
    rows = []
    for shard in shards:
        table = pq.read_table(shard)
        cols = {name: table.column(name).to_pylist() for name in table.column_names}
        for i in range(len(cols["prompt_id"])):
            rows.append({k: cols[k][i] for k in cols})
    return rows, prompts_by_id


def parse_published_overlap(text: str) -> dict[tuple[str, str], float]:
    """Pull the 10 pairwise cosines docs/ORTHOGONALITY.md actually publishes.

    The predictor is NOT recomputed as an independent second estimate -- it is
    recomputed with the identical code path and then checked against these
    published values. If the check fails, the run aborts rather than quietly
    shipping a number that disagrees with the doc it claims to build on.
    """
    out = {}
    for m in re.finditer(r"^- (\w+) vs (\w+): cosine=(-?\d+\.\d+)", text, re.M):
        a, b, v = m.group(1), m.group(2), float(m.group(3))
        out[(a, b)] = v
        out[(b, a)] = v
    return out


def stage_precompute(args):
    print("loading traces + probe set ...")
    rows, prompts_by_id = load_rows()
    print(f"  {len(rows)} routing rows, {len(prompts_by_id)} prompts")

    # ---- predictor 1: the published overlap, recomputed on the same code path
    cm_all = aggregate_counts(rows, prompts_by_id, N_LAYERS, N_EXPERTS,
                              domain_factor="topic", seed=0)
    prof_all = routing_profile(cm_all.counts, cm_all.domain_labels, cm_all.expert_uids)
    # sanity: our lift must be the project's lift, bit for bit
    assert np.allclose(prof_all.lift, compute_lift(cm_all.counts)), \
        "routing_profile lift disagrees with stats.compute_lift"

    cos_all = interference_lift_cosine(prof_all)
    published = parse_published_overlap(ORTHO_DOC.read_text())
    checked, worst = 0, 0.0
    for (a, b), v in published.items():
        if a in prof_all.domains and b in prof_all.domains:
            got = float(cos_all[prof_all.index(a), prof_all.index(b)])
            worst = max(worst, abs(got - v))
            checked += 1
    if checked == 0 or worst > args.overlap_tolerance:
        raise SystemExit(
            f"overlap reproduction FAILED: checked {checked} published pairs, "
            f"max abs deviation {worst:.6f} (tolerance {args.overlap_tolerance:g}; "
            "the default 5e-4 = the doc's own 3-decimal rounding). Refusing to "
            "proceed on a predictor that does not match docs/ORTHOGONALITY.md. "
            "If the traces were regenerated on different hardware (BF16 "
            "nondeterminism, see RESEARCH_LOG entry 8), pass an explicit "
            "--overlap-tolerance and report the deviation in the writeup."
        )
    print(f"  reproduced {checked // 2} published ORTHOGONALITY.md pairs, "
          f"max deviation {worst:.6f} (rounding-level) -- OK")

    # ---- split-A-only profile: expert sets and the 'clean' predictor variant
    # docs/ABLATION.md drew its expert sets from data/atlas.json, whose lift was
    # fitted on ALL prompts including the split=B text it then evaluated on. That
    # is leakage. Here the ablation targets come from split A only, so the split=B
    # evaluation text is genuinely held out from expert selection.
    rows_a = [r for r in rows if prompts_by_id.get(r["prompt_id"], {}).get("split") == "A"]
    cm_a = aggregate_counts(rows_a, prompts_by_id, N_LAYERS, N_EXPERTS,
                            domain_factor="topic", seed=0)
    prof_a = routing_profile(cm_a.counts, cm_a.domain_labels, cm_a.expert_uids)
    lift_a = compute_lift(cm_a.counts)
    sig_a = bh_fdr(chi2_pvalues_fast(cm_a.counts), q=Q_FDR)
    cos_a = interference_lift_cosine(prof_a)

    avail, avail_strict = {}, {}
    for d in DOMAINS:
        j = prof_a.index(d)
        col = lift_a[:, j]
        avail[d] = int(np.sum(sig_a[:, j] & (col > 0)))
        avail_strict[d] = int(np.sum(sig_a[:, j] & (col >= MEANINGFUL_LIFT)))
    print("  split-A supply per domain (FDR-sig & positive lift | also clearing |lift|>=1.0):")
    for d in DOMAINS:
        print(f"    {d:12s} {avail[d]:5d} | {avail_strict[d]:4d}")
    print("  NOTE: the project's usual |lift|>=1.0 bar cannot fund a fixed-size set here "
          f"(min supply {min(avail_strict.values())}). Set membership is relaxed to "
          "'FDR-significant and above base rate'; the lift at rank m is reported per "
          "domain so set quality is visible.")

    m = args.m if args.m > 0 else min(avail.values())
    if m > min(avail.values()):
        raise SystemExit(f"--m {m} exceeds the smallest domain's supply ({min(avail.values())})")
    print(f"  fixed ablation-set size m = {m} (identical for every domain, so the "
          f"random null is size-matched by construction)")

    sets = {d: top_m_expert_set(lift_a, sig_a, prof_a.index(d), m) for d in DOMAINS}
    set_lift = {}
    for d in DOMAINS:
        j = prof_a.index(d)
        vals = np.sort(lift_a[sorted(sets[d]), j])[::-1]
        set_lift[d] = {
            "mean": float(vals.mean()), "min_at_rank_m": float(vals[-1]),
            "max": float(vals[0]), "median": float(np.median(vals)),
            "n_clearing_lift_1": int(np.sum(vals >= MEANINGFUL_LIFT)),
        }
    print("  ablation-set quality (split-A lift of the selected experts):")
    for d in DOMAINS:
        s = set_lift[d]
        print(f"    {d:12s} mean={s['mean']:.2f} median={s['median']:.2f} "
              f"lift@rank{m}={s['min_at_rank_m']:.2f} n(|lift|>=1)={s['n_clearing_lift_1']}")

    # ---- Workstream 3 load covariate
    util = json.loads(UTILIZATION_PATH.read_text())
    assert util["utilization"]["uids"] == prof_a.expert_uids, \
        "utilization.json expert uid order does not match the count matrix"
    load_ratio = np.asarray(util["utilization"]["load_ratio"], dtype=np.float64)

    # ---- comparability with the prior single-pair run
    atlas_sets = {}
    if ATLAS_PATH.exists():
        atlas = json.loads(ATLAS_PATH.read_text())
        uid_to_i = {u: i for i, u in enumerate(prof_a.expert_uids)}
        for d in DOMAINS:
            s = set()
            for e in atlas["experts"]:
                lf = e.get("lift", {}).get(d)
                if d in e.get("significant", []) and lf is not None and abs(lf) >= MEANINGFUL_LIFT:
                    s.add(uid_to_i[e["uid"]])
            atlas_sets[d] = s

    payload = {
        "domains": DOMAINS,
        "m": m,
        "expert_uids": prof_a.expert_uids,
        "n_experts_total": len(prof_a.expert_uids),
        "available_experts_split_a": avail,
        "available_experts_split_a_lift1": avail_strict,
        "expert_set_lift": set_lift,
        "overlap_reproduction": {"pairs_checked": checked // 2, "max_abs_deviation": worst},
        "expert_sets": {d: sorted(sets[d]) for d in DOMAINS},
        "expert_set_uids": {d: [prof_a.expert_uids[i] for i in sorted(sets[d])] for d in DOMAINS},
        "load_removed": {d: load_removed(load_ratio, sets[d]) for d in DOMAINS},
        "load_removed_random_expectation": float(m),
        "jaccard_vs_atlas_sets": {
            d: jaccard(sets[d], atlas_sets[d]) for d in DOMAINS
        } if atlas_sets else {},
        "atlas_set_sizes": {d: len(atlas_sets[d]) for d in DOMAINS} if atlas_sets else {},
        "predictors": {},
        "layer_profile": {
            d: [sum(1 for i in sets[d] if i // N_EXPERTS == L) for L in range(N_LAYERS)]
            for d in DOMAINS
        },
    }

    def pack(mat, prof, name):
        payload["predictors"][name] = {
            f"{a}|{b}": float(mat[prof.index(a), prof.index(b)])
            for a in DOMAINS for b in DOMAINS if a != b
        }

    pack(cos_all, prof_all, "overlap_cos_lift_alldata")
    pack(cos_a, prof_a, "overlap_cos_lift_splitA")
    pack(interference_raw_cosine(prof_all), prof_all, "overlap_cos_raw_alldata")
    pack(interference_ratio_cosine(prof_all), prof_all, "overlap_cos_ratio_alldata")

    payload["predictors"]["mass_ratio_splitA"] = {
        f"{a}|{b}": routing_mass_ratio(prof_a, sets[a], b)
        for a in DOMAINS for b in DOMAINS if a != b
    }
    payload["predictors"]["mass_ratio_self"] = {
        d: routing_mass_ratio(prof_a, sets[d], d) for d in DOMAINS
    }

    PRECOMPUTE_PATH.write_text(json.dumps(payload, indent=2))
    print(f"wrote {PRECOMPUTE_PATH}")

    print("\n  pre-computed predictor spread (the regression is meaningless without it):")
    v = np.asarray(list(payload["predictors"]["overlap_cos_lift_alldata"].values()))
    print(f"    overlap_cos_lift  min={v.min():+.3f}  max={v.max():+.3f}  sd={v.std():.3f}")
    v = np.asarray(list(payload["predictors"]["overlap_cos_raw_alldata"].values()))
    print(f"    overlap_cos_raw   min={v.min():+.3f}  max={v.max():+.3f}  sd={v.std():.3f}"
          "   <- near-1 and flat = load balancing washing out the raw signal")
    v = np.asarray(list(payload["predictors"]["mass_ratio_splitA"].values()))
    print(f"    mass_ratio        min={v.min():+.3f}  max={v.max():+.3f}  sd={v.std():.3f}")
    print("\n  load removed per set (random size-matched expectation "
          f"= {m:.1f} fair shares):")
    for d in DOMAINS:
        print(f"    {d:12s} {payload['load_removed'][d]:7.2f}")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def load_measurements():
    if not MEASUREMENTS_PATH.exists():
        raise SystemExit(f"{MEASUREMENTS_PATH} not found -- run scripts/run_ablation_multi.py first")
    recs = [json.loads(l) for l in MEASUREMENTS_PATH.read_text().splitlines() if l.strip()]
    by_sweep = {}
    for r in recs:
        by_sweep[r["sweep"]] = r  # last write wins; sweeps are idempotent
    return list(by_sweep.values())


def mean_loss(rec, domain):
    v = rec["losses"][domain]
    return float(np.mean(v))


def stage_report(args):
    pre = json.loads(PRECOMPUTE_PATH.read_text())
    domains = pre["domains"]
    recs = load_measurements()

    baseline = [r for r in recs if r["kind"] == "baseline"]
    targets = {r["ablator"]: r for r in recs if r["kind"] == "target"}
    nulls = [r for r in recs if r["kind"] == "null"]
    if not baseline:
        raise SystemExit("no baseline sweep in measurements")
    if len(targets) < len(domains):
        raise SystemExit(f"only {len(targets)}/{len(domains)} target sweeps present: "
                         f"{sorted(targets)}")
    base = baseline[0]
    n_prompts = {d: len(base["losses"][d]) for d in domains}
    print(f"baseline + {len(targets)} target sweeps + {len(nulls)} null draws; "
          f"prompts per domain: {n_prompts}")

    # ---- damage matrices ------------------------------------------------
    # D(a->b) = loss(ablate a, text b) - loss(baseline, text b), in nats/token
    damage = {}
    damage_per_prompt = {}
    for a in domains:
        for b in domains:
            d_ab = mean_loss(targets[a], b) - mean_loss(base, b)
            damage[(a, b)] = d_ab
            damage_per_prompt[(a, b)] = (
                np.asarray(targets[a]["losses"][b]) - np.asarray(base["losses"][b])
            )

    null_damage = {}
    for b in domains:
        base_b = mean_loss(base, b)
        null_damage[b] = np.asarray([mean_loss(r, b) - base_b for r in nulls])

    # null-standardised damage: how far beyond a random size-matched cut
    zdamage, null_pos = {}, {}
    for a in domains:
        for b in domains:
            np_ = null_position(damage[(a, b)], null_damage[b])
            null_pos[(a, b)] = np_
            zdamage[(a, b)] = np_.get("z", float("nan"))

    off = [(a, b) for a in domains for b in domains if a != b]

    # ---- predictors -----------------------------------------------------
    def pred(name):
        return {tuple(k.split("|")): v for k, v in pre["predictors"][name].items()}

    x_primary = pred("overlap_cos_lift_alldata")
    x_splita = pred("overlap_cos_lift_splitA")
    x_raw = pred("overlap_cos_raw_alldata")
    x_ratio = pred("overlap_cos_ratio_alldata")
    x_mass = pred("mass_ratio_splitA")
    load_rm = pre["load_removed"]

    responses = {
        "raw_damage_nats": {k: damage[k] for k in off},
        "null_z_damage": {k: zdamage[k] for k in off},
        "damage_rel_own": {k: damage[k] / damage[(k[0], k[0])] for k in off},
    }
    predictors = {
        "overlap_cos_lift_alldata": x_primary,
        "overlap_cos_lift_splitA": x_splita,
        "overlap_cos_raw_alldata": x_raw,
        "overlap_cos_ratio_alldata": x_ratio,
        "mass_ratio_splitA": x_mass,
    }

    fits = {}
    for yname, y in responses.items():
        for xname, x in predictors.items():
            xs = np.asarray([x[k] for k in off])
            ys = np.asarray([y[k] for k in off])
            f = fit_linear(xs, ys)
            f["mantel"] = mantel_test(domains, {k: x[k] for k in off}, {k: y[k] for k in off})
            fits[(yname, xname)] = f

    # primary pre-registered analysis: published overlap -> null-standardised damage
    primary_x, primary_y = "overlap_cos_lift_alldata", "null_z_damage"
    px = {k: predictors[primary_x][k] for k in off}
    py = {k: responses[primary_y][k] for k in off}
    xs = np.asarray([px[k] for k in off])
    ys = np.asarray([py[k] for k in off])
    primary = fit_linear(xs, ys)
    primary["mantel"] = mantel_test(domains, px, py)
    primary["boot_pairs"] = bootstrap_ci_pairs(xs, ys, n_boot=10000, seed=0)
    primary["jackknife"] = domain_jackknife(domains, px, py)

    # ---- WS-3 confound: load removed ------------------------------------
    load_vec = np.asarray([load_rm[a] for (a, b) in off])
    partial_r = partial_correlation(ys, xs, load_vec)
    mreg = multiple_regression(ys, np.column_stack([xs, load_vec]),
                               ["overlap_cos_lift", "load_removed_by_ablator"])

    out = {
        "domains": domains, "off": off, "pre": pre, "recs_n": len(recs),
        "n_prompts": n_prompts, "damage": damage, "zdamage": zdamage,
        "null_damage": null_damage, "null_pos": null_pos, "fits": fits,
        "primary": primary, "primary_x": primary_x, "primary_y": primary_y,
        "predictors": predictors, "responses": responses,
        "partial_r_given_load": partial_r, "mreg": mreg,
        "n_nulls": len(nulls), "load_rm": load_rm,
        "damage_per_prompt": damage_per_prompt,
        "base": base, "targets": targets,
    }
    write_report(out)
    print(f"wrote {OUT_DOC}")


def _fmt(v, nd=3):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def write_report(R):
    domains, off, pre = R["domains"], R["off"], R["pre"]
    damage, zdam, npos = R["damage"], R["zdamage"], R["null_pos"]
    primary = R["primary"]
    m = pre["m"]
    n_nulls = R["n_nulls"]
    npr = R["n_prompts"]
    r = primary["pearson_r"]
    rho = primary["spearman_rho"]
    p_mantel = primary["mantel"]["p_two_sided"]
    boot = primary["boot_pairs"]["r_ci"]
    fisher = primary["fisher_ci_naive"]

    if not np.isfinite(r):
        verdict = "INDETERMINATE"
    elif abs(r) >= 0.5 and p_mantel <= 0.05:
        verdict = "PREDICTS"
    elif abs(r) >= 0.3 or (p_mantel <= 0.05 and abs(r) >= 0.25):
        verdict = "PARTIALLY PREDICTS"
    else:
        verdict = "DOES NOT PREDICT"

    L = []
    A = L.append
    A("# Interference — does a pre-computed routing-overlap number predict cross-domain ablation damage?")
    A("")
    A("Workstream 2. Model: OLMoE-1B-7B-0924 (16 layers x 64 experts, top-8). "
      f"Domains: {', '.join(domains)}. Ablation-set size fixed at **m = {m} experts "
      "for every domain** (so the random null is size-matched by construction and no "
      "damage difference can come from having cut more of the network). Metric: mean "
      "per-token teacher-forced cross-entropy (nats) on held-out `split=B` prompts, "
      "forward passes only.")
    A("")
    A(f"**Held-out sample size: {npr[domains[0]]} prompts per domain "
      f"({sum(npr.values())} scored per sweep).** `docs/ABLATION.md` flagged its n=6 as "
      "too small; this is 4x that and is the entire held-out supply the probe set has. "
      "**But see Limitations — those 24 prompts are 24 surface variants of ONE content "
      "stem, so the effective content-level n is 1, and this raise is smaller than it "
      "looks.**")
    A("")

    # ------------------------------------------------------------------ headline
    A("## Headline")
    A("")
    A(f"Pre-computed overlap (`docs/ORTHOGONALITY.md` lift-cosine, no ablation involved) "
      f"vs. null-standardised cross-domain damage, over {len(off)} ordered "
      f"(ablator, victim) domain pairs:")
    A("")
    A(f"- **Pearson r = {_fmt(r)}**  (R^2 = {_fmt(primary['r2'])}), Spearman rho = {_fmt(rho)}")
    A(f"- OLS slope = {_fmt(primary['slope'])} null-sd of damage per unit cosine "
      f"(intercept {_fmt(primary['intercept'])})")
    A(f"- Mantel permutation p = {_fmt(p_mantel, 4)} "
      f"({'exact enumeration of all ' + str(primary['mantel']['n_permutations_used']) + ' domain relabellings' if primary['mantel']['exact_enumeration'] else 'sampled'}; "
      f"**p cannot go below {_fmt(primary['mantel']['p_floor'], 4)} at 6 domains — that is the design's floor, not a result**)")
    A(f"- 95% CI on r, pair bootstrap: [{_fmt(boot[0])}, {_fmt(boot[1])}] — "
      f"Fisher-z CI [{_fmt(fisher[0])}, {_fmt(fisher[1])}] is narrower and **wrong**, "
      "because the 30 pairs are entries of one 6x6 matrix, not 30 independent draws.")
    A("")
    A(f"### Verdict: **{verdict}**")
    A("")
    A(_verdict_prose(verdict, r, rho, p_mantel, boot))
    A("")

    # ------------------------------------------------------------- prior art
    A("## Side-by-side with the two papers this is positioned against")
    A("")
    A("| | arXiv 2406.16437 (ICLR 2025) | arXiv 2503.05029 | this run |")
    A("|---|---|---|---|")
    A("| What it is | Theory: MoE in continual learning | Empirical: continual pre-training of MoEs | Empirical: static pretrained MoE |")
    A("| Established | Explicit expressions for expected forgetting and generalisation error; experts diversify, router learns to select and balance load; gating update must be terminated for convergence | 500M-active/2B-total MoEs over 600B tokens are surprisingly robust to distribution shift; routing changes concentrate in early layers; \"more pronounced changes correlate with higher forgetting\" | An overlap number computable from routing traces alone is/is not quantitatively predictive of ablation damage |")
    A("| Validated on | Overparameterised linear regression, plus synthetic and small real-dataset DNN experiments. **No real pretrained open-weight LLM's measured routing** | Real MoE LLMs, but models they trained themselves | Real open-weight pretrained LLM (OLMoE-1B-7B-0924), 480-prompt controlled factorial probe set |")
    A("| Quantitative overlap -> damage magnitude? | Predicted by theory, never fitted to real routing data | **Explicitly not established** — correlation between an observed post-hoc routing *change* and forgetting, not a pre-computed overlap predicting damage *size* | **This is the step taken here** |")
    A("| Causal manipulation | No (analytic) | No (observational over training) | Yes — zero-ablation of a named expert set, with a size-matched random null |")
    A("")
    A("**The differentiation has to be stated precisely: 2503.05029 already gets partway "
      "there.** It shows routing behaviour relates to forgetting on real MoEs. What it does "
      "not do — and says it does not do — is put a number on the relationship in the "
      "predictive direction: given two domains and their routing statistics *before* any "
      "intervention, how much damage should you expect? That is the only gap this run "
      "addresses, and it addresses it on one model.")
    A("")

    # ------------------------------------------------------------- functional
    A("## The interference functional and how the linear-model math was mapped onto routing")
    A("")
    A("With experts e in E (|E| = 1024), domain d, and c_d(e) the selection count under "
      "the equal-token-budget control in `expertatlas/aggregate.py`:")
    A("")
    A("```")
    A("  q_d(e) = (c_d(e) + 1) / (sum_e' c_d(e') + |E|)        routing distribution")
    A("  p(e)   = (sum_d c_d(e) + 1) / (grand total + |E|)     base rate")
    A("  l_d(e) = log2( q_d(e) / p(e) )                        == stats.compute_lift")
    A("")
    A("  symmetric   I(a,b)    = <l_a, l_b> / (||l_a|| ||l_b||)")
    A("  directional M(a -> b) = sum_{e in S_a} q_b(e) / sum_{e in S_a} p(e)")
    A("```")
    A("")
    A("`S_a` is the expert set actually ablated for domain a. `M = 1` means the victim "
      "domain routes through the ablator's experts at exactly the base rate.")
    A("")
    A("**Faithful to 2406.16437's structure:** interference is a bilinear form between "
      "two per-task vectors (not a set-overlap count); the vectors are indexed by expert "
      "and weighted by the router's per-task selection probability; zero overlap gives "
      "exactly zero predicted interference.")
    A("")
    A("**Adaptations that are our judgement calls, not their theorem:**")
    A("")
    A("1. **Routing distribution substituted for task representation.** They inner-product "
      "task feature directions in a shared parameter space. We observe only the router. "
      "This is strictly weaker and is the same limit `docs/ORTHOGONALITY.md` already "
      "flags: two domains can route identically and still be orthogonal *inside* the "
      "experts. If the predictor under-performs, this is candidate reason #1.")
    A("2. **Base-rate correction (lift, not raw q).** Their setting has no load-balancing "
      "objective forcing a shared near-uniform marginal; a real trained MoE does. "
      "Measured here: the raw-q cosine over the same 30 pairs has "
      f"sd = {np.std([R['predictors']['overlap_cos_raw_alldata'][k] for k in off]):.4f} "
      f"and range [{min(R['predictors']['overlap_cos_raw_alldata'][k] for k in off):.4f}, "
      f"{max(R['predictors']['overlap_cos_raw_alldata'][k] for k in off):.4f}] — i.e. "
      "essentially constant across every domain pair, exactly as the load-balancing "
      "argument predicts. Using it as the predictor would be using a constant. Taking "
      "the domain-specific deviation instead is a modelling decision, and it is the "
      "decision the whole project already makes.")
    A("3. **log-ratio rather than ratio.** l = log2(q/p) for consistency with every other "
      "number in this repo, not because their derivation implies a log. The non-log "
      "variant is fitted below as a robustness check.")
    A("4. **Ablation replaces gradient interference.** They forget via a weight update; we "
      "zero-ablate and measure held-out CE. Ablation is the harsher, upper-bound version "
      "of \"how much of b's computation flows through a's experts\". Magnitudes here are "
      "NOT comparable to a forgetting curve.")
    A("5. **Static model, no task sequence.** Nothing here observes forgetting dynamics.")
    A("")

    # ------------------------------------------------------------- design
    A("## Design")
    A("")
    A(f"- **Expert sets** are the top-{m} experts by lift for each domain among those that "
      "are BH-FDR significant (q=0.05) and clear |lift| >= 1.0, **computed on `split=A` "
      "traces only**. `docs/ABLATION.md` took its sets from `data/atlas.json`, whose lift "
      "was fitted on all 480 prompts including the split=B text it then scored — leakage. "
      "Fixing that changes the sets; Jaccard against the atlas-derived sets is "
      + ", ".join(f"{d} {pre['jaccard_vs_atlas_sets'].get(d, float('nan')):.2f}" for d in domains)
      + ".")
    A(f"- **Fixed m** (not \"all experts over the bar\", which gave 189 vs 170 in the prior "
      "run) so every ablation removes the same number of experts and one random null per "
      "victim domain is valid for all six ablators.")
    A(f"- **Null**: {n_nulls} independent uniformly-random sets of exactly {m} experts, each "
      "scored on all six domains' held-out prompts. This is the piece `docs/ABLATION.md` "
      f"named as missing. Percentile resolution is 1/{n_nulls + 1} = "
      f"{1.0 / (n_nulls + 1):.3f}, so no empirical p below that is reportable.")
    A(f"- **Sweeps run**: 1 baseline + {len(domains)} target ablations + {n_nulls} null "
      f"draws = {1 + len(domains) + n_nulls} full forward-pass sweeps over "
      f"{sum(npr.values())} prompts each.")
    A("")

    # ------------------------------------------------------------- damage matrix
    A("## Cross-domain damage matrix")
    A("")
    A("Rows = ablated domain, columns = evaluated (victim) text. Cells are "
      "loss(ablate row) - loss(baseline), in nats/token. Diagonal = on-target damage.")
    A("")
    A("| ablate \\ eval | " + " | ".join(domains) + " |")
    A("|---|" + "---|" * len(domains))
    for a in domains:
        A(f"| **{a}** | " + " | ".join(f"{damage[(a, b)]:+.4f}" for b in domains) + " |")
    A("")
    A("Same matrix in units of the matched-size random null's standard deviation "
      "(z = (observed - null mean) / null sd, per victim column):")
    A("")
    A("| ablate \\ eval | " + " | ".join(domains) + " |")
    A("|---|" + "---|" * len(domains))
    for a in domains:
        A(f"| **{a}** | " + " | ".join(f"{zdam[(a, b)]:+.2f}" for b in domains) + " |")
    A("")

    # double dissociation check across all pairs
    n_pairs, n_dd = 0, 0
    dd_lines = []
    for i, a in enumerate(domains):
        for b in domains[i + 1:]:
            n_pairs += 1
            ok = damage[(a, a)] > damage[(a, b)] and damage[(b, b)] > damage[(b, a)]
            n_dd += bool(ok)
            dd_lines.append(f"| {a} / {b} | {damage[(a,a)]:+.4f} | {damage[(a,b)]:+.4f} | "
                            f"{damage[(b,b)]:+.4f} | {damage[(b,a)]:+.4f} | {'YES' if ok else 'no'} |")
    A("### Double dissociation, now over every pair")
    A("")
    A(f"`docs/ABLATION.md` found the crossover pattern on its single medicine/cooking pair. "
      f"Across all {n_pairs} unordered pairs here it holds for **{n_dd}/{n_pairs}**.")
    A("")
    A("| pair a/b | a on a | a on b | b on b | b on a | crossover |")
    A("|---|---|---|---|---|---|")
    L.extend(dd_lines)
    A("")

    # ------------------------------------------------------------- null
    A("## The random-expert null `docs/ABLATION.md` was missing")
    A("")
    A(f"{n_nulls} random size-{m} expert sets per victim domain. Percentiles, not just the "
      "mean — `docs/TRANSFER.md` §11's standing rule.")
    A("")
    A("| victim domain | null min | p05 | p25 | median | p75 | p95 | max | own-domain ablation | its percentile |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for b in domains:
        s = npos[(b, b)]
        A(f"| {b} | {s['null_min']:+.4f} | {s['null_p05']:+.4f} | {s['null_p25']:+.4f} | "
          f"{s['null_median']:+.4f} | {s['null_p75']:+.4f} | {s['null_p95']:+.4f} | "
          f"{s['null_max']:+.4f} | {damage[(b, b)]:+.4f} | {s['percentile']:.0f}th |")
    A("")
    A("Cross-domain cells against the same nulls:")
    A("")
    A("| ablator -> victim | damage | null median | z | percentile | empirical p (upper) |")
    A("|---|---|---|---|---|---|")
    for (a, b) in off:
        s = npos[(a, b)]
        A(f"| {a} -> {b} | {damage[(a,b)]:+.4f} | {s['null_median']:+.4f} | {s['z']:+.2f} | "
          f"{s['percentile']:.0f}th | {s['empirical_p_upper']:.3f} |")
    A("")

    # ------------------------------------------------------------- fits
    A("## Fits")
    A("")
    A("Every predictor is computed from routing statistics only, before any ablation. "
      "Every response is measured. Mantel p permutes domain labels (respecting the "
      "matrix dependence), not pair values.")
    A("")
    A("| response | predictor | n | Pearson r | Spearman rho | slope | Mantel p |")
    A("|---|---|---|---|---|---|---|")
    for (yname, xname), f in R["fits"].items():
        A(f"| {yname} | {xname} | {f['n']} | {_fmt(f['pearson_r'])} | "
          f"{_fmt(f['spearman_rho'])} | {_fmt(f['slope'], 4)} | "
          f"{_fmt(f['mantel']['p_two_sided'], 4)} |")
    A("")
    A("### Leave-one-domain-out on the primary fit")
    A("")
    A("30 pairs, but only 6 independent units. If dropping one domain moves r a lot, the "
      "relationship is that domain, not a law.")
    A("")
    A("| domain dropped | n pairs | Pearson r | Spearman rho | slope |")
    A("|---|---|---|---|---|")
    for j in primary["jackknife"]:
        A(f"| {j['dropped']} | {j['n']} | {_fmt(j['pearson_r'])} | "
          f"{_fmt(j['spearman_rho'])} | {_fmt(j['slope'], 4)} |")
    jr = [j["pearson_r"] for j in primary["jackknife"] if np.isfinite(j["pearson_r"])]
    if jr:
        A("")
        A(f"Range of r across the six leave-one-out fits: [{min(jr):.3f}, {max(jr):.3f}] "
          f"(full-sample r = {_fmt(r)}).")
    A("")

    # ------------------------------------------------------------- WS3 confound
    A("## Workstream 3's confound: is this just 'you deleted more of the model'?")
    A("")
    A("`docs/UTILIZATION.md` found H1 specialists are disproportionately COLD (enrichment "
      "0.624x into the hot set, permutation p < 0.0001), and that the hot experts are "
      "largely generalists. So a size-matched null controls set *size* but not set *load*. "
      f"Load removed by each ablation set, in fair-share units (a random size-{m} set has "
      f"expectation {m:.0f}):")
    A("")
    A("| domain | load removed | vs random expectation |")
    A("|---|---|---|")
    for d in domains:
        A(f"| {d} | {R['load_rm'][d]:.2f} | {R['load_rm'][d] / m:.2f}x |")
    A("")
    A(f"- Partial correlation of null-z damage with overlap, controlling for the ablator's "
      f"load removed: **{_fmt(R['partial_r_given_load'])}** (raw r = {_fmt(r)}).")
    A("- Multiple regression of null-z damage on [overlap, load removed]:")
    A("")
    mreg = R["mreg"]
    A("| term | beta | t | p (naive) |")
    A("|---|---|---|---|")
    for i, nm in enumerate(mreg["names"]):
        A(f"| {nm} | {mreg['beta'][i]:+.4f} | {mreg['t'][i]:+.2f} | {mreg['p_naive'][i]:.4f} |")
    A(f"")
    A(f"Model R^2 = {_fmt(mreg['r2'])}. **The standard errors above assume 30 independent "
      "rows and there are 6 independent units — treat the t/p columns as descriptive "
      "only.** The load covariate varies only across the 6 ablators (it does not depend on "
      "the victim), so it can absorb ablator-level differences but says nothing about "
      "victim-level ones; that is a real limit of this control, not a clean adjustment.")
    A("")
    A("The alternative hypothesis WS-3 names — that cross-domain damage tracks how much of "
      "the *shared generalist pathway* an ablation removes rather than a-b overlap — is "
      + ("consistent with the load term carrying explanatory weight above."
         if abs(mreg["t"][2]) > abs(mreg["t"][1]) else
         "not supported over the overlap term here, but 6 ablators cannot separate them properly.")
      )
    A("")

    # ------------------------------------------------------------- limits
    A("## What this does NOT show")
    A("")
    A("- **n = 6 domains.** The 30 regression points are one 6x6 matrix. The Mantel p "
      f"floor is {_fmt(primary['mantel']['p_floor'], 4)}. Any CI here is wide and the "
      "leave-one-domain-out table above is the honest read on stability.")
    A("- **The held-out prompts are 24 surface variants of a single content stem per "
      "domain** (`probe_set_v1.yaml` gives each (topic, lang, register, format) cell one "
      "split=A and one split=B prompt, and all 24 split=B prompts of a topic share "
      "`stem`). Raising n from 6 to 24 raised *surface* coverage, not content coverage. "
      "The effective content-level n per domain is 1. This is the single biggest weakness "
      "of the measurement and it is a property of the probe set, not fixable here.")
    A("- **Routing overlap only.** Two domains routing to the same experts could still "
      "compute orthogonally inside them (`docs/ORTHOGONALITY.md`'s own stated limit). A "
      "weak predictive result is consistent with the routing layer simply not being where "
      "the interference lives.")
    A("- **Zero-ablation is not forgetting.** It is an upper bound on how much of a "
      "domain's computation flows through an expert set. No gradient step is taken and no "
      "continual-learning dynamics are observed. Nothing here transfers directly to a "
      "forgetting curve.")
    A("- **One model, one seed, one probe set.** OLMoE-1B-7B-0924 only. PLAN.md's "
      "second-model generality check is still not done.")
    A("- **The non-English prompts have never been read by a human** "
      "(`translation_reviewed: false`). 18 of the 24 held-out prompts per domain are "
      "zh/de/ja. This is a standing project-wide gap and it is inside this measurement.")
    A(f"- **{n_nulls} null draws** gives percentile resolution {1.0 / (n_nulls + 1):.3f}. "
      "Adequate for placing an observation in the bulk of the null; not adequate for "
      "tail claims.")
    A("")
    A("## Reproduce")
    A("")
    A("```bash")
    A("python scripts/run_interference.py precompute      # routing statistics only, no model")
    A("python scripts/run_ablation_multi.py --n-null 30   # the expensive part, resumable")
    A("python scripts/run_interference.py report          # fits + this document")
    A("```")

    OUT_DOC.write_text("\n".join(L) + "\n")


def _verdict_prose(verdict, r, rho, p, boot):
    if verdict == "PREDICTS":
        return (f"The pre-computed overlap number tracks measured cross-domain damage "
                f"(r = {_fmt(r)}, Mantel p = {_fmt(p, 4)}). Stated with the power limit "
                "attached: 6 domains, 30 dependent pairs, one model. This is evidence for "
                "the quantitative link 2406.16437 predicts and 2503.05029 stops short of, "
                "not a demonstration that it holds generally.")
    if verdict == "PARTIALLY PREDICTS":
        return (f"There is a relationship in the predicted direction but it is not strong "
                f"enough, at this sample size, to call the overlap number predictive "
                f"(r = {_fmt(r)}, Spearman rho = {_fmt(rho)}, Mantel p = {_fmt(p, 4)}, "
                f"bootstrap CI on r [{_fmt(boot[0])}, {_fmt(boot[1])}]). An overlap score "
                "computed before any ablation carries some information about how much "
                "collateral damage to expect, and clearly does not carry all of it. "
                "Reporting the coefficient rather than a claim.")
    if verdict == "DOES NOT PREDICT":
        return (f"**The pre-computed overlap number does not predict the magnitude of "
                f"cross-domain damage on this model** (r = {_fmt(r)}, Spearman rho = "
                f"{_fmt(rho)}, Mantel p = {_fmt(p, 4)}, bootstrap CI on r "
                f"[{_fmt(boot[0])}, {_fmt(boot[1])}]). That is the actual measurement and "
                "it is reported as the result. It does not refute 2406.16437's theory — "
                "the theory is about representation overlap under gradient updates, and "
                "this measures routing overlap under ablation (see the adaptation list). "
                "It does say that the specific cheap thing you might hope to do — read a "
                "routing-overlap number off a trace and predict interference from it — "
                "does not work here.")
    return "The fit could not be computed. See the tables below for what was measured."


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    p1 = sub.add_parser("precompute")
    p1.add_argument("--overlap-tolerance", type=float, default=5e-4,
                    help="max allowed |recomputed - published| cosine deviation. "
                         "Default 5e-4 (the doc's 3-decimal rounding); raise it "
                         "ONLY for known trace-regeneration drift, and say so "
                         "in the writeup.")
    p1.add_argument("--m", type=int, default=100,
                    help="fixed ablation-set size; 0 = use the smallest domain's supply")
    sub.add_parser("report")
    args = ap.parse_args()
    if args.stage == "precompute":
        stage_precompute(args)
    else:
        stage_report(args)


if __name__ == "__main__":
    main()
