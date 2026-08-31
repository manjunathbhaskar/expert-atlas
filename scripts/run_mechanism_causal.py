"""The MECHANISM.md causal test: does BOOSTING the needle-affine experts at the
needle's tokens recover accuracy?

`docs/MECHANISM.md` established a correlation and named its own next step:

    "bias the router toward the needle-affine experts when entropy spikes,
     measure whether accuracy recovers ... has not been run."

This script runs it, on the same 192-prompt hard-variant substrate
(`data/context_rot_hard.json` / `data/context_traces_hard/`), with the same
forced-choice scoring (`run_context_sweep.score_answer`, imported, not
reimplemented) and the same needle-affine expert set (`run_context_analyze.
needle_affine_set`, imported and re-derived at the shortest bucket, then
checked against the 75 experts that run reported -- if the count differs, this
script stops rather than quietly measuring a different set).

Design, and why each piece is there
-----------------------------------
The intervention is `expertatlas.steering.boost`: `+delta` on the needle-affine
experts' router logits, **before** softmax+topk, **only at the needle's token
window** (`needle_token_span`, ~12 tokens out of up to 3832).

`docs/ABLATION.md` sets this project's bar for a causal claim: a flattering
number in the treated condition is not enough, it has to beat the controls.
The four-condition structure here is the direct analogue:

  1. `baseline`            -- no steering. Also a reproduction check: it must
                              return the `answer_prob` already stored for that
                              prompt in `data/context_rot_hard.json`.
  2. `needle_boost@small`  -- the treatment, small magnitude.
  3. `needle_boost@large`  -- the treatment, large magnitude.
  4. `random_boost@*`      -- the analogue of `ablate_random`: the same number
                              of experts **per layer**, drawn from the
                              non-needle-affine complement, at the same
                              magnitude, in the same window. Redrawn per prompt
                              from a per-prompt seed, so across prompts this
                              samples a null over draws rather than testing one
                              arbitrary set.

...and it is run on TWO prompt subsets, because "does the fix help where the
mechanism says it should" is a different question from "does poking the router
help":

  * `low_affinity`  -- the lowest `needle_affinity_rate` prompts: where
                       MECHANISM.md's mechanism says the pathway was lost, so
                       this is where restoring it should help.
  * `high_affinity` -- the highest `needle_affinity_rate` prompts: the pathway
                       was already intact, so a mechanism-specific
                       intervention should do little or nothing here. If the
                       boost helps both equally, the claim is "poking the
                       router helps", not "restoring the needle pathway helps".

Both subsets are drawn with an EQUAL NUMBER OF PROMPTS PER LENGTH BUCKET, so
the two subsets are matched on length by construction -- `needle_affinity_rate`
falls with length, so an unmatched split would confound "low affinity" with
"long prompt".

Boost magnitudes are NOT tuned on the evaluated prompts
-------------------------------------------------------
`--calibrate` picks them on a **dev set from a length bucket that is not
evaluated at all**, using a purely routing-side, accuracy-blind criterion: the
median logit gap between the top-k selection threshold and the needle-affine
experts that just missed it. `delta_small` is that gap (roughly "just enough to
matter"), `delta_large` is 4x it. Accuracy on the dev prompts is never looked
at, and both magnitudes are reported in the results table regardless of which
one looks better.

Usage:
    HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 .venv/bin/python \
        scripts/run_mechanism_causal.py --calibrate
    HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 .venv/bin/python \
        scripts/run_mechanism_causal.py --run
    .venv/bin/python scripts/run_mechanism_causal.py --analyze
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from expertatlas.capture import load_model, route_from_logits  # noqa: E402
from expertatlas.context_metrics import router_entropy_bits, set_hit_rate  # noqa: E402
from expertatlas.steering import ExpertSteering, boost, by_layer  # noqa: E402

MODEL_ID = "allenai/OLMoE-1B-7B-0924"
TRACES_HARD = REPO_ROOT / "data" / "context_traces_hard"
PROBE_HARD = REPO_ROOT / "probes" / "probe_set_context_hard.yaml"
HARD_JSON = REPO_ROOT / "data" / "context_rot_hard.json"
OUT_DIR = REPO_ROOT / "data" / "mechanism_causal"
RECORDS = OUT_DIR / "records.jsonl"
CALIB = OUT_DIR / "calibration.json"
OUT_MD = REPO_ROOT / "docs" / "MECHANISM_CAUSAL.md"

N_LAYERS, N_EXPERTS, TOP_K = 16, 64, 8
N_TOTAL = N_LAYERS * N_EXPERTS
EXPECTED_AFFINE = 75          # what docs/MECHANISM.md's run reported
EVAL_BUCKETS = (2048, 3072, 3840)   # the degraded end of CONTEXT_ROT_HARD.md
DEV_BUCKET = 1024                   # calibration only; never evaluated
MIN_EFFECT_DZ = 0.8                 # this project's own Cohen's d floor
Q_FDR = 0.05
N_SIGNFLIP = 20000


# ---------------------------------------------------------------------------
# Reuse of WS1 code, by import (nothing in another lane is edited)
# ---------------------------------------------------------------------------


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _ws1_modules():
    analyze = _load_module(REPO_ROOT / "scripts" / "run_context_analyze.py", "_ws1_analyze")
    analyze.TRACES = TRACES_HARD
    analyze.PROBE_SET_PATH = PROBE_HARD
    sweep = _load_module(REPO_ROOT / "scripts" / "run_context_sweep.py", "_ws1_sweep")
    return analyze, sweep


def needle_affine_pairs(analyze) -> tuple[np.ndarray, set[tuple[int, int]]]:
    """Re-derive the needle-affine expert set exactly as WS1 defined it.

    Only the shortest bucket's traces are read, because that is the only bucket
    `needle_affine_set` uses (the set is defined at short context and then held
    fixed, so long context plays no part in choosing it).
    """
    recs_list, _ps = analyze.load_prompt_features()
    recs = {r["prompt_id"]: r for r in recs_list}
    short_bucket = min(r["bucket"] for r in recs_list)
    feats = [f for f in (analyze.per_prompt_routing(r) for r in recs_list
                         if r["bucket"] == short_bucket) if f is not None]
    if not feats:
        raise RuntimeError(f"no traces for bucket {short_bucket} under {TRACES_HARD}")
    mask, _lift, _counts = analyze.needle_affine_set(feats, recs, short_bucket)
    n = int(mask.sum())
    if n != EXPECTED_AFFINE:
        raise RuntimeError(
            f"re-derived needle-affine set has {n} experts, but docs/MECHANISM.md's "
            f"run reported {EXPECTED_AFFINE}. Refusing to run a causal test on a "
            "different expert set than the correlational result it follows up."
        )
    pairs = {(int(g) // N_EXPERTS, int(g) % N_EXPERTS) for g in np.flatnonzero(mask)}
    return mask, pairs


# ---------------------------------------------------------------------------
# Prompt subsets: length-matched low- vs high-affinity
# ---------------------------------------------------------------------------


def pick_subsets(per_bucket: int) -> dict[str, list[dict]]:
    pp = json.loads(HARD_JSON.read_text())["per_prompt"]
    low, high = [], []
    for b in EVAL_BUCKETS:
        rows = sorted((p for p in pp if p["bucket"] == b),
                      key=lambda p: p["needle_affinity_rate"])
        if len(rows) < 2 * per_bucket:
            raise RuntimeError(f"bucket {b}: only {len(rows)} prompts, need {2 * per_bucket}")
        low += rows[:per_bucket]
        high += rows[-per_bucket:]
    return {"low_affinity": low, "high_affinity": high}


def dev_prompt_ids(n: int) -> list[int]:
    pp = json.loads(HARD_JSON.read_text())["per_prompt"]
    rows = sorted((p for p in pp if p["bucket"] == DEV_BUCKET), key=lambda p: p["prompt_id"])
    return [p["prompt_id"] for p in rows[:n]]


# ---------------------------------------------------------------------------
# One scored forward pass, optionally under an intervention
# ---------------------------------------------------------------------------


def random_pairs_like(affine_pairs, seed: int) -> set[tuple[int, int]]:
    """Same number of experts PER LAYER as the needle-affine set, drawn from the
    complement. Per-layer matching matters: the affine set is not spread evenly
    over layers, and a globally-matched draw would also change *which layers*
    are intervened on, confounding the control."""
    rng = np.random.default_rng(seed)
    affine_by_layer = by_layer(affine_pairs)
    out = set()
    for layer in range(N_LAYERS):
        k = len(affine_by_layer.get(layer, ()))
        if not k:
            continue
        pool = np.array(sorted(set(range(N_EXPERTS)) - affine_by_layer.get(layer, set())))
        out |= {(layer, int(e)) for e in rng.choice(pool, size=k, replace=False)}
    return out


def scored_forward(loaded, text: str, candidate_ids, answer_id: int,
                   needle_span, question_span, affine_mask: np.ndarray,
                   plan: dict | None, sweep) -> dict:
    """One forward pass -> accuracy + the routing metrics needed as a
    manipulation check (did the boost actually move selection?)."""
    enc = loaded.tokenizer(text, return_tensors="pt")
    inputs = {k: v for k, v in enc.items()}
    n_tokens = int(inputs["input_ids"].shape[1])

    steer_ctx = (ExpertSteering(loaded.model, plan, seq_len=n_tokens)
                 if plan else None)
    try:
        with torch.no_grad():
            try:
                out = loaded.model(**inputs, output_router_logits=True, logits_to_keep=1)
            except TypeError:
                out = loaded.model(**inputs, output_router_logits=True)
        if steer_ctx is not None:
            fired = sum(steer_ctx.call_counts.values())
            if fired != len(steer_ctx.patched_layers):
                raise RuntimeError(
                    f"steering fired {fired} times over {len(steer_ctx.patched_layers)} "
                    "patched layers -- expected exactly one call each; the "
                    "intervention did not run as intended"
                )
        router_logits = list(out.router_logits)
        last = out.logits[0, -1, :].float()
    finally:
        if steer_ctx is not None:
            steer_ctx.remove()

    acc = sweep.score_answer(last, candidate_ids, answer_id)

    # Routing metrics from the SAME logits the selection was made from, so an
    # intervened run reports the intervened routing, not a counterfactual.
    n0, n1 = needle_span
    q0, q1 = question_span
    draws_needle, draws_q, ent_needle, ent_all = [], [], [], []
    for layer, lg in enumerate(router_logits):
        lg = lg.reshape(-1, N_EXPERTS)
        ids, _w, _m = route_from_logits(lg, TOP_K, loaded.shape.norm_topk_prob)
        gidx = layer * N_EXPERTS + ids.cpu().numpy()
        draws_needle.append(gidx[n0:n1])
        draws_q.append(gidx[q0:q1])
        ent = router_entropy_bits(lg.float().cpu().numpy())
        ent_needle.append(ent[n0:n1])
        ent_all.append(ent)

    return {
        **acc,
        "n_tokens": n_tokens,
        "needle_affinity_rate": set_hit_rate(np.concatenate(draws_needle), affine_mask),
        "needle_affinity_rate_q": set_hit_rate(np.concatenate(draws_q), affine_mask),
        "entropy_needle": float(np.mean(np.concatenate(ent_needle))),
        "entropy_all": float(np.mean(np.concatenate(ent_all))),
    }


# ---------------------------------------------------------------------------
# Calibration: pick boost magnitudes from routing only, on dev prompts
# ---------------------------------------------------------------------------


def calibrate(loaded, prompts_by_id, affine_pairs, n_dev: int) -> dict:
    """Median logit gap between the top-k threshold and the near-miss
    needle-affine experts, measured in the needle window on dev-bucket prompts.

    Accuracy is not read, and the dev bucket is never evaluated, so this cannot
    be threshold-tuning against the reported result (handoff §5).
    """
    affine_by_layer = by_layer(affine_pairs)
    gaps: list[float] = []
    ids = dev_prompt_ids(n_dev)
    for pid in ids:
        p = prompts_by_id[pid]
        enc = loaded.tokenizer(p["text"], return_tensors="pt")
        with torch.no_grad():
            try:
                out = loaded.model(**enc, output_router_logits=True, logits_to_keep=1)
            except TypeError:
                out = loaded.model(**enc, output_router_logits=True)
        n0, n1 = _spans(p, loaded)[0]
        for layer, lg in enumerate(out.router_logits):
            members = sorted(affine_by_layer.get(layer, ()))
            if not members:
                continue
            win = lg.reshape(-1, N_EXPERTS)[n0:n1].float().cpu().numpy()
            thresh = np.sort(win, axis=-1)[:, -TOP_K]          # k-th largest
            sub = win[:, members]
            miss = thresh[:, None] - sub                        # >0 == not selected
            miss = miss[miss > 0]
            if miss.size:
                gaps.append(float(np.median(miss)))
        print(f"  dev pid={pid} n_tok={int(enc['input_ids'].shape[1])} "
              f"running median gap={np.median(gaps):.3f}", flush=True)

    med = float(np.median(gaps))
    payload = {
        "criterion": "median (over layers x needle-window tokens) logit gap from the "
                     "top-k selection threshold to the needle-affine experts that "
                     "missed it; accuracy never inspected",
        "dev_bucket": DEV_BUCKET,
        "dev_prompt_ids": ids,
        "n_gap_samples": len(gaps),
        "median_gap": med,
        "p25_gap": float(np.percentile(gaps, 25)),
        "p75_gap": float(np.percentile(gaps, 75)),
        "delta_small": round(med, 2),
        "delta_large": round(4.0 * med, 2),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CALIB.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return payload


def _spans(prompt: dict, loaded):
    """(needle_span, question_span) in TOKEN units, from the capture-time
    character spans, using the same offset-mapping helper WS1 used."""
    from expertatlas.context_metrics import token_span_from_chars

    enc = loaded.tokenizer(prompt["text"], return_tensors="pt", return_offsets_mapping=True)
    offsets = [tuple(x) for x in enc["offset_mapping"][0].tolist()]
    return (token_span_from_chars(offsets, prompt["needle_char_span"]),
            token_span_from_chars(offsets, prompt["question_char_span"]))


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def done_keys() -> set[tuple[int, str]]:
    if not RECORDS.exists():
        return set()
    out = set()
    for line in RECORDS.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out.add((r["prompt_id"], r["condition"]))
    return out


def run(args) -> None:
    analyze, sweep = _ws1_modules()
    affine_mask, affine_pairs = needle_affine_pairs(analyze)
    print(f"needle-affine experts: {int(affine_mask.sum())}/{N_TOTAL} "
          f"across {len(by_layer(affine_pairs))} layers", flush=True)

    ps = yaml.safe_load(PROBE_HARD.read_text())
    prompts_by_id = {p["prompt_id"]: p for p in ps["prompts"]}
    stored = {p["prompt_id"]: p for p in json.loads(HARD_JSON.read_text())["per_prompt"]}

    print(f"loading {MODEL_ID} ...", flush=True)
    t0 = time.time()
    loaded = load_model(MODEL_ID, device="cpu", dtype="bfloat16")
    loaded.model.eval()
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)

    candidate_ids = [loaded.tokenizer(" " + w, add_special_tokens=False)["input_ids"][0]
                     for w in ps["candidate_words"]]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise RuntimeError("candidate words collide under the tokenizer")

    if args.calibrate:
        calibrate(loaded, prompts_by_id, affine_pairs, args.n_dev)
        return

    if not CALIB.exists():
        raise SystemExit("run --calibrate first (boost magnitudes must be picked "
                         "on the dev bucket, not on the evaluated prompts)")
    cal = json.loads(CALIB.read_text())
    deltas = {"small": cal["delta_small"], "large": cal["delta_large"]}
    print(f"boost magnitudes from {CALIB.name}: {deltas}", flush=True)

    subsets = pick_subsets(args.per_bucket)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    already = done_keys()

    # Job order is deliberate. Each prompt's baseline is run immediately before
    # its intervened conditions at the same magnitude, and the two subsets are
    # interleaved, so an interrupted run leaves a BALANCED, fully paired design
    # (complete prompts across both subsets at the large magnitude) rather than
    # one subset finished and the other missing. The large magnitude goes first
    # because it is the one the controls are most needed for.
    tiers = [
        [("baseline", None),
         ("needle_boost_large", ("needle", deltas["large"])),
         ("random_boost_large", ("random", deltas["large"]))],
        [("needle_boost_small", ("needle", deltas["small"])),
         ("random_boost_small", ("random", deltas["small"]))],
    ]
    interleaved = [(name, r)
                   for pair in zip(subsets["low_affinity"], subsets["high_affinity"])
                   for name, r in zip(("low_affinity", "high_affinity"), pair)]
    jobs = [(subset, r["prompt_id"], cond, spec)
            for tier in tiers for subset, r in interleaved for cond, spec in tier]
    todo = [j for j in jobs if (j[1], j[2]) not in already]
    print(f"{len(jobs)} (prompt, condition) cells; {len(already)} already recorded; "
          f"{len(todo)} to run", flush=True)
    if args.limit:
        todo = todo[: args.limit]

    run_t0 = time.time()
    for i, (subset, pid, condition, spec) in enumerate(todo):
        p = prompts_by_id[pid]
        needle_span, question_span = _spans(p, loaded)
        if spec is None:
            plan, boosted = None, []
        else:
            kind, delta = spec
            pairs = (affine_pairs if kind == "needle"
                     else random_pairs_like(affine_pairs, seed=1000 + pid))
            plan = {l: [boost(idxs, delta, needle_span)]
                    for l, idxs in by_layer(pairs).items()}
            boosted = sorted(f"L{l:02d}E{e:02d}" for l, e in pairs)

        t = time.time()
        answer_id = loaded.tokenizer(" " + p["answer_word"],
                                     add_special_tokens=False)["input_ids"][0]
        res = scored_forward(loaded, p["text"], candidate_ids, answer_id,
                             needle_span, question_span, affine_mask, plan, sweep)
        rec = {
            "prompt_id": pid, "condition": condition, "subset": subset,
            "bucket": p["bucket"], "delta": None if spec is None else spec[1],
            "needle_token_span": list(needle_span),
            "n_boosted_experts": len(boosted),
            "boosted_experts_sha": _sha(boosted),
            "stored_answer_prob": stored[pid]["answer_prob"],
            "stored_needle_affinity_rate": stored[pid]["needle_affinity_rate"],
            "seconds": round(time.time() - t, 1),
            **res,
        }
        with RECORDS.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

        eta = (time.time() - run_t0) / (i + 1) * (len(todo) - i - 1) / 60
        print(f"[{i+1}/{len(todo)}] pid={pid:>4} {subset:<13} {condition:<19} "
              f"p={res['forced_choice_prob']:.3f} (stored {rec['stored_answer_prob']:.3f}) "
              f"aff={res['needle_affinity_rate']:.4f} "
              f"({rec['seconds']:.0f}s, ETA {eta:.0f} min)", flush=True)
    print("run complete", flush=True)


def _sha(items) -> str:
    import hashlib

    return hashlib.sha1("|".join(items).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _paired(vals_a: np.ndarray, vals_b: np.ndarray, seed: int = 0) -> dict:
    """Paired contrast b - a with an effect size and a sign-flip permutation null.

    Sign-flipping is the paired analogue of `stats.py::shuffle_labels`: under
    "the condition label carries no information", each prompt's difference is
    equally likely to have either sign. Exact for small n and makes no
    normality assumption.
    """
    d = np.asarray(vals_b, float) - np.asarray(vals_a, float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 2 or np.allclose(d, 0):
        return {"n": int(n), "mean_delta": float(d.mean()) if n else float("nan"),
                "dz": 0.0, "perm_p": 1.0}
    dz = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("inf")
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(N_SIGNFLIP, n))
    null = (signs * np.abs(d)).mean(axis=1)
    p = (np.sum(np.abs(null) >= abs(d.mean())) + 1) / (N_SIGNFLIP + 1)
    return {"n": int(n), "mean_delta": float(d.mean()), "dz": dz, "perm_p": float(p)}


def analyze_records() -> dict:
    from expertatlas.stats import bh_fdr

    rows = [json.loads(l) for l in RECORDS.read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"{RECORDS} is empty -- run --run first")

    by_cell = {(r["prompt_id"], r["condition"]): r for r in rows}
    conditions = sorted({r["condition"] for r in rows},
                        key=lambda c: (c != "baseline", c))
    subsets = {}
    for r in rows:
        subsets.setdefault(r["subset"], set()).add(r["prompt_id"])

    # reproduction check on the baseline condition
    base = [r for r in rows if r["condition"] == "baseline"]
    repro = [abs(r["forced_choice_prob"] - r["stored_answer_prob"]) for r in base]
    repro_aff = [abs(r["needle_affinity_rate"] - r["stored_needle_affinity_rate"])
                 for r in base]

    contrasts, cells = [], []
    for subset, pids in sorted(subsets.items()):
        pids = sorted(p for p in pids
                      if all((p, c) in by_cell for c in conditions))
        for cond in conditions:
            sub = [by_cell[(p, cond)] for p in pids]
            cells.append({
                "subset": subset, "condition": cond, "n": len(sub),
                "answer_prob": float(np.mean([r["forced_choice_prob"] for r in sub])),
                "accuracy": float(np.mean([r["forced_choice_correct"] for r in sub])),
                "margin": float(np.mean([r["forced_choice_margin"] for r in sub])),
                "affinity": float(np.mean([r["needle_affinity_rate"] for r in sub])),
                "entropy_needle": float(np.mean([r["entropy_needle"] for r in sub])),
            })
        for cond in [c for c in conditions if c != "baseline"]:
            for metric, key in (("answer_prob", "forced_choice_prob"),
                                ("accuracy", "forced_choice_correct"),
                                ("affinity", "needle_affinity_rate")):
                a = np.array([by_cell[(p, "baseline")][key] for p in pids], float)
                b = np.array([by_cell[(p, cond)][key] for p in pids], float)
                st = _paired(a, b)
                contrasts.append({"subset": subset, "condition": cond,
                                  "metric": metric, **st})

    # FDR across the whole family of accuracy-side tests (affinity contrasts are
    # manipulation checks, not candidate findings, so they are corrected
    # separately rather than diluting the family).
    fam = [c for c in contrasts if c["metric"] in ("answer_prob", "accuracy")]
    if fam:
        p = np.array([c["perm_p"] for c in fam]).reshape(-1, 1)
        sig = bh_fdr(p, q=Q_FDR).ravel()
        for c, s in zip(fam, sig):
            c["fdr_significant"] = bool(s)
            c["passes_effect_size"] = bool(abs(c["dz"]) >= MIN_EFFECT_DZ)
    for c in contrasts:
        c.setdefault("fdr_significant", None)
        c.setdefault("passes_effect_size", None)

    return {
        "conditions": conditions,
        "n_prompts_per_subset": {k: len(v) for k, v in subsets.items()},
        "reproduction_check": {
            "max_abs_answer_prob_diff": float(max(repro)) if repro else None,
            "max_abs_affinity_diff": float(max(repro_aff)) if repro_aff else None,
            "n_baseline_cells": len(base),
        },
        "cells": cells,
        "contrasts": contrasts,
        "calibration": json.loads(CALIB.read_text()) if CALIB.exists() else None,
    }


def _get(contrasts, subset, condition, metric):
    for c in contrasts:
        if (c["subset"], c["condition"], c["metric"]) == (subset, condition, metric):
            return c
    return None


def write_report(res: dict) -> None:
    cells, contrasts = res["cells"], res["contrasts"]
    conds = res["conditions"]
    treat = [c for c in conds if c.startswith("needle_boost")]
    rand = [c for c in conds if c.startswith("random_boost")]
    cal = res["calibration"] or {}

    def cell(subset, cond, key):
        for c in cells:
            if c["subset"] == subset and c["condition"] == cond:
                return c[key]
        return float("nan")

    L = [
        "# Causal test of the MECHANISM.md pathway: boosting the needle-affine experts",
        "",
        "Follow-up to `docs/MECHANISM.md`, which found that `needle_affinity_rate` "
        "predicts per-prompt correctness independent of length (partial rho=+0.64, "
        "p<0.0001) while raw router entropy does not, and which named this experiment "
        "as its own untested next step. Nothing in `docs/MECHANISM.md` or "
        "`docs/CONTEXT_ROT_HARD.md` is modified by this document.",
        "",
        "## Caveats, before any result",
        "",
        "- **This steers routing, not compute.** `expertatlas/steering.py` adds "
        "`+delta` to the needle-affine experts' router logits before softmax+top-k. "
        "Nothing is skipped or unloaded; this is not a speed or memory claim.",
        "- **A boost changes two things at once**: which experts are selected, and "
        "(because the biased softmax puts more mass on them) what gate weight they "
        "receive. Runs here use the default `weights_from=\"biased\"`, so a positive "
        "result does not attribute itself between the two. "
        "`steering.py` supports isolating the selection change; that variant has "
        "**not** been run.",
        "- **Small n, one model, one seed, one task design** -- same scope limits as "
        "`docs/CONTEXT_ROT_HARD.md` §Limits. "
        f"n={min(res['n_prompts_per_subset'].values())} prompts per subset.",
        "- **The needle-affine set is the same lift-based set as MECHANISM.md's** "
        "(re-derived at the shortest bucket and checked to be the same size, 75/1024, "
        "before running), so this inherits that pipeline's assumptions rather than "
        "testing them.",
        "",
        "## Method",
        "",
        f"- Substrate: the 192-prompt hard variant. Evaluated buckets: {list(EVAL_BUCKETS)} "
        "(the degraded end of the length sweep). Dev bucket for calibration: "
        f"{DEV_BUCKET}, never evaluated.",
        "- Intervention window: the needle's own token span only "
        "(`needle_char_span` -> tokens via the capture-time offset mapping), ~12 tokens.",
        f"- Boost magnitudes, chosen **blind to accuracy** on the dev bucket by a "
        f"routing-side criterion (median logit gap from the top-k threshold to the "
        f"near-miss needle-affine experts = {cal.get('median_gap', float('nan')):.3f}): "
        f"small={cal.get('delta_small')}, large={cal.get('delta_large')}. Both are "
        "reported below regardless of which looks better.",
        "- Subsets are **length-matched by construction**: equal prompts per length "
        "bucket in each of `low_affinity` (the intervention's target, where the "
        "pathway is weakest) and `high_affinity` (where it was already intact).",
        "- `random_boost_*` is the analogue of `ablate_random` in `docs/ABLATION.md`: "
        "the same number of experts **per layer**, drawn from the non-affine "
        "complement, same magnitude, same window, redrawn per prompt from a "
        "per-prompt seed.",
        "- Scoring: `run_context_sweep.score_answer` (imported, not reimplemented) -- "
        "forced choice over the same 8 single-token candidates.",
        "",
        "## Reproduction check (baseline condition vs the stored hard-variant run)",
        "",
        f"- max |answer_prob difference| = "
        f"{res['reproduction_check']['max_abs_answer_prob_diff']:.2e}",
        f"- max |needle_affinity_rate difference| = "
        f"{res['reproduction_check']['max_abs_affinity_diff']:.2e}",
        "",
        "A non-trivial difference here would mean this harness is not measuring the "
        "same thing `docs/MECHANISM.md` measured, and nothing below would be "
        "comparable to it.",
        "",
        "## Manipulation check: did the boost actually move routing?",
        "",
        "| subset | condition | needle_affinity_rate (needle window) | mean delta vs baseline | dz | perm p |",
        "|---|---|---|---|---|---|",
    ]
    for subset in sorted(res["n_prompts_per_subset"]):
        for cond in conds:
            c = _get(contrasts, subset, cond, "affinity")
            base_v = cell(subset, cond, "affinity")
            if cond == "baseline":
                L.append(f"| {subset} | {cond} | {base_v:.4f} | — | — | — |")
            else:
                L.append(f"| {subset} | {cond} | {base_v:.4f} | {c['mean_delta']:+.4f} | "
                         f"{c['dz']:+.2f} | {c['perm_p']:.4f} |")

    L += [
        "",
        "## Results: accuracy",
        "",
        "Both bars, side by side, per this project's rules: a paired sign-flip "
        f"permutation p (BH-FDR across the whole accuracy family, q={Q_FDR}) AND a "
        f"paired effect size (|dz| >= {MIN_EFFECT_DZ}).",
        "",
        "| subset | condition | answer_prob | delta | dz | perm p | FDR sig | passes effect size | accuracy |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for subset in sorted(res["n_prompts_per_subset"]):
        for cond in conds:
            ap = cell(subset, cond, "answer_prob")
            acc = cell(subset, cond, "accuracy")
            if cond == "baseline":
                L.append(f"| {subset} | {cond} | {ap:.4f} | — | — | — | — | — | {acc:.3f} |")
                continue
            c = _get(contrasts, subset, cond, "answer_prob")
            L.append(f"| {subset} | {cond} | {ap:.4f} | {c['mean_delta']:+.4f} | "
                     f"{c['dz']:+.2f} | {c['perm_p']:.4f} | {c['fdr_significant']} | "
                     f"{c['passes_effect_size']} | {acc:.3f} |")

    # ---- the multi-condition bar, ABLATION.md style ------------------------
    L += ["", "## Verdict: the four-number bar", "",
          "`docs/ABLATION.md` requires a causal claim to beat its controls, not just "
          "to look good in the treated cell. The analogous bar here:", ""]
    verdict_lines, all_pass = [], []
    for t in treat:
        tag = t.replace("needle_boost_", "")
        r = f"random_boost_{tag}"
        tgt = _get(contrasts, "low_affinity", t, "answer_prob")
        ctl_rand = _get(contrasts, "low_affinity", r, "answer_prob")
        ctl_high = _get(contrasts, "high_affinity", t, "answer_prob")
        if not (tgt and ctl_rand and ctl_high):
            continue
        checks = [
            (f"`{t}` improves answer_prob on low_affinity prompts",
             tgt["mean_delta"] > 0, f"{tgt['mean_delta']:+.4f}"),
            ("...and clears FDR significance AND the effect-size floor",
             bool(tgt["fdr_significant"]) and bool(tgt["passes_effect_size"]),
             f"p={tgt['perm_p']:.4f}, dz={tgt['dz']:+.2f}"),
            (f"...and beats `{r}` on the same prompts",
             tgt["mean_delta"] > ctl_rand["mean_delta"],
             f"{tgt['mean_delta']:+.4f} vs {ctl_rand['mean_delta']:+.4f}"),
            ("...and is selective: helps low_affinity more than high_affinity",
             tgt["mean_delta"] > ctl_high["mean_delta"],
             f"{tgt['mean_delta']:+.4f} vs {ctl_high['mean_delta']:+.4f}"),
        ]
        verdict_lines.append(
            f"**{t}** (mean answer_prob on low_affinity: "
            f"{cell('low_affinity', t, 'answer_prob'):.4f}, baseline "
            f"{cell('low_affinity', 'baseline', 'answer_prob'):.4f}):")
        for desc, ok, num in checks:
            verdict_lines.append(f"- {desc}: {num} — {'YES' if ok else 'NO'}")
        verdict_lines.append("")
        all_pass.append(all(ok for _d, ok, _n in checks))

    L += verdict_lines
    supported = any(all_pass)
    L += [
        f"**Causal claim {'SUPPORTED' if supported else 'NOT SUPPORTED'} at any tested "
        "boost magnitude on this run.**",
        "",
        ("Every clause above is load-bearing: the treatment beating baseline is not "
         "enough, because a random same-size boost also perturbs routing, and an "
         "intervention that helps the already-intact `high_affinity` prompts just as "
         "much is not evidence about the needle pathway specifically."),
        "",
        "## Honest limits",
        "",
        f"- n={min(res['n_prompts_per_subset'].values())} prompts per subset, one model, "
        "one seed, one task design, one needle-affine set. Directional.",
        "- The manipulation itself is coarse: a fixed `+delta` on every needle-affine "
        "expert at every needle token, not an entropy-triggered or per-token-adaptive "
        "policy. `docs/MECHANISM.md` phrased its next step as boosting *when entropy "
        "spikes*; the always-on version tested here is the simpler, more conservative "
        "variant, and a negative result for it does not rule out the triggered one.",
        "- `weights_from=\"biased\"` confounds the selection change with gate-weight "
        "inflation (see caveats).",
        "- Only the needle window is intervened on. A boost applied at the question "
        "window, or across the whole prompt, is a different experiment and was not run.",
        "- No second domain / second task. `PLAN.md` §9b's generalisation standard is "
        "not met by this document alone.",
        "",
        f"Raw per-cell records: `data/mechanism_causal/records.jsonl`. "
        f"Calibration: `data/mechanism_causal/calibration.json`.",
    ]
    OUT_MD.write_text("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true",
                    help="pick boost magnitudes on the dev bucket (routing only)")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--per-bucket", type=int, default=4,
                    help="prompts per length bucket per subset")
    ap.add_argument("--n-dev", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.calibrate or args.run:
        run(args)
    if args.analyze:
        res = analyze_records()
        (OUT_DIR / "summary.json").write_text(json.dumps(res, indent=2))
        write_report(res)
        print(json.dumps({k: v for k, v in res.items()
                          if k in ("n_prompts_per_subset", "reproduction_check")}, indent=2))
        print(f"wrote {OUT_MD}")
    if not (args.calibrate or args.run or args.analyze):
        ap.error("nothing to do: pass --calibrate, --run or --analyze")


if __name__ == "__main__":
    main()
