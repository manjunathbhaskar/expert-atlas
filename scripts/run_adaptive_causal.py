"""The remaining MECHANISM.md causal variant: ENTROPY-TRIGGERED adaptive boost.

`docs/MECHANISM.md` proposed biasing the router toward the needle-affine
experts "when entropy spikes". `docs/MECHANISM_CAUSAL.md` tested the always-on
fixed-window variant and it FAILED at both magnitudes; the triggered policy is
the one remaining untested causal handle, and this script runs it.

Design (pre-registered before looking at any result)
----------------------------------------------------
* Trigger: per layer, per token, boost fires iff the UNBIASED full-softmax
  router entropy exceeds that layer's threshold `tau_l`, calibrated as the
  90th percentile of per-token entropy on DEV-bucket (1024) prompts — a
  routing-only, accuracy-blind criterion on a bucket that is never evaluated.
  "Spike" = top decile of the layer's own entropy distribution.
* Scope: the WHOLE sequence. The trigger, not a hand-chosen window, decides
  where to act — that is the difference from the failed fixed variant, which
  boosted a fixed ~12-token needle window unconditionally.
* Magnitude: `delta_small` from `data/mechanism_causal/calibration.json`
  (0.93 as reported in docs/MECHANISM_CAUSAL.md), NOT re-tuned — the large
  magnitude was already shown to be indiscriminately destructive, so carrying
  it forward would test nothing new.
* Conditions: `baseline` (repro gate vs the stored hard-variant answer_prob),
  `adaptive_needle` (treatment), `adaptive_random` (same per-layer set sizes
  drawn from the complement, same trigger, same magnitude, redrawn per prompt).
* Subsets: the same length-matched `low_affinity` / `high_affinity` subsets as
  docs/MECHANISM_CAUSAL.md (imported, not reimplemented), same EVAL buckets
  (2048, 3072, 3840), n=12 prompts per subset by default.
* Stats: paired sign-flip permutation p + dz, BH-FDR (q=0.05) across the
  accuracy family; manipulation checks (trigger rate, needle_affinity_rate)
  reported first.

Honest limits: one model, one seed, small n, teacher-forced forced choice —
same scope as docs/MECHANISM_CAUSAL.md. A null here closes out MECHANISM.md's
proposed intervention family at these magnitudes, it does not prove no routing
intervention can work.

Usage:
    HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 .venv/bin/python \
        scripts/run_adaptive_causal.py --calibrate
    HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 .venv/bin/python \
        scripts/run_adaptive_causal.py --run
    .venv/bin/python scripts/run_adaptive_causal.py --analyze
"""

from __future__ import annotations

import argparse
import hashlib
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

from expertatlas.adaptive_steering import AdaptiveEntropySteering  # noqa: E402
from expertatlas.capture import load_model, route_from_logits  # noqa: E402
from expertatlas.context_metrics import router_entropy_bits, set_hit_rate  # noqa: E402
from expertatlas.steering import by_layer  # noqa: E402

MODEL_ID = "allenai/OLMoE-1B-7B-0924"
PROBE_HARD = REPO_ROOT / "probes" / "probe_set_context_hard.yaml"
HARD_JSON = REPO_ROOT / "data" / "context_rot_hard.json"
MC_DIR = REPO_ROOT / "data" / "mechanism_causal"
OUT_DIR = REPO_ROOT / "data" / "adaptive_causal"
RECORDS = OUT_DIR / "records.jsonl"
CALIB = OUT_DIR / "calibration.json"
OUT_MD = REPO_ROOT / "docs" / "ADAPTIVE_CAUSAL.md"

N_LAYERS, N_EXPERTS, TOP_K = 16, 64, 8
N_TOTAL = N_LAYERS * N_EXPERTS
DELTA_SMALL_DOC = 0.93     # docs/MECHANISM_CAUSAL.md's reported small magnitude
TAU_PCTL = 90.0
DEV_BUCKET = 1024
MIN_EFFECT_DZ = 0.8
Q_FDR = 0.05
N_SIGNFLIP = 20000


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _mc():
    return _load_module(REPO_ROOT / "scripts" / "run_mechanism_causal.py", "_mc")


def needle_affine_pairs_local(analyze):
    """Same derivation as run_mechanism_causal.needle_affine_pairs but WITHOUT
    the ==75 guard: this box's regenerated hard traces yield 78 experts (the
    old box's run reported 75 — BF16 cross-machine drift moves a few
    borderline experts across the lift threshold). The count and this
    discrepancy are recorded in the output doc rather than hidden; the set
    used here is the one consistent with THIS box's traces, i.e. the same
    substrate the baselines below are reproduced on."""
    recs_list, _ps = analyze.load_prompt_features()
    recs = {r["prompt_id"]: r for r in recs_list}
    short_bucket = min(r["bucket"] for r in recs_list)
    feats = [f for f in (analyze.per_prompt_routing(r) for r in recs_list
                         if r["bucket"] == short_bucket) if f is not None]
    mask, _lift, _counts = analyze.needle_affine_set(feats, recs, short_bucket)
    pairs = {(int(g) // N_EXPERTS, int(g) % N_EXPERTS) for g in np.flatnonzero(mask)}
    return mask, pairs


def _delta_small() -> float:
    mc_cal = MC_DIR / "calibration.json"
    if mc_cal.exists():
        return float(json.loads(mc_cal.read_text())["delta_small"])
    return DELTA_SMALL_DOC


def _sha(items: list[str]) -> str:
    return hashlib.sha256("|".join(items).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Calibration: per-layer entropy thresholds from dev-bucket prompts only
# ---------------------------------------------------------------------------


def calibrate(loaded, mc, prompts_by_id, n_dev: int) -> dict:
    ids = mc.dev_prompt_ids(n_dev)
    ent_by_layer: dict[int, list[np.ndarray]] = {l: [] for l in range(N_LAYERS)}
    for pid in ids:
        p = prompts_by_id[pid]
        enc = loaded.tokenizer(p["text"], return_tensors="pt")
        with torch.no_grad():
            try:
                out = loaded.model(**enc, output_router_logits=True, logits_to_keep=1)
            except TypeError:
                out = loaded.model(**enc, output_router_logits=True)
        for layer, lg in enumerate(out.router_logits):
            ent = router_entropy_bits(lg.reshape(-1, N_EXPERTS).float().cpu().numpy())
            ent_by_layer[layer].append(ent)
        print(f"  dev pid={pid} n_tok={int(enc['input_ids'].shape[1])}", flush=True)
    tau = {str(l): float(np.percentile(np.concatenate(v), TAU_PCTL))
           for l, v in ent_by_layer.items()}
    payload = {
        "criterion": f"per-layer {TAU_PCTL:.0f}th percentile of per-token full-softmax "
                     "router entropy (bits) on dev-bucket prompts; accuracy never "
                     "inspected; dev bucket never evaluated",
        "dev_bucket": DEV_BUCKET,
        "dev_prompt_ids": ids,
        "tau_pctl": TAU_PCTL,
        "tau_by_layer": tau,
        "delta_small": _delta_small(),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CALIB.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: v for k, v in payload.items() if k != "tau_by_layer"}, indent=2))
    return payload


# ---------------------------------------------------------------------------
# One scored forward pass under the adaptive policy
# ---------------------------------------------------------------------------


def scored_forward(loaded, mc, text: str, candidate_ids, answer_id: int,
                   needle_span, question_span, affine_mask, pairs, delta,
                   tau_by_layer, sweep) -> dict:
    enc = loaded.tokenizer(text, return_tensors="pt")
    n_tokens = int(enc["input_ids"].shape[1])
    ctx = None
    if pairs is not None:
        experts_by_layer = {l: set(es) for l, es in by_layer(pairs).items()}
        ctx = AdaptiveEntropySteering(loaded.model, experts_by_layer, delta,
                                      {l: tau_by_layer[l] for l in experts_by_layer})
    try:
        with torch.no_grad():
            try:
                out = loaded.model(**enc, output_router_logits=True, logits_to_keep=1)
            except TypeError:
                out = loaded.model(**enc, output_router_logits=True)
        trigger_rate = float("nan")
        if ctx is not None:
            if any(c != 1 for c in ctx.call_counts.values()):
                raise RuntimeError("adaptive steering did not fire exactly once per layer")
            trigger_rate = ctx.trigger_rate()
        router_logits = list(out.router_logits)
        last = out.logits[0, -1, :].float()
    finally:
        if ctx is not None:
            ctx.remove()

    acc = sweep.score_answer(last, candidate_ids, answer_id)
    n0, n1 = needle_span
    q0, q1 = question_span
    draws_needle, draws_q = [], []
    for layer, lg in enumerate(router_logits):
        lg = lg.reshape(-1, N_EXPERTS)
        ids, _w, _m = route_from_logits(lg, TOP_K, loaded.shape.norm_topk_prob)
        gidx = layer * N_EXPERTS + ids.cpu().numpy()
        draws_needle.append(gidx[n0:n1])
        draws_q.append(gidx[q0:q1])
    return {
        **acc,
        "n_tokens": n_tokens,
        "trigger_rate": trigger_rate,
        "needle_affinity_rate": set_hit_rate(np.concatenate(draws_needle), affine_mask),
        "needle_affinity_rate_q": set_hit_rate(np.concatenate(draws_q), affine_mask),
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def done_keys() -> set[tuple[int, str]]:
    if not RECORDS.exists():
        return set()
    return {(json.loads(l)["prompt_id"], json.loads(l)["condition"])
            for l in RECORDS.read_text().splitlines() if l.strip()}


def run(args) -> None:
    mc = _mc()
    analyze, sweep = mc._ws1_modules()
    affine_mask, affine_pairs = needle_affine_pairs_local(analyze)
    print(f"needle-affine experts (this box's traces): {int(affine_mask.sum())}/{N_TOTAL}",
          flush=True)

    ps = yaml.safe_load(PROBE_HARD.read_text())
    prompts_by_id = {p["prompt_id"]: p for p in ps["prompts"]}
    stored = {p["prompt_id"]: p for p in json.loads(HARD_JSON.read_text())["per_prompt"]}

    print(f"loading {MODEL_ID} ...", flush=True)
    loaded = load_model(MODEL_ID, device="cpu", dtype="bfloat16")
    loaded.model.eval()

    candidate_ids = [loaded.tokenizer(" " + w, add_special_tokens=False)["input_ids"][0]
                     for w in ps["candidate_words"]]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise RuntimeError("candidate words collide under the tokenizer")

    if args.calibrate:
        calibrate(loaded, mc, prompts_by_id, args.n_dev)
        return

    if not CALIB.exists():
        raise SystemExit("run --calibrate first")
    cal = json.loads(CALIB.read_text())
    tau_by_layer = {int(k): float(v) for k, v in cal["tau_by_layer"].items()}
    delta = float(cal["delta_small"])
    print(f"delta={delta}, tau p{TAU_PCTL:.0f} range "
          f"[{min(tau_by_layer.values()):.2f}, {max(tau_by_layer.values()):.2f}] bits",
          flush=True)

    subsets = mc.pick_subsets(args.per_bucket)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    already = done_keys()
    conds = ["baseline", "adaptive_needle", "adaptive_random"]
    interleaved = [(name, r)
                   for pair in zip(subsets["low_affinity"], subsets["high_affinity"])
                   for name, r in zip(("low_affinity", "high_affinity"), pair)]
    jobs = [(subset, r["prompt_id"], cond) for subset, r in interleaved for cond in conds]
    todo = [j for j in jobs if (j[1], j[2]) not in already]
    print(f"{len(jobs)} cells; {len(already)} recorded; {len(todo)} to run", flush=True)
    if args.limit:
        todo = todo[: args.limit]

    t_run = time.time()
    for i, (subset, pid, cond) in enumerate(todo):
        p = prompts_by_id[pid]
        needle_span, question_span = mc._spans(p, loaded)
        if cond == "baseline":
            pairs, tags = None, []
        elif cond == "adaptive_needle":
            pairs = affine_pairs
            tags = sorted(f"L{l:02d}E{e:02d}" for l, e in pairs)
        else:
            pairs = mc.random_pairs_like(affine_pairs, seed=2000 + pid)
            tags = sorted(f"L{l:02d}E{e:02d}" for l, e in pairs)
        answer_id = loaded.tokenizer(" " + p["answer_word"],
                                     add_special_tokens=False)["input_ids"][0]
        t0 = time.time()
        res = scored_forward(loaded, mc, p["text"], candidate_ids, answer_id,
                             needle_span, question_span, affine_mask, pairs,
                             delta, tau_by_layer, sweep)
        rec = {
            "prompt_id": pid, "condition": cond, "subset": subset,
            "bucket": p["bucket"], "delta": None if pairs is None else delta,
            "n_boosted_experts": len(tags), "boosted_experts_sha": _sha(tags),
            "stored_answer_prob": stored[pid]["answer_prob"],
            "seconds": round(time.time() - t0, 1),
            **res,
        }
        if cond == "baseline":
            rec["repro_abs_dev"] = abs(res["forced_choice_prob"] - stored[pid]["answer_prob"])
        with RECORDS.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        el = (time.time() - t_run) / 60
        eta = el / (i + 1) * (len(todo) - i - 1)
        print(f"[{i+1}/{len(todo)}] pid={pid:>4} {subset:<13} {cond:<15} "
              f"p={res['forced_choice_prob']:.3f} trig={res['trigger_rate']:.3f} "
              f"aff={res['needle_affinity_rate']:.4f} {rec['seconds']:6.1f}s "
              f"ETA {eta:6.1f}m", flush=True)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _signflip_p(diffs: np.ndarray, n_perm: int, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    obs = abs(diffs.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, diffs.size))
    return float((np.sum(np.abs((signs * diffs).mean(axis=1)) >= obs) + 1) / (n_perm + 1))


def _bh_fdr(pvals: list[float], q: float) -> list[bool]:
    order = np.argsort(pvals)
    m = len(pvals)
    thresh = 0
    for rank, idx in enumerate(order, start=1):
        if pvals[idx] <= q * rank / m:
            thresh = rank
    return [rank <= thresh for rank in
            (int(np.where(order == i)[0][0]) + 1 for i in range(m))]


def analyze() -> None:
    recs = [json.loads(l) for l in RECORDS.read_text().splitlines() if l.strip()]
    cal = json.loads(CALIB.read_text())
    by_key = {(r["prompt_id"], r["condition"]): r for r in recs}
    subsets = sorted({r["subset"] for r in recs})
    conds = ["adaptive_needle", "adaptive_random"]

    lines = [
        "# Entropy-triggered adaptive boost: the remaining MECHANISM.md causal variant",
        "",
        "Follow-up to `docs/MECHANISM_CAUSAL.md` (fixed-window boost: FAILED at both",
        "magnitudes), testing the policy `docs/MECHANISM.md` actually proposed:",
        "boost the needle-affine experts only where router entropy spikes.",
        "",
        "## Caveats, before any result",
        "",
        "- Same scope limits as docs/MECHANISM_CAUSAL.md: one model, one seed, one",
        f"  task design, n={len({r['prompt_id'] for r in recs if r['subset'] == subsets[0]})}"
        " prompts per subset, teacher-forced forced choice.",
        f"- Trigger: per-layer p{cal['tau_pctl']:.0f} entropy threshold calibrated on",
        "  dev-bucket prompts (accuracy-blind, bucket never evaluated); magnitude is",
        f"  the SMALL delta ({cal['delta_small']}) carried over from the fixed test,",
        "  not re-tuned. Whole-sequence scope: the trigger decides where to act.",
        "- This steers routing, not compute; `weights_from='biased'` semantics only.",
        f"- Needle-affine set: {max(r['n_boosted_experts'] for r in recs)} experts,",
        "  re-derived from THIS box's regenerated hard traces by the same WS1",
        "  procedure. docs/MECHANISM.md's original run reported 75; BF16",
        "  cross-machine drift moves a few borderline experts across the lift",
        "  threshold. The set used matches the substrate the baselines below",
        "  reproduce exactly.",
        "",
    ]
    devs = [r.get("repro_abs_dev", 0.0) for r in recs if r["condition"] == "baseline"]
    lines += [f"Baseline reproduction: max |forced_choice_prob - stored| = {max(devs):.2e}", ""]

    lines += ["## Manipulation checks", "",
              "| subset | condition | trigger rate | needle_affinity_rate | delta vs baseline |",
              "|---|---|---|---|---|"]
    for sub in subsets:
        pids = sorted({r["prompt_id"] for r in recs if r["subset"] == sub})
        base = np.array([by_key[(pid, "baseline")]["needle_affinity_rate"] for pid in pids])
        lines.append(f"| {sub} | baseline | — | {base.mean():.4f} | — |")
        for cond in conds:
            aff = np.array([by_key[(pid, cond)]["needle_affinity_rate"] for pid in pids])
            trig = np.mean([by_key[(pid, cond)]["trigger_rate"] for pid in pids])
            lines.append(f"| {sub} | {cond} | {trig:.4f} | {aff.mean():.4f} "
                         f"| {(aff - base).mean():+.4f} |")

    lines += ["", "## Results: accuracy", "",
              "| subset | condition | answer_prob | delta | dz | perm p | FDR sig | "
              "passes effect size | accuracy |",
              "|---|---|---|---|---|---|---|---|---|"]
    pvals, cells = [], []
    for sub in subsets:
        pids = sorted({r["prompt_id"] for r in recs if r["subset"] == sub})
        base = np.array([by_key[(pid, "baseline")]["forced_choice_prob"] for pid in pids])
        bacc = np.mean([by_key[(pid, "baseline")]["forced_choice_correct"] for pid in pids])
        cells.append((sub, "baseline", base.mean(), None, None, None, bacc))
        for cond in conds:
            probs = np.array([by_key[(pid, cond)]["forced_choice_prob"] for pid in pids])
            acc = np.mean([by_key[(pid, cond)]["forced_choice_correct"] for pid in pids])
            d = probs - base
            dz = d.mean() / (d.std(ddof=1) + 1e-12)
            p = _signflip_p(d, N_SIGNFLIP)
            pvals.append(p)
            cells.append((sub, cond, probs.mean(), d.mean(), dz, p, acc))
    sig = _bh_fdr(pvals, Q_FDR)
    si = iter(sig)
    for sub, cond, mp, dm, dz, p, acc in cells:
        if dm is None:
            lines.append(f"| {sub} | baseline | {mp:.4f} | — | — | — | — | — | {acc:.3f} |")
        else:
            s = next(si)
            lines.append(f"| {sub} | {cond} | {mp:.4f} | {dm:+.4f} | {dz:+.2f} | {p:.4f} "
                         f"| {s} | {abs(dz) >= MIN_EFFECT_DZ} | {acc:.3f} |")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--n-dev", type=int, default=8)
    ap.add_argument("--per-bucket", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.analyze:
        analyze()
    else:
        run(args)


if __name__ == "__main__":
    main()
