"""Multi-domain ablation sweep with a proper random-expert null.

This is the expensive half of the interference analysis. It is the generalisation of
`scripts/run_ablation_harness.py` (ONE domain pair, n=6 held-out prompts, ONE
random draw, no significance test) to:

  * 6 domains -> 30 ordered (ablator, victim) pairs, chosen to span the
    published overlap range in docs/ORTHOGONALITY.md (python/rust/sql/
    math_proof at cosine 0.74-0.90, history at -0.19 to -0.25 against every
    code domain, cooking in between). Without spread in the predictor the
    regression in run_interference.py would be meaningless.
  * all 24 held-out (`split=B`) prompts per domain, the probe set's entire
    held-out supply, versus the prior n=6.
  * a real null: many independent uniformly-random expert sets of exactly the
    same size, each scored on every domain. docs/ABLATION.md names this as
    the missing piece.

Design choices that differ from the single-pair harness, all deliberate:

  * **Expert sets come from split=A traces only.** The prior run drew them
    from data/atlas.json, whose lift was fitted on all 480 prompts including
    the split=B text it then scored -- leakage. Here the evaluation text is
    genuinely unseen by the selection step.
  * **Fixed set size m for every domain** (the prior run's sets were 189 and
    170). Size-matched by construction, so ONE random null per victim domain
    is valid for every ablator and no damage difference can come from having
    cut more of the network.
  * The whole design is domain-major, not pair-major: one ablation of domain
    a is scored on all six domains' text in the same sweep, so K ablations
    yield K*(K-1) ordered pairs instead of needing a run per pair.

The ablation mechanism itself is NOT re-implemented -- `ExpertAblator` is
imported from `scripts/run_ablation_harness.py`, whose hook placement was
verified against this exact model and transformers version (it zeroes
`router_scores` at the selected positions, because masking `router_logits`
in a forward hook on `OlmoeTopKRouter` is a silent no-op). Re-deriving that
would risk silently getting a no-op ablation.

Resumable: every completed sweep is appended to data/ablation_multi.jsonl and
skipped on restart. Kill and restart freely; run_interference.py report works
with however many null draws finished.

Usage:
    python scripts/run_interference.py precompute        # first
    python scripts/run_ablation_multi.py --n-null 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from expertatlas.interference import matched_load_null_sets
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from expertatlas.capture import load_model
from run_ablation_harness import ExpertAblator  # verified hook -- do not re-derive

REPO_ROOT = Path(__file__).parent.parent
PROBE_SET_PATH = REPO_ROOT / "probes" / "probe_set_v1.yaml"
PRECOMPUTE_PATH = REPO_ROOT / "data" / "interference_precompute.json"
OUT_PATH = REPO_ROOT / "data" / "ablation_multi.jsonl"

MODEL_ID = "allenai/OLMoE-1B-7B-0924"
N_EXPERTS_PER_LAYER = 64
N_LAYERS = 16


def load_split_b(domains: list[str]) -> dict[str, list[dict]]:
    """Every split=B prompt for each domain, in probe-set order.

    HONEST LIMIT, carried into docs/INTERFERENCE.md: the probe set gives each
    (topic, lang, register, format) cell exactly one split=A and one split=B
    prompt, and all 24 split=B prompts of a topic share the same `stem`. So
    n=24 is 24 SURFACE variants (4 languages x 2 registers x 3 formats) of ONE
    content stem. It is a real 4x increase over the prior n=6 in surface
    coverage and NOT a 4x increase in content coverage.
    """
    ps = yaml.safe_load(PROBE_SET_PATH.read_text())
    out = {}
    for d in domains:
        out[d] = [p for p in ps["prompts"] if p["topic"] == d and p.get("split") == "B"]
        if not out[d]:
            raise SystemExit(f"no split=B prompts for domain {d!r}")
    return out


def flat_to_layer_expert(idx: int) -> tuple[int, int]:
    return divmod(int(idx), N_EXPERTS_PER_LAYER)


@torch.no_grad()
def per_prompt_losses(
    model, tokenizer, texts: list[str], batch_size: int, max_length: int
) -> list[float]:
    """Mean per-token teacher-forced cross-entropy (nats) for each prompt.

    Batched because a single forward call on this machine costs ~50s of fixed
    weight-streaming overhead against ~0.027s per token position -- unbatched
    scoring would make the null distribution unaffordable. Right padding plus
    causal attention means padded positions cannot influence any real token,
    and padded positions are masked out of the loss; `--verify-padding`
    asserts that empirically rather than assuming it.
    """
    order = np.argsort([-len(t) for t in texts])  # group similar lengths together
    losses = [0.0] * len(texts)
    for start in range(0, len(order), batch_size):
        chunk = [int(i) for i in order[start:start + batch_size]]
        enc = tokenizer([texts[i] for i in chunk], return_tensors="pt",
                        padding=True, truncation=True, max_length=max_length)
        ids, am = enc["input_ids"], enc["attention_mask"]
        out = model(input_ids=ids, attention_mask=am)
        logits = out.logits[:, :-1, :]
        labels = ids[:, 1:]
        mask = am[:, 1:].to(torch.float32)
        for r, i in enumerate(chunk):
            ce = F.cross_entropy(logits[r].float(), labels[r], reduction="none")
            denom = float(mask[r].sum())
            losses[i] = float((ce * mask[r]).sum() / denom) if denom > 0 else float("nan")
        del out, logits
    return losses


def score_all(model, tokenizer, prompts, domains, batch_size, max_length) -> dict:
    return {d: per_prompt_losses(model, tokenizer,
                                 [p["text"] for p in prompts[d]], batch_size, max_length)
            for d in domains}


def completed_sweeps() -> set[str]:
    if not OUT_PATH.exists():
        return set()
    done = set()
    for line in OUT_PATH.read_text().splitlines():
        if line.strip():
            done.add(json.loads(line)["sweep"])
    return done


def append(rec: dict):
    with OUT_PATH.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


def verify_padding(model, tokenizer, prompts, domains, max_length, tolerance=1e-3):
    """Empirical check that batched+padded scoring equals unpadded scoring.

    Runs the same 3 prompts (a) in one batch padded to their natural max and
    (b) in one batch force-padded much longer. If padding leaked into the loss
    these would differ. Cheaper than scoring each prompt alone, and tests the
    thing that could actually be wrong.
    """
    texts = [prompts[domains[0]][0]["text"], prompts[domains[1]][0]["text"],
             prompts[domains[2]][0]["text"]]
    a = per_prompt_losses(model, tokenizer, texts, batch_size=3, max_length=max_length)
    old = tokenizer.model_max_length
    enc = tokenizer(texts, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=max_length)
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    logits = out.logits[:, :-1, :]
    labels = enc["input_ids"][:, 1:]
    mask = enc["attention_mask"][:, 1:].to(torch.float32)
    b = []
    for r in range(len(texts)):
        ce = F.cross_entropy(logits[r].float(), labels[r], reduction="none")
        b.append(float((ce * mask[r]).sum() / float(mask[r].sum())))
    tokenizer.model_max_length = old
    dev = max(abs(x - y) for x, y in zip(a, b))
    print(f"  padding invariance check: max |natural-pad - forced-pad| = {dev:.2e}")
    if dev > tolerance:
        raise SystemExit(
            f"padded batching changes per-prompt loss by {dev:.4f} nats (tolerance "
            f"{tolerance:g}) -- batched scoring is not trustworthy, do not use "
            "these numbers. BF16 matmul reduction order varies with batch/padding "
            "shape across hardware; if the deviation is of that magnitude, pass an "
            "explicit --padding-tolerance and report it in the writeup."
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-null", type=int, default=30,
                    help="random size-matched expert sets for the null distribution")
    ap.add_argument("--n-load-null", type=int, default=0,
                    help="PER-DOMAIN random sets matched on both size AND total load. "
                         "A size-matched null removes ~m fair-shares by construction, "
                         "but per-domain load removed spans 5.3x (WS3), so it cannot "
                         "separate 'which experts' from 'how much network'. Costs "
                         "n_domains x this many extra sweeps. 0 disables.")
    ap.add_argument("--load-null-tolerance", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--max-length", type=int, default=112)
    ap.add_argument("--verify-padding", action="store_true", default=True)
    ap.add_argument("--no-verify-padding", dest="verify_padding", action="store_false")
    ap.add_argument("--padding-tolerance", type=float, default=1e-3,
                    help="max allowed |natural-pad - forced-pad| loss deviation. "
                         "Raise ONLY for known BF16 hardware drift, and report "
                         "the measured deviation in the writeup.")
    args = ap.parse_args()

    if not PRECOMPUTE_PATH.exists():
        raise SystemExit("run `python scripts/run_interference.py precompute` first")
    pre = json.loads(PRECOMPUTE_PATH.read_text())
    domains, m = pre["domains"], pre["m"]
    sets = {d: {flat_to_layer_expert(i) for i in pre["expert_sets"][d]} for d in domains}
    for d in domains:
        assert len(sets[d]) == m, f"{d}: set size {len(sets[d])} != m {m}"

    prompts = load_split_b(domains)
    print(f"domains={domains}  m={m}  prompts/domain=" +
          str({d: len(prompts[d]) for d in domains}))

    rng = np.random.default_rng(args.seed)
    all_pairs = [(l, e) for l in range(N_LAYERS) for e in range(N_EXPERTS_PER_LAYER)]
    null_sets = []
    for _ in range(args.n_null):
        idx = rng.choice(len(all_pairs), size=m, replace=False)
        null_sets.append({all_pairs[i] for i in idx})

    plan = [("baseline", "baseline", None, set())]
    plan += [(f"target::{d}", "target", d, sets[d]) for d in domains]
    plan += [(f"null::{i:04d}", "null", i, s) for i, s in enumerate(null_sets)]

    # Matched-LOAD nulls: one distribution per domain, since each domain's set
    # removes a different amount of routed traffic (sql 192 vs history 36).
    load_null_diagnostics = {}
    if args.n_load_null > 0:
        util_p = REPO_ROOT / "data" / "utilization.json"
        if not util_p.exists():
            raise SystemExit("--n-load-null needs data/utilization.json "
                             "(run scripts/run_utilization.py)")
        lvec = np.asarray(json.loads(util_p.read_text())["utilization"]["load_ratio"],
                          dtype=np.float64)
        lrng = np.random.default_rng(args.seed + 1)
        for d in domains:
            flat = {l * N_EXPERTS_PER_LAYER + e for l, e in sets[d]}
            msets, diag = matched_load_null_sets(
                lvec, flat, args.n_load_null, lrng,
                tolerance=args.load_null_tolerance)
            load_null_diagnostics[d] = diag
            if not diag["feasible"] or diag["n_returned"] < args.n_load_null:
                print(f"  WARNING {d}: {diag.get('warning', 'underpowered')}")
            for i, ms in enumerate(msets):
                plan.append((f"loadnull::{d}::{i:04d}", "load_null", d,
                             {flat_to_layer_expert(x) for x in ms}))
        (REPO_ROOT / "data" / "load_null_diagnostics.json").write_text(
            json.dumps(load_null_diagnostics, indent=1))
        print(f"  matched-load nulls: {sum(x['n_returned'] for x in load_null_diagnostics.values())} "
              f"sets across {len(domains)} domains")

    done = completed_sweeps()
    todo = [p for p in plan if p[0] not in done]
    print(f"{len(done)} sweeps already done, {len(todo)} to run "
          f"({len(plan)} total: 1 baseline + {len(domains)} targets + {args.n_null} nulls)")
    if not todo:
        print("nothing to do")
        return

    print(f"loading {MODEL_ID} ...")
    t0 = time.time()
    loaded = load_model(MODEL_ID, device="cpu", dtype="bfloat16")
    model, tokenizer = loaded.model, loaded.tokenizer
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"  loaded in {time.time() - t0:.0f}s")

    if args.verify_padding:
        verify_padding(model, tokenizer, prompts, domains, args.max_length,
                       tolerance=args.padding_tolerance)

    load_ratio = None
    util_path = REPO_ROOT / "data" / "utilization.json"
    if util_path.exists():
        u = json.loads(util_path.read_text())
        load_ratio = np.asarray(u["utilization"]["load_ratio"], dtype=np.float64)

    for n_done, (sweep, kind, who, ablate) in enumerate(todo, 1):
        t1 = time.time()
        lr = (float(load_ratio[[l * N_EXPERTS_PER_LAYER + e for l, e in ablate]].sum())
              if load_ratio is not None and ablate else 0.0)
        with ExpertAblator(model, ablate):
            losses = score_all(model, tokenizer, prompts, domains,
                               args.batch_size, args.max_length)
        rec = {
            "sweep": sweep, "kind": kind,
            "ablator": who if kind == "target" else None,
            "null_index": who if kind == "null" else None,
            "load_null_domain": who if kind == "load_null" else None,
            "n_ablated": len(ablate),
            "load_removed": lr,
            "losses": losses,
            "seconds": round(time.time() - t1, 1),
            "batch_size": args.batch_size, "max_length": args.max_length,
            "expert_set": sorted(l * N_EXPERTS_PER_LAYER + e for l, e in ablate),
        }
        append(rec)
        means = {d: round(float(np.mean(losses[d])), 4) for d in domains}
        el = time.time() - t1
        print(f"[{n_done}/{len(todo)}] {sweep:16s} n={len(ablate):3d} load={lr:7.2f} "
              f"{el:6.1f}s  " + " ".join(f"{d[:4]}={means[d]:.4f}" for d in domains),
              flush=True)

    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
