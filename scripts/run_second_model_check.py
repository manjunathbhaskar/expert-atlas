"""Second-model generality check (PLAN.md §5 Phase 3, §9b; NOVELTY.md's
"single largest threat to every claim").

Runs the H6 (split-half replication) and H1 (per-expert topic affinity)
analyses on a second, architecturally-different MoE: IBM Granite-3.0-3B-A800M
(base), 32 MoE layers x 40 experts, top-8. This tests whether the atlas
method's headline findings are OLMoE-specific or a property of MoE routing
more broadly.

Why Granite and not PLAN.md's named candidates: Qwen1.5-MoE-A2.7B (14.3B
params, ~28.6 GB bf16) and DeepSeek-V2-Lite (15.7B, ~31 GB) do not fit this
box's 31 GB of RAM. Granite-3.0-3B-A800M is 3.3B params (~6.6 GB bf16),
fine-grained-expert (40/layer, top-8), and a base (non-instruct) model like
OLMoE-0924 -- the closest available substrate for a like-for-like check.

Capture notes (all verified against transformers source, not assumed):
  - GraniteMoe does NOT honor output_router_logits at inference; raw logits
    are captured via a forward hook on each `block_sparse_moe.router`
    (GraniteMoeTopKRouter), whose forward returns
    (top_k_index, top_k_weights, router_logits).
  - GraniteMoeTopKRouter does topk on raw logits THEN softmax over the
    selected top-k. Because softmax is strictly monotone, the selected
    expert ids equal Mixtral-convention softmax-all-then-topk, and
    softmax(top_k_logits) equals the renormalised full softmax restricted
    to the top-k -- i.e. exactly route_from_logits(norm_topk_prob=True).

Writes (new files only, never touching the OLMoE artifacts):
  - data/traces_granite/          (resumable parquet shards, same engine)
  - data/second_model_granite.json
  - docs/SECOND_MODEL.md

Usage:
    HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 \
        .venv/bin/python scripts/run_second_model_check.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from expertatlas.aggregate import FACTORS, aggregate_counts
from expertatlas.capture import LoadedModel, capture_to_dir
from expertatlas.schemas import ModelShape
from expertatlas.stats import bh_fdr, chi2_pvalues_fast, compute_lift, split_half_replication

REPO_ROOT = Path(__file__).parent.parent
MODEL_ID = "ibm-granite/granite-3.0-3b-a800m-base"
GATE_SUFFIX = ".block_sparse_moe.router"
PROBE_SET_PATH = REPO_ROOT / "probes" / "probe_set_v1.yaml"
OUT_DIR = REPO_ROOT / "data" / "traces_granite"
JSON_OUT = REPO_ROOT / "data" / "second_model_granite.json"
DOC_OUT = REPO_ROOT / "docs" / "SECOND_MODEL.md"

Q_FDR = 0.05
MEANINGFUL_LIFT = 1.0  # same practical-significance bar as the OLMoE run


class _RouterLogitsHook:
    """Forward hooks on every GraniteMoeTopKRouter, recording raw router
    logits (the THIRD element of its return tuple -- verified against the
    transformers source, see module docstring) in layer order."""

    def __init__(self, model):
        self.captured: list[torch.Tensor] = []
        self._handles = []
        for name, module in model.named_modules():
            if name.endswith(GATE_SUFFIX):
                self._handles.append(module.register_forward_hook(self._hook))

    def _hook(self, module, inputs, output):
        self.captured.append(output[2].detach())

    def remove(self):
        for h in self._handles:
            h.remove()


def load_granite(device: str = "cpu", dtype: str = "bfloat16") -> LoadedModel:
    """Local loader: capture.load_model() reads config.num_experts, which
    GraniteMoe calls num_local_experts, and its '.gate' hook fallback does
    not match Granite's '.router' modules -- so this script carries its own
    loader + hook rather than editing WS-A's capture.py (interface request
    filed in docs/interface-requests.md)."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(MODEL_ID)
    shape = ModelShape(
        n_layers=config.num_hidden_layers,
        n_experts=config.num_local_experts,
        top_k=config.num_experts_per_tok,
        # Granite's topk-then-softmax == renormalised softmax-all-then-topk
        norm_topk_prob=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=getattr(torch, dtype))
    model = model.to(device)
    model.eval()

    n_routers = sum(1 for name, _ in model.named_modules() if name.endswith(GATE_SUFFIX))
    if n_routers != shape.n_layers:
        raise RuntimeError(
            f"{MODEL_ID}: found {n_routers} '{GATE_SUFFIX}' modules but config says "
            f"{shape.n_layers} layers -- refusing to capture with silent layer skips"
        )

    loaded = LoadedModel(
        model=model,
        tokenizer=tokenizer,
        shape=shape,
        capture_method="forward_hook",
        model_id=MODEL_ID,
        model_revision="main",
    )

    # Prove the hook actually yields one (n_tokens, n_experts) logit tensor
    # per layer before committing to a long capture.
    hook = _RouterLogitsHook(model)
    try:
        probe = tokenizer("def foo():", return_tensors="pt").to(device)
        with torch.no_grad():
            model(**probe)
        got = hook.captured
    finally:
        hook.remove()
    if len(got) != shape.n_layers or got[0].shape[-1] != shape.n_experts:
        raise RuntimeError(
            f"router hook sanity check failed: {len(got)} tensors, "
            f"last dim {got[0].shape[-1] if got else 'n/a'} "
            f"(expected {shape.n_layers} x (*, {shape.n_experts}))"
        )
    return loaded


def _patch_capture_hook():
    """capture_to_dir's forward_hook path uses _GateHookCapture, which hooks
    '*.gate' modules and takes output[0]. Point the module-level name at our
    Granite-aware hook for the duration of this script (this process only --
    capture.py on disk is untouched)."""
    import expertatlas.capture as cap

    class _Adapter(_RouterLogitsHook):
        def __init__(self, model, suffix=None):
            super().__init__(model)

        def reset(self):
            self.captured = []

    cap._GateHookCapture = _Adapter


def analyze(rows: list[dict], prompts_by_id: dict, shape: ModelShape) -> dict:
    n_layers, n_experts = shape.n_layers, shape.n_experts
    n_total = n_layers * n_experts

    # H6 -- the gate, run first
    rows_a = [r for r in rows if prompts_by_id.get(r["prompt_id"], {}).get("split") == "A"]
    rows_b = [r for r in rows if prompts_by_id.get(r["prompt_id"], {}).get("split") == "B"]
    cm_a = aggregate_counts(rows_a, prompts_by_id, n_layers, n_experts, domain_factor="topic", seed=0)
    cm_b = aggregate_counts(rows_b, prompts_by_id, n_layers, n_experts, domain_factor="topic", seed=0)
    assert cm_a.domain_labels == cm_b.domain_labels
    pooled_rho, _ = split_half_replication(compute_lift(cm_a.counts), compute_lift(cm_b.counts))

    # per-factor lift + FDR + effect-size filter (same pipeline as OLMoE)
    factors = {}
    for f in FACTORS:
        cm = aggregate_counts(rows, prompts_by_id, n_layers, n_experts, domain_factor=f, seed=0)
        lift = compute_lift(cm.counts)
        sig = bh_fdr(chi2_pvalues_fast(cm.counts), q=Q_FDR)
        meaningful = sig & (np.abs(lift) >= MEANINGFUL_LIFT)
        factors[f] = {
            "n_cells": int(sig.size),
            "n_sig": int(sig.sum()),
            "n_meaningful": int(meaningful.sum()),
            "n_tokens_used": cm.n_tokens_used,
        }
        if f == "topic":
            per_expert = np.any(meaningful, axis=1)
            h1_n = int(per_expert.sum())

    return {
        "h6_pooled_rho": float(pooled_rho),
        "h6_pass": bool(pooled_rho >= 0.5),
        "h1_experts_with_hit": h1_n,
        "h1_n_experts_total": n_total,
        "h1_rate": h1_n / n_total,
        "h1_pass": bool(h1_n / n_total >= 0.05),
        "factors": factors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    probe_set = yaml.safe_load(PROBE_SET_PATH.read_text())
    prompts = [(p["prompt_id"], p["text"]) for p in probe_set["prompts"]]
    prompts_by_id = {p["prompt_id"]: p for p in probe_set["prompts"]}
    print(f"{len(prompts)} prompts from {PROBE_SET_PATH.name}")

    print(f"loading {MODEL_ID} ...")
    t0 = time.time()
    loaded = load_granite()
    print(f"loaded in {time.time() - t0:.1f}s. n_layers={loaded.shape.n_layers} "
          f"n_experts={loaded.shape.n_experts} top_k={loaded.shape.top_k}")

    _patch_capture_hook()
    t0 = time.time()
    result = capture_to_dir(
        loaded, prompts, out_dir=OUT_DIR, device="cpu", dtype="bfloat16",
        seed=args.seed, limit=args.limit, resume=True,
    )
    print(f"capture: {result} in {time.time() - t0:.1f}s")

    # load rows back (same as run_phase3_analyze.load_rows, pointed at granite dir)
    import pyarrow.parquet as pq
    rows = []
    for shard in sorted(OUT_DIR.glob("trace_*.parquet")):
        table = pq.read_table(shard)
        cols = {name: table.column(name).to_pylist() for name in table.column_names}
        rows.extend({k: cols[k][i] for k in cols} for i in range(len(cols["prompt_id"])))
    covered = {r["prompt_id"] for r in rows}
    prompts_by_id = {k: v for k, v in prompts_by_id.items() if k in covered}
    print(f"{len(rows)} rows, {len(prompts_by_id)} prompts")

    res = analyze(rows, prompts_by_id, loaded.shape)
    res["model_id"] = MODEL_ID
    res["n_rows"] = len(rows)
    res["n_prompts"] = len(prompts_by_id)
    JSON_OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

    write_doc(res, loaded.shape)
    print(f"wrote {DOC_OUT}")


def write_doc(res: dict, shape: ModelShape):
    olmoe = {"h6": 0.667, "h1_n": 557, "h1_total": 1024, "h1_rate": 0.544}
    f = res["factors"]
    DOC_OUT.write_text(f"""# Second-model generality check — Granite-3.0-3B-A800M

## Limitations first

- **One additional model, one architecture family.** This tests whether the
  OLMoE findings survive on a second fine-grained-expert MoE; it does not
  license claims about MoE models in general.
- **Not PLAN.md's named candidates.** Qwen1.5-MoE-A2.7B (~28.6 GB bf16) and
  DeepSeek-V2-Lite (~31 GB) exceed this machine's 31 GB RAM; Granite-3.0-3B-A800M
  (base, 3.3B params, {shape.n_layers} MoE layers x {shape.n_experts} experts, top-{shape.top_k})
  was the closest fitting substrate. Rerunning on a named candidate on a
  larger machine remains open.
- **Same probe set, different tokenizer.** probe_set_v1 was designed for
  OLMoE; token budgets per domain differ after Granite tokenization
  (equal-budget subsampling still applies, so base-rate correction holds).
- Router logits were captured via a forward hook on GraniteMoeTopKRouter
  (Granite ignores `output_router_logits`); the hook was sanity-checked to
  yield exactly one (n_tokens, {shape.n_experts}) tensor per layer. Granite's
  topk-then-softmax gating selects identical expert ids to the
  softmax-all-then-topk convention used in this repo (softmax is monotone),
  and its gate weights equal `route_from_logits(norm_topk_prob=True)`.

## Result

Same pipeline as the OLMoE run: base-rate-corrected lift, chi-squared +
BH-FDR (q={Q_FDR}), practical-significance bar |lift| >= {MEANINGFUL_LIFT}.

| metric | OLMoE-1B-7B-0924 | Granite-3.0-3B-A800M | verdict |
|---|---|---|---|
| H6 split-half pooled rho (gate, threshold 0.5) | {olmoe['h6']:.3f} | {res['h6_pooled_rho']:.3f} | {'PASS' if res['h6_pass'] else 'FAIL'} |
| H1 experts with >=1 meaningful topic affinity | {olmoe['h1_n']}/{olmoe['h1_total']} ({olmoe['h1_rate']:.1%}) | {res['h1_experts_with_hit']}/{res['h1_n_experts_total']} ({res['h1_rate']:.1%}) | {'PASS' if res['h1_pass'] else 'FAIL'} (falsified if <5%) |

Per-factor (FDR-significant cells vs. cells also clearing |lift| >= {MEANINGFUL_LIFT}):

| factor | cells | FDR-sig | sig AND meaningful |
|---|---|---|---|
""" + "\n".join(
        f"| {name} | {d['n_cells']} | {d['n_sig']} ({d['n_sig']/d['n_cells']:.1%}) | {d['n_meaningful']} ({d['n_meaningful']/d['n_cells']:.1%}) |"
        for name, d in f.items()
    ) + f"""

Run: {res['n_prompts']} prompts, {res['n_rows']} trace rows. Raw numbers in
`data/second_model_granite.json`; traces under `data/traces_granite/`.

## Interpretation

{'Both the replication gate and the per-expert affinity finding hold on a second, independently-trained, architecturally-distinct MoE. The atlas method and the specialization signal it measures are not OLMoE-specific artifacts -- within the limits above.' if res['h6_pass'] and res['h1_pass'] else 'The OLMoE findings did NOT fully replicate on Granite. Whatever fails here bounds every downstream claim: treat the OLMoE results as substrate-specific until the discrepancy is understood.'}
""")


if __name__ == "__main__":
    main()
