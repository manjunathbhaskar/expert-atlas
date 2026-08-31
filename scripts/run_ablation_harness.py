"""Ablation harness (Tier 3, TRANSFER.md §6.6) -- the causal test.

Everything so far (H1, orthogonality) is CORRELATIONAL: expert X's routing
correlates with domain D. This tests CAUSATION: does removing X's
contribution selectively hurt the model's predictions on D, and NOT on
other domains?

Method: register forward hooks on individual expert FFN submodules
(`model.layers[L].mlp.experts[E]`, verified against the loaded OLMoE
model's actual module names before use -- see docs/ABLATION.md for what
was checked) that zero their output tensor. This removes exactly that
expert's contribution to the token's hidden state regardless of its gate
weight -- the standard "zero ablation" technique -- without touching
routing itself (the router still selects it, its contribution is just
dropped, which is the correct causal manipulation: "what if this expert
didn't exist" rather than "what if the router never picked it").

Metric: mean per-token cross-entropy loss (teacher-forced NLL) on held-out
('B' split) prompts. Forward passes only -- no training, no gradients.

Conditions compared, evaluated on BOTH target-domain and control-domain
held-out text, so the comparison is about SELECTIVITY not just magnitude:
  - baseline:            no ablation
  - ablate_target:       experts identified as significant+meaningful for
                          the target domain (from data/atlas.json, i.e. the
                          real Phase 3 run -- not recomputed here, avoiding
                          a second circular pass over the same data)
  - ablate_random:       same COUNT of experts, chosen uniformly at random
                          (fixed seed) -- controls for "ablating anything
                          hurts everything a bit"
  - ablate_other_domain: experts significant+meaningful for a DIFFERENT
                          domain, same-ish count -- controls for
                          "ablating any topically-loaded experts hurts"

The causal claim (X is selectively load-bearing for D) is supported only
if: loss(ablate_target, target_text) - loss(baseline, target_text) is
clearly larger than the SAME delta computed for ablate_random and
ablate_other_domain, AND clearly larger than
loss(ablate_target, control_text) - loss(baseline, control_text).
Report all four numbers -- do not report only the one that looks good.

Usage:
    python scripts/run_ablation_harness.py --target medicine --control cooking [--n-prompts 15]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from expertatlas.capture import load_model

REPO_ROOT = Path(__file__).parent.parent
PROBE_SET_PATH = REPO_ROOT / "probes" / "probe_set_v1.yaml"
ATLAS_PATH = REPO_ROOT / "data" / "atlas.json"
OUT_PATH = REPO_ROOT / "docs" / "ABLATION.md"

MEANINGFUL_LIFT = 1.0  # same bar as Phase 3 analysis -- >=2x fold, not just FDR-significant
MODEL_ID = "allenai/OLMoE-1B-7B-0924"


class ExpertAblator:
    """Suppresses specific (layer, expert_idx) pairs by hooking each
    layer's router (`model.layers.{L}.mlp.gate`, an `OlmoeTopKRouter`) and
    zeroing the ablated experts' SCORE in its already-selected top-k, not
    by masking logits pre-selection.

    Why this specific hook, verified against this model/transformers
    version before trusting it (do not port this assumption elsewhere
    without re-checking):
      - OLMoE has no per-expert FFN submodule to hook -- `.mlp.experts` is
        one batched module, not a ModuleList (checked via named_modules()).
      - `OlmoeTopKRouter.forward()` does softmax + top-k INTERNALLY and
        returns `(router_logits, router_scores, router_indices)` -- the
        selection already happened by the time a forward hook fires.
        Masking `router_logits` in a hook would be a SILENT NO-OP: nothing
        downstream re-reads it. Checked via reading
        OlmoeSparseMoeBlock.forward() source directly, which unpacks
        `_, top_k_weights, top_k_index = self.gate(...)` -- it uses
        `router_scores`/`router_indices` (weights and indices), not the
        logits.
      - So the hook instead zeros `router_scores` at the positions where
        `router_indices` matches an ablated expert id. Each expert's FFN
        output is weighted by its score before summing
        (`self.experts(hidden_states, top_k_index, top_k_weights)`), so a
        zeroed score makes that expert's contribution to the token's
        hidden state exactly zero -- true ablation, not just re-routing.

    Layer-scoped: only touches gate modules for layers that actually have
    >=1 ablated expert.
    """

    def __init__(self, model, ablate_set: set[tuple[int, int]]):
        self.handles = []
        by_layer: dict[int, set[int]] = {}
        for layer, idx in ablate_set:
            by_layer.setdefault(layer, set()).add(idx)

        by_name = dict(model.named_modules())
        for layer, idxs in by_layer.items():
            pattern = f"model.layers.{layer}.mlp.gate"
            if pattern not in by_name:
                raise RuntimeError(
                    f"gate module '{pattern}' not found -- module path assumption is "
                    "wrong for this model/transformers version, check named_modules() "
                    "before trusting any result from this run"
                )
            hook = self._make_hook(idxs)
            self.handles.append(by_name[pattern].register_forward_hook(hook))

        if ablate_set and not self.handles:
            raise RuntimeError(
                f"asked to ablate {len(ablate_set)} experts but registered 0 hooks -- "
                "do not trust a silent no-op ablation"
            )

    @staticmethod
    def _make_hook(idxs: set[int]):
        idx_set = set(idxs)

        def hook(module, inputs, output):
            router_logits, router_scores, router_indices = output
            router_scores = router_scores.clone()
            # router_indices: (n_tokens, top_k) expert ids actually selected.
            # Zero the score wherever a selected slot is an ablated expert --
            # this is the actual weight multiplied into that expert's FFN
            # output downstream, so zeroing it == zero contribution.
            mask = torch.zeros_like(router_indices, dtype=torch.bool)
            for e in idx_set:
                mask |= router_indices == e
            router_scores = torch.where(mask, torch.zeros_like(router_scores), router_scores)
            return router_logits, router_scores, router_indices

        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.remove()


def mean_nll(loaded, prompts: list[str], device: str = "cpu") -> float:
    """Mean per-token teacher-forced cross-entropy (nats) across prompts."""
    losses = []
    for text in prompts:
        inputs = loaded.tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        if input_ids.shape[1] < 2:
            continue
        with torch.no_grad():
            out = loaded.model(**inputs, labels=input_ids)
        losses.append(float(out.loss))
    if not losses:
        raise RuntimeError("no prompts long enough to score")
    return float(np.mean(losses))


def load_expert_set_for_domain(domain: str) -> set[tuple[int, int]]:
    """Pull already-computed, meaningful (FDR-sig AND |lift|>=1.0) experts
    for a domain straight from the real atlas.json -- avoids re-deriving
    significance in a second, potentially-inconsistent pass."""
    atlas = json.loads(ATLAS_PATH.read_text())
    out = set()
    for e in atlas["experts"]:
        lift = e.get("lift", {}).get(domain)
        if domain in e.get("significant", []) and lift is not None and abs(lift) >= MEANINGFUL_LIFT:
            out.add((e["layer"], e["idx"]))
    return out


def load_split_b_prompts(domain: str, n: int) -> list[str]:
    probe_set = yaml.safe_load(PROBE_SET_PATH.read_text())
    matched = [p["text"] for p in probe_set["prompts"] if p["topic"] == domain and p.get("split") == "B"]
    return matched[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="target domain, e.g. medicine")
    parser.add_argument("--control", required=True, help="control domain, e.g. cooking")
    parser.add_argument("--n-prompts", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not ATLAS_PATH.exists():
        raise SystemExit(f"{ATLAS_PATH} not found -- run scripts/run_phase3_analyze.py first")

    target_experts = load_expert_set_for_domain(args.target)
    control_domain_experts = load_expert_set_for_domain(args.control)
    print(f"target ({args.target}) meaningful experts: {len(target_experts)}")
    print(f"control-domain ({args.control}) meaningful experts: {len(control_domain_experts)}")
    if not target_experts:
        raise SystemExit(f"no meaningful experts found for target domain {args.target!r} in atlas.json")

    rng = np.random.default_rng(args.seed)
    all_pairs = [(l, e) for l in range(16) for e in range(64)]
    random_idx = rng.choice(len(all_pairs), size=len(target_experts), replace=False)
    random_experts = {all_pairs[i] for i in random_idx}

    target_prompts = load_split_b_prompts(args.target, args.n_prompts)
    control_prompts = load_split_b_prompts(args.control, args.n_prompts)
    print(f"target prompts: {len(target_prompts)}, control prompts: {len(control_prompts)}")
    if not target_prompts or not control_prompts:
        raise SystemExit("not enough held-out (split=B) prompts for target/control domain")

    print(f"loading {MODEL_ID} ...")
    loaded = load_model(MODEL_ID, device="cpu", dtype="bfloat16")
    model = loaded.model
    model.eval()

    conditions = {
        "baseline": set(),
        "ablate_target": target_experts,
        "ablate_random": random_experts,
        "ablate_other_domain": control_domain_experts,
    }

    results = {}
    for name, ablate_set in conditions.items():
        print(f"condition={name} (n_ablated={len(ablate_set)}) ...")
        with ExpertAblator(model, ablate_set):
            loss_on_target = mean_nll(loaded, target_prompts)
            loss_on_control = mean_nll(loaded, control_prompts)
        results[name] = {"loss_on_target": loss_on_target, "loss_on_control": loss_on_control}
        print(f"  loss_on_target={loss_on_target:.4f}  loss_on_control={loss_on_control:.4f}")

    write_report(args.target, args.control, results, len(target_experts), len(control_domain_experts))
    print(f"wrote {OUT_PATH}")


def write_report(target, control, results, n_target_experts, n_control_experts):
    base_t = results["baseline"]["loss_on_target"]
    base_c = results["baseline"]["loss_on_control"]

    def delta(cond, which):
        return results[cond][which] - (base_t if which == "loss_on_target" else base_c)

    lines = [
        "# Ablation harness -- causal test (Tier 3, TRANSFER.md §6.6)",
        "",
        f"Target domain: `{target}` ({n_target_experts} experts ablated). "
        f"Control domain: `{control}` ({n_control_experts} experts ablated in the "
        "ablate_other_domain condition). Metric: mean per-token teacher-forced "
        "cross-entropy (nats), held-out (split=B) prompts, forward passes only.",
        "",
        "**The causal claim requires ALL of:** ablate_target hurts target text more than "
        "it hurts control text, AND more than ablate_random hurts target text, AND more "
        "than ablate_other_domain hurts target text. Reporting one flattering number alone "
        "would not support a causal claim -- all four deltas below are load-bearing.",
        "",
        "## Results",
        "",
        "| condition | loss on target text | delta vs baseline | loss on control text | delta vs baseline |",
        "|---|---|---|---|---|",
    ]
    for cond in ("baseline", "ablate_target", "ablate_random", "ablate_other_domain"):
        r = results[cond]
        dt = delta(cond, "loss_on_target")
        dc = delta(cond, "loss_on_control")
        lines.append(f"| {cond} | {r['loss_on_target']:.4f} | {dt:+.4f} | {r['loss_on_control']:.4f} | {dc:+.4f} |")

    dt_target = delta("ablate_target", "loss_on_target")
    dc_target = delta("ablate_target", "loss_on_control")
    dt_random = delta("ablate_random", "loss_on_target")
    dt_other = delta("ablate_other_domain", "loss_on_target")

    selective = dt_target > dc_target and dt_target > dt_random and dt_target > dt_other
    lines += [
        "",
        "## Verdict",
        f"- ablate_target hurts target text more than it hurts control text: "
        f"{dt_target:.4f} vs {dc_target:.4f} -- {'YES' if dt_target > dc_target else 'NO'}",
        f"- ablate_target hurts target text more than ablate_random does: "
        f"{dt_target:.4f} vs {dt_random:.4f} -- {'YES' if dt_target > dt_random else 'NO'}",
        f"- ablate_target hurts target text more than ablate_other_domain does: "
        f"{dt_target:.4f} vs {dt_other:.4f} -- {'YES' if dt_target > dt_other else 'NO'}",
        "",
        f"**{'CAUSAL claim supported' if selective else 'CAUSAL claim NOT supported'} on this "
        f"single run.** No significance test is applied here (no null distribution over "
        f"ablation sets was built -- this is a first pass, not a finished statistical test); "
        "treat this as directional evidence, not a p-value-backed claim. A proper version "
        "would repeat ablate_random over many random sets to build a null and report where "
        "ablate_target falls in that distribution, and would run on more than one target domain.",
    ]
    OUT_PATH.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
