"""Keep-top-K probe -- quick first pass on "how much of the model can we
drop and still work" (Tier 3 extension, not a finished result).

Inverts run_ablation_harness.py's mechanism: instead of ablating a small
target set to show it's load-bearing, this KEEPS only a small set --
top-N% experts by measured lift for a domain (data/atlas.json), unioned
with the generalist/hot core (data/utilization.json) -- and ablates
everything else. Reuses the exact same proven ExpertAblator hook.

Known limitation, stated up front: ablation happens AFTER the router's
top-8 selection (zeroing the score of an already-chosen expert), not by
restricting the candidate pool BEFORE selection. When most of the model is
ablated, many tokens will have most of their 8 selected slots land on
now-zeroed experts -- a harsher test than "let the router choose only
among the kept experts" would be. Treat this as a pessimistic first pass.

Usage:
    python scripts/probe_keep_topk.py --domain medicine --control cooking --keep-frac 0.10 0.20 0.30
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
ATLAS_PATH = REPO_ROOT / "data" / "atlas.json"
UTIL_PATH = REPO_ROOT / "data" / "utilization.json"
PROBE_SET_PATH = REPO_ROOT / "probes" / "probe_set_v1.yaml"
OUT_PATH = REPO_ROOT / "docs" / "KEEP_TOPK_PROBE.md"
MODEL_ID = "allenai/OLMoE-1B-7B-0924"
N_LAYERS = 16
N_EXPERTS_PER_LAYER = 64
HOT_THRESHOLD = 2.0  # matches UTILIZATION.md's own "hot" definition


class ExpertAblator:
    """Verbatim copy of the proven hook from run_ablation_harness.py --
    zeroes router_scores for ablated (layer, expert) pairs post-selection.
    See that file's docstring for why this hook point (not router_logits)
    is the correct one for this model/transformers version."""

    def __init__(self, model, ablate_set: set[tuple[int, int]]):
        self.handles = []
        by_layer: dict[int, set[int]] = {}
        for layer, idx in ablate_set:
            by_layer.setdefault(layer, set()).add(idx)

        by_name = dict(model.named_modules())
        for layer, idxs in by_layer.items():
            pattern = f"model.layers.{layer}.mlp.gate"
            if pattern not in by_name:
                raise RuntimeError(f"gate module '{pattern}' not found")
            hook = self._make_hook(idxs)
            self.handles.append(by_name[pattern].register_forward_hook(hook))

        if ablate_set and not self.handles:
            raise RuntimeError(
                f"asked to ablate {len(ablate_set)} experts but registered 0 hooks"
            )

    @staticmethod
    def _make_hook(idxs: set[int]):
        idx_set = set(idxs)

        def hook(module, inputs, output):
            router_logits, router_scores, router_indices = output
            router_scores = router_scores.clone()
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


def load_split_b_prompts(domain: str, n: int) -> list[str]:
    probe_set = yaml.safe_load(PROBE_SET_PATH.read_text())
    matched = [p["text"] for p in probe_set["prompts"] if p["topic"] == domain and p.get("split") == "B"]
    return matched[:n]


def load_lift_ranked(domain: str) -> list[tuple[int, int, float]]:
    atlas = json.loads(ATLAS_PATH.read_text())
    out = []
    for e in atlas["experts"]:
        lift = e.get("lift", {}).get(domain)
        if lift is not None:
            out.append((e["layer"], e["idx"], lift))
    out.sort(key=lambda x: x[2], reverse=True)
    return out


def load_hot_core() -> set[tuple[int, int]]:
    util = json.loads(UTIL_PATH.read_text())
    load_ratio = util["utilization"]["load_ratio"]
    hot = set()
    for i, lr in enumerate(load_ratio):
        if lr >= HOT_THRESHOLD:
            layer, idx = divmod(i, N_EXPERTS_PER_LAYER)
            hot.add((layer, idx))
    return hot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="medicine")
    parser.add_argument("--control", default="cooking")
    parser.add_argument("--keep-frac", type=float, nargs="+", default=[0.10, 0.20, 0.30])
    parser.add_argument("--n-prompts", type=int, default=15)
    args = parser.parse_args()

    if not ATLAS_PATH.exists() or not UTIL_PATH.exists():
        raise SystemExit("atlas.json / utilization.json not found -- run Phase 3 analysis first")

    ranked = load_lift_ranked(args.domain)
    hot_core = load_hot_core()
    print(f"hot/generalist core: {len(hot_core)} experts (load_ratio >= {HOT_THRESHOLD}x fair share)")

    target_prompts = load_split_b_prompts(args.domain, args.n_prompts)
    control_prompts = load_split_b_prompts(args.control, args.n_prompts)
    print(f"target prompts: {len(target_prompts)}, control prompts: {len(control_prompts)}")
    if not target_prompts or not control_prompts:
        raise SystemExit("not enough held-out (split=B) prompts for target/control domain")

    print(f"loading {MODEL_ID} ...")
    loaded = load_model(MODEL_ID, device="cpu", dtype="bfloat16")
    model = loaded.model
    model.eval()

    all_pairs = {(l, e) for l in range(N_LAYERS) for e in range(N_EXPERTS_PER_LAYER)}

    results = []

    print("condition=baseline ...")
    with ExpertAblator(model, set()):
        base_t = mean_nll(loaded, target_prompts)
        base_c = mean_nll(loaded, control_prompts)
    print(f"  loss_on_target={base_t:.4f}  loss_on_control={base_c:.4f}")
    results.append({"keep_frac": None, "n_kept": len(all_pairs), "loss_on_target": base_t, "loss_on_control": base_c})

    for frac in args.keep_frac:
        n_top = int(round(frac * len(all_pairs)))
        top_by_lift = {(l, i) for l, i, _ in ranked[:n_top]}
        keep_set = top_by_lift | hot_core
        ablate_set = all_pairs - keep_set
        print(f"condition=keep_{frac:.0%} (top_by_lift={len(top_by_lift)} + hot_core={len(hot_core)} "
              f"-> keep={len(keep_set)} [{len(keep_set)/len(all_pairs):.1%}], ablate={len(ablate_set)}) ...")
        with ExpertAblator(model, ablate_set):
            loss_t = mean_nll(loaded, target_prompts)
            loss_c = mean_nll(loaded, control_prompts)
        print(f"  loss_on_target={loss_t:.4f} (baseline {base_t:.4f})  loss_on_control={loss_c:.4f}")
        results.append({
            "keep_frac": frac, "n_kept": len(keep_set),
            "loss_on_target": loss_t, "loss_on_control": loss_c,
        })

    write_report(args.domain, args.control, results, len(all_pairs))
    print(f"wrote {OUT_PATH}")


def write_report(domain, control, results, n_total):
    base_t = results[0]["loss_on_target"]
    base_c = results[0]["loss_on_control"]

    lines = [
        "# Keep-top-K probe -- quick first pass, NOT a finished result",
        "",
        f"Target domain: `{domain}`. Control domain: `{control}`. Metric: mean per-token "
        "teacher-forced cross-entropy (nats), held-out (split=B) prompts, forward passes only.",
        "",
        "**Method caveat, stated up front:** this reuses the `ExpertAblator` hook from "
        "`run_ablation_harness.py`, which zeroes an expert's contribution AFTER the router's "
        "top-8 selection, not by restricting the candidate pool BEFORE selection. When most of "
        "the model is ablated, many tokens will have most of their 8 selected slots land on "
        "now-zeroed experts -- a harsher test than 'let the router choose only among the kept "
        "experts' would be. Read this as a pessimistic first pass, not the final answer on "
        "whether keep-top-K is viable.",
        "",
        "## Results",
        "",
        "| condition | n kept | loss on target | delta vs baseline | loss on control | delta vs baseline |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        cond = "baseline" if r["keep_frac"] is None else f"keep_{r['keep_frac']:.0%}"
        dt = r["loss_on_target"] - base_t
        dc = r["loss_on_control"] - base_c
        lines.append(
            f"| {cond} | {r['n_kept']}/{n_total} | {r['loss_on_target']:.4f} | {dt:+.4f} | "
            f"{r['loss_on_control']:.4f} | {dc:+.4f} |"
        )
    lines += [
        "",
        "## Reading this",
        "- If loss on target stays close to baseline even at a low keep_frac, that's a real "
        "signal the idea has legs, even under this pessimistic mechanism.",
        "- If loss explodes even at a high keep_frac, that plausibly reflects the post-hoc-zeroing "
        "mechanism wasting slots rather than proving the idea is dead -- the next step would be "
        "the fairer pre-selection-restricted version (mask logits to -inf for non-kept experts "
        "before top-k, so all 8 selected slots come from the kept set) before drawing a real "
        "conclusion either way.",
        "- n is small (held-out split=B prompts only), one domain pair, one model, one seed -- "
        "directional only, same caveats as docs/ABLATION.md.",
    ]
    OUT_PATH.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
