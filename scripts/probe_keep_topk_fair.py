"""Keep-top-K probe -- FAIR version (restricts the router's candidate pool
BEFORE top-k selection, unlike probe_keep_topk.py's post-hoc zeroing).

probe_keep_topk.py showed loss exploding toward near-random at every keep
fraction tested, but flagged a real confound: its ExpertAblator hook zeroes
an expert's contribution AFTER the router's top-8 selection, so at high
ablation fractions most tokens waste most of their 8 chosen slots on
now-dead experts. That is not the same question as "what if the router
could only ever choose from a small kept set."

This version answers that question directly: for each targeted layer, it
monkey-patches OlmoeTopKRouter.forward (source verified against the
installed transformers==5.15.0 at
.venv/lib/python3.13/site-packages/transformers/models/olmoe/modeling_olmoe.py,
lines 328-346) to mask non-kept experts' logits to -inf BEFORE softmax+topk,
so all top_k selected slots are guaranteed to come from the kept set. The
patched forward is otherwise byte-for-byte identical to the original
(same softmax dtype upcast, same norm_topk_prob branch) -- only the mask
line is new.

Known, reported (not hidden) limitation: keep sets are chosen GLOBALLY by
lift rank, then split per layer. A layer can end up with fewer than top_k
(8) kept experts, in which case there aren't enough real candidates to fill
a normal top-8 selection. Policy: such layers are left COMPLETELY
UNRESTRICTED (skipped, not force-shrunk to a smaller top_k) and reported by
name -- shrinking top_k would introduce a second, uncontrolled confound
(less total contribution mass reaching the residual stream) on top of the
one this script exists to remove.

Usage:
    python scripts/probe_keep_topk_fair.py --domain medicine --control cooking --keep-frac 0.10 0.20 0.30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from expertatlas.capture import load_model

REPO_ROOT = Path(__file__).parent.parent
ATLAS_PATH = REPO_ROOT / "data" / "atlas.json"
UTIL_PATH = REPO_ROOT / "data" / "utilization.json"
PROBE_SET_PATH = REPO_ROOT / "probes" / "probe_set_v1.yaml"
OUT_PATH = REPO_ROOT / "docs" / "KEEP_TOPK_FAIR_PROBE.md"
MODEL_ID = "allenai/OLMoE-1B-7B-0924"
N_LAYERS = 16
N_EXPERTS_PER_LAYER = 64
HOT_THRESHOLD = 2.0  # matches UTILIZATION.md's own "hot" definition


class RestrictedGate:
    """Context manager: temporarily replaces each targeted layer's
    gate.forward with a version that masks non-allowed experts to -inf
    BEFORE softmax+topk, so every selected slot comes from the allowed set.

    Layers whose allowed set has fewer than gate.top_k experts are left
    unpatched (full open-book routing) and recorded in `skipped_layers`.
    """

    def __init__(self, model, keep_by_layer: dict[int, set[int]]):
        self.originals: dict[int, tuple] = {}
        self.skipped_layers: list[int] = []
        by_name = dict(model.named_modules())

        for layer, allowed in keep_by_layer.items():
            pattern = f"model.layers.{layer}.mlp.gate"
            if pattern not in by_name:
                raise RuntimeError(f"gate module '{pattern}' not found")
            gate = by_name[pattern]
            if len(allowed) < gate.top_k:
                self.skipped_layers.append(layer)
                continue
            self.originals[layer] = (gate, gate.forward)
            gate.forward = self._make_forward(gate, allowed)

    @staticmethod
    def _make_forward(gate, allowed: set[int]):
        allowed_idx = sorted(allowed)
        num_experts = gate.num_experts
        hidden_dim = gate.hidden_dim
        norm_topk_prob = gate.norm_topk_prob
        top_k = gate.top_k
        weight = gate.weight

        def forward(hidden_states):
            hidden_states = hidden_states.reshape(-1, hidden_dim)
            router_logits = F.linear(hidden_states, weight)  # (seq_len, num_experts)
            mask = torch.full((num_experts,), float("-inf"), dtype=router_logits.dtype)
            mask[allowed_idx] = 0.0
            router_logits = router_logits + mask  # broadcasts over seq_len; -inf outside allowed set
            router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
            router_top_value, router_indices = torch.topk(router_probs, top_k, dim=-1)
            if norm_topk_prob:
                router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
            router_top_value = router_top_value.to(router_logits.dtype)
            router_scores = router_top_value
            return router_logits, router_scores, router_indices

        return forward

    def remove(self):
        for layer, (gate, orig) in self.originals.items():
            gate.forward = orig
        self.originals = {}

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


def by_layer(pairs: set[tuple[int, int]]) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    for layer, idx in pairs:
        out.setdefault(layer, set()).add(idx)
    return out


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
    top_k = dict(model.named_modules())["model.layers.0.mlp.gate"].top_k

    n_total = N_LAYERS * N_EXPERTS_PER_LAYER
    results = []

    print("condition=baseline (fully open, no restriction) ...")
    base_t = mean_nll(loaded, target_prompts)
    base_c = mean_nll(loaded, control_prompts)
    print(f"  loss_on_target={base_t:.4f}  loss_on_control={base_c:.4f}")
    results.append({"keep_frac": None, "n_kept": n_total, "skipped_layers": 0,
                     "loss_on_target": base_t, "loss_on_control": base_c})

    for frac in args.keep_frac:
        n_top = int(round(frac * n_total))
        top_by_lift = {(l, i) for l, i, _ in ranked[:n_top]}
        keep_set = top_by_lift | hot_core
        keep_layers = by_layer(keep_set)
        # ensure every layer has an entry (possibly empty) so RestrictedGate sees it
        for l in range(N_LAYERS):
            keep_layers.setdefault(l, set())

        with RestrictedGate(model, keep_layers) as rg:
            n_restricted_layers = N_LAYERS - len(rg.skipped_layers)
            print(f"condition=keep_{frac:.0%} (top_by_lift={len(top_by_lift)} + hot_core={len(hot_core)} "
                  f"-> keep={len(keep_set)} [{len(keep_set)/n_total:.1%}] globally; "
                  f"{n_restricted_layers}/{N_LAYERS} layers actually restricted, "
                  f"skipped (< {top_k} kept experts) = {sorted(rg.skipped_layers)}) ...")
            loss_t = mean_nll(loaded, target_prompts)
            loss_c = mean_nll(loaded, control_prompts)
        print(f"  loss_on_target={loss_t:.4f} (baseline {base_t:.4f})  loss_on_control={loss_c:.4f}")
        results.append({
            "keep_frac": frac, "n_kept": len(keep_set),
            "skipped_layers": len(rg.skipped_layers),
            "loss_on_target": loss_t, "loss_on_control": loss_c,
        })

    write_report(args.domain, args.control, results, n_total)
    print(f"wrote {OUT_PATH}")


def write_report(domain, control, results, n_total):
    base_t = results[0]["loss_on_target"]
    base_c = results[0]["loss_on_control"]

    lines = [
        "# Keep-top-K probe -- FAIR version (pre-selection restriction)",
        "",
        f"Target domain: `{domain}`. Control domain: `{control}`. Metric: mean per-token "
        "teacher-forced cross-entropy (nats), held-out (split=B) prompts, forward passes only.",
        "",
        "Companion to `docs/KEEP_TOPK_PROBE.md` (the post-hoc-zeroing version). This one "
        "restricts each layer's router to choose its top-k **only from the kept set**, by "
        "masking non-kept experts' logits to -inf before softmax+topk -- verified against "
        "`OlmoeTopKRouter.forward` source directly (transformers==5.15.0), not guessed. Every "
        "selected slot in a restricted layer is therefore a real, non-zero expert -- the "
        "wasted-slots problem in the post-hoc version cannot happen here.",
        "",
        "**Reported limitation:** keep sets are chosen globally by lift rank, then split per "
        "layer -- a layer can end up with fewer than top_k kept experts. Such layers are left "
        "fully unrestricted rather than force-shrunk to a smaller top_k (shrinking would add a "
        "second, uncontrolled confound). Skipped-layer counts are reported per condition below; "
        "read the results next to that number, not in isolation.",
        "",
        "## Results",
        "",
        "| condition | n kept (global) | layers skipped (< top_k kept) | loss on target | delta vs baseline | loss on control | delta vs baseline |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        cond = "baseline" if r["keep_frac"] is None else f"keep_{r['keep_frac']:.0%}"
        dt = r["loss_on_target"] - base_t
        dc = r["loss_on_control"] - base_c
        lines.append(
            f"| {cond} | {r['n_kept']}/{n_total} | {r['skipped_layers']}/16 | "
            f"{r['loss_on_target']:.4f} | {dt:+.4f} | {r['loss_on_control']:.4f} | {dc:+.4f} |"
        )
    lines += [
        "",
        "## Reading this against the post-hoc version",
        "- If loss here is substantially better than the matching keep_frac row in "
        "`KEEP_TOPK_PROBE.md`, that confirms the post-hoc mechanism's own bluntness -- not "
        "the underlying idea -- was the main source of damage there.",
        "- If loss is still close to random even here, with few layers skipped, that's a much "
        "stronger negative result than the post-hoc version could ever produce, because the "
        "wasted-slots confound has been removed.",
        "- Watch the skipped-layers column: a condition where many layers had to be left "
        "unrestricted isn't really testing 'keep only K% everywhere' -- it's testing a mix of "
        "restricted and unrestricted layers, which is a real result but a different, weaker "
        "claim than the headline keep_frac number implies.",
        "- Same caveats as everywhere else in this project: one domain pair, one model, one "
        "seed, small held-out n -- directional only.",
    ]
    OUT_PATH.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
