"""Expert steering: intervene on a MoE router's logits BEFORE top-k selection.

This is the shared primitive behind two different experiments:

  * **restriction** ("the router may only choose from this set") — generalised
    from `RestrictedGate` in `scripts/probe_keep_topk_fair.py`, which was
    verified line-by-line against `OlmoeTopKRouter.forward`
    (transformers==5.15.0, `models/olmoe/modeling_olmoe.py`), and
  * **boosting** ("make these experts more likely to be chosen, by `delta`
    logits") — new here, needed for the causal follow-up named at the end of
    `docs/MECHANISM.md`.

Both are the same operation: an **additive bias on `router_logits`, applied
before `softmax` + `topk`**. Masking is a bias of `-inf` outside the allowed
set; boosting is a bias of `+delta` on the named set. Nothing downstream of the
selection is touched.

Why before selection, not after
-------------------------------
`ExpertAblator` (`scripts/run_ablation_harness.py`) and
`scripts/probe_keep_topk.py` zero an expert's *contribution* after the router
has already spent one of its top-k slots on it. `docs/KEEP_TOPK_PROBE.md`
flagged that as a confound (wasted slots) and `docs/KEEP_TOPK_FAIR_PROBE.md`
measured how much it mattered. This module only ever intervenes on the logits,
so every selected slot in an intervened layer is a real, live expert.

Three things this module does that the original `RestrictedGate` did not
-----------------------------------------------------------------------
1. **Boost as well as mask**, behind one interface (both are additive biases).
2. **Token-window scoping.** A bias can be restricted to a half-open token
   span `[start, end)` of the sequence, so an intervention can be applied
   *only at the needle's tokens* and nowhere else. This requires knowing the
   flat sequence length up front (`seq_len=`), and the patched forward
   **raises** if the tensor it receives disagrees — a silently mis-scoped
   window would look exactly like a real effect. Scoped interventions
   therefore only work for single-sequence (batch=1) full-sequence forward
   passes, which is what every capture in this repo does; they are not safe
   under cached incremental decoding, and they refuse rather than guess.
3. **`weights_from="original"`** (optional). A boost changes two things at
   once: *which* experts are selected, and *what gate weight* they get
   (the biased softmax puts more mass on them). Those are different
   interventions with different mechanisms. With `weights_from="original"`
   the selection is made from the biased logits but the gate weights are read
   off the **unbiased** softmax at the selected indices, isolating the
   selection change. Default is `"biased"` (the plain, obvious mechanism);
   whichever is used must be stated in the report, because they are not the
   same experiment.

Honest limitations (state these wherever results from this module are reported)
------------------------------------------------------------------------------
* This zeroes/raises *routing*, not compute. Nothing here makes the model
  faster or smaller — non-selected experts are simply never chosen. Real
  conditional compute is a separate piece of work (handoff Task 5).
* `weights_from="biased"` (the default) confounds selection change with gate-
  weight change, as described above. This is a property of the mechanism, not
  a bug, but it means "boosting helped" does not by itself say *which* of the
  two did the work. Run both if the distinction matters.
* The `router_logits` returned by a patched gate are the **biased** logits
  (the ones selection was actually made from). Anything that consumes them —
  the MoE load-balancing aux loss, `output_router_logits=True` capture — sees
  the intervened values. Router entropy measured during an intervention is
  therefore entropy of the intervened distribution, not of the model's own.
* Masking below `top_k` allowed experts in a layer is not representable
  without also shrinking `top_k` (a second, uncontrolled change in how much
  contribution mass reaches the residual stream). Such layers are **skipped
  entirely** (left fully unrestricted) and reported in `skipped_layers`, the
  same policy and the same reporting requirement as
  `docs/KEEP_TOPK_FAIR_PROBE.md`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

GATE_PATTERN = "model.layers.{layer}.mlp.gate"


# ---------------------------------------------------------------------------
# Intervention spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Intervention:
    """One additive-logit-bias intervention on one layer's router.

    kind:
        "mask"  — only `experts` may be selected (bias -inf elsewhere).
        "boost" — `experts` get `+delta` logits; everything else untouched.
    token_span:
        None applies the intervention at every token. `(start, end)` applies it
        only at flat token positions in the half-open interval, and requires
        `ExpertSteering(..., seq_len=...)`.
    """

    kind: str
    experts: frozenset[int]
    delta: float | None = None
    token_span: tuple[int, int] | None = None

    def __post_init__(self):
        if self.kind not in ("mask", "boost"):
            raise ValueError(f"unknown intervention kind {self.kind!r}")
        if self.kind == "boost" and self.delta is None:
            raise ValueError("boost requires delta")
        if not self.experts:
            raise ValueError(f"{self.kind} intervention has an empty expert set")
        if self.token_span is not None:
            s, e = self.token_span
            if not (0 <= s < e):
                raise ValueError(f"bad token_span {self.token_span}")


def mask_to(allowed, token_span: tuple[int, int] | None = None) -> Intervention:
    """Restrict a layer's router to `allowed` (per-layer expert indices)."""
    return Intervention("mask", frozenset(int(x) for x in allowed), None, token_span)


def boost(experts, delta: float, token_span: tuple[int, int] | None = None) -> Intervention:
    """Add `delta` to the router logits of `experts` (per-layer indices)."""
    return Intervention("boost", frozenset(int(x) for x in experts), float(delta), token_span)


def by_layer(pairs) -> dict[int, set[int]]:
    """`{(layer, idx), ...}` -> `{layer: {idx, ...}}` (same helper shape as the
    keep-top-K probes, so a global expert set can be fed straight in)."""
    out: dict[int, set[int]] = {}
    for layer, idx in pairs:
        out.setdefault(int(layer), set()).add(int(idx))
    return out


def boost_global(pairs, delta: float, token_span=None) -> dict[int, list[Intervention]]:
    """Build a per-layer boost plan from a global `{(layer, idx)}` set."""
    return {l: [boost(idxs, delta, token_span)] for l, idxs in by_layer(pairs).items()}


def mask_to_global(pairs, token_span=None) -> dict[int, list[Intervention]]:
    """Build a per-layer restriction plan from a global `{(layer, idx)}` set."""
    return {l: [mask_to(idxs, token_span)] for l, idxs in by_layer(pairs).items()}


# ---------------------------------------------------------------------------
# The steering context manager
# ---------------------------------------------------------------------------


class ExpertSteering:
    """Context manager applying router-logit interventions to chosen layers.

    Args:
        model: an OLMoE-style model whose gates live at
            `model.layers.{i}.mlp.gate` and whose gate exposes
            `top_k / num_experts / hidden_dim / norm_topk_prob / weight`.
        plan: `{layer: Intervention | [Intervention, ...]}`. Layers absent from
            the plan (and layers with an empty list) are left untouched.
        seq_len: flat token count of the forward passes made inside the
            context. Required if any intervention is token-scoped; the patched
            forward raises if the actual tensor disagrees.
        weights_from: `"biased"` (default) or `"original"` — see module
            docstring.

    Attributes:
        skipped_layers: layers left fully unpatched because a mask left fewer
            than `top_k` selectable experts. **Report this number.**
        call_counts: `{layer: n_forward_calls}`. Non-zero proves the patch
            actually fired; a zero here with a non-empty plan means the
            intervention never ran and any "no effect" reading is vacuous.
    """

    def __init__(self, model, plan: dict, *, seq_len: int | None = None,
                 weights_from: str = "biased"):
        if weights_from not in ("biased", "original"):
            raise ValueError("weights_from must be 'biased' or 'original'")
        self.weights_from = weights_from
        self.seq_len = seq_len
        self.skipped_layers: list[int] = []
        self.call_counts: dict[int, int] = {}
        self._originals: dict[int, tuple] = {}

        by_name = dict(model.named_modules())
        for layer, spec in sorted(plan.items()):
            ivs = [spec] if isinstance(spec, Intervention) else list(spec)
            if not ivs:
                continue
            name = GATE_PATTERN.format(layer=layer)
            if name not in by_name:
                raise RuntimeError(f"gate module '{name}' not found on this model")
            gate = by_name[name]

            if any(iv.token_span is not None for iv in ivs) and seq_len is None:
                raise ValueError(
                    f"layer {layer}: token-scoped intervention needs seq_len="
                )
            for iv in ivs:
                bad = [e for e in iv.experts if not 0 <= e < gate.num_experts]
                if bad:
                    raise ValueError(f"layer {layer}: expert ids out of range: {sorted(bad)[:5]}")

            selectable = self._selectable_after_masks(ivs, gate.num_experts)
            if selectable is not None and len(selectable) < gate.top_k:
                # Same policy as docs/KEEP_TOPK_FAIR_PROBE.md: leave the layer
                # fully open rather than shrink top_k, and report it.
                self.skipped_layers.append(layer)
                continue

            bias = self._build_bias(ivs, gate.num_experts, seq_len, gate.weight.dtype)
            self._originals[layer] = (gate, gate.forward)
            self.call_counts[layer] = 0
            gate.forward = self._make_forward(gate, bias, layer)

    # -- bias construction --------------------------------------------------

    @staticmethod
    def _selectable_after_masks(ivs, num_experts: int) -> set[int] | None:
        """Intersection of all unscoped masks, or None if there are none.

        Token-scoped masks are excluded from this check on purpose: they only
        constrain part of the sequence, so a global `top_k` feasibility test
        would be wrong. Their feasibility is checked per-op instead.
        """
        masks = [iv for iv in ivs if iv.kind == "mask"]
        if not masks:
            return None
        allowed: set[int] = set(range(num_experts))
        for iv in masks:
            if iv.token_span is None:
                allowed &= set(iv.experts)
        return allowed

    @staticmethod
    def _build_bias(ivs, num_experts: int, seq_len: int | None, dtype) -> torch.Tensor:
        scoped = any(iv.token_span is not None for iv in ivs)
        shape = (seq_len, num_experts) if scoped else (1, num_experts)
        bias = torch.zeros(shape, dtype=torch.float32)

        for iv in ivs:
            if iv.token_span is None:
                rows = slice(None)
            else:
                s, e = iv.token_span
                if e > seq_len:
                    raise ValueError(
                        f"token_span {iv.token_span} exceeds seq_len={seq_len}"
                    )
                rows = slice(s, e)
            idx = torch.tensor(sorted(iv.experts), dtype=torch.long)
            if iv.kind == "mask":
                blocked = torch.ones(num_experts, dtype=torch.bool)
                blocked[idx] = False
                bias[rows, blocked] = float("-inf")
            else:
                bias[rows, idx] += float(iv.delta)
        return bias.to(dtype=torch.float32)

    # -- the patched forward ------------------------------------------------

    def _make_forward(self, gate, bias: torch.Tensor, layer: int):
        """Byte-for-byte `OlmoeTopKRouter.forward` plus one additive bias line.

        Kept deliberately close to the original source (same reshape, same
        softmax dtype upcast, same `norm_topk_prob` branch) so a transformers
        upgrade that changes the router is easy to diff against this.
        """
        num_experts = gate.num_experts
        hidden_dim = gate.hidden_dim
        norm_topk_prob = gate.norm_topk_prob
        top_k = gate.top_k
        weight = gate.weight
        expect_len = self.seq_len
        scoped = bias.shape[0] > 1
        weights_from = self.weights_from
        counts = self.call_counts

        def forward(hidden_states):
            hidden_states = hidden_states.reshape(-1, hidden_dim)
            if scoped and hidden_states.shape[0] != expect_len:
                raise RuntimeError(
                    f"steering: token-scoped intervention on layer {layer} expected "
                    f"seq_len={expect_len} flat tokens, got {hidden_states.shape[0]}. "
                    "Refusing to apply a mis-scoped window (batch>1 or cached "
                    "incremental decoding is not supported for scoped steering)."
                )
            counts[layer] += 1
            router_logits = F.linear(hidden_states, weight)  # (seq_len, num_experts)
            biased = router_logits + bias.to(router_logits.dtype)
            router_probs = torch.nn.functional.softmax(biased, dtype=torch.float, dim=-1)
            router_top_value, router_indices = torch.topk(router_probs, top_k, dim=-1)
            if weights_from == "original":
                orig_probs = torch.nn.functional.softmax(
                    router_logits, dtype=torch.float, dim=-1
                )
                router_top_value = torch.gather(orig_probs, -1, router_indices)
            if norm_topk_prob:
                router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
            router_top_value = router_top_value.to(router_logits.dtype)
            router_scores = router_top_value
            return biased, router_scores, router_indices

        assert num_experts == bias.shape[1]
        return forward

    # -- lifecycle ----------------------------------------------------------

    @property
    def patched_layers(self) -> list[int]:
        return sorted(self._originals)

    def remove(self) -> None:
        for _layer, (gate, orig) in self._originals.items():
            gate.forward = orig
        self._originals = {}

    def __enter__(self) -> "ExpertSteering":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()


# ---------------------------------------------------------------------------
# Diagnostic helper
# ---------------------------------------------------------------------------


def selection_rate(model, hidden_states, layers, member_mask_by_layer) -> float:
    """Fraction of a gate's top-k draws landing in a per-layer reference set.

    The same quantity as `context_metrics.set_hit_rate`, but computed directly
    from a gate's own forward rather than from a stored trace — used to check
    that a boost actually moved selection, on the exact tensors the model saw.
    """
    hits = total = 0
    by_name = dict(model.named_modules())
    for layer in layers:
        gate = by_name[GATE_PATTERN.format(layer=layer)]
        with torch.no_grad():
            _logits, _scores, idx = gate.forward(hidden_states)
        members = torch.zeros(gate.num_experts, dtype=torch.bool)
        ref = sorted(member_mask_by_layer.get(layer, set()))
        if ref:
            members[torch.tensor(ref, dtype=torch.long)] = True
        hits += int(members[idx.reshape(-1)].sum())
        total += int(idx.numel())
    return hits / total if total else float("nan")
