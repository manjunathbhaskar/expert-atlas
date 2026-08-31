"""Entropy-triggered adaptive router steering.

`docs/MECHANISM.md` proposed biasing the router toward the needle-affine
experts "when entropy spikes"; `docs/MECHANISM_CAUSAL.md` tested the simpler
always-on variant and it FAILED (large boost catastrophic, small boost not
significant), explicitly noting the triggered policy remained untested. This
module implements that remaining variant.

Mechanism
---------
Each patched gate computes its own unbiased router logits, measures the
full-softmax entropy per token, and adds `+delta` to the target experts'
logits ONLY at tokens whose entropy exceeds a per-layer threshold `tau`.
Selection then proceeds exactly as `OlmoeTopKRouter.forward` does. The
thresholds must be calibrated on data that is never evaluated (see the
runner script), keeping the policy accuracy-blind.

Differences from `expertatlas.steering.ExpertSteering` (static plans):
* the intervention mask is computed at forward time from the model's own
  routing state — it cannot be prebuilt, so this is a separate class rather
  than a new `Intervention` kind;
* there is no token-span scoping: the trigger decides where to act;
* per-layer trigger counts are recorded (`trigger_counts`, `token_counts`)
  as the manipulation check's raw material.

Shared honest limitations (same as steering.py): this steers routing, not
compute; the biased logits are what selection is made from and what any
`output_router_logits=True` consumer sees; `weights_from="biased"` semantics
only (the boost changes both selection and gate weight).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

GATE_PATTERN = "model.layers.{layer}.mlp.gate"
_LOG2E = 1.0 / math.log(2.0)


def entropy_bits(logits: torch.Tensor) -> torch.Tensor:
    """Full-softmax entropy per row, in bits (float32 math)."""
    logp = F.log_softmax(logits.float(), dim=-1)
    return (-(logp.exp() * logp).sum(dim=-1)) * _LOG2E


class AdaptiveEntropySteering:
    """Context manager: boost `experts_by_layer` by `delta` at tokens whose
    unbiased router entropy exceeds `tau_by_layer[layer]` (bits).

    Args:
        model: OLMoE-style model (gates at `model.layers.{i}.mlp.gate`).
        experts_by_layer: `{layer: {expert_idx, ...}}` targets. Layers absent
            are untouched.
        delta: additive logit boost at triggered tokens.
        tau_by_layer: `{layer: threshold_bits}`. A layer in `experts_by_layer`
            without a threshold is an error (a silently-untriggered layer
            would look like a null result).

    Attributes:
        trigger_counts: `{layer: n_tokens_triggered}` across all forwards.
        token_counts: `{layer: n_tokens_seen}`.
        call_counts: `{layer: n_forward_calls}` — must be non-zero for a
            non-vacuous run.
    """

    def __init__(self, model, experts_by_layer: dict[int, set[int]],
                 delta: float, tau_by_layer: dict[int, float]):
        self.delta = float(delta)
        self.trigger_counts: dict[int, int] = {}
        self.token_counts: dict[int, int] = {}
        self.call_counts: dict[int, int] = {}
        self._originals: dict[int, tuple] = {}

        missing = sorted(set(experts_by_layer) - set(tau_by_layer))
        if missing:
            raise ValueError(f"layers without an entropy threshold: {missing}")

        by_name = dict(model.named_modules())
        for layer, experts in sorted(experts_by_layer.items()):
            if not experts:
                continue
            name = GATE_PATTERN.format(layer=layer)
            if name not in by_name:
                raise RuntimeError(f"gate module '{name}' not found on this model")
            gate = by_name[name]
            bad = [e for e in experts if not 0 <= e < gate.num_experts]
            if bad:
                raise ValueError(f"layer {layer}: expert ids out of range: {sorted(bad)[:5]}")

            idx = torch.tensor(sorted(int(e) for e in experts), dtype=torch.long)
            tau = float(tau_by_layer[layer])
            self._originals[layer] = (gate, gate.forward)
            self.call_counts[layer] = 0
            self.trigger_counts[layer] = 0
            self.token_counts[layer] = 0
            gate.forward = self._make_forward(gate, idx, tau, layer)

    def _make_forward(self, gate, expert_idx: torch.Tensor, tau: float, layer: int):
        hidden_dim = gate.hidden_dim
        norm_topk_prob = gate.norm_topk_prob
        top_k = gate.top_k
        weight = gate.weight
        delta = self.delta
        trig, seen, calls = self.trigger_counts, self.token_counts, self.call_counts

        def forward(hidden_states):
            hidden_states = hidden_states.reshape(-1, hidden_dim)
            calls[layer] += 1
            router_logits = F.linear(hidden_states, weight)
            ent = entropy_bits(router_logits)
            fire = ent > tau                                # (seq_len,)
            trig[layer] += int(fire.sum())
            seen[layer] += int(fire.numel())
            bias = torch.zeros_like(router_logits, dtype=torch.float32)
            bias[:, expert_idx] = delta
            bias = bias * fire.unsqueeze(-1).to(bias.dtype)
            biased = router_logits + bias.to(router_logits.dtype)
            router_probs = torch.nn.functional.softmax(biased, dtype=torch.float, dim=-1)
            router_top_value, router_indices = torch.topk(router_probs, top_k, dim=-1)
            if norm_topk_prob:
                router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
            router_top_value = router_top_value.to(router_logits.dtype)
            return biased, router_top_value, router_indices

        return forward

    @property
    def patched_layers(self) -> list[int]:
        return sorted(self._originals)

    def trigger_rate(self) -> float:
        total_seen = sum(self.token_counts.values())
        return (sum(self.trigger_counts.values()) / total_seen) if total_seen else float("nan")

    def remove(self) -> None:
        for _layer, (gate, orig) in self._originals.items():
            gate.forward = orig
        self._originals = {}

    def __enter__(self) -> "AdaptiveEntropySteering":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()
