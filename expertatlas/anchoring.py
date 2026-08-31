"""Residual-stream anchor injection (upstream-of-router intervention).

The two router-level interventions (`docs/MECHANISM_CAUSAL.md`,
`docs/ADAPTIVE_CAUSAL.md`) moved routing as designed and did not restore
accuracy, which points upstream: by the time the router sees the hidden
state, whatever it needed may already be gone. This module intervenes on the
residual stream itself: ADD a content-bearing "anchor" vector (derived from a
clean, short-context encoding of the needle) into the hidden states entering
a chosen decoder layer, at chosen token positions.

Design constraints, mirroring `steering.py` / `adaptive_steering.py`:

  * pure forward hooks -- no weights are modified, hooks are always removed;
  * norm-matched injection: the anchor is scaled to `alpha` times the mean
    norm of the hidden states it is added to, so treatment and control
    vectors are the same size by construction and "it helped because it was
    bigger" is excluded;
  * every experiment must pair the true anchor with (a) a random-direction
    anchor and (b) a wrong-content anchor of identical scale (built by the
    caller; this module just injects whatever vector it is given).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AnchorSpec:
    """One injection: `vector` (d_model,) added to positions [pos_start, pos_end)
    of the hidden states entering decoder layer `layer`, scaled so its norm is
    `alpha` x the mean hidden-state norm over those positions."""

    layer: int
    pos_start: int
    pos_end: int
    vector: torch.Tensor
    alpha: float


class AnchorInjector:
    """Context manager that installs forward_pre_hooks on the specified decoder
    layers and adds each spec's (norm-matched) vector to the residual stream.

    Only fires on full-sequence forwards (seq_len > max pos_end), so
    incremental decoding steps are untouched -- same guard as steering.py.
    """

    def __init__(self, model, specs: list[AnchorSpec]):
        self._model = model
        self._by_layer: dict[int, list[AnchorSpec]] = {}
        for s in specs:
            if s.pos_end <= s.pos_start:
                raise ValueError(f"empty position span in spec {s}")
            self._by_layer.setdefault(s.layer, []).append(s)
        self._handles: list = []
        self.n_fired = 0

    def _make_hook(self, specs: list[AnchorSpec]):
        def hook(module, args, kwargs):
            hs = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            need = max(s.pos_end for s in specs)
            if hs.shape[1] < need:
                return None
            hs = hs.clone()
            for s in specs:
                seg = hs[:, s.pos_start:s.pos_end, :]
                mean_norm = seg.norm(dim=-1).mean()
                v = s.vector.to(dtype=hs.dtype, device=hs.device)
                v = v / v.norm() * (s.alpha * mean_norm)
                hs[:, s.pos_start:s.pos_end, :] = seg + v
                self.n_fired += 1
            if "hidden_states" in kwargs:
                kwargs = dict(kwargs)
                kwargs["hidden_states"] = hs
                return args, kwargs
            return (hs,) + tuple(args[1:]), kwargs

        return hook

    def __enter__(self):
        layers = self._model.model.layers
        for li, specs in self._by_layer.items():
            self._handles.append(
                layers[li].register_forward_pre_hook(
                    self._make_hook(specs), with_kwargs=True
                )
            )
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False
