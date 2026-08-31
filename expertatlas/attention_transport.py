"""Attention transport tools (WS1): last-row attention capture and head boost.

Two pieces, both implemented as custom attention interfaces (registered via
`transformers.AttentionInterface`) that reproduce the eager attention math
exactly and either (a) record the final query row's post-softmax attention
per head, or (b) add a pre-softmax bias steering selected heads' attention
from selected query positions onto a key span (the needle).

Memory note: capture stores ONLY `attn[:, :, -1, :]` (one row per layer/head),
never the full (seq, seq) matrix, so a 3840-token prompt costs
16 layers x 16 heads x 3840 floats ~= 4 MiB, not ~15 GiB.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from transformers import AttentionInterface
from transformers.models.olmoe.modeling_olmoe import repeat_kv

CAPTURE_IMPL = "expertatlas_capture_last_row"
BOOST_IMPL = "expertatlas_head_boost"

_ACTIVE_CAPTURE: list["LastRowAttentionCapture"] = []
_ACTIVE_BOOST: list["HeadBoost"] = []


def _eager_scores(module, query, key, attention_mask, scaling):
    key_states = repeat_kv(key, module.num_key_value_groups)
    scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask
    else:
        # Custom attention interfaces receive attention_mask=None for the
        # pure-causal no-padding case (like sdpa's is_causal path); apply
        # the causal mask explicitly.
        q_len, k_len = scores.shape[2], scores.shape[3]
        if q_len > 1:
            i = torch.arange(q_len, device=scores.device).unsqueeze(1)
            j = torch.arange(k_len, device=scores.device)
            scores = scores.masked_fill(
                j > i + (k_len - q_len), torch.finfo(scores.dtype).min
            )
    return scores, key_states


def _transport_attention_forward(module, query, key, value, attention_mask,
                                 scaling, dropout=0.0, **kwargs):
    """Eager attention with optional pre-softmax head boost and last-row
    capture; boost and capture nest freely (both consult the active stacks).
    """
    scores, key_states = _eager_scores(module, query, key, attention_mask, scaling)
    value_states = repeat_kv(value, module.num_key_value_groups)
    if _ACTIVE_BOOST:
        b = _ACTIVE_BOOST[-1]
        heads = b.heads_by_layer.get(module.layer_idx)
        if heads:
            ks, ke = b.key_span
            qs = b.query_start if b.query_start is not None else scores.shape[2] - 1
            for h in heads:
                scores[0, h, qs:, ks:ke] += b.beta
            b.n_fired += len(heads)
    attn = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    if _ACTIVE_CAPTURE:
        _ACTIVE_CAPTURE[-1].rows[module.layer_idx] = attn[0, :, -1, :].float().cpu()
    out = torch.matmul(attn, value_states).transpose(1, 2).contiguous()
    return out, None


AttentionInterface.register(CAPTURE_IMPL, _transport_attention_forward)
AttentionInterface.register(BOOST_IMPL, _transport_attention_forward)


class LastRowAttentionCapture:
    """Context manager: capture post-softmax attention from the FINAL query
    position, per layer and head, during one forward pass.

    After the pass, `rows[layer]` is a (n_heads, seq_len) float32 tensor.
    """

    def __init__(self, model):
        self.model = model
        self.rows: dict[int, torch.Tensor] = {}
        self._prev_impl = model.config._attn_implementation

    def __enter__(self):
        self.model.set_attn_implementation(CAPTURE_IMPL)
        _ACTIVE_CAPTURE.append(self)
        return self

    def __exit__(self, *exc):
        _ACTIVE_CAPTURE.remove(self)
        self.model.set_attn_implementation(self._prev_impl)
        return False

    def needle_mass(self, span: tuple[int, int]) -> torch.Tensor:
        """(n_layers, n_heads) total attention mass on `span` from the last
        position."""
        layers = sorted(self.rows)
        return torch.stack(
            [self.rows[i][:, span[0]:span[1]].sum(dim=-1) for i in layers]
        )


@dataclass
class HeadBoost:
    """Context manager: add `beta` to pre-softmax attention scores at the
    given (layer, head) cells, from query positions `query_start:` onto keys
    `key_span[0]:key_span[1]`. `query_start=None` boosts only the final row.
    """

    model: object
    heads_by_layer: dict[int, list[int]]
    key_span: tuple[int, int]
    beta: float
    query_start: int | None = None
    n_fired: int = field(default=0, init=False)

    def __enter__(self):
        self._prev_impl = self.model.config._attn_implementation
        self.model.set_attn_implementation(BOOST_IMPL)
        _ACTIVE_BOOST.append(self)
        return self

    def __exit__(self, *exc):
        _ACTIVE_BOOST.remove(self)
        self.model.set_attn_implementation(self._prev_impl)
        return False


def heads_to_by_layer(heads: list[tuple[int, int]]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for layer, head in heads:
        out.setdefault(int(layer), []).append(int(head))
    return out
