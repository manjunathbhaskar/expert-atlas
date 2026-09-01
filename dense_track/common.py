"""Dense-transformer track (Pythia-2.8B): shared tools.

Parallel, independent replication track: tests whether the attention-transport
causal pattern found on OLMoE (docs/ATTENTION_TRANSPORT.md,
docs/ATTENTION_BOOST_CAUSAL.md) also appears in a small dense transformer.
Nothing in this directory imports from or modifies the MoE experimental code
paths; probe templates are shared read-only via `probes.probe_set_context`
constants so the substrates stay comparable.

Model: EleutherAI/pythia-2.8b (GPT-NeoX architecture, dense, 32 layers x
32 heads, max_position_embeddings=2048, no MoE routing). fp32, CPU, eager
math, teacher-forced single forward pass, deterministic.

Attention capture stores ONLY the final query row per (layer, head):
32 x 32 x seq floats (~15 MiB at 1900 tokens), never the full matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "EleutherAI/pythia-2.8b"
N_LAYERS = 32
N_HEADS = 32

TRACK_ROOT = Path(__file__).parent
DATA_DIR = TRACK_ROOT / "data"
PROBE_SET = TRACK_ROOT / "probe_set_dense.yaml"
RECORDS = DATA_DIR / "records.jsonl"

IMPL_NAME = "dense_track_transport"

_ACTIVE_CAPTURE: list["LastRowAttentionCapture"] = []
_ACTIVE_BOOST: list["HeadBoost"] = []


def _dense_attention_forward(module, query, key, value, attention_mask,
                             scaling, dropout=0.0, head_mask=None, **kwargs):
    """GPT-NeoX eager attention math with optional pre-softmax head boost and
    last-row capture. Mirrors expertatlas.attention_transport for OLMoE, minus
    repeat_kv (GPT-NeoX has no grouped KV heads)."""
    scores = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : key.shape[-2]]
    else:
        # Custom attention interfaces receive attention_mask=None for the
        # pure-causal no-padding case; apply the causal mask explicitly.
        q_len, k_len = scores.shape[2], scores.shape[3]
        if q_len > 1:
            i = torch.arange(q_len, device=scores.device).unsqueeze(1)
            j = torch.arange(k_len, device=scores.device)
            scores = scores.masked_fill(
                j > i + (k_len - q_len), torch.finfo(scores.dtype).min
            )
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
    out = torch.matmul(attn, value).transpose(1, 2).contiguous()
    return out, None


AttentionInterface.register(IMPL_NAME, _dense_attention_forward)


class LastRowAttentionCapture:
    """Context manager: capture post-softmax attention from the FINAL query
    position, per layer and head, during one forward pass."""

    def __init__(self, model):
        self.model = model
        self.rows: dict[int, torch.Tensor] = {}
        self._prev_impl = model.config._attn_implementation

    def __enter__(self):
        self.model.set_attn_implementation(IMPL_NAME)
        _ACTIVE_CAPTURE.append(self)
        return self

    def __exit__(self, *exc):
        _ACTIVE_CAPTURE.remove(self)
        self.model.set_attn_implementation(self._prev_impl)
        return False

    def needle_mass(self, span: tuple[int, int]) -> torch.Tensor:
        layers = sorted(self.rows)
        return torch.stack(
            [self.rows[i][:, span[0]:span[1]].sum(dim=-1) for i in layers]
        )


@dataclass
class HeadBoost:
    """Context manager: add `beta` to pre-softmax scores at the given
    (layer, head) cells, queries `query_start:` onto keys `key_span`."""

    model: object
    heads_by_layer: dict[int, list[int]]
    key_span: tuple[int, int]
    beta: float
    query_start: int | None = None
    n_fired: int = field(default=0, init=False)

    def __enter__(self):
        self._prev_impl = self.model.config._attn_implementation
        self.model.set_attn_implementation(IMPL_NAME)
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


def load_dense_model():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    return model, tok


def score_answer(last_logits, candidate_ids, answer_id) -> dict:
    lg = last_logits.detach().cpu().numpy().astype(np.float64)
    cand = np.array(candidate_ids, dtype=np.int64)
    cand_logits = lg[cand]
    ans_pos = int(np.where(cand == answer_id)[0][0])
    m = cand_logits.max()
    p = np.exp(cand_logits - m)
    p = p / p.sum()
    return {
        "forced_choice_correct": bool(int(np.argmax(cand_logits)) == ans_pos),
        "forced_choice_prob": float(p[ans_pos]),
    }


def char_span_to_token_span(tok, text: str, char_span: tuple[int, int]) -> tuple[int, int]:
    enc = tok(text, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    cs, ce = char_span
    idxs = [i for i, (a, b) in enumerate(offs) if a < ce and cs < b]
    if not idxs:
        raise RuntimeError(f"char span {char_span} maps to no tokens")
    return idxs[0], idxs[-1] + 1


def paired_stats(a: np.ndarray, b: np.ndarray, rng, n_perm: int = 2000) -> dict:
    d = a - b
    dz = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("nan")
    obs = d.mean()
    flips = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = (flips * d).mean(axis=1)
    return {"mean_delta": float(obs), "dz": dz,
            "perm_p": float((np.abs(null) >= abs(obs)).mean()), "n": int(len(d))}


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]
