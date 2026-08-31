"""PLAN.md Checkpoint 0 — the actual gate: does the target model load and
emit router logits at all?

Separated from the rest of tests/sanity because it needs the real model
weights on disk (~14 GB) and is slow (seconds-to-minutes, not milliseconds).
Skips cleanly if the weights aren't present yet rather than failing the
whole suite -- but per PLAN.md, `pytest tests/sanity -v` is not considered
green until this test has actually run and passed at least once.
"""

from __future__ import annotations

import os

import pytest

MODEL_ID = "allenai/OLMoE-1B-7B-0924"


def _weights_available() -> bool:
    # Cheap local check so this doesn't try to hit the network / fail slowly
    # in environments with no weights cached yet; load_model() itself does
    # the real, authoritative check.
    cache_dir = os.environ.get("HF_HOME", "data/hf_cache")
    return os.path.isdir(cache_dir) and any(
        "OLMoE" in name for root, dirs, files in os.walk(cache_dir) for name in dirs
    )


@pytest.mark.skipif(
    not _weights_available(),
    reason=f"{MODEL_ID} weights not found locally -- run the download first",
)
def test_model_loads_and_emits_router_logits():
    from expertatlas.capture import load_model, route_from_logits

    loaded = load_model(MODEL_ID, device="cpu", dtype="bfloat16")

    assert loaded.capture_method in ("output_router_logits", "forward_hook")
    assert loaded.shape.n_layers == 16
    assert loaded.shape.n_experts == 64
    assert loaded.shape.top_k == 8

    import torch

    inputs = loaded.tokenizer("def foo():", return_tensors="pt")
    with torch.no_grad():
        out = loaded.model(**inputs, output_router_logits=True)

    assert out.router_logits is not None
    assert len(out.router_logits) == loaded.shape.n_layers

    n_tokens = inputs["input_ids"].shape[1]
    for layer_logits in out.router_logits:
        assert layer_logits.shape == (n_tokens, loaded.shape.n_experts)
        ids, weights, mass = route_from_logits(
            layer_logits, loaded.shape.top_k, loaded.shape.norm_topk_prob
        )
        assert ids.shape == (n_tokens, loaded.shape.top_k)
        assert bool((mass > 0).all())
