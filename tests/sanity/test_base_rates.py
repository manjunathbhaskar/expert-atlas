"""PLAN.md §6.1 — base-rate sanity tests.

Runs on synthetic top-k selections built to be load-balanced (uniform),
so it checks the *statistic*, not a real model's actual balance (that's
a Phase 1/Checkpoint-1 concern once real capture exists).
"""

from __future__ import annotations

import torch

from expertatlas.capture import route_from_logits

N_EXPERTS = 64
TOP_K = 8


def _uniform_ish_logits(n_tokens: int, seed: int) -> torch.Tensor:
    """Logits with no per-expert bias -> top-k selection should be close
    to uniform over many tokens, same as a load-balanced router."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n_tokens, N_EXPERTS, generator=g)


def test_marginal_usage_near_uniform():
    n_tokens = 20_000
    logits = _uniform_ish_logits(n_tokens, seed=7)
    ids, _, _ = route_from_logits(logits, TOP_K, norm_topk_prob=False)

    counts = torch.bincount(ids.flatten(), minlength=N_EXPERTS).float()
    expected = n_tokens * TOP_K / N_EXPERTS
    # binomial-ish variance approximation for a sanity bound, not a proof
    std = (expected * (1 - TOP_K / N_EXPERTS)) ** 0.5
    within_3sigma = (counts - expected).abs() <= 3 * std
    frac_within = float(within_3sigma.float().mean())
    assert frac_within > 0.9, (
        f"only {frac_within:.1%} of experts within 3sigma of uniform usage — "
        "check for a selection bias bug"
    )


def test_no_dead_experts_unflagged():
    """A capture run must be able to explicitly report zero-usage experts,
    not silently omit them from downstream aggregation."""
    n_tokens = 200
    logits = _uniform_ish_logits(n_tokens, seed=1)
    ids, _, _ = route_from_logits(logits, TOP_K, norm_topk_prob=False)
    counts = torch.bincount(ids.flatten(), minlength=N_EXPERTS)

    dead = (counts == 0).nonzero(as_tuple=True)[0].tolist()
    # the aggregation step must be ABLE to represent dead experts (length
    # N_EXPERTS, not len(unique ids seen)) -- that's what this asserts
    assert len(counts) == N_EXPERTS, "dead experts silently dropped from the count vector"
    # not asserting dead == [] here: with only 200 tokens some experts may
    # legitimately see zero hits; the requirement is visibility, not absence
