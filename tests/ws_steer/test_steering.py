"""Steering primitives, tested against the REAL router class, not a stand-in.

`OlmoeTopKRouter` is instantiated directly from
`transformers.models.olmoe.modeling_olmoe` with a tiny hand-built config, so
these tests exercise the exact code path `ExpertSteering` monkey-patches
without downloading the 13GB checkpoint (this whole file runs in ~1s).

The four properties worth testing, per the handoff brief:
  1. a masked expert is NEVER selected,
  2. a boosted expert is selected MORE OFTEN than in an unboosted control,
  3. removing the patch restores byte-identical original behaviour,
  4. a token-scoped intervention changes selection inside its window and
     leaves every token outside it byte-identical.

(4) is the one that matters most for `docs/MECHANISM_CAUSAL.md`: a window that
silently slipped by a few tokens would produce a "diffuse boost helps a bit"
result that looks like a real finding.
"""

import pytest
import torch
import torch.nn as nn
from transformers.models.olmoe.modeling_olmoe import OlmoeTopKRouter

from expertatlas.steering import (
    ExpertSteering,
    Intervention,
    boost,
    boost_global,
    by_layer,
    mask_to,
    mask_to_global,
)

N_LAYERS = 3
N_EXPERTS = 16
TOP_K = 4
HIDDEN = 8
SEQ = 12


class _Cfg:
    num_experts_per_tok = TOP_K
    num_experts = N_EXPERTS
    norm_topk_prob = False          # OLMoE's real setting -- see docs/ANALYSIS_GUARDRAILS.md §5
    hidden_size = HIDDEN


def _make_model(seed: int = 0):
    """An nn.Module tree whose named_modules() match the real gate paths."""
    torch.manual_seed(seed)
    root = nn.Module()
    root.model = nn.Module()
    layers = nn.ModuleList()
    for _ in range(N_LAYERS):
        layer = nn.Module()
        layer.mlp = nn.Module()
        gate = OlmoeTopKRouter(_Cfg())
        with torch.no_grad():
            gate.weight.copy_(torch.randn(N_EXPERTS, HIDDEN) * 0.5)
        layer.mlp.gate = gate
        layers.append(layer)
    root.model.layers = layers
    return root


def _gate(model, layer=0):
    return dict(model.named_modules())[f"model.layers.{layer}.mlp.gate"]


@pytest.fixture
def model():
    return _make_model()


@pytest.fixture
def hidden():
    torch.manual_seed(99)
    return torch.randn(SEQ, HIDDEN)


# ---------------------------------------------------------------------------
# 1. masking
# ---------------------------------------------------------------------------


def test_masked_experts_are_never_selected(model, hidden):
    allowed = {1, 3, 5, 7, 9, 11}
    with ExpertSteering(model, {0: mask_to(allowed)}):
        _lg, _sc, idx = _gate(model).forward(hidden)
    assert set(idx.reshape(-1).tolist()) <= allowed


def test_mask_keeps_top_k_slots_full_and_real(model, hidden):
    allowed = set(range(TOP_K))          # exactly top_k allowed
    with ExpertSteering(model, {0: mask_to(allowed)}) as st:
        assert st.skipped_layers == []
        _lg, sc, idx = _gate(model).forward(hidden)
    assert idx.shape == (SEQ, TOP_K)
    # every row selects each allowed expert exactly once, and no weight is -inf/nan
    for row in idx.tolist():
        assert sorted(row) == sorted(allowed)
    assert torch.isfinite(sc).all()


def test_layer_with_fewer_than_top_k_allowed_is_skipped_not_shrunk(model, hidden):
    before = _gate(model).forward(hidden)
    with ExpertSteering(model, {0: mask_to({2, 4})}) as st:
        assert st.skipped_layers == [0]
        assert st.patched_layers == []
        during = _gate(model).forward(hidden)
    for a, b in zip(before, during):
        assert torch.equal(a, b), "skipped layer must be left completely unrestricted"


def test_mask_beats_boost_when_they_conflict(model, hidden):
    """-inf + delta is still -inf: a boost cannot resurrect a masked expert."""
    allowed = {0, 1, 2, 3, 4, 5}
    plan = {0: [mask_to(allowed), boost({7, 8}, 50.0)]}
    with ExpertSteering(model, plan):
        _lg, _sc, idx = _gate(model).forward(hidden)
    assert set(idx.reshape(-1).tolist()) <= allowed


# ---------------------------------------------------------------------------
# 2. boosting
# ---------------------------------------------------------------------------


def _hit_rate(idx, targets) -> float:
    flat = idx.reshape(-1).tolist()
    return sum(1 for e in flat if e in targets) / len(flat)


def test_boost_increases_selection_rate_vs_unboosted_control(model, hidden):
    targets = {12, 13, 14}
    _lg, _sc, idx0 = _gate(model).forward(hidden)
    base = _hit_rate(idx0, targets)
    with ExpertSteering(model, {0: boost(targets, 3.0)}):
        _lg, _sc, idx1 = _gate(model).forward(hidden)
    boosted = _hit_rate(idx1, targets)
    assert boosted > base, f"boost did not raise selection rate ({base} -> {boosted})"


def test_boost_rate_is_monotone_in_delta(model, hidden):
    targets = {12, 13, 14}
    rates = []
    for delta in (0.0, 1.0, 3.0, 20.0):
        with ExpertSteering(model, {0: boost(targets, delta)}):
            _lg, _sc, idx = _gate(model).forward(hidden)
        rates.append(_hit_rate(idx, targets))
    assert rates == sorted(rates), rates
    # a huge boost must saturate: all three targets in every row's top-4
    assert rates[-1] == pytest.approx(len(targets) / TOP_K)


def test_zero_boost_is_a_no_op(model, hidden):
    before = _gate(model).forward(hidden)
    with ExpertSteering(model, {0: boost({5}, 0.0)}):
        during = _gate(model).forward(hidden)
    for a, b in zip(before, during):
        assert torch.equal(a, b)


def test_boost_does_not_touch_unpatched_layers(model, hidden):
    before = _gate(model, 1).forward(hidden)
    with ExpertSteering(model, {0: boost({1, 2}, 5.0)}) as st:
        assert st.patched_layers == [0]
        during = _gate(model, 1).forward(hidden)
    for a, b in zip(before, during):
        assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# 3. restoration
# ---------------------------------------------------------------------------


def test_removal_restores_byte_identical_behaviour(model, hidden):
    before = [t.clone() for t in _gate(model).forward(hidden)]
    orig_fn = _gate(model).forward
    with ExpertSteering(model, {0: mask_to({0, 1, 2, 3, 4}), 2: boost({9}, 4.0)}):
        pass
    after = _gate(model).forward(hidden)
    for a, b in zip(before, after):
        assert torch.equal(a, b)
    assert _gate(model).forward == orig_fn, "the original bound method was not restored"


def test_unpatched_forward_matches_the_real_class_output(model, hidden):
    """The patched forward with a no-op bias must reproduce the real router
    exactly -- this is what makes it safe to attribute any difference under a
    real intervention to the intervention rather than to a reimplementation
    bug."""
    reference = [t.clone() for t in _gate(model).forward(hidden)]
    with ExpertSteering(model, {0: boost(set(range(N_EXPERTS)), 1.0)}):
        # boosting EVERY expert by the same amount shifts all logits equally,
        # so softmax, top-k order and gate weights are unchanged.
        patched = _gate(model).forward(hidden)
    assert torch.equal(reference[2], patched[2])
    torch.testing.assert_close(reference[1], patched[1])


def test_exception_inside_context_still_restores(model, hidden):
    before = [t.clone() for t in _gate(model).forward(hidden)]
    with pytest.raises(ValueError):
        with ExpertSteering(model, {0: boost({3}, 9.0)}):
            raise ValueError("boom")
    after = _gate(model).forward(hidden)
    for a, b in zip(before, after):
        assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# 4. token-window scoping
# ---------------------------------------------------------------------------


def test_scoped_boost_changes_only_its_window(model, hidden):
    span = (4, 7)
    targets = {12, 13, 14}
    _lg0, sc0, idx0 = _gate(model).forward(hidden)
    with ExpertSteering(model, {0: boost(targets, 20.0, span)}, seq_len=SEQ):
        _lg1, sc1, idx1 = _gate(model).forward(hidden)

    inside = slice(*span)
    assert not torch.equal(idx0[inside], idx1[inside]), "no effect inside the window"
    for rows in (slice(0, span[0]), slice(span[1], SEQ)):
        assert torch.equal(idx0[rows], idx1[rows]), "leaked outside the window"
        assert torch.equal(sc0[rows], sc1[rows])
    assert _hit_rate(idx1[inside], targets) == pytest.approx(len(targets) / TOP_K)


def test_scoped_mask_applies_only_inside_window(model, hidden):
    span = (2, 5)
    allowed = {0, 1, 2, 3, 4, 5}
    with ExpertSteering(model, {0: mask_to(allowed, span)}, seq_len=SEQ):
        _lg, _sc, idx = _gate(model).forward(hidden)
    assert set(idx[slice(*span)].reshape(-1).tolist()) <= allowed
    outside = set(idx[span[1]:].reshape(-1).tolist())
    assert not outside <= allowed, "unscoped tokens should be unrestricted"


def test_scoped_intervention_refuses_wrong_sequence_length(model):
    torch.manual_seed(1)
    wrong = torch.randn(SEQ + 3, HIDDEN)
    with ExpertSteering(model, {0: boost({1}, 5.0, (0, 2))}, seq_len=SEQ):
        with pytest.raises(RuntimeError, match="expected seq_len"):
            _gate(model).forward(wrong)


def test_scoped_intervention_requires_seq_len(model):
    with pytest.raises(ValueError, match="seq_len"):
        ExpertSteering(model, {0: boost({1}, 5.0, (0, 2))})


def test_span_beyond_seq_len_is_rejected(model):
    with pytest.raises(ValueError, match="exceeds seq_len"):
        ExpertSteering(model, {0: boost({1}, 5.0, (0, SEQ + 1))}, seq_len=SEQ)


# ---------------------------------------------------------------------------
# weights_from: separating "which experts" from "how much weight"
# ---------------------------------------------------------------------------


def test_weights_from_original_keeps_selection_but_not_inflated_weights(model, hidden):
    targets = {12, 13, 14}
    with ExpertSteering(model, {0: boost(targets, 3.0)}) as _st:
        _lg_b, sc_b, idx_b = _gate(model).forward(hidden)
    with ExpertSteering(model, {0: boost(targets, 3.0)}, weights_from="original"):
        _lg_o, sc_o, idx_o = _gate(model).forward(hidden)

    assert torch.equal(idx_b, idx_o), "selection must be identical -- only weights differ"
    assert not torch.allclose(sc_b, sc_o)

    # the 'original' weights must equal the UNBIASED softmax at those indices
    unbiased = torch.nn.functional.softmax(
        torch.nn.functional.linear(hidden, _gate(model).weight), dtype=torch.float, dim=-1
    )
    torch.testing.assert_close(sc_o, torch.gather(unbiased, -1, idx_o).to(sc_o.dtype))


def test_bad_weights_from_rejected(model):
    with pytest.raises(ValueError, match="weights_from"):
        ExpertSteering(model, {0: boost({1}, 1.0)}, weights_from="nonsense")


# ---------------------------------------------------------------------------
# bookkeeping, plan helpers, validation
# ---------------------------------------------------------------------------


def test_call_counts_prove_the_patch_fired(model, hidden):
    with ExpertSteering(model, {0: boost({1}, 1.0), 1: mask_to(set(range(8)))}) as st:
        assert st.call_counts == {0: 0, 1: 0}
        _gate(model, 0).forward(hidden)
        _gate(model, 0).forward(hidden)
        _gate(model, 1).forward(hidden)
        assert st.call_counts == {0: 2, 1: 1}


def test_by_layer_and_global_plan_helpers():
    pairs = {(0, 1), (0, 2), (2, 5)}
    assert by_layer(pairs) == {0: {1, 2}, 2: {5}}
    plan = boost_global(pairs, 2.5)
    assert set(plan) == {0, 2}
    assert plan[0][0].kind == "boost" and plan[0][0].delta == 2.5
    assert mask_to_global(pairs)[2][0].experts == frozenset({5})


def test_global_plan_runs_on_all_its_layers(model, hidden):
    pairs = {(l, e) for l in range(N_LAYERS) for e in (12, 13, 14)}
    with ExpertSteering(model, boost_global(pairs, 20.0)) as st:
        assert st.patched_layers == list(range(N_LAYERS))
        for l in range(N_LAYERS):
            _lg, _sc, idx = _gate(model, l).forward(hidden)
            assert _hit_rate(idx, {12, 13, 14}) == pytest.approx(3 / TOP_K)


def test_empty_plan_entries_are_left_alone(model):
    with ExpertSteering(model, {0: [], 1: boost({1}, 1.0)}) as st:
        assert st.patched_layers == [1]


def test_out_of_range_expert_ids_rejected(model):
    with pytest.raises(ValueError, match="out of range"):
        ExpertSteering(model, {0: boost({N_EXPERTS}, 1.0)})


def test_missing_gate_path_is_loud(model):
    with pytest.raises(RuntimeError, match="not found"):
        ExpertSteering(model, {N_LAYERS + 5: boost({1}, 1.0)})


def test_intervention_validation():
    with pytest.raises(ValueError, match="unknown intervention kind"):
        Intervention("scale", frozenset({1}))
    with pytest.raises(ValueError, match="boost requires delta"):
        Intervention("boost", frozenset({1}))
    with pytest.raises(ValueError, match="empty expert set"):
        mask_to(set())
    with pytest.raises(ValueError, match="bad token_span"):
        boost({1}, 1.0, (5, 5))
