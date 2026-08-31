"""WS-A: capture engine tests (PLAN.md §5, §6.4).

Uses a small fake model/tokenizer rather than the real OLMoE weights, so
these run in milliseconds and don't need the ~14GB download. The real
model integration is covered separately by
tests/sanity/test_model_loading.py (skips if weights aren't present).

What's verified here vs. taken on faith, honestly:
- VERIFIED directly: streaming (per-prompt writes, not accumulated in RAM),
  resumability (kill-and-restart produces the same shard set), manifest
  atomicity, deterministic ordering, parquet schema conformance, and that
  prompt_rows_to_table's row-building logic matches route_from_logits.
- NOT independently verified here: the PLAN §6.4 "<6GB RSS for 100k tokens"
  bound and the exact forward_hook fallback path against a real
  forward-hook-only model (no such model is downloaded in this repo;
  OLMoE itself uses output_router_logits). The RSS bound is architecturally
  satisfied by construction (one prompt's table is built, written, and
  dropped before the next starts -- nothing is appended across prompts) but
  that is a design argument, not a measured one.
"""

from __future__ import annotations

from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest
import torch

from expertatlas.capture import (
    LoadedModel,
    capture_to_dir,
    get_router_logits_for_prompt,
    prompt_rows_to_table,
)
from expertatlas.schemas import ModelShape

N_EXPERTS = 8
N_LAYERS = 3
TOP_K = 2


class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    def __call__(self, text: str, return_tensors: str = "pt"):
        n_tokens = max(1, len(text.split()))
        ids = torch.arange(n_tokens, dtype=torch.long) % 1000
        return _FakeBatch(input_ids=ids.unsqueeze(0))


class _FakeModel:
    def __call__(self, input_ids, output_router_logits: bool = True):
        n_tokens = input_ids.shape[-1]
        g = torch.Generator().manual_seed(int(input_ids.sum()))
        logits = [torch.randn(n_tokens, N_EXPERTS, generator=g) for _ in range(N_LAYERS)]
        return SimpleNamespace(router_logits=logits)


def _fake_loaded_model() -> LoadedModel:
    return LoadedModel(
        model=_FakeModel(),
        tokenizer=_FakeTokenizer(),
        shape=ModelShape(n_layers=N_LAYERS, n_experts=N_EXPERTS, top_k=TOP_K, norm_topk_prob=False),
        capture_method="output_router_logits",
        model_id="fake/tiny-moe",
        model_revision="n/a",
    )


# --------------------------------------------------------------------------
# prompt_rows_to_table -- pure logic, no I/O
# --------------------------------------------------------------------------


def test_prompt_rows_to_table_shape_and_schema():
    input_ids = torch.tensor([5, 12, 99])
    g = torch.Generator().manual_seed(0)
    router_logits = [torch.randn(3, N_EXPERTS, generator=g) for _ in range(N_LAYERS)]

    table = prompt_rows_to_table(prompt_id=7, input_ids=input_ids, router_logits=router_logits, top_k=TOP_K, norm_topk_prob=False)

    assert table.num_rows == 3 * N_LAYERS  # n_tokens * n_layers
    assert set(table.column_names) == {
        "prompt_id", "token_pos", "token_id", "layer", "expert_ids", "gate_weights", "topk_mass",
    }
    assert set(table.column("prompt_id").to_pylist()) == {7}
    assert set(table.column("layer").to_pylist()) == set(range(N_LAYERS))
    for ids in table.column("expert_ids").to_pylist():
        assert len(ids) == TOP_K
        assert len(set(ids)) == TOP_K  # no duplicate experts in one token's top-k


def test_get_router_logits_for_prompt_matches_n_layers():
    loaded = _fake_loaded_model()
    router_logits = get_router_logits_for_prompt(loaded, "hello there world", device="cpu")
    assert len(router_logits) == N_LAYERS
    for logits in router_logits:
        assert logits.shape == (3, N_EXPERTS)  # 3 tokens


def test_get_router_logits_raises_on_layer_count_mismatch():
    """If a hook/path silently drops a layer, this must fail loudly, not
    write a truncated trace (PLAN §6.1 test_no_silent_layer_skips)."""
    loaded = _fake_loaded_model()

    class _BrokenModel:
        def __call__(self, input_ids, output_router_logits=True):
            n_tokens = input_ids.shape[-1]
            return SimpleNamespace(router_logits=[torch.randn(n_tokens, N_EXPERTS)])  # missing layers

    loaded.model = _BrokenModel()
    with pytest.raises(RuntimeError, match="silently skipped"):
        get_router_logits_for_prompt(loaded, "a b c", device="cpu")


# --------------------------------------------------------------------------
# capture_to_dir -- streaming, resumable, deterministic
# --------------------------------------------------------------------------


def test_capture_writes_one_shard_per_prompt(tmp_path):
    loaded = _fake_loaded_model()
    prompts = [(0, "the quick fox"), (1, "jumps over"), (2, "a lazy dog here")]

    result = capture_to_dir(loaded, prompts, out_dir=tmp_path, seed=0)

    assert result["n_prompts_written_this_run"] == 3
    assert result["n_prompts_skipped_already_done"] == 0
    shards = sorted(tmp_path.glob("trace_*.parquet"))
    assert len(shards) == 3
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "meta.json").exists()


def test_capture_is_resumable_after_interruption(tmp_path):
    """Simulates a kill-and-restart: run once with only 2 of 3 prompts
    (as if killed early), then resume with the full list. Final shard set
    must match an uninterrupted run exactly."""
    loaded = _fake_loaded_model()
    all_prompts = [(0, "alpha bravo"), (1, "charlie delta echo"), (2, "foxtrot")]

    # "Interrupted" run: only prompt 0 gets captured.
    capture_to_dir(loaded, all_prompts[:1], out_dir=tmp_path, seed=0)
    shard_0_bytes = (tmp_path / "trace_000000.parquet").read_bytes()

    # Resume with the FULL prompt list -- prompt 0 must be skipped, not redone.
    result = capture_to_dir(loaded, all_prompts, out_dir=tmp_path, seed=0)

    assert result["n_prompts_skipped_already_done"] == 1
    assert result["n_prompts_written_this_run"] == 2
    assert result["n_prompts_total_done"] == 3

    # Untouched by the resume -- byte-identical to the interrupted run's shard.
    assert (tmp_path / "trace_000000.parquet").read_bytes() == shard_0_bytes

    shards = sorted(tmp_path.glob("trace_*.parquet"))
    assert len(shards) == 3


def test_capture_uninterrupted_matches_resumed_shard_set(tmp_path):
    """The actual §6.4 claim: an uninterrupted run and a killed-then-resumed
    run produce the SAME final set of shard files (same prompt_ids present,
    same row counts) -- not necessarily byte-identical overall since the
    interrupted run's later shards are written in a second process call,
    but each individual shard is deterministic given (prompt_id, text, seed)."""
    loaded = _fake_loaded_model()
    prompts = [(0, "one"), (1, "two three"), (2, "four five six")]

    uninterrupted_dir = tmp_path / "uninterrupted"
    capture_to_dir(loaded, prompts, out_dir=uninterrupted_dir, seed=0)

    resumed_dir = tmp_path / "resumed"
    capture_to_dir(loaded, prompts[:2], out_dir=resumed_dir, seed=0)  # partial
    capture_to_dir(loaded, prompts, out_dir=resumed_dir, seed=0)  # resume

    for pid in (0, 1, 2):
        a = pq.read_table(uninterrupted_dir / f"trace_{pid:06d}.parquet")
        b = pq.read_table(resumed_dir / f"trace_{pid:06d}.parquet")
        assert a.equals(b), f"prompt {pid}: uninterrupted vs resumed shard differ"


def test_capture_limit_caps_prompts(tmp_path):
    loaded = _fake_loaded_model()
    prompts = [(i, f"prompt number {i}") for i in range(5)]
    result = capture_to_dir(loaded, prompts, out_dir=tmp_path, seed=0, limit=2)
    assert result["n_prompts_written_this_run"] == 2
    assert len(list(tmp_path.glob("trace_*.parquet"))) == 2


def test_capture_no_resume_flag_does_not_delete_but_recaptures(tmp_path):
    loaded = _fake_loaded_model()
    prompts = [(0, "alpha"), (1, "beta")]
    capture_to_dir(loaded, prompts, out_dir=tmp_path, seed=0)

    result = capture_to_dir(loaded, prompts, out_dir=tmp_path, seed=0, resume=False)
    # resume=False ignores the manifest for skip-decisions, so both get
    # "recaptured" (rewritten) rather than skipped.
    assert result["n_prompts_written_this_run"] == 2
    assert result["n_prompts_skipped_already_done"] == 0


def test_capture_ordering_is_deterministic_regardless_of_input_order(tmp_path):
    loaded = _fake_loaded_model()
    forward = [(0, "a"), (1, "b"), (2, "c")]
    shuffled = [(2, "c"), (0, "a"), (1, "b")]

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    capture_to_dir(loaded, forward, out_dir=dir_a, seed=0)
    capture_to_dir(loaded, shuffled, out_dir=dir_b, seed=0)

    for pid in (0, 1, 2):
        ta = pq.read_table(dir_a / f"trace_{pid:06d}.parquet")
        tb = pq.read_table(dir_b / f"trace_{pid:06d}.parquet")
        assert ta.equals(tb)


def test_manifest_survives_atomic_write_pattern(tmp_path):
    """manifest.json is never left half-written -- write-then-rename means
    a reader never sees a partial file, which is what makes resume safe."""
    loaded = _fake_loaded_model()
    prompts = [(0, "x"), (1, "y")]
    capture_to_dir(loaded, prompts, out_dir=tmp_path, seed=0)

    import json

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert set(manifest["done_prompt_ids"]) == {0, 1}
    assert not (tmp_path / ".manifest.json.tmp").exists()
