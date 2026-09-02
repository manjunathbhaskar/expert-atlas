# Phase 0 — Foundation: status

Per PLAN.md §5 Checkpoint 0.

## Done

- [x] `pyproject.toml` — deps pinned (torch, transformers, pyarrow, numpy, scipy,
      statsmodels, umap-learn, networkx, python-louvain, pytest, typer, pydantic).
      venv on Python 3.13 (not 3.14 — numba/llvmlite for umap-learn need it).
- [x] Contracts §3 written into `expertatlas/schemas.py`:
      - `ROUTING_TRACE_SCHEMA` (pyarrow) + `RoutingTraceMeta` (pydantic) for Contract 1
      - `Atlas` + nested models for Contract 2 (atlas.json)
      - **One addition vs. the plan's own table:** `topk_mass` column on
        RoutingTrace — required by the plan's own §6.1 sanity test
        (`test_topk_mass_is_recorded`) but missing from the §3 schema table.
        Logged here rather than silently patched, per the "don't silently
        edit a frozen contract" rule in §4.
- [x] `tests/fixtures/atlas_synthetic.json` generated with planted structure
      (§6.2): expert L03E17 boosted 5x on domain `code.python`, everything
      else noise. Validated against the `Atlas` schema.
- [x] Minimal `expertatlas/stats.py` (lift, chi2, BH-FDR, label-shuffle null) —
      pulled forward from Phase 2/WS-C scope because Checkpoint 0 requires
      the §6.1 null-model tests to pass, and those need lift + FDR to exist.
      Full WS-C scope (co-activation, split-half, effect sizes) is still
      Phase 2, not implemented here.
- [x] `expertatlas/capture.py` — `load_model()` with output_router_logits
      primary path + forward-hook fallback detection (fallback path itself
      is WS-A scope, not implemented — Phase 0 only needs to prove the
      primary path works for OLMoE and detect if it doesn't).
- [x] `atlas capture --prompt "..." --dry-run` CLI works end-to-end.
- [x] `pytest tests/sanity -v` — 19/19 passing (all tests not requiring the
      real model weights on disk).

## Real Checkpoint 0 gate — CONFIRMED PASSING

`pytest tests/sanity/test_model_loading.py -v` — **1 passed in 58.28s**,
against the actual downloaded OLMoE-1B-7B-0924 weights (not synthetic data).
Confirms: model loads, `output_router_logits=True` works natively (no
forward-hook fallback needed for this model), n_layers=16, n_experts=64,
top_k=8, and a real forward pass emits well-formed router logits per layer.

**A real bug was found and fixed getting here, worth recording:** the first
attempt appeared to hang for ~30 minutes at low CPU. It was actually
*silently re-downloading the full 14GB model* — `snapshot_download` was
called with `cache_dir='data/hf_cache'` directly, but the test sets
`HF_HOME=data/hf_cache`, and `transformers` looks for weights under
`HF_HOME/hub/...`, a different path. ~5GB was re-downloaded before this was
caught (via `lsof` showing `.incomplete` files growing in the wrong
location). Fixed by symlinking the real snapshot into the `hub/` path
`HF_HOME` expects, and adding `HF_HUB_OFFLINE=1` when running the test so a
future cache-path mismatch fails fast and loud instead of silently
re-downloading again.

**Phase 0 / Checkpoint 0 is now fully complete.**

## Two things fixed during test-writing (worth recording, not just silently patching)

1. **Null-model test needed more tokens.** At 2000 tokens/domain, pure
   sampling noise alone produced mean|lift| ~0.18 — failing a naive <0.15
   bound for a reason that had nothing to do with a real bug. Fixed by
   using enough tokens (20k) that noise and signal are actually
   distinguishable, not by loosening the threshold to match noise.
2. **Planted-structure test needed more domains.** With only 6 domains,
   the boosted domain is ~17% of that expert's total activity, which
   contaminates the marginal P(expert) used as the lift denominator and
   attenuates the observed lift well below the true fold-change (1.4 vs
   expected 2.32 for a 5x fold). This is a real, inherent property of the
   lift formula when domain count is small — not a computation bug. Fixed
   by using domain counts closer to the real project's scale (150 vs the
   real plan's 240 factorial cells), which is also more representative of
   actual Phase 3 conditions than an arbitrary small number would have been.

## Not yet started (Phase 1+, per plan)

WS-A full capture engine (batched, streaming parquet, resumable), WS-B
probe set (zero prompts written yet — §7, the highest-risk item per §9b),
WS-C full statistics (co-activation, split-half, effect sizes), WS-D
visualiser, WS-E packaging.
