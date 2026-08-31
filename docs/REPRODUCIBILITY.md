# Reproducibility

Everything a stranger needs beyond the README to rerun the results.

## Models

| Model | HF id | Facts (verified) |
|---|---|---|
| OLMoE | `allenai/OLMoE-1B-7B-0924` | 16 layers × 64 experts, top-8 routing, 16 attention heads, `max_position_embeddings=4096`, **`norm_topk_prob=False`** (top-k gate weights do NOT sum to 1 — they sum to the top-k probability mass; never assert `== 1.0`) |
| Granite | `ibm-granite/granite-3.0-3b-a800m-base` | 32 layers × 40 experts, top-8 routing, grouped-query attention (24 query / 8 KV heads), max 4096 |

Weights are cached under `data/hf_cache/` (NOT the default `~/.cache/huggingface`
layout). Always run model scripts as:

```bash
HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 .venv/bin/python scripts/<script>.py
```

`HF_HUB_OFFLINE=1` makes a cache-path mistake fail loudly instead of silently
re-downloading ~14 GB.

## Hardware assumptions

- Everything was run on CPU (no GPU required). OLMoE full-model forward passes
  need ~13 GiB RAM (or use the on-demand runtime, `expertatlas/ondemand.py`,
  at ~0.7 GiB).
- Long runs (Phase 3 capture ~13 h; the boost evaluations 30–90 min each) are
  resumable via per-item manifests/JSONL — rerunning a script skips completed
  work.
- Do not run two full-model CPU jobs concurrently; they thrash.

## Determinism and seeds

- All forward passes are teacher-forced, single-pass, greedy — deterministic
  given the same library versions.
- Attention capture uses `attn_implementation="eager"` (SDPA does not return
  attention weights); eager vs SDPA outputs were verified equivalent for the
  measured quantities.
- All sampling (subsampling, permutation nulls, random-head/wrong-span
  controls) is seeded; defaults are seed 0 unless a script documents
  otherwise. Permutation nulls use ≥200 permutations (2000 for the head
  specificity nulls).
- Phase 3 statistical constants: Laplace smoothing 1.0, BH-FDR q=0.05, effect
  floor |lift| ≥ 1.0, H6 gate ρ ≥ 0.5.
- Preregistered intervention bar (all boost/repair claims): paired sign-flip
  permutation p < 0.05 AND |dz| ≥ 0.8 against the matched control, registered
  before running. Calibration (e.g. the boost strength β) uses the dev bucket
  only and is frozen before held-out evaluation.
- BF16 batch-shape nondeterminism exists at the third decimal of logits when
  batching varies (see `notes/RESEARCH_LOG.md` entry 8); all comparisons are
  made within a single internally consistent run.

## Software

- Python ≥ 3.11 (developed on 3.13), dependencies pinned by floor in
  `pyproject.toml` (`torch>=2.4`, `transformers>=4.44`, ...).
- Install: `uv pip install --python .venv/bin/python -e ".[dev]"`.
- The transformers OLMoE facts the code depends on (router returns
  scores/indices, no per-expert FFN submodule) are documented in
  `docs/METHOD.md` §12 and guarded by tests.

## Non-obvious requirements

- The 231-test fast suite excludes `tests/sanity/test_model_loading.py`
  (downloads the full checkpoint).
- Probe generators (`probes/generate_context_probes*.py`) are deterministic;
  generated probe JSONL files are committed where small.
- The memory-constraint benchmark (`docs/ONDEMAND.md`) uses
  `systemd-run --scope -p MemoryMax=2G`; requires systemd and (on some hosts)
  sudo.
- `viz/atlas.html` is fully offline/no-CDN; open it directly in a browser.
