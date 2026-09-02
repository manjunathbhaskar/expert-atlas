# ANALYSIS_GUARDRAILS.md

## Read first

`PLAN.md` is the spec. `docs/TRANSFER.md` is the running session log — **read §11**,
which documents two analysis errors that were caught before being reported, and which
you can easily repeat. `docs/FINDINGS.md` has the real results.

## Current state (Phase 3 complete, real run)

Real capture done: **480/480 prompts, 662,048 rows, 41,378 distinct tokens,
5,296,384 expert selections** on `allenai/OLMoE-1B-7B-0924` (16 layers x 64 experts,
top-8). ~12.9h on CPU. Traces in `data/traces/`, atlas in `data/atlas.json`.

| result | verdict |
|---|---|
| H6 split-half replication (the gate) | **PASS**, rho = 0.667 |
| H1 domain affinity | **PASS**, 557/1024 experts (FDR sig AND \|lift\|>=1.0) |
| H3 factor separability | topic 22% / lang 5% / register 0.6% / **format 0%** meaningful |
| H4 co-activation communities | **UNRELIABLE** — usage skew 227x vs PMI's 2.0x validity limit |
| Orthogonality | domains overlap MORE than chance (z=180.6), code-ish cluster 0.74–0.90 |
| Ablation (medicine/cooking) | double dissociation, directional only — n=6, no null |
| Utilization / hot experts | **hypothesis REFUTED** — specialists are *cold*, enrichment 0.62x, p<1e-4 |

Verify: `pytest tests/ws_b tests/ws_c tests/ws_d tests/ws_util -q` (66 fast tests).
`tests/sanity/test_model_loading.py` downloads the real 14GB checkpoint — slow, exclude
it for quick runs.

## Non-negotiables

1. **Lift, not heat.** Primary metric is `log2 P(e|d)/P(e)`, never raw counts. Load
   balancing makes raw usage near-uniform *by design*, so heat carries almost no
   information. Publishing heat as affinity is the fastest way to be confidently wrong.
2. **Significance AND effect size.** Every headline number needs BH-FDR **and**
   `|lift| >= 1.0`. The first analysis pass reported "70.7% of cells significant" —
   median lift among them was **0.79x**, i.e. no effect. See TRANSFER.md §11.
   **Always check percentiles, never just the mean, when n is large.**
3. **Every claim needs a null.** Permutation/label-shuffle, >=200 perms
   (`stats.py::shuffle_labels`). A raw overlap count is not a finding.
4. **Respect the tools' own validity gates.** `coactivation.py` warns above 2.0x usage
   skew; this run is 227x, so PMI-based results stay untrusted *regardless of which way
   they come out*. `docs/FINDINGS.md` reports H4 as UNRELIABLE against the project's
   own interest — match that.
5. **`norm_topk_prob=False` on OLMoE** — top-k gate weights do NOT sum to 1, they sum
   to the top-k probability mass. Never assert `== 1.0`. Read the config.
6. **Negative results are results.** Do not tune an analysis until it looks exciting.
   WS3's commissioning brief predicted enrichment; the data showed depletion, and that
   is what the doc says.

## Workstream file ownership (exclusive — never edit outside your lane)

| WS | Owns |
|----|------|
| A — capture | `expertatlas/capture.py`, `cli.py`, `schemas.py` |
| B — probes | `probes/` |
| C — statistics | `expertatlas/{stats,aggregate,coactivation}.py` |
| D — visualiser | `viz/`, `expertatlas/layout.py` |
| 1 — context rot | `expertatlas/context_metrics.py`, `probes/probe_set_context.*`, `scripts/run_context_*.py`, `docs/CONTEXT_ROT.md`, `tests/ws_ctx/` |
| 2 — interference | `expertatlas/interference.py`, `scripts/run_interference.py`, `scripts/run_ablation_multi.py`, `docs/INTERFERENCE.md`, `tests/ws_int/` |
| 3 — utilization | `expertatlas/utilization.py`, `scripts/run_utilization.py`, `docs/UTILIZATION.md`, `tests/ws_util/` |

Cross-lane changes go in `docs/interface-requests.md` (append-only) for the owner to action.

## Model facts (verified — do not re-derive)

`OlmoeForCausalLM(..., output_router_logits=True)` returns `.router_logits`: a tuple of
one tensor per layer, shape `(batch, seq, n_experts)`. `max_position_embeddings=4096`.
Capture is teacher-forced (single forward pass), greedy, deterministic.

## Model cache (important — disk is at 100%)

The OLMoE weights (13GB, 3 shards) live at `data/hf_cache/models--allenai--OLMoE-1B-7B-0924`,
which is **not** the default `$HF_HOME/hub/...` layout. `~/.cache/huggingface`'s copy has
config+tokenizer but **no weight shards**, so a script that misses this re-downloads 14GB
onto a disk with ~3.6GB free and fails.

Always run model scripts with:

```bash
HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 .venv/bin/python scripts/<script>.py
```

Verified: resolves offline to 16 layers x 64 experts, top-8.
