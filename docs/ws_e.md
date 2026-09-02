# WS-E — Packaging, Notebook, Writeup

**Owner:** WS-E (this session)
**Owns exclusively:** `notebooks/expert_atlas_colab.ipynb`, `docs/METHOD.md`, `docs/ws_e.md`

---

## Deliverables completed

| File | Purpose |
|------|---------|
| `docs/METHOD.md` | Full statistical method so a stranger can reproduce every number in `docs/FINDINGS.md` and `docs/ORTHOGONALITY.md`. Covers: lift formula and why base-rate correction is essential (MoE load balancing makes raw heat near-meaningless); equal-token-budget subsampling and why (prompt length varies 1.73× across topics); Benjamini–Hochberg FDR across ~10k simultaneous tests; effect-size requirement (|lift|≥1.0) and why it was necessary (TRANSFER.md §11: first pass reported 70%+ cells significant with median lift 0.79×); split-half replication (H6, ρ=0.667); orthogonality method (per-domain lift vectors, pairwise cosine, label-shuffle null); ablation method including both OLMoE gotchas (no per-expert FFN submodule; OlmoeTopKRouter.forward does softmax+topk internally, hooking logits is a silent no-op — must zero the SCORE at selected top-k slots). |
| `notebooks/expert_atlas_colab.ipynb` | Self-contained Colab demonstration: pip installs deps, loads OLMoE-1B-7B-0924, captures ~40 prompts (subset of 480), computes lift matrices with equal-token-budget subsampling, runs chi-squared + BH-FDR + effect-size filter, reports H6 split-half replication and H1 per-expert meaningful affinity, renders inline 3-D UMAP atlas coloured by max-lift domain. **Explicitly states in a markdown cell that it is a method-demonstration on a reduced subset and will NOT reproduce `docs/FINDINGS.md`'s exact numbers.** |
| `docs/ws_e.md` | This file. |

---

## What was NOT done / could not be verified

- **The Colab notebook has NOT been executed end-to-end on a T4 (or any GPU) to verify the ~10–15 minute runtime claim.** The runtime estimate is based on: model load ~30s, 40 prompts × ~15s/prompt on T4 ≈ 10 min, plus analysis/plotting overhead. This was not measured. A future session should run the notebook on a free Colab T4 and update the runtime note with actuals.

- **The notebook's GitHub-raw fetch URL for `probe_set_v1.yaml` is a placeholder (`<your-repo>`).** The repository is not public at the time of writing. A future session must replace this with the actual raw GitHub URL once the repo is published, or bundle the probe set into the notebook as a literal.

- **No GitHub Pages deployment of the visualiser (`viz/atlas.html`)** — this was scoped to Phase 4 in PLAN.md and remains undone.

- **No Zenodo DOI for `atlas.json`** — also Phase 4.

- **No coordination with Colibrì (#175)** — the method-offer step in PLAN.md §10 remains undone.

---

## Known issues in the notebook (for the next session to fix)

1. **Probe set fetch URL** — replace `<your-repo>` with actual GitHub raw URL once public.
2. **No error handling** for missing probe set / network failure — should fall back gracefully or give a clear instruction.
3. **The demo subset selection** (4 prompts per topic from split=A) is hardcoded; could be made configurable.
4. **UMAP import** — if `umap-learn` fails to install on Colab (numba/llvmlite issues), the notebook falls back to PCA silently. Could add a visible notice.
5. **No inline display of the visualiser HTML** — the notebook only shows a matplotlib 3-D scatter. Embedding the real `viz/atlas.html` via `IPython.display.HTML` would be nicer but requires the real `atlas.json` (which the demo doesn't produce).

---

## Honest framing

WS-E's task was **documentation and packaging only** — the science (capture, statistics, orthogonality, ablation) was completed by prior sessions and is reflected in `docs/FINDINGS.md`, `docs/ORTHOGONALITY.md`, `docs/ABLATION.md`, and `docs/TRANSFER.md` §11. This session did not run any new analyses, did not touch the capture engine, statistics, or visualiser code, and did not generate new data.

The two artefacts produced (`METHOD.md`, `expert_atlas_colab.ipynb`) are **faithful transcriptions of the existing pipeline** — they describe what the code *already does*, verified by reading the source (`expertatlas/stats.py`, `expertatlas/aggregate.py`, `expertatlas/capture.py`, `scripts/run_ablation_harness.py`, `scripts/run_orthogonality_analysis.py`, `expertatlas/layout.py`). No new methodology was invented here.

**If you are the next session:** run the notebook on a real T4, measure actual time, fix the probe-set URL, and consider embedding the real visualiser once `data/atlas.json` exists from the full run.