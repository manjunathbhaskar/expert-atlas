# Public Release Checklist (v1.0)

## Claims vs controls

- [x] Degradation curve (0.938 → 0.688) reported with its own miss: d = −0.67
      vs the preregistered |d| ≥ 0.8 floor (README, METHOD.md §15).
- [x] Routing correlate reported as correlational; three failed router/residual
      interventions preserved (ADAPTIVE_CAUSAL.md, ANCHOR_CAUSAL.md).
- [x] Storage-vs-transport probes cross-paired (no entity shortcut).
- [x] Head identification on short correct prompts only; collapse test has a
      2000-draw random-head specificity null (p < 0.0005).
- [x] Boost claims each have wrong-span, random-head, no-boost, and
      oracle-ceiling anchors, with the p < 0.05 AND |dz| ≥ 0.8 bar registered
      before running; β frozen on dev.
- [x] Granite replication caveat visible: failing subset n = 5, subset p floor
      0.0625; full-set inference stated.
- [x] Full-set identified-vs-random dz = 0.72 miss (ceiling effect) disclosed.

## Negatives and scope limits visible

- [x] Detector negatives preserved: expert-activation (0%), residual-cosine
      (harmful), lexical v1 (1.3%), attention detector circularity (~55%).
- [x] Multi-hop lexical failure (0% hit, bridge sentence) and the L8 fallback's
      labeled-dev-set dependence stated.
- [x] Paraphrase/multi-hop capability floor framing (assisted capability, not
      recovered rot).
- [x] H4 co-activation UNRELIABLE (usage skew 227× vs 2× gate).
- [x] Model/context/task scope limits in README "Scope and limitations".

## Code runs / reproducible

- [x] Full fast test suite green (see final run recorded below).
- [x] README gives exact setup + per-result commands.
- [x] docs/REPRODUCIBILITY.md: models, seeds, hardware, cache layout,
      determinism notes.
- [x] All scripts resumable; all randomness seeded.
- [x] Figures regenerate from scripts/make_paper_figures.py and
      scripts/make_architecture_figure.py.

## Repo hygiene

- [x] LICENSE (MIT) present; README links it.
- [x] No agent/tooling-specific names in tracked files.
- [x] logs/ and data/ untracked (results live in docs/, regenerable via scripts).
- [ ] Owner: make repo public, set default branch to the clean release branch,
      delete stale branches, create GitHub release for tag v1.0.

## Final verification

- Fast suite: `pytest tests/ -q --ignore=tests/sanity/test_model_loading.py`
  — 231 passed (recorded at tagging time).
