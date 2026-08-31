# Findings — Expert Atlas (real run)

Run over 480 prompts, 41378 distinct tokens, OLMoE-1B-7B-0924.
Per PLAN.md §0: negatives are reported as prominently as positives. Every number below is filtered through both FDR significance AND a practical-significance bar (|lift| >= 1.0, i.e. >=2x fold change) — raw significance-cell counts are reported too, but flagged as inflated by sample size, not treated as the finding. See docs/TRANSFER.md §4 for why this distinction was necessary: the first pass of this script reported raw significance rates (70%+ of cells) that collapsed to a median lift of ~0.8x (i.e. no real effect) once actually inspected.

## H6 — split-half replication (the project gate, run first)
Pooled Spearman rho = 0.667 (threshold >= 0.5). **PASS**

## H1 — domain affinity beyond chance (topic factor, PLAN.md's actual per-expert definition)
557/1024 experts (54.4%) have at least one topic domain that is both BH-FDR significant (q=0.05) AND clears |lift| >= 1.0 (>=2x fold change vs. base rate). Falsified if <5%. **PASS**

## H3 — factor separability (raw significance vs. practically-meaningful significance)
Raw BH-FDR significance rate is reported alongside the effect-size-filtered rate specifically because they diverge a lot here — the gap itself is informative (large-sample statistical significance without practical effect size).
- topic: 7243/10240 FDR-significant (70.73%) — but only 2270 (22.17%) also clear |lift|>=1.0
- lang: 3030/4096 FDR-significant (73.97%) — but only 215 (5.25%) also clear |lift|>=1.0
- register: 888/2048 FDR-significant (43.36%) — but only 12 (0.59%) also clear |lift|>=1.0
- format: 1197/3072 FDR-significant (38.96%) — but only 0 (0.00%) also clear |lift|>=1.0

## H4 — co-activation communities vs. degree-preserving null
usage_skew=227.49x. coactivation.py's own documented validity limit for PMI is 2.0x (measured: separation from noise collapses by ~10x skew). At 227x here, this result is **known-contaminated and must not be trusted either way**, regardless of which way the raw comparison comes out. Reported for completeness, not as evidence.
- raw modularity=0.9371 vs. null 0.3298 +/- 0.0037 (raw comparison: exceeds null)
- **verdict: UNRELIABLE — usage skew (227x) far exceeds the tool's own 2.0x validity limit.** This is plausibly a real property of inference-time routing on a narrow 480-prompt evaluation set (load balancing is a *training*-time objective over the full training distribution; nothing enforces balance on any specific narrow prompt sample at inference) rather than a capture bug — verified directly against raw trace counts: all 1024 experts fired at least once, and the usage distribution is smooth (no dead experts, no single outlier), consistent with genuine skew rather than a counting error. But that explanation does not make the PMI-based community result trustworthy; it explains why the tool's own validity check correctly refused to trust it.

## Orthogonality (extension, see docs/ORTHOGONALITY.md for full detail)
Tested whether different topics' routing signatures (base-rate-corrected lift vectors,
so immune to the H4 usage-skew problem) are orthogonal to each other, against a
label-shuffle null (200 permutations) — the mechanism from the continual-learning
literature (near-orthogonal task subspaces prevent interference).

**Result: domains overlap MORE than chance, not less.** Observed mean |cosine|=0.272 vs.
null 0.108±0.0009 (z=180.55). This argues against a simple "freeze old experts, add new
ones" continual-learning strategy on this substrate — the same experts are shared across
nominally different topics more than random routing would produce.

**But the overlap is not uniform, and the pattern is coherent, not noisy:**
python/rust/sql/math_proof cluster tightly (pairwise cosine 0.74–0.90), while history sits
apart (negative cosine vs. every code-ish domain, down to -0.25 vs. python). This lines up
with — and gives a second, independent angle on — the H3 finding that `format` shows ~0%
meaningful lift: it's not that json/prose/bulleted structure drives routing, it's that
certain *topics* (the symbol/notation-dense ones) share routing regardless of surface
format, most likely via shared subword-token statistics rather than "syntax" in the
format-field sense tested by H3. Honest limit: this is ROUTING orthogonality only — two
domains routing to the same experts could still produce orthogonal activations inside
them; not tested here.

## Headline
*"We mapped what experts in OLMoE-1B-7B-0924 actually do — base-rate corrected, FDR controlled, and it replicates across held-out prompts (H6 rho=0.67). 54% of experts show a real (>=2x), statistically robust affinity for at least one topic — well above the raw significance-cell rate would suggest at face value, and well above what a naive uncorrected 'percent significant' headline would honestly support. Co-activation community structure could not be assessed reliably on this run due to extreme inference-time usage skew."*
