# Which pathway degrades with context length: generalist or specialist?

## Limits (read first)

1. **One model (OLMoE-1B-7B-0924), one seed, one task design.** Same scope
   limits as `docs/CONTEXT_ROT_HARD.md`.
2. **The substrate verdict carries over**: forced-choice accuracy declines
   but below the preregistered effect-size bar, so every routing trend here
   is descriptive of routing itself; the accuracy link is the per-prompt
   partial correlation, which is correlational, not causal.
3. **`load_ratio` and lift come from the same count matrix** (WS-3's stated
   caveat): hot/cold vs specialist are partly definitionally opposed.
4. **The affine set is defined at the shortest bucket** on this run's own
   traces (192 prompts), so it inherits that pipeline's assumptions.
5. BF16 CPU regeneration of the traces drifts slightly from the committed
   run (see RESEARCH_LOG entry 8); all numbers here are from one internally
   consistent regeneration.

## Expert-set partition (disjoint)

| set | # experts | of which hot | of which cold |
|---|---|---|---|
| affine (specialist pathway) | 78 | 5 | 32 |
| hot_nonaffine | 95 | - | - |
| cold_nonaffine | 276 | - | - |
| mid_nonaffine | 575 | - | - |

## Length trends (permutation null, BH-FDR across this family, effect-size floor)

| share | window | short | long | delta | rho | perm p | FDR | d | verdict |
|---|---|---|---|---|---|---|---|---|---|
| affine | needle | 0.0417 | 0.0335 | -0.0082 | -0.402 | 0.0005 | yes | -1.39 | **SIGNIFICANT-BUT-NON-MONOTONE** |
| affine | question | 0.0057 | 0.0082 | +0.0024 | +0.473 | 0.0005 | yes | +1.30 | **SIGNIFICANT-BUT-NON-MONOTONE** |
| hot_nonaffine | needle | 0.1120 | 0.1407 | +0.0287 | +0.427 | 0.0005 | yes | +1.33 | **SIGNIFICANT-BUT-NON-MONOTONE** |
| hot_nonaffine | question | 0.1401 | 0.1484 | +0.0083 | +0.302 | 0.0005 | yes | +0.95 | **SIGNIFICANT-BUT-NON-MONOTONE** |
| cold_nonaffine | needle | 0.2335 | 0.2162 | -0.0173 | -0.284 | 0.0005 | yes | -0.68 | **SIGNIFICANT-BUT-TRIVIAL** |
| cold_nonaffine | question | 0.1939 | 0.1808 | -0.0131 | -0.286 | 0.0005 | yes | -0.99 | **SIGNIFICANT-BUT-NON-MONOTONE** |
| mid_nonaffine | needle | 0.6128 | 0.6097 | -0.0032 | -0.089 | 0.2259 | no | -0.23 | **FLAT** |
| mid_nonaffine | question | 0.6603 | 0.6627 | +0.0023 | -0.000 | 0.9970 | no | +0.23 | **FLAT** |

## Does each pathway's share predict accuracy, independent of length?

Partial Spearman vs `answer_prob`, controlling log2(length) by
rank-residual regression (same procedure as `docs/MECHANISM.md`),
permutation p (2000 shuffles, two-sided).

| share | window | raw rho | partial rho | perm p |
|---|---|---|---|---|
| affine | needle | +0.631 | +0.651 | 0.0005 |
| affine | question | +0.222 | +0.303 | 0.0005 |
| hot_nonaffine | needle | +0.029 | +0.076 | 0.2824 |
| hot_nonaffine | question | +0.125 | +0.161 | 0.0245 |
| cold_nonaffine | needle | -0.002 | -0.030 | 0.6767 |
| cold_nonaffine | question | +0.007 | -0.021 | 0.7836 |
| mid_nonaffine | needle | -0.393 | -0.404 | 0.0005 |
| mid_nonaffine | question | -0.210 | -0.211 | 0.0025 |

## Reading the two tables together

The length trends say where routing mass MOVES as context grows; the
partial correlations say which of those movements tracks correctness.
A pathway is implicated in degradation only if it shows both: a length
trend AND a length-independent association with `answer_prob`. A length
trend alone (like `entropy_all` in `docs/MECHANISM.md`) is movement that
explains nothing; a partial correlation alone is prompt-to-prompt
variation that length does not touch. Numbers above are the evidence;
this section only states which cells clear both requirements.

| share | window | length trend? | predicts accuracy (partial)? | implicated |
|---|---|---|---|---|
| affine | needle | yes | yes | **YES** |
| affine | question | yes | yes | **YES** |
| hot_nonaffine | needle | yes | no | no |
| hot_nonaffine | question | yes | no | no |
| cold_nonaffine | needle | no | no | no |
| cold_nonaffine | question | yes | no | no |
| mid_nonaffine | needle | no | yes | no |
| mid_nonaffine | question | no | no | no |

(`predicts accuracy` bar: perm p < 0.05 AND |partial rho| >= 0.3 -- the
same practical-effect floor spirit as everywhere else in this repo.)

