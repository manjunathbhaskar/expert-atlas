# The mechanism behind context rot on this model

**Question:** WS1 (`docs/CONTEXT_ROT_HARD.md`) established that accuracy declines
with input length (93.8% -> 68.8%, a real, FDR-significant, near-trend-threshold
effect) and that several routing metrics also trend with length. This document
asks the mechanism question directly: **does the routing change explain the
accuracy change, at the level of individual prompts** -- not just "both trend
with length," but "does a prompt's routing behaviour predict whether it gets
the right answer, independent of how long it is?"

**Data:** the 192-prompt hard-variant run (`data/context_rot_hard.json`),
per-prompt Spearman correlation between each routing metric and `answer_prob`
(the graded, softmax-based accuracy measure -- more sensitive than the binary
forced-choice accuracy), both raw and **partial** (controlling for
log2(length) via rank-residual regression, since length correlates with both
routing and accuracy and would otherwise confound the association).

## Result

| metric | raw rho | partial rho (length controlled) | p |
|---|---|---|---|
| `needle_affinity_rate` | +0.617 | **+0.640** | <0.0001 |
| `mass_q` | +0.578 | +0.586 | <0.0001 |
| `entropy_needle` | -0.589 | -0.570 | <0.0001 |
| `entropy_q` | -0.473 | -0.487 | <0.0001 |
| `hot_load_share_q` | +0.097 | +0.126 | 0.082 (ns) |
| `entropy_all` | -0.039 | +0.072 | 0.32 (ns), permutation p=0.325 |

**The key contrast:** `entropy_all` (mean router entropy over the whole prompt)
showed the *largest* length-trend in the original analysis (rho=0.887 vs
length) -- but it does **not** predict accuracy once length is controlled.
`needle_affinity_rate` (the rate at which the router's top-k selections land
on the specific experts affine to the needle's content) has a strong,
highly significant, length-independent correlation with accuracy: **rho=0.64**.

## Interpretation

**Context rot on this model is not global router confusion.** The router's
overall decisiveness (`entropy_all`) genuinely degrades with length -- that
part of the earlier finding stands -- but that degradation, by itself,
doesn't predict which prompts the model gets wrong. What predicts correctness
is much more specific: **whether the router keeps sending the needle's tokens
to the small cluster of experts that are actually affine to that content.**
When it does, accuracy is high regardless of length; when it doesn't, accuracy
drops, and it does this more often as length grows.

This is a more precise (and more actionable) mechanism than "the router gets
confused": it points at losing a specific specialist pathway, not a generic
capacity limit.

## Honest limits

- **Correlational, within one run.** This shows routing behaviour and
  correctness co-vary across prompts; it does not yet show that *forcing*
  `needle_affinity_rate` up would *cause* accuracy to recover. That causal
  step is the natural next experiment (bias the router toward the needle-
  affine experts when entropy spikes, measure whether accuracy recovers) and
  has not been run.
- **One model, one seed, one task design.** Same scope limits as
  `docs/CONTEXT_ROT_HARD.md` §Limits.
- **`needle_affinity_rate` is itself derived from the same lift-based analysis
  as the rest of this project** (which expert set counts as "needle-affine"
  is defined via the earlier significance+effect-size pipeline), so this
  result inherits that pipeline's assumptions rather than being independent
  of them.
