# Causal test of the MECHANISM.md pathway: boosting the needle-affine experts

Follow-up to `docs/MECHANISM.md`, which found that `needle_affinity_rate` predicts per-prompt correctness independent of length (partial rho=+0.64, p<0.0001) while raw router entropy does not, and which named this experiment as its own untested next step. Nothing in `docs/MECHANISM.md` or `docs/CONTEXT_ROT_HARD.md` is modified by this document.

## Caveats, before any result

- **This steers routing, not compute.** `expertatlas/steering.py` adds `+delta` to the needle-affine experts' router logits before softmax+top-k. Nothing is skipped or unloaded; this is not a speed or memory claim.
- **A boost changes two things at once**: which experts are selected, and (because the biased softmax puts more mass on them) what gate weight they receive. Runs here use the default `weights_from="biased"`, so a positive result does not attribute itself between the two. `steering.py` supports isolating the selection change; that variant has **not** been run.
- **Small n, one model, one seed, one task design** -- same scope limits as `docs/CONTEXT_ROT_HARD.md` §Limits. n=12 prompts per subset.
- **The needle-affine set is the same lift-based set as MECHANISM.md's** (re-derived at the shortest bucket and checked to be the same size, 75/1024, before running), so this inherits that pipeline's assumptions rather than testing them.

## Method

- Substrate: the 192-prompt hard variant. Evaluated buckets: [2048, 3072, 3840] (the degraded end of the length sweep). Dev bucket for calibration: 1024, never evaluated.
- Intervention window: the needle's own token span only (`needle_char_span` -> tokens via the capture-time offset mapping), ~12 tokens.
- Boost magnitudes, chosen **blind to accuracy** on the dev bucket by a routing-side criterion (median logit gap from the top-k threshold to the near-miss needle-affine experts = 0.930): small=0.93, large=3.72. Both are reported below regardless of which looks better.
- Subsets are **length-matched by construction**: equal prompts per length bucket in each of `low_affinity` (the intervention's target, where the pathway is weakest) and `high_affinity` (where it was already intact).
- `random_boost_*` is the analogue of `ablate_random` in `docs/ABLATION.md`: the same number of experts **per layer**, drawn from the non-affine complement, same magnitude, same window, redrawn per prompt from a per-prompt seed.
- Scoring: `run_context_sweep.score_answer` (imported, not reimplemented) -- forced choice over the same 8 single-token candidates.

## Reproduction check (baseline condition vs the stored hard-variant run)

- max |answer_prob difference| = 0.00e+00
- max |needle_affinity_rate difference| = 0.00e+00

A non-trivial difference here would mean this harness is not measuring the same thing `docs/MECHANISM.md` measured, and nothing below would be comparable to it.

## Manipulation check: did the boost actually move routing?

| subset | condition | needle_affinity_rate (needle window) | mean delta vs baseline | dz | perm p |
|---|---|---|---|---|---|
| high_affinity | baseline | 0.0450 | — | — | — |
| high_affinity | needle_boost_large | 0.5777 | +0.5326 | +225.96 | 0.0006 |
| high_affinity | needle_boost_small | 0.2753 | +0.2303 | +17.61 | 0.0006 |
| high_affinity | random_boost_large | 0.0202 | -0.0248 | -4.62 | 0.0006 |
| high_affinity | random_boost_small | 0.0274 | -0.0176 | -4.75 | 0.0006 |
| low_affinity | baseline | 0.0295 | — | — | — |
| low_affinity | needle_boost_large | 0.5775 | +0.5480 | +269.52 | 0.0006 |
| low_affinity | needle_boost_small | 0.2974 | +0.2679 | +16.47 | 0.0006 |
| low_affinity | random_boost_large | 0.0174 | -0.0121 | -2.43 | 0.0006 |
| low_affinity | random_boost_small | 0.0172 | -0.0123 | -4.50 | 0.0006 |

## Results: accuracy

Both bars, side by side, per this project's rules: a paired sign-flip permutation p (BH-FDR across the whole accuracy family, q=0.05) AND a paired effect size (|dz| >= 0.8).

| subset | condition | answer_prob | delta | dz | perm p | FDR sig | passes effect size | accuracy |
|---|---|---|---|---|---|---|---|---|
| high_affinity | baseline | 0.9907 | — | — | — | — | — | 1.000 |
| high_affinity | needle_boost_large | 0.1349 | -0.8557 | -7.88 | 0.0006 | True | True | 0.083 |
| high_affinity | needle_boost_small | 0.9864 | -0.0042 | -0.50 | 0.1134 | False | False | 1.000 |
| high_affinity | random_boost_large | 0.1274 | -0.8633 | -7.46 | 0.0006 | True | True | 0.083 |
| high_affinity | random_boost_small | 0.9887 | -0.0020 | -0.17 | 0.5363 | False | False | 1.000 |
| low_affinity | baseline | 0.5055 | — | — | — | — | — | 0.583 |
| low_affinity | needle_boost_large | 0.0114 | -0.4941 | -1.67 | 0.0006 | True | True | 0.000 |
| low_affinity | needle_boost_small | 0.4145 | -0.0910 | -0.51 | 0.1071 | False | False | 0.583 |
| low_affinity | random_boost_large | 0.0101 | -0.4954 | -1.68 | 0.0006 | True | True | 0.000 |
| low_affinity | random_boost_small | 0.3961 | -0.1094 | -0.91 | 0.0132 | True | True | 0.417 |

## Verdict: the four-number bar

`docs/ABLATION.md` requires a causal claim to beat its controls, not just to look good in the treated cell. The analogous bar here:

**needle_boost_large** (mean answer_prob on low_affinity: 0.0114, baseline 0.5055):
- `needle_boost_large` improves answer_prob on low_affinity prompts: -0.4941 — NO
- ...and clears FDR significance AND the effect-size floor: p=0.0006, dz=-1.67 — YES
- ...and beats `random_boost_large` on the same prompts: -0.4941 vs -0.4954 — YES
- ...and is selective: helps low_affinity more than high_affinity: -0.4941 vs -0.8557 — YES

**needle_boost_small** (mean answer_prob on low_affinity: 0.4145, baseline 0.5055):
- `needle_boost_small` improves answer_prob on low_affinity prompts: -0.0910 — NO
- ...and clears FDR significance AND the effect-size floor: p=0.1071, dz=-0.51 — NO
- ...and beats `random_boost_small` on the same prompts: -0.0910 vs -0.1094 — YES
- ...and is selective: helps low_affinity more than high_affinity: -0.0910 vs -0.0042 — NO

**Causal claim NOT SUPPORTED at any tested boost magnitude on this run.**

Every clause above is load-bearing: the treatment beating baseline is not enough, because a random same-size boost also perturbs routing, and an intervention that helps the already-intact `high_affinity` prompts just as much is not evidence about the needle pathway specifically.

## Honest limits

- n=12 prompts per subset, one model, one seed, one task design, one needle-affine set. Directional.
- The manipulation itself is coarse: a fixed `+delta` on every needle-affine expert at every needle token, not an entropy-triggered or per-token-adaptive policy. `docs/MECHANISM.md` phrased its next step as boosting *when entropy spikes*; the always-on version tested here is the simpler, more conservative variant, and a negative result for it does not rule out the triggered one.
- `weights_from="biased"` confounds the selection change with gate-weight inflation (see caveats).
- Only the needle window is intervened on. A boost applied at the question window, or across the whole prompt, is a different experiment and was not run.
- No second domain / second task. `PLAN.md` §9b's generalisation standard is not met by this document alone.

Raw per-cell records: `data/mechanism_causal/records.jsonl`. Calibration: `data/mechanism_causal/calibration.json`.
