# Span-Free Attention Boost: how much of the oracle repair survives without the label?

**Verdict up front: partial.** The best label-free detector (the identified
retrieval heads' own attention) recovers **55.3% of the oracle effect**,
repairs **8/14** of the baseline-failing prompts (oracle: 14/14), and beats
the wrong-span control with p<=0.02 — but its effect sizes (dz 0.53 full set,
0.71 failing subset) fall **below this project's |dz|>=0.8 floor**, so by the
registered bar this is *not yet* a deployable fix. The other two detectors
failed outright. Span discovery, not attention boosting, is the hard
remaining problem.

## Why this test exists

`docs/ATTENTION_BOOST_CAUSAL.md` proved the mechanism with an oracle:
boosting the 16 identified retrieval heads' attention onto the needle's
*labeled* token span repaired 14/14 failing 3840-token prompts
(answer prob 0.238 -> 0.986). A real question never comes with that label.
This test derives the boost target from signals the model produces on its
own, and measures how much of the effect survives without cheating.

## Registered design (declared before evaluation)

Script: `scripts/run_spanfree_boost.py`. Output: `data/spanfree_boost.json`.

Three detectors, each computed from ONE unlabeled forward pass, candidate
windows over positions [1, question_start) (position 0 excluded as the known
attention-sink; the question's own location is part of the query, not a
label):

1. **heads** — total final-position attention of the 16 identified retrieval
   heads per position; candidate = argmax sliding-window sum.
2. **experts** — per-token fraction of the 128 routing draws (16 layers x
   top-8) landing in the needle-affine expert set (defined on the HARD probe
   set's short bucket — an independent substrate); argmax window mean.
3. **resid** — cosine similarity between the mean layer-8 hidden state over
   the question window and each position's layer-8 hidden state; argmax
   window mean.

Window width per detector chosen on the 1024-token DEV bucket only (first 16
prompt_ids), by span hit rate, grid {8, 12, 16, 24}. Boost identical to the
oracle test: same 16 heads, same beta*=4.0, queries = question span through
final position, keys = the DETECTED span.

**Nulls, stated first:** (1) a detector's boost does no better than baseline;
(2) — the decisive one — it does no better than a strength-matched
**wrong-span control** (width-16 span drawn uniformly away from the true
needle, same heads, same beta). The random-head result already showed
boosting almost anything helps a little, so beating the plain baseline is not
evidence of span *discovery*. Bar per contrast: paired sign-flip permutation
(2000) p<0.05 AND |dz|>=0.8.

## Dev calibration (span hit rate, 1024 bucket, n=16)

| detector | w=8 | w=12 | w=16 | w=24 | w* |
|---|---|---|---|---|---|
| heads | **0.938** | 0.938 | 0.938 | 0.938 | 8 |
| experts | 0.000 | 0.000 | 0.000 | 0.000 | 8 |
| resid | **0.688** | 0.500 | 0.500 | 0.500 | 8 |

The experts detector failed already at dev: the needle-affine expert
activation signal does not localize the needle at all (0/16 windows overlap
it, at any width). It was still evaluated as registered.

## Held-out results (3840 bucket, n=64; oracle from the prior run)

| condition | span hit | acc | mean prob | acc (14 failing) | prob (14 failing) |
|---|---|---|---|---|---|
| baseline (floor) | — | 0.781 | 0.678 | 0/14 | 0.238 |
| **oracle (ceiling)** | 1.000 | **1.000** | **0.981** | **14/14** | **0.986** |
| wrong-span control | 0.000 | 0.719 | 0.668 | 0/14 | 0.241 |
| **heads detector** | **0.859** | **0.859** | **0.846** | **8/14** | **0.569** |
| experts detector | 0.000 | 0.766 | 0.673 | 0/14 | 0.235 |
| resid detector | 0.516 | 0.734 | 0.625 | 2/14 | 0.198 |

Paired statistics (answer probability, 2000 sign-flip permutations):

| contrast | mean delta | dz | perm p | passes bar? |
|---|---|---|---|---|
| heads vs baseline (n=64) | +0.168 | +0.499 | <0.0005 | p yes, dz **no** |
| heads vs wrong-span (n=64) | +0.178 | +0.527 | <0.0005 | p yes, dz **no** |
| heads vs wrong-span, failing subset (n=14) | +0.329 | +0.709 | 0.020 | p yes, dz **no** |
| experts vs wrong-span (n=64) | +0.006 | +0.068 | 0.559 | no |
| resid vs baseline (n=64) | -0.054 | -0.549 | <0.0005 | harmful |

Percent of the oracle effect (mean-prob gain over baseline):
**heads 55.3%**, experts -1.6%, resid -17.7%.

## What this rules in and out

* **The heads' own attention is a real span signal.** It finds the needle on
  86% of held-out long prompts, and the resulting boost beats the wrong-span
  control specifically (p<=0.02 on both the full set and the failing subset)
  — so the gain is span discovery, not generic perturbation. But the effect
  sizes miss the registered floor, and 6/14 failing prompts stay broken.
* **The failing-subset outcome is a perfect dichotomy on span detection**:
  on the 14 failing prompts, every one of the 8 where the detector found
  the span was fully repaired (answer prob >= 0.984), and every one of the
  6 where it missed stayed fully broken (answer prob <= 0.009). Conditional
  on finding the span, the repair is total; the entire residual failure is
  span discovery on prompts where the detector's own signal — the broken
  pathway's attention — has itself collapsed. That is the honest
  circularity limit of using the failing mechanism as its own label.
* **Boosting the wrong span at the identified heads repairs nothing**
  (0/14, acc 0.719 vs baseline 0.781 — slightly harmful). Contrast with the
  oracle test's random-HEAD control (8/14): pushing generic attention around
  can help, but pushing the *retrieval heads specifically* onto the wrong
  place does not. The mechanism is span-specific, which strengthens the
  causal story and raises the bar for any span-free method.
* **The needle-affine expert signal carries no positional information**
  (0% hit at dev and eval). Router-side signals do not localize the fact.
* **The resid-similarity detector is net harmful**: at a 52% hit rate, its
  misses (boosting a wrong span) cost more than its hits gain.

## Honest bottom line

The mechanism transfers partially to the label-free setting: 55% of the
oracle effect, 8/14 repairs, statistically real span discovery — but below
the project's own effect-size floor and far from the 14/14 ceiling. This is
**not yet a deployable fix**. The measured gap says the remaining problem is
span discovery on the prompts where the retrieval heads have collapsed —
the detector and the failure share a cause, so an independent signal (or the
targeted-training escalation already flagged) is what would close it.

## Follow-up: the depth sweep's early-needle failures (worst baseline in the project)

`scripts/run_spanfree_depth.py` applied the fully frozen repair (same 16
cells, beta 4.0, width 8 — no new calibration) to the 16 depth-0.15
3840-token prompts from `docs/CONTEXT_DEPTH.md` (baseline acc 0.375).
Nulls first: (1) the oracle boost does not transfer off the substrate the
heads were identified on; (2) the span-free variant beats nothing.

| condition | span hit | acc | mean prob |
|---|---|---|---|
| baseline | — | 0.375 | 0.207 |
| **oracle boost** | 1.000 | **1.000** | **0.959** |
| heads detector | 0.375 | 0.375 | 0.369 |
| wrong-span control | 0.000 | 0.438 | 0.214 |

Oracle vs baseline: delta +0.752, dz=4.03, p<0.0005; vs wrong-span
dz=3.85, p<0.0005 — **the oracle repair transfers completely to the
hardest substrate**, clearing the bar by a wide margin. Null (1) rejected.

The span-free variant does NOT transfer: detection collapses to 37.5% hit
rate at depth 0.15 (from 86% at depth 0.5), and its aggregate effect is not
significant (dz=0.35, p=0.19). Null (2) stands. The per-prompt dichotomy
repeats exactly: all 6 detector hits reach prob >=0.88, all 10 misses stay
near zero. The harder the retrieval, the more the label-free signal — the
failing pathway's own attention — has collapsed, which is the same
circularity measured above, now at its extreme.

## Limitations

* One model (OLMoE), one substrate (repaired forced-choice probe), n=14
  failing prompts.
* The heads detector reuses the 16 cells identified on this same substrate's
  short bucket (label-free at inference, but substrate-tuned).
* Wrong-span control is width 16; detected spans are width 8 (dev-chosen).
* beta fixed at the oracle test's dev-calibrated 4.0; no re-calibration for
  the span-free setting.
