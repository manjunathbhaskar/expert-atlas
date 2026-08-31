# Entropy-triggered adaptive boost: the remaining MECHANISM.md causal variant

Follow-up to `docs/MECHANISM_CAUSAL.md` (fixed-window boost: FAILED at both
magnitudes), testing the policy `docs/MECHANISM.md` actually proposed:
boost the needle-affine experts only where router entropy spikes.

## Caveats, before any result

- Same scope limits as docs/MECHANISM_CAUSAL.md: one model, one seed, one
  task design, n=12 prompts per subset, teacher-forced forced choice.
- Trigger: per-layer p90 entropy threshold calibrated on
  dev-bucket prompts (accuracy-blind, bucket never evaluated); magnitude is
  the SMALL delta (0.93) carried over from the fixed test,
  not re-tuned. Whole-sequence scope: the trigger decides where to act.
- This steers routing, not compute; `weights_from='biased'` semantics only.
- Needle-affine set: 78 experts,
  re-derived from THIS box's regenerated hard traces by the same WS1
  procedure. docs/MECHANISM.md's original run reported 75; BF16
  cross-machine drift moves a few borderline experts across the lift
  threshold. The set used matches the substrate the baselines below
  reproduce exactly.

Baseline reproduction: max |forced_choice_prob - stored| = 0.00e+00

## Manipulation checks

| subset | condition | trigger rate | needle_affinity_rate | delta vs baseline |
|---|---|---|---|---|
| high_affinity | baseline | — | 0.0444 | — |
| high_affinity | adaptive_needle | 0.1850 | 0.0673 | +0.0229 |
| high_affinity | adaptive_random | 0.1866 | 0.0428 | -0.0016 |
| low_affinity | baseline | — | 0.0268 | — |
| low_affinity | adaptive_needle | 0.1729 | 0.0587 | +0.0320 |
| low_affinity | adaptive_random | 0.1767 | 0.0262 | -0.0006 |

## Results: accuracy

| subset | condition | answer_prob | delta | dz | perm p | FDR sig | passes effect size | accuracy |
|---|---|---|---|---|---|---|---|---|
| high_affinity | baseline | 0.9946 | — | — | — | — | — | 1.000 |
| high_affinity | adaptive_needle | 0.9938 | -0.0007 | -0.33 | 0.3037 | False | False | 1.000 |
| high_affinity | adaptive_random | 0.9936 | -0.0010 | -0.44 | 0.1599 | False | False | 1.000 |
| low_affinity | baseline | 0.4872 | — | — | — | — | — | 0.500 |
| low_affinity | adaptive_needle | 0.5152 | +0.0280 | +0.25 | 0.4009 | False | False | 0.667 |
| low_affinity | adaptive_random | 0.4996 | +0.0124 | +0.25 | 0.3999 | False | False | 0.583 |

## Verdict: NOT SUPPORTED

The manipulation worked and the outcome did not move:

- **The trigger is real and selective**: it fired on 17–19% of (layer, token)
  cells (vs the 10% dev-bucket calibration rate — long degraded prompts are
  higher-entropy, as MECHANISM.md predicted), and the needle boost roughly
  doubled needle-affine selection in the needle window (+0.032 on
  low_affinity) while the random control left it unchanged (-0.0006).
- **Accuracy did not recover**: the treatment's +0.028 answer_prob on the
  target subset is not significant (perm p=0.40, dz=+0.25, no FDR survivor,
  fails the |dz|>=0.8 floor) and is not distinguishable from the random-boost
  control (+0.012). On the high-affinity subset both interventions are inert,
  as expected.
- Combined with docs/MECHANISM_CAUSAL.md (fixed-window, both magnitudes:
  FAILED), this closes out the intervention family MECHANISM.md proposed:
  at these magnitudes, pushing the router back onto the needle-affine pathway
  — always-on or entropy-triggered — does not repair long-context accuracy.
  The pathway loss looks like a symptom of an upstream failure (attention or
  residual-stream state), not a routing-local cause that can be reversed at
  the gate.
- Not tested and still open: larger n, other magnitudes between small and
  large, other trigger designs, intervening on attention rather than routing,
  other models.
