# Relative dynamic-k on the long-context substrate

## Limitations (read first)

- One model, one seed, teacher-forced forced-choice scoring; n=32 prompts per bucket.
- Thresholds are the pre-registered DYNAMIC_K_RELATIVE.md grid; nothing
  was tuned on these prompts.
- The baseline condition reproduced the stored hard-variant answer_prob
  (max dev reported below); if it had not, nothing here would be
  comparable to docs/CONTEXT_ROT_HARD.md.

Baseline reproduction: max |forced_choice_prob - stored| = 0.00e+00

## Mean kept k by bucket (question 1: does the router demand more experts with length?)

| threshold | bucket 256 | bucket 3840 |
|---|---|---|
| rk0.9 | 6.960 | 7.036 |
| rk0.7 | 4.922 | 5.062 |
| rk0.5 | 3.287 | 3.396 |

## Accuracy cost by bucket (question 2: does truncation hurt long context more?)

| bucket | condition | mean answer_prob | delta vs baseline | dz | perm p | FDR sig | accuracy |
|---|---|---|---|---|---|---|---|
| 256 | baseline | 0.7353 | — | — | — | — | 0.969 |
| 3840 | baseline | 0.6749 | — | — | — | — | 0.688 |
| 256 | rk0.9 | 0.7420 | +0.0067 | +0.13 | 0.5295 | False | 0.938 |
| 256 | rk0.7 | 0.7563 | +0.0210 | +0.21 | 0.2619 | False | 0.969 |
| 256 | rk0.5 | 0.7487 | +0.0134 | +0.05 | 0.7767 | False | 0.812 |
| 3840 | rk0.9 | 0.6829 | +0.0080 | +0.16 | 0.3809 | False | 0.750 |
| 3840 | rk0.7 | 0.6901 | +0.0152 | +0.12 | 0.5244 | False | 0.750 |
| 3840 | rk0.5 | 0.7002 | +0.0253 | +0.13 | 0.4779 | False | 0.750 |

## Findings

- **Question 1 (compute): yes, slightly.** At every threshold the router keeps
  more experts per token at 3840 than at 256 (e.g. rk0.7: 5.06 vs 4.92, ~+2-3%).
  Direction is consistent with docs/MECHANISM.md's finding that routing entropy
  rises with length, but the shift is small; adaptive-k does not blow up at
  long context.
- **Question 2 (quality): no measurable cost, at either length.** Every
  dynamic-k delta vs baseline is small, positive-signed, and not significant
  (all perm p > 0.26, no FDR survivor, all |dz| <= 0.21). The positive signs
  are NOT evidence dynamic-k helps — with n=32 and these effect sizes this is
  noise; the honest claim is "no detectable cost at 3.3-7.0 mean experts on
  this substrate". This differs from the domain-prompt NLL curve
  (docs/DYNAMIC_K_RELATIVE.md, +0.06..+0.23 nats): forced-choice answer
  probability on a retrieval task appears far less sensitive to dropping
  low-mass experts than teacher-forced NLL over all tokens.
- **Context rot is not aggravated by adaptive truncation**: the baseline
  degradation (0.969 -> 0.688 accuracy) is unchanged under dynamic-k. The
  low-mass tail of the router distribution is apparently not where the
  needle-retrieval computation lives.
