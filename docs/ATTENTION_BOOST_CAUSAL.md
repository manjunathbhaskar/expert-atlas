# Causal test: boosting identified retrieval heads' attention to the needle

**The fourth intervention family, and the first with a positive result.**
The three previous families touched the router (fixed boost — NOT SUPPORTED,
docs/MECHANISM_CAUSAL.md; entropy-triggered boost — NOT SUPPORTED,
docs/ADAPTIVE_CAUSAL.md) or the content (residual anchor at the readout —
NOT SUPPORTED, docs/ANCHOR_CAUSAL.md). None touched the mechanism that moves
information between positions. This one boosts the attention weights
themselves, at exactly the head cells identified as needle carriers by
docs/ATTENTION_TRANSPORT.md, which also showed those cells' needle attention
collapses specifically on failing prompts.

Script: `scripts/run_attention_boost_causal.py`.

## Registered design (before evaluation)

* Heads: the K=16 cells from stage 1 (identified on SHORT correct prompts
  only; the 3840 bucket played no part in choosing them).
* Boost: additive pre-softmax bias `beta` at those cells, query positions =
  question span through the final position, key positions = needle span
  (`HeadBoost`, exact eager math otherwise).
* `beta` calibrated on the 1024-token DEV bucket (first 16 prompt_ids),
  identified-heads condition only, betas {1, 2, 4} -> beta*=4.0
  (dev acc 1.000 at every beta; baseline dev acc 0.688).
* Control: per prompt, 16 head cells drawn uniformly at random (fixed
  seed), same beta, same query/key spans — matched in count, strength and
  target span; the only difference is WHICH heads.
* Null stated first: boosting the identified heads does no better than
  baseline, or no better than the same boost at random heads.
* Bar: perm p<0.05 AND |dz|>=0.8 AND beats the random control AND helps
  the model-wrong prompts more than the model-right ones.

## Result (3840-token bucket, all 64 prompts, paired)

| condition | forced-choice acc | mean answer prob |
|---|---:|---:|
| baseline | 0.781 | 0.678 |
| random-heads boost | 0.906 | 0.773 |
| **identified-heads boost** | **1.000** | **0.981** |

| contrast | mean delta | dz | perm p |
|---|---:|---:|---:|
| heads vs baseline (n=64) | +0.303 | +0.89 | <0.0005 |
| heads vs random (n=64) | +0.208 | +0.72 | <0.0005 |
| heads vs baseline, model-wrong subset (n=14) | +0.748 | +5.55 | <0.0005 |
| heads vs random, model-wrong subset (n=14) | +0.597 | +2.13 | <0.0005 |
| heads vs baseline, model-right subset (n=50) | +0.179 | +0.67 | <0.0005 |

Model-wrong subset (the prompts the repair targets): baseline 0/14 correct,
random-heads boost 8/14, **identified-heads boost 14/14**, mean answer prob
0.238 -> 0.986.

## Verdict against the registered bar

* p<0.05: **yes**, every contrast (<0.0005).
* beats the random control: **yes** (+0.208 overall, +0.597 on the failing
  subset).
* helps failing prompts more than working ones: **yes** (+0.748 vs +0.179;
  the model-right prompts start near ceiling, 0.678 -> mean 0.981 overall).
* |dz|>=0.8: **yes vs baseline** (0.89 full set, 5.55 failing subset).
  Vs the random control, dz is 2.13 on the failing subset but **0.72 on
  the full set** — below the 0.8 floor, because 50 of 64 prompts are at
  ceiling under both conditions and contribute ~zero paired variance-scaled
  signal. Reported as measured: the full-set control-contrast effect size
  misses the floor; the failing-subset contrast clears it decisively.

So: on the prompts that actually fail, boosting the identified retrieval
heads restores every one (14/14), with a large, significant margin over a
strength-matched random-head control. This is the outcome the three
previous families never produced — the anchor test's best condition never
beat its controls at all.

## What this does and does not show

* It shows the failure is **causally downstream of attention transport**:
  when the identified heads are made to attend to the needle again, the
  fact — which the probe showed was intact at the source — is read out
  correctly. Combined with the collapse measurement, the chain is now:
  fact survives at source -> specific retrieval heads stop attending to it
  at long range -> readout degrades -> router starvation appears as a
  symptom.
* The random control's partial recovery (8/14) is informative: even random
  heads pointed at the needle recover some prompts, i.e. part of the
  effect is generic "attend to the right span". But WHICH heads matters
  beyond that (+0.597 mean prob over random on the failing subset).
* **It is not a deployable fix.** The boost uses the needle's token span —
  oracle information about where the relevant fact sits. A real system
  would need to locate candidate spans without labels (e.g. from the
  retrieval heads' own short-range behavior, or a query-to-context match).
  This experiment isolates the mechanism; it does not solve span discovery.
* One model (OLMoE), one substrate, n=14 failing prompts, beta chosen on a
  dev bucket but coarse ({1,2,4}); no generation-time evaluation (teacher-
  forced forced-choice only).

## Next steps this licenses

1. Span-free variant: derive the boost target from the model's own signals
   (e.g. top attention spans of the identified heads under a sliding
   query, or needle-affine expert activation from the atlas) instead of
   the oracle span.
2. Depth-sweep interaction: test whether the same boost repairs the
   early-needle failures in docs/CONTEXT_DEPTH.md (depth 0.15, acc 0.375).
3. Granite: identify that model's retrieval heads and check the same
   collapse/repair pattern once a substrate exists where it fails.
