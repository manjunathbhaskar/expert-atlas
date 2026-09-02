
# Context Rot at the Routing Level — OLMoE-1B-7B-0924

**Verdict: SUBSTRATE CANNOT TEST THE QUESTION**

## The gap this addresses

Chroma's *Context Rot: How Increasing Input Tokens Impacts LLM Performance* (2025) measured, across 18 models including the MoE Qwen3-235B-A22B, that accuracy degrades as input length grows **even when task difficulty is held fixed**. On the mechanism, the report says:

> "we do not have a definitive answer for why that occurs... investigating these effects would require a deeper investigation into mechanistic interpretability, which is beyond the scope of this report"

This workstream runs that deeper investigation on one small, fully open MoE, at the one place a MoE can degrade that a dense model cannot: **the router**. It asks whether the length-accuracy curve is accompanied by a length-routing curve — whether the router becomes less decisive, or less specialised, as context grows.

## Design

`probes/probe_set_context.yaml` — 112 prompts, 7 length buckets (128, 256, 512, 1024, 2048, 3072, 3840 tokens), 2 x 2 conditions, 4 replicates. `probe_set_v1.yaml` is untouched, so every other result in this repo stays comparable.

Two of Chroma's conditions are reproduced so the results land on named comparisons rather than 'context rot in general':

1. **The distractor comparison** — their needle-in-a-haystack result contrasting a clean haystack with one seeded with distractors (they use 1 and 4; we use 0 and 4). Distractors here are the needle's own sentence template with a different entity and a *wrong answer drawn from the same forced-choice pool*, so they compete directly in scoring.
2. **The needle/haystack similarity comparison** — the same needle buried in either a topically similar haystack (corporate facilities/security prose) or a dissimilar one (glacial geology prose).

Needle depth is fixed at 50%. Depth is a large effect in the NIAH literature and is deliberately held constant: this set has one independent variable. That is a scope limit, stated, not a claim that depth does not matter.

## The normalisation trap, and how it is avoided

Everywhere else in this project, `aggregate.py` equalises the token budget per cell so that length cannot masquerade as signal. **Here length is the independent variable, so that control is unavailable** — and the risk inverts. The failure mode is manufacturing a 'more experts fire at long context' result that is pure arithmetic:

> The number of *distinct experts touched* grows with token count under a > frozen, length-blind router, because more tokens means more top-8 draws > (coupon collector). Any sum over tokens has this problem.

Two defences, both structural rather than post-hoc:

**1. A content-identical measurement window.** Within a replicate, the needle sentence and the trailing question block are *byte-identical* across every length bucket and condition. All primary metrics are computed on those windows only, so the token multiset being measured is literally the same string everywhere; only the amount of preceding context differs. Verified, not assumed:

- question-window tokens per bucket: {128: 448, 256: 448, 512: 448, 1024: 448, 2048: 448, 3072: 448, 3840: 448} — **equal by construction**
- needle-window tokens per bucket: {128: 184, 256: 184, 512: 184, 1024: 184, 2048: 184, 3072: 184, 3840: 184} — **equal by construction**

**2. Every metric is a mean or a rate, never a sum.** Per-metric normalisation:

| metric | normalisation | length-invariant under a null router? |
|---|---|---|
| `entropy_q`, `entropy_needle` | mean over window tokens x layers of H(full 64-way softmax), bits | yes |
| `mass_q`, `mass_needle` | mean over window tokens x layers of top-k probability mass | yes |
| `needle_affinity_rate` | fraction of window top-k draws landing in a fixed reference set, [0,1] | yes |
| `hot_load_share_q` | fraction of window draws on WS-3 hot experts, [0,1] | yes |
| co-activation | built from equal-size windows; PMI already divides out base rate | yes, subject to the skew gate |
| `entropy_all`, `mass_all` | per-token mean, but over the **whole prompt** | yes arithmetically, **no** in content: long prompts are mostly haystack, so this confounds length with token mix. Reported as secondary only. |
| `distinct_experts_TRAP` | **none — this is the trap** | **no** |

### The trap, measured

`distinct_experts_touched` is implemented in `context_metrics.py` as a negative control and reported here so the artefact is visible rather than merely asserted to have been avoided:

| bucket | distinct experts touched | coupon-collector expectation under a null router |
|---|---|---|
| 128 | 963.9 | 1024.0 |
| 256 | 987.4 | 1024.0 |
| 512 | 995.2 | 1024.0 |
| 1024 | 997.3 | 1024.0 |
| 2048 | 1000.3 | 1024.0 |
| 3072 | 1001.0 | 1024.0 |
| 3840 | 1001.8 | 1024.0 |

Observed rho vs length = +0.648. This is what a confident, entirely fake context-rot result looks like: the observed curve tracks the null expectation, so essentially all of the growth is arithmetic. **It is excluded from the FDR family and is never used as evidence.**

## Accuracy: does context rot replicate here at all?

Accuracy is read off the final-position logits of the *same* forward pass that produces the routing trace. Every candidate answer is a single token by construction, so this is an exact logit comparison — no generation, no sampling. Four measures, because a base model can fail in different ways:

- `accuracy` — argmax over the 8 candidates. Chance = 0.125.
- `answer_prob` — softmax over the 8 candidates; graded, far more sensitive.
- `strict_top1` — argmax over the full 50,304-token vocabulary, i.e. what greedy decoding would actually emit.
- `answer_margin` — logit gap, correct vs. best distractor.

| bucket | n | accuracy | answer_prob | strict_top1 | mean answer rank |
|---|---|---|---|---|---|
| 128 | 16 | 0.688 | 0.634 | 0.688 | 2.6 |
| 256 | 16 | 0.875 | 0.757 | 0.875 | 1.2 |
| 512 | 16 | 0.875 | 0.720 | 0.875 | 1.1 |
| 1024 | 16 | 0.875 | 0.713 | 0.875 | 1.2 |
| 2048 | 16 | 0.812 | 0.701 | 0.812 | 1.8 |
| 3072 | 16 | 0.750 | 0.710 | 0.688 | 1.3 |
| 3840 | 16 | 0.812 | 0.717 | 0.812 | 1.4 |

### By condition (the direct Chroma comparison)

Mean `answer_prob` (top) and forced-choice `accuracy` (bottom) per cell.

| condition | 128 | 256 | 512 | 1024 | 2048 | 3072 | 3840 |
|---|---|---|---|---|---|---|---|
| dissimilar haystack, 0 distractors — prob | 0.984 | 0.995 | 0.996 | 0.992 | 0.995 | 0.994 | 0.993 |
| dissimilar haystack, 4 distractors — prob | 0.268 | 0.574 | 0.467 | 0.399 | 0.241 | 0.308 | 0.438 |
| similar haystack, 0 distractors — prob | 0.983 | 0.989 | 0.997 | 0.997 | 0.998 | 0.994 | 0.992 |
| similar haystack, 4 distractors — prob | 0.299 | 0.468 | 0.420 | 0.465 | 0.572 | 0.545 | 0.445 |
| dissimilar haystack, 0 distractors — acc | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| dissimilar haystack, 4 distractors — acc | 0.25 | 0.75 | 0.75 | 0.75 | 0.50 | 0.25 | 0.75 |
| similar haystack, 0 distractors — acc | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| similar haystack, 4 distractors — acc | 0.50 | 0.75 | 0.75 | 0.75 | 0.75 | 0.75 | 0.50 |

## The figure

Chroma's accuracy-vs-length curve, overlaid with routing-entropy-vs-length and specialisation-vs-length. Each series is min-max normalised to its own range, so the plot shows **shape agreement** — whether these move together — not magnitudes, which are in incomparable units. Absolute values are in the tables above and below; this figure is never the only report of a number. (matplotlib is not installed in this venv; this is a text figure, deliberately, so the report stays self-contained.)

```
    1.0 |+           #            *           *                        
        |                                                              
        |                                                              
        |                                                              
        |#                        #                        *          #
        |                                     o                   o    
        |                         +                        o           
        |            +                        +                        
        |                                                              
        |                                     #                   *    
        |                                                              
        |                                                  +      #    
        |                                                         +    
    0.0 |o                                                 #          +
        +--------------------------------------------------------------
         128        256          512        1024         2048   3073840   (input tokens, log2 axis)

   *  accuracy (forced choice)  [0.6875 .. 0.875]
   o  answer_prob  [0.6337 .. 0.7567]
   #  router entropy (question window)  [5.467 .. 5.494]
   +  needle-affine hit rate (specialisation)  [0.03466 .. 0.04396]
```

## Routing metrics vs length — both bars applied

Every trend below clears (or fails) BOTH a permutation null (2000 shuffles of the length labels) with Benjamini-Hochberg FDR at q=0.05 across the whole family, AND a practical effect-size floor (|Cohen's d| >= 0.8 between shortest and longest bucket, |rho| >= 0.5 for monotonicity). `docs/FINDINGS.md` records this project once reporting 70% of cells 'significant' at a median lift of 0.79x; the both-bars rule exists because of that.

| metric | short | long | delta | % | rho | perm p | FDR | Cohen's d | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `accuracy` | 0.6875 | 0.8125 | +0.125 | +18.2% | +0.039 | 0.6837 | no | +0.28 | **FLAT** |
| `answer_prob` | 0.6337 | 0.7169 | +0.08325 | +13.1% | +0.019 | 0.8306 | no | +0.24 | **FLAT** |
| `answer_margin` | 2.562 | 3.031 | +0.4688 | +18.3% | +0.008 | 0.9370 | no | +0.15 | **FLAT** |
| `strict_top1` | 0.6875 | 0.8125 | +0.125 | +18.2% | +0.018 | 0.8571 | no | +0.28 | **FLAT** |
| `entropy_q` | 5.486 | 5.485 | -0.0002414 | -0.0% | -0.196 | 0.0460 | no | -0.01 | **FLAT** |
| `mass_q` | 0.3776 | 0.376 | -0.001665 | -0.4% | +0.082 | 0.4128 | no | -0.30 | **FLAT** |
| `entropy_needle` | 5.485 | 5.545 | +0.05976 | +1.1% | +0.361 | 0.0005 | yes | +1.32 | **SIGNIFICANT-BUT-NON-MONOTONE** |
| `mass_needle` | 0.3776 | 0.3559 | -0.0217 | -5.7% | -0.368 | 0.0005 | yes | -1.31 | **SIGNIFICANT-BUT-NON-MONOTONE** |
| `needle_affinity_rate` | 0.04396 | 0.03466 | -0.0093 | -21.2% | -0.472 | 0.0005 | yes | -2.15 | **SIGNIFICANT-BUT-NON-MONOTONE** |
| `needle_affinity_rate_q` | 0.00556 | 0.008333 | +0.002773 | +49.9% | +0.477 | 0.0005 | yes | +1.46 | **SIGNIFICANT-BUT-NON-MONOTONE** |
| `entropy_all` | 5.41 | 5.601 | +0.1909 | +3.5% | +0.887 | 0.0005 | yes | +5.90 | **TREND** |
| `mass_all` | 0.4003 | 0.3334 | -0.06694 | -16.7% | -0.879 | 0.0005 | yes | -5.54 | **TREND** |
| `hot_load_share_q` | 0.133 | 0.1503 | +0.01736 | +13.1% | +0.420 | 0.0005 | yes | +1.80 | **SIGNIFICANT-BUT-NON-MONOTONE** |
| `cold_load_share_q` | 0.2045 | 0.1826 | -0.0219 | -10.7% | -0.412 | 0.0005 | yes | -1.63 | **SIGNIFICANT-BUT-NON-MONOTONE** |

`TREND` = cleared both bars. `SIGNIFICANT-BUT-TRIVIAL` = survived FDR but the effect is too small to matter — reported as *not* a finding, which is exactly the distinction this project got wrong once before. `FLAT` = no trend.

## Co-activation community stability

| bucket | usage skew | modularity | n communities | ARI vs shortest | reliable? |
|---|---|---|---|---|---|
| 128 | 369.0x | 0.9370 | 154 | 1.000 | **NO** |
| 256 | 385.0x | 0.9370 | 157 | 0.964 | **NO** |
| 512 | 381.0x | 0.9366 | 187 | 0.938 | **NO** |
| 1024 | 389.0x | 0.9365 | 190 | 0.930 | **NO** |
| 2048 | 390.0x | 0.9361 | 198 | 0.920 | **NO** |
| 3072 | 383.0x | 0.9360 | 193 | 0.923 | **NO** |
| 3840 | 414.0x | 0.9361 | 196 | 0.922 | **NO** |

**verdict: UNRELIABLE.** `coactivation.py`'s own documented PMI validity limit is 2.0x usage skew (measured there: separation from noise collapses by ~10x). Skew here is far outside that range, so the community numbers are known-contaminated by base rate and must not be trusted **in either direction** — neither as evidence that communities blur at long context nor as evidence that they are stable. This is the same call `docs/FINDINGS.md` made for H4, made here for the same reason and against this workstream's interest.

## Hot-expert concentration vs length

`data/utilization.json` (Workstream 3) was available. **Its headline result reframes this section**: the hypothesised 'specialisation lives in hot experts' mechanism is refuted — specialists are disproportionately *cold* (observed overlap 34 vs null 54.5+/-4.7, enrichment 0.624x, p<0.0001). So hot-expert concentration is measured here **without** the prior that it is the mechanism; the prior is that hot experts are generalists.

WS-3's open question, which this sweep can speak to: if routing degrades with length, does it concentrate in the generalist (high-load) pathway or the specialist (low-load) one?

| bucket | share of draws on hot experts | share on cold experts |
|---|---|---|
| 128 | 0.1330 | 0.2045 |
| 256 | 0.1405 | 0.1939 |
| 512 | 0.1422 | 0.1894 |
| 1024 | 0.1442 | 0.1849 |
| 2048 | 0.1457 | 0.1871 |
| 3072 | 0.1459 | 0.1881 |
| 3840 | 0.1503 | 0.1826 |

Trend: hot share rho=+0.420, d=+1.80 -> **SIGNIFICANT-BUT-NON-MONOTONE**; cold share rho=-0.412, d=-1.63 -> **SIGNIFICANT-BUT-NON-MONOTONE**.

Note the honest limit WS-3 states and which carries over here: `load_ratio` and `max|lift|` come from the same count matrix and are partly definitionally opposed. Read hot/cold as 'load and specialisation are opposed on this substrate', not as two independent variables.

## Verdict

### SUBSTRATE CANNOT TEST THE QUESTION

Task accuracy does **not** degrade with input length on this model under this design, so there is no context rot here to find a mechanism for. Every routing number above is therefore descriptive only: with no accuracy curve to explain, a routing curve would explain nothing, and a flat routing curve would rule nothing out.

- `accuracy`: 0.688 at 128 tokens -> 0.812 at 3840 (rho=+0.039, d=+0.28, FLAT)
- `answer_prob`: 0.634 -> 0.717 (rho=+0.019, d=+0.24, FLAT)

**This is a substrate limitation and it bounds every downstream claim in this workstream.** It is reported here, not buried.

## Limits

1. **One model, one seed.** OLMoE-1B-7B-0924 has 64 experts per layer; frontier MoEs have hundreds. PLAN.md §9b flags the second-model check as not optional, and it has not been run for this workstream either.
2. **4 replicates per cell** (112 prompts total). Sized to finish — see the wall-clock section below. Powered for the pooled length trend, not for per-condition trends, which are shown for shape and should not be significance-tested individually at this n.
3. **Needle depth fixed at 50%.** Depth is known to matter; it is held constant here so that length is the only independent variable.
4. **A base model on a retrieval task.** OLMoE-1B-7B-0924 is not instruction-tuned. Forced-choice scoring is used precisely because it does not require the model to follow an instruction, but the task is still easier than what Chroma ran on instruction-tuned frontier models.
5. **Routing only.** As `docs/TRANSFER.md` §6.3 says of the orthogonality result: this is evidence about the routing layer specifically, not the full computation. Two prompts routing identically can still compute differently inside the experts.
6. **Co-activation results are gated out** by the usage-skew check above and should not be cited in either direction.

## Wall clock and what was cut

Throughput was measured on 3 real prompts at 126 / 1008 / 3831 tokens **before** launching anything, per the brief. Fitted cost on this machine:

```
seconds(T) = 61.38 - 0.0149*T + 1.323e-05*T^2
```

A fixed ~58 s/forward dominates below ~1k tokens — the 16 x 64 expert loop runs regardless of token count — and quadratic attention takes over above it. So on this substrate short buckets are nearly free and the long tail is the entire cost.

| replicates | prompts | nominal | +20% fit margin | x1.6 worst case |
|---|---|---|---|---|
| 3 | 84 | 2.20h | 2.64h | 4.22h |
| **4** | **112** | **2.93h** | **3.52h** | **5.63h** |
| 5 | 140 | 3.66h | 4.40h | 7.04h |
| 6 | 168 | 4.40h | 5.28h | 8.44h |

The 1.6x worst case is not arbitrary: `docs/TRANSFER.md` §11 records the prior 480-prompt run taking 12.9h against a ~8h fixed-cost floor, an unexplained ~1.6x slowdown attributed to thermal or contention effects. Budgeting for it rather than assuming it away is the difference between finishing and not.

**What was cut: replicates, from 6 to 4.** Bucket count (7) and both condition axes (2 x 2) were preserved, as the brief requires — those are what make the result map onto Chroma's named comparisons, and cutting them would have made the run cheaper and worthless. Replicates are the one axis where less costs only statistical power. 6 replicates projected to 8.44h worst case, over the ~8h budget; 4 lands at 5.63h worst case with real margin. A completed smaller sweep beats an unfinished larger one.

## Reproducing

```bash
export HF_HOME="$PWD/data/hf_cache" HF_HUB_OFFLINE=1
python probes/probe_set_context.py --replicates 4
python scripts/run_context_sweep.py      # resumable; safe to kill and rerun
python scripts/run_context_analyze.py
pytest tests/ws_ctx -q
```
