
# Context Rot at the Routing Level — OLMoE-1B-7B-0924

**Verdict: SUBSTRATE CANNOT TEST THE QUESTION**

## The gap this addresses

Chroma's *Context Rot: How Increasing Input Tokens Impacts LLM Performance* (2025) measured, across 18 models including the MoE Qwen3-235B-A22B, that accuracy degrades as input length grows **even when task difficulty is held fixed**. On the mechanism, the report says:

> "we do not have a definitive answer for why that occurs... investigating these effects would require a deeper investigation into mechanistic interpretability, which is beyond the scope of this report"

This workstream runs that deeper investigation on one small, fully open MoE, at the one place a MoE can degrade that a dense model cannot: **the router**. It asks whether the length-accuracy curve is accompanied by a length-routing curve — whether the router becomes less decisive, or less specialised, as context grows.

## Design

`probes/probe_set_context.yaml` — 192 prompts, 6 length buckets (256, 512, 1024, 2048, 3072, 3840 tokens), 2 x 2 conditions, 8 replicates. `probe_set_v1.yaml` is untouched, so every other result in this repo stays comparable.

Two of Chroma's conditions are reproduced so the results land on named comparisons rather than 'context rot in general':

1. **The distractor comparison** — their needle-in-a-haystack result contrasting a clean haystack with one seeded with distractors (they use 1 and 4; we use 0 and 4). Distractors here are the needle's own sentence template with a different entity and a *wrong answer drawn from the same forced-choice pool*, so they compete directly in scoring.
2. **The needle/haystack similarity comparison** — the same needle buried in either a topically similar haystack (corporate facilities/security prose) or a dissimilar one (glacial geology prose).

Needle depth is fixed at 50%. Depth is a large effect in the NIAH literature and is deliberately held constant: this set has one independent variable. That is a scope limit, stated, not a claim that depth does not matter.

## The normalisation trap, and how it is avoided

Everywhere else in this project, `aggregate.py` equalises the token budget per cell so that length cannot masquerade as signal. **Here length is the independent variable, so that control is unavailable** — and the risk inverts. The failure mode is manufacturing a 'more experts fire at long context' result that is pure arithmetic:

> The number of *distinct experts touched* grows with token count under a > frozen, length-blind router, because more tokens means more top-8 draws > (coupon collector). Any sum over tokens has this problem.

Two defences, both structural rather than post-hoc:

**1. A content-identical measurement window.** Within a replicate, the needle sentence and the trailing question block are *byte-identical* across every length bucket and condition. All primary metrics are computed on those windows only, so the token multiset being measured is literally the same string everywhere; only the amount of preceding context differs. Verified, not assumed:

- question-window tokens per bucket: {256: 904, 512: 904, 1024: 904, 2048: 904, 3072: 904, 3840: 904} — **equal by construction**
- needle-window tokens per bucket: {256: 372, 512: 372, 1024: 372, 2048: 372, 3072: 372, 3840: 372} — **equal by construction**

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
| 256 | 987.8 | 1024.0 |
| 512 | 994.7 | 1024.0 |
| 1024 | 997.4 | 1024.0 |
| 2048 | 999.7 | 1024.0 |
| 3072 | 1001.2 | 1024.0 |
| 3840 | 1001.2 | 1024.0 |

Observed rho vs length = +0.395. This is what a confident, entirely fake context-rot result looks like: the observed curve tracks the null expectation, so essentially all of the growth is arithmetic. **It is excluded from the FDR family and is never used as evidence.**

## Accuracy: does context rot replicate here at all?

Accuracy is read off the final-position logits of the *same* forward pass that produces the routing trace. Every candidate answer is a single token by construction, so this is an exact logit comparison — no generation, no sampling. Four measures, because a base model can fail in different ways:

- `accuracy` — argmax over the 8 candidates. Chance = 0.125.
- `answer_prob` — softmax over the 8 candidates; graded, far more sensitive.
- `strict_top1` — argmax over the full 50,304-token vocabulary, i.e. what greedy decoding would actually emit.
- `answer_margin` — logit gap, correct vs. best distractor.

| bucket | n | accuracy | answer_prob | strict_top1 | mean answer rank |
|---|---|---|---|---|---|
| 256 | 32 | 0.938 | 0.735 | 0.938 | 1.1 |
| 512 | 32 | 0.875 | 0.772 | 0.875 | 1.3 |
| 1024 | 32 | 0.812 | 0.749 | 0.781 | 1.3 |
| 2048 | 32 | 0.781 | 0.708 | 0.781 | 1.4 |
| 3072 | 32 | 0.781 | 0.701 | 0.781 | 1.4 |
| 3840 | 32 | 0.688 | 0.673 | 0.688 | 2.6 |

### By condition (the direct Chroma comparison)

Mean `answer_prob` (top) and forced-choice `accuracy` (bottom) per cell.

| condition | 256 | 512 | 1024 | 2048 | 3072 | 3840 |
|---|---|---|---|---|---|---|
| dissimilar haystack, 0 distractors — prob | 0.995 | 0.994 | 0.995 | 0.992 | 0.991 | 0.990 |
| dissimilar haystack, 8 distractors — prob | 0.474 | 0.541 | 0.533 | 0.334 | 0.394 | 0.364 |
| similar haystack, 0 distractors — prob | 0.992 | 0.996 | 0.996 | 0.998 | 0.994 | 0.991 |
| similar haystack, 8 distractors — prob | 0.477 | 0.557 | 0.471 | 0.508 | 0.424 | 0.347 |
| dissimilar haystack, 0 distractors — acc | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| dissimilar haystack, 8 distractors — acc | 0.88 | 0.75 | 0.75 | 0.75 | 0.62 | 0.38 |
| similar haystack, 0 distractors — acc | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| similar haystack, 8 distractors — acc | 0.88 | 0.75 | 0.50 | 0.38 | 0.50 | 0.38 |

## The figure

Chroma's accuracy-vs-length curve, overlaid with routing-entropy-vs-length and specialisation-vs-length. Each series is min-max normalised to its own range, so the plot shows **shape agreement** — whether these move together — not magnitudes, which are in incomparable units. Absolute values are in the tables above and below; this figure is never the only report of a number. (matplotlib is not installed in this venv; this is a text figure, deliberately, so the report stays self-contained.)

```
    1.0 |+               o                                             
        |                                                              
        |                #                                             
        |                *              o                              
        |                +                                            #
        |o                                                             
        |                               +                              
        |                               *                              
        |                               #               o        *     
        |                                                        o     
        |                                               +              
        |                                                        #     
        |                                                        +     
    0.0 |                                               #             +
        +--------------------------------------------------------------
         256            512           1024            2048     30723840   (input tokens, log2 axis)

   *  accuracy (forced choice)  [0.6875 .. 0.9375]
   o  answer_prob  [0.6732 .. 0.7718]
   #  router entropy (question window)  [5.47 .. 5.492]
   +  needle-affine hit rate (specialisation)  [0.03597 .. 0.04335]
```

## Routing metrics vs length — both bars applied

Every trend below clears (or fails) BOTH a permutation null (2000 shuffles of the length labels) with Benjamini-Hochberg FDR at q=0.05 across the whole family, AND a practical effect-size floor (|Cohen's d| >= 0.8 between shortest and longest bucket, |rho| >= 0.5 for monotonicity). `docs/FINDINGS.md` records this project once reporting 70% of cells 'significant' at a median lift of 0.79x; the both-bars rule exists because of that.

| metric | short | long | delta | % | rho | perm p | FDR | Cohen's d | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `accuracy` | 0.9375 | 0.6875 | -0.25 | -26.7% | -0.191 | 0.0080 | yes | -0.67 | **SIGNIFICANT-BUT-TRIVIAL** |
| `answer_prob` | 0.7345 | 0.6732 | -0.06137 | -8.4% | -0.090 | 0.1964 | no | -0.19 | **FLAT** |
| `answer_margin` | 3.562 | 2.814 | -0.748 | -21.0% | -0.095 | 0.1754 | no | -0.23 | **FLAT** |
| `strict_top1` | 0.9375 | 0.6875 | -0.25 | -26.7% | -0.183 | 0.0120 | yes | -0.67 | **SIGNIFICANT-BUT-TRIVIAL** |
| `entropy_q` | 5.492 | 5.485 | -0.007007 | -0.1% | -0.140 | 0.0615 | no | -0.30 | **FLAT** |
| `mass_q` | 0.3742 | 0.3756 | +0.001364 | +0.4% | +0.092 | 0.2279 | no | +0.18 | **FLAT** |
| `entropy_needle` | 5.514 | 5.548 | +0.03387 | +0.6% | +0.196 | 0.0080 | yes | +0.59 | **SIGNIFICANT-BUT-TRIVIAL** |
| `mass_needle` | 0.3664 | 0.3542 | -0.01217 | -3.3% | -0.206 | 0.0055 | yes | -0.57 | **SIGNIFICANT-BUT-TRIVIAL** |
| `needle_affinity_rate` | 0.04335 | 0.03597 | -0.00738 | -17.0% | -0.426 | 0.0005 | yes | -1.45 | **SIGNIFICANT-BUT-NON-MONOTONE** |
| `needle_affinity_rate_q` | 0.006088 | 0.008786 | +0.002697 | +44.3% | +0.498 | 0.0005 | yes | +1.49 | **SIGNIFICANT-BUT-NON-MONOTONE** |
| `entropy_all` | 5.447 | 5.601 | +0.1535 | +2.8% | +0.854 | 0.0005 | yes | +4.53 | **TREND** |
| `mass_all` | 0.3896 | 0.3334 | -0.05623 | -14.4% | -0.854 | 0.0005 | yes | -4.50 | **TREND** |
| `hot_load_share_q` | 0.1407 | 0.15 | +0.009346 | +6.6% | +0.315 | 0.0005 | yes | +1.04 | **SIGNIFICANT-BUT-NON-MONOTONE** |
| `cold_load_share_q` | 0.197 | 0.1841 | -0.01295 | -6.6% | -0.246 | 0.0010 | yes | -0.91 | **SIGNIFICANT-BUT-NON-MONOTONE** |

`TREND` = cleared both bars. `SIGNIFICANT-BUT-TRIVIAL` = survived FDR but the effect is too small to matter — reported as *not* a finding, which is exactly the distinction this project got wrong once before. `FLAT` = no trend.

## Co-activation community stability

| bucket | usage skew | modularity | n communities | ARI vs shortest | reliable? |
|---|---|---|---|---|---|
| 256 | 740.0x | 0.9370 | 135 | 1.000 | **NO** |
| 512 | 755.0x | 0.9367 | 149 | 0.979 | **NO** |
| 1024 | 765.0x | 0.9363 | 162 | 0.952 | **NO** |
| 2048 | 763.0x | 0.9360 | 172 | 0.940 | **NO** |
| 3072 | 744.0x | 0.9360 | 174 | 0.934 | **NO** |
| 3840 | 829.0x | 0.9359 | 179 | 0.933 | **NO** |

**verdict: UNRELIABLE.** `coactivation.py`'s own documented PMI validity limit is 2.0x usage skew (measured there: separation from noise collapses by ~10x). Skew here is far outside that range, so the community numbers are known-contaminated by base rate and must not be trusted **in either direction** — neither as evidence that communities blur at long context nor as evidence that they are stable. This is the same call `docs/FINDINGS.md` made for H4, made here for the same reason and against this workstream's interest.

## Hot-expert concentration vs length

`data/utilization.json` (Workstream 3) was available. **Its headline result reframes this section**: the hypothesised 'specialisation lives in hot experts' mechanism is refuted — specialists are disproportionately *cold* (observed overlap 34 vs null 54.5+/-4.7, enrichment 0.624x, p<0.0001). So hot-expert concentration is measured here **without** the prior that it is the mechanism; the prior is that hot experts are generalists.

WS-3's open question, which this sweep can speak to: if routing degrades with length, does it concentrate in the generalist (high-load) pathway or the specialist (low-load) one?

| bucket | share of draws on hot experts | share on cold experts |
|---|---|---|
| 256 | 0.1407 | 0.1970 |
| 512 | 0.1428 | 0.1911 |
| 1024 | 0.1442 | 0.1864 |
| 2048 | 0.1452 | 0.1878 |
| 3072 | 0.1476 | 0.1884 |
| 3840 | 0.1500 | 0.1841 |

Trend: hot share rho=+0.315, d=+1.04 -> **SIGNIFICANT-BUT-NON-MONOTONE**; cold share rho=-0.246, d=-0.91 -> **SIGNIFICANT-BUT-NON-MONOTONE**.

Note the honest limit WS-3 states and which carries over here: `load_ratio` and `max|lift|` come from the same count matrix and are partly definitionally opposed. Read hot/cold as 'load and specialisation are opposed on this substrate', not as two independent variables.

## Verdict

### SUBSTRATE CANNOT TEST THE QUESTION

Task accuracy does **not** degrade with input length on this model under this design, so there is no context rot here to find a mechanism for. Every routing number above is therefore descriptive only: with no accuracy curve to explain, a routing curve would explain nothing, and a flat routing curve would rule nothing out.

- `accuracy`: 0.938 at 256 tokens -> 0.688 at 3840 (rho=-0.191, d=-0.67, SIGNIFICANT-BUT-TRIVIAL)
- `answer_prob`: 0.735 -> 0.673 (rho=-0.090, d=-0.19, FLAT)

**This is a substrate limitation and it bounds every downstream claim in this workstream.** It is reported here, not buried.

## Limits

1. **One model, one seed.** OLMoE-1B-7B-0924 has 64 experts per layer; frontier MoEs have hundreds. PLAN.md §9b flags the second-model check as not optional, and it has not been run for this workstream either.
2. **8 replicates per cell** (192 prompts total). Sized to finish — see the wall-clock section below. Powered for the pooled length trend, not for per-condition trends, which are shown for shape and should not be significance-tested individually at this n.
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
