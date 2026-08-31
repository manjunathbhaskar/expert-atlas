# Offloading baseline: real conditional compute

Target domain: `medicine`. Control domain: `cooking`. Metric: mean per-token teacher-forced cross-entropy (nats), held-out (split=B) prompts. Hot core: 100 experts.

This is the next step after `docs/KEEP_TOPK_FAIR_PROBE.md`: that document showed that *masking the router* to a kept set is still not enough to preserve accuracy. This document tests whether the same keep-sets, when realised as an actually smaller FFN (non-kept experts are not loaded and not computed), perform the same, better, or worse.

## Method caveats

- The full 13GB checkpoint is still loaded once from disk. The savings here are the **runtime FFN parameters** that the offloaded block keeps in memory and the **FFN matmuls** it executes, not the initial disk read. A state-dict filter that only deserialises the kept experts would be the genuine 'do not load the whole model' step; that is not implemented yet.
- Offloading is applied to **all 16 layers** with the same keep fraction. A layer-wise or token-adaptive policy is possible but not tested.
- Wall-clock times include the tokenisation and teacher-forced forward pass; they are rough and should not be over-interpreted for a single run.

## Results

| condition | n kept | kept frac | loss on target | delta | loss on control | delta | wall (s) | FFN bytes frac | FFN FLOPs frac |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 1024/1024 | 100.0% | 2.8935 | +0.0000 | 2.7979 | +0.0000 | 5.0 | 1.000 | 1.000 |
| keep_10_global | 202/1024 | 19.7% | 7.4018 | +4.5083 | 7.6917 | +4.8938 | 4.4 | 0.197 | 1.000 |
| keep_10_per_layer_quota | 196/1024 | 19.1% | 8.2060 | +5.3125 | 8.3486 | +5.5507 | 3.5 | 0.191 | 1.000 |
| keep_20_global | 305/1024 | 29.8% | 6.5885 | +3.6950 | 7.0336 | +4.2357 | 4.6 | 0.298 | 1.000 |
| keep_20_per_layer_quota | 306/1024 | 29.9% | 6.4399 | +3.5464 | 6.9961 | +4.1982 | 4.3 | 0.299 | 1.000 |
| keep_30_global | 403/1024 | 39.4% | 5.5524 | +2.6589 | 6.3296 | +3.5317 | 5.1 | 0.394 | 1.000 |
| keep_30_per_layer_quota | 402/1024 | 39.3% | 5.1293 | +2.2358 | 5.9044 | +3.1065 | 5.2 | 0.393 | 1.000 |

## Reading this

- If the `per_layer_quota` condition is closer to baseline than `global`, that supports the 'selection unit, not mechanism' hypothesis from `docs/KEEP_TOPK_FAIR_PROBE.md`.
- If even the offloaded versions lose accuracy badly, the keep-top-K idea is likely limited by *which* experts are needed, not by whether the router was free to choose among them.
- Wall-clock speedup should track `FFN FLOPs frac` if the offloaded block is the bottleneck; if it does not, measurement noise or non-FFN overheads dominate.

## Honest limits

- n=7 conditions, one model, one seed, one domain pair, split-B prompts only. Directional.
- No measured RSS; the byte fraction is an upper-bound estimate from the kept-set sizes, not an observed memory footprint.
- No random-draw null for the keep-sets. A 'top-by-lift beats random' claim is not supported by this run alone.
