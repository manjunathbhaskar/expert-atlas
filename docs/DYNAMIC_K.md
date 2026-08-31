# Dynamic top-k: per-token adaptive FFN truncation

Target domain: `medicine`. Control domain: `cooking`. Metric: mean per-token teacher-forced cross-entropy (nats), held-out (split=B) prompts.

This is the per-token adaptive counterpart to `docs/OFFLOADING.md`. `OffloadedMoe` keeps a fixed subset of experts for every token. `DynamicKMoe` keeps a variable number `k_t` per token, determined by the router's own softmax mass: the smallest prefix of the top-8 probabilities that exceeds the `mass_threshold`.

## Method caveats

- The gate is still evaluated over the full candidate pool (or the fixed kept set, if used with `OffloadedMoe`). The savings are in FFN matmuls, not in the router projection.
- The current implementation uses a per-token Python loop inside the FFN. Wall-clock time is therefore dominated by Python overhead and is **not expected to improve** over the original in this prototype. The reported `mean_kept` is the FFN-compute saving; a fused CUDA/Metal kernel would be needed to realise it as wall-clock speed.
- The threshold is a hyperparameter. It was not tuned on the evaluated prompts.

## Results

| condition | mass threshold | mean kept | loss on target | delta | loss on control | delta | wall (s) |
|---|---|---|---|---|---|---|---|
| baseline | 1.0 | 8.00 | 2.9749 | — | 2.8792 | — | 5.2 |
| dynamic_k_0.9 | 0.9 | 8.00 | 2.9760 | +0.0011 | 2.8797 | +0.0005 | 5.3 |
| dynamic_k_0.95 | 0.95 | 8.00 | 2.9767 | +0.0018 | 2.8801 | +0.0009 | 5.8 |
| dynamic_k_0.99 | 0.99 | 8.00 | 2.9749 | +0.0000 | 2.8792 | -0.0000 | 5.8 |

## Reading this

- If `mean_kept` drops substantially (e.g. 8 -> 4) and loss stays close to baseline, the router is often concentrated and most of the top-8 mass is carried by a small prefix. That is evidence for an *adaptive* sparsity policy.
- If `mean_kept` stays near 8 even at 0.99, the router is usually diffuse and dynamic-k saves little.
- Wall-clock is not the metric here: this is a quality + FLOP-count experiment. A production implementation would fuse the variable-k loop.

## Post-run diagnosis (read before the table above)

This first run's thresholds **never fired**: `mean_kept` is 8.00 everywhere
because the truncation compares cumulative *raw* top-8 weights against the
threshold, and on OLMoE (`norm_topk_prob=False`) those weights sum to the
top-8 share of a 64-way softmax — measured mean **0.416** (p5 0.287, p95
0.622), with only **0.13%** of tokens reaching 0.9. Thresholds of
0.90/0.95/0.99 on that scale are unreachable, so this run is a no-op by
construction, not evidence that dynamic-k is lossless. This is exactly the
`norm_topk_prob=False` trap noted in the README's methodology conventions
(OLMoE's top-k gate weights do not sum to 1).
`docs/DYNAMIC_K_RELATIVE.md` reruns this with thresholds measured relative
to the top-8 mass, which is the reachable scale.

## Honest limits

- n=3 thresholds, one model, one seed, one domain pair, split-B prompts only. Directional.
- No wall-clock speed claim is made for this Python prototype.
- No random or held-out threshold selection; thresholds are reported as given.
