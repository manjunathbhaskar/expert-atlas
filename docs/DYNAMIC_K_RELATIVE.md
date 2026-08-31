# Dynamic top-k: per-token adaptive FFN truncation (relative-mass thresholds)

Target domain: `medicine`. Control domain: `cooking`. Metric: mean per-token teacher-forced cross-entropy (nats), held-out (split=B) prompts.

This is the per-token adaptive counterpart to `docs/OFFLOADING.md`. `OffloadedMoe` keeps a fixed subset of experts for every token. `DynamicKMoe` keeps a variable number `k_t` per token, determined by the router's own softmax mass: the smallest prefix of the top-8 probabilities that exceeds the `mass_threshold`.

## Method caveats

- The gate is still evaluated over the full candidate pool (or the fixed kept set, if used with `OffloadedMoe`). The savings are in FFN matmuls, not in the router projection.
- The current implementation uses a per-token Python loop inside the FFN. Wall-clock time is therefore dominated by Python overhead and is **not expected to improve** over the original in this prototype. The reported `mean_kept` is the FFN-compute saving; a fused CUDA/Metal kernel would be needed to realise it as wall-clock speed.
- The threshold is a hyperparameter. It was not tuned on the evaluated prompts.
- Thresholds are RELATIVE: a threshold of 0.9 keeps the smallest prefix carrying 90% of the mass the router gave its top-8, per token. Absolute thresholds never fire on OLMoE (`norm_topk_prob=False`; top-8 absolute mass is ~0.42 on average) — see `docs/DYNAMIC_K.md`.

## Results

| condition | mass threshold | mean kept | loss on target | delta | loss on control | delta | wall (s) |
|---|---|---|---|---|---|---|---|
| baseline | 1.0 | 8.00 | 2.9749 | — | 2.8792 | — | 5.1 |
| dynamic_k_0.5 | 0.5 | 3.06 | 3.2009 | +0.2260 | 3.0530 | +0.1738 | 3.9 |
| dynamic_k_0.7 | 0.7 | 4.73 | 3.0335 | +0.0586 | 2.9549 | +0.0757 | 4.8 |
| dynamic_k_0.8 | 0.8 | 5.77 | 3.0391 | +0.0642 | 2.9239 | +0.0447 | 4.8 |
| dynamic_k_0.9 | 0.9 | 6.97 | 3.0083 | +0.0334 | 2.8861 | +0.0069 | 5.0 |
| dynamic_k_0.95 | 0.95 | 7.88 | 2.9821 | +0.0072 | 2.8803 | +0.0011 | 6.1 |

## Reading this

- If `mean_kept` drops substantially (e.g. 8 -> 4) and loss stays close to baseline, the router is often concentrated and most of the top-8 mass is carried by a small prefix. That is evidence for an *adaptive* sparsity policy.
- If `mean_kept` stays near 8 even at 0.99, the router is usually diffuse and dynamic-k saves little.
- Wall-clock is not the metric here: this is a quality + FLOP-count experiment. A production implementation would fuse the variable-k loop.

## Verdict for this run

The truncation fires on the reachable scale, and the trade-off curve is real
but modest: at threshold 0.7 the model executes a mean **4.7 of 8** FFN experts
per token (~41% fewer expert-FFN matmuls) for **+0.06–0.08 nats** over
baseline (~2.9–3.0 nats, i.e. ~2% relative); at 0.5 it runs ~3 of 8 for
+0.17–0.23 nats. There is no free region — every truncation level costs some
loss — but the curve is smooth, with no collapse, and the cost is far below
the near-random damage the static keep-top-K probes produced at comparable
compute fractions (`docs/KEEP_TOPK_FAIR_PROBE.md`). Consistent with the
measured router diffuseness (reaching 90% of top-8 mass needs ~6.9 of 8
experts), the big savings tiers require accepting real loss; dynamic-k is a
knob, not a win. The target/control deltas are similar in size, so nothing
here is domain-selective.

## Honest limits

- n=5 thresholds, one model, one seed, one domain pair, split-B prompts only. Directional.
- No wall-clock speed claim is made for this Python prototype.
- No random or held-out threshold selection; thresholds are reported as given.
