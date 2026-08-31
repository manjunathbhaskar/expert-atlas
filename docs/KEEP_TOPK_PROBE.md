# Keep-top-K probe -- quick first pass, NOT a finished result

Target domain: `medicine`. Control domain: `cooking`. Metric: mean per-token teacher-forced cross-entropy (nats), held-out (split=B) prompts, forward passes only.

**Method caveat, stated up front:** this reuses the `ExpertAblator` hook from `run_ablation_harness.py`, which zeroes an expert's contribution AFTER the router's top-8 selection, not by restricting the candidate pool BEFORE selection. When most of the model is ablated, many tokens will have most of their 8 selected slots land on now-zeroed experts -- a harsher test than 'let the router choose only among the kept experts' would be. Read this as a pessimistic first pass, not the final answer on whether keep-top-K is viable.

## Results

| condition | n kept | loss on target | delta vs baseline | loss on control | delta vs baseline |
|---|---|---|---|---|---|
| baseline | 1024/1024 | 2.7846 | +0.0000 | 2.8352 | +0.0000 |
| keep_10% | 202/1024 | 8.7380 | +5.9534 | 8.6081 | +5.7730 |
| keep_20% | 305/1024 | 7.6141 | +4.8295 | 8.1100 | +5.2749 |
| keep_30% | 403/1024 | 5.0067 | +2.2221 | 5.8836 | +3.0485 |

## Reading this
- If loss on target stays close to baseline even at a low keep_frac, that's a real signal the idea has legs, even under this pessimistic mechanism.
- If loss explodes even at a high keep_frac, that plausibly reflects the post-hoc-zeroing mechanism wasting slots rather than proving the idea is dead -- the next step would be the fairer pre-selection-restricted version (mask logits to -inf for non-kept experts before top-k, so all 8 selected slots come from the kept set) before drawing a real conclusion either way.
- n is small (held-out split=B prompts only), one domain pair, one model, one seed -- directional only, same caveats as docs/ABLATION.md.
