# Keep-top-K probe -- FAIR version (pre-selection restriction)

Target domain: `medicine`. Control domain: `cooking`. Metric: mean per-token teacher-forced cross-entropy (nats), held-out (split=B) prompts, forward passes only.

Companion to `docs/KEEP_TOPK_PROBE.md` (the post-hoc-zeroing version). This one restricts each layer's router to choose its top-k **only from the kept set**, by masking non-kept experts' logits to -inf before softmax+topk -- verified against `OlmoeTopKRouter.forward` source directly (transformers==5.15.0), not guessed. Every selected slot in a restricted layer is therefore a real, non-zero expert -- the wasted-slots problem in the post-hoc version cannot happen here.

**Reported limitation:** keep sets are chosen globally by lift rank, then split per layer -- a layer can end up with fewer than top_k kept experts. Such layers are left fully unrestricted rather than force-shrunk to a smaller top_k (shrinking would add a second, uncontrolled confound). Skipped-layer counts are reported per condition below; read the results next to that number, not in isolation.

## Results

| condition | n kept (global) | layers skipped (< top_k kept) | loss on target | delta vs baseline | loss on control | delta vs baseline |
|---|---|---|---|---|---|---|
| baseline | 1024/1024 | 0/16 | 2.7846 | +0.0000 | 2.8352 | +0.0000 |
| keep_10% | 202/1024 | 2/16 | 7.0500 | +4.2654 | 7.2178 | +4.3827 |
| keep_20% | 305/1024 | 0/16 | 6.9607 | +4.1761 | 7.1903 | +4.3552 |
| keep_30% | 403/1024 | 0/16 | 5.6033 | +2.8186 | 6.1811 | +3.3459 |

## Reading this against the post-hoc version
- If loss here is substantially better than the matching keep_frac row in `KEEP_TOPK_PROBE.md`, that confirms the post-hoc mechanism's own bluntness -- not the underlying idea -- was the main source of damage there.
- If loss is still close to random even here, with few layers skipped, that's a much stronger negative result than the post-hoc version could ever produce, because the wasted-slots confound has been removed.
- Watch the skipped-layers column: a condition where many layers had to be left unrestricted isn't really testing 'keep only K% everywhere' -- it's testing a mix of restricted and unrestricted layers, which is a real result but a different, weaker claim than the headline keep_frac number implies.
- Same caveats as everywhere else in this project: one domain pair, one model, one seed, small held-out n -- directional only.
