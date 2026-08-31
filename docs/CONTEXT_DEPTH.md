# Needle depth: context rot depends strongly on WHERE the needle sits

Depths 0.15 / 0.50 / 0.85 and buckets 256 / 3840 were registered in
`probes/generate_context_probes_depth.py` before any capture (the whole
project previously fixed depth at 50%, a flagged scope limit). 96 prompts,
8 distractors, both haystack types; analysis
`scripts/run_context_depth_analyze.py`, results `data/context_depth.json`.

## Results (forced-choice accuracy / mean answer probability, n=16 per cell)

| depth | 256 tokens | 3840 tokens |
|---|---|---|
| 0.15 (early) | 1.000 / 0.773 | **0.375 / 0.207** |
| 0.50 (middle) | 0.750 / 0.543 | 0.563 / 0.329 |
| 0.85 (late) | 1.000 / 0.771 | **0.813 / 0.583** |

Permutation test on the between-depth spread of answer probability
(2000 shuffles): p=0.001 at 256, p<0.0005 at 3840.

## Reading

- **At long context the effect is large and monotone with recency**: an early
  needle answers at 0.375 while a late needle answers at 0.813 in the SAME
  3840-token bucket. Context rot as measured all project (depth 0.50) is a
  middle point of a much wider range; "length" alone understates the
  phenomenon — needle-to-question DISTANCE is a first-order factor.
- This is consistent with the transport interpretation from
  docs/PROBE_REPAIRED.md and the failure of readout-position anchoring
  (docs/ANCHOR_CAUSAL.md): the further the needle is from the readout, the
  less reliably its (still-decodable-at-source) content arrives there.
- The 256-token middle-depth dip (0.750/0.543, both haystacks) is unexpected
  and unexplained; at 256 tokens the depth-0.5 needle sits nearest the dense
  distractor block, so it may be a distractor-adjacency artifact of the
  generator rather than a depth effect. Flagged, not interpreted further.

## Limitations

- n=16 per cell; one model; one task family; two buckets; accuracy-level only
  (no routing/pathway analysis was run on these traces yet).
- Depth and needle-to-question distance are confounded by design (the question
  is always at the end) — this data cannot separate "absolute position" from
  "distance to readout".
