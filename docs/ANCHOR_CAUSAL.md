# Residual-stream anchoring at the readout position: NOT SUPPORTED

The follow-up causal test to docs/PROBE_REPAIRED.md. The repaired probe showed
the needle's fact stays linearly decodable at its source position (layer 8,
99.5% cross-pair accuracy) even on long prompts the model answers wrong, while
final-position decodability degrades. Hypothesis tested here: injecting a
clean content-bearing vector for the answer into the residual stream at the
final (readout) position restores accuracy.

Script: `scripts/run_anchor_causal.py`; results: `data/anchor_causal.json`.

## Design (registered before running)

- Anchor vectors: per-answer-word centroids of `needle_last` layer-8 hidden
  states from SHORT (256-token) prompts only — the representation the probe
  showed is intact.
- Injection: added at the final position's residual stream entering layer 8,
  scaled to `alpha * ||h||` (`expertatlas/anchoring.py`, `n_fired` manipulation
  check asserted per forward).
- Alpha calibrated on a 16-prompt 1024-token development subset only
  (alpha in {0.25, 0.5, 1.0}; chosen alpha* = 0.5). Evaluation: all 64
  held-out 3840-token prompts.
- Conditions: baseline (no injection), TRUE answer centroid, WRONG-content
  centroid (fixed shift 3), RANDOM direction — all norm-matched at the same
  alpha.
- Paired sign-flip permutation tests (2000 perms), Cohen dz, bar: perm p<0.05
  AND |dz|>=0.8 AND true beats BOTH controls.

## Results (n=64 long prompts, forced-choice answer probability)

| condition | acc | mean answer prob |
|---|---|---|
| baseline | 0.781 | 0.678 |
| true anchor | 0.688 | 0.633 |
| wrong-content | 0.703 | 0.637 |
| random | 0.703 | 0.652 |

| contrast | mean delta | dz | perm p |
|---|---|---|---|
| true vs baseline | **-0.045** | -0.50 | 0.000 |
| true vs random | -0.019 | -0.21 | 0.108 |
| true vs wrong | -0.003 | -0.15 | 0.220 |

Model-wrong subset (n=14): baseline 0/14 correct; true, wrong, and random each
flip the SAME 2/14 — a content-independent perturbation effect, not repair.

## Verdict

**NOT SUPPORTED.** The true-content anchor significantly *lowers* answer
probability vs baseline and is statistically indistinguishable from the
wrong-content and random controls. Whatever the injection does, it does it
regardless of content. The dev-set calibration gain (0.69 -> 0.81 acc, n=16)
did not transfer to evaluation — small-n calibration noise, reported here
rather than re-tuned.

## What this rules in/out

- Combined with docs/MECHANISM_CAUSAL.md and docs/ADAPTIVE_CAUSAL.md, every
  single-site intervention tried so far (router boost x2, residual content
  injection at the readout position) fails its controls. The needle's
  information survives at its source but naively pasting it at the readout
  position does not make the model use it.
- Consistent with the transport interpretation: the readout position's
  computation may need the fact delivered through its normal attention path
  (with the right phase/geometry), not superimposed as a raw source-position
  centroid.
- Next candidates in evidence order: (a) identify the attention heads that
  move needle content to the final position on short prompts and test whether
  their attention to the needle collapses on long wrong prompts; (b) inject at
  the source position or along the layer where the probe shows final-position
  decodability first degrading, rather than at layer 8.

## Limitations

- One model, one task family, CPU BF16, n=64 eval / n=14 model-wrong.
- One injection site (entering layer 8, final position); a layer/position
  sweep was not run.
- Centroids average over 24 short prompts per word; a per-prompt (rather than
  per-word) anchor was not tested.
