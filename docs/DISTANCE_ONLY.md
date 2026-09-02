# Distance-only trigger test: do the retrieval heads collapse without distractors?

**Question.** Every failure this project causally explained was
distractor-driven: the OLMoE hard set rots only in its 8-distractor arm (the
0-distractor arm is 1.000 at every length, docs/CONTEXT_ROT_HARD.md), and
Granite needed a 24-distractor escalation to fail at all
(docs/GRANITE_TRANSPORT.md). Published long-context work reports degradation
with *clean* haystacks, i.e. from positional distance alone. Which is it for
the mechanism identified here — do the 16 retrieval heads
(docs/ATTENTION_TRANSPORT.md) also collapse under pure distance, or is the
collapse gated on distractor competition?

Script: `scripts/run_distance_only.py`; substrate generator:
`probes/generate_context_probes_distance.py`. The 16 head cells are FROZEN —
loaded from `data/attention_transport.json`, never re-identified here.

## Registered design (declared before any capture)

* Substrate: 0 distractors, needle depths {0.15, 0.50, 0.85} x length
  buckets {256, 1024, 2048, 3840} x 2 haystack domains x 8 replicates
  = 192 prompts (seed 9). `distance := n_tokens − needle_token_end`.
* Per prompt, one teacher-forced forward pass (CPU, BF16) yields
  forced-choice correctness/prob and the final query row's post-softmax
  attention: `M_top` (mean needle mass over the 16 frozen heads) and
  `M_rest` (other 240 heads).
* **Primary test** (fixed length, distance varied via depth): within the
  3840 bucket (n=48), Spearman rho of `M_top` vs distance, permutation
  p (2000 shuffles). Chance mass is constant within the bucket.
* **Secondary test** (collapse index vs matched-length references): the
  8-distractor 3840 run gives reference levels for the same heads —
  M_right = 0.432 (model-right) and M_wrong = 0.187 (collapsed). For each
  3840 cell, `c := (M_right − M_top_obs) / (M_right − M_wrong)`.
  Registered bands: c < 0.2 distractor-gated; c > 0.5 substantial
  distance-driven collapse; between = partial.
* **Specificity**: the same rho for `M_rest` — a decline at least as strong
  outside the identified heads is a global length artefact, not a
  retrieval-head effect.
* Registered claim floor for "distance alone collapses the heads":
  perm p < 0.05 on the primary rho with rho < 0 **and** c >= 0.5 at the
  max-distance cell (3840 / depth 0.15).

## Result: distractor-gated. Distance alone breaks nothing here.

**Registered verdict: `distractor_gated`.**

* **Accuracy: 192/192 correct (1.000 in every cell)**, mean answer prob
  0.990–0.998 everywhere — including the max-distance cell (depth 0.15 at
  3840 tokens, mean distance 3253 tokens). With zero distractors, OLMoE
  simply does not fail at these lengths, at any depth.
* **The heads do not collapse.** At max distance, `M_top` = **0.470**
  (sd 0.036) — *above* the matched-length 8-distractor model-right
  reference (0.432), and 2.5x the collapsed level (0.187). Collapse index
  at the max-distance cell: **c = −0.16** (< 0.2 band; negative means
  healthier than the healthy reference).
* **The distance trend that does exist is not head-specific.** Within the
  3840 bucket, `M_top` declines with distance (rho = −0.745, perm
  p < 0.0005: 0.598 at depth 0.85 -> 0.470 at depth 0.15), but `M_rest`
  declines *more steeply* (rho = −0.894, perm p < 0.0005; 0.044 -> 0.013,
  tracking chance mass). The specificity control therefore reads the
  decline as a global attention-spreading effect of long prefixes, not a
  selective failure of the retrieval heads: relative to chance mass
  (0.0030 at 3840), the identified heads at max distance still put
  **155x chance** attention on the needle.

Per-cell `M_top` (all cells 1.000 accuracy, n=16 each):

| depth \ bucket | 256 | 1024 | 2048 | 3840 | c @3840 |
|---|---:|---:|---:|---:|---:|
| 0.15 | 0.621 | 0.646 | 0.610 | 0.470 | −0.16 |
| 0.50 | 0.622 | 0.641 | 0.613 | 0.559 | −0.52 |
| 0.85 | 0.636 | 0.658 | 0.601 | 0.598 | −0.68 |

Full numbers: `data/distance_only.json`.

## Interpretation and scope

Within this project's regime (OLMoE-1B-7B, <= 3840 tokens, synthetic
forced-choice needle retrieval), **the retrieval-head collapse is
distractor-gated**: pure positional distance neither breaks accuracy nor
brings the identified heads anywhere near their collapsed level. Combined
with the earlier results, the precise statement is: the transport failure
this project localized and repaired is triggered by *competition* (same-form
distractor sentences), not by *distance* — and any distance-only degradation
reported elsewhere at much longer contexts (>> 4096 tokens) is not explained
by, and cannot be assumed to share, this mechanism.

What this does **not** show: models tested at 32k–1M tokens may well exhibit
a distance-driven failure mode; this model's 4096-token window cannot reach
that regime. The two-pathway question (competition vs distance) is settled
here only for the short-window case: at these lengths there is exactly one
pathway, and it is competition.
