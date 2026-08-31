# Attention transport: retrieval-head identification and collapse

**Question.** The repaired probe (docs/PROBE_REPAIRED.md) showed the needle's
content survives at its source position at every length but arrives degraded
at the readout exactly on model-wrong long prompts, and the residual anchor
(docs/ANCHOR_CAUSAL.md) showed that pasting content at the readout is
content-independent — true fact, wrong fact and random noise all did the
same thing. Both point at the transport, not the content. This experiment
measures the transport directly: post-softmax attention from the final
(readout) position onto the needle's tokens, per layer and head.

Script: `scripts/run_attention_transport.py`; tools:
`expertatlas/attention_transport.py` (custom attention interface that stores
only the final query row — ~4 MiB per 3840-token prompt, exact eager math,
verified bit-identical logits in `tests/ws_attn/`).

## Registered design (declared before any long-bucket data was inspected)

* **Stage 1 — identification.** Repaired probe set, 256-token bucket,
  model-correct prompts only (n=61). Score per (layer, head) cell = mean
  total attention mass on `needle_token_span` from the final position.
  The top **K=16** of the 256 cells (6.25%) are the candidate retrieval
  heads; K was fixed in advance.
* **Stage 2 — collapse test.** 3840-token bucket, model-right (n=50) vs
  model-wrong (n=14). Nulls stated first:
  * group null: identified heads' needle mass does not differ between
    right and wrong prompts (label-shuffle permutation, 2000 perms);
  * **specificity null: the drop is diffuse — a random set of K heads
    shows the same right-minus-wrong drop as the identified set**
    (2000 random K-cell subsets). Only rejecting this second null says the
    problem is localized; a drop that any random head set matches would
    instead support the "no single fixable site" reading of the anchor
    result.

## Stage 1 result: strongly concentrated retrieval heads exist

Chance mass (needle span / prompt length) is 0.047 on the short bucket.
The top cells put up to **14x chance** attention on the 12-token needle:

| rank | cell | mean needle mass | x chance |
|---|---|---:|---:|
| 1 | L12 H14 | 0.670 | 14.3x |
| 2 | L9 H6 | 0.651 | 13.9x |
| 3 | L9 H1 | 0.591 | 12.6x |
| 4 | L12 H15 | 0.496 | 10.6x |
| 5 | L3 H15 | 0.444 | 9.5x |
| ... | (16 cells total, all >= 6.2x chance) | | |

The identified set concentrates in layers 9–14 (13 of 16 cells), i.e. the
same mid-to-late depth range where the repaired probe found genuine
answer-position decoding emerging (L13–16).

## Stage 2 result: the collapse is concentrated in the identified heads

At 3840 tokens (chance mass 0.0030):

| statistic | model-right (n=50) | model-wrong (n=14) | perm p | d |
|---|---:|---:|---:|---:|
| mean needle mass, identified 16 heads | **0.432** | **0.187** | <0.0005 | 1.55 |
| mean needle mass, other 240 heads | 0.022 | 0.012 | <0.0005 | 1.28 |

Both drop, but the absolute collapse lives almost entirely in the
identified set: right-minus-wrong is 0.245 there vs 0.010 elsewhere.
Specificity test against the head-identity null: observed excess drop
0.235 vs null mean -0.0005, null 95th percentile 0.037 — **perm p <
0.0005**. Per-head BH-FDR with a practical floor (drop >= half the head's
short-prompt mass): 21 of 256 cells collapse, **12 of them among the 16
identified cells** (75% of the identified set vs 3.8% of the rest).

**Verdict: LOCALIZED.** On failing long prompts the drop in needle
attention is concentrated in the specific heads that carry needle content
on short correct prompts, far beyond what any random equal-sized head set
shows. The diffuse alternative — which would have said no single site is
fixable — is rejected by its own registered null. Note the identified
heads still place 0.187 (62x chance) on the needle even when the model is
wrong, so the failure is a partial, not total, loss of transport.

## Limitations

* One model (OLMoE), one substrate (the repaired probe set), n=14
  model-wrong prompts in stage 2's wrong group.
* Correlational as measured; the causal test of these heads is
  `scripts/run_attention_boost_causal.py` (docs/ATTENTION_BOOST_CAUSAL.md).
* "Needle mass" is attention from the final position only; transport via
  intermediate positions is not measured here.
