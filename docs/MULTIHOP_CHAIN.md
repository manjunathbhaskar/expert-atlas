# Two-stage lexical chain: a training-free detector for multi-hop retrieval

**Open problem being closed.** docs/CONTEXT_VARIANTS.md documented that the
zero-forward-pass lexical detector fails completely on multi-hop composition:
0% needle hit rate, because the question's rare tokens point at the *bridge*
sentence ("the Zurich office is designated Site Kestrel"), not at the fact
sentence that actually carries the answer ("the codeword for Site Kestrel is
silver"). The only working fallback was the L8 residual probe (61.4% of
oracle effect) — which needs a labeled dev set and model forward passes.

**Idea.** If the detector reliably finds the bridge, use the bridge itself as
the second query: (A) run the existing IDF-weighted lexical detector with the
question tokens to locate a bridge window; (B) extract the rare alphabetic
tokens of that window that do *not* appear in the question (i.e. the linking
entity introduced by the bridge), zero out the stage-A window, and run the
same detector again with those tokens as the query. Still zero model forward
passes. Script: `scripts/run_multihop_chain.py`.

## Design

* Same multihop substrate as docs/CONTEXT_VARIANTS.md: eval = 32 prompts at
  3840 tokens (8 distractors, depths {0.15, 0.50}, both haystacks); dev = 16
  prompts at 1024 tokens, used **only** to calibrate the two window widths
  (grid {8,12,16,24}^2, selected by dev hop-2 hit rate).
* Boost: the same 16 frozen heads and beta = 4.0 as every prior experiment;
  baseline / oracle / wrong-span values reused unchanged from
  `data/context_variants.json` (identical prompts and scoring).
* Registered bar (same as all interventions): chain beats wrong-span with
  paired sign-flip p < 0.05 and |dz| >= 0.8.
* Registered failure decomposition: (i) stage A missed the bridge,
  (ii) bridge found but hop 2 missed, (iii) span found but boost failed.

## Result: bar met. 0% -> 50% hit rate, 45% of oracle, zero training.

Calibration picked w_A=8, w_B=24 (dev hit 0.500; smaller stage-A windows
dominate the grid — a tight bridge window gives a cleaner hop-2 query).

Eval (n=32, 3840 tokens):

| condition | accuracy | mean answer prob |
|---|---:|---:|
| baseline | 0.031 | 0.079 |
| wrong-span control | 0.031 | 0.086 |
| **chain-detected boost** | **0.500** | **0.378** |
| oracle boost | 1.000 | 0.744 |

* Needle (fact-sentence) hit rate: **0.500** (was 0.000 for the single-stage
  detector). Stage-A bridge hit rate: 0.5625; **given a bridge hit, hop 2
  finds the fact 88.9% of the time** (16/18).
* Chain vs wrong-span: mean delta +0.292, **dz = 0.92, perm p < 0.0005**
  (n=32; failing subset n=31: dz = 0.89, p < 0.0005) — passes the
  registered |dz| >= 0.8 floor. Chain vs baseline: dz = 0.95, p < 0.0005.
* Oracle-effect fraction: **45.0%**, vs 61.4% for the L8 residual probe —
  but with no labels, no dev-set training, and no forward passes.
* Failing-subset repair accuracy: 15/31 (0.484).

**Failure decomposition** (registered): the bottleneck is stage A — 43.75%
of prompts miss the bridge (14/32), and nearly every bridge hit converts
(hop-2 conditional hit 88.9%, i.e. only 2/18 hop misses). Improving bridge
detection, not the hop, is where the remaining 55%
of the oracle effect lives.

## Interpretation

The multi-hop failure of the lexical detector is not fundamental to
lexical detection — it was a *one-hop* limitation. Composing the same
detector with itself recovers half the hit rate and 45% of the oracle
effect at zero cost, and cleanly passes the same pre-registered bar the
oracle and one-hop repairs were held to. The L8 probe remains stronger
(61.4%) where a labeled dev set is affordable; the chain is the better
default when it is not. Scope: same as all multihop results — OLMoE,
3840 tokens, synthetic two-hop forced-choice, n=32.

Full numbers: `data/multihop_chain.json`.
