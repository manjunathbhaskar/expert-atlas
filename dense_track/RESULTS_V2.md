# Dense Track v2 — Registered Main Result (Pythia-2.8B)

Registered design: `REGISTRATION_V2.md` (written and committed before any v2
measurement). Substrate: `probe_set_dense_v2.yaml` (seed 23, 144 prompts,
24 replicates x 2 haystack domains x 3 buckets, 8 same-entity
confusable-attribute distractors per prompt). Raw outputs:
`data/records_v2.jsonl`, `data/needle_mass_v2.npz`, `data/transport_v2.json`,
`data/boost_v2.json`, `data/spanfree_v2.json`.

Relation to v1: the v1 registered substrate (`RESULTS.md`) did NOT induce
context rot (only 1 long failure), and a clearly-labelled v1 exploratory arm
found that same-entity confusable distractors do induce failures. That
exploratory result motivated the v2 design but its data are NOT reused here:
v2 is a fresh substrate (new seed, new prompts, 24 vs 16 replicates, all
three buckets) with the full pipeline registered before measurement.

## 1. Length-dependent failure (registered gate: PASSED)

Forced-choice accuracy (8 candidates, chance 0.125), n=48 per bucket:

| bucket (tokens) | accuracy | failures |
|---|---|---|
| 256 | 0.979 | 1 |
| 1024 | 0.833 | 8 |
| 1900 | 0.771 | 11 |

Gate required >=6 failures at 1900 AND a drop >=0.10 vs 256. Observed:
11 failures, drop 0.208. The dense model rots monotonically with length on
this substrate.

## 2. Retrieval-head identification (stage 1)

Top K=16 of 1024 (layer, head) cells by mean final-position needle attention
mass over the 47 short-bucket model-correct prompts. Top cell L11H18 carries
mass 0.744 vs chance 0.047 (~16x). All 16 cells sit in layers 11–26 (13 of 16
in layers 13–18); 13/16 cells coincide with the v1 identified set, i.e. the
head set is stable across substrates.

## 3. Localized attention collapse on long prompts (stage 2)

Long bucket (1900 tokens): n_right=37, n_wrong=11.

- Identified 16 cells: right mean needle mass 0.421 vs wrong 0.372,
  Cohen d = 0.70, label-permutation p = 0.048.
- Other 1008 cells: 0.0248 vs 0.0220, d = 0.52.
- Specificity: observed excess drop 0.0454 vs random-16-cell null
  (mean -0.0004, p95 0.0135), p < 0.0005 (0/2000 permutations reached it).

Honest note: the collapse contrast is significant and highly specific to the
identified cells, but its effect size (d = 0.70) is BELOW the registered
d >= 0.8 floor for the collapse stage. The correlational collapse claim is
therefore reported as suggestive, not confirmed. The causal stage below does
not depend on it and clears its own registered bar decisively.

## 4. Causal oracle-span boost (stage 3: registered bar CLEARED)

Beta calibrated on the first 16 sorted 1024-token dev prompts:
beta* = 4.0 (dev acc 0.875 -> 1.000). Evaluation: all 48 long prompts,
conditions baseline / identified heads / matched random 16 heads /
wrong span (same heads, same beta, width-16 non-overlapping span).

| condition | accuracy | mean answer prob |
|---|---|---|
| baseline | 0.771 | 0.588 |
| identified heads + oracle span | **1.000** | **0.892** |
| random heads + oracle span | 0.792 | 0.610 |
| identified heads + wrong span | 0.771 | 0.580 |

Paired effects (all n=48 unless noted, 2000-perm p):

- heads vs baseline: mean delta +0.304, dz = 1.44, p < 0.0005
- heads vs random-head control: delta +0.282, dz = 1.33, p < 0.0005
- heads vs wrong-span control: delta +0.312, dz = 1.41, p < 0.0005
- baseline-wrong subset (n=11): acc 0.000 -> 1.000 (11/11 failures
  repaired), prob 0.260 -> 0.826, dz = 5.64 vs baseline (p < 0.0005),
  dz = 3.82 vs random control (p = 0.001)
- baseline-right subset (n=37): dz = 1.34 — benefit is larger on failing
  prompts, as registered.

All registered causal criteria are met: significance, |dz| >= 0.8, beats
both matched controls, and larger benefit on baseline-wrong prompts.

## 5. Span-free lexical detector + boost (stage 4, conditional: TRIGGERED)

The lexical IDF-overlap detector (widths 8/12/16/24, width selected on the v2
1024-token dev bucket: width* = 24, dev hit rate 0.75) locates a candidate
span with no oracle knowledge, then applies the same identified-head boost
(beta = 4.0) to it. All 48 long prompts:

| condition | accuracy | mean answer prob |
|---|---|---|
| baseline | 0.771 | 0.588 |
| oracle span (reference) | 1.000 | 0.892 |
| wrong span (control) | 0.771 | 0.580 |
| lexical detected span | **0.875** | **0.742** |

- Detected-span hit rate (overlap with true needle span): 0.500.
- lexical vs baseline: delta +0.154, dz = 0.80, p < 0.0005 (n=48) — at the
  registered |dz| >= 0.8 floor.
- lexical vs wrong-span control: delta +0.162, dz = 0.82, p < 0.0005.
- Recovers 50.6% of the oracle effect on mean answer probability.
- Failing subset (n=11): repairs 5/11 failures (acc 0.000 -> 0.455, prob
  0.260 -> 0.482); vs wrong-span control dz = 1.01, p = 0.0005.

The fully span-free repair is real (clears both controls at the registered
floor) but partial: detector misses on half the prompts cap the recovery at
about half the oracle effect.

## 6. Verdict and scope

Under the registered v2 conditions, Pythia-2.8B (dense, no router) shows:
length-dependent retrieval failure (0.979 -> 0.771), a stable set of 16
mid-stack retrieval heads, and full causal repair of all 11 long-context
failures by boosting only those heads' attention to the needle span — with
matched random-head and wrong-span controls both flat. This supports the
claim that the attention-transport collapse and recoverable-by-boost pattern
found in OLMoE also appears in a dense transformer under conditions that
induce length-dependent retrieval failure.

What this does NOT show: anything about the MoE router-starvation component
(dense models have no router), generality beyond Pythia-2.8B, this substrate,
depth 0.50, and CPU fp32 teacher-forced evaluation. The correlational
collapse effect size fell below its registered floor (d = 0.70 < 0.8) and is
reported as suggestive only.
