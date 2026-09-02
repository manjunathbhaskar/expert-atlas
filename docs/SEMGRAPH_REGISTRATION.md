# Registered design: semantic + graph-walk span detectors (SEMGRAPH)

Status: REGISTERED before any evaluation-bucket measurement on this track.
This file is committed before `scripts/run_semgraph.py` produces any
eval-bucket number. Amendments, if any, will be labeled as such and will not
replace this text.

## Question

The training-free lexical (IDF) detector recovers ~100% of the oracle boost
effect when the question shares rare tokens with the needle, but fails on
direct multi-hop composition (0% needle hit single-stage; 50% hit / 45% of
oracle with the hardcoded two-stage chain, `docs/MULTIHOP_CHAIN.md`). Can a
(a) semantic embedding detector and (b) a general graph-walk detector locate
the answer span better — while the repair itself stays frozen — making the
locate-and-repair pipeline task-general instead of lexical-overlap-bound?

This track changes ONLY the span locator. It says nothing new about the
transport mechanism; oracle repair, detector repair, and causal localization
remain separate claims.

## Substrate (fixed, committed)

`probes/probe_set_context_variants.yaml` (unchanged): variants `paraphrase`
and `multihop`; dev arm = 16 prompts/variant at 1024 tokens (`dev: true`);
eval = 32 prompts/variant at 3840 tokens. Model: `allenai/OLMoE-1B-7B-0924`,
CPU, bfloat16, teacher-forced greedy forced-choice scoring (unchanged).

## Provenance note (declared up front)

The frozen raw artifacts from the original OLMoE runs
(`data/attention_transport.json`, `data/context_variants.json`,
`data/context_variants/records.jsonl`) exist only on the machine that ran
them and are not in the repository. They are re-derived HERE, before any
detector evaluation, by the identical registered procedures and code paths:

1. Retrieval-head set: re-run stage 1 of `scripts/run_attention_transport.py`
   logic — repaired probe set (`probes/probe_set_context_repaired.yaml`),
   256-token bucket, model-correct prompts only, final-row post-softmax
   needle mass per (layer, head), top K=16 by mean mass. K fixed in advance.
   The re-derived cell list is recorded in the output for comparison with
   the published top-5 (`docs` report: L9H13 etc. — deterministic procedure,
   expected identical).
2. Baseline / oracle-span / wrong-span conditions on the variants eval
   bucket: recomputed with the same code (`run_boost` semantics: frozen 16
   cells, beta=4.0, queries = question span through final position),
   wrong-span = width-16 uniform random non-overlapping span, seed 0.

## Detectors (label-free at evaluation; ZERO target-model forward passes)

Sentence segmentation: the context region [0, question_char_start) is split
into sentences at `. ` boundaries via regex with character offsets; each
sentence maps to a token span through the tokenizer's offset mapping.
Sentences overlapping the question span are excluded. Position-0 (BOS/sink)
is never selectable.

Embedder: `sentence-transformers/all-MiniLM-L6-v2` (22M params, frozen,
off-the-shelf). Disclosed trade: detection is no longer tokenization-only;
it uses a small auxiliary encoder, never the target model.

### Detector S (semantic)

score(sentence) = cosine(emb(question), emb(sentence)); detected span =
token span of the argmax sentence; ties break to the LATEST sentence (same
rationale as span-discovery v2). No hyperparameters.

### Detector G (graph walk)

Nodes = context sentences + the question. Directed hop score from node u to
candidate v (v unvisited):

    hop(u, v) = lex(u, v) + alpha * cosine(emb(u), emb(v))

where lex(u, v) = sum over shared token ids t of 1/df(t), restricted to
"linking tokens": df(t) < MAX_DF (=8, unchanged), decoded token has >= 3
alphabetic characters, and — for hops after the first — t does not appear in
the question (a hop must leave its source, same rule as the two-stage
chain). Walk: start at the question node, greedily take argmax hop(u, v)
for h hops (visited nodes excluded; ties to the LATEST sentence). Detected
span = token span of the final node.

Calibration (DEV arms only, per variant, before any eval-bucket contact):
h in {1, 2, 3}, alpha in {0, 0.5, 1.0}, selected by needle-span hit rate on
the 16 dev prompts; ties break to larger h, then larger alpha. The 3840
bucket plays no part in any choice.

## Boost (frozen, unchanged)

Same 16 re-derived head cells, beta = 4.0, queries = question start through
final position, keys = the DETECTED span. Identical `HeadBoost` code path.

## Conditions on each 3840 eval prompt (per variant)

baseline, oracle (true needle span), wrong-span control (as above),
semantic-boost, graph-boost. Both detectors are evaluated on BOTH variants.

## Nulls (stated first) and registered bar

1. A detector's boost does no better than baseline.
2. Decisive: no better than the wrong-span control (generic perturbation,
   not span discovery).

Bar per contrast: paired sign-flip permutation (2000 draws) p < 0.05 AND
|dz| >= 0.8 vs wrong-span. Primary endpoints: graph-vs-wrong on multihop;
semantic-vs-wrong on paraphrase. All other contrasts are secondary.

Reported however they come out: hit rates, accuracy, mean forced-choice
probability, repair rate on the baseline-failing subset, fraction of the
oracle delta, hop-level failure decomposition for G (bridge hit vs final
hit; missed-bridge vs missed-hop), and comparison rows for the stored
references: single-stage lexical (0% multihop hit), two-stage chain (50%
hit, 45% of oracle, dz=0.92 vs wrong), L8 probe (61.4% of oracle, labeled).

## Success criteria (fixed in advance)

- SUCCESS (multihop): primary bar cleared AND needle hit rate > 0.50 AND
  fraction of oracle > 0.45 (i.e. beats the two-stage chain).
- SUCCESS (paraphrase): primary bar cleared AND hit rate >= 0.90 (i.e. does
  not lose what the lexical detector already has).
- PARTIAL: bar cleared but reference not beaten.
- FAILURE: bar not cleared. Reported as a registered negative.

Seeds: wrong-span rng seed 0; stats rng seed 1; 2000 permutations. No
eval-bucket data is inspected before calibration is locked.
