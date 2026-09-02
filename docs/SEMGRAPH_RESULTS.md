# Results: semantic + graph-walk span detectors (registered)

Registration: docs/SEMGRAPH_REGISTRATION.md and docs/COREF_REGISTRATION.md,
both committed before any evaluation-bucket measurement. Raw outputs:
`data/semgraph.json`, `data/semgraph_coref.json`,
`data/semgraph/records*.jsonl`, `data/semgraph/top_cells.json`. Runner:
`scripts/run_semgraph.py`, `scripts/run_semgraph_coref.py`.

Everything downstream of span selection is frozen: the same 16 identified
OLMoE retrieval heads (re-derived deterministically; top-5 masses match
docs/ATTENTION_TRANSPORT.md exactly: L12H14 0.670, L9H6 0.651, L9H1 0.591,
L12H15 0.496, L3H15 0.444), pre-softmax boost beta = 4.0, query region from
the question start. This track changes ONLY the span locator; it makes no
claim about the strength of the transport mechanism.

Auxiliary model: sentence-transformers/all-MiniLM-L6-v2 (~22M params,
frozen, CPU). Detection uses zero forward passes of the target model but is
no longer "single tokenization only" — that trade is explicit.

## Dev-only calibration (16 dev prompts per variant, 1024 tokens)

| variant | selected | graph dev hit | semantic dev hit |
|---|---|---:|---:|
| paraphrase | h*=1, alpha*=1.0 | 1.000 | 1.000 |
| multihop | h*=2, alpha*=0.0 | 0.500 | 0.000 |
| coref | h*=2, alpha*=0.0, gamma*=1.0 | 1.000 | 0.000 |

The multihop calibration already shows the registered structure: a single
semantic hop never reaches the answer sentence (0%), while a 2-hop walk
(question -> bridge -> fact) reaches it half the time.

## Evaluation (3840-token bucket, 32 prompts per variant)

### Paraphrase — registered SUCCESS

| condition | acc | mean p(answer) | span hit |
|---|---:|---:|---:|
| baseline | 0.000 | 0.069 | |
| oracle boost | 1.000 | 0.860 | |
| wrong-span boost | 0.000 | 0.072 | |
| semantic boost | 1.000 | 0.860 | 1.000 |
| graph boost | 1.000 | 0.860 | 1.000 |

Semantic-vs-wrong: dz = 6.29, perm p < 0.0005 (bar |dz| >= 0.8 cleared);
hit rate 1.00 >= 0.90; 100.0% of oracle effect; 32/32 baseline failures
repaired. The semantic detector matches the lexical detector's prior
paraphrase result (dz = 6.46, ~100% of oracle) without relying on any
shared token.

### Multihop — registered PARTIAL (bar cleared; hit rate at, not above, 0.50)

| condition | acc | mean p(answer) | answer hit | bridge hit |
|---|---:|---:|---:|---:|
| baseline | 0.031 | 0.079 | | |
| oracle boost | 1.000 | 0.744 | | |
| wrong-span boost | 0.031 | 0.081 | | |
| semantic boost | 0.031 | 0.102 | 0.000 | 0.875 |
| graph boost (h=2) | 0.531 | 0.398 | 0.500 | path 0.563 |

Primary endpoint graph-vs-wrong: dz = 1.01, perm p < 0.0005 — the
registered bar (p < 0.05, |dz| >= 0.8) is CLEARED. Oracle fraction 48.0%
(> 0.45 registered). Answer hit rate is exactly 0.50, which does not
exceed the registered "> 0.50", so the outcome is classified PARTIAL, not
SUCCESS. Repair: 16/31 baseline failures (51.6%).

Versus the existing two-stage lexical chain (docs/MULTIHOP_CHAIN.md: hit
0.50, 45.0% of oracle, dz = 0.92): the graph walk ties the hit rate and is
slightly ahead on oracle fraction (48.0%) and effect size (1.01), with one
principled mechanism instead of a hand-built two-stage pipeline. It does
not decisively beat the chain.

Failure decomposition: the pure semantic detector lands on the BRIDGE
sentence 87.5% of the time but never on the answer (0%) — semantic
similarity follows the question's entity, not the fact. The graph walk's
path visits the bridge on 56.3% of prompts; when the first hop misses the
bridge, the second hop cannot recover (missed-bridge remains the dominant
failure, as in the two-stage chain).

## Coreference (registered in docs/COREF_REGISTRATION.md)

New substrate (`probes/probe_set_context_coref.yaml`): the answer sentence
shares ZERO contentful tokens with the question; only a pronoun link to
the adjacent anchor sentence (which carries the entity) identifies it.
Registered lexical prediction: 0% answer hit. Graph walk extended with a
discourse-adjacency edge (gamma * adj, adj = 1 iff the candidate
immediately follows the previously selected sentence), calibrated dev-only
over hops x alpha x gamma.

### Coreference — registered SUCCESS on ADJACENT anaphora only (AMENDED)

AMENDMENT (post-hoc review, result preserved unchanged): the v1 substrate
constructs every needle and every distractor as anchor + next-sentence
referent with no variation in distance and no competing antecedent. The
walk's adjacency edge (`adj = 1 iff j == prev + 1`) is therefore
guaranteed correct BY CONSTRUCTION once any anchor is found: the step is
entity matching plus a hard-coded "next sentence", not coreference
resolution the data could falsify. The v1 result below stands as recorded,
but its claim is narrowed to: a discourse-adjacency prior solves ADJACENT,
UNAMBIGUOUS anaphora. Variable-distance and multi-candidate coreference
were NOT tested by v1; they are the subject of the registered v2 substrate
(docs/COREF_V2_REGISTRATION.md), which removes the tautology.

Dev calibration is decisive: the ONLY grid cell with a non-zero dev hit
rate is h=2, alpha=0, gamma>0 (gamma=1.0: 100%; gamma=0.5: 50%; all 25
other cells: 0%). A pure semantic hop never finds the answer sentence.

Evaluation (3840 bucket, 32 prompts):

| condition | acc | mean p(answer) | answer hit | anchor hit |
|---|---:|---:|---:|---:|
| baseline | 0.313 | 0.271 | | |
| oracle boost | 1.000 | 0.948 | | |
| wrong-span boost | 0.281 | 0.245 | | |
| lexical boost | 0.781 | 0.728 | 1.000 | 1.000 |
| semantic boost | 0.219 | 0.205 | 0.000 | 1.000 |
| graph boost (h=2, gamma=1) | 1.000 | 0.948 | 1.000 | path 1.000 |

Primary endpoint graph-vs-wrong: dz = 3.37, perm p < 0.0005 (failing
subset dz = 5.59); hit rate 1.00; 100.0% of oracle effect; 22/22 baseline
failures repaired. Registered SUCCESS.

Within the adjacent-anaphora setting, the verdict question — does
coreference behave like paraphrase or like multi-hop? — points to
MULTI-HOP, but per the amendment above this does NOT answer the
manuscript's general question. Semantic-only
lands on the anchor 100% of the time but on the answer 0% of the time
(and its boost is slightly WORSE than wrong-span: dz = -0.37, because it
concentrates attention on the wrong sentence). Only the two-hop walk with
the discourse-adjacency edge (question -> anchor -> next sentence)
recovers the full oracle effect.

One registered prediction was WRONG and is reported as such: we predicted
the single-stage lexical detector would score 0% answer hits. It scored
100% "hits" — but only by geometric spillover: its fixed 16-token window
locks onto the ANCHOR (the only question-token match) and, because the
answer sentence is immediately adjacent by construction, the window
overlaps it. The partial coverage recovers just 67.5% of the oracle effect
(acc 0.781, failing-subset repair 16/22) vs the graph walk's 100.0%
(22/22). On substrates where the coreferent sentence is not adjacent, the
window spillover would not help; the walk's adjacency edge targets the
resolved sentence directly.

### Coreference v2, Experiment 1 — registered prediction CONFIRMED: the frozen v1 detector does NOT generalize past distance 1

Registration: docs/COREF_V2_REGISTRATION.md (committed with the substrate
before evaluation). Substrate: identical to v1 except the
anchor-to-referent distance is drawn uniformly from {1,2,3} per pair, per
emitted prompt (true pair and each of the 4 distractors independently),
with d-1 haystack-domain filler sentences in between; buckets 384/3840
(short bucket raised from 256 — disclosed deviation), dev arm 1024. A
replicate-level-draw bug in the first generated set is disclosed in the
registration; the corrected per-prompt-draw set is the registered result
(prelim run preserved as data/semgraph_coref_v2_prelim_repdraw.json and
shows the same pattern: graph hit 1.0 at d=1, 0.0 at d=2/3).

The v1-calibrated detector ran EXACTLY as frozen (h=2, alpha=0, gamma=1,
`adj = 1 iff j == prev + 1`), no recalibration.

Evaluation (3840 bucket, 32 prompts; eval-arm true distances
d=1: 6, d=2: 11, d=3: 15):

| condition | acc | mean p(answer) | answer hit | anchor hit |
|---|---:|---:|---:|---:|
| baseline | 0.125 | 0.156 | | |
| oracle boost | 1.000 | 0.940 | | |
| wrong-span boost | 0.125 | 0.156 | | |
| lexical boost | 0.250 | 0.230 | 0.188 | 1.000 |
| semantic boost | 0.125 | 0.156 | 0.000 | 1.000 |
| graph boost (frozen v1) | 0.281 | 0.311 | 0.188 | path 1.000 |

Answer hit rate by true distance — graph (frozen): d=1 1.000,
d=2 0.000, d=3 0.000. Lexical: 1.000 / 0.000 / 0.000 (its d=1 "hits" are
the same window-spillover artifact as v1). Semantic: 0.000 everywhere.

Primary endpoint graph-vs-wrong: dz = 0.48, perm p = 0.0015 — MISSES the
registered bar (|dz| >= 0.8); only 19.7% of the oracle effect; 5/28
baseline failures repaired (all at d=1). The walk still finds the anchor
on 100% of paths, then steps to the literal next sentence, which is the
referent only when d=1.

VERDICT: the registered degradation prediction is CONFIRMED. The v1
detector solves distance-1 (adjacent) anaphora specifically and falls to
the semantic-only floor at d>=2. The v1 "coreference" result is
substrate-dependent, exactly as the amendment states. General
variable-distance coreference remains UNSOLVED by the frozen detector;
a distance-tolerant edge is the subject of the separately registered
Experiment 2 (docs/COREF_V2_EXP2_REGISTRATION.md), not run before this
result was recorded.

Raw artifacts: data/semgraph_coref_v2.json,
data/semgraph/records_coref_v2.jsonl.

## Honest scope

- All substrates are synthetic forced-choice needle tasks; no claim of
  generality to natural documents.
- Calibration used dev arms only; evaluation buckets were never consulted
  before freezing (h, alpha, gamma).
- Locator performance and transport-mechanism strength remain separate
  claims; nothing here strengthens the causal chain.
