# Registration: coreference v2, Experiment 2 — distance-tolerant discourse edge

Registered AFTER Experiment 1's result was recorded
(docs/SEMGRAPH_RESULTS.md: frozen v1 detector degrades exactly as
predicted — answer hit 1.000 at d=1, 0.000 at d=2 and d=3; dz = 0.48
misses the bar) and BEFORE any evaluation of the detector below. Kept
separate from Experiment 1 per the project rule that "does X work" and
"here's the fix for X" never share one registration.

## Detector change (the ONLY change)

Replace the hard adjacency term in the graph-walk edge with a decaying
forward-proximity term:

    prox(j) = decay^(j - (prev + 1))   if prev is not None and j > prev
              0                        otherwise

    score(j) = lex(t, j) + alpha * cos(emb_t, emb_j) + gamma * prox(j)

At decay -> 0 this recovers the v1 hard adjacency (prox = 1 only at
j = prev + 1), so v1 is a special case; decay > 0 lets the walk reach a
referent several sentences past the anchor while still preferring nearer
candidates. The edge remains backward-blind (prox = 0 for j <= prev),
matching the anaphora construction.

Everything else is frozen and identical to Experiment 1: substrate
probes/probe_set_context_coref_v2.yaml (corrected per-prompt draws), same
16 OLMoE retrieval heads, beta = 4.0, baseline / oracle / wrong-span
conditions, WRONG_WIDTH = 16, paired sign-flip permutations (2000 draws,
seed 0), evaluation only on the 3840 bucket, bar p < 0.05 AND |dz| >= 0.8.

## Calibration (dev arm only)

Grid: h in {1, 2, 3} x alpha in {0.0, 0.5, 1.0} x gamma in {0.5, 1.0} x
decay in {0.3, 0.5, 0.7}, selected by answer hit rate on the 16-prompt
dev arm (1024 bucket) only, ties broken by smaller h, then smaller alpha,
then smaller gamma, then smaller decay. The evaluation bucket is never
consulted before freezing.

## Registered prediction

The decaying edge should restore answer hits at d = 2 and d = 3 (the
anchor-finding first hop already works: path anchor-hit was 100% in
Exp 1). Failure mode to watch honestly: at larger decay the edge may
prefer a distractor's referent or a filler sentence over the true
referent, since fillers between anchor and referent are ordinary haystack
sentences; the per-distance decomposition (registered endpoint) will show
this. Success = graph-vs-wrong clears p < 0.05 AND |dz| >= 0.8 on the
3840 bucket with hit rate improved at d >= 2; anything less is reported
as partial or failure.

## Variant B (still separate, later)

Competing pronoun-bearing near-antecedent; registered separately after
Experiment 2's result is recorded.

## Scope

Changes only the walk's discourse edge. Does not alter the substrate,
head set, boost, v1 results, Exp 1 results, main MoE results, or any
prior registration. Locator performance and transport-mechanism strength
remain separate claims.
