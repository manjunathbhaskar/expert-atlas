# Dense track v2 registration (written BEFORE any v2 measurement)

## Context and disclosure

The v1 registered substrate did not induce context rot in Pythia-2.8B
(RESULTS.md §1). A v1 exploratory arm with same-entity confusable-attribute
distractors did induce failures and showed collapse + repair with frozen
components. This v2 run is a **confirmatory replication on a fresh,
registered substrate of that design**: the substrate design choice is
informed by the v1 exploratory result (declared here), but every v2
measurement — head identification, collapse, boost, span-free — is made on
new prompts (new seed, new replicates) with the full pipeline re-run from
scratch, registered before any v2 long-context result is observed.

## Model

`EleutherAI/pythia-2.8b` (unchanged from v1): dense GPT-NeoX, 32 layers x
32 heads = 1024 head cells, 2048-token range, CPU fp32, eager attention,
teacher-forced, deterministic.

## Substrate (probe_set_dense_v2.yaml, generate_probes_v2.py, seed 23)

- 24 replicates x 2 haystack domains (similar, dissimilar) x 3 length
  buckets (256, 1024, 1900) = 144 prompts; 48 prompts per bucket.
- Needle depth 0.50; needle and question byte-identical across buckets
  within a replicate.
- 8 distractors per prompt, all using the SAME entity as the needle with a
  confusable attribute word ("The visitor codeword for the Zurich office is
  copper."); the question names the SECURITY codeword, so the task stays
  uniquely answerable. Attribute pool: visitor, loading, maintenance,
  evening, backup, delivery, archive, weekend. Wrong answer words drawn
  from the candidate pool.
- Forced-choice pool unchanged (8 single-token words, chance = 0.125).
- Replicate needle assignment: entity = ENTITIES[rep % 8],
  word = CANDIDATE_WORDS[(rep + rep // 8) % 8], so no two replicates in a
  block of 8 share a needle and the 3 blocks differ.

## Gate before proceeding (registered)

Head identification and all downstream stages run ONLY if the 1900 bucket
shows at least 6 failing prompts AND accuracy at 1900 is at least 0.10
below accuracy at 256. Otherwise the run stops and is reported as a failed
induction.

## Pipeline (identical to v1 registration except substrate)

1. **Stage 1 — head identification.** Rank all 1024 (layer, head) cells by
   mean final-position attention mass on the needle token span over the
   256-token model-CORRECT prompts; take the top K = 16.
2. **Stage 2 — localized collapse.** On the 1900 bucket, compare identified
   -cell needle mass on model-right vs model-wrong prompts. Report Cohen d
   and a 2000-permutation label null. Specificity: excess right-minus-wrong
   drop on the identified 16 cells vs 2000 random 16-cell subsets drawn
   from the other 1008 cells.
3. **Stage 3 — oracle boost.** Pre-softmax bias +beta on the identified
   cells toward the true needle span, query rows from question start. Beta
   calibrated on the first 16 sorted 1024-bucket prompt ids, grid
   {1, 2, 4}. Evaluate all 48 1900-token prompts under: baseline / heads /
   random (16 random cells per prompt, fixed seed) / wrong span (identified
   cells, width-16 non-needle span).
4. **Stage 4 — span-free (conditional).** Lexical idf-overlap detector,
   widths {8, 12, 16, 24} selected on the 1024 dev prompts by span hit
   rate (ties -> largest), then detector + frozen boost on the 1900 bucket.
   Runs only if stage 3 clears the causal bar.

## Registered statistical bar (unchanged from v1 and the MoE work)

- Effect-size floors: Cohen |d| >= 0.8 (collapse), Cohen |dz| >= 0.8
  (paired causal contrasts); permutation p < 0.05 (2000 perms).
- Oracle causal success requires ALL of: p < 0.05, |dz| >= 0.8 vs baseline
  AND vs the random-head control AND vs the wrong-span control, and larger
  benefit on the baseline-wrong subset than the baseline-right subset.
- Negative or partial results are reported as such.

## Scope

Claims are limited to Pythia-2.8B under this substrate. No generality
claim to other dense models, longer contexts, generation mode, or
naturalistic tasks.
