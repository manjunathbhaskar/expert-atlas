# Dense-track registration (declared BEFORE any long-context measurement)

Goal of this track only: test whether the causal pattern found on OLMoE
(retrieval-head attention collapse causes long-context needle failure;
oracle-span attention boost repairs it) also appears in a small **dense**
transformer. This track is independent: it does not modify the MoE code or
results.

## Registered choices

1. **Model.** EleutherAI/pythia-2.8b — the largest dense model that fits and
   runs on this box (fp32 forward at 1900 tokens: ~14 s, 27.5 GiB peak with
   full attentions; verified before this registration). GPT-NeoX, 32 layers x
   32 heads (1024 head cells), max positions 2048.
2. **Substrate.** `dense_track/probe_set_dense.yaml`: same templates,
   candidate pool (all 8 words verified single-token under the NeoX
   tokenizer), 2 haystack x 2 distractor (0 vs 8) crossing, needle depth
   0.50, byte-identical needle/question across buckets as the OLMoE repaired
   substrate. Buckets (256, 1024, 1900) — the long bucket is capped by the
   model's 2048 positional range (93% of range vs OLMoE's 3840/4096 = 94%).
   192 prompts. Seed 7.
3. **Head identification (stage 1).** SHORT bucket (256) model-CORRECT
   prompts only. Rank all 1024 (layer, head) cells by mean final-position
   attention mass on the needle token span. **K = 16**, fixed in advance,
   same K as OLMoE.
4. **Collapse test (stage 2).** LONG bucket (1900): compare mean needle mass
   of the identified 16 cells between model-right and model-wrong prompts.
   Specificity null: 2000 random 16-cell subsets of the remaining cells;
   observed statistic = (right - wrong drop of identified cells) minus
   (right - wrong drop of the other cells); p from the null.
5. **Oracle boost.** Additive pre-softmax bias `beta` at the identified 16
   cells, queries = question span start through final position, keys = the
   labeled needle token span. `beta` calibrated on the 1024-token DEV bucket
   (first 16 prompt_ids), grid {1.0, 2.0, 4.0}, highest dev forced-choice
   accuracy wins. Evaluation on the 1900 bucket (all 64 prompts). Controls:
   (a) **random-head** — 16 cells drawn uniformly per prompt (fixed seed),
   same beta and spans; (b) **wrong-span** — same 16 identified cells and
   beta, key span = width-16 window drawn uniformly from the context with no
   overlap with the true needle span (fixed seed).
6. **Span-free test (conditional).** Run ONLY if the oracle boost clears the
   causal bar. Detector = the existing lexical idf-overlap detector from
   `scripts/run_span_discovery.py` (v2: ties break toward the LATEST window;
   width chosen on the 1024 DEV bucket from {8, 12, 16, 24}, ties toward the
   largest). Boost frozen from the oracle test.
7. **Statistics.** Forced-choice accuracy (chance 0.125) and answer
   probability over the 8 candidates. Paired sign-flip permutation, 2000
   permutations. Effect-size floor |dz| >= 0.8 (Cohen dz) for causal claims;
   Cohen d >= 0.8 for the group collapse contrast; p < 0.05.
   Causal bar for the boost: p < 0.05 AND |dz| >= 0.8 AND beats the
   random-head control AND helps the model-wrong subset more than the
   model-right subset.
8. **Fallbacks (registered now).** If the long bucket produces fewer than 8
   model-wrong prompts in the distractor arm, the collapse and boost
   contrasts are reported descriptively without effect-size claims and the
   track concludes that the dense model does not context-rot enough on this
   substrate to test the mechanism (the Granite outcome). If short-bucket
   accuracy is below 0.5, stage 1 has too few correct prompts and the track
   stops with that negative reported.

Negative results at any stage are reported as results. No claim of
generality beyond Pythia-2.8B under these exact conditions will be made.
