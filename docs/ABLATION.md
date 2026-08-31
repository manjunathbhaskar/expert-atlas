# Ablation harness -- causal test (Tier 3, TRANSFER.md §6.6)

Target domain: `medicine` (189 experts ablated). Control domain: `cooking` (170 experts ablated in the ablate_other_domain condition). Metric: mean per-token teacher-forced cross-entropy (nats), held-out (split=B) prompts, forward passes only.

**The causal claim requires ALL of:** ablate_target hurts target text more than it hurts control text, AND more than ablate_random hurts target text, AND more than ablate_other_domain hurts target text. Reporting one flattering number alone would not support a causal claim -- all four deltas below are load-bearing.

## Results

| condition | loss on target text | delta vs baseline | loss on control text | delta vs baseline |
|---|---|---|---|---|
| baseline | 2.9779 | +0.0000 | 2.8796 | +0.0000 |
| ablate_target | 3.7041 | +0.7262 | 3.4685 | +0.5889 |
| ablate_random | 3.2340 | +0.2560 | 3.0962 | +0.2167 |
| ablate_other_domain | 3.6103 | +0.6324 | 3.9119 | +1.0323 |

## Verdict
- ablate_target hurts target text more than it hurts control text: 0.7262 vs 0.5889 -- YES
- ablate_target hurts target text more than ablate_random does: 0.7262 vs 0.2560 -- YES
- ablate_target hurts target text more than ablate_other_domain does: 0.7262 vs 0.6324 -- YES

**CAUSAL claim supported on this single run.** No significance test is applied here (no null distribution over ablation sets was built -- this is a first pass, not a finished statistical test); treat this as directional evidence, not a p-value-backed claim. A proper version would repeat ablate_random over many random sets to build a null and report where ablate_target falls in that distribution, and would run on more than one target domain.

## Double dissociation (noticed after the fact, not designed in -- worth more than the single-direction check above)

`ablate_other_domain` ablates cooking's meaningful experts and is scored on BOTH texts too.
Reading that row the other way round gives an independent, second test in the opposite
direction:

- Ablating **medicine** experts hurts medicine text (+0.726) more than cooking text (+0.589).
- Ablating **cooking** experts hurts cooking text (+1.032) more than medicine text (+0.632).

Each domain's own experts matter most for its own text, in both directions — a classic
double-dissociation pattern (the standard evidentiary structure in lesion studies:
damage to region A impairs task A more than task B, and damage to region B impairs task B
more than task A). This is more convincing than either single-direction result alone,
because a pattern that were really just "ablating any 189 topically-loaded experts causes
generic damage" would not produce this crossover — it would hurt both texts roughly
equally regardless of which expert set was cut.

**Still not a finished result:** one model, one seed, n=6 held-out prompts per domain, one
domain pair, no repeated-random-draw null, no significance test. Directional evidence
worth a properly powered follow-up (more prompts, several domain pairs, a real null over
many random 189-expert draws), not a publishable claim on its own.
