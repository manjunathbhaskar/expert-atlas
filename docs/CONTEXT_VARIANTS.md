# Beyond the lexical needle: paraphrase and multi-hop variants (OLMoE)

**Verdict, two-sided.** (1) The lexical detector's scope limit is now
measured, not hypothesized: it still drives full repair on *paraphrased*
questions (100% hit, 100.8% of oracle) but fails completely on *multi-hop*
retrieval (0% hit — it locks onto the bridge sentence, exactly the
registered failure mode). (2) The registered L8 source-probe fallback earns
its place precisely there: 90.6% hit rate on multi-hop, 61.4% of the oracle
effect, dz=1.20–1.35, p<0.0005 — passing the project bar where the lexical
detector cannot. (3) A finding that reframes both: on these harder tasks
OLMoE has a **capability floor, not context rot** — accuracy is 0–12.5%
even at 256 tokens — yet the span-targeted head boost repairs prompts the
model could never do unassisted (oracle 32/32 on both variants).

## Design (preregistered in `scripts/run_context_variants.py`)

`probes/probe_set_context_variants.yaml` (seed 7): 160 prompts, 2 variants
x depths {0.15, 0.50} x buckets {256, 3840} + a 1024 dev arm, 8 replicates,
8 distractors, 2 haystack domains.

- **paraphrase**: needle "Staff entering the {entity} branch must present
  the passphrase {word} at the desk." vs question "What is the security
  codeword for the {entity} office?" — only the entity is shared; codeword/
  passphrase, office/branch are paraphrases.
- **multihop**: two sentences at independent depths — bridge "The {entity}
  office is designated Site {site}." and answer "The security codeword for
  Site {site} is {word}." The answer sentence shares **no** contentful
  token with the question. Registered prediction: the lexical detector
  finds the bridge, not the answer.

Conditions per 3840-bucket prompt (frozen 16 heads, beta=4.0 from
ATTENTION_BOOST_CAUSAL.md): baseline / oracle span / wrong span (>=200
tokens from truth) / lexical-detected span / L8-probe-detected span.
Detector widths calibrated on the dev arm only. Bar: paired sign-flip
p<0.05 AND |dz|>=0.8 vs wrong-span. 2000 perms, seed 0.

**L8 probe honesty note**: the fallback is a logistic probe on layer-8
residuals, trained on the dev arm *using ground-truth span labels* (needle
positions as positives). It is label-free at *evaluation* time but, unlike
the lexical detector (zero training, zero forward passes), it requires a
small labeled dev set per task family. That distinction is part of the
result, not a footnote.

## Stage 0 — the substrate is a capability floor, not rot

| variant | depth | acc @256 | acc @3840 |
|---|---|---|---|
| paraphrase | 0.15 | 0.125 | 0.000 |
| paraphrase | 0.50 | 0.062 | 0.000 |
| multihop | 0.15 | 0.125 | 0.000 |
| multihop | 0.50 | 0.000 | 0.062 |

At/below the 1/5 forced-choice chance floor even short. There is no
length-dependent degradation to "recover" — the model cannot do these tasks
unassisted at any length tested. All boost results below are therefore
reported as *assisted capability*, not rot repair.

## Paraphrase (n=32 eval @3840)

| condition | acc | mean prob |
|---|---|---|
| baseline | 0.000 | 0.069 |
| oracle span | **1.000** | 0.860 |
| wrong span | 0.000 | 0.074 |
| lexical span | **1.000** | 0.866 |
| L8 probe span | 0.375 | 0.348 |

- Lexical: 32/32 hits (the entity is anchor enough), **100.8% of oracle**,
  vs wrong-span dz=6.46, p<0.0005. The detector survives paraphrase.
- L8 probe: 37.5% hit rate held-out (despite 1.000 on dev) — bimodal as
  always: hits repair, misses stay broken.
- Wrong span repairs 0/32: the effect is span-specific, not generic.

## Multi-hop (n=32 eval @3840)

| condition | acc | mean prob |
|---|---|---|
| baseline | 0.031 | 0.079 |
| oracle span (answer sentence) | **1.000** | 0.744 |
| wrong span | 0.031 | 0.086 |
| lexical span | 0.031 | 0.073 |
| L8 probe span | **0.688** | 0.487 |

- Lexical: **0% answer-span hits**; 56% of its picks are the *bridge*
  sentence — the registered failure mode, confirmed. Effect vs wrong-span
  dz=-0.29, ns. Boosting the bridge does not repair (the model still fails
  the second hop on its own).
- L8 probe: **90.6% hit rate**, failing-subset repair 22/31, **61.4% of the
  oracle effect**, vs wrong-span dz=1.20 (full) / 1.35 (failing), p<0.0005
  — passes the registered bar. The residual stream at L8 marks the answer
  sentence even when no lexical route exists.
- Even with a hit, repair is not always total (oracle mean prob is 0.744
  here vs 0.86–0.99 elsewhere): forcing attention onto the answer sentence
  supplies the fact, but the model must still resolve the bridge.

## What this changes

1. The lexical detector's scope is now a measured boundary: robust to
   paraphrase, broken by compositional (multi-hop) structure.
2. The two detectors are complementary, and the union covers both variants:
   lexical (training-free) for anything with a question anchor, L8
   source-probe (needs a labeled dev set) for compositional retrieval.
3. The head-boost mechanism itself generalizes past its origin: it repairs
   tasks the model fails at *every* length, i.e. the identified transport
   heads are a general lever for directing retrieval, not only a rot fix.
4. Honest gaps: synthetic templates only (no naturalistic multi-document
   task yet); the capability floor means these variants say nothing about
   length-dependent rot; L8 probe needs per-task labeled dev data; n=32
   per variant.

## Reproduction

```bash
HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 .venv/bin/python \
  probes/generate_context_probes_variants.py         # seed 7
HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 .venv/bin/python \
  scripts/run_context_variants.py --stage all
```

Output: `data/context_variants.json`.
