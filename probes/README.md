# Probe set — design rationale (WS-B)

## The question this set exists to answer

Mixtral's routing analysis found expert assignment aligned with **syntax**, not topic
([arXiv 2401.04088](https://arxiv.org/abs/2401.04088)). The OLMoE paper reports **domain
and vocabulary** specialisation ([arXiv 2409.02060](https://arxiv.org/abs/2409.02060)).
Both cannot be the whole story.

A one-dimensional probe set ("10 categories × 3 prompts") **cannot distinguish them**,
because in such a set topic, language, register and format all move together. If an expert
fires on your Python prompts, a flat design cannot tell you whether it is a *Python* expert,
a *code-syntax* expert, an *English technical prose* expert, or a *formal register* expert.

This set is factorial so those four factors can be marginalised independently.

## Design

| factor | levels | n |
|---|---|---|
| `topic` | python, rust, sql, math_proof, law, medicine, music_theory, cooking, history, poetry | 10 |
| `lang` | en, zh, de, ja | 4 |
| `register` | formal, casual | 2 |
| `format` | prose, json, bulleted | 3 |

Full crossing = **240 cells**, 2 prompts per cell = **480 prompts**.

## The control that makes it work

For the four syntax-heavy topics (python, rust, sql, math_proof), **the embedded code or
notation is byte-identical across all four language cells.** Only the surrounding natural
language changes.

This is a within-subjects control and it is the scientific core of the set:

- An expert firing on the code tokens **regardless of surrounding language** → syntax expert.
- An expert firing on the prose but not the code, varying by language → language expert.
- An expert firing on Python code but not Rust or SQL code → genuine topic expert.

No published probe set does this, and it is what lets the atlas make a claim stronger than
"experts differ somewhat."

## Secondary contrast: `format` crosses the code/prose divide

`format=json` applies to *every* topic, including the prose ones. So a "structured output"
expert (fires on braces, quotes, delimiters) is separable from a "programming language"
expert (fires on `def`, `fn`, `SELECT`). Without this, the two are hopelessly confounded —
and conflating them is the most likely way to publish a wrong atlas.

## Known limitations — state these in the paper

1. **Translations are model-assisted, not native-reviewed.** zh/ja/de framing text should
   be checked by a native speaker before publication. Errors would appear as spurious
   language-affinity. Flagged in `probe_set_v1.yaml` as `translation_reviewed: false`.
2. **Not all cells are equally natural.** "Casual Japanese JSON about music theory" is a
   somewhat artificial request. This is accepted: the factorial's value is separability,
   and naturalness is sacrificed deliberately. Report it.
3. **Prompts are teacher-forced, not generated from.** We measure routing over *given*
   text. Routing during free generation may differ — out of scope for v1.
4. **Two prompts per cell is thin** for per-cell estimates. The design is powered for
   *marginal* effects (topic averaged over lang/register/format), which is what the
   hypotheses in PLAN.md §1 actually test. Do not over-interpret single cells.

## Held-out split

`split: A|B` is declared per prompt, balanced within every cell. Hypothesis **H6**
(split-half replication) uses it, and H6 is the gate for the entire project: if lift
vectors computed on A do not correlate with those from B, everything downstream is noise.

## Measured confound: prompt length varies by topic and by language

Run `python probes/validate.py` to reproduce.

| axis | ratio | cause |
|---|---|---|
| longest topic (`math_proof`) / shortest (`history`) | **1.73×** | payload sizes differ (code/notation blocks vs one-line scenarios) |
| `ja` / `en` mean tokens | **1.53×** | OLMoE BPE is English-centric |
| `zh` / `en` | 1.38× | same |
| `de` / `en` | 1.27× | same |

**Why this matters.** Every token is an independent routing observation. Pooled naively,
`math_proof` contributes 73% more observations than `history`, and any expert that responds
to *sequence position* or *long context* would appear as a topic specialist. That is a
plausible way to produce a confident, wrong atlas.

**Required mitigation (implemented in WS-C, not here):** compute lift over an **equal token
budget per cell**. Subsample each (topic, lang, register, format) cell to the minimum token
count across cells, with a fixed seed, before accumulating counts. Report the budget in
`atlas.json` under `stats.tokens_per_cell`.

Do **not** "fix" this by padding or truncating prompts — that would damage the payload
invariance control, which matters more. Handle it at the counting stage.
