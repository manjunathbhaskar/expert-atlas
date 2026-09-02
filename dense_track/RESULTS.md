# Dense track results: Pythia-2.8B

Parallel experiment track testing whether the causal pattern found on OLMoE
(retrieval-head attention collapse at long context, repairable by an
oracle-span attention boost) also appears in a small dense transformer.

Model: `EleutherAI/pythia-2.8b` (GPT-NeoX, dense, 32 layers x 32 heads = 1024
head cells, 2048-token context, CPU fp32, teacher-forced, deterministic).
Design registered before long-context results in `REGISTRATION.md`.
Raw outputs in `dense_track/data/`.

## 1. Registered substrate: Pythia-2.8B does NOT context-rot

192 prompts (16 replicates x 2 haystacks x {0, 8} distractors x
{256, 1024, 1900} tokens), same templates, candidate pool (chance = 0.125),
needle depth 0.50, and forced-choice metric as the OLMoE substrate; 1900
tokens is ~93% of Pythia's range, matching OLMoE's 3840/4096 long bucket.

| bucket | acc (all) | acc dist=0 | acc dist=8 |
|---|---|---|---|
| 256 | 0.938 | 1.000 | 0.875 |
| 1024 | 0.938 | 1.000 | 0.875 |
| 1900 | 0.984 | 1.000 | 0.969 |

Accuracy does not fall with length (OLMoE: 0.938 -> 0.688). Only 1/64 long
prompts fail — the registered fallback case ("model does not context-rot on
this substrate"), the same outcome Granite gave on its substrate.

## 2. Retrieval heads exist and are cleanly identifiable

Stage 1 (registered): rank all 1024 head cells by mean final-position
attention mass on the needle span over the 60 model-correct 256-token
prompts; take the top K=16.

- Top cell (L13, H9) carries mean mass 0.881; the 16th carries 0.526.
  Chance mass (needle tokens / context) is 0.047 short, 0.006 long.
- Cells concentrate in mid-stack layers (12/16 in layers 10-18 of 32),
  mirroring OLMoE (13/16 in layers 9-14 of 16).
- Identified cells: `transport.json` `top_cells`.

## 3. Localized collapse on the registered set: not testable (n_wrong = 1)

Long bucket has 63 right / 1 wrong, so the registered right-vs-wrong
contrast has no power. Descriptively, the one failing prompt has identified
mass 0.409 vs 0.728 for right prompts, while non-identified cells barely
differ (0.024 vs 0.050). Specificity of the drop concentrating on the
identified 16 cells vs 2000 random 16-cell subsets: observed excess drop
0.293 vs null p95 = 0.034, p < 0.0005 — but with n_wrong = 1 this is
descriptive only. No collapse effect-size claim is made on this set.

## 4. Oracle boost on the registered set: significant but below the bar

Beta calibrated on the 1024 DEV bucket (beta* = 4.0); evaluated on all 64
1900-token prompts; conditions: identified heads + true span ("heads"),
16 random cells ("random"), identified heads + non-needle span ("wrong").

| condition | acc | mean p(answer) |
|---|---|---|
| baseline | 0.984 | 0.945 |
| heads | 1.000 | 0.977 |
| random | 1.000 | 0.948 |
| wrong span | 0.984 | 0.946 |

Paired on p(answer): heads vs baseline dz = 0.33, vs random dz = 0.46, vs
wrong span dz = 0.32 (all perm p <= 0.0005, n = 64). All below the
registered |dz| >= 0.8 floor: with baseline near ceiling there is almost
nothing to repair. The single failing prompt: 0.257 -> 0.923 under the head
boost vs 0.581 random / 0.258 wrong span (anecdote, n = 1). **The registered
causal bar was NOT met**, so the registered span-free stage did not trigger.

## 5. EXPLORATORY harder arm: rot appears and the frozen heads repair it

Declared exploratory before results (`generate_probes_hard.py`): one knob
changed — the 8 distractors use the SAME entity as the needle with a
confusable attribute ("The visitor codeword for the Zurich office is
copper."), keeping the task uniquely answerable. 64 prompts
(16 replicates x 2 haystacks x {256, 1900}). Head set and beta* remain
FROZEN from the registered pipeline; nothing was re-identified or
re-calibrated on this set.

### 5a. It rots

acc 256 = 1.000 -> acc 1900 = 0.781 (25 right / 7 wrong at 1900).

### 5b. The SAME 16 heads show localized collapse (`transport_hard.json`)

- Identified-cell needle mass: right 0.418 vs wrong 0.352, Cohen d = 0.80,
  perm p = 0.036.
- Non-identified cells: 0.0242 vs 0.0231 (no drop).
- Specificity vs 2000 random 16-cell subsets: observed drop 0.0656, null
  mean 0.0013, null p95 0.0146, p < 0.0005.

### 5c. The frozen oracle boost repairs every failure (`boost_hard.json`)

| condition | acc | mean p(answer) |
|---|---|---|
| baseline | 0.781 | 0.578 |
| heads | **1.000** | 0.864 |
| random | 0.813 | 0.591 |
| wrong span | 0.781 | 0.576 |

Full set (n = 32): heads vs baseline dz = 1.47, vs random dz = 1.38, vs
wrong span dz = 1.45 (all perm p <= 0.0005).
Failing subset (n = 7): 7/7 repaired (random control: 2/7; wrong span: 0/7);
p(answer) 0.259 -> 0.760; dz = 4.01 vs baseline (p = 0.016), dz = 4.04 vs
random (p = 0.014). This clears every element of the registered causal bar,
but on an exploratory substrate, so it is reported as exploratory.

### 5d. Span-free lexical detector + boost (`spanfree_hard.json`)

Detector and width selected on the REGISTERED set's DEV bucket (no tuning on
the hard set); same frozen heads and beta*.

Width* = 24 (all widths hit 16/16 on the registered DEV bucket; ties break
toward the largest width, as registered).

| condition | acc | mean p(answer) | failing subset (n = 7) acc |
|---|---|---|---|
| baseline | 0.781 | 0.578 | 0.000 |
| lexical detector + boost | 0.969 | 0.731 | **0.857 (6/7 repaired)** |
| oracle | 1.000 | 0.864 | 1.000 |
| wrong span | 0.781 | 0.576 | 0.000 |

Span hit rate on the hard set: 0.50 (the same-entity distractors make
lexical detection genuinely harder). Full set (n = 32): lexical vs baseline
dz = 0.76 (perm p <= 0.0005), recovering 53.7% of the oracle effect.
Failing subset: lexical vs wrong-span dz = 1.40, p = 0.034. Exploratory,
but the label-free detector repairs 6/7 failures despite hitting the true
span only half the time.

## 6. What this does and does not show

**Shows (under the tested conditions):**

- A dense 2.8B GPT-NeoX model has a small set of mid-stack retrieval heads
  identifiable by needle attention on short correct prompts, like OLMoE.
- On a substrate hard enough to induce failures, those same heads lose
  needle attention specifically on failing prompts (d = 0.80, specificity
  p < 0.0005), and adding pre-softmax attention toward the true needle span
  on exactly those 16 of 1024 cells repairs 7/7 failures while matched
  random-head and wrong-span controls do not. The attention-transport link
  of the OLMoE causal chain therefore does not require MoE routing.

**Does not show:**

- Pythia-2.8B does not context-rot on the registered OLMoE-style substrate
  within its 2048-token range; the repair evidence comes from the
  exploratory confusable-distractor arm with frozen components, not from
  the registered test, and the registered causal bar was not met.
- Nothing here speaks to the router-starvation part of the MoE chain
  (dense models have no router), to other dense models, sizes, longer
  contexts, generation mode, or naturalistic tasks.
- The distractor-confusion failure mode induced here may differ from pure
  length-driven rot; on this substrate length alone (dist = 8, generic
  distractors) was not sufficient to cause failures.
