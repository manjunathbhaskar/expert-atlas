# Context rot, traced to the router — and past it

*The complete arc of this project's context-rot investigation, in one place.
Every number below is reported in full, with its null and its limits, in the
five source documents: `CONTEXT_ROT_HARD.md`, `MECHANISM.md`,
`CONTEXT_PATHWAY.md`, `MECHANISM_CAUSAL.md`, `ADAPTIVE_CAUSAL.md`. Nothing
here supersedes them; this is the narrative they add up to.*

## Scope, before the story

One model (OLMoE-1B-7B-0924, 16 layers x 64 experts, top-8 routing), one seed,
CPU, teacher-forced forced-choice scoring, needle depth fixed at 50%, n=192
prompts. Everything below is bounded by that. Each claim cleared this
project's standard bar — a permutation null (>=2000 shuffles), BH-FDR at
q=0.05 across the metric family, AND an effect-size floor — or it is labeled
as having failed it.

## 1. The open question

Chroma's *Context Rot* report (2025) measured, across 18 models, that accuracy
degrades as input grows **even when task difficulty is held fixed**. On why,
they wrote:

> "we do not have a definitive answer for why that occurs... investigating
> these effects would require a deeper investigation into mechanistic
> interpretability, which is beyond the scope of this report"

That is the question this project took up, on the one component a MoE has
that a dense model does not: the router. A 192-prompt needle-in-a-haystack
substrate reproducing two of Chroma's conditions (similar/dissimilar haystack,
0/8 distractors) across six length buckets reproduced the phenomenon locally:
forced-choice accuracy falls from **93.8% at 256 tokens to 68.8% at 3,840**
(FDR-significant), with the entire drop carried by the distractor conditions
(`CONTEXT_ROT_HARD.md`).

## 2. The diagnosis: one specific pathway, not general confusion

The obvious suspect — the router getting globally "less decisive" with length
— is real but turned out to be a red herring. Whole-prompt router entropy has
the *largest* length trend of any metric (rho=+0.854 vs length), yet once
length is controlled it does **not** predict which prompts fail (partial
rho=+0.072, permutation p=0.325).

What does predict failure is much narrower (`MECHANISM.md`): the rate at which
the router's top-k selections, at the needle's own tokens, land on the small
set of experts affine to the needle's content (~75/1024, defined by
base-rate-corrected lift at the shortest bucket and held fixed).

| metric | partial rho vs answer_prob (length controlled) | p |
|---|---|---|
| `needle_affinity_rate` | **+0.640** | <0.0001 |
| `entropy_all` (global confusion) | +0.072 | 0.32 (ns) |

`CONTEXT_PATHWAY.md` sharpened this by decomposing the routing shift: with
length, needle-window routing mass migrates OUT of the needle-affine
specialist set (0.0417 → 0.0335, d=-1.39, FDR-sig) and INTO hot generalist
experts (0.1120 → 0.1407, d=+1.33, FDR-sig) — but only the specialist loss
tracks correctness (partial rho +0.651, p=0.0005); the generalist takeover
predicts nothing once length is controlled (partial +0.076, p=0.28). The
generalist influx is a bystander; the specialist dropout is the correlate.

This is more specific than anything in the length-degradation literature we
could find: not "the model gets confused", but "the router stops delivering
one identifiable specialist pathway at the retrieval-critical tokens, and
prompts where that happens are the prompts that fail, independent of length."

## 3. The two fixes that failed

If losing the pathway causes the failure, forcing it back should recover
accuracy. That was tested twice, with the full control structure (matched
per-layer random-expert boosts, length-matched low/high-affinity subsets,
accuracy-blind magnitude calibration on a never-evaluated dev bucket):

**Fix attempt 1 — always-on boost at the needle window**
(`MECHANISM_CAUSAL.md`). +delta on the needle-affine experts' router logits
across the needle's ~12 tokens, at two magnitudes. The manipulation check
passed decisively (needle-affine selection went from ~0.03 to 0.28–0.58). The
outcome did not: the large boost was catastrophic (answer_prob 0.506 → 0.011
on the target subset) and statistically indistinguishable from boosting
random experts — indiscriminate damage, not repair. The small boost was
gentle and not significant (p=0.11).

**Fix attempt 2 — entropy-triggered adaptive boost** (`ADAPTIVE_CAUSAL.md`).
The policy `MECHANISM.md` had actually proposed: boost only at tokens where
that layer's router entropy exceeds an accuracy-blind p90 threshold, whole
sequence, small magnitude. Again the manipulation worked — the trigger fired
on 17–19% of cells, needle-affine selection roughly doubled under treatment
and was unchanged under the random control — and again accuracy did not
move: +0.028 answer_prob on the target subset, p=0.40, dz=+0.25,
indistinguishable from the random control.

Both experiments share the same shape, and that shape is the finding: **the
routing intervention did exactly what it was designed to do at the router,
and the model's answers did not improve.**

## 4. Why the failures matter

A correlation with a strong effect size, plus two well-controlled
interventions that move the correlate without moving the outcome, is real
information about where the problem lives:

- **The router is (at these magnitudes and designs) ruled out as the fixable
  layer.** If restoring specialist selection were sufficient, one of these
  interventions should have shown it — the manipulation checks prove the
  restoration happened.
- **The pathway loss is therefore best read as a symptom, not a cause.** The
  router selects experts from the hidden state it is handed. If that hidden
  state no longer carries the needle's content distinctly by the time it
  reaches the gate, then routing "correctly" delivers the right experts the
  wrong input — which is exactly what both null results look like.
- That points the search upstream: **attention (is the needle's content still
  being carried forward to its position?) or the residual stream itself (has
  the representation faded?).** Distinguishing those two is a probing
  question, not a routing question.

Ruling a layer out with controlled evidence is how a mechanism search is
supposed to narrow. A field where only the successful intervention gets
written up cannot do this; these two documents exist so nobody re-runs the
obvious experiment.

## 5. Where this stands

- **Named and characterized**: context rot on this model is the
  length-dependent loss of one specific specialist routing pathway at the
  retrieval-critical tokens, with a length-independent per-prompt correlation
  to failure (partial rho ~0.64–0.65) — not global router confusion, and not
  a generalist takeover.
- **Ruled out**: repairing it at the router, always-on or entropy-triggered,
  at a calibrated gentle magnitude and a 4x magnitude.
- **Not yet established**: where upstream the degradation originates
  (attention vs residual stream), whether the needle's information is still
  linearly recoverable at the failing prompts' hidden states, whether the
  same mechanism appears on a second architecture (Granite replicates the
  *specialization* findings, but the mechanism itself has only been tested on
  OLMoE), and how any of this depends on needle depth (fixed at 50%
  throughout).
- **No working fix exists yet.** The honest current position is a precise
  diagnosis, a well-evidenced negative on the most obvious cure, and a
  specific, testable hypothesis about where to look next.

## 6. Postscript: following the evidence upstream found the mechanism

Everything after section 5 was run in evidence order; each step's full
writeup is linked.

- **The fact survives at its source.** A deconfounded cross-pair probe
  (`PROBE_REPAIRED.md`) decodes the needle's content from its own position
  at 99.5% at every length — including on the prompts the model gets wrong.
  The readout position, not the source representation, degrades.
- **Pasting the content at the readout does not help**
  (`ANCHOR_CAUSAL.md`): true-content, wrong-content and random injections
  are indistinguishable — content-independence, pointing at transport.
- **Rot scales with source-to-readout distance** (`CONTEXT_DEPTH.md`):
  at 3840 tokens, accuracy is 0.375/0.563/0.813 for early/middle/late
  needles (p<0.0005).
- **The transport is carried by specific, identifiable heads, and they
  collapse on failing prompts** (`ATTENTION_TRANSPORT.md`): 16 head cells
  identified on short correct prompts put up to 14x-chance attention on
  the needle; on long failing prompts their needle attention drops from
  0.432 to 0.187 (d=1.55), and that drop is concentrated in exactly those
  cells (specificity vs random head sets: p<0.0005).
- **Re-opening those heads repairs the failure**
  (`ATTENTION_BOOST_CAUSAL.md`): boosting the identified heads' attention
  onto the needle restores all 14/14 failing prompts (answer prob
  0.238 -> 0.986, dz=5.55 vs baseline, dz=2.13 vs a strength-matched
  random-head control, p<0.0005). The fourth intervention family tested,
  and the first to beat its controls.

The causal chain, as measured on this model and substrate: the fact stays
intact at its source; specific retrieval heads stop attending to it at
long range; the readout degrades; the router's specialist starvation is a
downstream symptom. The remaining honest gap between this and a deployable
fix is that the repair uses the needle's span — oracle knowledge of where
the fact sits. Locating that span without labels is the open engineering
problem; the mechanism question this document opened with now has a
controlled, positive answer.

The span-free follow-up (`SPANFREE_BOOST.md`) measured how much of the
repair survives without the label. Of three label-free detectors, only the
retrieval heads' own attention works: it finds the needle on 86% of
held-out long prompts, recovers 55% of the oracle effect, repairs 8/14
failing prompts, and beats a strength-matched wrong-span control
(p<=0.02) — real span discovery, not generic perturbation — but its
effect sizes (dz 0.53/0.71) miss the project's 0.8 floor. The remaining
6/14 are exactly the prompts where the detector's own signal has
collapsed: the detector and the failure share a cause. Router-side
signals carry no positional information at all (0% localization). So the
current honest position: attention boosting is solved, span discovery on
the collapsed prompts is the hard remaining problem, and an independent
span signal (or targeted training) is the next escalation.

That escalation closed the gap (`SPAN_DISCOVERY_SOLVED.md`). A detector
that runs no forward pass at all — idf-weighted lexical overlap between
the question and each context window, ties extended forward from the
peak — locates the needle on 100% of prompts on every substrate,
including the depth-0.15 set where the attention detector had collapsed
to 37.5%. Driving the frozen boost with its predicted span repairs
14/14 failing prompts on the held-out 3840 set (100.8% of the oracle
effect; vs wrong-span dz=0.89 full set, dz=5.52 failing subset,
p<0.0005) and 10/10 on depth-0.15 (99.4% of oracle; dz=3.75/10.5). The
detector's signal is the prompt text, which does not degrade with
length — that is why it escapes the circularity that capped the
model-internal detectors. A first version that nominally "hit" the span
but truncated the answer word recovered only 1.3% of the effect and is
logged as a negative. Scope limit, stated plainly: this solves span
discovery for retrieval questions with lexical anchors to the needle;
paraphrase-heavy or multi-hop tasks may need the L8 residual
source-probe, which was registered as the fallback and not needed here.

The full causal chain, with the label removed: **fact survives at
source → specific retrieval heads stop attending at long range →
readout degrades (router starvation downstream) → locate the span from
the question's own rare tokens → re-open the identified heads on that
span → retrieval restored, at oracle strength, on every failing prompt
tested.**

The chain then replicated on a second model (`GRANITE_TRANSPORT.md`).
Granite-3.0-3B-A800M (GQA, 32 layers, a different MoE router) resisted
both length and needle depth — 1.000 accuracy at 3840 tokens at every
depth, a documented negative — and only failed under a registered
distractor-load escalation (24 competing same-template facts: accuracy
0.688). On that substrate the same pipeline, preregistered end to end,
found retrieval heads at the same relative depth of the stack (7/16 in
L22; L22/32 ≈ OLMoE's L12/16), measured a localized collapse on failing
prompts (identified heads 0.225 → 0.101, d=2.17; specificity vs random
head sets p<0.0005), and repaired 5/5 failing prompts by boosting them
(oracle acc 0.688 → 1.000, dz≈0.99–1.10 vs baseline/random/wrong-span
controls, p<0.0005), with the lexical detector recovering 99.1% of the
oracle effect at a 16/16 hit rate. The mechanism, the repair, and the
label-free span discovery are not OLMoE quirks — with the scope note
that Granite's rot trigger is interference under distractor load rather
than distance, and everything is still a lexical needle task at
<=4096 tokens.

Finally, the detector's scope was measured past the lexical needle
(`CONTEXT_VARIANTS.md`). Two harder variants exposed a different regime:
OLMoE fails paraphrased and multi-hop retrieval even at 256 tokens
(0–12.5% — a capability floor, not rot), yet the span-targeted boost
repairs both at 32/32 with the oracle span. The lexical detector
survives paraphrase intact (100% hit, 100.8% of oracle, dz=6.46) but
fails completely on multi-hop (0% hit; it locks onto the bridge
sentence, the failure mode registered before the run) — and there the
L8 residual source-probe fallback, trained on a small labeled dev arm,
takes over: 90.6% hit rate, 61.4% of the oracle effect, dz=1.20–1.35,
p<0.0005, passing the project bar. The two detectors are complementary
and their union covers every task variant tested; the boost itself
turns out to be a general lever for directing retrieval, not only a
rot repair. Open, stated plainly: naturalistic multi-document tasks,
lengths beyond 4096, and a fully training-free detector for
compositional retrieval.
