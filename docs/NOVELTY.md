# NOVELTY — what this project claims, and what it does not

Last updated 2026-08-26. Companion to `docs/FINDINGS.md` (results),
`docs/METHOD.md` (how), `docs/HANDOVER.md` (state).

---

## From specialization mapping to mechanism

**This project started by asking whether MoE experts specialise.** It built an
instrument to measure that properly — base-rate-corrected lift instead of raw
routing heat, a factorial probe set that separates topic from language from
register from format, BH-FDR across ~10k simultaneous tests, and a practical
effect-size floor on top of significance.

**That question is no longer the contribution.** Several parallel 2026 efforts
answer it — EASY-EP (NeurIPS 2025) publishes domain-specific expert pruning,
Priyanshu & Vijay shipped 200+ domain-pruned checkpoints with a routing
dashboard, and "Half the Experts, All the Code" (arXiv 2607.16721) does the
coding-domain version with open code. Our H1 result (557/1024 experts carry a
>=2x, FDR-significant topic affinity) is a careful *replication* with better
statistical hygiene, not a novel claim. It should be presented that way.

**The question this project now asks is why performance degrades as more is fed
into an MoE model** — and specifically whether that degradation has a
*routing-level* mechanism. Two named papers leave exactly this open:

- **Chroma, "Context Rot"** measured accuracy decay with input length across 18
  models (including the MoE Qwen3-235B-A22B) at fixed task difficulty, and said:
  > "we do not have a definitive answer for why that occurs... investigating
  > these effects would require a deeper investigation into mechanistic
  > interpretability, which is beyond the scope of this report"
- **"Continual Pre-training of MoEs: How robust is your router?"** (arXiv
  2503.05029) found early-layer routing instability *correlates* with
  forgetting on real MoE models, but explicitly did **not** establish a
  quantitative relationship between routing overlap and forgetting severity.
- **"Theory on MoE in Continual Learning"** (arXiv 2406.16437, ICLR 2025) proved
  forgetting decomposes via a task-overlap interference functional — but
  validated only on synthetic linear models and small ResNet/MNIST/CIFAR runs,
  never on a real pretrained LLM's routing data.

The pivot is from *"do experts specialise"* (answered, in parallel, by others)
to *"what breaks when you feed the model more"* (open, and named as open).

---

## Verdicts

Every workstream gets one of: **mechanism found / mechanism ruled out /
partial / null**. No result is upgraded beyond what its numbers support.

| WS | Question | Verdict |
|----|----------|---------|
| 1 | Does context rot have a routing-level mechanism? | **PARTIAL — promising but conditional** |
| 2 | Does routing overlap quantitatively predict interference? | **NOT YET TESTED — infrastructure only** |
| 3 | Does specialisation concentrate load onto hot experts? | **MECHANISM RULED OUT** |

---

### WS1 — context rot. Verdict: PARTIAL, and the caveat is load-bearing

**Two findings that must be read together.**

**(a) The accuracy effect is real but under our own pre-registered bar.**
On the hard variant (192 prompts, 8 distractors), needle accuracy falls
**93.8% -> 68.8%** from 256 to 3840 tokens. That clears BH-FDR (perm p=0.008)
but gives **Cohen's d = -0.67 against a pre-registered floor of 0.8**, and
`answer_prob` — the graded, more sensitive measure — is outright FLAT
(rho=-0.090, p=0.196).

`docs/CONTEXT_ROT_HARD.md` therefore records the verdict **"SUBSTRATE CANNOT
TEST THE QUESTION"**, honouring the bar that was fixed in advance. That is the
right call and it is not softened here.

**Note an internal tension, stated rather than hidden:** `docs/MECHANISM.md`
describes the same effect as "a real, FDR-significant, near-trend-threshold
effect." Both descriptions are accurate about different things — significance
vs effect size — but a reader could take the second as stronger than the
pre-registered rule allows. **The binding statement is the pre-registered one:
the accuracy effect did not clear the effect-size floor.**

**(b) Conditional on there being an effect, the mechanism is specific — and
this is the genuinely novel part.** Per-prompt, controlling for length via
rank-residual partial correlation:

| metric | raw rho | partial rho (length controlled) | p |
|---|---|---|---|
| `needle_affinity_rate` | +0.617 | **+0.640** | <0.0001 |
| `mass_q` | +0.578 | +0.586 | <0.0001 |
| `entropy_needle` | -0.589 | -0.570 | <0.0001 |
| `entropy_all` | -0.039 | +0.072 | 0.32 (ns) |

The contrast is the result. **`entropy_all` has the largest length-trend of any
metric (rho=0.887 vs length) yet does not predict which prompts fail.**
`needle_affinity_rate` — the rate at which the router sends needle tokens to
the experts affine to that content — predicts correctness at **rho=0.64,
independent of length**.

So *if* there is degradation here, it is **not generic router confusion**; it is
the router losing a specific specialist pathway. That distinction is finer than
anything in the cited prior art, and it is directly actionable: it names a
quantity to intervene on.

**What would upgrade this to "mechanism found":** (i) a substrate where the
accuracy effect clears the effect-size floor, and (ii) the causal test —
forcing `needle_affinity_rate` up and measuring whether accuracy recovers.
The steering primitive (`expertatlas/steering.py`, pre-selection router-logit
mask/boost) exists and the causal run is **in progress at time of writing**;
its result is not included here and must not be assumed.

---

### WS2 — interference prediction. Verdict: NOT YET TESTED

**No predictive claim is made, because the experiment has not been run.**
`data/ablation_multi.jsonl` does not exist.

What was built:
- An interference functional over real per-token routing distributions, with the
  mapping from arXiv 2406.16437's linear-model math documented explicitly.
- Pairwise overlap predictors, reproduced against `docs/ORTHOGONALITY.md`.
- A **matched-load null** (`interference.py::matched_load_null_sets`) that WS3's
  finding made necessary — see below.

**One real methodological result did come out of the setup.** Every null in this
project had been *size*-matched only. But per-domain load removed spans **5.3x**
(sql 192.4 fair-shares vs history 36.3, against a random expectation of 100), so
a size-matched null cannot separate "these experts matter" from "this ablation
deleted twice as much of the routed network." Validated: for `sql`, the
size-matched null is biased **-92.3 fair-shares** against target; the matched-load
null is **-2.9** — a 32x reduction, all six domains feasible.

The overlap/load collinearity was also quantified, and is milder than first
feared: on **directed** pairs (ablate A, score B — the real experimental unit)
`r=+0.564, R^2=0.318, VIF=1.47`, comfortably under the VIF>5 danger line, because
load depends only on the ablator while overlap varies with the pair. An earlier
warning in `docs/HANDOVER.md` that the two would be inseparable was based on
undirected pairs and was too pessimistic.

**When the sweep runs, the regression must include `load_removed` as a
covariate.** Reporting an overlap coefficient without it would be confounded.

---

### WS3 — hot experts. Verdict: MECHANISM RULED OUT

The commissioning brief hypothesised that *"specialization concentrates load
onto a small load-bearing subset... more load keeps landing on the same
specialized few."* **The data says the opposite.**

| | value |
|---|---|
| hot experts (>= 2x fair share) | 100 / 1024 |
| H1 specialists among them | **34** |
| expected by chance | 54.5 +/- 4.7 |
| enrichment | **0.624x** |
| permutation p | **< 0.0001** |

Specialists are disproportionately **cold**. The hot experts are largely
generalists. The proposed mechanism is not available on this substrate.

**Confound check** (because lift is base-rate corrected, so usage sits in its
denominator): `spearman(load, max|lift|) = -0.546` (p=1e-80) but
`spearman(load, n_significant_domains) = -0.022` (p=0.48). FDR *detection* of
specialisation is usage-independent; only its *magnitude* varies. So this is not
a statistical-power artifact.

**Honest limit, repeated from `docs/UTILIZATION.md`:** `load_ratio` and
`max|lift|` derive from the same count matrix and are partly *definitionally*
opposed — an expert firing 3x above fair share everywhere cannot also show 2x
concentration in one domain. Read this as "load and specialisation are opposed
on this substrate," **not** as two independent variables that happen to
anti-correlate. The non-circular part is the FDR-vs-usage null.

A ruled-out mechanism is a real contribution: it removes the cheapest available
explanation and redirects WS1/WS2 away from it.

---

## What this project does NOT claim

1. **Not that expert specialisation is a novel discovery.** It is replicated
   here with better hygiene; others published it in parallel.
2. **Not that context rot is explained.** WS1 is PARTIAL and the accuracy effect
   did not clear the pre-registered effect-size floor.
3. **Not that routing overlap predicts interference.** That experiment has not
   been run.
4. **Not that any of this generalises beyond OLMoE-1B-7B-0924.** One model, one
   seed. PLAN.md §9b calls the second-model check non-optional and it remains
   undone. OLMoE has 64 experts/layer; frontier MoEs have hundreds. **This is
   the single largest threat to every claim above.**
5. **Not that co-activation communities are meaningful here.** H4 measured
   227x usage skew against `coactivation.py`'s own documented 2.0x PMI validity
   limit, and is reported UNRELIABLE regardless of which way the raw comparison
   fell.

---

## The defensible one-sentence claim

> On one small open MoE, degradation under longer input is predicted by the
> router failing to keep content-relevant tokens on their specialist experts
> (partial rho=0.64, length-controlled) rather than by any global loss of router
> decisiveness (`entropy_all`, ns) — a distinction the context-rot literature
> names as open, measured with a base-rate-corrected, FDR-controlled instrument,
> and reported against an accuracy effect that did not itself clear this
> project's pre-registered effect-size floor.

Every qualifier in that sentence is load-bearing. Removing any one of them
overstates the result.
