# Interference — does a pre-computed routing-overlap number predict cross-domain ablation damage?

Workstream 2. Model: OLMoE-1B-7B-0924 (16 layers x 64 experts, top-8). Domains: python, rust, sql, math_proof, history, cooking. Ablation-set size fixed at **m = 100 experts for every domain** (so the random null is size-matched by construction and no damage difference can come from having cut more of the network). Metric: mean per-token teacher-forced cross-entropy (nats) on held-out `split=B` prompts, forward passes only.

**Held-out sample size: 24 prompts per domain (144 scored per sweep).** `docs/ABLATION.md` flagged its n=6 as too small; this is 4x that and is the entire held-out supply the probe set has. **But see Limitations — those 24 prompts are 24 surface variants of ONE content stem, so the effective content-level n is 1, and this raise is smaller than it looks.**

## Headline

Pre-computed overlap (`docs/ORTHOGONALITY.md` lift-cosine, no ablation involved) vs. null-standardised cross-domain damage, over 30 ordered (ablator, victim) domain pairs:

- **Pearson r = 0.816**  (R^2 = 0.666), Spearman rho = 0.759
- OLS slope = 6.779 null-sd of damage per unit cosine (intercept -0.402)
- Mantel permutation p = 0.0139 (exact enumeration of all 720 domain relabellings; **p cannot go below 0.0014 at 6 domains — that is the design's floor, not a result**)
- 95% CI on r, pair bootstrap: [0.727, 0.898] — Fisher-z CI [0.645, 0.909] is narrower and **wrong**, because the 30 pairs are entries of one 6x6 matrix, not 30 independent draws.

### Verdict: **PREDICTS**

The pre-computed overlap number tracks measured cross-domain damage (r = 0.816, Mantel p = 0.0139). Stated with the power limit attached: 6 domains, 30 dependent pairs, one model. This is evidence for the quantitative link 2406.16437 predicts and 2503.05029 stops short of, not a demonstration that it holds generally.

## Side-by-side with the two papers this is positioned against

| | arXiv 2406.16437 (ICLR 2025) | arXiv 2503.05029 | this run |
|---|---|---|---|
| What it is | Theory: MoE in continual learning | Empirical: continual pre-training of MoEs | Empirical: static pretrained MoE |
| Established | Explicit expressions for expected forgetting and generalisation error; experts diversify, router learns to select and balance load; gating update must be terminated for convergence | 500M-active/2B-total MoEs over 600B tokens are surprisingly robust to distribution shift; routing changes concentrate in early layers; "more pronounced changes correlate with higher forgetting" | An overlap number computable from routing traces alone is/is not quantitatively predictive of ablation damage |
| Validated on | Overparameterised linear regression, plus synthetic and small real-dataset DNN experiments. **No real pretrained open-weight LLM's measured routing** | Real MoE LLMs, but models they trained themselves | Real open-weight pretrained LLM (OLMoE-1B-7B-0924), 480-prompt controlled factorial probe set |
| Quantitative overlap -> damage magnitude? | Predicted by theory, never fitted to real routing data | **Explicitly not established** — correlation between an observed post-hoc routing *change* and forgetting, not a pre-computed overlap predicting damage *size* | **This is the step taken here** |
| Causal manipulation | No (analytic) | No (observational over training) | Yes — zero-ablation of a named expert set, with a size-matched random null |

**The differentiation has to be stated precisely: 2503.05029 already gets partway there.** It shows routing behaviour relates to forgetting on real MoEs. What it does not do — and says it does not do — is put a number on the relationship in the predictive direction: given two domains and their routing statistics *before* any intervention, how much damage should you expect? That is the only gap this run addresses, and it addresses it on one model.

## The interference functional and how the linear-model math was mapped onto routing

With experts e in E (|E| = 1024), domain d, and c_d(e) the selection count under the equal-token-budget control in `expertatlas/aggregate.py`:

```
  q_d(e) = (c_d(e) + 1) / (sum_e' c_d(e') + |E|)        routing distribution
  p(e)   = (sum_d c_d(e) + 1) / (grand total + |E|)     base rate
  l_d(e) = log2( q_d(e) / p(e) )                        == stats.compute_lift

  symmetric   I(a,b)    = <l_a, l_b> / (||l_a|| ||l_b||)
  directional M(a -> b) = sum_{e in S_a} q_b(e) / sum_{e in S_a} p(e)
```

`S_a` is the expert set actually ablated for domain a. `M = 1` means the victim domain routes through the ablator's experts at exactly the base rate.

**Faithful to 2406.16437's structure:** interference is a bilinear form between two per-task vectors (not a set-overlap count); the vectors are indexed by expert and weighted by the router's per-task selection probability; zero overlap gives exactly zero predicted interference.

**Adaptations that are our judgement calls, not their theorem:**

1. **Routing distribution substituted for task representation.** They inner-product task feature directions in a shared parameter space. We observe only the router. This is strictly weaker and is the same limit `docs/ORTHOGONALITY.md` already flags: two domains can route identically and still be orthogonal *inside* the experts. If the predictor under-performs, this is candidate reason #1.
2. **Base-rate correction (lift, not raw q).** Their setting has no load-balancing objective forcing a shared near-uniform marginal; a real trained MoE does. Measured here: the raw-q cosine over the same 30 pairs has sd = 0.1023 and range [0.7032, 0.9865] — i.e. essentially constant across every domain pair, exactly as the load-balancing argument predicts. Using it as the predictor would be using a constant. Taking the domain-specific deviation instead is a modelling decision, and it is the decision the whole project already makes.
3. **log-ratio rather than ratio.** l = log2(q/p) for consistency with every other number in this repo, not because their derivation implies a log. The non-log variant is fitted below as a robustness check.
4. **Ablation replaces gradient interference.** They forget via a weight update; we zero-ablate and measure held-out CE. Ablation is the harsher, upper-bound version of "how much of b's computation flows through a's experts". Magnitudes here are NOT comparable to a forgetting curve.
5. **Static model, no task sequence.** Nothing here observes forgetting dynamics.

## Design

- **Expert sets** are the top-100 experts by lift for each domain among those that are BH-FDR significant (q=0.05) and clear |lift| >= 1.0, **computed on `split=A` traces only**. `docs/ABLATION.md` took its sets from `data/atlas.json`, whose lift was fitted on all 480 prompts including the split=B text it then scored — leakage. Fixing that changes the sets; Jaccard against the atlas-derived sets is python 0.02, rust 0.05, sql 0.07, math_proof 0.14, history 0.34, cooking 0.22.
- **Fixed m** (not "all experts over the bar", which gave 189 vs 170 in the prior run) so every ablation removes the same number of experts and one random null per victim domain is valid for all six ablators.
- **Null**: 30 independent uniformly-random sets of exactly 100 experts, each scored on all six domains' held-out prompts. This is the piece `docs/ABLATION.md` named as missing. Percentile resolution is 1/31 = 0.032, so no empirical p below that is reportable.
- **Sweeps run**: 1 baseline + 6 target ablations + 30 null draws = 37 full forward-pass sweeps over 144 prompts each.

## Cross-domain damage matrix

Rows = ablated domain, columns = evaluated (victim) text. Cells are loss(ablate row) - loss(baseline), in nats/token. Diagonal = on-target damage.

| ablate \ eval | python | rust | sql | math_proof | history | cooking |
|---|---|---|---|---|---|---|
| **python** | +1.8971 | +1.7762 | +1.6675 | +0.8846 | +0.1499 | +0.1624 |
| **rust** | +2.0437 | +1.8187 | +1.7645 | +1.0575 | +0.1285 | +0.1212 |
| **sql** | +2.0149 | +1.9217 | +2.3340 | +0.7284 | +0.1544 | +0.1525 |
| **math_proof** | +0.4075 | +0.4934 | +0.5343 | +1.1987 | +0.2846 | +0.1254 |
| **history** | +0.0001 | -0.0193 | +0.0373 | -0.0229 | +0.2476 | +0.0280 |
| **cooking** | +0.0354 | +0.0130 | +0.0389 | -0.0164 | +0.1576 | +0.6758 |

Same matrix in units of the matched-size random null's standard deviation (z = (observed - null mean) / null sd, per victim column):

| ablate \ eval | python | rust | sql | math_proof | history | cooking |
|---|---|---|---|---|---|---|
| **python** | +9.39 | +5.89 | +7.90 | +3.46 | -0.49 | -0.56 |
| **rust** | +10.24 | +6.06 | +8.46 | +4.45 | -0.58 | -0.73 |
| **sql** | +10.08 | +6.47 | +11.77 | +2.56 | -0.48 | -0.60 |
| **math_proof** | +0.75 | +0.73 | +1.32 | +5.27 | +0.06 | -0.71 |
| **history** | -1.61 | -1.33 | -1.57 | -1.75 | -0.09 | -1.10 |
| **cooking** | -1.41 | -1.20 | -1.56 | -1.72 | -0.46 | +1.50 |

### Double dissociation, now over every pair

`docs/ABLATION.md` found the crossover pattern on its single medicine/cooking pair. Across all 15 unordered pairs here it holds for **14/15**.

| pair a/b | a on a | a on b | b on b | b on a | crossover |
|---|---|---|---|---|---|
| python / rust | +1.8971 | +1.7762 | +1.8187 | +2.0437 | no |
| python / sql | +1.8971 | +1.6675 | +2.3340 | +2.0149 | YES |
| python / math_proof | +1.8971 | +0.8846 | +1.1987 | +0.4075 | YES |
| python / history | +1.8971 | +0.1499 | +0.2476 | +0.0001 | YES |
| python / cooking | +1.8971 | +0.1624 | +0.6758 | +0.0354 | YES |
| rust / sql | +1.8187 | +1.7645 | +2.3340 | +1.9217 | YES |
| rust / math_proof | +1.8187 | +1.0575 | +1.1987 | +0.4934 | YES |
| rust / history | +1.8187 | +0.1285 | +0.2476 | -0.0193 | YES |
| rust / cooking | +1.8187 | +0.1212 | +0.6758 | +0.0130 | YES |
| sql / math_proof | +2.3340 | +0.7284 | +1.1987 | +0.5343 | YES |
| sql / history | +2.3340 | +0.1544 | +0.2476 | +0.0373 | YES |
| sql / cooking | +2.3340 | +0.1525 | +0.6758 | +0.0389 | YES |
| math_proof / history | +1.1987 | +0.2846 | +0.2476 | -0.0229 | YES |
| math_proof / cooking | +1.1987 | +0.1254 | +0.6758 | -0.0164 | YES |
| history / cooking | +0.2476 | +0.0280 | +0.6758 | +0.1576 | YES |

## The random-expert null `docs/ABLATION.md` was missing

30 random size-100 expert sets per victim domain. Percentiles, not just the mean — `docs/TRANSFER.md` §11's standing rule.

| victim domain | null min | p05 | p25 | median | p75 | p95 | max | own-domain ablation | its percentile |
|---|---|---|---|---|---|---|---|---|---|
| python | +0.0034 | +0.0720 | +0.1618 | +0.2422 | +0.3205 | +0.6317 | +0.7058 | +1.8971 | 100th |
| rust | +0.0515 | +0.0558 | +0.1455 | +0.2054 | +0.3930 | +0.8046 | +0.8855 | +1.8187 | 100th |
| sql | +0.0409 | +0.1066 | +0.2020 | +0.2721 | +0.3780 | +0.6584 | +0.7390 | +2.3340 | 100th |
| math_proof | +0.0458 | +0.0706 | +0.1546 | +0.2682 | +0.3736 | +0.5973 | +0.7447 | +1.1987 | 100th |
| history | +0.0193 | +0.0334 | +0.1316 | +0.2051 | +0.3110 | +0.8501 | +0.9815 | +0.2476 | 67th |
| cooking | +0.0131 | +0.0736 | +0.1853 | +0.2333 | +0.2801 | +0.9112 | +1.0820 | +0.6758 | 90th |

Cross-domain cells against the same nulls:

| ablator -> victim | damage | null median | z | percentile | empirical p (upper) |
|---|---|---|---|---|---|
| python -> rust | +1.7762 | +0.2054 | +5.89 | 100th | 0.032 |
| python -> sql | +1.6675 | +0.2721 | +7.90 | 100th | 0.032 |
| python -> math_proof | +0.8846 | +0.2682 | +3.46 | 100th | 0.032 |
| python -> history | +0.1499 | +0.2051 | -0.49 | 30th | 0.710 |
| python -> cooking | +0.1624 | +0.2333 | -0.56 | 20th | 0.806 |
| rust -> python | +2.0437 | +0.2422 | +10.24 | 100th | 0.032 |
| rust -> sql | +1.7645 | +0.2721 | +8.46 | 100th | 0.032 |
| rust -> math_proof | +1.0575 | +0.2682 | +4.45 | 100th | 0.032 |
| rust -> history | +0.1285 | +0.2051 | -0.58 | 20th | 0.806 |
| rust -> cooking | +0.1212 | +0.2333 | -0.73 | 7th | 0.935 |
| sql -> python | +2.0149 | +0.2422 | +10.08 | 100th | 0.032 |
| sql -> rust | +1.9217 | +0.2054 | +6.47 | 100th | 0.032 |
| sql -> math_proof | +0.7284 | +0.2682 | +2.56 | 97th | 0.065 |
| sql -> history | +0.1544 | +0.2051 | -0.48 | 30th | 0.710 |
| sql -> cooking | +0.1525 | +0.2333 | -0.60 | 13th | 0.871 |
| math_proof -> python | +0.4075 | +0.2422 | +0.75 | 80th | 0.226 |
| math_proof -> rust | +0.4934 | +0.2054 | +0.73 | 80th | 0.226 |
| math_proof -> sql | +0.5343 | +0.2721 | +1.32 | 87th | 0.161 |
| math_proof -> history | +0.2846 | +0.2051 | +0.06 | 70th | 0.323 |
| math_proof -> cooking | +0.1254 | +0.2333 | -0.71 | 7th | 0.935 |
| history -> python | +0.0001 | +0.2422 | -1.61 | 0th | 1.000 |
| history -> rust | -0.0193 | +0.2054 | -1.33 | 0th | 1.000 |
| history -> sql | +0.0373 | +0.2721 | -1.57 | 0th | 1.000 |
| history -> math_proof | -0.0229 | +0.2682 | -1.75 | 0th | 1.000 |
| history -> cooking | +0.0280 | +0.2333 | -1.10 | 7th | 0.935 |
| cooking -> python | +0.0354 | +0.2422 | -1.41 | 3th | 0.968 |
| cooking -> rust | +0.0130 | +0.2054 | -1.20 | 0th | 1.000 |
| cooking -> sql | +0.0389 | +0.2721 | -1.56 | 0th | 1.000 |
| cooking -> math_proof | -0.0164 | +0.2682 | -1.72 | 0th | 1.000 |
| cooking -> history | +0.1576 | +0.2051 | -0.46 | 30th | 0.710 |

## Fits

Every predictor is computed from routing statistics only, before any ablation. Every response is measured. Mantel p permutes domain labels (respecting the matrix dependence), not pair values.

| response | predictor | n | Pearson r | Spearman rho | slope | Mantel p |
|---|---|---|---|---|---|---|
| raw_damage_nats | overlap_cos_lift_alldata | 30 | 0.833 | 0.800 | 1.3127 | 0.0042 |
| raw_damage_nats | overlap_cos_lift_splitA | 30 | 0.858 | 0.759 | 1.5385 | 0.0083 |
| raw_damage_nats | overlap_cos_raw_alldata | 30 | 0.887 | 0.777 | 6.1134 | 0.0014 |
| raw_damage_nats | overlap_cos_ratio_alldata | 30 | 0.913 | 0.784 | 1.3177 | 0.0042 |
| raw_damage_nats | mass_ratio_splitA | 30 | 0.872 | 0.832 | 1.2634 | 0.0125 |
| null_z_damage | overlap_cos_lift_alldata | 30 | 0.816 | 0.759 | 6.7793 | 0.0139 |
| null_z_damage | overlap_cos_lift_splitA | 30 | 0.841 | 0.728 | 7.9449 | 0.0153 |
| null_z_damage | overlap_cos_raw_alldata | 30 | 0.873 | 0.808 | 31.7489 | 0.0056 |
| null_z_damage | overlap_cos_ratio_alldata | 30 | 0.894 | 0.780 | 6.8072 | 0.0125 |
| null_z_damage | mass_ratio_splitA | 30 | 0.853 | 0.850 | 6.5199 | 0.0125 |
| damage_rel_own | overlap_cos_lift_alldata | 30 | 0.852 | 0.777 | 0.6689 | 0.0014 |
| damage_rel_own | overlap_cos_lift_splitA | 30 | 0.874 | 0.710 | 0.7802 | 0.0056 |
| damage_rel_own | overlap_cos_raw_alldata | 30 | 0.917 | 0.826 | 3.1506 | 0.0042 |
| damage_rel_own | overlap_cos_ratio_alldata | 30 | 0.937 | 0.837 | 0.6738 | 0.0028 |
| damage_rel_own | mass_ratio_splitA | 30 | 0.905 | 0.873 | 0.6534 | 0.0083 |

### Leave-one-domain-out on the primary fit

30 pairs, but only 6 independent units. If dropping one domain moves r a lot, the relationship is that domain, not a law.

| domain dropped | n pairs | Pearson r | Spearman rho | slope |
|---|---|---|---|---|
| python | 20 | 0.815 | 0.616 | 5.4682 |
| rust | 20 | 0.765 | 0.643 | 5.8920 |
| sql | 20 | 0.780 | 0.634 | 5.5241 |
| math_proof | 20 | 0.916 | 0.628 | 8.7033 |
| history | 20 | 0.827 | 0.903 | 9.6991 |
| cooking | 20 | 0.804 | 0.888 | 6.3806 |

Range of r across the six leave-one-out fits: [0.765, 0.916] (full-sample r = 0.816).

## Workstream 3's confound: is this just 'you deleted more of the model'?

`docs/UTILIZATION.md` found H1 specialists are disproportionately COLD (enrichment 0.624x into the hot set, permutation p < 0.0001), and that the hot experts are largely generalists. So a size-matched null controls set *size* but not set *load*. Load removed by each ablation set, in fair-share units (a random size-100 set has expectation 100):

| domain | load removed | vs random expectation |
|---|---|---|
| python | 171.54 | 1.72x |
| rust | 189.91 | 1.90x |
| sql | 192.39 | 1.92x |
| math_proof | 134.55 | 1.35x |
| history | 38.09 | 0.38x |
| cooking | 52.87 | 0.53x |

- Partial correlation of null-z damage with overlap, controlling for the ablator's load removed: **0.721** (raw r = 0.816).
- Multiple regression of null-z damage on [overlap, load removed]:

| term | beta | t | p (naive) |
|---|---|---|---|
| intercept | -1.8412 | -1.97 | 0.0597 |
| overlap_cos_lift | +5.7285 | +5.41 | 0.0000 |
| load_removed_by_ablator | +0.0133 | +1.77 | 0.0888 |

Model R^2 = 0.700. **The standard errors above assume 30 independent rows and there are 6 independent units — treat the t/p columns as descriptive only.** The load covariate varies only across the 6 ablators (it does not depend on the victim), so it can absorb ablator-level differences but says nothing about victim-level ones; that is a real limit of this control, not a clean adjustment.

The alternative hypothesis WS-3 names — that cross-domain damage tracks how much of the *shared generalist pathway* an ablation removes rather than a-b overlap — is not supported over the overlap term here, but 6 ablators cannot separate them properly.

## What this does NOT show

- **n = 6 domains.** The 30 regression points are one 6x6 matrix. The Mantel p floor is 0.0014. Any CI here is wide and the leave-one-domain-out table above is the honest read on stability.
- **The held-out prompts are 24 surface variants of a single content stem per domain** (`probe_set_v1.yaml` gives each (topic, lang, register, format) cell one split=A and one split=B prompt, and all 24 split=B prompts of a topic share `stem`). Raising n from 6 to 24 raised *surface* coverage, not content coverage. The effective content-level n per domain is 1. This is the single biggest weakness of the measurement and it is a property of the probe set, not fixable here.
- **Routing overlap only.** Two domains routing to the same experts could still compute orthogonally inside them (`docs/ORTHOGONALITY.md`'s own stated limit). A weak predictive result is consistent with the routing layer simply not being where the interference lives.
- **Zero-ablation is not forgetting.** It is an upper bound on how much of a domain's computation flows through an expert set. No gradient step is taken and no continual-learning dynamics are observed. Nothing here transfers directly to a forgetting curve.
- **One model, one seed, one probe set.** OLMoE-1B-7B-0924 only. PLAN.md's second-model generality check is still not done.
- **The non-English prompts have never been read by a human** (`translation_reviewed: false`). 18 of the 24 held-out prompts per domain are zh/de/ja. This is a standing project-wide gap and it is inside this measurement.
- **30 null draws** gives percentile resolution 0.032. Adequate for placing an observation in the bulk of the null; not adequate for tail claims.

## Reproduce

```bash
python scripts/run_interference.py precompute      # routing statistics only, no model
python scripts/run_ablation_multi.py --n-null 30   # the expensive part, resumable
python scripts/run_interference.py report          # fits + this document
```
