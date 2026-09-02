# Workstream 3 — expert utilization and the hot/specialist cross-reference

Run over the existing 480-prompt capture (662,048 rows, 5,296,384 expert
selections). No new capture. Reproduce with `python scripts/run_utilization.py`.

## The hypothesis this workstream was built to test

> *"If specialization concentrates load onto a small load-bearing subset, that's a
> direct, cheap explanation for why adding more data/domains increases collapse
> risk — more load keeps landing on the same specialized few."*

**That hypothesis is refuted on this substrate. The relationship runs the other way.**

## Utilization

`load_ratio = 1.0` means an expert carries exactly its fair share
(`1/(n_layers*n_experts)` of all selections; `top_k` cancels out of the baseline).

| statistic | value |
|---|---|
| load ratio min / median / max | 0.026 / 0.734 / 5.806 |
| skew (max/min) | **227.5x** |
| Gini | 0.4282 |
| coefficient of variation | 0.8975 |
| hot (>= 2x fair share) | 100 / 1024 |
| cold (<= 0.5x) | 309 / 1024 |
| dead (never fired) | **0** |

No dead experts and a smooth distribution — consistent with `docs/FINDINGS.md`'s
conclusion that the 227x skew is genuine inference-time behaviour on a narrow
evaluation set, not a counting bug. It remains far outside `coactivation.py`'s
2.0x PMI validity limit, so anything PMI-based on this run stays untrusted.

## Hot / specialist cross-reference

Specialists = the H1 set from `docs/FINDINGS.md`: >=1 domain that is BH-FDR
significant **and** clears `|lift| >= 1.0`. n = 557.

Tested against a permutation null (10,000 draws of 557 random experts) rather
than a raw overlap count, since with 54% of experts specialised a large raw
overlap is expected by chance.

| | value |
|---|---|
| observed overlap (hot AND specialist) | 34 |
| null | 54.5 +/- 4.7 |
| enrichment | **0.624x** |
| permutation p | < 0.0001 |

**Verdict: DEPLETED. Specialists are disproportionately *cold*, not hot.**

## Is this an artifact of base-rate correction?

It has to be asked: lift is `log2(P(e|d)/P(e))`, so usage appears in the
denominator. Two checks:

| test | result | reading |
|---|---|---|
| `spearman(load_ratio, max\|lift\|)` | **-0.546**, p=1.1e-80 | strong anti-correlation |
| `spearman(load_ratio, n_significant_domains)` | **-0.022**, p=0.48 | **no** relationship |

max\|lift\| by usage decile (1 = coldest):

| decile | load | max\|lift\| | sig domains | n specialists |
|---|---|---|---|---|
| 1 | 0.17x | 2.92 | 7.4 | 94 |
| 5 | 0.67x | 1.53 | 7.4 | 62 |
| 10 | 3.23x | 0.88 | 8.0 | 35 |

**Reading, stated carefully.** If this were a statistical-power artifact — rare
experts having noisier estimates — FDR significance would track usage. It does
not (r = -0.022, p = 0.48): experts pass FDR at the same rate at every usage
level, and the number of significant domains is flat (~6-8) across all deciles.
What changes monotonically is the *effect size*.

That is the substantive reading: **a high-load expert fires above its fair share
across many domains, which is what "generalist" means.** An expert cannot both
absorb 3x average load everywhere and show >=2x concentration in one domain.
So load and specialisation are close to definitionally opposed, and the measured
anti-correlation is partly structural rather than a surprising empirical fact.

**Honest limit:** `load_ratio` and `max|lift|` are derived from the same count
matrix and are not independent quantities. This result should be read as
"specialisation and load are opposed on this substrate", **not** as an
independent discovery that two unrelated variables happen to anti-correlate.
The load-bearing, non-circular part is the FDR-vs-usage null result: the
*detection* of specialisation is usage-independent even though its *magnitude*
is not.

## Consequence for Workstreams 1 and 2

The cheap mechanism this workstream was meant to supply — *specialisation piles
load onto a few experts, so more data means more collapse risk* — **is not
available.** The 100 hot experts are largely generalists (only 34 are
specialists, against 54.5 expected by chance).

If context rot or cross-domain interference has a routing-level mechanism here,
hot-expert concentration is **not** it, and WS1/WS2 should not assume it is.
A different candidate this does leave open, untested here: because the hot set
is generalist, load growth may degrade the *shared* pathway all domains rely on
rather than any domain's specialists — which would predict fairly uniform
degradation across domains rather than domain-specific damage. WS2's
multi-pair ablation is the natural test of that and it is not run here.

`data/utilization.json` carries the full per-expert vectors for both workstreams.
