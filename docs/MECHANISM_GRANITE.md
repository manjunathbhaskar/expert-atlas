# Does the context-rot mechanism replicate on a second model? (Granite-3.0-3B-A800M)

## Limits (read first)

1. **Different candidate pool.** 7/8 of the OLMoE forced-choice words are
   multi-token under the Granite tokenizer, so this run uses a pool verified
   single-token under BOTH tokenizers (see
   `probes/generate_context_probes_granite.py`). Same design, same seed
   structure, different surface content -- a conceptual replication, not a
   byte-identical one.
2. **No WS-3 utilization run exists for Granite**, so the hot/cold pathway
   partition from `docs/CONTEXT_PATHWAY.md` is not reproduced; only the
   needle-affine specialist set (the causally-relevant pathway on OLMoE) is.
3. One seed, one task design, CPU BF16 -- same scope limits as the OLMoE run.
4. Correlational throughout. No intervention was run on Granite.

## Q1 -- substrate: accuracy vs length (192 prompts)

| bucket | n | forced-choice acc | mean answer prob |
|---|---|---|---|
| 256 | 32 | 0.938 | 0.930 |
| 512 | 32 | 0.969 | 0.950 |
| 1024 | 32 | 1.000 | 0.995 |
| 2048 | 32 | 1.000 | 0.994 |
| 3072 | 32 | 1.000 | 0.999 |
| 3840 | 32 | 1.000 | 1.000 |

## Q2 -- length trends (needle-affine set: 79 of 1280 experts, defined at bucket 256 only)

| metric | short | long | delta | rho | perm p | FDR | d | verdict |
|---|---|---|---|---|---|---|---|---|
| affine_share_needle | 0.0303 | 0.0251 | -0.0053 | -0.255 | 0.0010 | yes | -0.73 | **SIGNIFICANT-BUT-TRIVIAL** |
| affine_share_q | 0.0043 | 0.0049 | +0.0005 | +0.308 | 0.0005 | yes | +0.54 | **SIGNIFICANT-BUT-TRIVIAL** |
| entropy_all | 4.8461 | 4.9184 | +0.0724 | +0.838 | 0.0005 | yes | +4.44 | **TREND** |

## Q3 -- does each metric predict answer probability, independent of length?

Partial Spearman controlling log2(length), permutation p (2000 shuffles, two-sided), practical floor |partial rho| >= 0.3.

| metric | raw rho | partial rho | perm p |
|---|---|---|---|
| affine_share_needle | +0.291 | +0.465 | 0.0005 |
| affine_share_q | +0.320 | +0.216 | 0.0030 |
| entropy_all | +0.315 | -0.108 | 0.1474 |

## Verdict: **MIXED — the substrate is absent, the pathway-correctness link replicates**

- Q1: **Granite does not context-rot at these lengths.** Accuracy is 1.000 at
  every bucket >= 1024 (vs OLMoE's 0.938 -> 0.688 decline). The task tops out
  before 4096 tokens on this model, so there is almost no failure variance for
  the mechanism to explain. The registered Q2/Q3 criteria were still scored:
- Q2 (specialist share declines with length, FDR + |d|>=0.8): **FAIL** — the
  decline exists and is FDR-significant (rho=-0.255, p=0.001) but small
  (d=-0.73 vs the 0.8 floor), consistent with a model that is not failing.
- Q3 (specialist share predicts answer probability independent of length,
  perm p<0.05 + partial rho>=+0.3): **PASS** (partial rho=+0.465, p=0.0005).
  Entropy again does not (partial -0.108, ns) — same dissociation as OLMoE.

Honest reading: this run cannot confirm the full mechanism (no rot means no
mechanism-of-rot to replicate at n<=3840), but the specific correlational
signature — needle-affine specialist routing tracks correctness while global
entropy does not — appears on both models. Testing the decline itself would need a substrate
this model actually fails on (harder distractors, or a longer-context
Granite variant).

OLMoE reference values (docs/CONTEXT_PATHWAY.md): specialist share 0.0417 -> 0.0335 (d=-1.39, FDR yes); partial rho +0.651, p=0.0005.

