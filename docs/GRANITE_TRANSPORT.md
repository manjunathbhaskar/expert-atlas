# Granite head-level replication of the transport mechanism

**Verdict: the full causal chain replicates on a second model.** On
`ibm-granite/granite-3.0-3b-a800m-base` (32 layers x 24 heads, GQA, 40
experts/top-8), retrieval heads identified purely from short correct prompts
collapse on failing long prompts (localized, not diffuse), and boosting them
onto the needle span repairs **5/5** failing prompts. The label-free lexical
detector drives the same repair at **99.1% of the oracle effect**.

Everything below was preregistered in `scripts/run_granite_transport.py`
(module docstring) before any capture: buckets, K=16, betas, widths, the
>=4-failing-prompt gate, 2000 permutations, seed 0, dev-arm-only calibration,
and the statistical bar (paired sign-flip p<0.05 AND |dz|>=0.8).

## Getting Granite to fail at all (two substrates, one negative first)

Granite previously did not context-rot at 50% needle depth at any length up
to 3840 (docs/MECHANISM_GRANITE.md). Two registered escalations:

1. **Depth-factor set** (`probes/probe_set_context_granite_depth.yaml`,
   depths 0.15/0.50/0.85 x 256/3840, 8 distractors): **still no rot** —
   1.000 accuracy in every 3840 bucket, at every depth (misses only in the
   256 bucket). The depth factor that breaks OLMoE (0.375 at depth 0.15)
   does not break Granite. This negative is a finding: Granite is
   substantially more robust to needle depth at these lengths.
2. **Distractor-load escalation** (`probes/probe_set_context_granite_hard.yaml`,
   24 distinct same-template distractors, all wrong entities AND wrong
   answer words — unambiguous by construction; depth 0.15, 3840 tokens,
   n=16): accuracy **0.688**, 5 model-wrong prompts. Gate (>=4) passed.

So Granite's rot trigger is competitive interference under load, not
distance alone — a substrate difference vs OLMoE, reported as measured.

## Stage 1 — identification (short correct prompts, all depths pooled)

46 model-correct 256-token prompts from the depth set. Top-16 of 768
(layer, head) cells by mean last-row attention mass on the needle:

- Top cell **L22 H8: 0.901 mass (17.9x chance)**; then L22 H4 (0.779),
  L22 H3, L22 H16, L22 H1, L25 H13, ...
- 7/16 cells in layer 22; 16/16 in layers 19–28. Same relative-depth band
  as OLMoE (L22/32 = 0.69 vs OLMoE L12/16 = 0.75): retrieval heads sit at
  ~70% of the stack in both models.

## Stage 2 — collapse test (3840 bucket: 11 right vs 5 wrong)

Nulls stated first: (1) identified-head needle mass does not differ between
right and wrong prompts; (2) any drop is not specific to the identified set.

- Identified heads: right 0.225 vs wrong 0.101 — **d=2.17, perm p=0.0005**.
- Remaining 752 cells: 0.0112 vs 0.0068 (d=1.63, p=0.015) — a small ambient
  drop exists, but the identified-set excess drop (0.120) is far outside the
  random-16-cell null (null p95 = 0.021, **specificity p<0.0005**).
- **Collapse is localized**, replicating OLMoE.

## Stage 3 — boost repair (frozen calibration, all controls)

beta*=8.0 and detector width*=8 chosen on the 16-prompt dev arm only
(depth-set 1024 bucket; dev was at ceiling so beta resolved by mean prob).
Evaluation on the 16 hard-set prompts:

| condition | acc | mean prob | failing subset (n=5) |
|---|---|---|---|
| baseline | 0.688 | 0.644 | 0/5, prob 0.178 |
| identified + oracle span | **1.000** | 0.988 | **5/5**, prob 0.975 |
| random 16 cells + oracle span | 0.625 | 0.561 | 0/5, prob 0.136 |
| identified + wrong span | 0.688 | 0.631 | 0/5, prob 0.137 |
| identified + lexical-detected span | **1.000** | 0.984 | **5/5**, prob 0.968 |

- Oracle vs baseline: dz=0.99, p<0.0005; vs random heads: dz=1.10,
  p<0.0005; vs wrong span: dz=0.99, p<0.0005 (full set, n=16).
- Lexical detector: 16/16 span hits; **99.1% of the oracle effect**;
  vs wrong span dz=0.99, p<0.0005.
- Failing subset (n=5): dz ~ 8.2–8.8 for all key contrasts; the sign-flip
  permutation floor at n=5 is p=1/16=0.0625, so subset p-values (0.063–
  0.066) are at their theoretical minimum — the full-set tests carry the
  significance, and they pass the registered bar (|dz|>=0.8, p<0.05).
- Random heads at beta=8 are net harmful (0.688 -> 0.625 acc), as on OLMoE.

## What this does and does not show

Shown: the mechanism (specific retrieval heads, localized collapse,
span-targeted attention repair, lexical span discovery) is not an OLMoE
quirk — it replicates across architecture (GQA vs MHA attention, 32 vs 16
layers, different MoE router) on a substrate that fails for a different
reason (distractor load vs depth).

Not shown: generality beyond the lexical needle task (see
docs/CONTEXT_ROT_STORY.md and the paraphrase/multi-hop variants run), or
beyond 4096-token models, or on naturally occurring documents.

## Reproduction

```bash
HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 .venv/bin/python \
  probes/generate_context_probes_granite_depth.py           # seed 6
HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 .venv/bin/python \
  probes/generate_context_probes_granite_hard.py            # seed 8
HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 .venv/bin/python \
  scripts/run_granite_transport.py --stage 0                # depth set
HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 .venv/bin/python \
  scripts/run_granite_transport.py --stage 0 \
  --probe-set probes/probe_set_context_granite_hard.yaml --out-dir data/granite_hard
HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 .venv/bin/python \
  scripts/run_granite_transport.py --stage 12 --hard
HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 .venv/bin/python \
  scripts/run_granite_transport.py --stage 3 --hard
```

Outputs: `data/granite_transport.json`, `data/granite_boost.json`.
