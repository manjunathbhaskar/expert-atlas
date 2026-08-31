# Deconfounded residual-stream probe: the needle survives; the readout fails

## Limitations (read first)

- One model (OLMoE-1B-7B-0924), one seed, CPU/BF16, teacher-forced,
  single-needle forced-choice substrate. No claim beyond it.
- Probe recoverability is a lower bound on "information present"; a probe
  failure does not prove absence, and probe success does not prove the model
  USES the information — that distinction is the whole point of this doc.
- 192 prompts, 64 per length bucket {256, 1024, 3840}; the model-wrong cell
  at long context has n=14.

## Why the first probe run was thrown away

The original hard set maps entity i deterministically to answer word i, and
the question names the entity. Any representation encoding *entity identity*
therefore decodes the "answer" without reading the needle. Proof: a probe on
the question window at LAYER 0 — before any attention has run — scored 100%.
That run is reported nowhere as evidence of needle retention.

## The repaired design

`probes/probe_set_context_repaired.yaml`: two pairing systems. Group A maps
entity i -> word i; group B maps entity i -> word (i+3) mod 8. Probes
(multinomial logistic regression, standardized features, C=1.0) are trained
on one group and evaluated on the other, both directions averaged. Under
cross-pair evaluation an entity shortcut predicts a specific WRONG answer, so
it shows up as a measured `shortcut rate`, not as fake accuracy. 200
label-shuffle permutations give the null (mean 0.125, max ~0.23).

Manipulation check: the question window (`q_mean`) now scores 0.000 at every
layer with shortcut rate 1.000 — the design catches the confound exactly as
intended.

## Results

Model forced-choice accuracy by bucket: 0.953 (256) -> 0.797 (1024) ->
0.781 (3840). The probe grid (cross-pair accuracy at 3840 tokens):

| position | layers 1–9 | layers 10–12 | layers 13–16 |
|---|---|---|---|
| needle last token | **0.98–1.00** | 0.64–0.81 | 0.22–0.45 |
| final (answer) position | 0.00–0.03 | 0.64–0.75 | **0.91–0.94** |

Registered primary cells:

- `needle_last` L8: **acc = 0.995**, perm p = 0.0050, null mean 0.126.
- `final` L16: acc = 0.922, perm p = 0.0050, null mean 0.124.

The crux — prompts where the model answers WRONG at long context:

| cell | model wrong (n=14) | model right (n=50) |
|---|---|---|
| needle_last L8 | **1.000** | 1.000 |
| final L16 | **0.714** | 0.960 |

## Reading this

1. **The needle's content is fully intact at its source position at every
   tested length, including on every prompt the model gets wrong.** Context
   rot on this substrate is not upstream representational decay at the needle.
2. **The failure is transport/readout.** Needle information appears at the
   answer position only in layers ~10–16 (attention has to carry it there),
   and precisely on model-wrong prompts that arrival is degraded (0.71 vs
   0.96). The router-level specialist starvation documented in
   `docs/CONTEXT_PATHWAY.md` is downstream of — or concurrent with — this
   transport failure, which is consistent with both router boosts failing:
   forcing the router cannot help if the needle content never arrives at the
   position being routed.
3. The `needle_last` decodability FALLS across layers 10–16 while `final`
   decodability RISES over the same layers — consistent with mid-stack
   attention moving the content from source to destination.

## What this licenses next

The information is present and localized; the deficient link is the
mid-stack transfer into the final position. That makes a representation-level
intervention well-posed: inject the needle's content representation at the
final position in the mid stack on long prompts, against random-direction and
wrong-content controls at matched norm (`expertatlas/anchoring.py`), and test
whether forced-choice accuracy recovers. A positive result must beat both
controls under a paired permutation test with the project's effect-size floor.

Artifacts: `data/context_probe_repaired/{records.jsonl,analysis.json}`,
scripts `run_context_probe_capture.py`, `run_context_probe_repaired_analyze.py`,
generator `probes/generate_context_probes_repaired.py`.
