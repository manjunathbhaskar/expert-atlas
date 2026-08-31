# Second-model generality check — Granite-3.0-3B-A800M

## Limitations first

- **One additional model, one architecture family.** This tests whether the
  OLMoE findings survive on a second fine-grained-expert MoE; it does not
  license claims about MoE models in general.
- **Not PLAN.md's named candidates.** Qwen1.5-MoE-A2.7B (~28.6 GB bf16) and
  DeepSeek-V2-Lite (~31 GB) exceed this machine's 31 GB RAM; Granite-3.0-3B-A800M
  (base, 3.3B params, 32 MoE layers x 40 experts, top-8)
  was the closest fitting substrate. Rerunning on a named candidate on a
  larger machine remains open.
- **Same probe set, different tokenizer.** probe_set_v1 was designed for
  OLMoE; token budgets per domain differ after Granite tokenization
  (equal-budget subsampling still applies, so base-rate correction holds).
- Router logits were captured via a forward hook on GraniteMoeTopKRouter
  (Granite ignores `output_router_logits`); the hook was sanity-checked to
  yield exactly one (n_tokens, 40) tensor per layer. Granite's
  topk-then-softmax gating selects identical expert ids to the
  softmax-all-then-topk convention used in this repo (softmax is monotone),
  and its gate weights equal `route_from_logits(norm_topk_prob=True)`.

## Result

Same pipeline as the OLMoE run: base-rate-corrected lift, chi-squared +
BH-FDR (q=0.05), practical-significance bar |lift| >= 1.0.

| metric | OLMoE-1B-7B-0924 | Granite-3.0-3B-A800M | verdict |
|---|---|---|---|
| H6 split-half pooled rho (gate, threshold 0.5) | 0.667 | 0.611 | PASS |
| H1 experts with >=1 meaningful topic affinity | 557/1024 (54.4%) | 490/1280 (38.3%) | PASS (falsified if <5%) |

Per-factor (FDR-significant cells vs. cells also clearing |lift| >= 1.0):

| factor | cells | FDR-sig | sig AND meaningful |
|---|---|---|---|
| topic | 12800 | 8932 (69.8%) | 1572 (12.3%) |
| lang | 5120 | 3357 (65.6%) | 100 (2.0%) |
| register | 2560 | 1178 (46.0%) | 4 (0.2%) |
| format | 3840 | 1497 (39.0%) | 1 (0.0%) |

Run: 480 prompts, 1229568 trace rows. Raw numbers in
`data/second_model_granite.json`; traces under `data/traces_granite/`.

## Interpretation

Both the replication gate and the per-expert affinity finding hold on a second, independently-trained, architecturally-distinct MoE. The atlas method and the specialization signal it measures are not OLMoE-specific artifacts -- within the limits above.
