# Interface change requests

Per PLAN.md §4: file ownership is exclusive. If a workstream needs a change
to a contract or file it doesn't own, it requests it here instead of editing
directly, and the owner makes the change.

Format:

```
## <date> — requested by <workstream> — target: <owning workstream>
**File/contract:** ...
**Requested change:** ...
**Why:** ...
**Status:** open | accepted | rejected (with reason)
```

---

(none yet)

## 2026-08-11 — requested by WS-B — target: WS-C
**File/contract:** `expertatlas/stats.py` — count aggregation, and `atlas.json` `stats` block

**Requested change:** Before accumulating expert×domain counts, subsample every
(topic, lang, register, format) cell to the **minimum token count across all cells**,
using a fixed seed. Record the value as `stats.tokens_per_cell` in `atlas.json`.

**Why:** Measured in `probes/validate.py` — prompt length varies **1.73×** across topics
(`math_proof` 126 tokens vs `history` 73) and **1.53×** across languages (OLMoE's BPE is
English-centric). Each token is an independent routing observation, so pooling raw counts
weights long-prompt cells more heavily. Any expert responding to *sequence position* or
*long context* would then appear as a topic specialist — a confident, wrong atlas.

Not fixable in WS-B: padding or truncating prompts would break the payload-invariance
control (identical code bytes across language cells), which is the more important property.

**Status:** accepted — implemented in `stats.py::subsample_cells` + `aggregate_counts`

## 2026-08-11 — requested by WS-D — target: WS-A / WS-C
**File/contract:** `atlas.json` — a new optional `replay` block, or a trace sidecar

**Requested change:** expose per-token expert firing for a small set of example
prompts, e.g. `replay: [{prompt_id, text, steps: [[layer, [expert_uid, ...]], ...]}]`,
capped at ~20 prompts to stay inside the 5 MB budget.

**Why:** PLAN.md §8 specifies a replay mode (step a prompt, flash experts as they
fire). It is the most legible feature for a non-specialist audience and the closest
analogue to Colibri's Brain page. `atlas.json` currently carries only aggregate lift,
so replay cannot be built from it. Deferred rather than faked.

**Status:** open

## 2026-08-29 — requested by second-model check — target: WS-A
**File/contract:** `expertatlas/capture.py` — `load_model()` + `_GateHookCapture`

**Requested change:** (1) `load_model()` should fall back to `config.num_local_experts`
when `config.num_experts` is absent (GraniteMoe, Mixtral-family configs use the former).
(2) The forward-hook fallback should accept a configurable gate-module suffix and a
configurable tuple index for the logits — GraniteMoeTopKRouter modules end in
`.block_sparse_moe.router` (not `.gate`) and return `(top_k_index, top_k_weights,
router_logits)`, i.e. the raw logits are element 2, not element 0.

**Why:** `scripts/run_second_model_check.py` needed a private loader + hook adapter to
capture Granite-3.0-3B-A800M. Generalizing capture.py would let future model checks
reuse the tested WS-A path directly instead of carrying script-local copies.

**Status:** open
