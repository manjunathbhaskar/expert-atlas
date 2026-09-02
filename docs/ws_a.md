# WS-A — capture engine: status

Per PLAN.md §5 Phase 1 WS-A bullets.

**Note on authorship:** this was built directly (not via the Nemotron/OpenCode
dispatch attempted first) — NVIDIA's free `nemotron-3-ultra` endpoint was
returning transient `404 Not Found` on a large fraction of calls at the time
(observed ~2/5 to ~3/6 failure rate across direct tests, including mid-task,
not just at startup), which reliably killed multi-step agentic runs before
they could write anything (confirmed via clean `git status` after each failed
attempt — zero partial/corrupt output). A retry wrapper was added at
`notes/run_nemotron_retrying.sh` for future use, but whole-run
retries can't realistically get a long task through a ~35%-per-call failure
rate (probability of zero failures across dozens of LLM calls is negligible),
so this workstream was done directly instead of keeping the endpoint blocked.

## Done

- [x] `expertatlas/capture.py::get_router_logits_for_prompt` — unifies the
      output_router_logits and forward_hook paths behind one call, used by
      both the CLI dry-run and the real batched writer so they can't diverge.
- [x] `_GateHookCapture` — implements the forward-hook fallback for real
      (previously only *detected* whether a model needed it, in `load_model`,
      without an actual capture implementation). **Caveat, stated honestly**:
      not exercised against a real forward-hook-only model — none is
      downloaded in this repo, since OLMoE itself uses
      `output_router_logits` natively. Covered by a unit test that a broken/
      mismatched-layer-count model raises loudly rather than silently
      truncating (`test_get_router_logits_raises_on_layer_count_mismatch`),
      which is the property that actually matters for trace integrity.
- [x] `prompt_rows_to_table` — builds one prompt's full (all layers, all
      tokens) RoutingTrace as a pyarrow Table against the frozen
      `ROUTING_TRACE_SCHEMA`, independently testable with no model/disk I/O.
- [x] `capture_to_dir` — batched, streaming, resumable capture:
      - **Streaming**: one prompt's table is built → written → dropped
        before the next prompt starts. Nothing accumulates across prompts.
      - **Resumable**: `manifest.json` (write-then-rename, atomic) tracks
        completed prompt_ids; a killed-and-restarted run skips everything
        already done and produces byte-identical shards for those prompts.
      - **Deterministic ordering**: prompts are sorted by id before capture,
        regardless of input list order.
      - **Greedy-only by construction**: each prompt gets exactly one
        forward pass over its own tokens — there is no autoregressive
        sampling loop to have gotten wrong.
- [x] CLI (`atlas capture`): added `--prompts-file`, `--out`, `--limit`,
      `--layers` (dry-run summary filter only — the real per-prompt trace
      always writes all layers per the frozen §3 schema, which has no
      per-layer filter field), `--no-resume`. The old `--prompt --dry-run`
      single-prompt debug path is preserved unchanged.
- [x] `tests/ws_a/test_capture_engine.py` — 10 tests, all passing, using a
      small fake model/tokenizer (no OLMoE download needed): schema
      conformance, layer-mismatch detection, streaming shard-per-prompt,
      resume-after-interruption (byte-identical untouched shard, correct
      skip/write counts), full uninterrupted-vs-resumed equivalence,
      `--limit`, `--no-resume` semantics, deterministic ordering
      independent of input order, and manifest atomicity.
- [x] `pytest tests/ -v --ignore=tests/sanity/test_model_loading.py` —
      **83/83 passing** (73 pre-existing from WS-B/C/D + 10 new). Confirmed
      directly, nothing from WS-B/C/D touched or broken.

## Honestly not verified (stated per PLAN.md's own honesty requirement)

- **PLAN §6.4 `test_capture_memory_bounded` (<6GB RSS for 100k tokens)** —
  not measured against real memory usage. The streaming design satisfies
  this *by construction* (no cross-prompt accumulation, one table in memory
  at a time), but that's an architectural argument, not a measurement. A
  real RSS-bounded test needs either the actual OLMoE model captured at
  scale or a memory-profiling harness around the fake model with enough
  synthetic tokens to be meaningful — not done here for time.
- **Forward-hook fallback against a real gate-hook model** — see caveat
  above. Implemented for real, unit-tested for the failure mode that
  matters (silent layer loss), but never run against actual hook-only
  model weights.

## Interface request response

`docs/interface-requests.md` has an open request from WS-D for a `replay`
block in `atlas.json` (per-token expert firing for ~20 example prompts).
**Left open** — building it would mean either (a) re-running capture for a
curated prompt subset and threading that into the WS-C aggregation/export
step, which is WS-C's file (`aggregate.py`/export logic), or (b) WS-A adding
an export path of its own that WS-C would then need to consume, and I don't
want to guess which shape WS-C's `atlas.json` writer expects without
coordinating. Flagging in `docs/interface-requests.md` rather than picking
a shape unilaterally and risking a contract WS-C then has to unwind.

## Not done (still open, per PLAN.md)

- Phase 3 real capture run (≥250k tokens across the full probe set) — this
  workstream provides the mechanism; running it for real against OLMoE is
  the Phase 3 integration step, not WS-A itself.
- WS-E packaging/notebook.
