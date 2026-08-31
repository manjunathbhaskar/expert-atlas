# Expert Atlas — Method

This document describes the complete experimental pipeline. **Part I** (§1–§14) is the specialization-atlas statistical method behind `docs/FINDINGS.md` and `docs/ORTHOGONALITY.md`. **Part II** (§15–§21) is the context-degradation diagnosis-and-repair pipeline behind `docs/CONTEXT_ROT_STORY.md` and the technical report.

---

# Part I — Specialization atlas

---

## 1. Data: what is measured and how

**Model:** OLMoE-1B-7B-0924 (16 layers × 64 experts, top-8 routing, `norm_topk_prob=False`).

**Probe set:** 480 prompts forming a 10 (topic) × 4 (language) × 2 (register) × 3 (format) factorial design (240 cells, 2 prompts per cell). Each cell has a held-out split (A/B, 50/50) for H6 replication. Payloads (code/notation) are byte-identical across language cells — this is the within-subjects control separating syntax from language affinity.

**Capture:** Teacher-forced single forward pass per prompt (greedy, deterministic). Router logits extracted via `output_router_logits=True` (native HF path). For each token and layer: full softmax over 64 experts, then top-8 selection. The top-8 weights **do not sum to 1** — they sum to the top-k probability mass (`topk_mass`), stored per token because low mass = router uncertainty. This matches `OlmoeSparseMoeBlock.forward` exactly.

**Output:** Parquet traces (`RoutingTrace` schema: `prompt_id, token_pos, token_id, layer, expert_ids[8], gate_weights[8], topk_mass`). One shard per prompt, streaming write, resumable via manifest.

---

## 2. From traces to count matrices — equal token budget

**The confound:** Prompt length varies 1.73× across topics (`math_proof` 126 tokens vs `history` 73) and 1.53× across languages (English-centric BPE). Each token is an independent routing observation, so raw pooling weights long-prompt cells more heavily. An expert responding to *sequence position* or *long context* would masquerade as a topic specialist.

**The fix (`expertatlas/aggregate.py::subsample_cells`):** Before any counting, subsample every factorial cell to the **minimum token count across all cells**, using a fixed seed. Subsampling is over *distinct tokens* (prompt_id, token_pos) — every layer of a kept token is kept, so per-token routing patterns are never partially observed. The budget (`tokens_per_cell`) is recorded in `atlas.json` `stats` block.

**Counting (`aggregate_counts`):** For each kept token, increment the (expert, domain) cell by 1 per selected expert (default; `weight_by_gate=False` — selection is the routing decision, weight is confidence, mixing them makes lift harder to interpret). Domains are defined by one factor at a time (`topic`, `lang`, `register`, `format`) — the factorial design's payoff is computing *marginal* lift per factor.

**Result:** Four count matrices (one per factor), each `(n_experts_total=1024, n_domains)` with equal token budget per column.

---

## 3. Primary metric: lift (base-rate-corrected affinity)

**Why not raw counts ("heat"):** MoE training uses a load-balancing objective whose explicit purpose is to make marginal expert usage `P(expert)` uniform. The model was optimised to defeat naive specialisation metrics. Raw usage carries almost no information by design.

**Formula (`expertatlas/stats.py::compute_lift`):**

```
lift(e, d) = log2( P(e | d) / P(e) )
```

With Laplace smoothing (`laplace=1.0`):

```
P(e | d) = (count(e, d) + laplace) / (total(d) + laplace * n_experts)
P(e)     = (total(e) + laplace) / (grand_total + laplace * n_experts)
```

- Positive lift = expert fires more on this domain than its base rate
- 0 = exactly base rate
- Negative = fires less
- `lift = 1.0` means 2× fold change; `lift = -1.0` means 0.5×

This is the **only** primary metric. Every headline number in `FINDINGS.md` is lift.

---

## 4. Significance testing and multiple-comparison correction

### 4.1 Per-cell test: chi-squared independence

For each (expert, domain) pair, a 2×2 contingency table:

| | this domain | other domains |
|---|---|---|
| this expert | a | b |
| other experts | c | d |

where `a = count(e,d)`, `b = total(e) - a`, `c = total(d) - a`, `d = grand_total - a - b - c`.

Yates-corrected chi-squared statistic (vectorised in `chi2_pvalues_fast`):

```
stat = n * (|a*d - b*c| - n/2)^2 / (row * col * (n-row) * (n-col))
p = chi2.sf(stat, df=1)
```

Degenerate tables (empty row/col) → p = 1.0.

### 4.2 Benjamini–Hochberg FDR across the whole matrix

1,024 experts × D domains = tens of thousands of simultaneous tests. Without correction, ~5% false specialists are manufactured.

**Procedure (`bh_fdr`):** Flatten the entire (experts × domains) p-value matrix, apply BH-FDR at `q=0.05` (`statsmodels.stats.multitest.multipletests(method="fdr_bh")`), reshape back. This is the correct unit of multiple comparison — not per-expert, not per-domain, but the whole atlas at once.

**Output:** Boolean significance mask, same shape as lift matrix.

---

## 5. Effect-size requirement: why significance alone is not enough

**The trap (caught in TRANSFER.md §11):** The first analysis pass reported "70.7% of topic cells significant" — looked like a strong finding. Checking the *distribution* of lift among those "significant" cells (not just the mean, which was pulled up by a right tail) showed the **median was 0.79×** — i.e., no real effect, just statistically distinguishable from base rate because the equal-token-budget subsampling left ~19,440 tokens per domain, which is enough for chi-squared to flag trivial deviations.

**The fix (non-negotiable, enforced in `run_phase3_analyze.py`):** Every headline number requires **BOTH**:

1. BH-FDR significance (q=0.05)
2. `|lift| >= 1.0` (≥2× fold change)

Raw significance-cell counts are reported in `FINDINGS.md` but explicitly flagged as inflated by sample size, not treated as the finding. Always check percentiles, never just the mean, when n is large.

---

## 6. H6 — Split-half replication (the project gate)

**Why first:** If affinity does not replicate across held-out prompts, everything downstream is noise.

**Method:** The probe set declares a 50/50 split (A/B) per cell. Compute lift matrices `lift_A` and `lift_B` independently on each half (each gets its own equal-token-budget subsampling). Spearman correlation between the two flattened lift vectors.

**Implementation (`split_half_replication`):** Returns pooled rho (primary) and per-expert rho.

**Threshold (pre-registered in PLAN.md):** ρ ≥ 0.5 → PASS. Real result: **ρ = 0.667** → PASS.

---

## 7. H1 — Domain affinity beyond chance (per-expert definition)

**PLAN.md's actual H1 definition:** "max lift per EXPERT vs. shuffled-label null" — a single test per expert, not per cell.

**Procedure:** For each expert, take its maximum lift across all domains of a factor (e.g., 10 topics). Check whether that max-lift domain is **both** FDR-significant **and** clears `|lift| >= 1.0`. Count experts passing this dual bar.

**Falsification bar (pre-registered):** < 5% of experts → H1 fails.

**Real result:** 557/1024 experts (54.4%) pass for `topic` → PASS.

---

## 8. H3 — Factor separability (factorial design payoff)

Compute the dual-bar (FDR + effect size) pass rate **per factor**:

| Factor | Domains | FDR-significant | + |lift|≥1.0 |
|--------|---------|-----------------|---|----------|
| topic  | 10      | 70.73%          | 22.17% |
| lang   | 4       | 73.97%          | 5.25%  |
| register | 2     | 43.36%          | 0.59%  |
| format | 3       | 38.96%          | **0.00%** |

The gap between raw FDR rate and effect-size-filtered rate *is* the finding — large-sample statistical significance without practical effect size. Format showing ~0% meaningful lift while topic shows 22% directly addresses the Mixtral syntax-vs-topic question.

---

## 9. Null model: label-shuffle permutation (used everywhere)

**Function (`shuffle_labels`):** Redistribute each domain's total count across experts according to the **global expert marginal** `P(expert)` — what you'd see if domain labels carried no information about routing. Preserves the shape of expert usage randomness while destroying any real expert↔domain association.

**Used for:**
- H6: not directly (split-half is the replication gate)
- Empirical lift null distribution (`null_lift_distribution`): 1000 permutations, returns per-cell null mean/std and two-sided empirical p-value (does not assume chi-squared approximation holds in sparse cells)
- Orthogonality analysis: 200 permutations, pairwise cosine similarity of lift vectors recomputed each time
- Co-activation: degree-preserving rewired null (different, see below)

---

## 10. Orthogonality analysis (extension, `docs/ORTHOGONALITY.md`)

**Question:** Are different topics' routing signatures near-orthogonal? (Mechanism from continual-learning literature: near-orthogonal task subspaces prevent interference.)

**Routing signature:** Base-rate-corrected **lift vector per domain** (columns of the lift matrix). This is immune to the 227× usage-skew problem that made H4 unreliable — a generically popular expert contributes ~0 to every domain's lift vector.

**Metric:** Pairwise cosine similarity between domain lift vectors.

**Null:** 200 label-shuffle permutations (same `shuffle_labels`). Observed pairwise similarities compared against this empirical null, not against 0 (cosine between random sparse-ish vectors is not 0 in general).

**Reported:** Mean |cosine|, max |cosine|, z-score vs null, domain-pair deltas (observed - null).

**Honest limit (must ship with any result):** This measures **ROUTING orthogonality only**. Two domains could route to identical experts yet produce orthogonal activations *inside* them. This is evidence about the routing layer specifically, not the full computation.

---

## 11. H4 — Co-activation communities (and why it's UNRELIABLE)

**Method (`expertatlas/coactivation.py`):** Within-layer and adjacent-layer co-firing counts → PMI → Louvain communities. Null: degree-preserving rewired graph (not Erdős–Rényi — naive null would inflate modularity).

**Validity gate (documented in `coactivation.py`):** PMI-based community detection requires usage skew ≤ 2.0×. The tool itself warns above this.

**This run:** `usage_skew = 227.5×` (max usage 30,029 vs min 132). Verified against raw trace counts: all 1,024 experts fired at least once, distribution is smooth (no dead experts, no single outlier) — this is real inference-time skew, not a counting bug. Plausible explanation: load balancing is a *training*-time objective over the full training distribution; nothing enforces balance on a narrow 480-prompt evaluation sample.

**Verdict:** **UNRELIABLE** — reported for completeness, not as evidence either way. The tool's own validity check correctly refused to trust it.

---

## 12. Ablation harness — causal test (Tier 3, `docs/ABLATION.md`)

**Goal:** Does removing an expert's contribution selectively hurt its domain?

**Method:** Forward hooks on `OlmoeTopKRouter` (layer gate modules). **Two critical implementation gotchas, verified against this model/transformers version:**

1. **OLMoE has no per-expert FFN submodule** — `.mlp.experts` is one batched module, not a `ModuleList`. You cannot hook individual expert FFNs.
2. `OlmoeTopKRouter.forward()` does **softmax + top-k INTERNALLY** and returns `(router_logits, router_scores, router_indices)`. The selection has already happened by the time a forward hook fires. Masking `router_logits` in a hook would be a **silent no-op** — nothing downstream re-reads it. `OlmoeSparseMoeBlock.forward` unpacks `_, top_k_weights, top_k_index = self.gate(...)` and uses `router_scores`/`router_indices`.

**Correct hook:** Zero `router_scores` at positions where `router_indices` matches an ablated expert id. Each expert's FFN output is weighted by its score before summing, so a zeroed score makes that expert's contribution exactly zero — true ablation, not re-routing.

**Conditions (evaluated on BOTH target and control held-out text):**
- `baseline`: no ablation
- `ablate_target`: experts significant+meaningful (FDR + |lift|≥1.0) for target domain
- `ablate_random`: same count, uniform random (fixed seed)
- `ablate_other_domain`: experts significant+meaningful for a different domain, same-ish count

**Causal claim requires ALL:** `ablate_target` hurts target text more than it hurts control text, AND more than `ablate_random` hurts target text, AND more than `ablate_other_domain` hurts target text. All four deltas are load-bearing — reporting one flattering number alone does not support causation.

**Metric:** Mean per-token teacher-forced cross-entropy (nats) on held-out (split=B) prompts. Forward passes only.

**Real result (medicine vs cooking, n=6 prompts each):** Double dissociation — directional evidence, not a finished statistical test (no null over repeated random draws, one domain pair, one seed).

---

## 13. Layout for visualisation (presentational only)

**Input:** Lift matrix (never raw usage).

**Method:** UMAP (cosine metric, `n_neighbors=min(15, n_experts-1)`, fixed seed) with PCA fallback. Deterministic jitter (1e-3) so co-located points remain pickable. Scaled to ~[-10, 10].

**Colour:** Co-activation community (which has a degree-preserving null behind it). UMAP blobs are *not* evidence on their own — PLAN.md §9b requires saying so wherever the layout is displayed.

---

## 14. Reproducibility checklist

| Item | Value / Location |
|------|------------------|
| Model | `allenai/OLMoE-1B-7B-0924`, revision pinned in `meta.json` |
| Probe set | `probes/probe_set_v1.yaml` (480 prompts, 240 cells, A/B split) |
| Capture seed | 0 (default) |
| Subsample seed | 0 (default, `aggregate_counts`) |
| Permutation seeds | 0 (stats), 0 (orthogonality), 0 (ablation random draw) |
| FDR method | Benjamini–Hochberg, q=0.05 |
| Effect size bar | `|lift| >= 1.0` (≥2× fold) |
| H6 threshold | ρ ≥ 0.5 |
| H1 falsification | < 5% experts |
| Laplace smoothing | 1.0 |
| `norm_topk_prob` | False (OLMoE config) — top-k weights do NOT sum to 1 |

All randomness is seeded. Two identical runs produce byte-identical traces and identical `atlas.json`.

---

# Part II — Context-degradation diagnosis and repair

Stages, in order. Every positive claim has a matched control and a bar (paired sign-flip permutation p < 0.05 AND |dz| ≥ 0.8 vs the matched control) registered before the run; failed directions are preserved in the docs they belong to.

## 15. Probe substrate and degradation measurement

Forced-choice needle retrieval: a needle fact placed at depth δ inside filler plus `n_d` same-template distractors, context lengths L ∈ {256, …, 3840} tokens, question at the end, answer scored as argmax over 8 candidate completions (chance 0.125), teacher-forced single forward pass (`scripts/run_context_probe_capture.py`). OLMoE hard variant: accuracy 0.938 (256) → 0.688 (3840); significant, but Cohen's d = −0.67 misses the preregistered |d| ≥ 0.8 practical floor — reported as such.

## 16. Routing correlate (and why it is a symptom, not the cause)

Per-prompt `needle_affinity_rate` (fraction of needle-token routing that goes to the needle's lift-affine experts) predicts accuracy controlling for length (partial ρ = 0.64, p < 0.0001) while overall router entropy does not (partial ρ = 0.07, ns) — `scripts/run_context_pathway.py`. Three intervention families acting on this correlate failed their controls (fixed router boost, entropy-triggered adaptive boost, residual anchoring at readout; `docs/ADAPTIVE_CAUSAL.md`, `docs/ANCHOR_CAUSAL.md`), which is what demotes router starvation to a downstream symptom.

## 17. Storage vs transport (deconfounded probes)

Linear probes on the residual stream (`scripts/run_context_probe_repaired_analyze.py`, cross-paired so the probe cannot shortcut on the entity): at layer 8 at the needle's own position the fact is 0.995–1.000 decodable — including on every failing prompt — while decoding at the readout (final) position degrades on wrong answers (0.714 vs 0.960 on right answers). The fact is stored; it fails to be transported.

## 18. Retrieval-head identification and localized collapse

For every (layer, head) cell, needle attention mass `a_{l,h} = Σ_{k∈needle} A_{l,h}(q_final, k)` is measured on SHORT CORRECT prompts only; the top K=16 cells are the identified retrieval heads (`scripts/run_attention_transport.py`). OLMoE: peak L12H14 at 0.670 mass (14.3× chance), 13/16 in layers 9–14. On failing long prompts the identified heads' mass collapses 0.432 → 0.187 (d = 1.55); specificity null (2000 random 16-head sets) gives p < 0.0005 — collapse is localized to those heads, not diffuse.

## 19. Causal repair: targeted attention boost

Pre-softmax bias at the identified heads only: `logit'_{l,h}(q,k) = logit_{l,h}(q,k) + β·1[k ∈ S]` for span S (`expertatlas/attention_transport.py`); β calibrated on the dev bucket, frozen. Controls: random-head sets (matched count and β), wrong-span (matched β, wrong location), no-boost floor, oracle-span ceiling. With the oracle span: 14/14 failing prompts repaired (answer prob 0.238 → 0.986, dz = 5.55 vs baseline), transfers without recalibration to depth 0.15 (accuracy 0.375 → 1.000). Wrong-span repairs 0/14 — the mechanism is span-specific.

## 20. Label-free span detection

Registered detectors, in the order tried (`scripts/run_span_discovery.py`, `docs/SPAN_DISCOVERY_SOLVED.md`):
- Identified-heads' own attention: 85.9% span hits, 8/14 repairs, ~55% of oracle effect — blocked by a measured circularity (the signal is the collapsing pathway itself).
- Needle-affine expert activation: 0% localization (negative, preserved).
- Residual-cosine similarity: net harmful (negative, preserved).
- **IDF-weighted lexical overlap** (`s(w) = Σ_{t∈Q∩w} 1/df(t)`, argmax window, latest-window tie-break; zero forward passes, hence independent of the failing pathway by construction): 100% hits on lexical substrates; detector-driven boost recovers 100.8% of the oracle effect on OLMoE 3840 (14/14) and 99.4% at depth 0.15 (10/10), beating wrong-span at the registered bar. v1 (no tie-break) truncated the answer span and recovered 1.3% — preserved as the amendment record.
- Fallback for non-lexical (multi-hop) substrates: L8 residual probe, 90.6% hits, 61.4% of oracle effect — requires a small labeled dev set (not training-free).

## 21. Generality and scope

- Granite replication (`scripts/run_granite_transport.py`, `docs/GRANITE_TRANSPORT.md`): same pipeline end to end on a second MoE architecture — heads at the same relative stack depth (~0.7), localized collapse d = 2.17 (specificity p < 0.0005), oracle repair 5/5, lexical detector 16/16 hits at 99.1% of oracle. Caveat: failing subset n = 5, so subset-only sign-flip p bottoms at 0.0625; full-set tests carry inference. Depth alone does not break Granite (negative, preserved) — a registered 24-distractor escalation was required.
- Harder OLMoE variants (`scripts/run_context_variants.py`, `docs/CONTEXT_VARIANTS.md`): paraphrase and multi-hop sit at a capability floor (0–12.5% at 256 tokens), so results there are assisted capability, not recovered context rot. Lexical detector: 100% on paraphrase, 0% on multi-hop (locks onto the bridge sentence); L8 fallback covers multi-hop at 61.4% of oracle effect.
- Not shown: dense models, models beyond these two, contexts > 4096, naturalistic long-context tasks.