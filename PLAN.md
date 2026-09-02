# Expert Atlas — Execution Plan

**What it is:** the first published, statistically-controlled map of what individual
experts inside a Mixture-of-Experts LLM actually specialise in — shipped as a
reproducible pipeline, a self-contained 3-D visualiser, and a one-click notebook.

**Status:** plan. Nothing built yet.
**Target substrate:** OLMoE-1B-7B-0924 (1,024 experts: 16 layers × 64, top-8).
**Target hardware:** 16 GB M2 Pro. Everything below runs locally. No cloud, no GPU required.

---

## 0. Read this first: the honest framing

Prior evidence says expert specialisation is **weaker than people assume**:

- Mixtral's own routing analysis found no domain clustering — routing aligned with
  *syntax*, not topic ([arXiv 2401.04088](https://arxiv.org/abs/2401.04088)).
- Colibrì measured **"OLMoE has no hot expert zone"**
  ([issue #864](https://github.com/JustVugg/colibri/issues/864)).
- MoE training uses load-balancing objectives whose explicit purpose is to make
  marginal expert usage *uniform*. The model was optimised to defeat naive specialisation.
- Counterweight: the OLMoE paper itself reports domain and vocabulary specialisation
  ([arXiv 2409.02060](https://arxiv.org/abs/2409.02060)). So there is signal — the
  question is how much, and whether it survives a base-rate control.

**Therefore this project is designed so that a negative result is still a result.**
The contribution is *the measurement and the method*, not a particular outcome. If
specialisation turns out to be weak, "we built the instrument and specialisation is
weaker than the field assumes" is publishable, useful, and honest. That framing is
non-negotiable and shapes every design decision below.

**The single most important design consequence:** every number must be reported against
a **null model**. Raw `P(expert | domain)` is nearly meaningless under load balancing.
The primary metric is **lift**, and every claim carries a significance test with
multiple-comparison correction.

---

## 1. Pre-registered hypotheses

Write these down *before* looking at results. Do not edit them after seeing data;
add new ones as clearly-marked exploratory.

| ID | Hypothesis | Primary metric | Falsified if |
|----|-----------|----------------|--------------|
| **H1** | Some experts have domain affinity beyond chance | max lift per expert vs. shuffled-label null | < 5% of experts significant after BH-FDR at q=0.05 |
| **H2** | Specialisation is concentrated in specific layers | per-layer mean \|lift\|, ANOVA across layers | no layer differs from grand mean beyond CI |
| **H3** | Language affinity ≠ topic affinity (separable factors) | two-way factorial: lift on `lang` vs `topic` marginals | factors inseparable / fully confounded |
| **H4** | Expert co-activation forms communities beyond chance | Louvain modularity vs. degree-preserving rewired null | modularity within null CI |
| **H5** | Affinity is machine-independent | cosine sim of lift vectors across two hosts | cosine < 0.99 (would indicate nondeterminism bug, not science) |
| **H6** | Affinity replicates on held-out prompts | split-half Spearman of lift vectors | ρ < 0.5 → measuring noise |

**H6 is the gate.** If split-half correlation is low, everything downstream is noise
and the project reports that and stops. Run H6 as early as possible.

---

## 2. What is actually novel here

Be precise about the claim, because "measure expert specialisation" alone is not new.

**Taken / prior art:**
- Expert specialisation exists and is reported qualitatively (OLMoE paper, DeepSeekMoE).
- Routing traces and heat visualisation (Colibrì Brain page, [#126](https://github.com/JustVugg/colibri/issues/126)).
- Co-activation analysis for placement/prefetch (many systems papers).

**The gap this fills:**
1. **Base-rate-controlled affinity.** Prior visualisations show routing *heat* (how often
   an expert fires). Heat under load balancing is ~uniform and tells you almost nothing.
   Lift — `log P(e|d) / P(e)` — is the quantity that carries information, and nobody
   publishes an atlas built on it.
2. **A factorial probe design** that separates *language* from *topic* from *register*
   from *format*. Existing probe sets are one-dimensional category lists, which confound
   these completely. This is the methodological core.
3. **Statistical discipline at scale.** 1,024 experts × D domains is tens of thousands of
   simultaneous tests. Without FDR correction you manufacture ~5% false specialists and
   publish a map of noise. No existing effort does this.
4. **A shipped artifact.** Reproducible pipeline + self-contained interactive 3-D atlas +
   one-click notebook, versioned and citable.

**Honest scope note:** Colibrì [#175](https://github.com/JustVugg/colibri/issues/175) is
an open call for exactly this on GLM-5.2 (744B). This project is the rigorous, small-model,
fully-reproducible sibling. **Coordinate, don't compete** — the method here transfers to
their 19,456-expert run, and offering it upstream is a better outcome than racing them.

---

## 3. Architecture and frozen data contracts

**Freeze the schemas first.** Everything downstream parallelises only because these
are fixed on day one. Changing a schema mid-project serialises the whole team.

```
expert-atlas/
├── expertatlas/
│   ├── __init__.py
│   ├── capture.py        # router hooks → RoutingTrace
│   ├── probes.py         # probe set loading + validation
│   ├── stats.py          # lift, significance, FDR, null models
│   ├── coactivation.py   # co-firing graph + communities
│   ├── layout.py         # UMAP/t-SNE → 3-D coords
│   ├── export.py         # → atlas.json
│   └── cli.py            # `atlas capture|analyze|export|serve`
├── probes/probe_set_v1.yaml
├── viz/                  # self-contained HTML + Three.js
├── notebooks/expert_atlas_colab.ipynb
├── tests/
├── data/                 # gitignored: traces, atlas.json
└── docs/
```

### Contract 1 — `RoutingTrace` (capture output)

Parquet, one row per (token, layer). Columns:

| column | dtype | meaning |
|---|---|---|
| `prompt_id` | int32 | index into probe set |
| `token_pos` | int32 | position in sequence |
| `token_id` | int32 | vocabulary id |
| `layer` | int16 | MoE layer index |
| `expert_ids` | list[int16] | top-k selected, length k |
| `gate_weights` | list[float32] | post-softmax weights, length k |

Sidecar `meta.json`: model id + revision, dtype, device, k, n_layers, n_experts,
transformers version, torch version, seed, git commit, timestamp, host fingerprint.

### Contract 2 — `atlas.json` (visualiser input, versioned)

```jsonc
{
  "schema_version": "1.0",
  "model": { "id": "allenai/OLMoE-1B-7B-0924", "revision": "...",
             "n_layers": 16, "n_experts_per_layer": 64, "top_k": 8 },
  "probe_set": { "id": "probe_set_v1", "n_prompts": 480, "factors": ["topic","lang","register","format"] },
  "stats": { "n_tokens": 250000, "null_model": "label_shuffle", "n_permutations": 1000,
             "fdr_method": "benjamini_hochberg", "q": 0.05 },
  "experts": [
    { "uid": "L03E17", "layer": 3, "idx": 17,
      "usage": 0.0161,                       // marginal P(expert) — expect ~1/64
      "lift": { "code.python": 0.84, "lang.zh": -0.31 },
      "significant": ["code.python"],         // survived FDR
      "max_lift": 0.84, "specialisation": 0.42, // normalised, 0=generalist 1=specialist
      "top_tokens": [ {"token":"def","lift":1.9}, ... ],
      "community": 4,
      "xyz": [0.12, -0.88, 0.41] }
  ],
  "communities": [ { "id": 4, "size": 37, "label": "python-ish", "modularity_contrib": 0.03 } ],
  "coactivation": { "edges": [[uid_i, uid_j, weight], ...], "null_modularity_ci": [0.11, 0.14] }
}
```

**Rule:** the visualiser reads *only* `atlas.json`. It never touches traces. This is what
lets the frontend be built in parallel against a synthetic fixture before real data exists.

---

## 4. Agentic workflow — how to parallelise this

**Sequencing principle:** Phase 0 is strictly serial (contracts + sanity). Everything
after is parallel across five workstreams with no shared files.

```
Phase 0 (SERIAL, blocking) ── contracts frozen, sanity harness green
        │
        ├── WS-A  capture engine        (owns expertatlas/capture.py)
        ├── WS-B  probe set design      (owns probes/)
        ├── WS-C  statistics            (owns stats.py, coactivation.py)
        ├── WS-D  visualiser            (owns viz/)   ← unblocked by fixture, not real data
        └── WS-E  packaging/notebook    (owns notebooks/, README, CI)
        │
Phase 3 (SERIAL) ── integration, real run, H1–H6 evaluation
        │
Phase 4 ── writeup + publish
```

**File ownership is exclusive.** Two agents must never write the same file. If a
workstream needs a change in another's file, it files a note in `docs/interface-requests.md`
and the owner makes the change.

**Launch template (one agent per workstream, after Phase 0 is green):**

> You own `<paths>` exclusively. Do not edit files outside them.
> Contracts in PLAN.md §3 are frozen — read them, do not change them.
> Deliver: implementation + tests in `tests/<ws>/` + a 10-line summary in `docs/<ws>.md`.
> Every test must pass `pytest tests/<ws>`. Report honestly if a contract is unworkable —
> do not silently work around it.

**Critical:** WS-D (visualiser) builds against `tests/fixtures/atlas_synthetic.json` —
a hand-made atlas with *known planted structure* (see §6). This means the frontend is
fully built and testable before a single real token is captured, and the fixture doubles
as the visualiser's correctness test.

---

## 5. Phases and checkpoints

### Phase 0 — Foundation (SERIAL, ~1 session)

- [ ] `pyproject.toml`, deps pinned: `torch`, `transformers`, `pyarrow`, `numpy`,
      `scipy`, `statsmodels`, `umap-learn`, `networkx`, `python-louvain`, `pytest`, `typer`
- [ ] Contracts §3 written into `expertatlas/schemas.py` as dataclasses + JSON Schema
- [ ] `tests/fixtures/atlas_synthetic.json` generated with planted structure
- [ ] Model loads and one forward pass emits router logits
- [ ] **All of §6.1 sanity tests green**

**CHECKPOINT 0 — do not proceed until:**
`pytest tests/sanity -v` is fully green, and `atlas capture --prompt "def foo():" --dry-run`
prints a well-formed trace. If router logits can't be extracted for the chosen model,
**stop and re-select the model** — everything depends on this.

### Phase 1 — Capture + probes (PARALLEL: WS-A, WS-B)

**WS-A — capture engine**
- [ ] Router hook via `output_router_logits=True`; fall back to forward hooks on
      `*.mlp.gate` modules if unavailable
- [ ] Batched capture with deterministic ordering; greedy decode only (no sampling)
- [ ] Parquet writer, streaming (never hold all traces in RAM)
- [ ] Resume-from-checkpoint (long runs must survive interruption)
- [ ] `--limit`, `--layers`, `--seed`, `--device` flags

**WS-B — probe set** (the scientific core; see §7 for the design)
- [ ] `probes/probe_set_v1.yaml` — factorial, ≥480 prompts
- [ ] Validation: balance across factor levels, length-matched within ±15% tokens
- [ ] Held-out split (50/50) declared in the file, for H6

**CHECKPOINT 1:** capture 20k tokens on 40 prompts. Verify: exactly `top_k` experts per
(token, layer); gate weights sum to 1.0 ± 1e-4; marginal usage within 3σ of uniform;
two identical runs produce byte-identical traces.

### Phase 2 — Statistics + visualiser (PARALLEL: WS-C, WS-D)

**WS-C — statistics**
- [ ] `P(e|d)`, marginal `P(e)`, **lift** `log2(P(e|d)/P(e))` with Laplace smoothing
- [ ] Null model: label-shuffle permutation (≥1000 perms), preserving prompt lengths
- [ ] Per-expert-per-domain chi-squared; **Benjamini–Hochberg FDR at q=0.05**
- [ ] Effect size (Cramér's V) reported alongside every p-value
- [ ] Specialisation score: normalised entropy of the lift profile, 0=generalist 1=specialist
- [ ] Split-half replication (H6) — **report this first**
- [ ] Co-activation: within-layer and adjacent-layer co-firing, Louvain communities,
      degree-preserving rewired null for modularity CI

**WS-D — visualiser** (builds against the synthetic fixture)
- [ ] Three.js scene, 1,024 instanced points (`InstancedMesh` — do not use 1,024 meshes)
- [ ] Orbit + fly controls; layer as depth axis or filter
- [ ] Colour = community; size = usage; brightness = specialisation
- [ ] Hover → expert card: top domains by lift, top tokens, usage, significance flags
- [ ] Click → isolate expert + its co-activation edges
- [ ] Filter panel: by layer, by significance, by domain
- [ ] **Replay mode**: step a prompt token-by-token, flash experts as they fire
- [ ] Single self-contained HTML, atlas.json inlined, works offline, no CDN
- [ ] Renders at 60 fps with 1,024 points and ~5k edges on an M2 Pro

**CHECKPOINT 2:** visualiser loads the synthetic fixture and **visibly recovers the
planted structure** (§6.2). Stats module recovers planted lift values within tolerance
on the same fixture.

### Phase 3 — Integration and the real run (SERIAL)

- [ ] Full capture: ≥250k tokens across the full probe set
- [ ] Run H1–H6 in order. **H6 first** — if replication fails, stop and write that up.
- [ ] Generate real `atlas.json`
- [ ] Cross-host run (H5) — second machine or a Colab instance
- [ ] Second model (Qwen1.5-MoE-A2.7B or DeepSeek-V2-Lite) to test generality

**CHECKPOINT 3:** every hypothesis has a verdict with effect size and CI. Nothing is
reported without a null comparison.

### Phase 4 — Ship (PARALLEL: WS-E + writeup)

- [ ] Colab notebook: pip install → capture → analyse → inline 3-D atlas, < 15 min on free T4
- [ ] GitHub Pages deploy of the interactive atlas
- [ ] `README.md` with the headline number and one screenshot/GIF
- [ ] `docs/METHOD.md` — full statistical method, reproducible
- [ ] `docs/FINDINGS.md` — results including negatives
- [ ] Zenodo DOI for `atlas.json`
- [ ] Offer the method + code to Colibrì [#175](https://github.com/JustVugg/colibri/issues/175)

---

## 6. Test suite

### 6.1 Sanity tests (Phase 0 gate — all must pass before any science)

```python
# tests/sanity/test_capture_fidelity.py
def test_exactly_topk_experts_selected():
    """Every (token, layer) selects exactly k experts, no duplicates."""

def test_full_softmax_sums_to_one():
    """Softmax over ALL n_experts sums to 1.0 ± 1e-4.

    NOTE: OLMoE sets norm_topk_prob=False, so the TOP-K weights do NOT
    sum to 1 — they sum to the top-k probability mass (< 1). Asserting
    top-k sum == 1 is a real bug that would fail for the wrong reason.
    Models with norm_topk_prob=True (e.g. Mixtral) renormalise and DO
    sum to 1. Read the config; do not assume."""

def test_topk_mass_is_recorded():
    """Top-k probability mass (sum of selected gate weights) is stored per
    (token, layer). This is 'routing confidence' and is itself a signal:
    low mass = the router was undecided. Do not discard it."""

def test_expert_ids_in_range():
    """All expert ids in [0, n_experts). Catches off-by-one in hook indexing."""

def test_determinism_same_prompt():
    """Same prompt, greedy, same seed → byte-identical trace. Twice."""

def test_no_silent_layer_skips():
    """Trace contains every MoE layer for every token. Catches hooks not firing."""

# tests/sanity/test_base_rates.py
def test_marginal_usage_near_uniform():
    """Aggregate P(expert) within 3σ of 1/n_experts.
    Load balancing implies this. A large deviation means a capture bug
    OR a genuinely unbalanced model — investigate before proceeding."""

def test_no_dead_experts_unflagged():
    """Experts with zero activations are explicitly reported, not silently dropped."""

# tests/sanity/test_null_model.py
def test_shuffled_labels_give_zero_lift():
    """Under label shuffling, mean lift ≈ 0 and the significance test
    yields ≈ q*n false positives. Calibrates the whole statistical pipeline.
    If this fails, every downstream number is wrong."""

def test_fdr_controls_false_positives():
    """On pure-noise input, BH-FDR at q=0.05 flags ≤5% of experts."""
```

### 6.2 Planted-structure test (the key correctness test)

Generate a synthetic trace where **expert 17 in layer 3 fires 5× more often on
`code.python` prompts than base rate**, and nothing else is structured.

```python
def test_recovers_planted_specialist():
    """The pipeline must find L03E17 as significant for code.python,
    with lift ≈ log2(5) ≈ 2.32 ± 0.15, and must NOT flag more than
    q*n_experts other experts as significant."""

def test_visualiser_renders_planted_cluster():
    """atlas.json from planted data → the planted expert is visually
    separated in the 3-D layout (distance from centroid > 2σ)."""
```

This is the single most valuable test in the project. It proves the instrument works
before you trust anything it says about a real model.

### 6.3 Statistical integrity tests

```python
def test_split_half_replication_on_planted_data():
    """Planted signal replicates across halves (ρ > 0.9); noise does not (ρ ≈ 0)."""

def test_lift_is_base_rate_corrected():
    """An expert used 10× more overall but proportionally across domains
    must have lift ≈ 0 in every domain. Guards against reporting heat as affinity —
    the single most likely way to publish a wrong atlas."""

def test_coactivation_null_is_degree_preserving():
    """Rewired null preserves degree sequence; naive Erdős–Rényi null
    would inflate modularity and manufacture fake communities."""
```

### 6.4 Performance / integration

```python
def test_capture_memory_bounded():
    """Capturing 100k tokens never exceeds 6 GB RSS (streaming writer works)."""

def test_capture_resumable():
    """Kill at 50%, resume, result identical to uninterrupted run."""

def test_viz_frame_budget():
    """1,024 instanced points + 5k edges renders < 16 ms/frame."""

def test_atlas_json_schema_valid():
    """Output validates against the JSON Schema in schemas.py."""
```

---

## 7. Probe set design (the methodological core)

Naive probe sets ("10 categories × 3 prompts") confound language with topic with
register with format. The factorial design fixes this.

**Factors:**
- `topic` ∈ {python, rust, sql, regex, math_proof, law, medicine, music_theory, cooking, history}
- `lang` ∈ {en, zh, de, ja}
- `register` ∈ {formal, casual}
- `format` ∈ {prose, json, bulleted}

Full crossing is 10×4×2×3 = 240 cells. Use **≥2 prompts per cell = 480 prompts**,
length-matched within ±15% tokens.

This lets you compute **marginal** lift per factor and answer questions no
one-dimensional probe set can:
- Is an expert a *Chinese* expert, or a *Chinese-legal-text* expert?
- Is "code affinity" actually **syntax affinity** — i.e. does it also fire on JSON and
  regex? (This directly tests Mixtral's syntax-not-semantics finding, and is the most
  scientifically interesting question in the whole project.)

**Validation gates:** balanced cells, length-matched, no near-duplicate prompts
(cosine < 0.9 on embeddings), and 50/50 held-out split declared in the YAML.

---

## 8. Visualiser specification

**Non-negotiable:** single self-contained HTML. No CDN. Works offline, opens by
double-click, publishable to GitHub Pages and embeddable anywhere.

**Layout:** UMAP (3 components) on the **lift vectors** — not raw counts. Metric: cosine.
Add small deterministic jitter to co-located points. Layer available as a filter *and*
as an alternate "stacked layers" layout toggle.

**Encoding:**
| channel | variable |
|---|---|
| position | UMAP of lift vector |
| colour | co-activation community |
| size | marginal usage |
| brightness/opacity | specialisation score |
| ring/outline | survived FDR significance |

**Interactions:** orbit + WASD fly · hover card · click to isolate + show co-activation
edges · filter by layer/domain/significance · **replay mode** (token-by-token firing) ·
search by expert uid or domain.

**Perf:** `THREE.InstancedMesh` for points, `LineSegments` with a single BufferGeometry
for edges. Do not create per-expert objects.

**Accessibility:** community colours from a colourblind-safe palette; never encode
information in colour alone (significance also gets an outline).

---

## 9. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| **Specialisation is real but tiny** | **High** | Negative result is pre-framed as the contribution. H6 gates early. |
| Router logits unavailable for chosen model | Medium | Phase 0 gate; forward-hook fallback; 3 candidate models |
| Colibrì ships their atlas first | Medium | Different model + rigour angle; offer method upstream, don't race |
| Multiple-comparison noise mistaken for signal | **High if careless** | BH-FDR + planted-structure test + null model are mandatory |
| Confounded probe set → wrong conclusions | High | Factorial design + length matching + held-out split |
| UMAP layout is unstable / over-interpreted | Medium | Fixed seed; report that layout is presentational, clustering is from co-activation |
| Scope creep into a full inference engine | Medium | Out of scope. This is measurement + visualisation only. |
| 16 GB RAM insufficient | Low | OLMoE bf16 ≈ 14 GB; use 8-bit if tight; capture is streaming |

---

## 9b. Honest effort estimate and known gaps in this plan

**What is actually done:** Phase 0 only. `capture.py` works, 12 sanity tests pass. Every
test in §6.2–6.4 is a *specification*, not code. Do not mistake the docstrings for a suite.

**The critical path is the probe set, and it does not exist yet.** §7 describes the design;
there are zero prompts written. 480 length-matched prompts across a 240-cell factorial is
the single largest piece of human judgment in the project and the thing most likely to be
done badly. Everything downstream inherits its flaws. Budget accordingly and do not
delegate it blindly.

**Effort, assuming evenings/weekends with an agent doing most of the typing:**

| Phase | Estimate | Risk |
|---|---|---|
| 0 — foundation | **done** | — |
| 1 — capture engine (WS-A) | 1–2 sessions | low |
| 1 — probe set (WS-B) | 2–4 sessions | **high — needs your judgment** |
| 2 — statistics (WS-C) | 2–3 sessions | medium (null models are fiddly) |
| 2 — visualiser (WS-D) | 2–3 sessions | low (fixture unblocks it immediately) |
| 3 — real run + hypotheses | 1–2 sessions | **the run itself is cheap** |
| 4 — notebook, Pages, writeup | 2–3 sessions | low |

**≈2–4 weeks of evenings.** Note the compute is *not* the bottleneck: OLMoE is ~1B active
params, forward-pass only, so 250k tokens is well under an hour on an M2 Pro. This is a
measurement-and-rigour project, not a compute project. That is exactly why it is feasible
solo — and it is the main reason to prefer it over anything in the streaming/offloading space.

**Gaps this plan does not yet resolve — decide these in Phase 1:**

1. **`atlas.json` size budget.** 1,024 experts × 240 factor cells × float + top-tokens could
   reach several MB inlined into a single HTML file. Set a hard budget (target < 5 MB
   gzipped): store lift as sparse (significant entries only), quantise to float16, cap
   `top_tokens` at 10. Decide before WS-D hardcodes a loader.
2. **Is 240 domain-dimensions enough for a meaningful UMAP over only 1,024 points?**
   Likely yes, but verify on the planted fixture: if planted specialists don't separate
   visually, the layout is decorative and must be labelled as such.
3. **Does OLMoE generalise?** 64 experts/layer is far from frontier scale (Kimi K3: 896,
   GLM-5.2: 256). The second-model check in Phase 3 is the only defence against a
   substrate-specific artifact. Do not skip it.
4. **Coordinate with Colibrì [#175](https://github.com/JustVugg/colibri/issues/175) NOW,
   not in Phase 4.** Post the method — factorial probes, lift-not-heat, FDR — as a comment
   early. Two outcomes, both good: they adopt it and you have collaborators on a
   19,456-expert run, or they say it's covered and you save weeks. Racing silently is the
   worst option.

## 10. Definition of done

- [ ] `pytest` fully green, including planted-structure recovery
- [ ] Real atlas over ≥250k tokens with every hypothesis adjudicated against a null
- [ ] Self-contained HTML atlas live on GitHub Pages
- [ ] Colab notebook runs end-to-end on a free T4 in < 15 min
- [ ] `FINDINGS.md` reports negatives as prominently as positives
- [ ] `atlas.json` + code archived with a DOI
- [ ] Method offered to Colibrì #175

**The headline, if H1 holds:**
> *"We mapped what all 1,024 experts in an open MoE actually do — base-rate corrected,
> FDR controlled, and you can fly through it in your browser."*

**The headline, if H1 fails:**
> *"We built the instrument to measure expert specialisation properly. Under a
> base-rate-controlled null, most claimed specialisation does not survive."*

Both are worth publishing. That is the point.
