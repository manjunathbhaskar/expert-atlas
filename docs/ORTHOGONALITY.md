# Orthogonality analysis

Factor: `topic` (10 domains). Routing signature = base-rate-corrected lift vector per domain (immune to the usage-skew problem that made H4 unreliable — see docs/FINDINGS.md). Null: 200 label-shuffle permutations (same null used everywhere else in this project, expertatlas/stats.py::shuffle_labels).

**Honest limit**: this measures ROUTING orthogonality only. Two domains could route to identical experts yet produce orthogonal activations inside them — this is evidence about the routing layer specifically, not the full computation (TRANSFER.md §6.3).

## Headline
Observed mean |cosine similarity| across domain pairs: **0.2720**
Null distribution: 0.1082 +/- 0.0009
**z = 180.55** (domains are LESS orthogonal than chance (overlapping))

## Most orthogonal domain pairs (observed - null, most negative = most separated)
- history vs python: cosine=-0.250 (null=-0.105, delta=-0.145)
- history vs rust: cosine=-0.248 (null=-0.105, delta=-0.144)
- history vs sql: cosine=-0.202 (null=-0.110, delta=-0.092)
- history vs math_proof: cosine=-0.187 (null=-0.104, delta=-0.083)
- cooking vs history: cosine=-0.185 (null=-0.112, delta=-0.073)

## Least orthogonal (most overlapping) domain pairs
- math_proof vs python: cosine=0.743 (null=-0.105, delta=+0.849)
- math_proof vs rust: cosine=0.757 (null=-0.108, delta=+0.865)
- python vs sql: cosine=0.825 (null=-0.107, delta=+0.932)
- rust vs sql: cosine=0.862 (null=-0.106, delta=+0.968)
- python vs rust: cosine=0.898 (null=-0.109, delta=+1.007)

## Interpretation
Domain routing signatures show MORE overlap than chance, not less — this would argue against a simple 'freeze old experts, add new ones' continual-learning strategy on this substrate: the same experts are shared across nominally different domains more than random routing would produce.
