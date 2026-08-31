"""Generate tests/fixtures/atlas_synthetic.json — a hand-made atlas with
KNOWN planted structure (PLAN.md §6.2).

This is the single most valuable artifact in the project: it lets WS-D
(visualiser) be built and correctness-tested before a single real token is
captured, and it lets WS-C's stats pipeline prove it recovers a known signal
before it's trusted on a real model.

Planted structure:
  - Expert L03E17 fires 5x more often on domain 'code.python' than its
    base rate -> lift = log2(5) ~= 2.3219.
  - Every other expert has domain-uniform usage -> lift ~= 0 everywhere
    (small noise added so the pipeline has something realistic to reject).
  - Experts 0-9 in layer 0 form a co-activation community (community id 0);
    everything else is unclustered (community id -1) or singleton clusters.

Run: python tests/fixtures/make_synthetic_fixture.py
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

N_LAYERS = 16
N_EXPERTS_PER_LAYER = 64
TOP_K = 8
DOMAINS = ["code.python", "code.rust", "lang.zh", "lang.de", "math_proof", "cooking"]

PLANTED_LAYER = 3
PLANTED_IDX = 17
PLANTED_DOMAIN = "code.python"
PLANTED_LIFT = math.log2(5.0)  # ~= 2.3219

rng = random.Random(20260811)  # deterministic fixture


def expert_uid(layer: int, idx: int) -> str:
    return f"L{layer:02d}E{idx:02d}"


def make_experts() -> list[dict]:
    experts = []
    for layer in range(N_LAYERS):
        for idx in range(N_EXPERTS_PER_LAYER):
            uid = expert_uid(layer, idx)
            is_planted = layer == PLANTED_LAYER and idx == PLANTED_IDX

            lift = {}
            for d in DOMAINS:
                if is_planted and d == PLANTED_DOMAIN:
                    lift[d] = PLANTED_LIFT
                else:
                    # tiny symmetric noise, mean 0 -> nothing else should
                    # survive FDR correction
                    lift[d] = rng.gauss(0.0, 0.03)

            significant = [PLANTED_DOMAIN] if is_planted else []
            max_lift = max(lift.values())
            specialisation = 0.9 if is_planted else round(abs(rng.gauss(0.05, 0.02)), 4)

            # community 0 = a deliberate 10-expert clique in layer 0;
            # everything else singleton (-1 = unclustered)
            if layer == 0 and idx < 10:
                community = 0
            else:
                community = -1

            # deterministic layout: planted expert placed far from origin,
            # community-0 experts clustered together, everything else near
            # the origin with small jitter
            if is_planted:
                xyz = (8.0, 0.0, 0.0)
            elif community == 0:
                xyz = (
                    -5.0 + rng.uniform(-0.3, 0.3),
                    -5.0 + rng.uniform(-0.3, 0.3),
                    idx * 0.05,
                )
            else:
                xyz = (
                    rng.uniform(-1, 1),
                    rng.uniform(-1, 1),
                    rng.uniform(-1, 1),
                )

            experts.append(
                {
                    "uid": uid,
                    "layer": layer,
                    "idx": idx,
                    "usage": round(1.0 / N_EXPERTS_PER_LAYER + rng.gauss(0, 0.0005), 6),
                    "lift": {k: round(v, 4) for k, v in lift.items()},
                    "significant": significant,
                    "max_lift": round(max_lift, 4),
                    "specialisation": specialisation,
                    "top_tokens": (
                        [{"token": "def", "lift": 1.9}, {"token": "class", "lift": 1.2}]
                        if is_planted
                        else []
                    ),
                    "community": community,
                    "xyz": xyz,
                }
            )
    return experts


def make_communities() -> list[dict]:
    return [
        {"id": 0, "size": 10, "label": "planted-clique-layer0", "modularity_contrib": 0.08},
        {"id": -1, "size": N_LAYERS * N_EXPERTS_PER_LAYER - 10, "label": "unclustered", "modularity_contrib": 0.0},
    ]


def make_coactivation() -> dict:
    edges = []
    for i in range(10):
        for j in range(i + 1, 10):
            edges.append([expert_uid(0, i), expert_uid(0, j), round(rng.uniform(0.4, 0.9), 3)])
    return {"edges": edges, "null_modularity_ci": [0.11, 0.14]}


def main() -> None:
    atlas = {
        "schema_version": "1.0",
        "model": {
            "id": "synthetic/fixture",
            "revision": "n/a",
            "n_layers": N_LAYERS,
            "n_experts_per_layer": N_EXPERTS_PER_LAYER,
            "top_k": TOP_K,
        },
        "probe_set": {"id": "synthetic_v1", "n_prompts": 480, "factors": ["topic", "lang", "register", "format"]},
        "stats": {
            "n_tokens": 250_000,
            "null_model": "label_shuffle",
            "n_permutations": 1000,
            "fdr_method": "benjamini_hochberg",
            "q": 0.05,
        },
        "experts": make_experts(),
        "communities": make_communities(),
        "coactivation": make_coactivation(),
        # not part of the frozen schema — informational, for test assertions only
        "_planted": {
            "expert_uid": expert_uid(PLANTED_LAYER, PLANTED_IDX),
            "domain": PLANTED_DOMAIN,
            "expected_lift": PLANTED_LIFT,
            "expected_lift_tolerance": 0.15,
        },
    }

    out = Path(__file__).parent / "atlas_synthetic.json"
    out.write_text(json.dumps(atlas, indent=2))
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
