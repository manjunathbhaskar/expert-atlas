"""Workstream 3: expert utilization + hot/specialist cross-reference.

Uses only existing parquet traces and data/atlas.json — no new capture.
Emits data/utilization.json, which Workstreams 1 and 2 consume.

    python scripts/run_utilization.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from expertatlas.utilization import compute_utilization, hot_specialist_overlap  # noqa: E402

TRACES = ROOT / "data" / "traces"
ATLAS = ROOT / "data" / "atlas.json"
OUT = ROOT / "data" / "utilization.json"


def load_counts(n_layers: int, n_experts: int) -> tuple[np.ndarray, int]:
    """Accumulate per-(layer, expert) selection counts across every trace."""
    counts = np.zeros(n_layers * n_experts, dtype=np.float64)
    files = sorted(TRACES.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no traces in {TRACES} — run scripts/run_phase3_capture.py first")

    n_rows = 0
    for f in files:
        t = pq.read_table(f, columns=["layer", "expert_ids"])
        layers = t.column("layer").to_numpy()
        experts = t.column("expert_ids").to_pylist()
        n_rows += len(layers)
        for layer, ids in zip(layers, experts):
            base = int(layer) * n_experts
            for e in ids:
                counts[base + int(e)] += 1.0
    return counts, n_rows


def main() -> int:
    atlas = json.loads(ATLAS.read_text())
    n_layers = atlas["model"]["n_layers"]
    n_experts = atlas["model"]["n_experts_per_layer"]
    top_k = atlas["model"]["top_k"]

    print(f"model: {atlas['model']['id']} — {n_layers}x{n_experts}, top-{top_k}")
    counts, n_rows = load_counts(n_layers, n_experts)
    print(f"accumulated {int(counts.sum()):,} expert selections over {n_rows:,} rows")

    uids = [f"L{l:02d}E{e:02d}" for l in range(n_layers) for e in range(n_experts)]
    stats = compute_utilization(counts, n_experts, top_k, uids=uids)

    print("\n--- utilization ---")
    print(f"  expected share/expert : {stats.expected_share:.6f} (balanced router)")
    print(f"  load ratio  min/med/max: {stats.load_ratio.min():.3f} / "
          f"{np.median(stats.load_ratio):.3f} / {stats.load_ratio.max():.3f}")
    print(f"  skew (max/min)        : {stats.skew:.1f}x")
    print(f"  Gini                  : {stats.gini:.4f}")
    print(f"  coefficient of var    : {stats.coefficient_of_variation:.4f}")
    print(f"  hot (>=2x fair share) : {stats.n_hot} / {counts.size}")
    print(f"  cold (<=0.5x)         : {stats.n_cold}")
    print(f"  dead (never fired)    : {stats.n_dead}")

    # Specialists = experts with >=1 domain that is FDR-significant AND |lift|>=1.0,
    # matching docs/FINDINGS.md's H1 definition exactly (not the raw per-cell count).
    specialists = {
        e["uid"] for e in atlas["experts"]
        if any(abs(e.get("lift", {}).get(d, 0.0)) >= 1.0 for d in e.get("significant", []))
    }
    print(f"\n--- hot / specialist cross-reference ---")
    print(f"  specialists (FDR sig AND |lift|>=1.0): {len(specialists)}")

    overlap = hot_specialist_overlap(stats, specialists, n_permutations=10_000, seed=0)
    print(f"  observed overlap : {overlap['observed_overlap']}")
    print(f"  null             : {overlap['null_mean']:.1f} +/- {overlap['null_std']:.1f}")
    print(f"  enrichment       : {overlap['enrichment']:.3f}x")
    print(f"  p (permutation)  : {overlap['p_value']:.4f}")
    print(f"  VERDICT          : {overlap['verdict']}")

    payload = {
        "model": atlas["model"],
        "n_expert_selections": int(counts.sum()),
        "utilization": stats.to_json(),
        "hot_specialist_overlap": overlap,
        "specialist_uids": sorted(specialists),
        "notes": (
            "load_ratio == 1.0 means the expert carries exactly its fair share "
            "(1/(n_layers*n_experts) of all selections). hot >= 2.0, cold <= 0.5. "
            "Skew is reported so consumers can gate on coactivation.py's 2.0x PMI "
            "validity limit — see docs/FINDINGS.md H4."
        ),
    }
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
