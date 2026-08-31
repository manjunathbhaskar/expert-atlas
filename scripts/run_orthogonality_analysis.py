"""Orthogonality analysis (discussed with the user, TRANSFER.md §6.3).

Mechanism from the continual-learning literature (Continual task learning in
natural and artificial agents, PMC 2022, user-supplied citation): brains
avoid catastrophic interference not by strict "separate modules" but by
mapping tasks to near-orthogonal representational subspaces. If task A and
task B live in orthogonal subspaces, updating one barely disturbs the other.

This asks the MoE analogue: are different topics' ROUTING SIGNATURES
(which experts fire, and how much, relative to base rate) orthogonal to
each other, or do they overlap heavily?

Deliberate design choice: this uses LIFT vectors, not raw usage counts, as
each domain's routing signature. Lift is already base-rate corrected
(log2(P(e|d)/P(e))), so this analysis is immune to the 227x usage-skew
problem that made H4's raw-count-based PMI co-activation result unreliable
(see docs/FINDINGS.md) -- a generic expert that's simply popular everywhere
contributes ~0 to every domain's lift vector, so it can't manufacture false
"orthogonality" or "overlap" the way raw popularity would.

Null model: labels are shuffled (expertatlas.stats.shuffle_labels, the same
null used everywhere else in this project) N times; lift and pairwise cosine
similarity are recomputed each time. Observed pairwise similarities are
compared against this null distribution, not against 0 -- cosine similarity
between two random sparse-ish vectors is not 0 in general, so the null must
be empirical, not assumed.

Honest limit (must ship with any result, per TRANSFER.md §6.3): this
measures ROUTING orthogonality only. Two domains could route to identical
experts yet produce orthogonal activations INSIDE those experts. This is
evidence about the routing layer specifically, not the full computation.

Usage:
    python scripts/run_orthogonality_analysis.py [--factor topic] [--n-null 200]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from expertatlas.aggregate import aggregate_counts
from expertatlas.stats import compute_lift, shuffle_labels

REPO_ROOT = Path(__file__).parent.parent
TRACES_DIR = REPO_ROOT / "data" / "traces"
PROBE_SET_PATH = REPO_ROOT / "probes" / "probe_set_v1.yaml"
OUT_PATH = REPO_ROOT / "docs" / "ORTHOGONALITY.md"

N_LAYERS, N_EXPERTS = 16, 64


def load_rows():
    probe_set = yaml.safe_load(PROBE_SET_PATH.read_text())
    prompts_by_id = {p["prompt_id"]: p for p in probe_set["prompts"]}
    shards = sorted(TRACES_DIR.glob("trace_*.parquet"))
    if not shards:
        raise RuntimeError(f"no trace shards under {TRACES_DIR} -- run capture first")
    rows = []
    for shard in shards:
        table = pq.read_table(shard)
        cols = {name: table.column(name).to_pylist() for name in table.column_names}
        n = len(cols["prompt_id"])
        for i in range(n):
            rows.append({k: cols[k][i] for k in cols})
    return rows, prompts_by_id


def pairwise_cosine(lift: np.ndarray) -> np.ndarray:
    """lift: (n_experts_total, n_domains). Returns (n_domains, n_domains)
    cosine similarity between domain lift-vectors (columns)."""
    v = lift.T  # (n_domains, n_experts_total)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    vn = v / norms
    return vn @ vn.T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", default="topic", choices=["topic", "lang", "register", "format"])
    parser.add_argument("--n-null", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"loading traces + probe set (factor={args.factor})...")
    rows, prompts_by_id = load_rows()

    cm = aggregate_counts(rows, prompts_by_id, N_LAYERS, N_EXPERTS, domain_factor=args.factor, seed=args.seed)
    lift = compute_lift(cm.counts)
    domains = cm.domain_labels
    n_domains = len(domains)

    observed = pairwise_cosine(lift)
    off_diag_mask = ~np.eye(n_domains, dtype=bool)
    observed_mean_abs = float(np.abs(observed[off_diag_mask]).mean())
    observed_max_abs = float(np.abs(observed[off_diag_mask]).max())

    print(f"observed: mean|cosine|={observed_mean_abs:.4f}, max|cosine|={observed_max_abs:.4f} "
          f"across {n_domains} domains ({n_domains * (n_domains - 1)} off-diagonal pairs)")

    print(f"building empirical null ({args.n_null} label-shuffle permutations)...")
    rng = np.random.default_rng(args.seed)
    null_mean_abs = np.zeros(args.n_null)
    null_max_abs = np.zeros(args.n_null)
    null_pairwise_sum = np.zeros((n_domains, n_domains))

    for t in range(args.n_null):
        shuffled_counts = shuffle_labels(cm.counts, rng)
        shuffled_lift = compute_lift(shuffled_counts)
        sim = pairwise_cosine(shuffled_lift)
        null_mean_abs[t] = np.abs(sim[off_diag_mask]).mean()
        null_max_abs[t] = np.abs(sim[off_diag_mask]).max()
        null_pairwise_sum += sim

    null_pairwise_mean = null_pairwise_sum / args.n_null
    z = (observed_mean_abs - null_mean_abs.mean()) / max(null_mean_abs.std(), 1e-9)

    print(f"null: mean|cosine|={null_mean_abs.mean():.4f} +/- {null_mean_abs.std():.4f}")
    print(f"observed vs null: z={z:.2f} "
          f"({'LOWER (more orthogonal) than null' if observed_mean_abs < null_mean_abs.mean() else 'HIGHER (less orthogonal / more overlap) than null'})")

    # Which domain pairs are most/least orthogonal (observed vs null-corrected)?
    delta = observed - null_pairwise_mean
    pairs = []
    for i in range(n_domains):
        for j in range(i + 1, n_domains):
            pairs.append((domains[i], domains[j], float(observed[i, j]), float(null_pairwise_mean[i, j]), float(delta[i, j])))
    pairs.sort(key=lambda x: x[4])  # most negative delta = most orthogonal relative to null

    lines = [
        "# Orthogonality analysis",
        "",
        f"Factor: `{args.factor}` ({n_domains} domains). Routing signature = base-rate-corrected "
        "lift vector per domain (immune to the usage-skew problem that made H4 unreliable — "
        "see docs/FINDINGS.md). Null: {} label-shuffle permutations (same null used everywhere "
        "else in this project, expertatlas/stats.py::shuffle_labels).".format(args.n_null),
        "",
        "**Honest limit**: this measures ROUTING orthogonality only. Two domains could route "
        "to identical experts yet produce orthogonal activations inside them — this is evidence "
        "about the routing layer specifically, not the full computation (TRANSFER.md §6.3).",
        "",
        "## Headline",
        f"Observed mean |cosine similarity| across domain pairs: **{observed_mean_abs:.4f}**",
        f"Null distribution: {null_mean_abs.mean():.4f} +/- {null_mean_abs.std():.4f}",
        f"**z = {z:.2f}** ({'domains are MORE orthogonal than chance' if z < -2 else 'domains are LESS orthogonal than chance (overlapping)' if z > 2 else 'not distinguishable from chance'})",
        "",
        "## Most orthogonal domain pairs (observed - null, most negative = most separated)",
    ]
    for d1, d2, obs, null, delta_v in pairs[:5]:
        lines.append(f"- {d1} vs {d2}: cosine={obs:.3f} (null={null:.3f}, delta={delta_v:+.3f})")
    lines.append("")
    lines.append("## Least orthogonal (most overlapping) domain pairs")
    for d1, d2, obs, null, delta_v in pairs[-5:]:
        lines.append(f"- {d1} vs {d2}: cosine={obs:.3f} (null={null:.3f}, delta={delta_v:+.3f})")

    lines += [
        "",
        "## Interpretation",
    ]
    if z < -2:
        lines.append(
            "Domain routing signatures are **more orthogonal than a label-shuffled null** — "
            "this is consistent with (not proof of) the idea that different topics occupy "
            "separable regions of the routing space, which is the precondition modular "
            "continual-learning approaches (freeze old experts, add new ones) would need to "
            "hold. Does NOT establish representational orthogonality inside experts, and does "
            "NOT establish this generalises beyond OLMoE-1B-7B-0924 on this probe set."
        )
    elif z > 2:
        lines.append(
            "Domain routing signatures show MORE overlap than chance, not less — this would "
            "argue against a simple 'freeze old experts, add new ones' continual-learning "
            "strategy on this substrate: the same experts are shared across nominally "
            "different domains more than random routing would produce."
        )
    else:
        lines.append(
            "Observed orthogonality is not distinguishable from the null. No orthogonality "
            "claim is supported either way on this data."
        )

    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
