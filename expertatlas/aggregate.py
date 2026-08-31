"""Trace -> count matrix, with the equal-token-budget control.

This module is the bridge between WS-A's `RoutingTrace` and WS-C's statistics.
It exists because of a measured confound (docs/interface-requests.md):

    prompt length varies 1.73x across topics and 1.53x across languages

Every token is an independent routing observation, so pooling raw counts weights
long-prompt cells more heavily, and any expert responding to sequence position
would masquerade as a topic specialist. `subsample_cells` equalises the token
budget per factorial cell before any counting happens.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FACTORS = ("topic", "lang", "register", "format")


@dataclass(frozen=True)
class CountMatrix:
    """Expert x domain counts plus the provenance needed to reproduce them."""
    counts: np.ndarray           # (n_experts_total, n_domains) float64
    expert_uids: list[str]       # "L03E17", row order
    domain_labels: list[str]     # column order
    tokens_per_cell: int         # the enforced budget
    n_tokens_used: int


def cell_key(prompt: dict) -> tuple:
    return tuple(prompt[f] for f in FACTORS)


def subsample_cells(
    rows: list[dict],
    prompts_by_id: dict[int, dict],
    seed: int = 0,
    tokens_per_cell: int | None = None,
) -> tuple[list[dict], int]:
    """Equalise token count per factorial cell.

    Args:
        rows: routing records; each needs ``prompt_id`` and ``token_pos``.
        prompts_by_id: probe-set metadata keyed by ``prompt_id``.
        seed: fixed for reproducibility.
        tokens_per_cell: override the budget; default is the observed minimum.

    Returns:
        (kept_rows, tokens_per_cell)

    Subsampling is over *distinct tokens*, not rows — every layer of a kept token
    is kept, so the per-token routing pattern is never partially observed.
    """
    rng = np.random.default_rng(seed)

    # cell -> set of (prompt_id, token_pos)
    tokens_by_cell: dict[tuple, set] = {}
    for r in rows:
        key = cell_key(prompts_by_id[r["prompt_id"]])
        tokens_by_cell.setdefault(key, set()).add((r["prompt_id"], r["token_pos"]))

    if not tokens_by_cell:
        return [], 0

    budget = tokens_per_cell or min(len(v) for v in tokens_by_cell.values())

    keep: set = set()
    for key, toks in tokens_by_cell.items():
        ordered = sorted(toks)  # deterministic before sampling
        if len(ordered) <= budget:
            keep.update(ordered)
        else:
            idx = rng.choice(len(ordered), size=budget, replace=False)
            keep.update(ordered[i] for i in idx)

    kept = [r for r in rows if (r["prompt_id"], r["token_pos"]) in keep]
    return kept, budget


def aggregate_counts(
    rows: list[dict],
    prompts_by_id: dict[int, dict],
    n_layers: int,
    n_experts: int,
    domain_factor: str = "topic",
    seed: int = 0,
    weight_by_gate: bool = False,
) -> CountMatrix:
    """Build the expert x domain count matrix under an equal token budget.

    Args:
        domain_factor: which factor defines a "domain" column. Using one factor
            at a time is what the factorial design buys — see `marginal_counts`.
        weight_by_gate: count gate weight instead of 1 per selection. Off by
            default: selection is the routing decision, weight is confidence,
            and mixing them makes lift harder to interpret.
    """
    kept, budget = subsample_cells(rows, prompts_by_id, seed=seed)

    domains = sorted({p[domain_factor] for p in prompts_by_id.values()})
    d_index = {d: i for i, d in enumerate(domains)}

    counts = np.zeros((n_layers * n_experts, len(domains)), dtype=np.float64)
    for r in kept:
        col = d_index[prompts_by_id[r["prompt_id"]][domain_factor]]
        base = int(r["layer"]) * n_experts
        weights = r["gate_weights"] if weight_by_gate else None
        for j, e in enumerate(r["expert_ids"]):
            counts[base + int(e), col] += float(weights[j]) if weights is not None else 1.0

    uids = [f"L{l:02d}E{e:02d}" for l in range(n_layers) for e in range(n_experts)]
    n_tokens = len({(r["prompt_id"], r["token_pos"]) for r in kept})
    return CountMatrix(counts, uids, domains, budget, n_tokens)


def marginal_counts(
    rows: list[dict],
    prompts_by_id: dict[int, dict],
    n_layers: int,
    n_experts: int,
    seed: int = 0,
) -> dict[str, CountMatrix]:
    """One count matrix per factor — the payoff of the factorial design.

    Comparing lift on `topic` against lift on `lang` is what distinguishes a
    *syntax* expert from a *language* expert. A flat probe set cannot do this
    because the factors move together (probes/README.md).
    """
    return {
        f: aggregate_counts(rows, prompts_by_id, n_layers, n_experts,
                            domain_factor=f, seed=seed)
        for f in FACTORS
    }
