"""3-D layout of experts from their lift vectors (WS-D input).

Position is computed from **lift**, never from raw usage. Two experts sit close
together when they respond to the same domains relative to base rate — which is
the only thing the atlas claims to show.

The layout is presentational. Cluster *membership* comes from co-activation
communities (coactivation.py), which have a degree-preserving null behind them.
A UMAP blob is not evidence of anything on its own, and PLAN.md §9b requires
saying so wherever the layout is displayed.
"""

from __future__ import annotations

import numpy as np

SEED = 0


def _pca3(x: np.ndarray) -> np.ndarray:
    """Deterministic PCA to 3 components. Always available, no dependencies."""
    xc = x - x.mean(axis=0, keepdims=True)
    # SVD sign is arbitrary; pin it so layouts are reproducible across runs.
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    k = min(3, vt.shape[0])
    comps = vt[:k]
    for i in range(k):
        if comps[i][np.argmax(np.abs(comps[i]))] < 0:
            comps[i] = -comps[i]
    out = xc @ comps.T
    if out.shape[1] < 3:
        out = np.pad(out, ((0, 0), (0, 3 - out.shape[1])))
    return out


def compute_layout(
    lift: np.ndarray,
    method: str = "auto",
    seed: int = SEED,
    jitter: float = 1e-3,
) -> tuple[np.ndarray, str]:
    """Map (n_experts, n_domains) lift to (n_experts, 3) coordinates.

    Args:
        method: "umap", "pca", or "auto" (umap when installed, else pca).
        jitter: tiny deterministic offset so co-located points remain pickable.

    Returns:
        (coords, method_used) — coords scaled to roughly [-10, 10].
    """
    x = np.asarray(lift, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"lift must be 2-D, got {x.shape}")
    if x.shape[0] < 3:
        raise ValueError(f"need >= 3 experts to lay out, got {x.shape[0]}")

    used = "pca"
    coords = None
    if method in ("umap", "auto"):
        try:
            import umap  # noqa: PLC0415

            n_neighbors = int(min(15, max(2, x.shape[0] - 1)))
            reducer = umap.UMAP(
                n_components=3, n_neighbors=n_neighbors, min_dist=0.1,
                metric="cosine", random_state=seed,
            )
            coords = reducer.fit_transform(x)
            used = "umap"
        except Exception:
            if method == "umap":
                raise
            coords = None

    if coords is None:
        coords = _pca3(x)

    coords = np.asarray(coords, dtype=np.float64)
    span = np.ptp(coords, axis=0)
    scale = 10.0 / max(float(span.max()), 1e-9)
    coords = (coords - coords.mean(axis=0, keepdims=True)) * scale

    if jitter > 0:
        rng = np.random.default_rng(seed)
        coords = coords + rng.normal(scale=jitter, size=coords.shape)
    return coords, used


def lift_matrix(experts: list[dict], domains: list[str] | None = None):
    """Extract a dense lift matrix from atlas.json expert records."""
    if domains is None:
        keys: list[str] = []
        for e in experts:
            for k in e.get("lift", {}):
                if k not in keys:
                    keys.append(k)
        domains = sorted(keys)
    m = np.array([[e.get("lift", {}).get(d, 0.0) for d in domains] for e in experts],
                 dtype=np.float64)
    return m, domains
