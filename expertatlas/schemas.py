"""Frozen data contracts (PLAN.md §3).

These are the ONLY interfaces workstreams talk through. WS-A writes
RoutingTrace parquet + meta.json; WS-C reads it and writes atlas.json;
WS-D reads *only* atlas.json and never touches traces.

Changing a field here after Phase 0 is a cross-team-breaking change — if a
workstream needs a change, it goes through docs/interface-requests.md, not
a silent edit.

One deliberate addition vs. the PLAN.md §3 table: `topk_mass` on
RoutingTraceRow. Section 6.1's own sanity tests (test_topk_mass_is_recorded)
require it — "routing confidence" (sum of the top-k gate weights before
any renormalisation) is itself a signal and the original table omitted the
column it depends on. Fixed here in Phase 0 rather than discovered mid-WS-C.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------
# Contract 1 — RoutingTrace (capture output)
# --------------------------------------------------------------------------

#: One row per (token, layer). This is the pyarrow schema the parquet writer
#: (WS-A) must produce and every downstream reader may rely on verbatim.
ROUTING_TRACE_SCHEMA = pa.schema(
    [
        pa.field("prompt_id", pa.int32(), nullable=False),
        pa.field("token_pos", pa.int32(), nullable=False),
        pa.field("token_id", pa.int32(), nullable=False),
        pa.field("layer", pa.int16(), nullable=False),
        pa.field("expert_ids", pa.list_(pa.int16()), nullable=False),
        pa.field("gate_weights", pa.list_(pa.float32()), nullable=False),
        # Addition vs. PLAN.md §3 table — see module docstring.
        pa.field("topk_mass", pa.float32(), nullable=False),
    ]
)


class RoutingTraceMeta(BaseModel):
    """Sidecar meta.json written alongside every trace parquet file."""

    model_config = ConfigDict(frozen=True)

    model_id: str
    model_revision: str
    dtype: str
    device: str
    top_k: int
    n_layers: int
    n_experts: int
    transformers_version: str
    torch_version: str
    seed: int
    git_commit: str
    timestamp: str  # ISO 8601, UTC
    host_fingerprint: str
    # true when produced by the forward-hook fallback rather than
    # output_router_logits=True — downstream stats should not care, but
    # it matters for debugging capture discrepancies across hosts (H5).
    capture_method: str = Field(pattern="^(output_router_logits|forward_hook)$")


# --------------------------------------------------------------------------
# Contract 2 — atlas.json (visualiser input, versioned)
# --------------------------------------------------------------------------

ATLAS_SCHEMA_VERSION = "1.0"


class ModelInfo(BaseModel):
    id: str
    revision: str
    n_layers: int
    n_experts_per_layer: int
    top_k: int


class ProbeSetInfo(BaseModel):
    id: str
    n_prompts: int
    factors: list[str]


class StatsInfo(BaseModel):
    n_tokens: int
    null_model: str
    n_permutations: int
    fdr_method: str
    q: float


class TopToken(BaseModel):
    token: str
    lift: float


class ExpertRecord(BaseModel):
    uid: str  # e.g. "L03E17"
    layer: int
    idx: int
    usage: float  # marginal P(expert); expect ~= 1/n_experts under load balancing
    lift: dict[str, float]  # domain -> log2 lift, ALL computed domains
    significant: list[str]  # domain names that survived BH-FDR
    max_lift: float
    specialisation: float  # normalised entropy of lift profile, 0=generalist 1=specialist
    top_tokens: list[TopToken]
    community: int
    xyz: tuple[float, float, float]


class CommunityRecord(BaseModel):
    id: int
    size: int
    label: str
    modularity_contrib: float


class CoactivationInfo(BaseModel):
    edges: list[tuple[str, str, float]]
    null_modularity_ci: tuple[float, float]


class Atlas(BaseModel):
    """Root object serialised to atlas.json. The visualiser (WS-D) reads
    only this — never raw traces."""

    schema_version: str = ATLAS_SCHEMA_VERSION
    model: ModelInfo
    probe_set: ProbeSetInfo
    stats: StatsInfo
    experts: list[ExpertRecord]
    communities: list[CommunityRecord]
    coactivation: CoactivationInfo


# --------------------------------------------------------------------------
# Small helpers shared across workstreams
# --------------------------------------------------------------------------


def expert_uid(layer: int, idx: int) -> str:
    """Canonical expert identifier, e.g. layer=3, idx=17 -> 'L03E17'."""
    return f"L{layer:02d}E{idx:02d}"


@dataclass(frozen=True)
class ModelShape:
    """Router/topology facts about the target model, resolved once in
    capture.py and threaded through everything downstream instead of being
    re-derived (and possibly re-guessed) in five different places."""

    n_layers: int
    n_experts: int
    top_k: int
    norm_topk_prob: bool = field(
        metadata={
            "why": (
                "OLMoE sets norm_topk_prob=False: the top-k gate weights "
                "sum to the top-k probability MASS, not 1.0. Mixtral-style "
                "models renormalise and DO sum to 1. Read this from the "
                "model config at capture time — never assume either way."
            )
        },
        default=False,
    )
