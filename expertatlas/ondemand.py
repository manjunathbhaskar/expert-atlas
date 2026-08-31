"""On-demand expert runtime: run an MoE with expert FFNs materialized lazily.

The local-execution thread of this project (IDEAS_LOG §1) in its validated
form: `docs/KEEP_TOPK_*.md` ruled out *dropping* experts by a static list, and
`docs/DYNAMIC_K_RELATIVE.md` showed per-token adaptive selection is the shape
that works. The remaining engineering claim was never tested: if only a few
experts are needed at any moment, the rest do not need to be RESIDENT — they
can live on disk and be paged in per use. That claim is about memory, not
quality: the computation is exactly the dense model's (same weights, same
ops), so quality is verified by loss equality, and the honest question is the
measured RSS / wall-clock trade-off.

Mechanism
---------
* The model skeleton is built on the `meta` device; every non-expert tensor
  (embeddings, attention, layernorms, routers, lm_head) is loaded resident.
* Every expert FFN (`model.layers.L.mlp.experts.E`) is replaced by
  `OnDemandExpert`, which holds NO parameters. On forward it asks an
  `ExpertLRU` for its three weight matrices; on a cache miss the LRU reads
  them from the original safetensors shards via `safetensors.safe_open`
  (zero-copy mmap into the OS page cache, then a copy into the cache tensor).
* The LRU capacity is counted in experts. Capacity >= 1024 degenerates to
  "everything resident after first touch" (the dense memory footprint,
  reached lazily); capacity 0 refetches on every use.

Honest measurement notes (also printed by the benchmark script)
---------------------------------------------------------------
* RSS is the process's resident set; the mmap'd shard pages the OS caches on
  our behalf show up as page cache, not RSS. On a machine whose RAM exceeds
  the model size the OS will happily cache all shards, so wall-clock numbers
  here are a WARM-page-cache bound (optimistic vs. a machine that genuinely
  cannot hold the model, where misses become real disk reads). RSS numbers
  are not affected by that caveat: they measure what the process itself must
  hold, which is exactly the "run a model bigger than your memory budget"
  claim.
* BF16 CPU only, batch 1, teacher-forced. No claim about GPUs or generation.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

EXPERT_KEY = "model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
PROJS = ("gate_proj", "up_proj", "down_proj")


class ExpertStore:
    """Reads expert weight matrices straight from the checkpoint's safetensors
    shards (mmap; no copy of the checkpoint is ever made)."""

    def __init__(self, snapshot_dir: str | Path):
        from safetensors import safe_open

        self.snapshot_dir = Path(snapshot_dir)
        index = json.loads((self.snapshot_dir / "model.safetensors.index.json").read_text())
        self._weight_map: dict[str, str] = index["weight_map"]
        self._handles: dict[str, object] = {}
        self._safe_open = safe_open

    def _handle(self, shard: str):
        h = self._handles.get(shard)
        if h is None:
            h = self._safe_open(str(self.snapshot_dir / shard), framework="pt", device="cpu")
            self._handles[shard] = h
        return h

    def fetch(self, layer: int, expert: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = []
        for proj in PROJS:
            name = EXPERT_KEY.format(layer=layer, expert=expert, proj=proj)
            out.append(self._handle(self._weight_map[name]).get_tensor(name))
        return tuple(out)

    def non_expert_state_dict(self) -> dict[str, torch.Tensor]:
        sd = {}
        for name, shard in self._weight_map.items():
            if ".mlp.experts." not in name:
                sd[name] = self._handle(shard).get_tensor(name)
        return sd


@dataclass
class LRUStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    bytes_fetched: int = 0

    def to_dict(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "bytes_fetched": self.bytes_fetched,
            "hit_rate": (self.hits / total) if total else float("nan"),
        }


class ExpertLRU:
    """LRU over (layer, expert) -> weight tuple. Capacity in experts."""

    def __init__(self, store: ExpertStore, capacity: int):
        self.store = store
        self.capacity = int(capacity)
        self._cache: OrderedDict[tuple[int, int], tuple] = OrderedDict()
        self.stats = LRUStats()

    def get(self, layer: int, expert: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (layer, expert)
        w = self._cache.get(key)
        if w is not None:
            self._cache.move_to_end(key)
            self.stats.hits += 1
            return w
        self.stats.misses += 1
        w = self.store.fetch(layer, expert)
        self.stats.bytes_fetched += sum(t.numel() * t.element_size() for t in w)
        if self.capacity > 0:
            self._cache[key] = w
            while len(self._cache) > self.capacity:
                self._cache.popitem(last=False)
                self.stats.evictions += 1
        return w

    def resident_experts(self) -> int:
        return len(self._cache)

    def resident_bytes(self) -> int:
        return sum(t.numel() * t.element_size()
                   for ws in self._cache.values() for t in ws)


class PerLayerLRU:
    """LRU with a SEPARATE budget per layer, same interface as `ExpertLRU`.

    Why: a forward pass touches layers in a fixed order 0..L-1, once per
    token batch. Under a single global LRU, layer l's experts are the oldest
    entries by the time the next forward returns to layer l, so any capacity
    below "everything" evicts exactly the entries about to be reused --
    measured hit rate 0.000 at capacities 64-512 (docs/ONDEMAND.md). Giving
    each layer its own budget removes that cross-layer eviction: layer l's
    experts can only be evicted by other layer-l experts, so the top-k
    stationarity WITHIN a layer is what determines the hit rate.

    `capacity` is the TOTAL expert budget, split evenly across layers
    (floor); remainder capacity is left unused so the total is never
    exceeded. Per-layer capacity 0 means refetch-on-every-use.
    """

    def __init__(self, store: ExpertStore, capacity: int, n_layers: int):
        self.store = store
        self.capacity = int(capacity)
        self.n_layers = int(n_layers)
        self.per_layer_capacity = self.capacity // self.n_layers
        self._caches: dict[int, OrderedDict[int, tuple]] = {
            l: OrderedDict() for l in range(self.n_layers)
        }
        self.stats = LRUStats()

    def get(self, layer: int, expert: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache = self._caches[layer]
        w = cache.get(expert)
        if w is not None:
            cache.move_to_end(expert)
            self.stats.hits += 1
            return w
        self.stats.misses += 1
        w = self.store.fetch(layer, expert)
        self.stats.bytes_fetched += sum(t.numel() * t.element_size() for t in w)
        if self.per_layer_capacity > 0:
            cache[expert] = w
            while len(cache) > self.per_layer_capacity:
                cache.popitem(last=False)
                self.stats.evictions += 1
        return w

    def resident_experts(self) -> int:
        return sum(len(c) for c in self._caches.values())

    def resident_bytes(self) -> int:
        return sum(t.numel() * t.element_size()
                   for c in self._caches.values()
                   for ws in c.values() for t in ws)


class PinnedCache:
    """Fixed per-layer resident set chosen by measured usage; no eviction.

    Why: LRU (global or per-layer) measures hit rate 0.000 at every partial
    capacity on this workload — a full-sequence forward touches nearly all 64
    experts of a layer in ascending index order, the textbook worst case for
    LRU (sequential scan evicts each entry right before its reuse). A cache
    with no reuse-distance assumption sidesteps that: pin the top-N experts
    per layer by measured selection counts (`data/utilization.json`), serve
    those from RAM, and fetch-and-discard everything else. The achievable
    hit rate is then exactly the pinned set's measured selection coverage
    (load balancing keeps usage near-uniform, so coverage, not recency, is
    the honest ceiling at partial capacity).

    `capacity` is the TOTAL pinned budget, split evenly across layers.
    `counts` is an (n_layers, n_experts) selection-count array.
    """

    def __init__(self, store: ExpertStore, capacity: int, counts):
        counts = np.asarray(counts)
        self.store = store
        self.capacity = int(capacity)
        self.n_layers = counts.shape[0]
        per_layer = self.capacity // self.n_layers
        self.per_layer_capacity = per_layer
        self.pinned_sets: list[set[int]] = [
            set(np.argsort(row)[::-1][:per_layer].tolist()) for row in counts
        ]
        self._cache: dict[tuple[int, int], tuple] = {}
        self.stats = LRUStats()

    def get(self, layer: int, expert: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (layer, expert)
        w = self._cache.get(key)
        if w is not None:
            self.stats.hits += 1
            return w
        self.stats.misses += 1
        w = self.store.fetch(layer, expert)
        self.stats.bytes_fetched += sum(t.numel() * t.element_size() for t in w)
        if expert in self.pinned_sets[layer]:
            # Clone out of the checkpoint mmap into anonymous memory: an
            # mmap-backed "pinned" tensor is still reclaimable page cache, so
            # under memory pressure a cache hit would silently hit disk anyway.
            w = tuple(t.clone() for t in w)
            self._cache[key] = w
        return w

    def resident_experts(self) -> int:
        return len(self._cache)

    def resident_bytes(self) -> int:
        return sum(t.numel() * t.element_size()
                   for ws in self._cache.values() for t in ws)


class OnDemandExperts(nn.Module):
    """Drop-in replacement for `OlmoeExperts` holding NO parameters.

    Same forward contract as transformers' `OlmoeExperts.forward(hidden_states,
    top_k_index, top_k_weights)` and the same per-expert math: the checkpoint
    stores separate `gate_proj`/`up_proj` matrices which transformers fuses
    into one `gate_up_proj` linear and then chunks; applying the two linears
    separately produces the identical rows, verified by the benchmark's
    loss-equality check against the dense model.
    """

    def __init__(self, lru: ExpertLRU, layer: int, num_experts: int):
        super().__init__()
        self.lru = lru
        self.layer = layer
        self.num_experts = num_experts

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor,
                top_k_weights: torch.Tensor) -> torch.Tensor:
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            # Zero-weight entries (dynamic-k padding) must not trigger fetches.
            expert_mask = expert_mask * (top_k_weights != 0).long().unsqueeze(-1)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            e = int(expert_idx[0])
            top_k_pos, token_idx = torch.where(expert_mask[e])
            x = hidden_states[token_idx]
            gate_w, up_w, down_w = self.lru.get(self.layer, e)
            h = F.silu(F.linear(x, gate_w)) * F.linear(x, up_w)
            h = F.linear(h, down_w) * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, h.to(final.dtype))
        return final


@dataclass
class OnDemandModel:
    model: object
    tokenizer: object
    lru: ExpertLRU
    store: ExpertStore
    n_layers: int = 0
    n_experts: int = 0
    resident_param_bytes: int = field(default=0)


def snapshot_dir_for(model_id: str, hf_cache: str | Path) -> Path:
    base = Path(hf_cache) / f"models--{model_id.replace('/', '--')}" / "snapshots"
    snaps = sorted(p for p in base.iterdir() if p.is_dir())
    if not snaps:
        raise FileNotFoundError(f"no snapshot under {base}")
    return snaps[-1]


def load_ondemand(model_id: str, hf_cache: str | Path, cache_experts: int,
                  dtype: torch.dtype = torch.bfloat16,
                  cache_policy: str = "global",
                  usage_counts=None) -> OnDemandModel:
    """Build the model with all experts on disk and only `cache_experts` ever
    resident at once. `cache_policy`: "global" (single LRU, the measured
    thrash case below full capacity), "per_layer" (budget split across
    layers; see PerLayerLRU), or "pinned" (fixed usage-ranked resident set;
    requires `usage_counts`, an (n_layers, n_experts) selection-count array;
    see PinnedCache)."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    snap = snapshot_dir_for(model_id, hf_cache)
    config = AutoConfig.from_pretrained(snap)
    tokenizer = AutoTokenizer.from_pretrained(snap)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
    model.eval()

    store = ExpertStore(snap)
    n_layers = config.num_hidden_layers
    n_experts = config.num_experts
    if cache_policy == "global":
        lru = ExpertLRU(store, cache_experts)
    elif cache_policy == "per_layer":
        lru = PerLayerLRU(store, cache_experts, n_layers)
    elif cache_policy == "pinned":
        if usage_counts is None:
            raise ValueError("cache_policy='pinned' requires usage_counts")
        lru = PinnedCache(store, cache_experts, usage_counts)
    else:
        raise ValueError(f"unknown cache_policy {cache_policy!r}")

    # Swap the whole experts module for the parameter-free proxy BEFORE
    # loading weights, so no expert tensor is ever materialized resident.
    for l in range(n_layers):
        model.model.layers[l].mlp.experts = OnDemandExperts(lru, l, n_experts)

    sd = store.non_expert_state_dict()
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    unexpected = [k for k in unexpected if ".mlp.experts." not in k]
    leftover_meta = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if unexpected or leftover_meta:
        raise RuntimeError(f"bad selective load: unexpected={unexpected[:5]} "
                           f"meta={leftover_meta[:5]}")

    resident = sum(p.numel() * p.element_size() for p in model.parameters())
    return OnDemandModel(model=model, tokenizer=tokenizer, lru=lru, store=store,
                         n_layers=n_layers, n_experts=n_experts,
                         resident_param_bytes=resident)
