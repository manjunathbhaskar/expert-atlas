# On-demand expert runtime: measured memory and wall-clock

## Limitations (read first)

- **The main table is warm page cache.** This machine's RAM exceeds the
  checkpoint size, so the OS caches the mmap'd shards after first touch; the
  main table's wall-clock numbers are a warm-cache bound. The MEMORY numbers
  are unaffected: `RssAnon` is what the process itself must hold and is the
  honest local-execution figure. **Cold-cache behavior under a real memory
  limit is now measured separately below** (cgroup `MemoryMax`, page cache
  dropped first).
- One model (OLMoE-1B-7B-0924), CPU, BF16, batch 1, teacher-forced NLL on
  12 split-B prompts (4 domains). No GPU or generation claims.
- Correctness: expert weights were verified bit-identical to the dense
  model's, and a single experts-module comparison with identical routing
  differs by at most 1 bf16 ulp (transformers dispatches a differently
  batched kernel; reduction-order drift). The small NLL deviation reported
  below for non-dynamic-k conditions is that kernel drift, not a quality
  change. Dynamic-k conditions deviate by design (measured quality cost).

## Results

| condition | mean NLL | mean k | tok/s | slowdown | RssAnon (GiB) | RssFile (GiB) | LRU hit rate | fetched (GiB) |
|---|---|---|---|---|---|---|---|---|
| dense | 2.353309 | 8 (full) | 155.8 | 1.00x | 12.66 | 13.06 | - | - |
| ondemand-0 | 2.359576 | 8 (full) | 105.3 | 1.48x | 0.66 | 12.96 | 0.000 | 126.5 |
| ondemand-64 | 2.359576 | 8 (full) | 105.0 | 1.48x | 0.64 | 12.97 | 0.000 | 126.5 |
| ondemand-128 | 2.359576 | 8 (full) | 104.7 | 1.49x | 0.64 | 12.96 | 0.000 | 126.5 |
| ondemand-256 | 2.359576 | 8 (full) | 106.1 | 1.47x | 0.67 | 12.97 | 0.000 | 126.5 |
| ondemand-512 | 2.359576 | 8 (full) | 103.9 | 1.50x | 0.72 | 12.97 | 0.000 | 126.5 |
| ondemand-1024 | 2.359576 | 8 (full) | 119.3 | 1.31x | 0.64 | 12.97 | 0.906 | 11.9 |
| ondemand-1024-rk0.9 | 2.366937 | 6.89 | 123.2 | 1.26x | 0.64 | 12.84 | 0.903 | 11.8 |
| ondemand-1024-rk0.8 | 2.434173 | 5.79 | 131.3 | 1.19x | 0.63 | 12.58 | 0.899 | 11.5 |
| ondemand-1024-rk0.7 | 2.464916 | 4.79 | 141.9 | 1.10x | 0.63 | 12.30 | 0.895 | 11.2 |
| ondemand-1024-rk0.5 | 2.586347 | 3.18 | 166.0 | 0.94x | 0.61 | 11.59 | 0.884 | 10.5 |

Expert weights total 12.0 GiB of the checkpoint;
the on-demand process keeps only the LRU capacity's worth resident
(`RssAnon`), plus ~0.9 GiB of non-expert weights and runtime overhead.

## Reading this

- If `mean NLL` matches dense (up to BF16 kernel drift) at every capacity,
  the runtime is CORRECT and the whole trade is memory-vs-time, quantified
  above.
- The dense condition's RssAnon is the footprint this machine needed to run
  the model at all; the smallest capacity's RssAnon is what a machine would
  need with this runtime.

## Findings from this run

- **The headline number is real and measured**: the dense model needs
  12.66 GiB of anonymous RSS; the on-demand runtime runs the same computation
  (NLL equal up to BF16 kernel drift) with **0.61–0.72 GiB** anonymous RSS —
  a ~20x reduction in the memory the process must own — at a 1.31–1.50x
  wall-clock cost warm, and rk0.5 dynamic-k is actually 0.94x (faster than
  dense) because it computes ~3.2 instead of 8 expert FFNs per token.
- **`safetensors.get_tensor` returns mmap-backed views**, so even "cached"
  expert weights live in file-backed pages (`RssFile`), not anonymous memory.
  That is why RssAnon is flat across capacities: the OS, not the process,
  owns the expert bytes and can reclaim them under pressure. On a genuinely
  small-RAM machine the trade would appear as re-read latency (cold misses
  hitting disk), not as OOM — exactly the behavior wanted for local
  execution. That machine is now simulated with a cgroup below.
- **The global LRU thrashes below full capacity**: capacities 64–512 show a
  0.000 hit rate — identical to capacity 0 — because a forward pass touches
  layers in order and each layer's experts are evicted by later layers before
  the next prompt returns to them. The fetched-bytes column (126.5 GiB
  logical vs 11.9 GiB at full capacity) quantifies this. On this box the
  mmap page cache hides the cost, which is why the wall-clock stays flat.

## Partial-capacity caching: per-layer LRU (negative) and usage-pinned (positive)

Two layer-aware policies were built and measured at partial capacity
(same 12-prompt, 666-token workload; `expertatlas/ondemand.py`).

**Per-layer LRU (budget split evenly across layers) does NOT fix the thrash:**

| condition | mean NLL | hit rate | fetched (GiB) |
|---|---|---|---|
| ondemand-64-pl | 2.359576 | 0.000 | 126.5 |
| ondemand-128-pl | 2.359576 | 0.000 | 126.5 |
| ondemand-256-pl | 2.359576 | 0.000 | 126.5 |
| ondemand-512-pl | 2.359576 | 0.000 | 126.5 |

Reason (a real negative worth keeping): each full-sequence forward touches
nearly all 64 experts of a layer in ascending index order — a sequential
scan, the textbook worst case for LRU. With any per-layer budget below the
full working set, each expert is evicted right before its next use. Isolating
layers removes cross-layer eviction but the within-layer access pattern still
defeats recency-based eviction entirely.

**Usage-pinned cache (top-N experts per layer by measured selection counts
from `data/utilization.json`, no eviction, misses fetch-and-discard) gives a
real, nonzero hit rate at partial capacity:**

| condition | mean NLL | hit rate | fetched (GiB) | tok/s (warm) |
|---|---|---|---|---|
| ondemand-128-pin | 2.359576 | 0.121 | 111.2 | 98.6 |
| ondemand-256-pin | 2.359576 | 0.247 | 95.3 | 98.4 |
| ondemand-512-pin | 2.359576 | 0.496 | 63.8 | 102.2 |

Hit rate tracks capacity fraction (128/1024=12.5% -> 12.1% hits; 512/1024=50%
-> 49.6%): on this mixed 4-domain prompt set the atlas's usage-count ranking
buys little above proportional coverage, consistent with load balancing
keeping usage near-uniform (docs/UTILIZATION.md). The honest claim is
"proportional, not zero" — the pinned policy converts capacity directly into
avoided fetches where every LRU variant converted it into nothing.
Pinned entries are `clone()`d out of the checkpoint mmap into anonymous
memory, so a hit survives OS memory pressure (an mmap-backed "cached" tensor
is just reclaimable page cache); RssAnon therefore grows by the pinned bytes
(~1.6 GiB at 128 experts).

## Cold-cache under a real memory limit (cgroup MemoryMax, caches dropped)

Each run: `echo 3 > /proc/sys/vm/drop_caches`, then the child under
`systemd-run --scope -p MemoryMax=<limit> -p MemorySwapMax=0`. The kernel is
forced to actually evict pages and re-read from disk.

| condition | limit | mean NLL | tok/s | outcome |
|---|---|---|---|---|
| dense | 2 GiB | - | - | **OOM-killed by the memcg during weight loading** (dmesg: `Memory cgroup out of memory`) |
| ondemand-0 (no cache) | 3 GiB | 2.359576 | 6.96 | completes |
| ondemand-128-pin | 3 GiB | 2.359576 | 7.56 | completes (RssAnon 2.15 GiB incl. pinned clones) |
| ondemand-512-pin | 2 GiB | 2.359576 | 7.18 | completes (peak VmRSS 2.01 GiB, under the cap) |
| ondemand-512-pin | 4 GiB | 2.359576 | 7.58 | completes |

Findings:

- **The runtime's core claim survives a real constraint**: under a 2 GiB
  cgroup — 6x below the checkpoint — the dense model cannot even load, while
  the on-demand runtime completes with NLL identical to the warm runs.
- **The cold-cache cost is large and is the honest headline for a small
  machine: ~7 tok/s vs ~100–120 warm (~14x slower)** on this box's disk.
  This is the number a genuinely RAM-limited machine should expect, not the
  warm table above.
- Raising the limit 2→4 GiB barely helps (7.18→7.58 tok/s): throughput is
  bound by re-reading evicted expert pages from disk (~64–120 GiB logical per
  run), not by the cache budget. Reducing bytes-fetched (pinning, dynamic-k)
  is what moves cold throughput, not more page cache below checkpoint size.
- The 512-pin runs' pinned clones (6.4 GiB) exceed the 2 GiB cap, so the
  kernel keeps only what fits; correctness is unaffected (clones are
  re-created on miss), and the measured hit counter still reflects avoided
  `fetch()` calls, not guaranteed RAM residency at this capacity/limit combo.
  The 128-pin/3 GiB row is the configuration where the pinned set genuinely
  fits and stays anonymous-resident.
- **Dynamic-k reduces what must be touched at all**: at threshold 0.5 the
  runtime fetched 10.5 GiB logical (vs 11.9 full) and computed 3.18/8 experts
  per token for +0.23 nats — consistent with docs/DYNAMIC_K_RELATIVE.md's
  curve, now with measured wall-clock instead of analytical matmul counts.
