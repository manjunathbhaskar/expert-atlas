"""Phase 3 step 1: real capture run over the full probe set (PLAN.md §5 Phase 3).

Loads probes/probe_set_v1.yaml (480 prompts, real prompt_ids), loads OLMoE
via expertatlas.capture.load_model, and streams RoutingTrace to data/traces/
via capture_to_dir -- the same resumable/streaming engine WS-A built and
tested, just pointed at the real probe set instead of a fake model.

Usage:
    python scripts/run_phase3_capture.py [--limit N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from expertatlas.capture import capture_to_dir, load_model

MODEL_ID = "allenai/OLMoE-1B-7B-0924"
PROBE_SET_PATH = Path(__file__).parent.parent / "probes" / "probe_set_v1.yaml"
OUT_DIR = Path(__file__).parent.parent / "data" / "traces"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="cap number of prompts (debug)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    probe_set = yaml.safe_load(PROBE_SET_PATH.read_text())
    prompts = [(p["prompt_id"], p["text"]) for p in probe_set["prompts"]]
    print(f"loaded {len(prompts)} prompts from {PROBE_SET_PATH.name} "
          f"(probe_set_id={probe_set['probe_set_id']}, n_cells={probe_set['n_cells']})")

    print(f"loading {MODEL_ID} ...")
    t0 = time.time()
    loaded = load_model(MODEL_ID, device="cpu", dtype="bfloat16")
    print(f"loaded in {time.time() - t0:.1f}s. capture_method={loaded.capture_method} "
          f"n_layers={loaded.shape.n_layers} n_experts={loaded.shape.n_experts} "
          f"top_k={loaded.shape.top_k}")

    t0 = time.time()
    result = capture_to_dir(
        loaded,
        prompts,
        out_dir=OUT_DIR,
        device="cpu",
        dtype="bfloat16",
        seed=args.seed,
        limit=args.limit,
        resume=True,
    )
    elapsed = time.time() - t0
    print(f"capture done in {elapsed:.1f}s ({elapsed / max(result['n_prompts_written_this_run'], 1):.2f}s/prompt)")
    print(result)


if __name__ == "__main__":
    main()
