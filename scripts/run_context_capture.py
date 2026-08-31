"""WS1 capture: routing traces + needle-retrieval accuracy across context lengths.

Two things are recorded per prompt, in one forward/generate pass each:
  1. Router traces (same RoutingTrace contract as the main run) -> parquet.
  2. Whether the model actually retrieves the needle (greedy generation,
     checked for the expected answer string) -> accuracy.jsonl

Both are needed for the WS1 question: does retrieval accuracy degrade with
length, and if so, does any routing metric degrade in step? Recording
accuracy without routing (or vice versa) can't answer that.

Resumable: skips prompts already present in accuracy.jsonl, same spirit as
capture_to_dir's manifest.

Usage:
    python scripts/run_context_capture.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from expertatlas.capture import get_router_logits_for_prompt, load_model, prompt_rows_to_table

REPO_ROOT = Path(__file__).parent.parent
PROBE_PATH = REPO_ROOT / "probes" / "probe_set_context.yaml"
OUT_DIR = REPO_ROOT / "data" / "traces_context"
ACC_PATH = OUT_DIR / "accuracy.jsonl"
MODEL_ID = "allenai/OLMoE-1B-7B-0924"
MAX_NEW_TOKENS = 12


def already_done() -> set[int]:
    if not ACC_PATH.exists():
        return set()
    done = set()
    for line in ACC_PATH.read_text().splitlines():
        if line.strip():
            done.add(json.loads(line)["prompt_id"])
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    doc = yaml.safe_load(PROBE_PATH.read_text())
    prompts = doc["prompts"]
    expected = doc["expected_answer"]
    if args.limit:
        prompts = prompts[: args.limit]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = already_done()
    todo = [p for p in prompts if p["prompt_id"] not in done]
    print(f"{len(prompts)} prompts total, {len(done)} already done, {len(todo)} to run")
    if not todo:
        print("nothing to do")
        return

    print(f"loading {MODEL_ID} ...")
    t0 = time.time()
    loaded = load_model(MODEL_ID, device="cpu", dtype="bfloat16")
    print(f"loaded in {time.time() - t0:.1f}s")

    import pyarrow.parquet as pq

    for i, p in enumerate(todo):
        t_start = time.time()
        text = p["text"]

        # --- routing trace (prompt tokens only, one forward pass) ---
        router_logits = get_router_logits_for_prompt(loaded, text, "cpu")
        inputs = loaded.tokenizer(text, return_tensors="pt")
        n_tokens = int(inputs["input_ids"].shape[1])
        table = prompt_rows_to_table(
            p["prompt_id"], inputs["input_ids"][0], router_logits,
            loaded.shape.top_k, loaded.shape.norm_topk_prob,
        )
        pq.write_table(table, OUT_DIR / f"trace_{p['prompt_id']:06d}.parquet")

        # --- needle retrieval accuracy (greedy, deterministic) ---
        # load_model() sets config.output_router_logits=True, so EVERY forward
        # pass also computes the MoE load-balancing aux loss. That is fine for
        # a single full-sequence forward (the main capture run relies on it),
        # but it crashes inside generate()'s cached decoding: at single-token
        # steps load_balancing_loss_func indexes an attention mask slice that
        # is empty, giving "size of tensor a (16) must match tensor b (0)".
        # Routing for this prompt is already captured above from the full
        # forward pass, so the flag is simply switched off for generation and
        # restored afterwards -- no routing information is lost.
        prev_router_flag = loaded.model.config.output_router_logits
        loaded.model.config.output_router_logits = False
        try:
            with torch.no_grad():
                gen = loaded.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=loaded.tokenizer.eos_token_id,
                )
        finally:
            loaded.model.config.output_router_logits = prev_router_flag
        completion = loaded.tokenizer.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        # Normalised containment check -- the needle answer is a rare, exact
        # token string, so substring match after case/space normalisation is a
        # fair and unambiguous criterion here.
        norm = lambda s: s.upper().replace(" ", "").replace("\n", "")
        correct = norm(expected) in norm(completion)

        rec = {
            "prompt_id": p["prompt_id"],
            "length_bucket": p["length_bucket"],
            "condition": p["condition"],
            "replicate": p["replicate"],
            "split": p["split"],
            "n_tokens_actual": n_tokens,
            "completion": completion.strip()[:200],
            "correct": bool(correct),
        }
        with ACC_PATH.open("a") as f:
            f.write(json.dumps(rec) + "\n")

        print(f"[{i+1}/{len(todo)}] id={p['prompt_id']} bucket={p['length_bucket']} "
              f"cond={p['condition']} tokens={n_tokens} correct={correct} "
              f"({time.time() - t_start:.1f}s)")

    print("done")


if __name__ == "__main__":
    main()
