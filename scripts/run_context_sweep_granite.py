"""Context-rot capture sweep on the SECOND model (Granite-3.0-3B-A800M).

Everything about the context-rot mechanism so far -- the accuracy decline, the
needle-affine specialist loss, the length-independent link to correctness --
has only ever been measured on OLMoE. `docs/SECOND_MODEL.md` replicated the
BASE specialization findings on Granite; this script captures what is needed
to ask whether the MECHANISM replicates too.

Same design and output format as `scripts/run_context_sweep.py` (parquet
shards + accuracy.jsonl + manifest), pointed at
`probes/probe_set_context_granite.yaml` (same hard design, candidate pool
re-verified single-token under the Granite tokenizer -- 7/8 of the OLMoE pool
is multi-token under Granite, which would silently break forced-choice
scoring). Model loading and router-logit hooks are reused verbatim from
`scripts/run_second_model_check.py` (GraniteMoeTopKRouter returns raw logits
as the third element of its output tuple; verified there against the
transformers source, with a sanity forward before any capture).

Routing convention: Granite does topk-then-softmax; selecting top-k on raw
logits is identical either way (verified in run_second_model_check), and
`route_from_logits(..., norm_topk_prob=True)` reproduces its renormalised
gate weights.

Usage:
    HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 \
      .venv/bin/python scripts/run_context_sweep_granite.py
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
sys.path.insert(0, str(Path(__file__).parent))

from expertatlas.capture import route_from_logits
from expertatlas.context_metrics import router_entropy_bits, token_span_from_chars

import run_context_sweep as base
from run_second_model_check import _RouterLogitsHook, load_granite

REPO_ROOT = Path(__file__).parent.parent
PROBE_SET_PATH = REPO_ROOT / "probes" / "probe_set_context_granite.yaml"
OUT_DIR = REPO_ROOT / "data" / "context_traces_granite"


def process_prompt(loaded, prompt: dict, candidate_ids: list[int]):
    """Granite version of run_context_sweep.process_prompt: hook-captured
    router logits instead of output_router_logits, otherwise identical."""
    import numpy as np
    import pyarrow as pa

    tok = loaded.tokenizer
    enc = tok(prompt["text"], return_tensors="pt", return_offsets_mapping=True)
    offsets = [tuple(x) for x in enc["offset_mapping"][0].tolist()]
    inputs = {k: v for k, v in enc.items() if k != "offset_mapping"}
    input_ids = inputs["input_ids"][0]
    n_tokens = int(input_ids.shape[0])

    hook = _RouterLogitsHook(loaded.model)
    try:
        with torch.no_grad():
            try:
                out = loaded.model(**inputs, logits_to_keep=1)
            except TypeError:
                out = loaded.model(**inputs)
        router_logits = list(hook.captured)
    finally:
        hook.remove()
    last_logits = out.logits[0, -1, :].float()

    if len(router_logits) != loaded.shape.n_layers:
        raise RuntimeError(
            f"prompt {prompt['prompt_id']}: got {len(router_logits)} router layers, "
            f"expected {loaded.shape.n_layers} -- a layer was silently skipped"
        )

    cols = {k: [] for k in ("prompt_id", "token_pos", "token_id", "layer",
                            "expert_ids", "gate_weights", "topk_mass", "router_entropy")}
    for layer, logits in enumerate(router_logits):
        lg = logits.reshape(-1, loaded.shape.n_experts)
        ids, weights, mass = route_from_logits(
            lg, loaded.shape.top_k, loaded.shape.norm_topk_prob
        )
        ent = router_entropy_bits(lg.float().cpu().numpy())
        ids_l = ids.cpu().tolist()
        w_l = weights.float().cpu().tolist()
        m_l = mass.float().cpu().tolist()
        for t in range(n_tokens):
            cols["prompt_id"].append(int(prompt["prompt_id"]))
            cols["token_pos"].append(t)
            cols["token_id"].append(int(input_ids[t]))
            cols["layer"].append(layer)
            cols["expert_ids"].append(ids_l[t])
            cols["gate_weights"].append([float(x) for x in w_l[t]])
            cols["topk_mass"].append(float(m_l[t]))
            cols["router_entropy"].append(float(ent[t]))

    table = pa.table(
        {
            "prompt_id": pa.array(cols["prompt_id"], type=pa.int32()),
            "token_pos": pa.array(cols["token_pos"], type=pa.int32()),
            "token_id": pa.array(cols["token_id"], type=pa.int32()),
            "layer": pa.array(cols["layer"], type=pa.int16()),
            "expert_ids": pa.array(cols["expert_ids"], type=pa.list_(pa.int16())),
            "gate_weights": pa.array(cols["gate_weights"], type=pa.list_(pa.float32())),
            "topk_mass": pa.array(cols["topk_mass"], type=pa.float32()),
            "router_entropy": pa.array(cols["router_entropy"], type=pa.float32()),
        },
        schema=base._extended_schema(),
    )

    answer_id = tok(" " + prompt["answer_word"], add_special_tokens=False)["input_ids"][0]
    acc = base.score_answer(last_logits, candidate_ids, answer_id)
    q_span = token_span_from_chars(offsets, prompt["question_char_span"])
    n_span = token_span_from_chars(offsets, prompt["needle_char_span"])
    acc.update({
        "prompt_id": int(prompt["prompt_id"]),
        "bucket": int(prompt["bucket"]),
        "n_tokens": n_tokens,
        "haystack": prompt["haystack"],
        "n_distractors": int(prompt["n_distractors"]),
        "replicate": int(prompt["replicate"]),
        "entity": prompt["entity"],
        "answer_word": prompt["answer_word"],
        "question_token_span": list(q_span),
        "needle_token_span": list(n_span),
    })
    return table, acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=str, default=str(OUT_DIR))
    ap.add_argument("--probe-set", type=str, default=str(PROBE_SET_PATH))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pyarrow.parquet as pq

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ps = yaml.safe_load(Path(args.probe_set).read_text())
    prompts = sorted(ps["prompts"], key=lambda p: p["prompt_id"])
    if args.limit:
        prompts = prompts[: args.limit]
    print(f"probe set {ps['probe_set_id']}: {len(prompts)} prompts, "
          f"{sum(p['n_tokens'] for p in prompts)} tokens", flush=True)

    manifest = base._read_manifest(out_dir)
    done = set(manifest["done_prompt_ids"])
    todo = [p for p in prompts if p["prompt_id"] not in done]
    print(f"{len(done)} already done, {len(todo)} to run", flush=True)

    t0 = time.time()
    loaded = load_granite()
    print(f"loaded {loaded.model_id} in {time.time() - t0:.1f}s "
          f"n_layers={loaded.shape.n_layers} n_experts={loaded.shape.n_experts} "
          f"top_k={loaded.shape.top_k}", flush=True)

    torch.manual_seed(args.seed)
    candidate_ids = [
        loaded.tokenizer(" " + w, add_special_tokens=False)["input_ids"][0]
        for w in ps["candidate_words"]
    ]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise RuntimeError("candidate words collide under the tokenizer")

    acc_path = out_dir / "accuracy.jsonl"
    run_t0 = time.time()
    for i, prompt in enumerate(todo):
        t1 = time.time()
        table, acc = process_prompt(loaded, prompt, candidate_ids)
        pq.write_table(table, out_dir / f"trace_{prompt['prompt_id']:06d}.parquet")
        with acc_path.open("a") as fh:
            fh.write(json.dumps(acc) + "\n")
        done.add(prompt["prompt_id"])
        base._write_manifest(out_dir, {"done_prompt_ids": sorted(done)})
        dt = time.time() - t1
        eta = (time.time() - run_t0) / (i + 1) * (len(todo) - i - 1) / 60.0
        print(f"[{i + 1}/{len(todo)}] pid={prompt['prompt_id']:>4} "
              f"bucket={prompt['bucket']:>5} n_tok={acc['n_tokens']:>5} "
              f"{dt:6.1f}s  fc={int(acc['forced_choice_correct'])} "
              f"p={acc['forced_choice_prob']:.3f}  ETA {eta:6.1f} min", flush=True)

    (out_dir / "meta.json").write_text(json.dumps({
        "model_id": loaded.model_id,
        "probe_set_id": ps["probe_set_id"],
        "n_layers": loaded.shape.n_layers,
        "n_experts": loaded.shape.n_experts,
        "top_k": loaded.shape.top_k,
        "norm_topk_prob": loaded.shape.norm_topk_prob,
        "extra_columns": ["router_entropy"],
        "note": "Captured by scripts/run_context_sweep_granite.py (hook-based "
                "router logits; see module docstring).",
    }, indent=2))
    print(f"done: {len(done)} prompts, {(time.time() - run_t0) / 3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
