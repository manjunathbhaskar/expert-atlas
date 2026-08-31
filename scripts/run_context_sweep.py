"""Context-rot capture sweep.

Runs `probes/probe_set_context.yaml` through OLMoE and records, per prompt:

  * a routing trace extended with **`router_entropy`** — the entropy of the
    router's full 64-way softmax, per (token, layer), in bits;
  * the forced-choice **accuracy** of the needle answer, read off the
    final-position LM logits of the *same* forward pass.

Why this does not reuse `capture.py::capture_to_dir`
----------------------------------------------------
`capture.py` belongs to the specialisation-capture pipeline and is not edited here. Its writer
records `expert_ids / gate_weights / topk_mass` but discards the raw router
logits, so full-softmax entropy cannot be recovered from its output, and it
does not surface LM logits at all. This script therefore drives its own forward
pass while importing `load_model` and `route_from_logits` from `capture.py`, so
the top-k selection logic is literally the same code path as every other trace
in the repo and cannot silently diverge. An interface request to add
`router_entropy` to the frozen schema is filed in `docs/interface-requests.md`.

Cost model (measured on this machine, 3 real prompts at 126 / 1008 / 3831
tokens, before launching anything):

    seconds(T) = 61.38 - 0.0149*T + 1.323e-05*T^2

A fixed ~58 s/forward dominates below ~1k tokens (the 16 x 64 expert loop runs
regardless of token count); quadratic attention takes over above it. Projected
wall clock is reported in `docs/CONTEXT_ROT.md`.

Output (all new paths, nothing existing is touched):
    data/context_traces/trace_XXXXXX.parquet   extended routing trace
    data/context_traces/accuracy.jsonl         one record per prompt
    data/context_traces/manifest.json          resume state
    data/context_traces/meta.json              provenance

Resumable: re-running skips prompts already in the manifest. Prompts are
processed in prompt_id order, which is replicate-major, so an interrupted run
leaves a *balanced* design over the replicates that did finish rather than a
partial length sweep.

Usage:
    python scripts/run_context_sweep.py [--limit N] [--buckets 128,256]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from expertatlas.capture import build_meta, load_model, route_from_logits
from expertatlas.context_metrics import router_entropy_bits, token_span_from_chars
from expertatlas.schemas import ROUTING_TRACE_SCHEMA

MODEL_ID = "allenai/OLMoE-1B-7B-0924"
REPO_ROOT = Path(__file__).parent.parent
PROBE_SET_PATH = REPO_ROOT / "probes" / "probe_set_context.yaml"
OUT_DIR = REPO_ROOT / "data" / "context_traces"
MANIFEST = "manifest.json"


def _extended_schema():
    """ROUTING_TRACE_SCHEMA + router_entropy.

    A strict superset, so anything that can read a normal trace can read these
    shards and ignore the extra column.
    """
    import pyarrow as pa

    return pa.schema(list(ROUTING_TRACE_SCHEMA) + [
        pa.field("router_entropy", pa.float32(), nullable=False),
    ])


def _read_manifest(out_dir: Path) -> dict:
    p = out_dir / MANIFEST
    return json.loads(p.read_text()) if p.exists() else {"done_prompt_ids": []}


def _write_manifest(out_dir: Path, manifest: dict) -> None:
    tmp = out_dir / f".{MANIFEST}.tmp"
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(out_dir / MANIFEST)


def _forward(model, inputs, want_full_logits: bool = False):
    """One forward pass returning (router_logits list, last-position logits).

    `logits_to_keep=1` avoids materialising a (n_tokens x 50304) logit tensor —
    at 3840 tokens that is ~400 MB of pure waste, since only the final position
    is scored. Falls back cleanly if the installed transformers lacks the kwarg.
    """
    with torch.no_grad():
        try:
            out = model(**inputs, output_router_logits=True, logits_to_keep=1)
        except TypeError:
            out = model(**inputs, output_router_logits=True)
    last = out.logits[0, -1, :].float()
    return list(out.router_logits), last


def score_answer(last_logits: torch.Tensor, candidate_ids: list[int], answer_id: int) -> dict:
    """Forced-choice + strict accuracy for one prompt, from final-position logits.

    Every candidate answer word is a single token by construction
    (`probe_set_context.assert_single_token`), so this is an exact comparison of
    logits — no generation, no sampling, no length normalisation needed.

    Reports four things because a base model can fail this task in different
    ways and the difference matters for the substrate verdict:
      * `forced_choice_correct` — argmax restricted to the 8 candidates.
        Chance = 1/8 = 0.125.
      * `forced_choice_prob` — softmax over just the candidates; a graded
        version of the same thing, far more sensitive than the binary at low
        accuracy, which is exactly the regime a 1B-active base model is in.
      * `strict_top1` — argmax over the FULL vocabulary. Equivalent to greedy
        decoding of the answer token, so it answers "would the model actually
        say it" rather than "can it rank it".
      * `answer_rank` — rank of the correct token in the full vocabulary.
    """
    lg = last_logits.detach().cpu().numpy().astype(np.float64)
    cand = np.array(candidate_ids, dtype=np.int64)
    cand_logits = lg[cand]
    ans_pos = int(np.where(cand == answer_id)[0][0])

    m = cand_logits.max()
    p = np.exp(cand_logits - m)
    p = p / p.sum()

    others = np.delete(cand_logits, ans_pos)
    order = np.argsort(-lg)
    rank = int(np.where(order == answer_id)[0][0]) + 1

    return {
        "forced_choice_correct": bool(int(np.argmax(cand_logits)) == ans_pos),
        "forced_choice_prob": float(p[ans_pos]),
        "forced_choice_margin": float(cand_logits[ans_pos] - others.max()),
        "strict_top1": bool(int(np.argmax(lg)) == answer_id),
        "answer_rank": rank,
    }


def process_prompt(loaded, prompt: dict, candidate_ids: list[int]) -> tuple[object, dict]:
    """One prompt -> (parquet table, accuracy record)."""
    import pyarrow as pa

    tok = loaded.tokenizer
    text = prompt["text"]

    enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
    offsets = [tuple(x) for x in enc["offset_mapping"][0].tolist()]
    inputs = {k: v for k, v in enc.items() if k != "offset_mapping"}
    input_ids = inputs["input_ids"][0]
    n_tokens = int(input_ids.shape[0])

    router_logits, last_logits = _forward(loaded.model, inputs)
    if len(router_logits) != loaded.shape.n_layers:
        raise RuntimeError(
            f"prompt {prompt['prompt_id']}: got {len(router_logits)} router layers, "
            f"expected {loaded.shape.n_layers} — a layer was silently skipped"
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
        schema=_extended_schema(),
    )

    answer_id = loaded.tokenizer(" " + prompt["answer_word"],
                                 add_special_tokens=False)["input_ids"][0]
    acc = score_answer(last_logits, candidate_ids, answer_id)

    # Token spans for the byte-identical measurement windows. Resolved here,
    # at capture time, so the analysis never has to re-tokenise and risk drift.
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
    ap.add_argument("--buckets", type=str, default=None,
                    help="comma-separated bucket filter (debug)")
    ap.add_argument("--out", type=str, default=str(OUT_DIR))
    ap.add_argument("--probe-set", type=str, default=str(PROBE_SET_PATH),
                    help="path to an alternate probe_set_context*.yaml -- lets a "
                         "harder/larger variant be run without touching the file "
                         "docs/CONTEXT_ROT.md's reproduce instructions point at")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pyarrow.parquet as pq

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ps = yaml.safe_load(Path(args.probe_set).read_text())
    prompts = list(ps["prompts"])
    if args.buckets:
        keep = {int(x) for x in args.buckets.split(",")}
        prompts = [p for p in prompts if p["bucket"] in keep]
    prompts.sort(key=lambda p: p["prompt_id"])
    if args.limit:
        prompts = prompts[: args.limit]

    total_tokens = sum(p["n_tokens"] for p in prompts)
    print(f"probe set {ps['probe_set_id']}: {len(prompts)} prompts, "
          f"{total_tokens} tokens, buckets={ps['length_buckets']}", flush=True)

    manifest = _read_manifest(out_dir)
    done = set(manifest["done_prompt_ids"])
    todo = [p for p in prompts if p["prompt_id"] not in done]
    print(f"{len(done)} already done, {len(todo)} to run", flush=True)

    print(f"loading {MODEL_ID} ...", flush=True)
    t0 = time.time()
    loaded = load_model(MODEL_ID, device="cpu", dtype="bfloat16")
    print(f"loaded in {time.time() - t0:.1f}s method={loaded.capture_method} "
          f"n_layers={loaded.shape.n_layers} n_experts={loaded.shape.n_experts} "
          f"top_k={loaded.shape.top_k} norm_topk_prob={loaded.shape.norm_topk_prob}",
          flush=True)

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
        t0 = time.time()
        table, acc = process_prompt(loaded, prompt, candidate_ids)
        pq.write_table(table, out_dir / f"trace_{prompt['prompt_id']:06d}.parquet")
        with acc_path.open("a") as fh:
            fh.write(json.dumps(acc) + "\n")

        done.add(prompt["prompt_id"])
        _write_manifest(out_dir, {"done_prompt_ids": sorted(done)})

        dt = time.time() - t0
        elapsed = time.time() - run_t0
        rate = elapsed / (i + 1)
        eta = rate * (len(todo) - i - 1) / 60.0
        print(f"[{i + 1}/{len(todo)}] pid={prompt['prompt_id']:>4} "
              f"bucket={prompt['bucket']:>5} n_tok={acc['n_tokens']:>5} "
              f"{dt:6.1f}s  fc={int(acc['forced_choice_correct'])} "
              f"p={acc['forced_choice_prob']:.3f} rank={acc['answer_rank']:>6}  "
              f"ETA {eta:6.1f} min", flush=True)

    meta = build_meta(loaded, device="cpu", dtype="bfloat16", seed=args.seed,
                      repo_root=str(REPO_ROOT))
    meta_d = json.loads(meta.model_dump_json())
    meta_d["probe_set_id"] = ps["probe_set_id"]
    meta_d["extra_columns"] = ["router_entropy"]
    meta_d["note"] = (
        "Captured by scripts/run_context_sweep.py, not capture.py::capture_to_dir "
        "— same route_from_logits code path, plus full-softmax router entropy and "
        "final-position LM logits which the frozen schema does not carry."
    )
    (out_dir / "meta.json").write_text(json.dumps(meta_d, indent=2))

    print(f"done: {len(done)} prompts total, "
          f"{(time.time() - run_t0) / 3600:.2f}h this run", flush=True)


if __name__ == "__main__":
    main()
