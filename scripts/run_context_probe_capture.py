"""Hidden-state capture for the residual-stream needle probe (WS1, upstream test).

Motivation (docs/CONTEXT_ROT_STORY.md §4): two router-level interventions moved
routing as designed without recovering accuracy, so the working hypothesis is
that the failure sits upstream of the router — the hidden state no longer
carries the needle's content by the time it matters. This script collects the
raw material to test that directly: per-layer hidden states at the positions
where the needle's information (a) is written and (b) must be read.

For each of the 192 hard-variant prompts, one teacher-forced forward pass with
`output_hidden_states=True`, saving float32 vectors for every hidden-state
layer (embedding + 16 transformer layers = 17) at four positions:

  * `needle_last` — last token of the needle span (where the answer word sits);
  * `needle_mean` — mean over the needle span;
  * `q_mean`      — mean over the question span;
  * `final`       — the final position, whose LM logits score the answer.

Also records the same forced-choice scoring as `run_context_sweep.py` from the
same pass, as a consistency check against `data/context_traces_hard/accuracy.jsonl`.

Output: data/context_probe/hidden_XXXXXX.npz  (one per prompt, ~550 KB)
        data/context_probe/records.jsonl      (one line per prompt)
        data/context_probe/manifest.json      (resume state)

Nothing existing is touched. Resumable.

Usage:
    HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 \
      .venv/bin/python scripts/run_context_probe_capture.py
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

from expertatlas.capture import load_model
from expertatlas.context_metrics import token_span_from_chars

MODEL_ID = "allenai/OLMoE-1B-7B-0924"
REPO_ROOT = Path(__file__).parent.parent
PROBE_SET_PATH = REPO_ROOT / "probes" / "probe_set_context_hard.yaml"
OUT_DIR = REPO_ROOT / "data" / "context_probe"
POSITIONS = ("needle_last", "needle_mean", "q_mean", "final")


def score_answer(last_logits, candidate_ids, answer_id):
    lg = last_logits.detach().cpu().numpy().astype(np.float64)
    cand = np.array(candidate_ids, dtype=np.int64)
    cand_logits = lg[cand]
    ans_pos = int(np.where(cand == answer_id)[0][0])
    m = cand_logits.max()
    p = np.exp(cand_logits - m)
    p = p / p.sum()
    return {
        "forced_choice_correct": bool(int(np.argmax(cand_logits)) == ans_pos),
        "forced_choice_prob": float(p[ans_pos]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-set", type=str, default=str(PROBE_SET_PATH))
    ap.add_argument("--out", type=str, default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_p = out_dir / "manifest.json"
    done = set(json.loads(manifest_p.read_text())["done"]) if manifest_p.exists() else set()

    ps = yaml.safe_load(Path(args.probe_set).read_text())
    prompts = sorted(ps["prompts"], key=lambda p: p["prompt_id"])
    todo = [p for p in prompts if p["prompt_id"] not in done]
    print(f"{len(done)} done, {len(todo)} to run", flush=True)

    loaded = load_model(MODEL_ID, device="cpu", dtype="bfloat16")
    tok = loaded.tokenizer
    candidate_ids = [tok(" " + w, add_special_tokens=False)["input_ids"][0]
                     for w in ps["candidate_words"]]

    rec_path = out_dir / "records.jsonl"
    t_run = time.time()
    for i, prompt in enumerate(todo):
        t0 = time.time()
        enc = tok(prompt["text"], return_tensors="pt", return_offsets_mapping=True)
        offsets = [tuple(x) for x in enc["offset_mapping"][0].tolist()]
        inputs = {k: v for k, v in enc.items() if k != "offset_mapping"}
        n_tokens = int(inputs["input_ids"].shape[1])

        with torch.no_grad():
            out = loaded.model(**inputs, output_hidden_states=True, logits_to_keep=1)
        hs = out.hidden_states  # tuple of (1, T, H), len = n_layers + 1

        n_span = token_span_from_chars(offsets, prompt["needle_char_span"])
        q_span = token_span_from_chars(offsets, prompt["question_char_span"])

        arrs = {}
        for li, h in enumerate(hs):
            h = h[0].float()
            arrs[f"needle_last_{li}"] = h[n_span[1] - 1].numpy()
            arrs[f"needle_mean_{li}"] = h[n_span[0]:n_span[1]].mean(dim=0).numpy()
            arrs[f"q_mean_{li}"] = h[q_span[0]:q_span[1]].mean(dim=0).numpy()
            arrs[f"final_{li}"] = h[-1].numpy()
        np.savez_compressed(out_dir / f"hidden_{prompt['prompt_id']:06d}.npz", **arrs)

        answer_id = tok(" " + prompt["answer_word"], add_special_tokens=False)["input_ids"][0]
        acc = score_answer(out.logits[0, -1, :].float(), candidate_ids, answer_id)
        rec = {
            "prompt_id": int(prompt["prompt_id"]),
            "bucket": int(prompt["bucket"]),
            "n_tokens": n_tokens,
            "haystack": prompt["haystack"],
            "n_distractors": int(prompt["n_distractors"]),
            "replicate": int(prompt["replicate"]),
            "entity": prompt["entity"],
            "answer_word": prompt["answer_word"],
            "needle_token_span": list(n_span),
            "question_token_span": list(q_span),
            "n_hidden_layers": len(hs),
            **{k: prompt[k] for k in ("pairing_set",) if k in prompt},
            **acc,
        }
        with rec_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        done.add(prompt["prompt_id"])
        manifest_p.write_text(json.dumps({"done": sorted(done)}))

        dt = time.time() - t0
        eta = (time.time() - t_run) / (i + 1) * (len(todo) - i - 1) / 60
        print(f"[{i+1}/{len(todo)}] pid={prompt['prompt_id']:>4} "
              f"bucket={prompt['bucket']:>5} {dt:6.1f}s fc={int(acc['forced_choice_correct'])} "
              f"p={acc['forced_choice_prob']:.3f} ETA {eta:6.1f} min", flush=True)

    print(f"done in {(time.time()-t_run)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
