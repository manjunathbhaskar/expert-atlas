"""Dense track stage 0: accuracy sweep + last-row attention capture.

One teacher-forced forward pass per prompt (192 total). Records per prompt:
forced-choice correctness/probability, token spans, and the final-position
attention mass on the needle span per (layer, head) — captured in the SAME
pass so head identification and collapse analysis need no second sweep.

Outputs:
    dense_track/data/records.jsonl   (one row per prompt)
    dense_track/data/needle_mass.npz (prompt_id -> (32, 32) mass matrix)

Usage:
    .venv-dense/bin/python dense_track/run_sweep.py
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

from dense_track.common import (
    DATA_DIR, PROBE_SET, RECORDS, LastRowAttentionCapture,
    char_span_to_token_span, load_dense_model, score_answer,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-set", type=str, default=str(PROBE_SET))
    ap.add_argument("--records", type=str, default=str(RECORDS))
    ap.add_argument("--mass-out", type=str,
                    default=str(DATA_DIR / "needle_mass.npz"))
    args = ap.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ps = yaml.safe_load(Path(args.probe_set).read_text())
    model, tok = load_dense_model()
    candidate_ids = [tok(" " + w, add_special_tokens=False)["input_ids"][0]
                     for w in ps["candidate_words"]]

    rows, masses = [], {}
    t0 = time.time()
    prompts = ps["prompts"]
    for i, p in enumerate(prompts):
        text = p["text"]
        nspan = char_span_to_token_span(tok, text, tuple(p["needle_char_span"]))
        qspan = char_span_to_token_span(tok, text, tuple(p["question_char_span"]))
        ids = tok(text, return_tensors="pt")
        answer_id = tok(" " + p["answer_word"],
                        add_special_tokens=False)["input_ids"][0]
        with LastRowAttentionCapture(model) as cap, torch.no_grad():
            out = model(**ids, logits_to_keep=1)
        res = score_answer(out.logits[0, -1, :].float(), candidate_ids, answer_id)
        masses[str(p["prompt_id"])] = cap.needle_mass(nspan).numpy()
        rows.append({
            "prompt_id": p["prompt_id"], "bucket": p["bucket"],
            "haystack": p["haystack"], "n_distractors": p["n_distractors"],
            "replicate": p["replicate"], "answer_word": p["answer_word"],
            "n_tokens": p["n_tokens"],
            "needle_token_span": list(nspan),
            "question_token_span": list(qspan),
            **res,
        })
        el = time.time() - t0
        print(f"[{i + 1}/{len(prompts)}] pid={p['prompt_id']} b={p['bucket']} "
              f"d={p['n_distractors']} correct={res['forced_choice_correct']} "
              f"p={res['forced_choice_prob']:.3f} "
              f"ETA {(el / (i + 1)) * (len(prompts) - i - 1) / 60:.1f} min",
              flush=True)

    Path(args.records).write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    np.savez_compressed(args.mass_out, **masses)

    print("\naccuracy by bucket x distractors:")
    for b in ps["length_buckets"]:
        for d in sorted({r["n_distractors"] for r in rows}):
            sel = [r for r in rows if r["bucket"] == b and r["n_distractors"] == d]
            if not sel:
                continue
            acc = float(np.mean([r["forced_choice_correct"] for r in sel]))
            print(f"  bucket {b:>5} dist {d}: acc={acc:.3f} (n={len(sel)})")
    for b in ps["length_buckets"]:
        sel = [r for r in rows if r["bucket"] == b]
        acc = float(np.mean([r["forced_choice_correct"] for r in sel]))
        print(f"  bucket {b:>5} ALL: acc={acc:.3f} (n={len(sel)})")


if __name__ == "__main__":
    main()
