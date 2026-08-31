"""Emit probe_set_v1.yaml from the factorial design in content.py.

Run:  python probes/generate.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from content import FORMATS, LANGS, REGISTERS, TOPICS, render  # noqa: E402

OUT = Path(__file__).resolve().parent / "probe_set_v1.yaml"
PROBE_SET_ID = "probe_set_v1"


def build() -> dict:
    prompts, pid = [], 0
    for topic in TOPICS:
        for lang in LANGS:
            for register in REGISTERS:
                for fmt in FORMATS:
                    for stem_idx in range(len(TOPICS[topic])):
                        text = render(topic, stem_idx, lang, register, fmt)
                        prompts.append({
                            "prompt_id": pid,
                            "text": text,
                            "topic": topic,
                            "lang": lang,
                            "register": register,
                            "format": fmt,
                            "stem": stem_idx,
                            # Balanced within every cell: stem 0 -> A, stem 1 -> B.
                            # H6 then compares two halves that are matched on every
                            # factor and differ only in content, which is exactly
                            # the replication question.
                            "split": "A" if stem_idx % 2 == 0 else "B",
                            "sha1": hashlib.sha1(text.encode()).hexdigest()[:12],
                        })
                        pid += 1

    return {
        "probe_set_id": PROBE_SET_ID,
        "version": "1.0",
        "translation_reviewed": False,
        "factors": {
            "topic": sorted(TOPICS),
            "lang": list(LANGS),
            "register": list(REGISTERS),
            "format": list(FORMATS),
        },
        "n_cells": len(TOPICS) * len(LANGS) * len(REGISTERS) * len(FORMATS),
        "n_prompts": len(prompts),
        "notes": (
            "Payloads (code/notation/scenario) are byte-identical across all language "
            "cells. This is the within-subjects control that separates syntax affinity "
            "from language affinity. See probes/README.md."
        ),
        "prompts": prompts,
    }


if __name__ == "__main__":
    data = build()
    OUT.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=1000))
    print(f"wrote {OUT} — {data['n_prompts']} prompts across {data['n_cells']} cells")
