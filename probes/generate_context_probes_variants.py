"""Generate probes/probe_set_context_variants.yaml -- harder OLMoE probe
variants that remove the lexical-anchor advantage the span detector enjoyed.

The lexical detector (docs/SPAN_DISCOVERY_SOLVED.md) was validated on a
substrate where the question repeats the needle's wording almost verbatim.
These two pre-declared variants weaken that advantage in different, measured
ways:

  paraphrase -- the needle asserts the fact with completely different wording
    from the question; the ONLY shared contentful token is the entity name.
      needle  : "Staff entering the {entity} branch must present the
                 passphrase {word} at the desk."
      question: unchanged ("What is the security codeword for the {entity}
                 office?").
    Tests whether a single shared referent token is enough for the detector.

  multihop -- the fact is split across two sentences that never co-occur:
      hop1 (bridge): "The {entity} office is designated Site {site}."
      hop2 (fact)  : "The security codeword for Site {site} is {word}."
    The question mentions only {entity}; the answer lives in hop2, which
    shares NO contentful token with the question. hop1 sits at a different
    depth. The oracle span is hop2 (where the answer word is). Expected
    failure mode for the lexical detector: it finds hop1 and misses hop2 --
    that expectation is registered here, before any evaluation.

Design mirrors the depth set: depths 0.15/0.50 for the fact sentence,
buckets 256 and 3840, 8 distractors, 8 replicates, similar+dissimilar
haystacks => 2 variants x 2 depths x 2 buckets x 2 haystacks x 8 = 256
prompts, plus a 16-prompt DEV arm per variant (1024 tokens, depth 0.15) for
detector-width calibration only. Distractors are re-templated per variant so
they compete in the same form as the needle. In the multihop variant the
bridge sentence is placed at (needle_depth + 0.25) mod 0.9 + 0.05, i.e. far
from the fact; its char span is recorded as `bridge_char_span`.

Usage:
    HF_HUB_CACHE=$PWD/data/hf_cache HF_HUB_OFFLINE=1 \
      .venv/bin/python probes/generate_context_probes_variants.py
"""

from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from transformers import AutoTokenizer

import probes.probe_set_context as base
from probes.generate_context_probes_depth import _distractor_depths

OUT = Path(__file__).parent / "probe_set_context_variants.yaml"
REPLICATES = 8
SEED = 7
DEPTHS = (0.15, 0.50)
BUCKETS = (256, 3840)
DEV_BUCKET = 1024
N_DISTRACTORS = 8

PARA_NEEDLE = ("Staff entering the {entity} branch must present the "
               "passphrase {word} at the desk.")
HOP1 = "The {entity} office is designated Site {site}."
HOP2 = "The security codeword for Site {site} is {word}."
SITES = ("Kestrel", "Osprey", "Falcon", "Heron",
         "Plover", "Curlew", "Sandpiper", "Avocet")


def _bridge_depth(needle_depth: float) -> float:
    return (needle_depth + 0.25) % 0.90 + 0.05


def _assemble(sentences, inserts, entity):
    """Like base._assemble but with arbitrary (depth, sentence) inserts.

    Returns (text, spans) where spans[i] is the char span of inserts[i].
    """
    n = len(sentences)
    body = list(sentences)
    order = sorted(range(len(inserts)), key=lambda i: -inserts[i][0])
    for i in order:
        depth, sent = inserts[i]
        body.insert(min(int(round(depth * n)), len(body)), sent)
    text = base.PREAMBLE + " ".join(body)
    spans = []
    for _, sent in inserts:
        s = text.index(sent)
        spans.append((s, s + len(sent)))
    question = base.QUESTION_TEMPLATE.format(entity=entity)
    q_start = len(text)
    text = text + question
    return text, spans, (q_start, len(text))


def build(tokenizer) -> dict:
    def n_tok(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    base.assert_single_token(tokenizer)
    prompts = []
    pid = 0
    for variant in ("paraphrase", "multihop"):
        for rep in range(REPLICATES):
            rng = random.Random(SEED * 100003 + rep)
            entity = base.ENTITIES[rep % len(base.ENTITIES)]
            answer_word = base.CANDIDATE_WORDS[rep % len(base.CANDIDATE_WORDS)]
            site = SITES[rep % len(SITES)]
            other_e = [e for e in base.ENTITIES if e != entity]
            other_w = [w for w in base.CANDIDATE_WORDS if w != answer_word]
            other_s = [s for s in SITES if s != site]

            if variant == "paraphrase":
                needle = PARA_NEEDLE.format(entity=entity, word=answer_word)
                distractor_pool = [
                    PARA_NEEDLE.format(entity=other_e[i % len(other_e)],
                                       word=other_w[i % len(other_w)])
                    for i in range(N_DISTRACTORS)]
            else:
                needle = HOP2.format(site=site, word=answer_word)
                # distractors compete on BOTH hops: fake bridges + fake facts
                distractor_pool = []
                for i in range(N_DISTRACTORS // 2):
                    distractor_pool.append(HOP1.format(
                        entity=other_e[i % len(other_e)],
                        site=other_s[i % len(other_s)]))
                    distractor_pool.append(HOP2.format(
                        site=other_s[i % len(other_s)],
                        word=other_w[i % len(other_w)]))

            for depth in DEPTHS:
                d_depths = _distractor_depths(depth)
                for haystack in base.HAYSTACK_DOMAINS:
                    stream = base._generate_haystack_stream(
                        haystack, random.Random(rng.randrange(1 << 30)), 420)
                    for bucket in BUCKETS + ((DEV_BUCKET,)
                                                 if depth == 0.15 else ()):
                        inserts = [(depth, needle)]
                        if variant == "multihop":
                            inserts.append((_bridge_depth(depth),
                                            HOP1.format(entity=entity, site=site)))
                        for dd, d in zip(d_depths, distractor_pool):
                            inserts.append((dd, d))
                        lo, hi, best = 0, len(stream), None
                        while lo <= hi:
                            mid = (lo + hi) // 2
                            text, spans, qspan = _assemble(
                                stream[:mid], inserts, entity)
                            t = n_tok(text)
                            if t <= bucket:
                                best = (text, spans, qspan, t)
                                lo = mid + 1
                            else:
                                hi = mid - 1
                        if best is None:
                            raise RuntimeError(f"bucket {bucket} too small")
                        text, spans, qspan, t = best
                        p = {
                            "prompt_id": pid, "text": text, "variant": variant,
                            "bucket": bucket, "n_tokens": t,
                            "haystack": haystack,
                            "n_distractors": N_DISTRACTORS,
                            "replicate": rep, "entity": entity,
                            "answer_word": answer_word,
                            "needle_depth": depth,
                            "dev": bucket == DEV_BUCKET,
                            "needle_char_span": list(spans[0]),
                            "question_char_span": list(qspan),
                            "sha1": hashlib.sha1(text.encode()).hexdigest()[:12],
                        }
                        if variant == "multihop":
                            p["site"] = site
                            p["bridge_char_span"] = list(spans[1])
                        prompts.append(p)
                        pid += 1

    return {
        "probe_set_id": "probe_set_context_variants",
        "tokenizer_model_id": base.MODEL_ID,
        "variants": ["paraphrase", "multihop"],
        "length_buckets": list(BUCKETS), "dev_bucket": DEV_BUCKET,
        "needle_depths": list(DEPTHS),
        "candidate_words": list(base.CANDIDATE_WORDS),
        "chance_accuracy": 1.0 / len(base.CANDIDATE_WORDS),
        "n_replicates": REPLICATES, "seed": SEED,
        "n_prompts": len(prompts),
        "notes": (
            "Harder variants removing the lexical-anchor advantage: "
            "paraphrase (only the entity token is shared between question "
            "and needle) and multihop (the answer sentence shares no "
            "contentful token with the question; a bridge sentence links "
            "them). Registered before any evaluation."
        ),
        "prompts": prompts,
    }


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(base.MODEL_ID)
    ps = build(tokenizer)
    OUT.write_text(yaml.safe_dump(ps, sort_keys=False, allow_unicode=True))
    print(f"wrote {OUT}: {ps['n_prompts']} prompts")
    for v in ("paraphrase", "multihop"):
        for depth in DEPTHS:
            for b in BUCKETS + (DEV_BUCKET,):
                got = [p["n_tokens"] for p in ps["prompts"]
                       if p["variant"] == v and p["needle_depth"] == depth
                       and p["bucket"] == b]
                if got:
                    print(f"  {v:>10} depth {depth:.2f} bucket {b:>5}: "
                          f"n={len(got)} min={min(got)} max={max(got)}")


if __name__ == "__main__":
    main()
