"""Context-rot probe set generator.

Builds `probes/probe_set_context.yaml`: a needle-in-a-haystack set modelled on
Chroma's *Context Rot: How Increasing Input Tokens Impacts LLM Performance*
(2025), whose central design principle is:

    **hold task difficulty fixed and vary ONLY input length.**

`probe_set_v1.yaml` is deliberately left untouched (it is the factorial
topic/lang/register/format set that every other result in this repo is computed
on). This is a separate, additional set with a different independent variable.

---------------------------------------------------------------------------
Which Chroma results this maps onto
---------------------------------------------------------------------------
Two of the report's conditions are reproduced, so the routing measurements land
on a *named* comparison rather than "context rot in general":

1. **The distractor condition** — Chroma's needle-in-a-haystack section that
   contrasts a clean haystack against a haystack seeded with distractors
   (they run 1 and 4 distractors; we run 0 and 4). Their distractors are
   sentences that plausibly answer the question but are not the needle.
   Reproduced here exactly in that spirit: a distractor is the *same sentence
   template as the needle* with a different entity and a different answer word,
   and the wrong answer word is drawn from the same forced-choice candidate
   pool the needle's answer comes from, so distractors compete directly in
   scoring rather than being decorative.

2. **The needle/haystack similarity condition** — Chroma vary the semantic
   relationship between the needle and the haystack it is buried in. Here the
   *same* needle is buried in either a topically similar haystack (corporate
   facilities/security operations prose, i.e. the needle's own subject matter)
   or a topically dissimilar one (glacial geology prose). Nothing about the
   needle, the question, or the answer changes.

Crossed 2 x 2, over >= 6 length buckets. See `docs/CONTEXT_ROT.md`.

---------------------------------------------------------------------------
The control that makes routing measurable across lengths
---------------------------------------------------------------------------
This is the part that matters, and it is the reason this generator exists
rather than a quick script.

Within one `replicate`, the entity, the answer word, the needle sentence and
the question block are **byte-identical across every length bucket and every
condition**. Only the amount of haystack around them changes.

That buys two things that no post-hoc normalisation can:

* A **measurement window** (the trailing question block) whose token content is
  literally the same string in every cell. Per-token routing statistics over
  that window compare like with like: same tokens, same task, different amount
  of preceding context. Length is the only thing that varies.
* A **needle window** with the same property, for the specialisation metric.

Haystack content is also matched across buckets: one long sentence stream is
generated per (replicate, haystack domain) and shorter buckets take a *prefix*
of it, so the haystack's own topic mix does not drift with length.

Answers are single-token by construction (verified against the real OLMoE
tokenizer at generation time), so forced-choice accuracy is readable off the
final-position logits of the *same* forward pass that produces the routing
trace. No second pass, no generation, no sampling.

Usage:
    python probes/probe_set_context.py --replicates 3 --out probes/probe_set_context.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import yaml

MODEL_ID = "allenai/OLMoE-1B-7B-0924"

# --------------------------------------------------------------------------
# Design constants
# --------------------------------------------------------------------------

#: Target *total prompt length* in OLMoE tokens. 7 buckets, ~30x span, top end
#: kept clear of max_position_embeddings=4096 so nothing is silently truncated.
LENGTH_BUCKETS = (128, 256, 512, 1024, 2048, 3072, 3840)

#: Chroma run 1 and 4 distractors; 0 gives a clean control arm.
DISTRACTOR_LEVELS = (0, 4)

HAYSTACK_DOMAINS = ("similar", "dissimilar")

#: Needle sits at 50% depth in every prompt. Depth is a large effect in the
#: NIAH literature and is deliberately HELD FIXED here rather than varied --
#: this set has one independent variable. Stated as a scope limit in
#: docs/CONTEXT_ROT.md, not quietly omitted.
NEEDLE_DEPTH = 0.50

#: Distractor depths, chosen to straddle the needle without colliding with it.
DISTRACTOR_DEPTHS = (0.18, 0.34, 0.66, 0.82)

#: Forced-choice answer pool. Every entry is verified to be exactly ONE OLMoE
#: token when prefixed with a space (assert_single_token below). Chance
#: accuracy on the forced-choice metric is therefore exactly 1/8 = 12.5%.
CANDIDATE_WORDS = (
    "silver", "thunder", "marble", "velvet",
    "copper", "lantern", "cobalt", "magnet",
)

#: Single-word cities so the needle/question/distractor templates stay
#: byte-stable and tokenise identically wherever they appear.
ENTITIES = (
    "Zurich", "Osaka", "Toronto", "Helsinki",
    "Nairobi", "Lisbon", "Denver", "Bergen",
)

NEEDLE_TEMPLATE = "The security codeword for the {entity} office is {word}."
QUESTION_TEMPLATE = (
    "\n\nQuestion: What is the security codeword for the {entity} office?"
    "\nAnswer: The security codeword for the {entity} office is"
)

PREAMBLE = "The following is an internal reference document.\n\n"


# --------------------------------------------------------------------------
# Haystack generation
# --------------------------------------------------------------------------
#
# Sentences are generated combinatorially from templates x slot pools rather
# than drawn from a fixed list. A fixed list long enough for the 3840-token
# bucket would have to REPEAT for that bucket and not for the short ones,
# which would make "amount of verbatim repetition" covary with length -- a
# confound with the independent variable. Combinatorial generation keeps every
# haystack sentence distinct at every length.

_SIMILAR_TEMPLATES = (
    "The {region} facilities team completed the {quarter} inspection of the {asset} without incident.",
    "Badge access to the {asset} is reviewed by the {region} operations desk every {period}.",
    "Maintenance on the {asset} in the {region} building is scheduled for the {quarter} window.",
    "The {region} site manager filed the {quarter} occupancy report covering the {asset}.",
    "Visitor logs for the {asset} are retained by the {region} security office for {period}.",
    "A {quarter} audit of the {region} {asset} found no outstanding remediation items.",
    "The {region} front desk issues temporary passes for the {asset} on a {period} basis.",
    "Cleaning contracts for the {region} {asset} were renewed ahead of the {quarter} review.",
    "Fire drill attendance in the {region} building was logged during the {quarter} exercise.",
    "The {asset} in the {region} office was recatalogued as part of the {quarter} asset sweep.",
    "Procurement approved a {period} service agreement for the {region} {asset}.",
    "The {region} facilities budget for the {quarter} includes refurbishment of the {asset}.",
)

_DISSIMILAR_TEMPLATES = (
    "Meltwater channels beneath the {glacier} glacier shift on a {period} cycle during {season}.",
    "Sediment cores taken near the {glacier} terminus record {season} deposition over {period}.",
    "The {glacier} ice margin retreated measurably through the {season} of the survey period.",
    "Basal sliding rates under the {glacier} glacier increase sharply in late {season}.",
    "Moraine ridges downvalley of the {glacier} preserve a {period} record of advance.",
    "Crevasse fields on the upper {glacier} widen through {season} as accumulation thins.",
    "Isotope ratios in {glacier} firn track {season} temperature over roughly a {period}.",
    "Proglacial lakes fed by the {glacier} drain episodically across a {period} interval.",
    "Ablation stakes on the {glacier} were resurveyed after the {season} melt season.",
    "Till deposits exposed near the {glacier} indicate repeated {season} readvance.",
    "Radar sounding of the {glacier} bed resolved channels formed over a {period}.",
    "Suspended load in streams draining the {glacier} peaks during {season} discharge.",
)

_SLOTS = {
    "region": ("northern", "eastern", "riverside", "downtown", "harbour", "midtown",
               "western", "southern", "lakeside", "uptown", "central", "coastal"),
    "asset": ("loading bay", "server room", "archive vault", "print floor", "roof plant",
              "mail room", "parking deck", "atrium", "workshop", "chiller unit",
              "records store", "staff canteen"),
    "quarter": ("first quarter", "second quarter", "third quarter", "fourth quarter",
                "mid-year", "year-end"),
    "period": ("six months", "two years", "eighteen months", "one year",
               "three years", "nine months"),
    "glacier": ("Vatna", "Kenai", "Rhone", "Tasman", "Aletsch", "Perito",
                "Franz", "Malaspina", "Baltoro", "Fox", "Athabasca", "Hubbard"),
    "season": ("summer", "autumn", "winter", "spring"),
}


def _generate_haystack_stream(domain: str, rng: random.Random, n_sentences: int) -> list[str]:
    """Deterministic stream of distinct haystack sentences for one domain.

    Shorter length buckets consume a PREFIX of this stream, so the haystack's
    topic mixture is identical across buckets and cannot covary with length.
    """
    templates = _SIMILAR_TEMPLATES if domain == "similar" else _DISSIMILAR_TEMPLATES
    seen: set[str] = set()
    out: list[str] = []
    guard = 0
    while len(out) < n_sentences and guard < n_sentences * 200:
        guard += 1
        tpl = templates[rng.randrange(len(templates))]
        slots = {k: v[rng.randrange(len(v))] for k, v in _SLOTS.items()}
        s = tpl.format(**slots)
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    if len(out) < n_sentences:
        raise RuntimeError(
            f"haystack domain {domain!r} exhausted its combinatorial space at "
            f"{len(out)}/{n_sentences} distinct sentences -- add templates or slots "
            "rather than allowing repetition, which would confound length"
        )
    return out


# --------------------------------------------------------------------------
# Prompt assembly
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptSpec:
    prompt_id: int
    text: str
    bucket: int
    n_tokens: int
    haystack: str
    n_distractors: int
    replicate: int
    entity: str
    answer_word: str
    needle_char_span: tuple[int, int]
    question_char_span: tuple[int, int]


def _assemble(
    sentences: list[str],
    needle: str,
    distractors: list[str],
    entity: str,
) -> tuple[str, tuple[int, int], tuple[int, int]]:
    """Interleave needle + distractors into a haystack at fixed depths.

    Returns (text, needle_char_span, question_char_span). Character spans are
    resolved to token spans downstream via the tokenizer's offset mapping --
    storing char spans keeps this file independent of any tokenizer state.
    """
    n = len(sentences)
    body = list(sentences)

    # Insert deepest-first so earlier insertions do not shift later indices.
    inserts: list[tuple[int, str, bool]] = [(int(round(NEEDLE_DEPTH * n)), needle, True)]
    for depth, d in zip(DISTRACTOR_DEPTHS, distractors):
        inserts.append((int(round(depth * n)), d, False))
    inserts.sort(key=lambda x: -x[0])
    for idx, sent, _is_needle in inserts:
        body.insert(min(idx, len(body)), sent)

    haystack_text = " ".join(body)
    text = PREAMBLE + haystack_text

    needle_start = text.index(needle)
    needle_span = (needle_start, needle_start + len(needle))

    question = QUESTION_TEMPLATE.format(entity=entity)
    q_start = len(text)
    text = text + question
    question_span = (q_start, len(text))

    return text, needle_span, question_span


def assert_single_token(tokenizer, words=CANDIDATE_WORDS) -> None:
    """Every forced-choice candidate must be exactly one token with a leading
    space. If this ever fails, forced-choice accuracy silently stops being a
    single-logit comparison and the accuracy metric becomes wrong -- so it is
    an assertion, not a warning."""
    bad = {}
    for w in words:
        ids = tokenizer(" " + w, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            bad[w] = ids
    if bad:
        raise RuntimeError(
            f"candidate words are not single tokens under this tokenizer: {bad}. "
            "Pick different words -- do not weaken the accuracy metric to fit."
        )


def build_probe_set(tokenizer, replicates: int = 3, seed: int = 0) -> dict:
    """Build the full context probe set. Deterministic given (replicates, seed)."""
    assert_single_token(tokenizer)

    def n_tok(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    prompts: list[PromptSpec] = []
    pid = 0

    for rep in range(replicates):
        rng = random.Random(seed * 100003 + rep)
        entity = ENTITIES[rep % len(ENTITIES)]
        answer_word = CANDIDATE_WORDS[rep % len(CANDIDATE_WORDS)]
        needle = NEEDLE_TEMPLATE.format(entity=entity, word=answer_word)

        # Hard distractors: same template, different entity, and a WRONG answer
        # word taken from the forced-choice pool so it competes in scoring.
        other_entities = [e for e in ENTITIES if e != entity]
        other_words = [w for w in CANDIDATE_WORDS if w != answer_word]
        distractor_pool = [
            NEEDLE_TEMPLATE.format(entity=other_entities[i % len(other_entities)],
                                   word=other_words[i % len(other_words)])
            for i in range(max(DISTRACTOR_LEVELS))
        ]

        for haystack in HAYSTACK_DOMAINS:
            # One long stream per (replicate, domain); buckets take prefixes.
            stream = _generate_haystack_stream(
                haystack, random.Random(rng.randrange(1 << 30)), n_sentences=420
            )
            for n_dist in DISTRACTOR_LEVELS:
                distractors = distractor_pool[:n_dist]
                for bucket in LENGTH_BUCKETS:
                    # Grow the haystack prefix until the ASSEMBLED prompt hits
                    # the bucket. Measuring on the assembled text (not the
                    # haystack alone) is what makes the bucket a real total
                    # prompt length rather than an approximation.
                    lo, hi = 0, len(stream)
                    best: tuple | None = None
                    while lo <= hi:
                        mid = (lo + hi) // 2
                        text, nspan, qspan = _assemble(
                            stream[:mid], needle, distractors, entity
                        )
                        t = n_tok(text)
                        if t <= bucket:
                            best = (mid, text, nspan, qspan, t)
                            lo = mid + 1
                        else:
                            hi = mid - 1
                    if best is None:
                        raise RuntimeError(
                            f"bucket {bucket} is too small to hold needle + "
                            f"{n_dist} distractors + question for entity {entity}"
                        )
                    _, text, nspan, qspan, t = best
                    prompts.append(PromptSpec(
                        prompt_id=pid, text=text, bucket=bucket, n_tokens=t,
                        haystack=haystack, n_distractors=n_dist, replicate=rep,
                        entity=entity, answer_word=answer_word,
                        needle_char_span=nspan, question_char_span=qspan,
                    ))
                    pid += 1

    return {
        "probe_set_id": "probe_set_context",
        "version": "1.0",
        "design": "needle-in-haystack; task difficulty fixed, input length varied",
        "maps_to": (
            "Chroma, 'Context Rot: How Increasing Input Tokens Impacts LLM "
            "Performance' (2025) -- the needle-in-a-haystack distractor "
            "comparison (0 vs 4 distractors) and the needle/haystack "
            "similarity comparison. See docs/CONTEXT_ROT.md."
        ),
        "tokenizer_model_id": MODEL_ID,
        "length_buckets": list(LENGTH_BUCKETS),
        "distractor_levels": list(DISTRACTOR_LEVELS),
        "haystack_domains": list(HAYSTACK_DOMAINS),
        "needle_depth": NEEDLE_DEPTH,
        "distractor_depths": list(DISTRACTOR_DEPTHS),
        "candidate_words": list(CANDIDATE_WORDS),
        "chance_accuracy": 1.0 / len(CANDIDATE_WORDS),
        "n_replicates": replicates,
        "n_prompts": len(prompts),
        "seed": seed,
        "invariants": [
            "Within a replicate, the needle sentence and the question block are "
            "byte-identical across every length bucket and every condition.",
            "Shorter buckets use a PREFIX of the same haystack sentence stream, "
            "so haystack topic mix does not covary with length.",
            "Every haystack sentence is distinct: no bucket repeats text that a "
            "shorter bucket does not, which would confound length with repetition.",
            "Every candidate answer word is exactly one token under the OLMoE "
            "tokenizer, so forced-choice accuracy is a single-logit comparison.",
        ],
        "prompts": [
            {
                "prompt_id": p.prompt_id,
                "text": p.text,
                "bucket": p.bucket,
                "n_tokens": p.n_tokens,
                "haystack": p.haystack,
                "n_distractors": p.n_distractors,
                "replicate": p.replicate,
                "entity": p.entity,
                "answer_word": p.answer_word,
                "needle_char_span": list(p.needle_char_span),
                "question_char_span": list(p.question_char_span),
                "sha1": hashlib.sha1(p.text.encode()).hexdigest()[:12],
            }
            for p in prompts
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).parent / "probe_set_context.yaml"))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    ps = build_probe_set(tokenizer, replicates=args.replicates, seed=args.seed)
    Path(args.out).write_text(yaml.safe_dump(ps, sort_keys=False, allow_unicode=True))

    tot = sum(p["n_tokens"] for p in ps["prompts"])
    print(f"wrote {args.out}: {ps['n_prompts']} prompts, {tot} tokens total")
    for b in LENGTH_BUCKETS:
        got = [p["n_tokens"] for p in ps["prompts"] if p["bucket"] == b]
        print(f"  bucket {b:>5}: n={len(got)} actual tokens min={min(got)} max={max(got)}")


if __name__ == "__main__":
    main()
