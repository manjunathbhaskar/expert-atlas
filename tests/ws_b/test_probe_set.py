"""WS-B tests: the probe set must be balanced, controlled, and reproducible.

No tokenizer or model required — these run in under a second.
"""

import sys
from collections import Counter
from pathlib import Path

import pytest

PROBES = Path(__file__).resolve().parents[2] / "probes"
sys.path.insert(0, str(PROBES))

from validate import (  # noqa: E402
    check_balance, check_duplicates, check_payload_invariance, load,
)
from content import LANGS, REGISTERS, FORMATS, TOPICS, render  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return load()


def test_full_factorial_present(data):
    """240 cells = 10 topics x 4 langs x 2 registers x 3 formats, 2 prompts each."""
    assert data["n_cells"] == 240
    assert data["n_prompts"] == 480


def test_balanced(data):
    assert check_balance(data) == []


def test_no_duplicate_prompts(data):
    """Near-identical prompts would inflate apparent per-cell sample size."""
    assert check_duplicates(data) == []


def test_payload_invariance_control(data):
    """THE experimental control.

    For a fixed (topic, stem), the payload must appear byte-identically in every
    language cell. This is what separates 'syntax expert' from 'language expert':
    if the code bytes are constant and only prose changes, an expert firing on the
    code across all four languages is responding to syntax, not to language.

    If this test fails the atlas cannot make its central claim.
    """
    assert check_payload_invariance(data) == []


def test_payload_identical_across_languages_directly():
    """Stronger, independent form of the control — does not rely on validate.py."""
    for topic, stems in TOPICS.items():
        for i in range(len(stems)):
            rendered = [render(topic, i, lang, "formal", "prose") for lang in LANGS]
            payload = stems[i]["payload"]
            assert all(payload in r for r in rendered), f"{topic}[{i}] payload varies"


def test_every_cell_has_both_splits(data):
    """H6 (split-half replication) requires A and B matched on every factor."""
    cells = {}
    for p in data["prompts"]:
        key = (p["topic"], p["lang"], p["register"], p["format"])
        cells.setdefault(key, set()).add(p["split"])
    assert all(v == {"A", "B"} for v in cells.values())


def test_splits_globally_balanced(data):
    counts = Counter(p["split"] for p in data["prompts"])
    assert counts["A"] == counts["B"] == 240


def test_format_instructions_present(data):
    """json cells must name their keys; bulleted cells must name their points.
    A silently-missing instruction would make the format factor inert."""
    for p in data["prompts"]:
        if p["format"] == "json":
            assert '"' in p["text"], f"prompt {p['prompt_id']}: no JSON keys"
        elif p["format"] == "bulleted":
            assert "," in p["text"], f"prompt {p['prompt_id']}: no bullet points"


def test_register_actually_differs(data):
    """formal and casual must produce different text in the same cell, or the
    register factor is measuring nothing."""
    for topic in TOPICS:
        f = render(topic, 0, "en", "formal", "prose")
        c = render(topic, 0, "en", "casual", "prose")
        assert f != c, f"{topic}: register has no effect"


def test_generation_is_deterministic(data):
    """Regenerating must reproduce identical text — sha1 is the contract."""
    import hashlib
    for p in data["prompts"]:
        again = render(p["topic"], p["stem"], p["lang"], p["register"], p["format"])
        assert hashlib.sha1(again.encode()).hexdigest()[:12] == p["sha1"], (
            f"prompt {p['prompt_id']} not reproducible — regenerate probe_set_v1.yaml"
        )


def test_translation_review_flag_is_honest(data):
    """Must stay False until a native speaker has actually reviewed zh/ja/de.
    Flipping this without review would misrepresent the artifact."""
    assert data["translation_reviewed"] is False


def test_all_languages_non_empty(data):
    """Guards against a missing translation silently rendering as English."""
    for lang in LANGS:
        texts = [p["text"] for p in data["prompts"] if p["lang"] == lang]
        assert len(texts) == 120
        if lang != "en":
            en = {p["text"] for p in data["prompts"] if p["lang"] == "en"}
            assert not (set(texts) & en), f"{lang} prompts identical to English"
