"""Validates tests/fixtures/atlas_synthetic.json against the frozen
Atlas pydantic schema (expertatlas.schemas.Atlas) -- catches drift between
the fixture generator and the real atlas.json contract early, since WS-D
builds against this fixture before real data exists (PLAN.md §4)."""

from __future__ import annotations

import json
from pathlib import Path

from expertatlas.schemas import Atlas

FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "atlas_synthetic.json"


def test_fixture_exists():
    assert FIXTURE_PATH.exists(), (
        "run `python tests/fixtures/make_synthetic_fixture.py` to generate it"
    )


def test_fixture_matches_atlas_schema():
    raw = json.loads(FIXTURE_PATH.read_text())
    raw.pop("_planted", None)  # informational field, not part of the frozen schema
    atlas = Atlas.model_validate(raw)
    assert len(atlas.experts) == atlas.model.n_layers * atlas.model.n_experts_per_layer


def test_fixture_has_exactly_one_planted_specialist():
    raw = json.loads(FIXTURE_PATH.read_text())
    significant_experts = [e for e in raw["experts"] if e["significant"]]
    assert len(significant_experts) == 1
    assert significant_experts[0]["uid"] == raw["_planted"]["expert_uid"]
