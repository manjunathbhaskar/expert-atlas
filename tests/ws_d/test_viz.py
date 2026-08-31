"""WS-D tests: layout correctness and the self-containment guarantee.

Rendering itself is verified in a real browser (see docs/ws_d.md); these tests
cover everything checkable without one, and the offline guarantee — which is a
hard requirement, since the target user cannot send confidential material to a
third-party host.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "viz"))

from build import EXTERNAL, PLACEHOLDER, build, validate  # noqa: E402
from expertatlas.layout import compute_layout, lift_matrix  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "atlas_synthetic.json"


@pytest.fixture(scope="module")
def atlas():
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("viz") / "atlas.html"
    build(FIXTURE, out)
    return out.read_text()


# --- the offline guarantee ---------------------------------------------------

def test_output_has_no_external_references(built):
    """No CDN, no remote font, no analytics. The page must work air-gapped."""
    assert not EXTERNAL.findall(built)
    assert "cdn." not in built
    assert "googleapis" not in built


def test_placeholder_fully_replaced(built):
    assert PLACEHOLDER not in built
    assert '"schema_version"' in built


def test_data_is_inlined_not_fetched(built):
    """A fetch()/XHR would break offline use and leak the atlas path."""
    assert "fetch(" not in built
    assert "XMLHttpRequest" not in built


def test_size_within_budget(built):
    mb = len(built.encode()) / 1e6
    assert mb < 5.0, f"{mb:.1f} MB exceeds the 5 MB budget"


# --- validation catches malformed atlases ------------------------------------

def test_validate_accepts_fixture(atlas):
    assert validate(atlas) == []


def test_validate_rejects_missing_xyz(atlas):
    bad = json.loads(json.dumps(atlas))
    bad["experts"][0].pop("xyz")
    assert any("xyz" in e for e in validate(bad))


def test_validate_rejects_non_finite_coords(atlas):
    bad = json.loads(json.dumps(atlas))
    bad["experts"][0]["xyz"] = [float("nan"), 0.0, 0.0]
    assert any("non-finite" in e for e in validate(bad))


def test_validate_rejects_duplicate_uids(atlas):
    bad = json.loads(json.dumps(atlas))
    bad["experts"][1]["uid"] = bad["experts"][0]["uid"]
    assert any("duplicate" in e for e in validate(bad))


def test_validate_rejects_dangling_edges(atlas):
    bad = json.loads(json.dumps(atlas))
    bad.setdefault("coactivation", {}).setdefault("edges", []).append(["NOPE", "L00E00", 1.0])
    assert any("unknown expert" in e for e in validate(bad))


# --- layout ------------------------------------------------------------------

def test_layout_separates_planted_specialist(atlas):
    """PLAN.md §6.2: the planted expert must be visually separable.

    Layout is computed from lift, so an expert with strong single-domain lift
    should sit far from the generalist mass. If this fails the 3-D view is
    decorative and must be labelled as such rather than shipped as a finding.
    """
    experts = atlas["experts"]
    lift, _ = lift_matrix(experts)
    coords, _method = compute_layout(lift, method="pca", seed=0)

    idx = {e["uid"]: i for i, e in enumerate(experts)}[atlas["_planted"]["expert_uid"]]
    d = np.linalg.norm(coords - coords.mean(axis=0), axis=1)
    z = (d[idx] - d.mean()) / d.std()
    assert z > 2.0, f"planted expert only {z:.1f} sigma from the centroid"


def test_layout_is_deterministic(atlas):
    lift, _ = lift_matrix(atlas["experts"])
    a, _ = compute_layout(lift, method="pca", seed=0)
    b, _ = compute_layout(lift, method="pca", seed=0)
    np.testing.assert_allclose(a, b)


def test_layout_output_shape_and_finiteness(atlas):
    lift, _ = lift_matrix(atlas["experts"])
    coords, method = compute_layout(lift, method="pca")
    assert coords.shape == (len(atlas["experts"]), 3)
    assert np.isfinite(coords).all()
    assert method in ("pca", "umap")


def test_layout_rejects_degenerate_input():
    with pytest.raises(ValueError):
        compute_layout(np.zeros((2, 4)))


def test_lift_matrix_column_order_is_stable(atlas):
    _, d1 = lift_matrix(atlas["experts"])
    _, d2 = lift_matrix(atlas["experts"])
    assert d1 == d2 == sorted(d1)


# --- encoding correctness ----------------------------------------------------

def test_unassigned_community_is_not_palette_coloured(built):
    """community == -1 means Louvain assigned none. Colouring those from the
    palette would imply structure the null model never supported — and in JS
    `-1 % 8` is -1, which silently indexed past the array end (a real bug this
    test now pins)."""
    assert "communityColour" in built
    assert "UNASSIGNED" in built
    assert re.search(r"c\s*<\s*0", built), "no guard against negative community ids"


def test_significance_is_not_encoded_by_colour_alone(built):
    """Accessibility: significance also gets an outline ring."""
    assert "vSig > 0.5" in built


def test_layout_caveat_is_displayed(built):
    """PLAN.md §9b requires the presentational caveat to ship with the artifact,
    not just live in the docs."""
    assert "Proximity alone is not" in built
