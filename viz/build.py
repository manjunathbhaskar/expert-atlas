"""Inline an atlas.json into the visualiser template -> one self-contained HTML.

No CDN, no external fetch, no server. Opens by double-click and works offline,
which is a hard requirement (PLAN.md §8) both for GitHub Pages and because the
target user cannot send confidential material to a third-party host.

Usage:
    python viz/build.py data/atlas.json -o viz/atlas.html
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "template.html"
PLACEHOLDER = "__ATLAS_DATA__"

SIZE_BUDGET_MB = 5.0
REQUIRED_TOP = ("schema_version", "model", "stats", "experts")
REQUIRED_EXPERT = ("uid", "layer", "idx", "usage", "lift", "xyz")
# Any URL that would leave the page is a violation of the offline guarantee.
EXTERNAL = re.compile(r'(?:src|href)\s*=\s*["\']\s*(?:https?:)?//', re.I)


def validate(atlas: dict) -> list[str]:
    errs = [f"missing top-level key: {k}" for k in REQUIRED_TOP if k not in atlas]
    experts = atlas.get("experts") or []
    if not experts:
        errs.append("no experts in atlas")
        return errs

    for k in REQUIRED_EXPERT:
        if k not in experts[0]:
            errs.append(f"expert records missing key: {k}")

    uids = [e.get("uid") for e in experts]
    if len(set(uids)) != len(uids):
        errs.append("duplicate expert uids")

    for e in experts:
        xyz = e.get("xyz")
        if not (isinstance(xyz, (list, tuple)) and len(xyz) == 3):
            errs.append(f"{e.get('uid')}: xyz must be 3 floats — run compute_layout first")
            break
        if any(v != v or abs(v) == float("inf") for v in xyz):
            errs.append(f"{e.get('uid')}: non-finite xyz")
            break

    known = set(uids)
    for a, b, _w in (atlas.get("coactivation", {}) or {}).get("edges", []):
        if a not in known or b not in known:
            errs.append(f"edge references unknown expert: {a}-{b}")
            break
    return errs


def build(atlas_path: Path, out_path: Path, minify: bool = True) -> Path:
    atlas = json.loads(atlas_path.read_text())

    errs = validate(atlas)
    if errs:
        raise SystemExit("atlas.json failed validation:\n  " + "\n  ".join(errs))

    payload = json.dumps(atlas, separators=(",", ":") if minify else None)
    # A literal placeholder inside the data would corrupt the template splice.
    if PLACEHOLDER in payload:
        raise SystemExit(f"atlas data contains the reserved token {PLACEHOLDER}")

    html = TEMPLATE.read_text().replace(PLACEHOLDER, payload)

    leaked = EXTERNAL.findall(html)
    if leaked:
        raise SystemExit(f"template references external resources: {leaked[:3]}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)

    mb = out_path.stat().st_size / 1e6
    if mb > SIZE_BUDGET_MB:
        print(f"WARNING: {mb:.1f} MB exceeds the {SIZE_BUDGET_MB} MB budget. "
              f"Store lift sparsely (significant entries only), quantise to 3 dp, "
              f"and cap top_tokens at 10.", file=sys.stderr)
    print(f"wrote {out_path} — {len(atlas['experts'])} experts, {mb:.2f} MB")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("atlas", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=HERE / "atlas.html")
    ap.add_argument("--no-minify", action="store_true")
    a = ap.parse_args()
    build(a.atlas, a.out, minify=not a.no_minify)
