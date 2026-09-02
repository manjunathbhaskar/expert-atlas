# WS-D — visualiser

Self-contained WebGL atlas. `viz/template.html` + `viz/build.py` -> one HTML file.

## Why raw WebGL and not Three.js

PLAN.md §8 requires a single self-contained file with no CDN. Three.js is ~600 KB
to inline; a point cloud with lines needs a few hundred lines of WebGL. The result
opens by double-click, works air-gapped, and has zero supply-chain surface — which
matters because the target user cannot send confidential material to a third party.

## Build

```bash
python viz/build.py data/atlas.json -o viz/atlas.html
```

`build.py` validates the atlas (required keys, finite `xyz`, unique uids, no dangling
edges), refuses to emit if the template gained an external reference, and warns above
a 5 MB budget.

## Encoding

| channel | variable |
|---|---|
| position | PCA/UMAP of the **lift** vectors (never raw usage) |
| colour | co-activation community; grey = unassigned |
| size | marginal usage |
| brightness | specialisation score |
| ring outline | survived FDR |

Significance is never colour-alone — it also gets an outline, for accessibility and
because the ring survives greyscale printing.

## Browser verification (2026-08-11)

Rendered the synthetic fixture in Chromium at 1280x720 and confirmed:

- 1024 points render; generalist mass, the planted community (with its
  co-activation edges), and the planted specialist are all visually distinct.
- Hovering the gold-ringed point yields **L03E17 · code.python +2.322 · significant**,
  matching the fixture's `_planted.expected_lift` of 2.3219 exactly.
- Metadata line reports 1024 experts / 16 layers / top-8 / 1 significant / 250,000 tokens.

## Two real bugs the browser caught that unit tests could not

1. **`community: -1` crashed the page.** 1,014 of 1,024 fixture experts are
   unassigned, and JS `-1 % 8 === -1`, so `PALETTE[-1]` was `undefined`. Louvain
   genuinely leaves isolated nodes unassigned, so real data hits this too.
   Unassigned experts now render neutral grey — they must not read as an eighth
   community. Pinned by `test_unassigned_community_is_not_palette_coloured`.

2. **Canvas laid out at drawing-buffer size.** With `position:fixed;inset:0` but no
   CSS `width`/`height`, the canvas took its attribute size (innerWidth x dpr =
   2560x1440 CSS px) and overflowed the viewport, so only the top-left quadrant was
   visible and the cloud's centre sat exactly at the bottom-right corner. Fixed with
   `width:100vw;height:100vh`.

Neither is reachable without a real renderer. Re-run the browser check after any
template change.

## Not implemented

**Replay mode** (step a prompt token-by-token, flash experts as they fire) is in
PLAN.md §8 but is not built: `atlas.json` carries aggregate lift, not per-token
firing sequences. It needs either a trace sidecar or a new contract field, so it is
deferred rather than half-built. Filed for WS-A/WS-C in docs/interface-requests.md.
