"""Formal methodology architecture diagram (academic style, serif + math).

All quantities verbatim from the committed docs. Usage:
.venv/bin/python scripts/make_architecture_figure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).parent.parent / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})

EC = "#4a5568"

# header color, body color per role
INPUT = ("#5a5a9e", "#eceafb")
MODEL = ("#616161", "#f2f2f2")
MEAS = ("#c07a30", "#fdf3e4")
LOC = ("#2b6cb0", "#e7f0fa")
REPAIR = ("#2f855a", "#e9f6ee")
EVAL = ("#b0485a", "#fbeaee")

fig, ax = plt.subplots(figsize=(12.6, 8.9))
ax.set_xlim(0, 12.6)
ax.set_ylim(0.6, 9.4)
ax.axis("off")


def band(y0, y1, label):
    ax.add_patch(Rectangle((0.12, y0), 12.36, y1 - y0, fc="#fafbfc",
                           ec="#d5dbe3", lw=0.9, zorder=0))
    ax.text(0.28, y1 - 0.16, label, ha="left", va="top", fontsize=9.5,
            fontweight="bold", color="#6b7280", zorder=1)


def box(cx, cy, w, h, title, lines, role, fs=8.8, tfs=9.4):
    hc, bc = role
    hh = 0.36
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.045", fc=bc, ec=hc,
                                lw=1.3, zorder=2))
    ax.add_patch(Rectangle((cx - w / 2 + 0.01, cy + h / 2 - hh),
                           w - 0.02, hh + 0.02, fc=hc, ec="none", zorder=3))
    ax.text(cx, cy + h / 2 - hh / 2 + 0.01, title, ha="center", va="center",
            fontsize=tfs, fontweight="bold", color="white", zorder=4)
    ax.text(cx, cy - hh / 2 + 0.04, "\n".join(lines), ha="center",
            va="center", fontsize=fs, linespacing=1.55, zorder=4)


def arrow(x0, y0, x1, y1, label=None, lx=0.0, ly=0.13, fs=8.2, rad=0.0,
          color=EC, lw=1.3):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13,
        color=color, lw=lw, connectionstyle=f"arc3,rad={rad}", zorder=5))
    if label:
        ax.text((x0 + x1) / 2 + lx, (y0 + y1) / 2 + ly, label,
                ha="center", fontsize=fs, color="#374151",
                fontstyle="italic", zorder=6)


# ================= bands =================
band(7.35, 9.3, "Stage I — Instrumented inference")
band(4.25, 6.85, "Stage II — Failure localization (registered nulls)")
band(0.8, 3.75, "Stage III — Label-free repair and evaluation")

# ================= Stage I =================
box(1.95, 8.15, 3.1, 1.32, "Probe Construction",
    [r"needle $+$ haystack $+$ $n_d$ distractors",
     r"$L \in \{256,\dots,3840\}$ tokens, depth $\delta$",
     r"forced choice, 8 candidates (chance $0.125$)"], INPUT)
box(6.15, 8.15, 3.4, 1.32, "MoE Language Model (frozen)",
    [r"OLMoE $16{\times}64$  /  Granite $32{\times}40$, top-8",
     "single teacher-forced forward pass",
     "eager attention, verified bit-identical"], MODEL)
box(10.55, 8.15, 3.5, 1.32, "Instrumented Captures",
    [r"routing:  lift$(e,d)=\log_2 P(e|d)/P(e)$",
     r"residual stream:  $h^{(\ell)}_t$ at source / readout",
     r"attention:  $a_{\ell,h}=\Sigma_{k\in\mathrm{needle}}A_{\ell,h}(q_{\rm final},k)$"],
    MEAS, fs=8.4)

arrow(3.5, 8.15, 4.45, 8.15)
arrow(7.85, 8.15, 8.8, 8.15)

# ================= Stage II =================
box(2.25, 5.35, 3.7, 1.7, "1.  Storage Check",
    [r"L8 linear probe at needle position:",
     r"acc $0.995$–$1.000$, incl. every failing prompt",
     r"readout position degrades: $0.714$ vs $0.960$",
     r"$\Rightarrow$ fact stored; transport fails"], LOC, fs=8.4)
box(6.35, 5.35, 3.6, 1.7, "2.  Head Identification",
    [r"rank all $(\ell,h)$ by $\bar{a}_{\ell,h}$ on short",
     r"correct prompts; take top $K{=}16$",
     r"peak cell L12H14: $0.670$ needle mass",
     r"($14.3\times$ chance); $13/16$ in L9–14"], LOC, fs=8.4)
box(10.45, 5.35, 3.6, 1.7, "3.  Collapse Test",
    [r"same 16 cells on failing long prompts:",
     r"$0.432 \rightarrow 0.187$  ($d=1.55$)",
     r"specificity vs random 16-head sets:",
     r"$p<0.0005$ — localized, not diffuse"], LOC, fs=8.4)

arrow(10.55, 7.47, 10.45, 6.22)
arrow(6.15, 7.47, 6.35, 6.22)
arrow(1.95, 7.47, 2.25, 6.22)
arrow(4.1, 5.35, 4.53, 5.35)
arrow(8.15, 5.35, 8.63, 5.35)
ax.text(8.39, 4.85, "same 16 cells", ha="center", fontsize=8.2,
        color="#374151", fontstyle="italic", zorder=6)

# ================= Stage III =================
box(2.45, 1.95, 4.1, 1.75, "Span Detector (no labels)",
    [r"IDF overlap: $s(w)=\Sigma_{t\in Q\cap w}\,1/\mathrm{df}(t)$,",
     r"$\hat{S}=\arg\max_w s(w)$ — zero forward passes",
     r"hit rate $100\%$ on lexical substrates",
     "fallback for multi-hop: L8 residual probe",
     r"($90.6\%$ hit, needs small labeled dev set)"], REPAIR, fs=8.3)
box(7.05, 1.95, 4.0, 1.75, "Attention Boost  (Eq. 2)",
    [r"$\mathrm{logit}'_{\ell,h}(q,k)=\mathrm{logit}_{\ell,h}(q,k)+\beta\,\mathbf{1}[k\in\hat{S}]$",
     r"applied at the 16 identified heads only",
     r"$\beta$ calibrated on dev bucket, frozen",
     "controls: wrong-span, random-heads,",
     "oracle-span ceiling, no-boost floor"], REPAIR, fs=8.3)
box(11.05, 1.95, 2.7, 1.75, "Evaluation",
    [r"repairs $14/14$, $10/10$, $5/5$",
     r"failing prompts",
     r"$99$–$101\%$ of oracle effect",
     r"perm. $p<0.0005$",
     r"$|d_z|\geq 0.8$ vs controls"], EVAL, fs=8.3)

arrow(2.25, 4.48, 2.45, 2.85)
arrow(10.45, 4.48, 7.6, 2.85, rad=-0.06)
arrow(4.5, 1.95, 5.03, 1.95, label=r"$\hat{S}$", ly=0.18)
arrow(9.05, 1.95, 9.68, 1.95)

fig.savefig(OUT / "fig0_architecture.png")
fig.savefig(OUT / "fig0_architecture.pdf")
print("wrote", OUT / "fig0_architecture.png")
