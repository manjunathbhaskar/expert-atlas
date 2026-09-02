"""Publication figures for the TechRxiv report.

Every number is taken verbatim from the committed docs:
CONTEXT_ROT_HARD.md, ATTENTION_TRANSPORT.md, ATTENTION_BOOST_CAUSAL.md,
SPAN_DISCOVERY_SOLVED.md, SPANFREE_BOOST.md, GRANITE_TRANSPORT.md,
CONTEXT_VARIANTS.md, PROBE_REPAIRED.md, CONTEXT_DEPTH.md.

Usage: .venv/bin/python scripts/make_paper_figures.py
Writes PNG (300 dpi) + PDF to figures/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent.parent / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})

BLUE = "#2b6cb0"
RED = "#c53030"
GREEN = "#2f855a"
GRAY = "#718096"
ORANGE = "#dd6b20"
PURPLE = "#6b46c1"
LIGHT = "#e2e8f0"


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


# ---------------------------------------------------------------- Figure 1
# Causal-chain mechanism diagram
def fig1():
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    def box(x, y, w, h, title, lines, fc, ec, title_c="black"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    fc=fc, ec=ec, lw=1.4))
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top",
                fontsize=9.5, fontweight="bold", color=title_c)
        ax.text(x + w / 2, y + h - 0.62, "\n".join(lines), ha="center",
                va="top", fontsize=7.8, color="#1a202c", linespacing=1.4)

    def arrow(x0, y0, x1, y1, color="#2d3748", style="-|>", lw=1.6,
              label=None, lx=0, ly=0.14, lcolor="#2d3748"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                     mutation_scale=14, color=color, lw=lw))
        if label:
            ax.text((x0 + x1) / 2 + lx, (y0 + y1) / 2 + ly, label,
                    ha="center", fontsize=7.5, color=lcolor, fontstyle="italic")

    # top row: the failure chain
    box(0.15, 2.6, 2.2, 1.7, "1. Storage intact",
        ["needle content decodable", "at its source position",
         "L8 probe acc 0.995\u20131.000", "incl. every failing prompt"],
        "#ebf8ff", BLUE)
    box(2.85, 2.6, 2.4, 1.7, "2. Transport collapses",
        ["16 retrieval heads (of 256)", "needle attention on failing",
         "prompts: 0.432 \u2192 0.187", "localized: spec. p<0.0005"],
        "#fff5f5", RED)
    box(5.75, 2.6, 2.2, 1.7, "3. Readout degrades",
        ["answer-position decodability", "0.714 (wrong) vs 0.960 (right)",
         "accuracy 0.938 \u2192 0.688", "(256 \u2192 3,840 tokens)"],
        "#fffaf0", ORANGE)
    box(8.45, 2.6, 1.45, 1.7, "Symptom",
        ["router specialist", "starvation", "partial \u03c1 = 0.64",
         "(not repairable)"], "#faf5ff", PURPLE)

    arrow(2.42, 3.45, 2.80, 3.45)
    arrow(5.32, 3.45, 5.70, 3.45)
    arrow(8.02, 3.45, 8.40, 3.45, color=GRAY, style="->")

    # bottom row: the repair
    box(1.0, 0.25, 3.1, 1.5, "Label-free span detection",
        ["IDF-weighted lexical overlap,", "question vs context windows",
         "no forward pass \u00b7 hit rate 100%", "on all needle substrates"],
        "#f0fff4", GREEN)
    box(5.4, 0.25, 3.1, 1.5, "Attention boost repair",
        ["pre-softmax bias \u03b2 at the 16 heads", "onto the detected span",
         "OLMoE 14/14, Granite 5/5 repaired", "99\u2013101% of oracle effect"],
        "#f0fff4", GREEN)

    arrow(4.16, 1.0, 5.34, 1.0, color=GREEN, label="detected span",
          ly=0.16, lcolor=GREEN)
    arrow(6.95, 1.82, 4.6, 2.55, color=GREEN, style="-|>", lw=1.8,
          label="reopens the collapsed heads", lx=1.15, ly=-0.05,
          lcolor=GREEN)

    ax.text(0.15, 4.45, "Failure chain (measured)", fontsize=9,
            fontweight="bold", color="#2d3748")
    ax.text(1.0, 1.95, "Repair (causal, controlled)", fontsize=9,
            fontweight="bold", color=GREEN)
    save(fig, "fig1_causal_chain")


# ---------------------------------------------------------------- Figure 2
# (a) degradation curve, (b) collapse contrast on both models
def fig2():
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4))

    buckets = [256, 512, 1024, 2048, 3072, 3840]
    acc_all = [0.938, 0.875, 0.812, 0.781, 0.781, 0.688]
    acc_d0 = [1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
    acc_d8 = [0.88, 0.75, 0.625, 0.565, 0.56, 0.38]
    a.plot(buckets, acc_all, "o-", color=BLUE, lw=2, label="all conditions")
    a.plot(buckets, acc_d0, "s--", color=GRAY, lw=1.4,
           label="0 distractors")
    a.plot(buckets, acc_d8, "^--", color=RED, lw=1.4,
           label="8 distractors (mean)")
    a.axhline(0.125, color="k", lw=0.8, ls=":")
    a.text(3840, 0.06, "chance = 0.125", ha="right", fontsize=7.5)
    a.set_xscale("log", base=2)
    a.set_xticks(buckets)
    a.set_xticklabels(buckets, rotation=45)
    a.set_ylim(0, 1.05)
    a.set_xlabel("context length (tokens)")
    a.set_ylabel("forced-choice accuracy")
    a.set_title("(a) Length degradation, OLMoE")
    a.legend(frameon=False, fontsize=7.5, loc="center left",
             bbox_to_anchor=(0.02, 0.32))

    # collapse contrast
    groups = ["OLMoE\nidentified", "OLMoE\nother",
              "Granite\nidentified", "Granite\nother",
              "Pythia\nidentified", "Pythia\nother"]
    right = [0.432, 0.022, 0.225, 0.0112, 0.421, 0.0248]
    wrong = [0.187, 0.012, 0.101, 0.0068, 0.372, 0.0220]
    x = range(6)
    w = 0.36
    b.bar([i - w / 2 for i in x], right, w, color=BLUE,
          label="model-right prompts")
    b.bar([i + w / 2 for i in x], wrong, w, color=RED,
          label="model-wrong prompts")
    for i, (r, wv) in enumerate(zip(right, wrong)):
        b.text(i - w / 2, r + 0.010, f"{r:.3f}", ha="center", fontsize=7.5)
        b.text(i + w / 2, wv + 0.010, f"{wv:.3f}", ha="center", fontsize=7.5)
    ds = ["d=1.55", "d=1.28", "d=2.17", "d=1.63", "d=0.70†", "d=0.52"]
    for i, d in enumerate(ds):
        b.text(i, max(right[i], wrong[i]) + 0.045, d, ha="center",
               fontsize=8, fontstyle="italic")
    b.set_xticks(list(x))
    b.set_xticklabels(groups, fontsize=8.5)
    b.set_ylabel("mean needle attention (final position)")
    b.set_ylim(0, 0.62)
    b.set_title("(b) Collapse is localized (long bucket, all 3 models)",
                pad=34)
    b.legend(frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, 1.14), ncol=2)
    fig.text(0.99, 0.01, "† significant (perm p = 0.048), specific "
             "(p < 0.0005), but below the 0.8 effect floor — "
             "suggestive, not confirmatory", ha="right", va="bottom",
             fontsize=7.2, color="#4a5568", fontstyle="italic")
    fig.subplots_adjust(bottom=0.22, top=0.80, wspace=0.28)
    save(fig, "fig2_degradation_collapse")


# ---------------------------------------------------------------- Figure 3
# Repair vs controls, three substrates
def fig3():
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    conds = ["baseline", "wrong\nspan", "random\nheads", "lexical\ndetector",
             "oracle\nspan"]
    colors = [GRAY, ORANGE, PURPLE, GREEN, BLUE]

    data = [
        ("OLMoE 3,840 tok (n=64)", [0.678, 0.668, 0.773, 0.984, 0.981],
         "14/14 failing repaired"),
        ("OLMoE depth 0.15 (n=16)", [0.207, 0.214, None, 0.954, 0.959],
         "10/10 failing repaired"),
        ("Granite hard set (n=16)", [0.644, 0.631, 0.561, 0.984, 0.988],
         "5/5 failing repaired"),
        ("Pythia hard set (n=48)", [0.588, 0.580, 0.610, 0.742, 0.892],
         "11/11 failing repaired"),
    ]
    for ax, (title, vals, note) in zip(axes, data):
        xs, vs, cs, ls = [], [], [], []
        for i, v in enumerate(vals):
            if v is None:
                continue
            xs.append(i)
            vs.append(v)
            cs.append(colors[i])
            ls.append(conds[i])
        ax.bar(range(len(vs)), vs, color=cs, width=0.62)
        for j, v in enumerate(vs):
            ax.text(j, v + 0.015, f"{v:.3f}", ha="center", fontsize=7.5)
        ax.set_xticks(range(len(vs)))
        ax.set_xticklabels(ls, fontsize=7.5)
        ax.set_ylim(0, 1.1)
        ax.set_title(title)
        ax.text(0.5, 0.97, note, transform=ax.transAxes, ha="center",
                fontsize=7.5, color=GREEN, fontstyle="italic")
    axes[0].set_ylabel("mean answer probability")
    fig.suptitle("Attention-boost repair vs matched controls "
                 "(full evaluation sets)", fontsize=10, fontweight="bold",
                 y=1.04)
    save(fig, "fig3_repair_controls")


# ---------------------------------------------------------------- Figure 4
# Span detector comparison across substrates
def fig4():
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 3.6))

    # (a) hit rates per detector per substrate
    subs = ["3,840\nneedle", "depth\n0.15", "para-\nphrase", "multi-\nhop",
            "Pythia\nhard"]
    lex = [100, 100, 100, 0, 50]
    attn = [85.9, 37.5, None, None, None]
    l8 = [None, None, None, 90.6, None]
    expert = [0, None, None, None, None]
    x = range(5)
    w = 0.2
    a.bar([i - 1.5 * w for i in x], [v if v is not None else 0 for v in lex],
          w, color=GREEN, label="lexical (training-free)")
    a.bar([i - 0.5 * w for i in x],
          [v if v is not None else 0 for v in attn], w, color=BLUE,
          label="retrieval-head attention")
    a.bar([i + 0.5 * w for i in x],
          [v if v is not None else 0 for v in l8], w, color=PURPLE,
          label="L8 residual probe (labeled dev)")
    a.bar([i + 1.5 * w for i in x],
          [v if v is not None else 0 for v in expert], w, color=GRAY,
          label="needle-affine experts")
    for i, series, off in ((0, lex, -1.5), (1, attn, -0.5), (2, l8, 0.5),
                           (3, expert, 1.5)):
        for j, v in enumerate(series):
            if v is not None:
                a.text(j + off * w, v + 2, f"{v:g}", ha="center", fontsize=7)
    a.set_xticks(list(x))
    a.set_xticklabels(subs, fontsize=8)
    a.set_ylabel("span hit rate (%)")
    a.set_ylim(0, 118)
    a.set_title("(a) Span detectors by substrate", pad=34)
    a.legend(frameon=False, fontsize=7.5, loc="upper center",
             bbox_to_anchor=(0.5, 1.20), ncol=2)
    fig.subplots_adjust(top=0.76, wspace=0.22)

    # (b) fraction of oracle effect recovered by the label-free pipeline
    subs2 = ["OLMoE\n3,840", "OLMoE\ndepth 0.15", "Granite\nhard",
             "para-\nphrase", "multi-hop\nchain", "multi-hop\nL8 fallback",
             "Pythia\nhard"]
    frac = [100.8, 99.4, 99.1, 100.8, 45.0, 61.4, 51.0]
    cs = [GREEN, GREEN, GREEN, GREEN, GREEN, PURPLE, GREEN]
    b.bar(range(7), frac, color=cs, width=0.62)
    for j, v in enumerate(frac):
        b.text(j, v + 2, f"{v:g}%", ha="center", fontsize=7.5)
    b.axhline(100, color="k", lw=0.8, ls=":")
    b.text(6.45, 108, "oracle ceiling", ha="right", fontsize=7.5)
    b.set_xticks(range(7))
    b.set_xticklabels(subs2, fontsize=8)
    b.set_ylabel("% of oracle repair effect")
    b.set_ylim(0, 122)
    b.set_title("(b) Label-free pipeline recovery")
    save(fig, "fig4_detectors")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
