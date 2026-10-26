"""Builds report/MINNs_speech_summary.pdf: a plain-language, visual summary of the work so far.

    uv run --no-project --with matplotlib python report/make_report.py

The numbers are typed in from the cluster logs and result CSVs (5-fold CV on the 550 train
participants, 2026-10-20/21). Every "best" result is the best combination by fold AUC, chosen
separately for each split. The 124 test participants have not been scored yet.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent / "MINNs_speech_summary.pdf"

INK, MUTED, GRID = "#1f2933", "#616e7c", "#d9dee4"
BLUE, ORANGE, GREEN, RED, GREY, PURPLE = "#2f6db5", "#e08a2c", "#3a9a5b", "#c4473a", "#9aa5b1", "#7b5ea7"
PAGE = (11.69, 8.27)  # A4 landscape, inches
FLAG_ALL_F1 = 0.21    # F1 of calling everyone positive: 2p/(1+p) with p = 63/550

STRAT, HELD = "Stratified split (known sites)", "Held-out-site split (unseen sites)"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ------------------------------------------------------------------ results
# (auc, within-site auc, f1, setting)
BASE = {
    "site + language": {"strat": (0.619, 0.469, 0.219, ""), "held": (0.558, 0.542, 0.191, "")},
    "age, gender,\neducation": {"strat": (0.613, 0.600, 0.232, ""), "held": (0.626, 0.585, 0.152, "")},
    "earlier pipeline\n(Whisper)": {"strat": (0.611, 0.555, 0.203, ""), "held": (0.567, 0.551, 0.198, "")},
    "audio quality": {"strat": (0.462, 0.422, 0.200, ""), "held": (0.473, 0.452, 0.205, "")},
}
ENC = {  # probe, best layer x window x pooling
    "Whisper-medium": {"strat": (0.672, 0.617, 0.287, "L23 · 5 s · max"),
                       "held": (0.665, 0.647, 0.225, "L24 · 2 s · max")},
    "HuBERT-large": {"strat": (0.657, 0.606, 0.297, "L23 · 20 s · max"),
                     "held": (0.641, 0.595, 0.207, "L24 · whole · mean")},
    "mHuBERT-147": {"strat": (0.640, 0.579, 0.255, "L8 · 2 s · mm10"),
                    "held": (0.638, 0.603, 0.188, "L4 · 2 s · mm100")},
    "XLS-R-300m": {"strat": (0.617, 0.537, 0.223, "L12 · 2 s · p90"),
                   "held": (0.627, 0.591, 0.181, "L4 · 2 s · max")},
}
MIL = {  # Whisper-medium, best of each family
    "probe": ENC["Whisper-medium"],
    "MINN – small net": {"strat": (0.676, 0.642, 0.302, "L24 · 2 s · mi-lse"),
                                     "held": (0.630, 0.621, 0.174, "L24 · 2 s · mi-lse")},
    "MINN – linear": {"strat": (0.664, 0.622, 0.261, "L23 · 5 s · max"),
                      "held": (0.656, 0.644, 0.218, "L24 · 2 s · lse10")},
    "demographics": BASE["age, gender,\neducation"],
}
# AUC at the layer that was best overall: best window with max, best window with mean, whole-recording mean
POOL = {
    "Whisper": {"strat": ("L23", 0.672, 0.640, 0.607), "held": ("L24", 0.665, 0.600, 0.500)},
    "HuBERT-L": {"strat": ("L23", 0.657, 0.620, 0.608), "held": ("L24", 0.627, 0.620, 0.641)},
    "mHuBERT": {"strat": ("L8", 0.636, 0.623, 0.626), "held": ("L4", 0.637, 0.613, 0.602)},
    "XLS-R": {"strat": ("L12", 0.607, 0.607, 0.570), "held": ("L4", 0.627, 0.575, 0.550)},
}


# ------------------------------------------------------------------ drawing helpers

def new_page(title: str, subtitle: str = "", n: int = 0):
    fig = plt.figure(figsize=PAGE)
    fig.text(0.05, 0.93, title, fontsize=21, weight="bold")
    if subtitle:
        fig.text(0.05, 0.885, subtitle, fontsize=11.5, color=MUTED)
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.865, 0.865], color=GRID, lw=1))
    if n:
        fig.text(0.95, 0.025, f"{n}", fontsize=9, color=MUTED, ha="right")
    fig.text(0.05, 0.025, "MINNs – speech screening for depression / anxiety", fontsize=9, color=MUTED)
    return fig


def box(fig, x, y, w, h, text="", color=BLUE, fontsize=10.5, title="", ha="left"):
    for fc, ec, a in ((color, "none", 0.10), ("none", color, 1.0)):
        fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                      transform=fig.transFigure, fc=fc, ec=ec, alpha=a, lw=1.3))
    if title:
        fig.text(x + 0.012, y + h - 0.012, title, fontsize=fontsize + 1, weight="bold", color=color, va="top")
    if text:
        tx = x + w / 2 if ha == "center" else x + 0.012
        ty = y + h / 2 if not title else y + h - 0.05
        fig.text(tx, ty, text, ha=ha, va="center" if not title else "top", fontsize=fontsize, linespacing=1.45)


def arrow(fig, x0, y0, x1, y1, color=MUTED):
    fig.add_artist(FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=16, color=color, lw=1.6))


def style(ax, ylim):
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def label_bars(ax, pos, vals, size=8.5):
    for p, v in zip(pos, vals):
        ax.text(p, v + 0.004, f"{v:.2f}", ha="center", va="bottom", fontsize=size)


def split_grid(fig, table: dict, colors=None, auc_ylim=(0.4, 0.72), demo_line=False, left=0.06, width=0.42,
               gap=0.06, top=0.80, label_size=8.5):
    """Two columns (stratified | held-out-site); top row AUC + within-site AUC, bottom row F1."""
    names = list(table)
    x = np.arange(len(names))
    for col, (key, title) in enumerate((("strat", STRAT), ("held", HELD))):
        x0 = left + col * (width + gap)
        a1 = fig.add_axes([x0, top - 0.33, width, 0.33])
        a2 = fig.add_axes([x0, top - 0.33 - 0.25, width, 0.17])
        auc = [table[n][key][0] for n in names]
        site = [table[n][key][1] for n in names]
        f1 = [table[n][key][2] for n in names]
        a1.bar(x - 0.2, auc, 0.38, color=BLUE, label="AUC")
        a1.bar(x + 0.2, site, 0.38, color=GREEN, label="within-site AUC")
        label_bars(a1, x - 0.2, auc)
        label_bars(a1, x + 0.2, site)
        a1.axhline(0.5, color=RED, lw=1, ls="--", label="chance (0.50)")
        if demo_line:
            d = BASE["age, gender,\neducation"][key][1]
            a1.axhline(d, color=ORANGE, lw=1.2, ls=":", label=f"demographics within-site ({d:.2f})")
        style(a1, auc_ylim)
        a1.set_xticks(x, [""] * len(names))
        a1.set_title(title, loc="left", fontsize=12, weight="bold", pad=22)
        a1.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 0.99), ncol=4,
                  handlelength=1.5, columnspacing=1)
        a2.bar(x, f1, 0.5, color=colors or PURPLE)
        label_bars(a2, x, f1)
        a2.axhline(FLAG_ALL_F1, color=RED, lw=1, ls="--", label=f"F1 of flagging everyone ({FLAG_ALL_F1:.2f})")
        a2.legend(frameon=False, fontsize=8, loc="upper right")
        style(a2, (0, 0.45))
        a2.set_ylabel("F1", fontsize=9)
        settings = [table[n][key][3] for n in names]
        a2.set_xticks(x, [f"{n}\n{s}" if s else n for n, s in zip(names, settings)], fontsize=label_size)


# ------------------------------------------------------------------ pages

def page_title(pdf):
    fig = plt.figure(figsize=PAGE)
    fig.text(0.07, 0.80, "Can we hear depression and anxiety\nin how people speak?", fontsize=30,
             weight="bold", linespacing=1.3)
    fig.text(0.07, 0.68, "A plain-language summary of the MINNs speech experiments  ·  October 2026",
             fontsize=13, color=MUTED)
    box(fig, 0.07, 0.40, 0.86, 0.22, color=BLUE, fontsize=12.5, text=
        "We took the voice recordings of the participants, cut each recording into many short\n"
        "windows, and used large pre-trained speech models to turn every window into a list of\n"
        "numbers. Small classifiers then learned which windows go with a positive screen. We\n"
        "compared 4 speech models, window lengths, ways of combining windows, and the\n"
        "multiple-instance networks (MINN) from Alsarrani et al. 2025.")
    items = [("Ranking (AUC)", "best ≈ 0.67 on known sites\n≈ 0.66 on unseen sites", GREEN),
             ("Better than", "the earlier pipeline (0.57–0.61)\nand age/gender/education (0.60)", BLUE),
             ("But", "F1 only 0.18–0.30 (flag-everyone 0.21);\nMINNs add little over the probe", ORANGE)]
    for i, (h, t, c) in enumerate(items):
        x = 0.07 + i * 0.295
        box(fig, x, 0.14, 0.27, 0.18, color=c)
        fig.text(x + 0.135, 0.275, h, ha="center", fontsize=12, weight="bold", color=c)
        fig.text(x + 0.135, 0.20, t, ha="center", va="center", fontsize=10.8, linespacing=1.4)
    fig.text(0.07, 0.06, "Papers followed: Alsarrani et al., ICMI 2025 (MINN, \"punctual vs continuous\");  "
             "de Gennes et al. 2024 (probing speech foundation models).", fontsize=9, color=MUTED)
    pdf.savefig(fig)
    plt.close(fig)


def page_data(pdf):
    fig = new_page("1. The data and how we test", "Same participants, labels and splits as the mh-ML-pipeline", 2)
    ax = fig.add_axes([0.06, 0.62, 0.88, 0.13])
    ax.barh(0, 550, color=BLUE, height=0.55)
    ax.barh(0, 124, left=550, color=GREY, height=0.55)
    ax.text(275, 0, "550 participants used for building and testing ideas (cross-validation)",
            ha="center", va="center", color="white", fontsize=11, weight="bold")
    ax.text(612, 0, "124 locked\ntest set", ha="center", va="center", color="white", fontsize=10, weight="bold")
    ax.set_xlim(0, 674)
    ax.axis("off")
    ax.set_title("674 participants with a voice recording and a screening label", loc="left", fontsize=12)
    fig.text(0.06, 0.565, "63 of the 550 (11%) screen positive.  The 124 test participants (14 positive) have NOT "
             "been looked at – they are saved for one final check.", fontsize=10.5, color=MUTED)

    fig.text(0.06, 0.49, "5-fold cross-validation", fontsize=13, weight="bold")
    for k in range(5):
        y = 0.42 - k * 0.052
        fig.text(0.06, y + 0.012, f"round {k + 1}", fontsize=9.5, color=MUTED)
        for j in range(5):
            fig.add_artist(FancyBboxPatch((0.13 + j * 0.06, y), 0.054, 0.035,
                                          boxstyle="round,pad=0.002,rounding_size=0.006", transform=fig.transFigure,
                                          fc=ORANGE if j == k else BLUE, ec="none", alpha=0.85))
    fig.text(0.13, 0.135, "■ learn from these", color=BLUE, fontsize=10)
    fig.text(0.26, 0.135, "■ score this part", color=ORANGE, fontsize=10)
    fig.text(0.06, 0.095, "Every participant is scored once, by a model that never saw them.", fontsize=10)

    box(fig, 0.47, 0.30, 0.47, 0.22, title="Two ways of making the 5 folds", fontsize=10, color=BLUE, text=
        "Stratified split (known sites): every fold has the same mix of label,\n"
        "cohort, illness, site, language, education and age. Each site appears\n"
        "on both sides → new people at familiar sites.\n\n"
        "Held-out-site split (unseen sites): whole sites are kept out of\n"
        "learning, folds balanced on the label only → a site never seen before.")
    box(fig, 0.47, 0.07, 0.47, 0.20, title="Three scores on every chart", fontsize=10, color=GREEN, text=
        "AUC – ranking quality, no threshold: 0.5 = coin flip, 1.0 = perfect.\n"
        "Within-site AUC – the same, but people only compared with others from\n"
        "their own site, so site differences can't help (see page 3).\n"
        "F1 – after picking a yes/no cut-off (tuned for F2 on the other folds).\n"
        "Calling everyone positive already gives F1 ≈ 0.21.")
    pdf.savefig(fig)
    plt.close(fig)


def page_baselines(pdf):
    fig = new_page("2. First, a trap to avoid: \"cheating\" signals",
                   "Stage 0 – how well can we do WITHOUT the new speech models?", 3)
    split_grid(fig, BASE, colors=GREY)
    box(fig, 0.06, 0.055, 0.88, 0.11, color=ORANGE, fontsize=10, text=
        "Some sites simply have more positive participants: site + language alone gets AUC 0.62 without hearing a word.\n"
        "Within-site AUC removes this (0.47). Age, gender and education are a real bar: within-site 0.60 / 0.59\n"
        "(women screen positive about twice as often, 15% vs 7%). A speech model has to beat ≈ 0.60, not 0.50.")
    pdf.savefig(fig)
    plt.close(fig)


def page_pipeline(pdf):
    fig = new_page("3. From a recording to window embeddings", "The same for every speech model", 4)
    steps = [("Recording", "the full interview\naudio", GREY),
             ("Speech model", "frozen, pre-trained\n(Whisper, HuBERT …);\nreads 30 s at a time", BLUE),
             ("Every layer", "13–25 layers, each gives\none vector per 20 ms\n(768–1024 numbers)", BLUE),
             ("Windows", "average the 20 ms vectors\ninside each 2/5/10/20 s\nwindow (half overlap)", ORANGE),
             ("Window embedding", "one vector per window,\nper layer – e.g. up to\n~150 windows of 5 s", GREEN)]
    w, h, y = 0.165, 0.20, 0.56
    for i, (t, d, c) in enumerate(steps):
        x = 0.05 + i * 0.185
        box(fig, x, y, w, h, color=c)
        fig.text(x + w / 2, y + h - 0.035, t, ha="center", fontsize=11.5, weight="bold", color=c)
        fig.text(x + w / 2, y + 0.075, d, ha="center", va="center", fontsize=9.3, linespacing=1.35)
        if i:
            arrow(fig, x - 0.019, y + h / 2, x - 0.002, y + h / 2)
    box(fig, 0.05, 0.12, 0.42, 0.33, title="\"Whole recording\" (what the earlier pipeline did)", color=GREY,
        fontsize=10.5, text=
        "Instead of windows, average ALL 20 ms vectors of the\n"
        "recording into one vector per person, then train a\n"
        "classifier on that one vector.\n\n"
        "Simple, but any short moment that matters is diluted\n"
        "by minutes of ordinary speech.")
    box(fig, 0.53, 0.12, 0.42, 0.33, title="How \"best\" is chosen", color=BLUE, fontsize=10.5, text=
        "For each speech model we tried every combination of\n"
        "layer × window length × pooling and kept the one with\n"
        "the highest AUC – layer, window and pooling are chosen\n"
        "together, separately for each split.\n\n"
        "Labels like \"L23 · 5 s · max\" under bars name that choice.\n"
        "Best-of-many is slightly optimistic.")
    pdf.savefig(fig)
    plt.close(fig)


def page_probe_vs_minn(pdf):
    fig = new_page("4. Two ways to learn from windows: probe vs MINN",
                   "Both use the same frozen window embeddings – they differ in how they are trained", 5)
    rows = [("Probe", BLUE,
             "Every window gets its person's label\n(a positive person's windows all count as positive).\n"
             "A logistic regression learns to score single windows.\nPooling plays no part in training.",
             "Window scores are combined into one score per person.\n"
             "We try several poolings on the same trained model:\nmax, mean, min, p90, vote, mellowmax (mm)."),
            ("MINN\n(multiple-instance\nnetwork)", GREEN,
             "Window scorer → pooling → one score per person,\ntrained together on the person's label only.\n"
             "The model can learn that only some windows matter.\nScorer: linear, or a small neural network (32–128 units).",
             "The pooling it was trained with\n(max, mean or lse – a soft maximum).\n"
             "Variants: mi-Net (score windows, then pool),\nMI-Net / attention (pool features, then score).")]
    fig.text(0.29, 0.80, "Training", fontsize=13, weight="bold", ha="center")
    fig.text(0.72, 0.80, "Predicting", fontsize=13, weight="bold", ha="center")
    for i, (name, c, train, pred) in enumerate(rows):
        y = 0.55 - i * 0.25
        box(fig, 0.05, y, 0.11, 0.20, color=c)
        fig.text(0.105, y + 0.10, name, ha="center", va="center", fontsize=10, weight="bold", color=c)
        box(fig, 0.18, y, 0.38, 0.20, color=c, fontsize=9.8, text=train)
        box(fig, 0.59, y, 0.36, 0.20, color=c, fontsize=9.8, text=pred)
    box(fig, 0.05, 0.08, 0.90, 0.13, color=ORANGE, fontsize=10.3, text=
        "\"Probe + max\" and \"linear MINN + max\" have exactly the same form – one weighted sum per window, then the "
        "maximum.\nOnly the training differs, so comparing them tests one question: does learning through the pooling "
        "help? (page 8)\nWhy pooling matters: a sign of low mood may appear only in a few moments (\"punctual\"); "
        "an average hides them, the maximum keeps them.")
    pdf.savefig(fig)
    plt.close(fig)


def page_encoders(pdf):
    fig = new_page("5. Which speech model hears it best?",
                   "Stage B – probe on each speech model, best layer · window · pooling per split", 6)
    split_grid(fig, ENC, demo_line=True)
    box(fig, 0.06, 0.055, 0.88, 0.11, color=BLUE, fontsize=10, text=
        "Whisper-medium is best on both splits, with one of its last layers (23–24) and max pooling, and is the only "
        "model\nabove the demographics bar within site on both splits. XLS-R's middle layers (15–18) fall below chance on "
        "unseen sites.\nF1 is low everywhere (0.18–0.30); on unseen sites it is barely above flagging everyone (0.21).")
    pdf.savefig(fig)
    plt.close(fig)


def page_pooling(pdf):
    fig = new_page("6. Does keeping the strongest windows help?",
                   "AUC of the probe at each model's best layer: max vs mean over windows vs one whole-recording average",
                   7)
    for col, (key, title) in enumerate((("strat", STRAT), ("held", HELD))):
        ax = fig.add_axes([0.06 + col * 0.48, 0.30, 0.42, 0.48])
        names = list(POOL)
        x = np.arange(len(names))
        for j, (lab, c) in enumerate((("max over windows", BLUE), ("mean over windows", GREEN),
                                      ("whole-recording average", GREY))):
            vals = [POOL[n][key][1 + j] for n in names]
            ax.bar(x - 0.27 + j * 0.27, vals, 0.25, color=c, label=lab)
            label_bars(ax, x - 0.27 + j * 0.27, vals, size=7.5)
        ax.axhline(0.5, color=RED, lw=1, ls="--")
        style(ax, (0.45, 0.72))
        ax.set_xticks(x, [f"{n}\n{POOL[n][key][0]}" for n in names])
        ax.set_title(title, loc="left", fontsize=12, weight="bold")
        ax.legend(frameon=False, fontsize=8, loc="upper right" if col else "upper left", ncol=1)
    box(fig, 0.06, 0.06, 0.88, 0.15, color=ORANGE, fontsize=10.2, text=
        "• Whisper: max clearly beats mean (+0.03 known, +0.065 unseen sites); its whole-recording average is chance "
        "(0.50) on unseen sites.\n"
        "• HuBERT-L: max helps on known sites (+0.04), but on unseen sites its whole-recording average is its best result "
        "(0.64).\n"
        "• mHuBERT: small gain from max (+0.01 / +0.02).    • XLS-R: no gain on known sites, +0.05 on unseen sites.\n"
        "→ \"Punctual\" signal (max wins) is clear for Whisper, the best model; it is not universal.")
    pdf.savefig(fig)
    plt.close(fig)


def page_mil(pdf):
    fig = new_page("7. Do the multiple-instance networks (MINN) help?",
                   "Stage C – Whisper-medium, best setting of each approach per split", 8)
    split_grid(fig, MIL, colors=[BLUE, GREEN, GREEN, ORANGE], auc_ylim=(0.4, 0.74), label_size=8)
    box(fig, 0.06, 0.055, 0.88, 0.11, color=GREEN, fontsize=10, text=
        "All three land at AUC ≈ 0.63–0.68; the gaps (≤ 0.035) are smaller than the fold-to-fold spread (±0.03 to "
        "±0.12),\nso no MINN is reliably better than the probe. The small network wins slightly on known sites but loses "
        "on unseen sites.\nMI-Net and attention variants fell to ≈ chance on unseen sites; HuBERT-L MINNs were worse than "
        "its probe.")
    pdf.savefig(fig)
    plt.close(fig)


def page_table(pdf):
    fig = new_page("8. Everything in one table", "Each row: best setting per split (5-fold CV on the 550). "
                   "Darker green = better; red = at or below chance / flag-everyone.", 9)
    rows = [("Site + language only", BASE["site + language"], "no speech"),
            ("Age, gender, education", BASE["age, gender,\neducation"], "no speech"),
            ("Audio quality", BASE["audio quality"], "no speech"),
            ("Earlier pipeline (Whisper, whole recording)", BASE["earlier pipeline\n(Whisper)"], "earlier"),
            ("Whisper – probe", ENC["Whisper-medium"], "this project"),
            ("Whisper – MINN small network", MIL["MINN – small net"], "this project"),
            ("Whisper – MINN linear", MIL["MINN – linear"], "this project"),
            ("HuBERT-large – probe", ENC["HuBERT-large"], "this project"),
            ("mHuBERT-147 – probe", ENC["mHuBERT-147"], "this project"),
            ("XLS-R-300m – probe", ENC["XLS-R-300m"], "this project")]
    cols = [("strat", 0, "AUC"), ("strat", 1, "within-\nsite AUC"), ("strat", 2, "F1"),
            ("held", 0, "AUC"), ("held", 1, "within-\nsite AUC"), ("held", 2, "F1")]
    ax = fig.add_axes([0.05, 0.08, 0.90, 0.72])
    ax.set_xlim(0, 10)
    ax.set_ylim(len(rows) + 1.6, 0)
    ax.axis("off")
    cmap = plt.get_cmap("Greens")
    x0, cw = 3.6, 1.06
    ax.text(x0 + 1.5 * cw, -0.15, "Stratified (known sites)", ha="center", fontsize=11, weight="bold", color=BLUE)
    ax.text(x0 + 4.5 * cw, -0.15, "Held-out site (unseen sites)", ha="center", fontsize=11, weight="bold",
            color=BLUE)
    for j, (_, _, name) in enumerate(cols):
        ax.text(x0 + (j + 0.5) * cw, 0.75, name, ha="center", va="center", fontsize=9.5, color=MUTED)
    for i, (name, d, group) in enumerate(rows):
        y = i + 1.5
        ax.text(0.05, y + 0.5, name, va="center", fontsize=10.2, weight="bold" if group == "this project" else None)
        for j, (key, k, _) in enumerate(cols):
            v = d[key][k]
            if k < 2:
                low, good = v <= 0.5, (v - 0.5) / 0.18
            else:
                low, good = v <= FLAG_ALL_F1, (v - FLAG_ALL_F1) / 0.1
            fc = "#f6d5d1" if low else cmap(0.15 + 0.6 * min(max(good, 0), 1))
            ax.add_patch(Rectangle((x0 + j * cw + 0.03, y + 0.06), cw - 0.06, 0.88, fc=fc, ec="none"))
            ax.text(x0 + (j + 0.5) * cw, y + 0.5, f"{v:.2f}", ha="center", va="center", fontsize=10.5,
                    color="white" if (not low and good > 0.75) else INK)
    ax.plot([x0 + 3 * cw, x0 + 3 * cw], [0.3, len(rows) + 1.5], color=GRID, lw=1.5)
    ax.plot([0, 10], [4.5, 4.5], color=GRID, lw=1)
    ax.plot([0, 10], [5.5, 5.5], color=GRID, lw=1)
    fig.text(0.05, 0.05, "Bold rows = this project. F1 shaded red when ≤ 0.21 (no better than calling everyone "
             "positive). Each row's setting is the best of many, so values are slightly optimistic.",
             fontsize=9, color=MUTED)
    pdf.savefig(fig)
    plt.close(fig)


def page_next(pdf):
    fig = new_page("9. Honest caveats and next steps", "", 10)
    box(fig, 0.05, 0.40, 0.42, 0.42, title="Caveats", color=ORANGE, fontsize=10.5, text=
        "• Small numbers: about 13 positive people per fold,\n"
        "  so AUC moves ±0.03–0.12 between folds.\n\n"
        "• We tried many settings and report the best;\n"
        "  the true performance is probably a bit lower.\n\n"
        "• F1 is low (0.18–0.30). As a yes/no screener the\n"
        "  model is not usable yet – AUC shows a real but\n"
        "  weak signal.\n\n"
        "• Speech may partly pick up gender, which is\n"
        "  linked to the label – not yet ruled out.\n\n"
        "• Nothing has been checked on the 124 test people.")
    box(fig, 0.53, 0.40, 0.42, 0.42, title="Suggested next steps", color=GREEN, fontsize=10.5, text=
        "1. Speech vs demographics: does Whisper add\n"
        "   anything on top of age, gender and education?\n\n"
        "2. Remove site effects from the speech features\n"
        "   (normalise within each site) before scoring.\n\n"
        "3. Pick ONE final model with a rule fixed in advance,\n"
        "   retrain on all 550 people and score the 124\n"
        "   test participants once.")
    fig.text(0.05, 0.32, "Glossary", fontsize=13, weight="bold")
    fig.text(0.05, 0.29,
             "Window (instance) – one short slice of a recording.     Bag – all windows of one person.\n"
             "Layer – one processing stage inside the speech model (L23 = layer 23).     "
             "Embedding – the list of numbers describing a window.\n"
             "Probe – a simple classifier on frozen embeddings, trained window by window.     "
             "MINN / MIL – learns from whole bags labelled only as a whole.\n"
             "Pooling – how window scores become one person score: max, mean, min, p90 (90th percentile), "
             "vote, mm/lse (soft maximum).\n"
             "mi-Net – score each window, then pool.     MI-Net / attention – pool the window features, then score.",
             fontsize=9.8, va="top", linespacing=1.75)
    fig.text(0.05, 0.10, "Code: src/minns/ (extract, probe, mil, baseline, evaluation)  ·  Cluster jobs: "
             "scripts/cluster_{extract,probe,mil}.sh  ·  Results: ~/MIL/outputs/results/ on the cluster",
             fontsize=9, color=MUTED)
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    with PdfPages(OUT) as pdf:
        for page in (page_title, page_data, page_baselines, page_pipeline, page_probe_vs_minn, page_encoders,
                     page_pooling, page_mil, page_table, page_next):
            page(pdf)
        pdf.infodict()["Title"] = "MINNs speech screening – summary"
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
