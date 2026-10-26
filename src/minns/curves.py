"""ROC curves and confusion matrices for chosen configs of a probe / MIL run.

Nothing is re-trained: it reads the run's out-of-fold probabilities (<stem>_oof.npz,
next to the results CSV), rebuilds the same folds, and thresholds every fold
exactly as evaluation.run_cv does (F2-best threshold on the OTHER folds), so the
confusion matrix is the one behind the logged recall / specificity / PPV.

  ROC          one curve per config over all out-of-fold predictions (its area is
               `oof_auc` in the CSV), with the dot at the confusion matrix's
               operating point.
  confusion    counts summed over the folds; the shading and the percentage are
               per actual class, so the diagonal reads recall (top) and specificity
               (bottom).

    python -m minns.curves outputs/results/stage_b/probe_egemaps+prosody_random_C0.01.csv
    python -m minns.curves <csv> --top 4
    python -m minns.curves <csv> --configs "egemaps+prosody|utt|L0|max" "egemaps+prosody|full|L0|mean"

Writes <stem>_curves.png beside the CSV and prints each confusion matrix.
Single-repeat runs only (the default 5-fold CV), the only ones whose out-of-fold
vector maps back to one threshold per participant.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from minns.data import load_bucket
from minns.evaluation import best_threshold, cv_folds

# Reference data-viz palette: categorical slots 1-4 in fixed order, blue sequential
# ramp for the matrix shading, text inks and the light chart surface.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95")
INK, INK_2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e4e3df"


def operating_point(y: np.ndarray, prob: np.ndarray, folds, metric: str) -> np.ndarray:
    """Confusion matrix [[TP, FN], [FP, TN]] summed over folds, each fold thresholded on the others."""
    cm, rows = np.zeros((2, 2), int), np.arange(len(y))
    for _, va in folds:
        others = np.setdiff1d(rows, va)
        thr = best_threshold(y[others], prob[others], metric)
        pred = prob[va] >= thr
        yt = y[va] == 1
        cm += [[(pred & yt).sum(), (~pred & yt).sum()], [(pred & ~yt).sum(), (~pred & ~yt).sum()]]
    return cm


def short(config: str, configs: list[str]) -> str:
    """Drop the parts every chosen config shares, e.g. the encoder."""
    parts = [c.split("|") for c in configs]
    keep = [i for i in range(len(parts[0])) if len({p[i] for p in parts}) > 1] or [len(parts[0]) - 1]
    return " · ".join(config.split("|")[i] for i in keep)


def plot(path: Path, title: str, y: np.ndarray, curves: list[tuple[str, np.ndarray, np.ndarray]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "axes.edgecolor": GRID})
    n = len(curves)
    fig = plt.figure(figsize=(max(9, 3.2 * n), 8.6), facecolor=SURFACE)
    gs = fig.add_gridspec(2, n, height_ratios=(1.45, 1), hspace=0.45, wspace=0.35)
    fig.suptitle(title, x=0.06, ha="left", fontsize=12, weight="bold")

    roc = fig.add_subplot(gs[0, :])
    roc.set_facecolor(SURFACE)
    roc.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (4, 3)), label="chance (AUC 0.50)")
    for (label, prob, cm), color in zip(curves, SERIES):
        fpr, tpr, _ = roc_curve(y, prob)
        roc.plot(fpr, tpr, color=color, lw=2, label=f"{label}  (AUC {roc_auc_score(y, prob):.3f})")
        (tp, fn), (fp, tn) = cm
        roc.plot(fp / (fp + tn), tp / (tp + fn), "o", ms=9, color=color, mec=SURFACE, mew=2, zorder=3)
    roc.set(xlim=(0, 1), ylim=(0, 1.01), aspect="equal",
            xlabel="False positive rate (1 − specificity)", ylabel="True positive rate (recall)")
    roc.grid(color=GRID, lw=0.8)
    roc.spines[["top", "right"]].set_visible(False)
    roc.legend(loc="lower right", frameon=False, fontsize=9, title="dot = operating point (F2 threshold)",
               title_fontsize=8.5, alignment="left")

    cmap = LinearSegmentedColormap.from_list("seq", RAMP)
    for i, ((label, _, cm), color) in enumerate(zip(curves, SERIES)):
        ax = fig.add_subplot(gs[1, i])
        share = cm / cm.sum(1, keepdims=True)
        ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
        for r in range(2):
            for c in range(2):
                ax.text(c, r, f"{cm[r, c]}\n{share[r, c]:.0%}", ha="center", va="center", fontsize=11,
                        color=SURFACE if share[r, c] > 0.55 else INK)
        ax.set_xticks([0, 1], ["pred +", "pred −"])
        ax.set_yticks([0, 1], ["actual +", "actual −"])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(label, fontsize=9.5, color=INK, pad=8, loc="left", x=0.09)
        ax.add_patch(matplotlib.patches.Rectangle((0, 1.035), 0.06, 0.06, color=color, clip_on=False,
                                                  transform=ax.transAxes))
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, help="a probe_*.csv or mil_*.csv results file")
    ap.add_argument("--configs", nargs="+", help="config names (the CSV's first column); default: --top by fold AUC")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--threshold", default="f2", help="threshold rule the run used (default f2)")
    args = ap.parse_args()

    summary = pd.read_csv(args.csv, index_col=0)
    configs = args.configs or summary.sort_values("auc", ascending=False).index[:args.top].tolist()
    if len(configs) > len(SERIES):
        raise SystemExit(f"at most {len(SERIES)} configs per figure")
    oof_path = args.csv.with_name(args.csv.stem + "_oof.npz")
    z = np.load(oof_path, allow_pickle=True)    # our own run's file; older runs saved ids as objects
    names = z["config"].tolist()
    missing = [c for c in configs if c not in names]
    if missing:
        raise SystemExit(f"not in {oof_path.name}: {missing}")

    cv = re.search(r"_(random|site)(_|$)", args.csv.stem)
    if not cv or re.search(r"\dx\d", args.csv.stem):
        raise SystemExit("need a single-repeat run whose file name says _random_ or _site_")
    df = load_bucket("finetune")
    train = df[df["split"] == "train"].reset_index(drop=True)
    if train["participant_id"].tolist() != z["participant_id"].tolist():
        raise SystemExit("participants differ from the run's: the bucket changed since it was written")
    y = train["label"].to_numpy()
    folds = cv_folds(train, group_field="meta_site" if cv.group(1) == "site" else None)

    curves = []
    for c in configs:
        prob = z["prob"][names.index(c)]
        cm = operating_point(y, prob, folds, args.threshold)
        (tp, fn), (fp, tn) = cm
        print(f"\n{c}   oof AUC {roc_auc_score(y, prob):.3f}")
        print(f"               pred +   pred -\n   actual +   {tp:6d}   {fn:6d}    recall      {tp / (tp + fn):.3f}"
              f"\n   actual -   {fp:6d}   {tn:6d}    specificity {tn / (tn + fp):.3f}"
              f"\n   PPV {tp / max(tp + fp, 1):.3f}   NPV {tn / max(tn + fn, 1):.3f}")
        curves.append((short(c, configs), prob, cm))

    out = args.csv.with_name(args.csv.stem + "_curves.png")
    title = f"{args.csv.stem}: {len(y)} participants, {int(y.sum())} positive, {cv.group(1)} CV"
    plot(out, title, y, curves)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
