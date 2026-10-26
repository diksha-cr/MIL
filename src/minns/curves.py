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

Compare runs, e.g. averaging vs fixed windows vs utterances (same participants, same CV):

    python -m minns.curves <probe_whisper-medium-utt+prosody csv> <probe_whisper-medium csv> \
        --by-window full w5 utt

`--by-window` takes the best config (fold AUC) of each window. With two or more configs it
also prints a paired bootstrap of the oof AUC difference against the weakest one.

Writes <stem>_curves.png beside the (first) CSV and prints each confusion matrix.
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


def paired_bootstrap(y: np.ndarray, a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0):
    """AUC(a) - AUC(b) on the same participants: (delta, 95% CI, one-sided p of delta <= 0).

    Resamples participants within each class, so every replicate keeps 63 positives."""
    from scipy.stats import rankdata

    def auc(s, n_pos):                       # Mann-Whitney, rows = replicates, positives first
        r = rankdata(s, axis=1)
        return (r[:, :n_pos].sum(1) - n_pos * (n_pos + 1) / 2) / (n_pos * (s.shape[1] - n_pos))

    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    idx = np.hstack([rng.choice(pos, (n, len(pos))), rng.choice(neg, (n, len(neg)))])
    d = auc(a[idx], len(pos)) - auc(b[idx], len(pos))
    return roc_auc_score(y, a) - roc_auc_score(y, b), np.percentile(d, [2.5, 97.5]), float((d <= 0).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, nargs="+", help="probe_*.csv / mil_*.csv results files (same CV)")
    ap.add_argument("--configs", nargs="+", help="config names (the CSV's first column); default: --top by fold AUC")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--by-window", nargs="*", metavar="WINDOW",
                    help="instead of --top, the best config (fold AUC) of each window, e.g. "
                         "--by-window full w5 utt; with no names, every window found")
    ap.add_argument("--threshold", default="f2", help="threshold rule the run used (default f2)")
    args = ap.parse_args()

    summary, probs, ids, cvs = [], {}, None, set()
    for csv in args.csv:
        cv = re.search(r"_(random|site)(_|$)", csv.stem)
        if not cv or re.search(r"\dx\d", csv.stem):
            raise SystemExit(f"{csv.name}: need a single-repeat run whose file name says _random_ or _site_")
        cvs.add(cv.group(1))
        z = np.load(csv.with_name(csv.stem + "_oof.npz"), allow_pickle=True)  # our own runs; older ones saved ids as objects
        if ids is not None and z["participant_id"].tolist() != ids:
            raise SystemExit(f"{csv.name}: different participants from {args.csv[0].name}")
        ids = z["participant_id"].tolist()
        probs.update(zip(z["config"].tolist(), z["prob"]))
        summary.append(pd.read_csv(csv, index_col=0))
    if len(cvs) > 1:
        raise SystemExit("all runs must use the same CV (random or site)")
    summary = pd.concat(summary)
    summary = summary[summary.index.isin(probs)].sort_values("auc", ascending=False)

    if args.configs:
        configs = args.configs
    elif args.by_window is not None:
        best = summary.groupby("window", sort=False).head(1)      # summary is sorted by fold AUC
        best = pd.Series(best.index, index=best["window"].astype(str))
        absent = [w for w in args.by_window if w not in best.index]
        if absent:
            raise SystemExit(f"no results for windows {absent}; found {best.index.tolist()}")
        configs = best[args.by_window].tolist() if args.by_window else best.tolist()
    else:
        configs = summary.index[:args.top].tolist()
    if len(configs) > len(SERIES):
        raise SystemExit(f"at most {len(SERIES)} configs per figure; pick them with --by-window or --configs")
    missing = [c for c in configs if c not in probs]
    if missing:
        raise SystemExit(f"no out-of-fold predictions for {missing}")

    df = load_bucket("finetune")
    train = df[df["split"] == "train"].reset_index(drop=True)
    if train["participant_id"].tolist() != ids:
        raise SystemExit("participants differ from the run's: the bucket changed since it was written")
    y = train["label"].to_numpy()
    cv = cvs.pop()
    folds = cv_folds(train, group_field="meta_site" if cv == "site" else None)

    curves = []
    for c in configs:
        prob = probs[c]
        cm = operating_point(y, prob, folds, args.threshold)
        (tp, fn), (fp, tn) = cm
        row = summary.loc[c]
        site = f"   within-site AUC {row['site_auc']:.3f}" if "site_auc" in row and pd.notna(row["site_auc"]) else ""
        print(f"\n{c}   fold AUC {row['auc']:.3f}   oof AUC {roc_auc_score(y, prob):.3f}{site}")
        print(f"               pred +   pred -\n   actual +   {tp:6d}   {fn:6d}    recall      {tp / (tp + fn):.3f}"
              f"\n   actual -   {fp:6d}   {tn:6d}    specificity {tn / (tn + fp):.3f}"
              f"\n   PPV {tp / max(tp + fp, 1):.3f}   NPV {tn / max(tn + fn, 1):.3f}")
        curves.append((short(c, configs), prob, cm))

    if len(configs) > 1:
        # Against the weakest config: is each gain bigger than resampling noise?
        base = min(configs, key=lambda c: roc_auc_score(y, probs[c]))
        print(f"\nPaired bootstrap of oof AUC vs {base} (2000 resamples of participants):")
        for c in configs:
            if c != base:
                d, (lo, hi), p = paired_bootstrap(y, probs[c], probs[base])
                print(f"   {c:50s}  dAUC {d:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]   p {p:.3f}")
        print("   (the configs were picked as the best of many, so treat p as optimistic)")

    stem = args.csv[0].stem if len(args.csv) == 1 else "compare_" + "_vs_".join(
        re.sub(r"^(probe|mil)_|_(random|site)_.*$", "", c.stem) for c in args.csv)
    out = args.csv[0].with_name(stem + "_curves.png")
    title = f"{stem}: {len(y)} participants, {int(y.sum())} positive, {cv} CV"
    plot(out, title, y, curves)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
