"""Held-out TEST split: the final check of configs already chosen on the train CV.

For each named config (a row of a probe_*.csv or mil_*.csv run):
  1. refit it on ALL train participants (same features, layer, C / network as in CV);
  2. score every test participant once;
  3. threshold at the F2-best cut of that config's train out-of-fold probabilities
     (the run's _oof.npz), so nothing about the test split is chosen on the test split.

Prints AUC with a bootstrap 95% CI, the confusion matrix, recall / specificity / PPV /
NPV, a paired bootstrap of AUC differences, and writes results/figures/<name>_test.png (ROC +
confusion matrices, the same figure as minns.curves) and results/test/<name>_test.csv.

Choose the configs BEFORE running this, and run it once: picking the best of several
test results makes the test split another validation set. With ~15 positives in
test, the AUC's CI is about +-0.1 wide; read it next to the CV number, not instead of it.

    python -m minns.holdout --name utt_vs_full \
        outputs/results/stage_b/probe_whisper-medium-utt+prosody_random_C0.01.csv \
        --configs "whisper-medium-utt+prosody|full|L24|mean" "whisper-medium-utt+prosody|utt|L24|mean"
    python -m minns.holdout --name probe_vs_mil <probe csv> <mil csv> --configs <probe config> <mil config>

--seed-sets K (MIL configs only): how much the test AUC depends on the networks'
random initialisation. Each config is refitted K times, seed set k using seeds
SEED + 5k .. SEED + 5k + 4 (k = 0 is the run above), and EVERY set's test AUC is
reported, plus their mean +- SD. Quote the mean, decided before running; picking the
best set would make the test split a validation set again. Writes
results/test/<name>_seeds<K>_test.csv; the single-run outputs are not touched.

    python -m minns.holdout --name fusion_seg_top4 <mil csv> --configs <config> ... --seed-sets 10
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from minns.curves import display_labels, is_baseline, paired_bootstrap, plot, series_colors
from minns.data import OUTPUTS, load_bucket
from minns.evaluation import best_threshold


def auc_ci(y: np.ndarray, p: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Bootstrap 95% CI of the AUC, resampling within each class."""
    lo_hi = paired_bootstrap(y, p, np.zeros_like(p) + 0.5, n=n, seed=seed)[1]   # AUC(p) - AUC(constant 0.5)
    return float(lo_hi[0] + 0.5), float(lo_hi[1] + 0.5)


def probe_test(config: str, C: float, df: pd.DataFrame, train_rows, test_rows) -> np.ndarray:
    from minns.probe import DIALOGUE_ROLES, fit_layer, load_packed

    encoder, key, layer, pool = config.split("|")
    pids, y_bag = df["participant_id"].tolist(), df["label"].to_numpy()
    feats, owner = load_packed(encoder, key, pids)
    roles = None
    if key in DIALOGUE_ROLES:
        onehot, role_owner = load_packed(DIALOGUE_ROLES[key], key, pids)
        if not np.array_equal(role_owner, owner):
            raise SystemExit(f"{DIALOGUE_ROLES[key]} and {encoder} have different {key} instances; re-segment")
        roles = np.asarray(onehot[0], dtype=np.float32).argmax(1)
    scores = fit_layer(feats[int(layer[1:])], owner, y_bag, train_rows, test_rows, C, key != "full", roles)
    return scores[pool]                       # in sorted test-row order, i.e. test_rows' order


def mil_test(config: str, C: float, df: pd.DataFrame, train_rows, test_rows, seed_offset: int = 0) -> np.ndarray:
    from minns.extract import pick_device
    from minns.mil import EPOCHS, N_SEEDS, fit_predict_fold, padded_bags
    from minns.probe import load_packed

    encoder, key, layer, model = config.split("|")
    device = pick_device()
    feats, owner = load_packed(encoder, key, df["participant_id"].tolist())
    bags = np.concatenate([train_rows, test_rows])        # train bags first, then test
    X, mask, lengths = padded_bags(np.asarray(feats[int(layer[1:])]), owner, bags, device)
    y = df["label"].to_numpy()[bags].copy()
    tr, te = np.arange(len(train_rows)), np.arange(len(train_rows), len(bags))
    return fit_predict_fold(X, mask, lengths, y, tr, te, [model], N_SEEDS, EPOCHS, device, C, seed_offset)[model]


def seed_stability(args, source, df, train_rows, test_rows, y_tr, y_te) -> None:
    from minns.mil import N_SEEDS
    rows = []
    for c in args.configs:
        kind, C, oof = source[c]
        if kind != "mil":
            print(f"\n{c}: a probe is deterministic, no seeds to vary; skipped")
            continue
        thr = best_threshold(y_tr, oof, args.threshold)
        print(f"\n{c}   (threshold {thr:.3f} from train CV)")
        for k in range(args.seed_sets):
            p = mil_test(c, C, df, train_rows, test_rows, seed_offset=k * N_SEEDS)
            pred, yt = p >= thr, y_te == 1
            tp, fn, fp, tn = (pred & yt).sum(), (~pred & yt).sum(), (pred & ~yt).sum(), (~pred & ~yt).sum()
            auc = roc_auc_score(y_te, p)
            print(f"   seed set {k:2d}: AUC {auc:.3f}   TP/FN/FP/TN {tp}/{fn}/{fp}/{tn}", flush=True)
            rows.append(dict(config=c, seed_set=k, test_auc=auc, threshold=thr, tp=tp, fn=fn, fp=fp, tn=tn,
                             recall=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1)))
    if not rows:
        return
    res = pd.DataFrame(rows)
    agg = res.groupby("config", sort=False)[["test_auc", "recall", "specificity"]].agg(["mean", "std", "min", "max"])
    print(f"\nAcross {args.seed_sets} seed sets (quote the mean):")
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(agg.to_string(float_format=lambda v: f"{v:.3f}"))
    out_dir = OUTPUTS / "results/test"
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / f"{args.name}_seeds{args.seed_sets}_test.csv", index=False)
    print(f"\n-> {out_dir / args.name}_seeds{args.seed_sets}_test.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, nargs="+", help="the probe_*.csv / mil_*.csv runs the configs come from")
    ap.add_argument("--configs", nargs="+", required=True, help="config names (the CSV's first column)")
    ap.add_argument("--name", required=True, help="output file name, e.g. utt_vs_full")
    ap.add_argument("--C", type=float, default=0.01, help="L2 strength of MIL lin-* models (probe C comes from its CSV name)")
    ap.add_argument("--threshold", default="f2")
    ap.add_argument("--seed-sets", type=int, default=1, help="refit each MIL config with this many seed sets "
                    "and report every test AUC with their mean +- SD (see above)")
    args = ap.parse_args()

    df = load_bucket("finetune")
    split = df["split"].to_numpy()
    train_rows, test_rows = np.flatnonzero(split == "train"), np.flatnonzero(split == "test")
    y_tr, y_te = df["label"].to_numpy()[train_rows], df["label"].to_numpy()[test_rows]
    print(f"train {len(train_rows)} ({y_tr.sum()} positive) -> test {len(test_rows)} ({y_te.sum()} positive)")

    # Which run each config comes from: its kind (probe / MIL), C, and its train out-of-fold probabilities.
    source = {}
    for csv in args.csv:
        z = np.load(csv.with_name(csv.stem + "_oof.npz"), allow_pickle=True)
        if z["participant_id"].tolist() != df["participant_id"].to_numpy()[train_rows].tolist():
            raise SystemExit(f"{csv.name}: its train participants differ from the bucket's; re-run it")
        if len(z["prob"]) and z["prob"].shape[1] != len(train_rows) or re.search(r"\dx\d", csv.stem):
            raise SystemExit(f"{csv.name}: need a single-repeat run")
        m = re.search(r"_C([0-9.e-]+)$", csv.stem)
        kind = "mil" if csv.name.startswith("mil_") else "probe"
        C = float(m.group(1)) if kind == "probe" and m else args.C
        for c, p in zip(z["config"].tolist(), z["prob"]):
            source[c] = (kind, C, p)
    missing = [c for c in args.configs if c not in source]
    if missing:
        raise SystemExit(f"not in the given runs: {missing}")
    if args.seed_sets > 1:
        return seed_stability(args, source, df, train_rows, test_rows, y_tr, y_te)
    args.configs = sorted(args.configs, key=lambda c: not is_baseline(c))     # baseline first
    colors = series_colors(args.configs)

    rows, curves, probs = [], [], {}
    labels = display_labels(args.configs)
    for c, label, color in zip(args.configs, labels, colors):
        kind, C, oof = source[c]
        thr = best_threshold(y_tr, oof, args.threshold)
        print(f"\n{label.replace(chr(10), ' · ')}   [{c}]\n   refitting on all train ...", flush=True)
        p = (mil_test if kind == "mil" else probe_test)(c, C, df, train_rows, test_rows)
        probs[c] = p
        pred, yt = p >= thr, y_te == 1
        cm = np.array([[(pred & yt).sum(), (~pred & yt).sum()], [(pred & ~yt).sum(), (~pred & ~yt).sum()]])
        (tp, fn), (fp, tn) = cm
        auc, (lo, hi) = roc_auc_score(y_te, p), auc_ci(y_te, p)
        print(f"   test AUC {auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]   (train CV oof AUC {roc_auc_score(y_tr, oof):.3f})")
        print(f"               pred +   pred -\n   actual +   {tp:6d}   {fn:6d}    recall      {tp / max(tp + fn, 1):.3f}"
              f"\n   actual -   {fp:6d}   {tn:6d}    specificity {tn / max(tn + fp, 1):.3f}"
              f"\n   PPV {tp / max(tp + fp, 1):.3f}   NPV {tn / max(tn + fn, 1):.3f}   threshold {thr:.3f} (from train CV)")
        rows.append(dict(config=c, label=label, test_auc=auc, auc_lo=lo, auc_hi=hi,
                         cv_oof_auc=roc_auc_score(y_tr, oof), threshold=thr, tp=tp, fn=fn, fp=fp, tn=tn,
                         recall=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1),
                         ppv=tp / max(tp + fp, 1), npv=tn / max(tn + fn, 1)))
        curves.append((label, p, cm, color))

    if len(args.configs) > 1:
        base = args.configs[0] if is_baseline(args.configs[0]) else \
            min(args.configs, key=lambda c: roc_auc_score(y_te, probs[c]))
        print(f"\nPaired bootstrap of test AUC vs {labels[args.configs.index(base)]}:")
        for c, label in zip(args.configs, labels):
            if c != base:
                d, (lo, hi), pv = paired_bootstrap(y_te, probs[c], probs[base])
                print(f"   {label.replace(chr(10), ' · '):55s}  dAUC {d:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]   p {pv:.3f}")

    out_dir = OUTPUTS / "results/test"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).set_index("config").to_csv(out_dir / f"{args.name}_test.csv")
    np.savez_compressed(out_dir / f"{args.name}_test_prob.npz", participant_id=df["participant_id"].to_numpy(dtype=str)[test_rows],
                        config=np.array(args.configs), prob=np.stack([probs[c] for c in args.configs]))
    title = f"Held-out test split (models trained on all of train): {len(y_te)} participants, {int(y_te.sum())} positive"
    fig_dir = OUTPUTS / "results/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot(fig_dir / f"{args.name}_test.png", title, y_te, curves)
    print(f"\n-> {fig_dir / args.name}_test.png, {out_dir / args.name}_test.csv")


if __name__ == "__main__":
    main()
