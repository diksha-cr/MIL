"""Cross-validation protocol, following mh-ML-pipeline's MLTrainer.run_cv.

- Outer CV: 5-fold stratified (random_state=42) over the bucket's composite
  `cv_stratum` (label x cohort x illness x site x language x education x age), on
  the frozen TRAIN split only. The test split is never read here. n_repeats > 1
  gives the pipeline's repeated 5 x 5 scheme.
- Optional site-grouped CV (group_field="meta_site"): one StratifiedGroupKFold
  per repeat, reseeded per repeat, stratified on the label — no site appears on
  both sides of a fold, so the estimate is for an unseen site.
- Decision threshold for a fold comes from the OTHER folds of the same repeat
  (their out-of-fold probabilities), never from the fold being scored, so
  threshold-dependent metrics are not inflated. Default rule: F2.

Metric and threshold code is copied from the pipeline's src/utils/ml/ml_trainer.py
(best_threshold, compute_metrics) rather than imported, because importing it pulls
in the pipeline's config loader and reporting stack.

A model plugs in as fit_predict(tr_idx, va_idx) -> {config_name: va_probs}. One fit
may return many named configs (e.g. one probe scored under several poolings) and
each is evaluated as its own row.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, fbeta_score,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold

N_SPLITS, N_REPEATS, SEED = 5, 1, 42
METRICS = ("auc", "ap", "f2", "f1", "ppv", "recall", "specificity", "npv")
COUNTS = ("tp", "fn", "fp", "tn")   # confusion matrix, summed over a repeat's folds


def cv_folds(df: pd.DataFrame, group_field: str | None = None, n_splits: int = N_SPLITS,
             n_repeats: int = N_REPEATS, seed: int = SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    """Folds over the row positions of `df` (the train split, in load_bucket order)."""
    y = df["label"].to_numpy()
    if group_field is None:
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
        return list(rskf.split(np.zeros(len(df)), df["cv_stratum"].to_numpy()))
    groups = df[group_field].astype(str).to_numpy()
    folds = []
    for rep in range(n_repeats):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + rep)
        folds.extend(sgkf.split(np.zeros(len(df)), y, groups=groups))
    return folds


def best_threshold(y_true: np.ndarray, probs: np.ndarray, metric: str = "f2") -> float:
    p, r, t = precision_recall_curve(y_true, probs)
    if metric.startswith("npv_floor"):
        target = float(metric.split(":", 1)[1]) if ":" in metric else 0.93
        best_thr, best_ppv, fallback, best_npv = None, -1.0, None, -1.0
        for thr in t:
            tn, fp, fn, tp = confusion_matrix(y_true, probs >= thr, labels=[0, 1]).ravel()
            if tn + fn == 0:
                continue
            npv = tn / (tn + fn)
            if npv > best_npv:
                best_npv, fallback = npv, float(thr)
            if tp + fp and npv >= target and tp / (tp + fp) > best_ppv:
                best_ppv, best_thr = tp / (tp + fp), float(thr)
        return best_thr if best_thr is not None else (fallback if fallback is not None else 0.5)
    beta2 = {"f2": 4.0, "f1": 1.0, "f0.5": 0.25}[metric]
    score = (1 + beta2) * p * r / (beta2 * p + r + 1e-8)
    return float(t[np.argmax(score[:-1])])


def compute_metrics(y_true: np.ndarray, preds: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return {
        "auc": roc_auc_score(y_true, prob),
        "ap": average_precision_score(y_true, prob),
        "f2": fbeta_score(y_true, preds, beta=2.0, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "ppv": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "npv": tn / (tn + fn) if tn + fn else 1.0,
        **{k: int(v) for k, v in zip(COUNTS, (tp, fn, fp, tn))},
    }


def languages(df: pd.DataFrame) -> np.ndarray | None:
    """Recording language per row (meta_language), for lang_auc; None if the bucket has no such column."""
    return df["meta_language"].astype(str).to_numpy() if "meta_language" in df else None


def within_group_auc(y: np.ndarray, prob: np.ndarray, groups: np.ndarray) -> float:
    """AUC computed inside each group (site) and averaged, weighted by positives.

    Only compares participants recorded at the same site, so a model cannot score
    by learning that one site has more positives than another. Groups with a
    single class are skipped.
    """
    aucs, weights = [], []
    for g in np.unique(groups):
        m = groups == g
        if len(np.unique(y[m])) == 2:
            aucs.append(roc_auc_score(y[m], prob[m]))
            weights.append(y[m].sum())
    return float(np.average(aucs, weights=weights))


def run_cv(y: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]],
           fit_predict: Callable[[np.ndarray, np.ndarray], dict[str, np.ndarray]],
           threshold_metric: str = "f2", n_splits: int = N_SPLITS,
           sites: np.ndarray | None = None, langs: np.ndarray | None = None,
           verbose: bool = False, keep_repeats: bool = False):
    """Returns (summary, oof), plus repeat_oof if keep_repeats.

    summary: one row per config — mean and SD over folds of every metric, plus
    `oof_auc`, the AUC of each repeat's complete out-of-fold vector averaged over
    repeats (less noisy than per-fold AUC with ~13 positives per fold), and, when
    `sites` is given, `site_auc` — the same out-of-fold vectors scored within site
    (`lang_auc`: within language, when `langs` is given).
    oof: config -> out-of-fold probability per row, averaged over repeats.
    repeat_oof: config -> (n_repeats, rows) out-of-fold probabilities, one row per repeat.
    """
    n_repeats = len(folds) // n_splits
    fold_probs: list[dict[str, np.ndarray]] = []
    for k, (tr, va) in enumerate(folds):
        fold_probs.append(fit_predict(tr, va))
        if verbose:
            print(f"  fold {k + 1}/{len(folds)}", flush=True)
    names = list(fold_probs[0])

    repeat_oof = {m: np.full((n_repeats, len(y)), np.nan) for m in names}
    for k, (_, va) in enumerate(folds):
        for m in names:
            repeat_oof[m][k // n_splits, va] = fold_probs[k][m]

    rows, all_rows = [], np.arange(len(y))
    for m in names:
        per_fold = []
        for k, (_, va) in enumerate(folds):
            rep = k // n_splits
            others = np.setdiff1d(all_rows, va)
            ref = repeat_oof[m][rep, others]
            thr = best_threshold(y[others], ref, threshold_metric) if len(np.unique(y[others])) == 2 else 0.5
            prob = fold_probs[k][m]
            per_fold.append({**compute_metrics(y[va], (prob >= thr).astype(int), prob), "thresh": thr})
        pf = pd.DataFrame(per_fold)
        row = {"config": m, "oof_auc": np.mean([roc_auc_score(y, repeat_oof[m][r]) for r in range(n_repeats)])}
        if sites is not None:
            row["site_auc"] = np.mean([within_group_auc(y, repeat_oof[m][r], sites) for r in range(n_repeats)])
        if langs is not None:
            row["lang_auc"] = np.mean([within_group_auc(y, repeat_oof[m][r], langs) for r in range(n_repeats)])
        for c in (*METRICS, "thresh"):
            row[c], row[f"{c}_sd"] = pf[c].mean(), pf[c].std()
        for c in COUNTS:                 # every participant scored once per repeat
            row[c] = pf[c].sum() / n_repeats
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("config")
    oof = {m: np.nanmean(repeat_oof[m], axis=0) for m in names}
    return (summary, oof, repeat_oof) if keep_repeats else (summary, oof)


def format_summary(summary: pd.DataFrame, sort: str = "auc", top: int | None = None) -> str:
    s = summary.sort_values(sort, ascending=False)
    if top:
        s = s.head(top)
    cols = ["oof_auc", "auc", "ap", "f2", "f1", "ppv", "recall", "specificity", "npv"]
    for c in ("lang_auc", "site_auc"):
        if c in s:
            cols.insert(1, c)
    out = s[cols].copy()
    out["auc"] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(s["auc"], s["auc_sd"])]
    if "tp" in s:
        out["TP/FN/FP/TN"] = ["/".join(f"{s.at[i, c]:g}" for c in COUNTS) for i in s.index]
    return out.to_string(float_format=lambda v: f"{v:.3f}")
