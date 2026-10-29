"""Paired comparisons of out-of-fold predictions, and the text confound check.

delta     Paired bootstrap over participants of AUC(config) - AUC(reference) and of the
          within-site AUC difference, on the out-of-fold vectors any stage writes
          (*_oof.npz: participant_id, config, prob). Both vectors score the same
          participants, so the bootstrap resamples participants and recomputes both
          AUCs on each resample. Reported: delta, 95 % percentile CI, and the share of
          resamples where the config is not better (a one-sided bootstrap p).

confound  How well a feature set's participant-mean vector predicts site and language
          (5-fold balanced logistic regression, train split), against the majority
          class. Text embeddings carry the script, so language is expected near 100 %;
          if so, read a fused model's gain from site_auc / lang_auc, not from auc.

chance    Is a config better than chance? On one prediction vector (an *_oof.npz or a
          holdout *_test_prob.npz) with its labels:
            - AUC permutation test (threshold-free): shuffle labels, recompute AUC;
            - Cohen's kappa at a threshold, permutation test;
            - chi-square (Yates) and one-sided Fisher exact on the 2x2 confusion matrix.
          Each permutation p is given twice: labels shuffled over everyone (p_*), and
          shuffled only within each site (p_*_site). The site null keeps how many
          positives each site has, so a model that only knows the site is NOT better
          than it; with this cohort's site confound, p_*_site is the one to quote.
          Threshold: a holdout run's own (train CV F2 cut, from the _test.csv next to
          the npz), a fixed --threshold value, or else the F2-best cut of the vector
          itself, re-chosen inside every permutation so kappa's p stays valid
          (chi-square / Fisher are then optimistic: they can't account for the choice).
          With a fixed threshold the unrestricted kappa permutation is exactly the
          one-sided Fisher test (kappa is monotone in TP once the margins are fixed).
          OOF vectors: every probability is out-of-fold, so permuting labels after
          the fact is a fair test of the vector, but it does not re-run training.

    python -m minns.compare delta --ref outputs/results/stage_e/fusion_X_oof.npz:'<config>' \\
        --against outputs/results/stage_e/fusion_X_oof.npz
    python -m minns.compare confound --set xlmr-large-seg@16 --windows seg
    python -m minns.compare chance outputs/results/test/fusion_seg_top4_test_prob.npz
    python -m minns.compare chance outputs/results/stage_e/fusion_X_oof.npz:'<config>' --out chance.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from minns.data import load_bucket
from minns.evaluation import SEED, best_threshold, within_group_auc

N_BOOT = 2000
N_PERM = 10000


def load_oof(path: str) -> dict[str, pd.Series]:
    z = np.load(path, allow_pickle=True)
    pid = z["participant_id"].astype(str)
    return {str(c): pd.Series(p, index=pid) for c, p in zip(z["config"], z["prob"])}


def pick(oofs: dict[str, pd.Series], name: str | None, path: str) -> tuple[str, pd.Series]:
    if name is None:
        if len(oofs) != 1:
            raise SystemExit(f"{path} holds {len(oofs)} configs; name one with {path}:<config>")
        return next(iter(oofs.items()))
    hits = [c for c in oofs if c == name] or [c for c in oofs if name in c]
    if len(hits) != 1:
        raise SystemExit(f"{name!r} matches {len(hits)} configs in {path}: {hits[:5]}")
    return hits[0], oofs[hits[0]]


def split_spec(spec: str) -> tuple[str, str | None]:
    path, _, name = spec.partition(".npz:")
    return (path + ".npz", name or None) if name else (spec, None)


def delta(ref: pd.Series, other: pd.Series, y: np.ndarray, sites: np.ndarray, rng) -> dict[str, float]:
    a, b = ref.to_numpy(), other.to_numpy()
    out = {"auc": roc_auc_score(y, b), "d_auc": roc_auc_score(y, b) - roc_auc_score(y, a),
           "site_auc": within_group_auc(y, b, sites),
           "d_site_auc": within_group_auc(y, b, sites) - within_group_auc(y, a, sites)}
    boots = []
    n = len(y)
    while len(boots) < N_BOOT:
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        try:
            ds = within_group_auc(y[i], b[i], sites[i]) - within_group_auc(y[i], a[i], sites[i])
        except ZeroDivisionError:          # no site with both classes in this resample
            ds = np.nan
        boots.append((roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]), ds))
    boots = np.asarray(boots)
    for k, col in (("d_auc", 0), ("d_site_auc", 1)):
        v = boots[:, col][np.isfinite(boots[:, col])]
        out[f"{k}_lo"], out[f"{k}_hi"] = np.percentile(v, [2.5, 97.5])
        out[f"{k}_p"] = float((v <= 0).mean())
    return out


def run_delta(args) -> None:
    df = load_bucket("finetune").set_index("participant_id")
    ref_path, ref_name = split_spec(args.ref)
    ref_cfg, ref = pick(load_oof(ref_path), ref_name, ref_path)
    rows = {}
    for spec in args.against:
        path, name = split_spec(spec)
        oofs = load_oof(path)
        chosen = [pick(oofs, name, path)] if name else list(oofs.items())
        for cfg, s in chosen:
            if cfg == ref_cfg and path == ref_path:
                continue
            common = ref.index.intersection(s.index)
            if len(common) != len(ref):
                raise SystemExit(f"{cfg}: scores {len(common)} of the reference's {len(ref)} participants")
            y = df.loc[common, "label"].to_numpy()
            sites = df.loc[common, "meta_site"].astype(str).to_numpy()
            rows[cfg] = delta(ref.loc[common], s.loc[common], y, sites, np.random.default_rng(SEED))
    out = pd.DataFrame(rows).T.sort_values("d_site_auc", ascending=False)
    print(f"reference: {ref_cfg}\n  AUC {roc_auc_score(df.loc[ref.index, 'label'], ref):.3f}, "
          f"{N_BOOT} paired bootstrap resamples over participants\n")
    print(out.to_string(float_format=lambda v: f"{v:.3f}"))
    if args.out:
        out.to_csv(args.out)
        print(f"-> {args.out}")


def run_confound(args) -> None:
    from minns.probe import load_packed, window_key
    df = load_bucket("finetune")
    pids = df["participant_id"].tolist()
    train_rows = np.flatnonzero(df["split"].to_numpy() == "train")
    for w in args.windows:
        feats, owner = load_packed(args.set, window_key(w), pids)
        if feats.shape[0] != 1:
            raise SystemExit(f"{args.set} has {feats.shape[0]} layers; pick one with @")
        X = np.asarray(feats[0], dtype=np.float32)
        X[~np.isfinite(X)] = 0
        sums = np.zeros((len(pids), X.shape[1]))
        np.add.at(sums, owner, X)
        Xp = (sums / np.bincount(owner, minlength=len(pids))[:, None].clip(1))[train_rows]
        print(f"{args.set} | {w}: participant mean of {X.shape[1]}-D instances, {len(train_rows)} train participants")
        for field in ("meta_language", "meta_site"):
            target = df[field].astype(str).to_numpy()[train_rows]
            keep = pd.Series(target).map(pd.Series(target).value_counts()) >= 5     # classes CV can stratify
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced"))
            pred = cross_val_predict(clf, Xp[keep], target[keep],
                                     cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
            acc = (pred == target[keep]).mean()
            majority = pd.Series(target[keep]).value_counts(normalize=True).iloc[0]
            print(f"  {field:14s} accuracy {acc:.3f} (balanced {balanced_accuracy_score(target[keep], pred):.3f}) "
                  f"vs majority {majority:.3f}, {pd.Series(target[keep]).nunique()} classes")


def permuted_labels(y: np.ndarray, groups: np.ndarray, n_perm: int, rng) -> np.ndarray:
    """(n_perm, n) label matrix, each row y shuffled within every group."""
    perm = np.tile(np.arange(len(y)), (n_perm, 1))
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        perm[:, idx] = rng.permuted(np.tile(idx, (n_perm, 1)), axis=1)
    return y[perm]


def kappa(y: np.ndarray, pred: np.ndarray) -> float:
    n, po = len(y), (y == pred).mean()
    pe = (y.sum() * pred.sum() + (n - y.sum()) * (n - pred.sum())) / n ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else 0.0


def perm_p(null: np.ndarray, obs: float) -> float:
    return float((1 + (null >= obs - 1e-12).sum()) / (1 + len(null)))


def chance(y: np.ndarray, p: np.ndarray, sites: np.ndarray, thr: float | None, metric: str,
           n_perm: int, rng) -> dict[str, float]:
    """thr None: F2-best cut of (y, p), re-chosen under every permutation."""
    ranks = rankdata(p)
    n1, n0 = y.sum(), len(y) - y.sum()
    auc_of = lambda Y: (Y @ ranks - n1 * (n1 + 1) / 2) / (n1 * n0)     # Mann-Whitney AUC per row of Y
    t = best_threshold(y, p, metric) if thr is None else thr
    pred = (p >= t).astype(int)
    tp, fn = int((pred & y).sum()), int(((1 - pred) & y).sum())
    fp, tn = int((pred & (1 - y)).sum()), int(((1 - pred) & (1 - y)).sum())
    cm = np.array([[tp, fn], [fp, tn]])
    out = {"n": len(y), "n_pos": int(n1), "auc": roc_auc_score(y, p), "site_auc": within_group_auc(y, p, sites),
           "threshold": t, "tp": tp, "fn": fn, "fp": fp, "tn": tn, "kappa": kappa(y, pred)}
    for suffix, groups in (("", np.zeros(len(y))), ("_site", sites)):
        Y = permuted_labels(y, groups, n_perm, rng)
        out[f"p_auc{suffix}"] = perm_p(auc_of(Y), out["auc"])
        if thr is None:
            null = np.array([kappa(yp, (p >= best_threshold(yp, p, metric)).astype(int)) for yp in Y])
        else:
            null = np.array([kappa(yp, pred) for yp in Y])
        out[f"p_kappa{suffix}"] = perm_p(null, out["kappa"])
    degenerate = (cm.sum(0) == 0).any() or (cm.sum(1) == 0).any()     # everyone predicted one class
    out["chi2"], out["p_chi2"] = (np.nan, np.nan) if degenerate else chi2_contingency(cm)[:2]
    out["p_fisher"] = fisher_exact(cm, alternative="greater")[1]
    return out


def run_chance(args) -> None:
    df = load_bucket("finetune").set_index("participant_id")
    path, name = split_spec(args.preds)
    preds = load_oof(path)
    chosen = [pick(preds, name, path)] if name else list(preds.items())
    test_csv = Path(path).with_name(Path(path).stem.removesuffix("_prob") + ".csv")
    holdout_thr = pd.read_csv(test_csv, index_col="config")["threshold"] \
        if path.endswith("_test_prob.npz") and test_csv.exists() else None
    rows = {}
    for cfg, s in chosen:
        y = df.loc[s.index, "label"].to_numpy().astype(int)
        sites = df.loc[s.index, "meta_site"].astype(str).to_numpy()
        if args.threshold is not None:
            thr, src = args.threshold, "fixed"
        elif holdout_thr is not None and cfg in holdout_thr.index:
            thr, src = float(holdout_thr[cfg]), "train CV"
        else:
            thr, src = None, f"{args.metric} on itself, re-chosen per permutation"
        rows[cfg] = chance(y, s.to_numpy(), sites, thr, args.metric, args.n_perm, np.random.default_rng(SEED))
        rows[cfg]["threshold_from"] = src
    out = pd.DataFrame(rows).T
    cols = ["n", "n_pos", "auc", "p_auc", "p_auc_site", "site_auc", "threshold", "tp", "fn", "fp", "tn",
            "kappa", "p_kappa", "p_kappa_site", "chi2", "p_chi2", "p_fisher", "threshold_from"]
    out = out[cols]
    print(f"{path}: {args.n_perm} label permutations (p_*: over everyone, p_*_site: within site); "
          f"one-sided except chi-square\n")
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(out.to_string(float_format=lambda v: f"{v:.3g}" if abs(v) < 0.001 else f"{v:.3f}"))
    if args.out:
        out.to_csv(args.out)
        print(f"-> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("delta")
    d.add_argument("--ref", required=True, help="<oof.npz>[:<config>]")
    d.add_argument("--against", nargs="+", required=True, help="<oof.npz>[:<config>] (every config if none)")
    d.add_argument("--out", type=Path, default=None)
    c = sub.add_parser("confound")
    c.add_argument("--set", required=True, help="feature set at one layer, e.g. xlmr-large-seg@16")
    c.add_argument("--windows", nargs="+", default=["seg"])
    h = sub.add_parser("chance")
    h.add_argument("preds", help="<oof.npz | test_prob.npz>[:<config>] (every config if none)")
    h.add_argument("--threshold", type=float, default=None, help="fixed cut; default: holdout's train-CV cut, "
                   "else the --metric cut re-chosen per permutation")
    h.add_argument("--metric", default="f2")
    h.add_argument("--n-perm", type=int, default=N_PERM)
    h.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    {"delta": run_delta, "confound": run_confound, "chance": run_chance}[args.cmd](args)


if __name__ == "__main__":
    main()
