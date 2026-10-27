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

    python -m minns.compare delta --ref outputs/results/stage_e/fusion_X_oof.npz:'<config>' \\
        --against outputs/results/stage_e/fusion_X_oof.npz
    python -m minns.compare confound --set xlmr-large-seg@16 --windows seg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from minns.data import load_bucket
from minns.evaluation import SEED, within_group_auc

N_BOOT = 2000


def load_oof(path: str) -> dict[str, pd.Series]:
    z = np.load(path)
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
    args = ap.parse_args()
    run_delta(args) if args.cmd == "delta" else run_confound(args)


if __name__ == "__main__":
    main()
