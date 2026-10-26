"""Stage B: layer x window x pooling probe (de Gennes et al. 2024) on our cohort.

what exactly is the probe.py doing? is it following the de genne's paper which is based on the 2 hyperamaters choosing or the the other paper which uses MINN and then does MIL based on the max pooling.

For every encoder layer and window size:

  window probe   a balanced logistic regression trained on WINDOWS, each window
                 carrying its participant's label (single-instance learning).
                 Every participant gets the same total weight however many windows
                 they have, and the two classes get equal total weight. The
                 window probabilities of a held-out participant are then pooled
                 into one score:
                   mean, max, min, mellowmax(w) for w in POOL_OMEGAS, p90, and
                   vote (share of windows above 0.5 — the majority-vote rule of
                   Alsarrani et al. 2025, kept to show how it behaves at 11 %).
  full           the same regression on one mean-pooled embedding per
                 participant (the pipeline's representation, at every layer).
  dlg            dialogue utterances (minns.utterances): every speaker's utterances
                 in the bag. Besides the pools above over all of them, mean / max /
                 p90 over each role alone: mean@P (participant), mean@I (interviewer),
                 ... A bag without interviewer utterances uses its all-utterance value.
                 Give the regression the roles too with +role in --encoder.

Config names read  <encoder>|<window>|L<layer>|<pooling>, e.g. whisper-medium|w5|L18|mm10.

Picking the best row of this sweep is a selection over hundreds of configs on
the same folds, so its score is optimistic. It is for choosing where to look;
the honest number comes from a nested run.

    python -m minns.probe --encoder whisper-medium
    python -m minns.probe --encoder xlsr-300m --windows 5 10 --layers 12 18 24 --cv site
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from joblib import Parallel, delayed
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from minns.data import OUTPUTS, load_bucket
from minns.evaluation import SEED, cv_folds, format_summary, run_cv

POOL_OMEGAS = (-10, -1, 1, 10, 100)


# ── packed, layer-major feature cache ─────────────────────────────────────────

def window_key(w: str) -> str:
    """CLI window -> npz key: "full", "utt" / "dlg" (participant / dialogue utterances,
    minns.utterances) or "w<sec>"."""
    return w if w in ("full", "utt", "dlg") else f"w{float(w):g}"


def load_packed(encoder: str, key: str, pids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(features (L, N, D) memmap in the npz dtype, owner (N,) row index into `pids`).

    The per-participant npz files are repacked once per (encoder, key) into a
    layer-major .npy so a single layer is one contiguous read. `key` is "full",
    "utt" or "w<sec>". Rebuilt automatically if the participant list changes.

    "a+b" concatenates feature sets along the feature axis, e.g. egemaps+prosody
    or whisper-medium-utt+prosody. They must share their instances (the utterance
    sets do); a single-layer set is repeated across the other set's layers.
    """
    if "+" in encoder:
        parts = [load_packed(e, key, pids) for e in encoder.split("+")]
        owner = parts[0][1]
        if any(not np.array_equal(o, owner) for _, o in parts[1:]):
            raise SystemExit(f"{encoder}: the feature sets do not have the same instances for {key}")
        n_layers = max(f.shape[0] for f, _ in parts)
        if any(f.shape[0] not in (1, n_layers) for f, _ in parts):
            raise SystemExit(f"{encoder}: layer counts {[f.shape[0] for f, _ in parts]} cannot be combined")
        return np.concatenate([np.broadcast_to(f, (n_layers, *f.shape[1:])) for f, _ in parts], axis=2), owner

    src = OUTPUTS / "features/windows" / encoder
    dst = OUTPUTS / "features/packed" / encoder
    dst.mkdir(parents=True, exist_ok=True)
    arr_path, meta_path = dst / f"{key}.npy", dst / f"{key}.json"

    missing = [p for p in pids if not (src / f"{p}.npz").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} participants not extracted yet for {encoder} (e.g. {missing[0]})")

    # Reuse the pack unless the participants changed or a source file was rewritten
    # after it (re-extraction, re-segmentation with --overwrite).
    newest = max((src / f"{p}.npz").stat().st_mtime for p in pids)
    if (arr_path.exists() and meta_path.exists() and json.loads(meta_path.read_text())["pids"] == pids
            and arr_path.stat().st_mtime >= newest):
        owner = np.load(dst / f"{key}_owner.npy")
        return np.load(arr_path, mmap_mode="r"), owner

    if key == "full":
        counts = [1] * len(pids)
    else:
        counts = [len(np.load(src / f"{p}.npz")[f"{key}_span"]) for p in pids]
    first = np.load(src / f"{pids[0]}.npz")[key]
    n_layers, dim = first.shape[-2:]
    out = np.lib.format.open_memmap(arr_path, mode="w+", dtype=first.dtype,
                                    shape=(n_layers, sum(counts), dim))
    off = 0
    for p, n in zip(pids, counts):
        a = np.load(src / f"{p}.npz")[key]
        out[:, off:off + n] = a[:, None, :] if key == "full" else a.transpose(1, 0, 2)
        off += n
    out.flush()
    owner = np.repeat(np.arange(len(pids)), counts)
    np.save(dst / f"{key}_owner.npy", owner)
    meta_path.write_text(json.dumps({"pids": pids}))
    return np.load(arr_path, mmap_mode="r"), owner


# ── pooling of window probabilities ───────────────────────────────────────────

ROLE_POOLS = {"P": 0, "I": 1}           # data.ROLES indices pooled on their own for dialogue bags


def pool_scores(prob: np.ndarray, owner: np.ndarray, bags: np.ndarray,
                roles: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Pool window probabilities into one score per bag, for every pooling rule.

    With `roles` (one data.ROLES index per instance) also mean / max / p90 over each
    ROLE_POOLS role alone; a bag without that role falls back to its all-instance value.
    """
    names = ["mean", "max", "min", "p90", "vote", *[f"mm{w}" for w in POOL_OMEGAS]]
    if roles is not None:
        names += [f"{k}@{tag}" for tag in ROLE_POOLS for k in ("mean", "max", "p90")]
    out = {k: np.empty(len(bags)) for k in names}
    order = np.argsort(owner, kind="stable")
    owner_s, prob_s = owner[order], prob[order]
    roles_s = roles[order] if roles is not None else None
    starts = np.searchsorted(owner_s, bags, side="left")
    ends = np.searchsorted(owner_s, bags, side="right")
    for i, (s, e) in enumerate(zip(starts, ends)):
        p = prob_s[s:e]
        if roles_s is not None:
            for tag, r in ROLE_POOLS.items():
                q = p[roles_s[s:e] == r]
                q = q if len(q) else p
                out[f"mean@{tag}"][i], out[f"max@{tag}"][i] = q.mean(), q.max()
                out[f"p90@{tag}"][i] = np.quantile(q, 0.9)
        out["mean"][i], out["max"][i], out["min"][i] = p.mean(), p.max(), p.min()
        out["p90"][i] = np.quantile(p, 0.9)
        out["vote"][i] = (p > 0.5).mean()
        for w in POOL_OMEGAS:   # mellowmax: (1/w) log mean exp(w p); -> min / mean / max as w -> -inf / 0 / inf
            out[f"mm{w}"][i] = (logsumexp(w * p) - np.log(len(p))) / w
    return out


# ── one fold, one layer ───────────────────────────────────────────────────────

def finite(X: np.ndarray) -> np.ndarray:
    """float32 copy with inf / NaN as NaN. After standardising, np.nan_to_num turns
    them into 0, i.e. the training mean: a missing value says nothing either way."""
    X = np.array(X, dtype=np.float32)
    X[~np.isfinite(X)] = np.nan
    return X


def instance_weights(owner: np.ndarray, y_bag: np.ndarray) -> np.ndarray:
    """Each bag weighs 1 in total, split evenly over its windows; classes balanced at bag level."""
    bags, counts = np.unique(owner, return_counts=True)
    per_bag_n = dict(zip(bags, counts))
    yb = y_bag[bags]
    class_w = {c: len(yb) / (2.0 * max((yb == c).sum(), 1)) for c in (0, 1)}
    return np.array([class_w[y_bag[o]] / per_bag_n[o] for o in owner])


def fit_layer(X_layer: np.ndarray, owner: np.ndarray, y_bag: np.ndarray,
              tr_bags: np.ndarray, va_bags: np.ndarray, C: float, pooled: bool,
              roles: np.ndarray | None = None) -> dict[str, np.ndarray]:
    tr_m, va_m = np.isin(owner, tr_bags), np.isin(owner, va_bags)
    Xtr = finite(X_layer[tr_m])
    scaler = StandardScaler().fit(Xtr)                 # ignores NaN when fitting
    clf = LogisticRegression(C=C, max_iter=2000, random_state=SEED)
    clf.fit(np.nan_to_num(scaler.transform(Xtr)), y_bag[owner[tr_m]],
            sample_weight=instance_weights(owner[tr_m], y_bag))
    prob = clf.predict_proba(np.nan_to_num(scaler.transform(finite(X_layer[va_m]))))[:, 1]
    if not pooled:                       # one instance per bag: the probability is the score
        order = np.argsort(owner[va_m])
        return {"mean": prob[order]}
    return pool_scores(prob, owner[va_m], np.sort(va_bags), None if roles is None else roles[va_m])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--windows", nargs="+", default=["full", "2", "5", "10", "20"],
                    help="window sizes in seconds, 'full' (one mean embedding per participant) "
                         "and/or 'utt' (utterances, from minns.utterances)")
    ap.add_argument("--layers", nargs="+", default=["all"])
    ap.add_argument("--C", type=float, default=0.01, help="inverse L2 strength of every probe")
    ap.add_argument("--cv", choices=["random", "site"], default="random")
    ap.add_argument("--threshold", default="f2")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    df = load_bucket("finetune")
    pids = df["participant_id"].tolist()
    train_rows = np.flatnonzero(df["split"].to_numpy() == "train")   # bucket rows of the train split
    train = df.iloc[train_rows].reset_index(drop=True)
    y_bag = df["label"].to_numpy()
    y = y_bag[train_rows]
    folds = cv_folds(train, group_field="meta_site" if args.cv == "site" else None)

    keys = [window_key(w) for w in args.windows]
    data = {k: load_packed(args.encoder, k, pids) for k in keys}
    roles = {}
    if "dlg" in data:                  # speaker role of every dialogue utterance, for per-role pooling
        onehot, owner = load_packed("role", "dlg", pids)
        if not np.array_equal(owner, data["dlg"][1]):
            raise SystemExit("role and feature sets have different dialogue utterances; re-run minns.utterances")
        roles["dlg"] = np.asarray(onehot[0], dtype=np.float32).argmax(1)
    n_layers = next(iter(data.values()))[0].shape[0]
    layers = list(range(n_layers)) if args.layers == ["all"] else [int(l) for l in args.layers]
    if not all(0 <= l < n_layers for l in layers):
        raise SystemExit(f"{args.encoder} has layers 0..{n_layers - 1}; asked for {layers}")
    print(f"{args.encoder}: windows {keys}, layers {layers[0]}..{layers[-1]} ({len(layers)}), C={args.C}, "
          f"{args.cv} CV, {len(folds)} folds, train {len(y)} ({y.sum()} positive)")

    def fit_predict(tr, va):
        tr_bags, va_bags = train_rows[tr], train_rows[va]
        # va is not sorted in general; the pooled scores come back in sorted-bag order
        back = np.argsort(np.argsort(va_bags))
        jobs = [(k, l) for k in keys for l in layers]
        results = Parallel(n_jobs=args.jobs)(
            delayed(fit_layer)(data[k][0][l], data[k][1], y_bag, tr_bags, va_bags, args.C, k != "full",
                               roles.get(k))
            for k, l in jobs)
        out = {}
        for (k, l), res in zip(jobs, results):
            for pool, scores in res.items():
                out[f"{args.encoder}|{k}|L{l}|{pool}"] = scores[back]
        return out

    t0 = time.time()
    summary, oof = run_cv(y, folds, fit_predict, threshold_metric=args.threshold,
                          sites=train["meta_site"].to_numpy(), verbose=True)
    parts = summary.index.to_series().str.split("|", expand=True)
    summary.insert(0, "window", parts[1].values)
    summary.insert(1, "layer", parts[2].str[1:].astype(int).values)
    summary.insert(2, "pool", parts[3].values)

    out_dir = OUTPUTS / "results/stage_b"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"probe_{args.encoder}_{args.cv}_C{args.C:g}"
    summary.to_csv(out_dir / f"{stem}.csv")
    np.savez_compressed(out_dir / f"{stem}_oof.npz", participant_id=train["participant_id"].to_numpy(dtype=str),
                        config=np.array(list(oof)), prob=np.stack(list(oof.values())))
    print(f"\n{len(summary)} configs in {(time.time() - t0) / 60:.1f} min -> {out_dir / stem}.csv")
    print(f"\nTop 15 by fold AUC (selection-biased):\n{format_summary(summary, top=15)}")


if __name__ == "__main__":
    main()
