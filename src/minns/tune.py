"""Nested CV: an honest score for a MIL model whose instance type, audio layer, text
layer and network hyperparameters are all chosen from the data.

Everything selected so far (L24, text @13, seg instances, the network settings) was
picked on the same folds that score it, so those scores are optimistic. Here the
selection is part of the model: each OUTER training fold runs the whole search on
its own inner 5-fold CV and never sees its validation fold.

  outer CV   cv_folds over the train split (5 folds x --repeats), as in mil.py;
             it scores the procedure "tune, then train".
  stage 1    per instance type (--windows), the audio layer and the text layer that
             give the best inner-CV probe AUC (probe.fit_layer, best of mean / max
             pooling), each set probed on its own: picking the pair jointly would
             mean 24 x 24 fused probes.
  stage 2    Optuna (TPE sampler, median pruner) over the network settings in
             suggest() and the instance type, each instance type at its stage-1
             layers; --n-trials trials per model, the a-priori setting enqueued
             first for every instance type. A trial is one inner 5-fold CV with
             --inner-seeds initialisations; its value is the inner out-of-fold AUC
             (--select site_auc: within-site AUC, if the site confound should not
             count). After each inner fold the running mean fold AUC is reported, so
             trials far below the median stop after two folds.
  refit      the best trial on the whole outer training fold, --seeds
             initialisations averaged, scored on the outer validation fold.

Instance type sets the feature sets: seg / dseg use <audio>-seg + <text>-seg +
prosody-seg (dseg: + role-seg), word uses <audio>-word + <text>-word + wordpros.
Every searched instance type's bags stay on the GPU during the search.

Parallel and resumable. A study (one per outer fold and model) lives in the journal
file tune/<stem>/fold<k>.journal, which any number of processes can share, so:
  - outer folds run as separate jobs (--fold k);
  - several workers can search the same fold at once (same --fold, different
    --worker: each gets its own sampler seed; together they stop at --n-trials);
  - --hours stops a worker between trials before a wall-clock limit; resubmitting
    continues the study. Stage 1 is cached in fold<k>_stage1.json.
Whichever worker ends with the study complete refits and writes fold<k>_<model>.npz
(a later finisher overwrites it with the same or a better trial). --merge scores
the saved folds together. Results: outputs/results/tune/<stem>.csv (summary as
mil.py), <stem>_oof.npz (minns.compare delta works on it) and <stem>_choices.csv
(what every outer fold chose); per fold and model also fold<k>_<model>_trials.csv.

    python -m minns.tune --audio whisper-medium --text xlmr-large --windows seg word --fold 0
    python -m minns.tune --audio whisper-medium --text xlmr-large --windows seg word --merge
    python -m minns.tune --audio whisper-medium --text mbert-base --n-trials 3 --limit-folds 1
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import optuna
import pandas as pd
import torch
from joblib import Parallel, delayed
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState
from sklearn.metrics import roc_auc_score

from minns.data import OUTPUTS, load_bucket
from minns.evaluation import SEED, cv_folds, format_summary, languages, run_cv, within_group_auc
from minns.extract import pick_device
from minns.mil import BATCH, DROPOUT, EPOCHS, HIDDEN, LR, MODELS, N_SEEDS, WEIGHT_DECAY, fit_predict_fold, padded_bags
from minns.probe import fit_layer, load_packed

INSTANCES = {"seg": ("seg", "prosody-seg"), "dseg": ("seg", "prosody-seg+role-seg"), "word": ("word", "wordpros")}
PROBE_POOLS = ("mean", "max")
PROBE_C = 0.01
HIDDEN_CHOICES = ("32-64-128", "128-64-32", "64", "256-64", "128-128")
DEFAULT = {"lr": LR, "weight_decay": WEIGHT_DECAY, "dropout": DROPOUT, "hidden": "-".join(map(str, HIDDEN)),
           "batch_size": BATCH, "epochs": EPOCHS}
FINISHED = (TrialState.COMPLETE, TrialState.PRUNED)


def suggest(trial: optuna.Trial, windows: list[str]) -> dict:
    """The search space. Ranges bracket the a-priori setting (DEFAULT) on both sides."""
    return {
        "window": trial.suggest_categorical("window", windows),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.6),
        "hidden": trial.suggest_categorical("hidden", HIDDEN_CHOICES),
        "batch_size": trial.suggest_categorical("batch_size", (8, 16, 32)),
        "epochs": trial.suggest_int("epochs", 10, 80, step=10),
    }


def net_kwargs(p: dict) -> dict:
    return {"lr": p["lr"], "weight_decay": p["weight_decay"], "batch_size": p["batch_size"],
            "hidden": tuple(int(h) for h in p["hidden"].split("-")), "dropout": p["dropout"]}


def describe(p: dict) -> str:
    return (f"{p['window']} lr={p['lr']:.2g} wd={p['weight_decay']:.2g} drop={p['dropout']:.2f} "
            f"hidden={p['hidden']} batch={p['batch_size']} ep={p['epochs']}")


def score(y: np.ndarray, prob: np.ndarray, sites: np.ndarray, how: str) -> float:
    return within_group_auc(y, prob, sites) if how == "site_auc" else roc_auc_score(y, prob)


# ── stage 1: layers by inner-CV probe ─────────────────────────────────────────

def probe_layers(feats: np.ndarray, owner: np.ndarray, y_bag: np.ndarray, inner_bags, sites, select: str,
                 layers: list[int], jobs: int) -> tuple[int, dict[int, float]]:
    """Best layer of one feature set by inner-CV probe AUC; also every layer's AUC."""
    tasks = [(l, k) for l in layers for k in range(len(inner_bags))]
    res = Parallel(n_jobs=jobs)(
        delayed(fit_layer)(feats[l], owner, y_bag, *inner_bags[k], PROBE_C, True) for l, k in tasks)
    all_va = np.sort(np.concatenate([va for _, va in inner_bags]))
    aucs = {}
    for l in layers:
        best = -np.inf
        for pool in PROBE_POOLS:
            # fit_layer returns scores in sorted-bag order; collect them per bag, then score once
            prob = {}
            for (l2, k), r in zip(tasks, res):
                if l2 == l:
                    prob.update(zip(np.sort(inner_bags[k][1]), r[pool]))
            best = max(best, score(y_bag[all_va], np.array([prob[b] for b in all_va]), sites[all_va], select))
        aucs[l] = best
    return max(aucs, key=aucs.get), aucs


# ── one outer fold ────────────────────────────────────────────────────────────

def write_atomic(path, write) -> None:
    """write(tmp_path), then rename onto `path`, so concurrent workers never leave a half file."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}")
    write(tmp)
    os.replace(tmp, path)


def stage1(args, k: int, inner_bags, site_bag, y_bag, pids, fold_dir) -> tuple[dict, dict]:
    """{window: (audio layer, text layer)} and every layer's inner AUC, cached per fold."""
    cache = fold_dir / f"fold{k}_stage1.json"
    if cache.exists():
        c = json.loads(cache.read_text())
        if set(c["layers"]) >= set(args.windows):
            return {w: tuple(c["layers"][w]) for w in args.windows}, c["aucs"]
    t0 = time.time()
    chosen_layers, layer_aucs = {}, {}
    for w in args.windows:
        suffix, _ = INSTANCES[w]
        a_feats, owner = load_packed(f"{args.audio}-{suffix}", w, pids)
        la, a_aucs = probe_layers(a_feats, owner, y_bag, inner_bags, site_bag, args.select,
                                  args.audio_layers or list(range(a_feats.shape[0])), args.jobs)
        lt, t_aucs = None, {}
        if args.text:
            t_feats, _ = load_packed(f"{args.text}-{suffix}", w, pids)
            lt, t_aucs = probe_layers(t_feats, owner, y_bag, inner_bags, site_bag, args.select,
                                      args.text_layers or list(range(t_feats.shape[0])), args.jobs)
        chosen_layers[w], layer_aucs[w] = (la, lt), {"audio": a_aucs, "text": t_aucs}
        print(f"[fold {k}] stage 1 {w}: audio L{la} ({a_aucs[la]:.3f}), "
              f"text {f'L{lt} ({t_aucs[lt]:.3f})' if args.text else '-'}  {(time.time() - t0) / 60:.1f} min",
              flush=True)
    write_atomic(cache, lambda p: p.write_text(json.dumps({"layers": chosen_layers, "aucs": layer_aucs}, indent=1)))
    return chosen_layers, layer_aucs


def run_fold(args, k: int, tr: np.ndarray, va: np.ndarray, train: pd.DataFrame, train_rows: np.ndarray,
             pids: list[str], y_bag: np.ndarray, device: str, fold_dir, deadline: float) -> bool:
    """Tune on outer-train positions `tr`, refit, write the outer-validation predictions of
    every model whose study is complete. False if the time budget ran out first."""
    y, sites = train["label"].to_numpy().copy(), train["meta_site"].astype(str).to_numpy()
    group = "meta_site" if args.cv == "site" else None
    inner = cv_folds(train.iloc[tr].reset_index(drop=True), group_field=group, seed=SEED + 1 + k)
    inner_pos = [(tr[a], tr[b]) for a, b in inner]                       # train positions
    inner_bags = [(train_rows[a], train_rows[b]) for a, b in inner_pos]  # bucket rows, for the probe
    site_bag = np.empty(len(y_bag), dtype=object)
    site_bag[train_rows] = sites
    t0 = time.time()

    chosen_layers, layer_aucs = stage1(args, k, inner_bags, site_bag, y_bag, pids, fold_dir)
    bags = {}
    for w in args.windows:
        suffix, extra = INSTANCES[w]
        la, lt = chosen_layers[w]
        name = f"{args.audio}-{suffix}" + (f"+{args.text}-{suffix}@{lt}" if args.text else "") \
            + ("" if args.no_extra else f"+{extra}")
        feats, owner = load_packed(name, w, pids)
        bags[w] = padded_bags(np.asarray(feats[la]), owner, train_rows, device)

    # stage 2
    storage = JournalStorage(JournalFileBackend(str(fold_dir / f"fold{k}.journal"),
                                                lock_obj=JournalFileOpenLock(str(fold_dir / f"fold{k}.journal"))))
    complete = True
    for m in args.models:
        study = optuna.create_study(
            storage=storage, study_name=f"fold{k}-{m}", direction="maximize", load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=SEED + 100 * k + args.worker,
                                               n_startup_trials=args.startup_trials),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=args.startup_trials, n_warmup_steps=1))
        for w in args.windows:
            study.enqueue_trial({**DEFAULT, "window": w}, user_attrs={"a_priori": True}, skip_if_exists=True)

        def objective(trial):
            p = suggest(trial, args.windows)
            X, mask, lengths = bags[p["window"]]
            oof, fold_aucs = np.full(len(y), np.nan), []
            for step, (a, b) in enumerate(inner_pos):
                oof[b] = fit_predict_fold(X, mask, lengths, y, a, b, [m], args.inner_seeds, p["epochs"],
                                          device, **net_kwargs(p))[m]
                fold_aucs.append(score(y[b], oof[b], sites[b], args.select))
                trial.report(float(np.nanmean(fold_aucs)), step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            keep = ~np.isnan(oof)
            return score(y[keep], oof[keep], sites[keep], args.select)

        def log(study, trial):
            n_done = sum(t.state in FINISHED for t in study.get_trials(deepcopy=False))
            value = f"{trial.value:.3f}" if trial.value is not None else f"pruned at inner fold {len(trial.intermediate_values)}"
            print(f"[fold {k}|w{args.worker}] {m} trial {n_done}/{args.n_trials}: {value}  ({describe(trial.params)})  "
                  f"best {study.best_value:.3f}  {(time.time() - t0) / 60:.1f} min", flush=True)

        def n_finished():
            return sum(t.state in FINISHED for t in study.get_trials(deepcopy=False))

        if n_finished() < args.n_trials and time.time() < deadline:
            study.optimize(objective, timeout=max(deadline - time.time(), 1), gc_after_trial=True,
                           callbacks=[log, optuna.study.MaxTrialsCallback(args.n_trials, states=FINISHED)])
        if n_finished() < args.n_trials:
            print(f"[fold {k}|w{args.worker}] {m}: time budget used up at {n_finished()}/{args.n_trials} trials; "
                  f"resubmit to continue", flush=True)
            complete = False
            continue
        running = sum(t.state == TrialState.RUNNING for t in study.get_trials(deepcopy=False))
        if running:
            print(f"[fold {k}|w{args.worker}] {m}: {running} trial(s) still running elsewhere; refitting the "
                  f"best so far (their worker refits again when it ends)", flush=True)

        # refit the best trial on the whole outer training fold
        best = study.best_trial
        p = best.params
        X, mask, lengths = bags[p["window"]]
        prob = fit_predict_fold(X, mask, lengths, y, tr, va, [m], args.seeds, p["epochs"], device,
                                seed_offset=1000, **net_kwargs(p))[m]
        a_priori = [t.value for t in study.trials if t.user_attrs.get("a_priori") and t.params["window"] == p["window"]]
        la, lt = chosen_layers[p["window"]]
        choice = {"fold": k, "model": m, "audio_layer": la, "text_layer": lt, **p, "trial": best.number,
                  "inner_score": best.value, "inner_score_a_priori": a_priori[0] if a_priori else None,
                  "n_pruned": sum(t.state == TrialState.PRUNED for t in study.trials)}
        print(f"[fold {k}|w{args.worker}] {m}: chose audio L{la} text L{lt} {describe(p)} (inner {best.value:.3f}, "
              f"a-priori setting {choice['inner_score_a_priori']})", flush=True)
        tag = f"fold{k}_{m}"
        def save_npz(f):
            with open(f, "wb") as fh:            # a file object: savez would add .npz to a path
                np.savez_compressed(fh, va=va, prob=prob)
        write_atomic(fold_dir / f"{tag}.npz", save_npz)
        write_atomic(fold_dir / f"{tag}.json", lambda f: f.write_text(json.dumps(
            {"choice": choice, "layer_aucs": layer_aucs}, indent=1,
            default=lambda v: v.item() if hasattr(v, "item") else str(v))))
        write_atomic(fold_dir / f"{tag}_trials.csv", lambda f: study.trials_dataframe().to_csv(f, index=False))
    del bags
    if device == "cuda":
        torch.cuda.empty_cache()
    return complete


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", required=True, help="speech encoder, e.g. whisper-medium (-> whisper-medium-seg)")
    ap.add_argument("--text", default=None, help="text encoder, e.g. xlmr-large; omit for audio only")
    ap.add_argument("--windows", nargs="+", default=["seg"], choices=list(INSTANCES), help="instance types searched")
    ap.add_argument("--no-extra", action="store_true", help="leave out prosody-seg / wordpros / role-seg")
    ap.add_argument("--audio-layers", nargs="+", type=int, default=None, help="candidates (default: all)")
    ap.add_argument("--text-layers", nargs="+", type=int, default=None, help="candidates (default: all)")
    ap.add_argument("--models", nargs="+", default=["MI-mean"], choices=MODELS, help="each tuned on its own")
    ap.add_argument("--n-trials", type=int, default=50, help="Optuna trials per model and outer fold")
    ap.add_argument("--startup-trials", type=int, default=10, help="random trials before TPE and pruning start")
    ap.add_argument("--select", choices=["auc", "site_auc"], default="auc", help="inner-CV selection score")
    ap.add_argument("--inner-seeds", type=int, default=2, help="initialisations per inner fit")
    ap.add_argument("--seeds", type=int, default=N_SEEDS, help="initialisations of the outer refit")
    ap.add_argument("--cv", choices=["random", "site"], default="random")
    ap.add_argument("--repeats", type=int, default=1, help="repeats of the outer 5-fold CV")
    ap.add_argument("--threshold", default="f2")
    ap.add_argument("--fold", type=int, nargs="+", default=None, help="outer folds to run (array task); "
                    "default: all, then merge")
    ap.add_argument("--limit-folds", type=int, default=None, help="smoke test: only the first N outer folds")
    ap.add_argument("--worker", type=int, default=0, help="index of this worker among those sharing a fold "
                    "(sets the sampler seed)")
    ap.add_argument("--hours", type=float, default=None, help="stop starting trials after this many hours "
                    "(resubmit to continue)")
    ap.add_argument("--merge", action="store_true", help="score the saved outer folds together")
    ap.add_argument("--jobs", type=int, default=-1, help="parallel probe fits in stage 1")
    args = ap.parse_args()
    deadline = time.time() + 3600 * args.hours if args.hours else float("inf")
    optuna.logging.set_verbosity(optuna.logging.WARNING)       # one line per trial comes from the callback

    df = load_bucket("finetune")
    pids = df["participant_id"].tolist()
    train_rows = np.flatnonzero(df["split"].to_numpy() == "train")   # test split never loaded into a bag
    train = df.iloc[train_rows].reset_index(drop=True)
    y_bag = df["label"].to_numpy()
    y = y_bag[train_rows].copy()
    folds = cv_folds(train, group_field="meta_site" if args.cv == "site" else None, n_repeats=args.repeats)

    stem = (f"tune_{args.audio}" + (f"+{args.text}" if args.text else "") + f"_{'-'.join(args.windows)}"
            + ("_noextra" if args.no_extra else "") + f"_{args.cv}_{args.select}"
            + (f"_5x{args.repeats}" if args.repeats > 1 else ""))
    out_dir = OUTPUTS / "results/tune"
    fold_dir = out_dir / stem
    fold_dir.mkdir(parents=True, exist_ok=True)

    if not args.merge:
        device = pick_device()
        todo = args.fold if args.fold is not None else list(range(len(folds)))[:args.limit_folds]
        print(f"{stem}: outer folds {todo} of {len(folds)}, worker {args.worker}, {args.n_trials} trials x "
              f"5 inner folds x {args.inner_seeds} seeds per model, models {args.models}, device={device}, "
              f"train {len(y)} ({y.sum()} positive)", flush=True)
        complete = True
        for k in todo:
            tr, va = folds[k]
            complete &= run_fold(args, k, tr, va, train, train_rows, pids, y_bag, device, fold_dir, deadline)
        if args.fold is not None or args.limit_folds or not complete:
            return

    missing = [f"fold{k}_{m}" for k in range(len(folds)) for m in args.models
               if not (fold_dir / f"fold{k}_{m}.npz").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} fold x model results not written yet, e.g. {missing[:5]}")
    prefix = f"{args.audio}" + (f"+{args.text}" if args.text else "") + "|tuned"
    calls = iter(range(len(folds)))

    def fit_predict(tr, va):                 # replays the saved outer-fold predictions, in fold order
        k = next(calls)
        out = {}
        for m in args.models:
            z = np.load(fold_dir / f"fold{k}_{m}.npz")
            if not np.array_equal(z["va"], va):
                raise SystemExit("saved folds do not match the current outer folds (bucket or --repeats changed?)")
            out[f"{prefix}|{m}"] = z["prob"]
        return out

    summary, oof, repeat_oof = run_cv(y, folds, fit_predict, threshold_metric=args.threshold,
                                      sites=train["meta_site"].to_numpy(), langs=languages(train),
                                      keep_repeats=True)
    summary.to_csv(out_dir / f"{stem}.csv")
    extra = {"prob_repeats": np.stack(list(repeat_oof.values()))} if args.repeats > 1 else {}
    np.savez_compressed(out_dir / f"{stem}_oof.npz", participant_id=train["participant_id"].to_numpy(dtype=str),
                        config=np.array(list(oof)), prob=np.stack(list(oof.values())), **extra)
    choices = pd.DataFrame([json.loads((fold_dir / f"fold{k}_{m}.json").read_text())["choice"]
                            for k in range(len(folds)) for m in args.models])
    choices.to_csv(out_dir / f"{stem}_choices.csv", index=False)
    print(f"Nested CV ({len(folds)} outer folds) -> {out_dir / stem}.csv\n{format_summary(summary)}")
    print(f"\nWhat the outer folds chose:\n{choices.to_string(index=False)}")
    for col in ("audio_layer", "text_layer", "window", "hidden", "batch_size", "epochs"):
        print(f"  {col}: {choices[col].astype(str).value_counts().to_dict()}")


if __name__ == "__main__":
    main()
