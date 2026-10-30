"""Optuna hyperparameter search for the audio + text MIL (early fusion, as in mil.py).

Instances are whisper-medium | xlmr-large | hand-made prosody of the same window
(seg: TextGrid segments, word: words, utt: utterances). Every stage scores 5-fold CV
on the TRAIN split, participant = bag, so no person is on both sides of a fold
(same folds as mil.py). Studies live in one SQLite file; rerunning a stage resumes it
and tops it up to --trials.

    python -m minns.tune stage1       # 4. audio layer, text layer, instance type; MIL at DEFAULTS
    python -m minns.tune stage2       # 5. MIL hyperparameters, stage-1 best upstream fixed
    python -m minns.tune baselines    # 6. linear and probe, same folds, C tuned
    python -m minns.tune select       # 7. stage-2 top 5 re-run with 5 seeds -> best.json
    python -m minns.tune test         # 8. best config on the held-out test split, ONCE
    python -m minns.tune plots        # 9. parameter importances + Optuna plots (HTML)

    optuna-dashboard sqlite:///outputs/results/tune/optuna.db     # live view while it runs
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from functools import lru_cache, partial

import numpy as np
import optuna
from optuna.trial import TrialState
from sklearn.metrics import roc_auc_score

from minns.data import OUTPUTS, load_bucket
from minns.evaluation import best_threshold, compute_metrics, cv_folds
from minns.fusion import as_input
from minns.mil import fit_predict_fold, padded_bags
from minns.probe import finite, fit_layer, load_packed

OUT = OUTPUTS / "results/tune"
STORAGE = f"sqlite:///{OUT / 'optuna.db'}"


# ── 0. config / search spaces ─────────────────────────────────────────────────

HIDDEN_CHOICES = {"small": (32, 64, 128), "wide": (64, 128, 256), "deep": (32, 64, 128, 64), "narrow": (16, 32, 64)}
WINDOWS = ["seg", "word", "utt"]
AUDIO, TEXT = "whisper-medium", "xlmr-large"
EXTRA = {"seg": "prosody-seg", "word": "wordpros", "utt": "prosody"}     # hand-made features per instance type


@dataclass(frozen=True)
class Config:                       # field names = Optuna parameter names, so replace(cfg, **trial.params) works
    hidden: str = "small"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.25
    batch: int = 16
    attn_dim: int = 64
    model: str = "MI-mean"
    lse_r: float = 5.0
    audio_layer: int = 24
    text_layer: int = 13
    windows: str = "seg"
    epochs: int = 30
    n_seeds: int = 2


DEFAULTS = Config()


# ── 1. feature cache ──────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def bucket():
    """(bucket, train rows, test rows)."""
    df = load_bucket("finetune")
    split = df["split"].to_numpy()
    return df, np.flatnonzero(split == "train"), np.flatnonzero(split == "test")


@lru_cache(maxsize=1)
def folds():
    df, train_rows, _ = bucket()
    return cv_folds(df.iloc[train_rows].reset_index(drop=True))


@lru_cache(maxsize=2)
def instance_features(audio_layer: int, text_layer: int, windows: str):
    """(N, D) [audio ; text ; hand-made] for every instance in the bucket, owner (N,) its bucket row."""
    pids = bucket()[0]["participant_id"].tolist()
    (A, owner), (T, t_owner), (E, e_owner) = (load_packed(s, windows, pids)
                                              for s in (f"{AUDIO}-{windows}", f"{TEXT}-{windows}", EXTRA[windows]))
    if not (np.array_equal(owner, t_owner) and np.array_equal(owner, e_owner)):
        raise SystemExit(f"audio, text and extra sets do not share their {windows} instances")
    return as_input([A[audio_layer], T[text_layer], E[0]]), owner


@lru_cache(maxsize=2)               # padded bags sit on the GPU, so keep only the last two
def load_features(audio_layer: int, text_layer: int, windows: str, with_test: bool = False):
    """Padded bags (X, mask, lengths) and labels y: train bags, then test bags if with_test."""
    from minns.extract import pick_device
    df, train_rows, test_rows = bucket()
    X, owner = instance_features(audio_layer, text_layer, windows)
    bags = np.concatenate([train_rows, test_rows]) if with_test else train_rows
    return (*padded_bags(X, owner, bags, pick_device()), df["label"].to_numpy()[bags].copy())


# ── 2. train one model (one fold, one seed) ───────────────────────────────────

def train_mil(cfg: Config, data, tr: np.ndarray, va: np.ndarray, seed: int) -> np.ndarray:
    """Probabilities for bags `va` of one network trained on bags `tr`."""
    X, mask, lengths, y = data
    model = {"MI-attn": "attention", "MI-lse": f"MI-lse{cfg.lse_r:g}"}.get(cfg.model, cfg.model)
    return fit_predict_fold(X, mask, lengths, y, tr, va, [model], 1, cfg.epochs, X.device, seed_offset=seed,
                            lr=cfg.lr, weight_decay=cfg.weight_decay, batch_size=cfg.batch,
                            hidden=HIDDEN_CHOICES[cfg.hidden], dropout=cfg.dropout, attn_dim=cfg.attn_dim)[model]


# ── 3. cross-validation ───────────────────────────────────────────────────────

def run_cv(cfg: Config, trial: optuna.Trial | None = None) -> tuple[float, np.ndarray]:
    """Mean over folds of the seed-averaged validation AUROC, and the out-of-fold probabilities."""
    data = load_features(cfg.audio_layer, cfg.text_layer, cfg.windows)
    y = data[3]
    scores, oof = [], np.zeros(len(y))
    for k, (tr, va) in enumerate(folds()):
        probs = [train_mil(cfg, data, tr, va, seed) for seed in range(cfg.n_seeds)]
        scores.append(np.mean([roc_auc_score(y[va], p) for p in probs]))
        oof[va] = np.mean(probs, axis=0)
        if trial is not None:           # prune across folds
            trial.report(float(np.mean(scores)), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(scores)), oof


def make_study(name: str) -> optuna.Study:
    return optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0),
                               pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
                               storage=STORAGE, study_name=name, load_if_exists=True)


def optimize(name: str, objective, n_trials: int, hours: float, start: dict | None = None) -> optuna.Study:
    """Run until the study holds n_trials finished trials (resumes), or `hours` pass."""
    study = make_study(name)
    if start and not study.trials:      # first trial = the current a-priori setting, as a reference point
        study.enqueue_trial(start)
    done = (TrialState.COMPLETE, TrialState.PRUNED)
    study.optimize(objective, n_trials=n_trials, timeout=hours * 3600,
                   callbacks=[optuna.study.MaxTrialsCallback(n_trials, states=done)])
    print(f"{name}: best {study.best_value:.3f}  {study.best_params}")
    return study


# ── 4. stage 1: upstream choices (MIL at defaults) ────────────────────────────

def objective_stage1(trial: optuna.Trial) -> float:
    cfg = replace(DEFAULTS,
                  audio_layer=trial.suggest_int("audio_layer", 12, 24, step=4),
                  text_layer=trial.suggest_int("text_layer", 9, 17, step=2),
                  windows=trial.suggest_categorical("windows", WINDOWS))
    return run_cv(cfg, trial)[0]


# ── 5. stage 2: MIL hyperparameters (upstream fixed) ──────────────────────────

@lru_cache(maxsize=1)
def best_upstream() -> dict:
    return optuna.load_study(study_name="stage1", storage=STORAGE).best_params


def objective_stage2(trial: optuna.Trial) -> float:
    model = trial.suggest_categorical("model", ["MI-mean", "MI-attn", "MI-lse"])
    cfg = replace(DEFAULTS, **best_upstream(),
                  hidden=trial.suggest_categorical("hidden", list(HIDDEN_CHOICES)),
                  lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                  weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
                  dropout=trial.suggest_float("dropout", 0.0, 0.5),
                  batch=trial.suggest_categorical("batch", [8, 16, 32, 64]),
                  attn_dim=trial.suggest_categorical("attn_dim", [32, 64, 128]),
                  model=model,
                  lse_r=trial.suggest_float("lse_r", 1.0, 20.0, log=True) if model == "MI-lse" else DEFAULTS.lse_r)
    return run_cv(cfg, trial)[0]


# ── 6. baselines (same CV, same folds) ────────────────────────────────────────

def objective_baseline(trial: optuna.Trial, model_type: str) -> float:
    """linear: logistic regression on each participant's mean instance vector.
    probe: logistic regression on instances (each with its participant's label), mean-pooled."""
    windows = trial.suggest_categorical("windows", WINDOWS)
    X, owner = instance_features(trial.suggest_int("audio_layer", 12, 24, step=4),
                                 trial.suggest_int("text_layer", 9, 17, step=2), windows)
    C = trial.suggest_float("C", 1e-3, 1e1, log=True)
    df, train_rows, _ = bucket()
    y_bag = df["label"].to_numpy()
    if model_type == "linear":
        bags = np.arange(len(df))
        starts, ends = np.searchsorted(owner, bags, "left"), np.searchsorted(owner, bags, "right")
        X, owner = np.stack([np.nanmean(finite(X[s:e]), 0) for s, e in zip(starts, ends)]), bags
    scores = []
    for tr, va in folds():
        p = fit_layer(X, owner, y_bag, train_rows[tr], train_rows[va], C, pooled=model_type == "probe")["mean"]
        scores.append(roc_auc_score(y_bag[np.sort(train_rows[va])], p))     # fit_layer returns sorted-bag order
    return float(np.mean(scores))


# ── 7. re-evaluate top configs with full seeds ────────────────────────────────

def select(k: int = 5) -> None:
    study = optuna.load_study(study_name="stage2", storage=STORAGE)
    top = sorted(study.get_trials(states=(TrialState.COMPLETE,)), key=lambda t: t.value, reverse=True)[:k]
    results = []
    for t in top:
        cfg = replace(DEFAULTS, **best_upstream(), **t.params, n_seeds=5)
        score, oof = run_cv(cfg)
        print(f"trial {t.number:3d}: 2 seeds {t.value:.3f} -> 5 seeds {score:.3f}   {t.params}", flush=True)
        results.append((score, cfg, oof))
    score, cfg, oof = max(results, key=lambda r: r[0])
    # if the top scores are within noise, prefer the config sitting in a broad good region (see `plots`)
    (OUT / "best.json").write_text(json.dumps({"cv_auc": score, **asdict(cfg)}, indent=2))
    np.save(OUT / "best_oof.npy", oof)
    print(f"best: CV AUC {score:.3f} -> {OUT / 'best.json'}")


# ── 8. final test (once) ──────────────────────────────────────────────────────

def test() -> None:
    from minns.holdout import auc_ci
    best = json.loads((OUT / "best.json").read_text())
    best.pop("cv_auc")
    cfg = Config(**best)
    data = load_features(cfg.audio_layer, cfg.text_layer, cfg.windows, with_test=True)
    y, n_train = data[3], len(bucket()[1])
    tr, te = np.arange(n_train), np.arange(n_train, len(y))
    probs = np.mean([train_mil(cfg, data, tr, te, seed) for seed in range(cfg.n_seeds)], axis=0)
    thr = best_threshold(y[tr], np.load(OUT / "best_oof.npy"), "f2")    # chosen on train out-of-fold, never on test
    m = compute_metrics(y[te], (probs >= thr).astype(int), probs)
    lo, hi = auc_ci(y[te], probs)
    print(f"{cfg}\ntest {len(te)} ({int(y[te].sum())} positive): AUROC {m['auc']:.3f} [{lo:.3f}, {hi:.3f}]  "
          f"AUPRC {m['ap']:.3f}  F2 {m['f2']:.3f} at threshold {thr:.3f}  "
          f"(recall {m['recall']:.3f}, specificity {m['specificity']:.3f})")


# ── 9. diagnostics ────────────────────────────────────────────────────────────

def plots() -> None:
    import optuna.visualization as vis
    for name in optuna.get_all_study_names(STORAGE):
        study = optuna.load_study(study_name=name, storage=STORAGE)
        if len(study.get_trials(states=(TrialState.COMPLETE,))) < 2:
            continue
        print(f"{name}: best {study.best_value:.3f}  {study.best_params}\n   importance "
              + ", ".join(f"{k} {v:.2f}" for k, v in optuna.importance.get_param_importances(study).items()))
        per_fold = any(t.intermediate_values for t in study.trials)     # MIL studies report every fold
        for plot in (vis.plot_optimization_history, vis.plot_param_importances, vis.plot_slice,
                     vis.plot_parallel_coordinate, *[vis.plot_intermediate_values] * per_fold):
            plot(study).write_html(OUT / f"{name}_{plot.__name__.removeprefix('plot_')}.html")
    print(f"-> {OUT}/*.html")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["stage1", "stage2", "baselines", "select", "test", "plots"])
    ap.add_argument("--trials", type=int, help="total trials per study (default 30; stage2 80)")
    ap.add_argument("--hours", type=float, default=23, help="stop after this long (resubmit to continue)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.INFO)

    up = ("audio_layer", "text_layer", "windows")
    if args.stage == "stage1":
        optimize("stage1", objective_stage1, args.trials or 30, args.hours,
                 start={k: getattr(DEFAULTS, k) for k in up})
    elif args.stage == "stage2":
        print(f"upstream from stage 1: {best_upstream()}")
        optimize("stage2", objective_stage2, args.trials or 80, args.hours,
                 start={k: getattr(DEFAULTS, k) for k in ("model", "hidden", "lr", "weight_decay", "dropout",
                                                          "batch", "attn_dim")})
    elif args.stage == "baselines":
        for m in ("linear", "probe"):
            optimize(m, partial(objective_baseline, model_type=m), args.trials or 30, args.hours,
                     start={k: getattr(DEFAULTS, k) for k in up} | {"C": 0.01})
    elif args.stage == "select":
        select()
    elif args.stage == "test":
        test()
    else:
        plots()


if __name__ == "__main__":
    main()
