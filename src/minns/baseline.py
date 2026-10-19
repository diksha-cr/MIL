"""Stage 0: reference numbers every speech model in this repo has to beat.

All on the fine-tune TRAIN split (550 participants), same folds as everything else:

  confound:site+language   one-hot site and language. What a model can score by
                           recognising WHERE someone was recorded, not how they sound.
  confound:demographics    gender, age, education.
  quality:audio            recording length, participant-speech length, number of
                           participant turns, DNSMOS SIG/BAK/OVRL.
  pipeline:whisper-L23     the mh-ML-pipeline speech block as-is
                           (whisper_finetune674.npz: Whisper-medium, last block,
                           30 s windows at 15 s hop, mean-pooled), with the pipeline's
                           speech recipe: scaler -> PCA 0.90 -> balanced LogReg, C
                           searched inside each outer fold.

    python -m minns.baseline                 # random folds and site-grouped folds
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from minns.data import OUTPUTS, PIPELINE_ROOT, load_bucket, participant_spans
from minns.evaluation import SEED, cv_folds, format_summary, run_cv


def logreg(C: float = 1.0) -> LogisticRegression:
    return LogisticRegression(C=C, solver="liblinear", class_weight="balanced", max_iter=1000, random_state=SEED)


def nested_logreg(pca: float | None) -> GridSearchCV:
    """The pipeline's speech recipe; C picked by 3-fold search on the fold's training rows."""
    steps = [StandardScaler()] + ([PCA(n_components=pca, random_state=SEED)] if pca else []) + [logreg()]
    pipe = make_pipeline(*steps)
    return GridSearchCV(pipe, {"logisticregression__C": [0.001, 0.1, 1.0, 10.0]},
                        scoring=make_scorer(fbeta_score, beta=2.0, zero_division=0),
                        cv=StratifiedKFold(3, shuffle=True, random_state=SEED), n_jobs=-1)


def sklearn_fit_predict(blocks: dict[str, tuple[np.ndarray, callable]], y: np.ndarray):
    """fit_predict for run_cv from {name: (X, model_factory)}."""
    def fit_predict(tr, va):
        out = {}
        for name, (X, factory) in blocks.items():
            model = factory().fit(X[tr], y[tr])
            out[name] = model.predict_proba(X[va])[:, 1]
        return out
    return fit_predict


def build_blocks(df: pd.DataFrame) -> dict[str, tuple[np.ndarray, callable]]:
    onehot = lambda cols: OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit_transform(df[cols].astype(str))

    spans = [participant_spans(p)[0] for p in df["participant_id"]]
    quality = pd.DataFrame({
        "speech_sec": [sum(b - a for a, b in s) for s in spans],
        "n_turns": [len(s) for s in spans],
        "SIG": df["SIG"], "BAK": df["BAK"], "OVRL": df["OVRL"],
    })
    quality = quality.fillna(quality.median()).to_numpy(float)

    demog = np.column_stack([df["meta_gender"], df["meta_age"], df["meta_education"]]).astype(float)

    z = np.load(PIPELINE_ROOT / "outputs/features/speech/whisper_finetune674.npz", allow_pickle=True)
    lookup = dict(zip(z["participant_ids"].astype(str), z["layer_0"]))
    missing = [p for p in df["participant_id"] if p not in lookup]
    if missing:
        raise SystemExit(f"{len(missing)} train participants missing from whisper_finetune674.npz")
    whisper = np.stack([lookup[p] for p in df["participant_id"]]).astype(np.float32)

    scaled = lambda C=1.0: make_pipeline(StandardScaler(), logreg(C))
    return {
        "confound:site+language": (onehot(["meta_site", "meta_language"]), lambda: logreg(1.0)),
        "confound:demographics": (demog, scaled),
        "quality:audio": (quality, scaled),
        "pipeline:whisper-L23": (whisper, lambda: nested_logreg(pca=0.90)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", default="f2", help="f2 | f1 | npv_floor:0.90 ...")
    args = ap.parse_args()

    df = load_bucket("finetune")
    train = df[df["split"] == "train"].reset_index(drop=True)
    y = train["label"].to_numpy()
    print(f"train split: {len(train)} participants, {y.sum()} positive ({y.mean():.1%}); test split not used")

    blocks = build_blocks(train)
    out_dir = OUTPUTS / "results/stage0"
    out_dir.mkdir(parents=True, exist_ok=True)
    for cv_name, group in (("random", None), ("site-grouped", "meta_site")):
        folds = cv_folds(train, group_field=group)
        summary, _ = run_cv(y, folds, sklearn_fit_predict(blocks, y), threshold_metric=args.threshold,
                            sites=train["meta_site"].to_numpy())
        summary.to_csv(out_dir / f"stage0_{cv_name}.csv")
        print(f"\n== {cv_name} CV (5-fold), threshold={args.threshold}\n{format_summary(summary)}")


if __name__ == "__main__":
    main()
