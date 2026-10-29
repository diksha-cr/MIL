"""Stage C: multiple-instance neural networks, participant = bag, windows = instances.

Same window embeddings, folds, thresholds and metrics as the stage-B probe, so
every row here lines up with a probe row (encoder | window | layer). Difference
from the probe: the probe trains a classifier on windows that all carry the
participant's label and pools only at test time; these networks are trained
THROUGH the pooling, on the bag label, so a positive participant only needs
some of its windows to look positive.

  mi-<pool>    instance space (Wang et al. 2018 mi-Net): an MLP scores every
               window, the window probabilities are pooled into the bag probability.
  MI-<pool>    embedding space (MI-Net): window embeddings from the MLP are pooled
               component-wise into one bag vector, then classified. MI-max is the
               MINN of Alsarrani et al. 2025 (their Eq. 1).
  attention    gated attention pooling (Ilse et al. 2018): learnt weights over windows.
  lin-<pool>   linear mi-Net: one linear score per window (no hidden layers), window
               probabilities pooled, trained on the bag label with the probe's L2
               penalty (same C). Identical model class and regularisation to the probe;
               the only change is that the pooling is inside the training loss.
               Fitted by full-batch L-BFGS, so it is deterministic (no seeds).

  pools: max, mean, lse (log-mean-exp, r = LSE_R: between mean and max); lseN uses r = N.

Architecture and training follow Alsarrani et al.: hidden layers of 32, 64, 128
ReLU units, Adam, binary cross-entropy, 30 epochs, R = 5 random initialisations.
Their 5 runs are reported as mean ± SD; here the 5 runs' probabilities are averaged
per fold (one ensemble per config), which is what a deployed screener would use.
Nothing is tuned on the folds — the hyperparameters below are fixed a priori — so
the only selection is across the handful of configs in the output.
Bags are balanced by class weight in the loss, like the probe.

    python -m minns.mil --encoder whisper-medium --windows 2 5 --layers 23 24
    python -m minns.mil --encoder hubert-large --windows 20 --layers 23 --cv site
    python -m minns.mil --encoder whisper-medium --windows 2 5 --layers 23 24 --models lin-mean lin-max lin-lse5 lin-lse10
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from minns.data import OUTPUTS, load_bucket
from minns.evaluation import SEED, cv_folds, format_summary, languages, run_cv
from minns.extract import pick_device
from minns.probe import load_packed, window_key

HIDDEN = (32, 64, 128)
EPOCHS, BATCH, LR, WEIGHT_DECAY, DROPOUT = 30, 16, 1e-3, 1e-4, 0.25
N_SEEDS = 5
LSE_R = 5.0
MODELS = ("mi-max", "mi-mean", "mi-lse", "MI-max", "MI-mean", "MI-lse", "attention")
LINEAR_MODELS = ("lin-mean", "lin-max", "lin-lse5", "lin-lse10")


# ── bags ──────────────────────────────────────────────────────────────────────

def padded_bags(X: np.ndarray, owner: np.ndarray, bags: np.ndarray, device: str):
    """(n_bags, T_max, D) tensor, (n_bags, T_max) bool mask, lengths — for `bags`, in that order.

    float16 for encoder states, float32 for float32 inputs (hand-made features, whose
    ratios can exceed float16's range). Non-finite values become NaN, imputed after
    standardising (see fit_predict_fold).

    `owner` is non-decreasing (load_packed writes bags in participant order), so
    every bag is one contiguous slice of X.
    """
    starts = np.searchsorted(owner, bags, side="left")
    ends = np.searchsorted(owner, bags, side="right")
    lengths = ends - starts
    out = np.zeros((len(bags), lengths.max(), X.shape[1]), dtype=np.float32 if X.dtype == np.float32 else np.float16)
    for i, (s, e) in enumerate(zip(starts, ends)):
        out[i, :e - s] = X[s:e]
    out[~np.isfinite(out)] = np.nan
    mask = np.arange(lengths.max())[None, :] < lengths[:, None]
    return (torch.from_numpy(out).to(device), torch.from_numpy(mask).to(device),
            torch.from_numpy(lengths).to(device))


def masked_pool(h: torch.Tensor, mask: torch.Tensor, how: str) -> torch.Tensor:
    """Pool h (B, T, F) over valid instances -> (B, F)."""
    m = mask[..., None]
    if how == "max":
        return h.masked_fill(~m, float("-inf")).amax(1)
    n = mask.sum(1, keepdim=True).clamp(min=1)
    if how == "mean":
        return (h * m).sum(1) / n
    if how.startswith("lse"):          # (1/r) log mean exp(r h): mean as r -> 0, max as r -> inf
        r = float(how[3:] or LSE_R)
        z = (r * h).masked_fill(~m, float("-inf"))
        return (torch.logsumexp(z, 1) - torch.log(n)) / r
    raise ValueError(how)


# ── networks ──────────────────────────────────────────────────────────────────

class MINN(nn.Module):
    def __init__(self, dim: int, model: str):
        super().__init__()
        self.kind, _, self.pool = model.partition("-")
        layers, d = [], dim
        for h in HIDDEN:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(DROPOUT)]
            d = h
        self.body = nn.Sequential(*layers)
        if self.kind == "attention":
            self.att_v, self.att_u, self.att_w = nn.Linear(d, 64), nn.Linear(d, 64), nn.Linear(64, 1)
        self.head = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Bag probability of the positive class, (B,)."""
        h = self.body(x)
        if self.kind == "mi":
            return masked_pool(torch.sigmoid(self.head(h)), mask, self.pool).squeeze(-1)
        if self.kind == "MI":
            z = masked_pool(h, mask, self.pool)
        else:
            a = self.att_w(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h))).squeeze(-1)
            a = torch.softmax(a.masked_fill(~mask, float("-inf")), dim=1)
            z = (a[..., None] * h).sum(1)
        return torch.sigmoid(self.head(z)).squeeze(-1)


def fit_linear(Xtr, mtr, ytr, wtr, Xva, mva, pool: str, C: float) -> np.ndarray:
    """Linear mi-Net on standardised bags; returns validation-bag probabilities.

    Objective is sklearn LogisticRegression's divided by C * sum(weights):
    weighted mean bag log-loss + ||w||^2 / (2 C sum(weights)), so C means the same
    as in the probe (where every bag's windows also sum to its class weight).
    """
    w = torch.zeros(Xtr.shape[-1], device=Xtr.device, requires_grad=True)
    b = torch.zeros(1, device=Xtr.device, requires_grad=True)
    lam = 1.0 / (C * float(wtr.sum()))
    bag_prob = lambda Xb, mb: masked_pool(torch.sigmoid(Xb @ w + b)[..., None], mb, pool).squeeze(-1)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=500, tolerance_grad=1e-6, tolerance_change=1e-9,
                            history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        p = bag_prob(Xtr, mtr).clamp(1e-6, 1 - 1e-6)
        loss = (wtr * F.binary_cross_entropy(p, ytr, reduction="none")).sum() / wtr.sum() \
            + 0.5 * lam * w.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return bag_prob(Xva, mva).cpu().numpy()


# ── one fold ──────────────────────────────────────────────────────────────────

def make_standardiser(X: torch.Tensor, mask: torch.Tensor, tr_t: torch.Tensor):
    """Standardise with statistics of the training bags' instances only, skipping missing
    values; a missing value then becomes 0 (the training mean), as in the probe."""
    xs = X[tr_t].float()
    ok = mask[tr_t][..., None] & xs.isfinite()
    xs = xs.nan_to_num(0.0)
    n = ok.sum((0, 1)).clamp(min=1)
    mu = (xs * ok).sum((0, 1)) / n
    sd = (((xs - mu) * ok) ** 2).sum((0, 1)).div(n).sqrt().clamp(min=1e-6)
    del xs, ok
    return lambda x: ((x.float() - mu) / sd).nan_to_num(0.0)


def class_weights(y: np.ndarray, tr: np.ndarray, device: str) -> torch.Tensor:
    """Per-bag loss weight, both classes weighing the same in total over the training bags."""
    n_pos = float(y[tr].sum())
    class_w = torch.tensor([len(tr) / (2 * (len(tr) - n_pos)), len(tr) / (2 * n_pos)], device=device)
    return class_w[torch.as_tensor(y, device=device).long()]


def fit_predict_fold(X, mask, lengths, y, tr, va, models, n_seeds, epochs, device,
                     C: float = 0.01, seed_offset: int = 0) -> dict[str, np.ndarray]:
    """Train every model on bags `tr`, return {model: averaged-over-seeds probabilities for `va`}.

    Initialisations use seeds SEED + seed_offset + 0 .. n_seeds - 1.
    """
    tr_t, va_t = torch.as_tensor(tr, device=device), torch.as_tensor(va, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)

    standardise = make_standardiser(X, mask, tr_t)
    wt = class_weights(y, tr, device)

    def batch(idx):                     # trim padding to the longest bag in the batch
        t = int(lengths[idx].max())
        return standardise(X[idx, :t]), mask[idx, :t]

    out = {}
    linear = [m for m in models if m.startswith("lin-")]
    if linear:
        Xtr, Xva = standardise(X[tr_t]), standardise(X[va_t])
        for model_name in linear:
            out[model_name] = fit_linear(Xtr, mask[tr_t], yt[tr_t], wt[tr_t], Xva, mask[va_t],
                                         model_name.split("-", 1)[1], C)
        del Xtr, Xva

    for model_name in (m for m in models if not m.startswith("lin-")):
        probs = []
        for seed in range(n_seeds):
            torch.manual_seed(SEED + seed_offset + seed)
            net = MINN(X.shape[-1], model_name).to(device)
            opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            gen = torch.Generator().manual_seed(SEED + seed_offset + seed)
            for _ in range(epochs):
                net.train()
                for b in torch.randperm(len(tr), generator=gen).split(BATCH):
                    idx = tr_t[b.to(device)]
                    p = net(*batch(idx)).clamp(1e-6, 1 - 1e-6)
                    loss = F.binary_cross_entropy(p, yt[idx], weight=wt[idx])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            net.eval()
            with torch.no_grad():
                probs.append(torch.cat([net(*batch(idx)) for idx in va_t.split(64)]).cpu().numpy())
        out[model_name] = np.mean(probs, axis=0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--windows", nargs="+", default=["2", "5"],
                    help="window sizes in seconds, or 'utt' / 'dlg' (participant / dialogue utterances, "
                         "from minns.utterances; add +role to --encoder so the network sees who speaks)")
    ap.add_argument("--layers", nargs="+", type=int, required=True)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS + LINEAR_MODELS)
    ap.add_argument("--C", type=float, default=0.01, help="inverse L2 strength of the lin-* models (as in the probe)")
    ap.add_argument("--cv", choices=["random", "site"], default="random")
    ap.add_argument("--threshold", default="f2")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--repeats", type=int, default=1, help="repeats of the 5-fold CV, each with new folds "
                    "(> 1 writes a separate _5x<R> file; holdout.py needs the single-repeat one)")
    args = ap.parse_args()

    device = pick_device()
    df = load_bucket("finetune")
    pids = df["participant_id"].tolist()
    train_rows = np.flatnonzero(df["split"].to_numpy() == "train")   # test split never loaded into a bag
    train = df.iloc[train_rows].reset_index(drop=True)
    y = train["label"].to_numpy().copy()        # writable: torch warns on read-only arrays
    folds = cv_folds(train, group_field="meta_site" if args.cv == "site" else None, n_repeats=args.repeats)
    print(f"{args.encoder}: windows {args.windows}, layers {args.layers}, models {args.models}, "
          f"{args.cv} CV, {len(folds)} folds, {args.seeds} seeds x {args.epochs} epochs, device={device}, "
          f"train {len(y)} ({y.sum()} positive)")

    summaries, oofs, repeat_oofs, t0 = [], {}, {}, time.time()
    for w in args.windows:
        key = window_key(w)
        feats, owner = load_packed(args.encoder, key, pids)
        if not all(0 <= l < feats.shape[0] for l in args.layers):
            raise SystemExit(f"{args.encoder} has layers 0..{feats.shape[0] - 1}; asked for {args.layers}")
        for layer in args.layers:
            X, mask, lengths = padded_bags(np.asarray(feats[layer]), owner, train_rows, device)
            prefix = f"{args.encoder}|{key}|L{layer}"
            print(f"\n== {prefix}: {mask.shape[0]} bags, up to {mask.shape[1]} windows of {X.shape[-1]} dims")

            def fit_predict(tr, va):
                res = fit_predict_fold(X, mask, lengths, y, tr, va, args.models, args.seeds, args.epochs, device, args.C)
                return {f"{prefix}|{m}": p for m, p in res.items()}

            summary, oof, repeat_oof = run_cv(y, folds, fit_predict, threshold_metric=args.threshold,
                                              sites=train["meta_site"].to_numpy(),
                                              langs=languages(train), verbose=True, keep_repeats=True)
            print(format_summary(summary))
            print(f"   ({(time.time() - t0) / 60:.1f} min so far)", flush=True)
            summaries.append(summary)
            oofs.update(oof)
            repeat_oofs.update(repeat_oof)
            del X, mask, lengths
            if device == "cuda":
                torch.cuda.empty_cache()

    summary = pd.concat(summaries)
    parts = summary.index.to_series().str.split("|", expand=True)
    summary.insert(0, "window", parts[1].values)
    summary.insert(1, "layer", parts[2].str[1:].astype(int).values)
    summary.insert(2, "model", parts[3].values)

    out_dir = OUTPUTS / "results/stage_c"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"mil_{args.encoder}_{args.cv}" + ("_linear" if all(m.startswith("lin-") for m in args.models) else "") \
        + (f"_5x{args.repeats}" if args.repeats > 1 else "")
    summary.to_csv(out_dir / f"{stem}.csv")
    extra = {"prob_repeats": np.stack(list(repeat_oofs.values()))} if args.repeats > 1 else {}
    np.savez_compressed(out_dir / f"{stem}_oof.npz", participant_id=train["participant_id"].to_numpy(dtype=str),
                        config=np.array(list(oofs)), prob=np.stack(list(oofs.values())), **extra)
    print(f"\n{len(summary)} configs in {(time.time() - t0) / 60:.1f} min -> {out_dir / stem}.csv")
    print(f"\nAll configs by fold AUC:\n{format_summary(summary)}")


if __name__ == "__main__":
    main()
