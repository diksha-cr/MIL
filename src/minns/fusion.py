"""Stage E: audio + text multiple-instance networks (docs/text_audio_mil_plan.md).

Every instance carries an audio vector and a text vector for the same time window
(TextGrid segments / words from minns.segments + minns.text, or stage-D utterances).
Same bags, folds, thresholds, metrics and training recipe as stage C (mil.py), so a
row here lines up with a mil.py row on the same instances.

Each modality is first projected to PROJ dims and layer-normalised, so a 1024-D
encoder state and a 768-D text vector start on an equal footing; extra hand-made
features (--extra, e.g. prosody-seg+role-seg) are appended to the fused vector.

  audio-attn    audio only               \\  same network with one modality, the
  text-attn     text only                /   ablations every fused row is compared with
  concat-attn   [a ; t]                      Design 2a / 1a
  gate-attn     g * a + (1 - g) * t,         Design 2b: g = sigmoid(W [a ; t]) per dimension,
                                             forced to 1 (audio) where the instance has no text
  concat-hier   [a ; t], two-level attention Design 1b, --windows word only: attention over the
  gate-hier     gated, two-level attention   words of each segment, then over the segments
  xattn-attn    cross-attention, then        --windows word only: inside each TextGrid segment,
                attention over words         every word's text attends to the audio of the
  xattn-hier    cross-attention, then        segment's words and every word's audio to their
                two-level attention          text (HEADS heads, residual + LayerNorm); then
                                             [a' ; t'] goes through the same body and pooling.
                                             Attending over the whole segment, not one word's
                                             window, makes it robust to the syllable-split timings.

The per-bag diagnostic is the mean gate for gate-* (share taken from audio) and, for
xattn-*, the mean text->audio weight on the word's own audio window (chance is 1 / words
in the segment, reported alongside).

Then the stage-C body (HIDDEN ReLU layers, dropout), gated attention pooling (Ilse et
al. 2018) and a logistic head. Adam, BCE with balanced class weights, EPOCHS epochs,
N_SEEDS initialisations averaged per fold. Nothing tuned on the folds.

Config names read <audio>|<text>|<window>|L<audio layer>|<model>. The text layer is fixed
with @ (the stage-E text probe picks it); --layers sweeps the audio layer.

    python -m minns.fusion --audio whisper-medium-seg --text xlmr-large-seg@16 --extra prosody-seg \\
        --windows seg --layers 23 24
    python -m minns.fusion --audio whisper-medium-utt --text xlmr-large-utt@16 --extra prosody+role \\
        --windows dlg --layers 23 --models concat-attn gate-attn
    python -m minns.fusion --audio whisper-medium-word --text xlmr-large-word@16 --extra wordpros \\
        --windows word --layers 23 --models audio-attn text-attn concat-attn concat-hier
    python -m minns.fusion --audio whisper-medium-word --text xlmr-large-word@16 --extra wordpros \\
        --windows word --layers 23 --models concat-hier gate-hier xattn-attn xattn-hier
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from minns.data import OUTPUTS, load_bucket
from minns.evaluation import SEED, cv_folds, format_summary, languages, run_cv
from minns.extract import pick_device
from minns.mil import (BATCH, DROPOUT, EPOCHS, HIDDEN, LR, N_SEEDS, WEIGHT_DECAY, class_weights,
                       make_standardiser, padded_bags)
from minns.probe import load_packed, window_key
from minns.segments import WORD_DIR

PROJ, HEADS = 128, 4
MODELS = ("audio-attn", "text-attn", "concat-attn", "gate-attn")
HIER_MODELS = ("concat-hier", "gate-hier")
XATTN_MODELS = ("xattn-attn", "xattn-hier")
WORD_MODELS = HIER_MODELS + XATTN_MODELS          # need the word -> segment map


# ── bags ──────────────────────────────────────────────────────────────────────

def padded_aux(values: np.ndarray, owner: np.ndarray, bags: np.ndarray, fill, device: str) -> torch.Tensor:
    """Per-instance values (N,) laid out like padded_bags -> (n_bags, T_max)."""
    starts = np.searchsorted(owner, bags, side="left")
    ends = np.searchsorted(owner, bags, side="right")
    out = np.full((len(bags), (ends - starts).max()), fill, dtype=values.dtype)
    for i, (s, e) in enumerate(zip(starts, ends)):
        out[i, :e - s] = values[s:e]
    return torch.from_numpy(out).to(device)


def word_segments(pids: list[str]) -> np.ndarray:
    """Participant-segment index of every participant word, in load_packed order."""
    return np.concatenate([np.asarray(json.loads((WORD_DIR / f"{p}.json").read_text())["word_seg"])
                           for p in pids])


def as_input(parts: list[np.ndarray]) -> np.ndarray:
    """Concatenate (N, D_k) blocks. float16 unless a float32 block holds values beyond float16's range."""
    big = any(p.dtype == np.float32 and np.nanmax(np.abs(np.where(np.isfinite(p), p, 0)), initial=0) > 6e4
              for p in parts)
    return np.concatenate([np.asarray(p, dtype=np.float32 if big else np.float16) for p in parts], axis=1)


# ── networks ──────────────────────────────────────────────────────────────────

class GatedAttention(nn.Module):
    """Ilse et al. 2018 gated attention logits, (B, T, F) -> (B, T)."""

    def __init__(self, d: int, hidden: int = 64):
        super().__init__()
        self.v, self.u, self.w = nn.Linear(d, hidden), nn.Linear(d, hidden), nn.Linear(hidden, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.w(torch.tanh(self.v(h)) * torch.sigmoid(self.u(h))).squeeze(-1)


def attend(att: GatedAttention, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    a = torch.softmax(att(h).masked_fill(~mask, float("-inf")), dim=1)
    return (a[..., None] * h).sum(1)


class CrossAttention(nn.Module):
    """Multi-head attention of x over ctx where `allowed` (B, T, T); rows with no allowed key give 0."""

    def __init__(self, d: int, heads: int = HEADS):
        super().__init__()
        self.heads = heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x, ctx, allowed):
        """(B, T, D) update for x, (B, T, T) weights averaged over heads."""
        b, t, d = x.shape
        q, k, v = (f(z).view(b, t, self.heads, d // self.heads).transpose(1, 2)
                   for f, z in ((self.q, x), (self.k, ctx), (self.v, ctx)))
        s = (q @ k.transpose(-1, -2)) / (d // self.heads) ** 0.5
        w = torch.softmax(s.masked_fill(~allowed[:, None], float("-inf")), -1).nan_to_num(0.0)
        out = (w @ v).transpose(1, 2).reshape(b, t, d)
        return self.drop(self.o(out)), w.mean(1)


class FusionNet(nn.Module):
    def __init__(self, model: str, da: int, dt: int, de: int):
        super().__init__()
        self.kind, _, self.pool = model.partition("-")
        self.split = (da, dt, de)
        self.use_a, self.use_t = self.kind != "text", self.kind != "audio"
        if self.use_a:
            self.pa = nn.Sequential(nn.Linear(da, PROJ), nn.LayerNorm(PROJ))
        if self.use_t:
            self.pt = nn.Sequential(nn.Linear(dt, PROJ), nn.LayerNorm(PROJ))
        if self.kind == "gate":
            self.gate = nn.Linear(2 * PROJ, PROJ)
        if self.kind == "xattn":
            self.t2a, self.a2t = CrossAttention(PROJ), CrossAttention(PROJ)
            self.nt, self.na = nn.LayerNorm(PROJ), nn.LayerNorm(PROJ)
        d = (PROJ if self.kind in ("gate", "audio", "text") else 2 * PROJ) + de
        layers = []
        for h in HIDDEN:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(DROPOUT)]
            d = h
        self.body = nn.Sequential(*layers)
        self.att = GatedAttention(d)
        if self.pool == "hier":
            self.seg_att = GatedAttention(d)
        self.head = nn.Linear(d, 1)

    def forward(self, x, mask, has_text, groups=None):
        """(bag probability (B,), per-bag diagnostic (B,) or None: mean gate / own-word attention)."""
        da, dt, _ = self.split
        a, t, e = x[..., :da], x[..., da:da + dt], x[..., da + dt:]
        g_bag = None
        if self.use_t:
            t = self.pt(t) * has_text[..., None]
        if self.use_a:
            a = self.pa(a)
        if self.kind == "gate":
            g = torch.sigmoid(self.gate(torch.cat([a, t], -1)))
            g = torch.where(has_text[..., None] > 0, g, torch.ones_like(g))
            h = g * a + (1 - g) * t
            m = (mask & (has_text > 0)).float()
            g_bag = (g.mean(-1) * m).sum(1) / m.sum(1).clamp(min=1)
        elif self.kind == "xattn":
            same = (groups[..., :, None] == groups[..., None, :]) & mask[..., None, :]   # key in query's segment
            texted = has_text > 0
            ta, w = self.t2a(t, a, same)                                 # text queries the segment's audio
            at, _ = self.a2t(a, t, same & texted[..., None, :])          # audio queries the segment's text
            t = self.nt(t + ta) * has_text[..., None]
            a = self.na(a + at)
            h = torch.cat([a, t], -1)
            m = (mask & texted).float()
            g_bag = (torch.diagonal(w, dim1=1, dim2=2) * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            h = {"audio": a, "text": t}.get(self.kind)
            if h is None:
                h = torch.cat([a, t], -1)
        h = self.body(torch.cat([h, e], -1))
        if self.pool != "hier":
            return torch.sigmoid(self.head(attend(self.att, h, mask))).squeeze(-1), g_bag
        # words -> segments: attention restricted to the words of each segment
        n_seg = int(groups.max()) + 1
        member = (groups[..., None] == torch.arange(n_seg, device=h.device)) & mask[..., None]   # (B, T, S)
        logits = self.att(h)[..., None].masked_fill(~member, float("-inf"))
        w = torch.softmax(logits, dim=1).nan_to_num(0.0)            # empty segments: all -inf -> 0
        segs = torch.einsum("bts,btf->bsf", w, h)
        z = attend(self.seg_att, segs, member.any(1))
        return torch.sigmoid(self.head(z)).squeeze(-1), g_bag


# ── one fold ──────────────────────────────────────────────────────────────────

def fit_predict_fold(X, mask, lengths, has_text, groups, y, tr, va, models, split, n_seeds, epochs, device):
    """{model: averaged-over-seeds probabilities for `va`}, {model: mean gate per `va` bag}."""
    tr_t, va_t = torch.as_tensor(tr, device=device), torch.as_tensor(va, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    standardise = make_standardiser(X, mask, tr_t)
    wt = class_weights(y, tr, device)

    def batch(idx):                     # trim padding to the longest bag in the batch
        t = int(lengths[idx].max())
        return (standardise(X[idx, :t]), mask[idx, :t], has_text[idx, :t],
                None if groups is None else groups[idx, :t])

    probs, gates = {}, {}
    for model_name in models:
        p_runs, g_runs = [], []
        for seed in range(n_seeds):
            torch.manual_seed(SEED + seed)
            net = FusionNet(model_name, *split).to(device)
            opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            gen = torch.Generator().manual_seed(SEED + seed)
            for _ in range(epochs):
                net.train()
                for b in torch.randperm(len(tr), generator=gen).split(BATCH):
                    idx = tr_t[b.to(device)]
                    p = net(*batch(idx))[0].clamp(1e-6, 1 - 1e-6)
                    loss = F.binary_cross_entropy(p, yt[idx], weight=wt[idx])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            net.eval()
            with torch.no_grad():
                out = [net(*batch(idx)) for idx in va_t.split(32)]
            p_runs.append(torch.cat([p for p, _ in out]).cpu().numpy())
            if out[0][1] is not None:
                g_runs.append(torch.cat([g for _, g in out]).cpu().numpy())
        probs[model_name] = np.mean(p_runs, axis=0)
        if g_runs:
            gates[model_name] = np.mean(g_runs, axis=0)
    return probs, gates


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", required=True, help="audio feature set, e.g. whisper-medium-seg")
    ap.add_argument("--text", required=True, help="text feature set at one layer, e.g. xlmr-large-seg@16")
    ap.add_argument("--extra", default=None, help="hand-made sets appended after fusion, e.g. prosody-seg+role-seg")
    ap.add_argument("--windows", nargs="+", default=["seg"], help="seg, dseg, word, utt, dlg")
    ap.add_argument("--layers", nargs="+", type=int, required=True, help="audio layers")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS + WORD_MODELS)
    ap.add_argument("--cv", choices=["random", "site"], default="random")
    ap.add_argument("--threshold", default="f2")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    device = pick_device()
    df = load_bucket("finetune")
    pids = df["participant_id"].tolist()
    train_rows = np.flatnonzero(df["split"].to_numpy() == "train")   # test split never loaded into a bag
    train = df.iloc[train_rows].reset_index(drop=True)
    y = train["label"].to_numpy().copy()
    langs = languages(train)
    folds = cv_folds(train, group_field="meta_site" if args.cv == "site" else None)
    print(f"audio {args.audio} | text {args.text} | extra {args.extra}: windows {args.windows}, "
          f"audio layers {args.layers}, models {args.models}, {args.cv} CV, {len(folds)} folds, "
          f"{args.seeds} seeds x {args.epochs} epochs, device={device}, train {len(y)} ({y.sum()} positive)")

    summaries, oofs, gate_oof, t0 = [], {}, {}, time.time()
    for w in args.windows:
        key = window_key(w)
        models = [m for m in args.models if key == "word" or m not in WORD_MODELS]
        A, owner = load_packed(args.audio, key, pids)
        T, t_owner = load_packed(args.text, key, pids)
        E, e_owner = load_packed(args.extra, key, pids) if args.extra else (np.zeros((1, len(owner), 0)), owner)
        if T.shape[0] != 1:
            raise SystemExit(f"{args.text} has {T.shape[0]} layers; pick one with @, e.g. {args.text}@16")
        if not (np.array_equal(owner, t_owner) and np.array_equal(owner, e_owner)):
            raise SystemExit(f"audio, text and extra sets do not share their {key} instances; rebuild them")
        text = np.asarray(T[0])
        has_text_all = (np.abs(text.astype(np.float32)).sum(1) > 0).astype(np.float32)
        has_text = padded_aux(has_text_all, owner, train_rows, 0.0, device)
        groups = (padded_aux(word_segments(pids), owner, train_rows, -1, device)
                  if any(m in WORD_MODELS for m in models) else None)
        print(f"\n== {key}: {has_text_all.mean():.1%} of instances have text")
        chance = np.nan
        if groups is not None:                 # 1 / (words in the segment), averaged over words
            g_np = groups.cpu().numpy()
            per = []
            for r in g_np:
                r = r[r >= 0]
                _, inv, cnt = np.unique(r, return_inverse=True, return_counts=True)
                per.append(1 / cnt[inv])
            chance = float(np.concatenate(per).mean())
        for layer in args.layers:
            X, mask, lengths = padded_bags(as_input([A[layer], text, E[0]]), owner, train_rows, device)
            split = (A.shape[2], T.shape[2], E.shape[2])
            prefix = f"{args.audio}|{args.text}|{key}|L{layer}"
            print(f"== {prefix}: {mask.shape[0]} bags, up to {mask.shape[1]} instances, dims {split}")
            fold_gates = {}

            def fit_predict(tr, va):
                probs, gates = fit_predict_fold(X, mask, lengths, has_text, groups, y, tr, va, models, split,
                                                args.seeds, args.epochs, device)
                for m, g in gates.items():
                    fold_gates.setdefault(m, np.full(len(y), np.nan))[va] = g
                return {f"{prefix}|{m}": p for m, p in probs.items()}

            summary, oof = run_cv(y, folds, fit_predict, threshold_metric=args.threshold,
                                  sites=train["meta_site"].to_numpy(), langs=langs, verbose=True)
            print(format_summary(summary))
            for m, g in fold_gates.items():        # gate: share from audio; xattn: weight on own-word audio
                by_lang = pd.Series(g).groupby(langs).mean() if langs is not None else pd.Series(dtype=float)
                what = (f"own-word text->audio attention (chance {chance:.3f})" if m.startswith("xattn")
                        else "audio gate")
                print(f"   {m} mean {what} {np.nanmean(g):.3f}; by language "
                      + ", ".join(f"{k} {v:.3f}" for k, v in by_lang.items()))
                gate_oof[f"{prefix}|{m}"] = g
            print(f"   ({(time.time() - t0) / 60:.1f} min so far)", flush=True)
            summaries.append(summary)
            oofs.update(oof)
            del X, mask, lengths
            if device == "cuda":
                torch.cuda.empty_cache()

    summary = pd.concat(summaries)
    parts = summary.index.to_series().str.split("|", expand=True)
    summary.insert(0, "window", parts[2].values)
    summary.insert(1, "layer", parts[3].str[1:].astype(int).values)
    summary.insert(2, "model", parts[4].values)

    out_dir = OUTPUTS / "results/stage_e"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fusion_{args.audio}_{args.text.replace('@', 'L')}_{'-'.join(args.windows)}_{args.cv}"
    summary.to_csv(out_dir / f"{stem}.csv")
    np.savez_compressed(out_dir / f"{stem}_oof.npz", participant_id=train["participant_id"].to_numpy(dtype=str),
                        config=np.array(list(oofs)), prob=np.stack(list(oofs.values())),
                        gate_config=np.array(list(gate_oof), dtype=str),
                        gate=np.stack(list(gate_oof.values())) if gate_oof else np.zeros((0, len(y))))
    print(f"\n{len(summary)} configs in {(time.time() - t0) / 60:.1f} min -> {out_dir / stem}.csv")
    print(f"\nAll configs by fold AUC:\n{format_summary(summary)}")


if __name__ == "__main__":
    main()
