"""Stage D: utterance instances — speech cut at pauses, every utterance tagged with its speaker's role.

Instead of fixed 2-20 s windows, every instance is one utterance: a stretch of one
speaker's speech bounded by pauses. Each participant (recording) is still a bag.
Two instance sets per recording:

  utt   the participant's own utterances only.
  dlg   the whole dialogue: participant, interviewer and "other" utterances
        (third speakers, diarisation's unsure tier) in time order. Which is which
        is in the `role` feature set (one-hot ROLES); add it with +role.

Segmentation, per recording and per role:
  1. Roles from the TextGrid tiers (data.speaker_roles, rule "role" by default).
  2. A 10 ms frame is speech of a role when it lies in one of that role's segments,
     in no other speaker's segment (overlapped speech is nobody's), and its energy
     is above the recording's silence threshold (SILENCE_FRAC of the way from the
     10th to the 95th energy percentile, in dB).
  3. Speech runs separated by a silence shorter than PAUSE_SEC are joined
     (never across another speaker's turn). The result is one utterance per run.
  4. Utterances longer than MAX_SEC are split at their longest internal pause
     (evenly, if there is none); utterances with less than MIN_SPEECH_SEC of speech
     are dropped.

Feature sets, one per output folder, all on the same utterances so they can be
concatenated (probe / mil: --encoder egemaps+prosody+role):

  prosody          PROSODY_NAMES — timing and pitch measured directly:
                   speaking rate (syllables per second of the transcribed segments,
                   pauses included; missing without a transcript; syllables counted
                   from the TextGrid text: aksharas for Indic scripts, vowel groups
                   for English), pauses inside the utterance, the pause before it
                   and whether it answers another speaker's turn (pause_before_sec
                   is then the response latency; back-channels under TURN_SEC
                   don't count as turns), pitch
                   level / spread / range / slope / final slope (semitones),
                   loudness level and spread, voiced share.
  egemaps          openSMILE eGeMAPSv02 functionals (88-D) of the utterance audio:
                   pitch, loudness, jitter, shimmer, HNR, formants, spectral balance.
  role             one-hot speaker role (participant, interviewer, other).
  <encoder>-utt    a speech encoder's hidden states (every layer), run once over
                   the full recording as in extract.py, mean-pooled over the utterance.

Output (one file per participant, the layout extract.py writes, keys "utt" and "dlg"):
    outputs/features/utterances/<pid>.json        spans, roles (written once, reused by encoders)
    outputs/features/windows/<set>/<pid>.npz      utt / dlg (n, layers, dim) float32 (encoders:
                                                  float16), utt_span / dlg_span (n, 2),
                                                  full (layers, dim, participant mean), info

    python -m minns.utterances --features prosody egemaps --limit 3
    python -m minns.utterances --features whisper-medium          # GPU; reuses the spans
    python -m minns.probe --encoder egemaps+prosody+role --windows dlg utt
"""
from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata

import numpy as np
import soundfile as sf
from joblib import Parallel, delayed

from minns.data import OUTPUTS, ROLES, SAMPLE_RATE, load_bucket, speaker_roles

HOP = 0.01                 # frame step, seconds (also openSMILE's LLD step)
WIN = 0.025                # energy frame length, seconds
SILENCE_FRAC = 0.4         # silence threshold between the 10th and 95th dB percentiles
PAUSE_SEC = 0.5            # a silence at least this long ends an utterance
MIN_PAUSE_SEC = 0.15       # shortest silence counted as a pause inside an utterance
MAX_SEC = 15.0
MIN_SPEECH_SEC = 1.0
TURN_SEC = 1.0             # another speaker's run this long is a turn, shorter is a back-channel
PROSODY_NAMES = ("dur_sec", "speech_ratio", "syll_rate", "pauses_per_sec", "mean_pause_sec",
                 "pause_before_sec", "after_other", "f0_mean", "f0_sd", "f0_range", "f0_slope",
                 "f0_final_slope", "loud_mean", "loud_sd", "voiced_ratio")
UTT_DIR = OUTPUTS / "features/utterances"
WIN_DIR = OUTPUTS / "features/windows"


# ── text ──────────────────────────────────────────────────────────────────────

_LATIN_WORD, _VOWELS = re.compile(r"[a-z]+"), re.compile(r"[aeiouy]+")


def syllables(text: str) -> int:
    """Syllable count: one per Indic akshara (a consonant or independent vowel not
    followed by a virama, so a conjunct counts once), one per vowel group of a Latin word."""
    n = 0
    for i, ch in enumerate(text):
        if unicodedata.category(ch) == "Lo" and not (
                i + 1 < len(text) and "VIRAMA" in unicodedata.name(text[i + 1], "")):
            n += 1
    for w in _LATIN_WORD.findall(text.lower()):
        k = len(_VOWELS.findall(w))
        n += max(1, k - (k > 1 and w.endswith("e") and not w.endswith("le")))   # silent final e
    return n


# ── segmentation ──────────────────────────────────────────────────────────────

def frame_db(audio: np.ndarray) -> np.ndarray:
    """Energy per HOP frame in dB."""
    win, hop = int(WIN * SAMPLE_RATE), int(HOP * SAMPLE_RATE)
    n = max(1 + (len(audio) - win) // hop, 1)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = np.pad(audio, (0, max(idx.max() + 1 - len(audio), 0)))[idx]
    return 10 * np.log10((frames ** 2).mean(1) + 1e-10)


def span_mask(spans, n: int) -> np.ndarray:
    m = np.zeros(n, bool)
    for a, b, *_ in spans:
        m[int(round(a / HOP)):int(round(b / HOP))] = True
    return m


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) frame ranges where mask is True."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def split_long(s: int, e: int, speech: np.ndarray) -> list[tuple[int, int]]:
    """Split [s, e) at its longest internal silence until every piece is at most MAX_SEC."""
    max_f = int(MAX_SEC / HOP)
    if e - s <= max_f:
        return [(s, e)]
    gaps = runs(~speech[s:e])
    gaps = [(a, b) for a, b in gaps if a > 0 and b < e - s]
    if gaps:
        a, b = max(gaps, key=lambda g: g[1] - g[0])
        return split_long(s, s + a, speech) + split_long(s + b, e, speech)
    k = int(np.ceil((e - s) / max_f))
    edges = np.linspace(s, e, k + 1).round().astype(int)
    return list(zip(edges[:-1], edges[1:]))


def segment(audio: np.ndarray, mine, others, fallback: bool = True
            ) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    """Utterance frame ranges of `mine`, plus the frame-level speech mask and other-speaker mask.

    fallback=True (the participant) never returns an empty list, see below; without
    it (interviewer, other) a speaker with no usable speech simply has no utterances.
    """
    db = frame_db(audio)
    n = len(db)
    lo, hi = np.percentile(db, [10, 95])
    loud = db > lo + SILENCE_FRAC * (hi - lo)
    other = span_mask(others, n)
    mine_m = span_mask(mine, n) if mine or not fallback else np.ones(n, bool)   # no transcript: all of it
    speech = loud & mine_m & ~other

    utts, cur = [], None
    for s, e in runs(speech):
        if cur and (s - cur[1]) * HOP < PAUSE_SEC and not other[cur[1]:s].any():
            cur = (cur[0], e)
        else:
            if cur:
                utts.append(cur)
            cur = (s, e)
    if cur:
        utts.append(cur)
    utts = [p for s, e in utts for p in split_long(s, e, speech)]
    kept = [(s, e) for s, e in utts if speech[s:e].sum() * HOP >= MIN_SPEECH_SEC]
    if not fallback:
        return kept, speech, other
    # Never an empty bag (near-silent audio or a transcript with almost nothing for
    # the participant): fall back to the longest run, then the participant's own
    # segments as transcribed, then the whole recording. build() records which.
    if not kept and utts:
        kept = [max(utts, key=lambda u: speech[u[0]:u[1]].sum())]
    if not kept and mine:
        frames = [(int(round(a / HOP)), int(round(b / HOP))) for a, b, _ in mine]
        kept = [p for s, e in frames if e > s for p in split_long(s, e, speech)]
    if not kept:
        kept = split_long(0, n, speech)
    return kept, speech, other


# ── prosody ───────────────────────────────────────────────────────────────────

def lld_frames(audio: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """openSMILE eGeMAPS LLDs on the HOP grid: F0 (semitones, 0 = unvoiced), loudness."""
    import opensmile
    smile = opensmile.Smile(opensmile.FeatureSet.eGeMAPSv02, opensmile.FeatureLevel.LowLevelDescriptors)
    lld = smile.process_signal(audio, SAMPLE_RATE)
    f0 = np.zeros(n)
    loud = np.zeros(n)
    m = min(n, len(lld))
    f0[:m] = lld["F0semitoneFrom27.5Hz_sma3nz"].to_numpy()[:m]
    loud[:m] = lld["Loudness_sma3"].to_numpy()[:m]
    return f0, loud


def slope(y: np.ndarray, t: np.ndarray) -> float:
    return float(np.polyfit(t, y, 1)[0]) if len(y) >= 5 and np.ptp(t) > 0 else 0.0


def prosody(utts, speech, other, mine, f0, loud) -> np.ndarray:
    """(n_utts, len(PROSODY_NAMES)) — see the module docstring."""
    n = len(speech)
    # Syllables per second of the transcribed segment they were said in, over its whole
    # duration. Dividing by detected speech frames instead inflated the rate wherever the
    # interviewer overlapped (those frames are removed from `speech`, the words are not).
    density, covered = np.zeros(n), np.zeros(n, bool)
    for a, b, text in mine:
        s, e = int(round(a / HOP)), int(round(b / HOP))
        if e > s:
            density[s:e] = syllables(text) / ((e - s) * HOP)
            covered[s:e] = True
    # Another speaker's run counts as a turn (a question the utterance answers) only
    # if it lasts TURN_SEC; shorter ones are back-channels ("hmm") over the participant.
    turn = np.zeros(n, bool)
    for a, b in runs(other):
        turn[a:b] = (b - a) * HOP >= TURN_SEC
    any_speech = speech | turn
    rows = []
    for s, e in utts:
        sp = speech[s:e]
        n_sp = max(sp.sum(), 1)
        pauses = [b - a for a, b in runs(~sp) if a > 0 and b < e - s and (b - a) * HOP >= MIN_PAUSE_SEC]
        before = np.flatnonzero(any_speech[:s])
        last = before[-1] if len(before) else None
        voiced = sp & (f0[s:e] > 0)
        t = np.flatnonzero(voiced) * HOP
        p = f0[s:e][voiced]
        tail = p[int(len(p) * 0.7):], t[int(len(t) * 0.7):]
        rows.append([
            (e - s) * HOP,
            sp.mean(),
            density[s:e][covered[s:e]].mean() if covered[s:e].any() else np.nan,
            len(pauses) / ((e - s) * HOP),
            np.mean(pauses) * HOP if pauses else 0.0,
            min((s - last) * HOP, 5.0) if last is not None else 5.0,
            float(last is not None and turn[last]),
            p.mean() if len(p) else 0.0,
            p.std() if len(p) else 0.0,
            np.ptp(np.percentile(p, [5, 95])) if len(p) else 0.0,
            slope(p, t),
            slope(*tail),
            loud[s:e][sp].mean() if sp.any() else 0.0,
            loud[s:e][sp].std() if sp.any() else 0.0,
            voiced.sum() / n_sp,
        ])
    return np.asarray(rows, dtype=np.float32)


# ── per participant ───────────────────────────────────────────────────────────

def read_audio(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz, got {sr}")
    return audio


def save(name: str, pid: str, sets: dict[str, tuple[np.ndarray, np.ndarray]], info: dict,
         dtype=np.float32) -> None:
    """{key: (features (n, layers, dim), spans (n, 2))} -> outputs/features/windows/<name>/<pid>.npz,
    extract.py's layout; `full` is the mean over the participant's utterances.

    Hand-made features stay float32: some eGeMAPS columns are ratios (stddevNorm =
    sd / mean) that reach 3e4 when the mean is near 0 and overflow float16 (max 65504).
    """
    d = WIN_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{pid}.tmp.npz"
    arrays = {}
    for key, (feats, spans) in sets.items():
        arrays[key] = feats.astype(dtype)
        arrays[f"{key}_span"] = spans.astype(np.float32)
    arrays["full"] = sets["utt"][0].astype(np.float64).mean(0).astype(dtype)
    np.savez(tmp, info=json.dumps(info), **arrays)
    tmp.rename(d / f"{pid}.npz")


def build(pid: str, audio_path: str, rule: str, egemaps: bool) -> str:
    """Segment one recording by role, write its spans, prosody, roles (and eGeMAPS). Returns a log line."""
    audio = read_audio(audio_path)
    segs, roles = speaker_roles(pid, rule)
    f0 = loud = None
    utts, pros, role_idx, part = [], [], [], {}
    for r, name in enumerate(ROLES):
        mine = [(a, b, text) for a, b, t, text in segs if roles[t] == name]
        others = [(a, b) for a, b, t, _ in segs if roles[t] != name]
        if name != "participant" and not mine:
            continue
        u, speech, other = segment(audio, mine, others, fallback=name == "participant")
        if f0 is None:
            f0, loud = lld_frames(audio, len(speech))
        if name == "participant":
            part = {"speech": speech, "utts": u, "mine": mine}
        if u:
            utts += u
            pros.append(prosody(u, speech, other, mine, f0, loud))
            role_idx += [r] * len(u)
    order = np.argsort([s for s, _ in utts], kind="stable")
    spans = (np.asarray(utts, dtype=np.float32) * HOP)[order]
    pros = np.concatenate(pros)[order]
    role_idx = np.asarray(role_idx)[order]
    is_p = role_idx == 0
    onehot = np.eye(len(ROLES), dtype=np.float32)[role_idx]

    speech = part["speech"]
    kept_speech = sum(int(speech[s:e].sum()) for s, e in part["utts"]) * HOP
    info = {"tier": next((t for t, r in roles.items() if r == "participant"), None), "roles": roles,
            "rule": rule, "has_transcript": bool(part["mine"]), "n_utts": int(is_p.sum()),
            "n_dlg": {name: int((role_idx == r).sum()) for r, name in enumerate(ROLES)},
            "fallback": kept_speech < MIN_SPEECH_SEC * int(is_p.sum()),
            "speech_sec": float(speech.sum() * HOP), "kept_sec": float(kept_speech),
            "total_sec": len(audio) / SAMPLE_RATE}
    UTT_DIR.mkdir(parents=True, exist_ok=True)
    (UTT_DIR / f"{pid}.json").write_text(json.dumps({**info, "spans": spans[is_p].tolist(),
                                                     "dlg_spans": spans.tolist(), "dlg_roles": role_idx.tolist(),
                                                     "prosody_names": PROSODY_NAMES}))

    def both(feats):                    # dialogue set and its participant subset
        return {"utt": (feats[is_p], spans[is_p]), "dlg": (feats, spans)}

    save("prosody", pid, both(pros[:, None, :]), {**info, "names": PROSODY_NAMES})
    save("role", pid, both(onehot[:, None, :]), {**info, "names": ROLES})
    if egemaps:
        import opensmile
        smile = opensmile.Smile(opensmile.FeatureSet.eGeMAPSv02, opensmile.FeatureLevel.Functionals)
        feats = np.stack([smile.process_signal(audio[int(a * SAMPLE_RATE):int(b * SAMPLE_RATE)],
                                               SAMPLE_RATE).to_numpy()[0] for a, b in spans])
        save("egemaps", pid, both(feats[:, None, :]), info)
    return (f"{pid}  participant={info['tier']}  " + " ".join(f"{k}:{v}" for k, v in info["n_dlg"].items())
            + f" utts  {info['speech_sec']:.0f}/{info['total_sec']:.0f}s participant speech")


def encode(rows, encoder_name: str) -> None:
    """Encoder hidden states over the full recording, mean-pooled over each saved utterance."""
    import torch
    from minns.extract import FRAME_RATE, load_encoder, pick_device, pool_windows
    device = pick_device()
    enc = load_encoder(encoder_name, device)
    name = f"{encoder_name}-utt"
    for i, (pid, path) in enumerate(rows, 1):
        meta = json.loads((UTT_DIR / f"{pid}.json").read_text())
        if "dlg_spans" not in meta:
            raise SystemExit(f"{pid}: segmented before role tagging; re-run --features prosody egemaps --overwrite")
        spans = np.asarray(meta["dlg_spans"], dtype=np.float32)
        is_p = np.asarray(meta["dlg_roles"]) == 0
        hs = enc(read_audio(path))
        n_frames, bounds = hs.shape[1], []
        for a, b in spans:
            s = min(int(a * FRAME_RATE), n_frames - 1)
            bounds.append((s, max(min(int(round(b * FRAME_RATE)), n_frames), s + 1)))
        feats = pool_windows(hs, bounds)
        save(name, pid, {"utt": (feats[is_p], spans[is_p]), "dlg": (feats, spans)},
             {k: meta[k] for k in ("tier", "rule", "n_utts", "n_dlg")}, dtype=np.float16)
        if device == "mps":
            torch.mps.empty_cache()
        print(f"[{i}/{len(rows)}] {pid}  {len(spans)} utterances, {int(is_p.sum())} participant", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", nargs="+", default=["prosody", "egemaps"],
                    help="prosody, egemaps, and/or an encoder from extract.ENCODERS (needs prosody run first)")
    ap.add_argument("--rule", choices=["role", "dominant"], default="role", help="participant tier rule")
    ap.add_argument("--bucket", default="finetune", choices=["finetune", "pretrain"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    df = load_bucket(args.bucket)
    if args.limit:
        df = df.head(args.limit)
    rows = list(zip(df["participant_id"], df["audio_path"]))
    t0 = time.time()

    if {"prosody", "egemaps"} & set(args.features):
        egemaps = "egemaps" in args.features
        target = WIN_DIR / ("egemaps" if egemaps else "prosody")
        todo = [r for r in rows if args.overwrite or not (target / f"{r[0]}.npz").exists()]
        print(f"segmenting {len(todo)}/{len(rows)} participants (rule={args.rule}, egemaps={egemaps})", flush=True)
        for line in Parallel(n_jobs=args.jobs, return_as="generator")(
                delayed(build)(pid, path, args.rule, egemaps) for pid, path in todo):
            print(line, flush=True)

    for enc in [f for f in args.features if f not in ("prosody", "egemaps")]:
        missing = [pid for pid, _ in rows if not (UTT_DIR / f"{pid}.json").exists()]
        if missing:
            raise SystemExit(f"{len(missing)} participants not segmented yet (run --features prosody first)")
        todo = [r for r in rows if args.overwrite or not (WIN_DIR / f"{enc}-utt" / f"{r[0]}.npz").exists()]
        print(f"{enc}: encoding {len(todo)}/{len(rows)} participants", flush=True)
        encode(todo, enc)

    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
