"""Stage E: TextGrid segments and words as instances, for audio + text MIL (docs/text_audio_mil_plan.md).

Every TextGrid interval with text is a segment: one speaker, one stretch of speech
(median 1.6 s, 3-4 words) with its transcript, so its audio and its text cover the
same time window by construction. A few intervals are whole monologues (up to ~100 s):
those longer than MAX_SEC are cut at word boundaries into near-equal pieces of at
most MAX_SEC, as utterances.py does. Each participant (recording) is still a bag.

  seg    the participant's own segments.
  dseg   every speaker's segments in time order; who speaks is in role-seg.
  word   the participant's words. The TextGrid has no word timings, so a segment's
         duration is split over its words in proportion to their syllables
         (utterances.syllables: aksharas, or vowel groups for Latin script). That is
         approximate within a segment and exact at its edges. Forced alignment is the
         upgrade if word-level models turn out to beat segment-level ones.

A bag never comes out empty: a recording without a usable participant segment gets
one empty-text instance over the whole recording, a participant without words one
empty-text word over their first segment (info: fallback). Text encoders give those
a zero vector.

Words are also mapped to the stage-D utterances (minns.utterances, when they have
been built): a word belongs to the utterance of its own speaker that contains its
midpoint, else to the nearest one within UTT_SNAP_SEC (the energy-based utterance
edges and the TextGrid's disagree by a few hundred ms), else to none. text.py uses this for the `<text encoder>-utt` sets (Design 2 on utterances).

Feature sets (outputs/features/windows/<set>/<pid>.npz, extract.py's layout):
  prosody-seg    utterances.PROSODY_NAMES of every segment (keys seg, dseg)
  role-seg       one-hot speaker role (keys seg, dseg)
  egemaps-seg    openSMILE eGeMAPSv02 functionals per segment (with --egemaps)
  wordpros       WORDPROS_NAMES of every participant word (key word)
  <encoder>-seg  a speech encoder's hidden states (every layer), one pass over the full
                 recording as in extract.py, mean-pooled over each segment (seg, dseg)
  <encoder>-word the same pooled over each word (key word), at least one frame

Tables: outputs/features/words/<pid>.json (segments, words, word -> segment / utterance).

    python -m minns.segments --features tables --limit 3
    python -m minns.segments --features whisper-medium     # GPU; reuses the tables
    python -m minns.text --encoder xlmr-large              # text side, see text.py
"""
from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata

import numpy as np
from joblib import Parallel, delayed

from minns.data import OUTPUTS, ROLES, SAMPLE_RATE, load_bucket, speaker_roles
from minns.utterances import (HOP, MAX_SEC, PROSODY_NAMES, SILENCE_FRAC, UTT_DIR, frame_db, lld_frames, prosody, read_audio, save,
                              span_mask, syllables)

WORD_DIR = OUTPUTS / "features/words"
WORDPROS_NAMES = ("dur_sec", "syllables", "seg_syll_rate", "position", "f0_mean", "f0_sd", "loud_mean",
                  "voiced_ratio")
UTT_SNAP_SEC = 1.0
_WORD = re.compile(r"\S+")


def is_word(tok: str) -> bool:
    """A whitespace token with at least one letter or digit (not a lone dash or ellipsis)."""
    return any(unicodedata.category(c)[0] in "LN" for c in tok)


def split_words(text: str, a: float, b: float) -> list[tuple[str, int, int, float, float, int]]:
    """(word, char start, char end, start sec, end sec, syllables) of a segment, the duration
    shared out in proportion to syllables."""
    toks = [(m.group(), m.start(), m.end()) for m in _WORD.finditer(text) if is_word(m.group())]
    if not toks:
        return []
    syl = np.array([max(syllables(w), 1) for w, _, _ in toks], dtype=float)
    edges = a + (b - a) * np.concatenate([[0], np.cumsum(syl)]) / syl.sum()
    return [(w, cs, ce, float(edges[i]), float(edges[i + 1]), int(syl[i]))
            for i, (w, cs, ce) in enumerate(toks)]


def split_long(a: float, b: float, text: str) -> list[tuple[float, float, str]]:
    """A segment longer than MAX_SEC -> near-equal pieces at word boundaries (by the
    syllable-split word times), each with its own text."""
    k = int(np.ceil((b - a) / MAX_SEC))
    ws = split_words(text, a, b)
    if k <= 1 or len(ws) < 2:
        return [(a, b, text)]
    piece = np.minimum(((np.array([(w[3] + w[4]) / 2 for w in ws]) - a) / ((b - a) / k)).astype(int), k - 1)
    out = []
    for j in np.unique(piece):
        mine = [w for w, q in zip(ws, piece) if q == j]
        out.append((mine[0][3], mine[-1][4], text[mine[0][1]:mine[-1][2]]))
    return out


# ── tables ────────────────────────────────────────────────────────────────────

def build_table(pid: str, total_sec: float, rule: str) -> dict:
    """Segments (every role, time order) and their words, plus the participant subsets."""
    segs, roles = speaker_roles(pid, rule)
    segs = [(max(a, 0.0), min(b, total_sec), t, text.strip()) for a, b, t, text in segs]
    segs = [(a2, b2, t, x) for a, b, t, text in segs if b > a and text
            for a2, b2, x in split_long(a, b, text)]
    fallback = []
    if not any(roles[t] == "participant" for _, _, t, _ in segs):
        segs.append((0.0, total_sec, "<none>", ""))      # no participant text: one empty instance
        roles = {**roles, "<none>": "participant"}
        segs.sort(key=lambda s: s[0])
        fallback.append("segment")
    seg_rows = [{"start": a, "end": b, "role": ROLES.index(roles[t]), "tier": t, "text": text}
                for a, b, t, text in segs]

    words = {k: [] for k in ("text", "start", "end", "char", "syllables", "dseg", "position", "role")}
    for i, s in enumerate(seg_rows):
        ws = split_words(s["text"], s["start"], s["end"])
        for j, (w, cs, ce, a, b, n_syl) in enumerate(ws):
            for k, v in zip(words, (w, a, b, [cs, ce], n_syl, i, j / max(len(ws) - 1, 1), s["role"])):
                words[k].append(v)
    p_seg = [i for i, s in enumerate(seg_rows) if s["role"] == 0]
    seg_of = {d: k for k, d in enumerate(p_seg)}                    # dseg index -> seg index
    p_word = [i for i, r in enumerate(words["role"]) if r == 0]
    if not p_word:                                                  # participant segments with no words
        i = p_seg[0]
        for k, v in zip(words, ("", seg_rows[i]["start"], seg_rows[i]["end"], [0, 0], 1, i, 0.0, 0)):
            words[k].append(v)
        p_word = [len(words["text"]) - 1]
        fallback.append("word")

    # word -> stage-D dialogue utterance of the same speaker containing its midpoint
    utt_path = UTT_DIR / f"{pid}.json"
    words["dlg_utt"] = [-1] * len(words["text"])
    meta = json.loads(utt_path.read_text()) if utt_path.exists() else {}
    has_utts = "dlg_spans" in meta          # stage D run before role tagging has no dialogue spans
    if has_utts:
        spans = np.asarray(meta["dlg_spans"], dtype=float).reshape(-1, 2)
        u_roles = np.asarray(meta["dlg_roles"])
        mid = (np.asarray(words["start"]) + np.asarray(words["end"])) / 2
        for w, (m, r) in enumerate(zip(mid, words["role"])):
            gap = np.where(u_roles == r, np.maximum(spans[:, 0] - m, m - spans[:, 1]).clip(min=0), np.inf)
            if len(gap) and gap.min() <= UTT_SNAP_SEC:
                words["dlg_utt"][w] = int(gap.argmin())
    return {"rule": rule, "roles": roles, "fallback": fallback, "has_utterances": has_utts,
            "segments": seg_rows, "words": words, "p_seg": p_seg, "p_word": p_word,
            "word_seg": [seg_of[words["dseg"][w]] for w in p_word]}


def seg_spans(table: dict) -> np.ndarray:
    return np.asarray([[s["start"], s["end"]] for s in table["segments"]], dtype=np.float32)


def word_spans(table: dict) -> np.ndarray:
    w = table["words"]
    return np.asarray([[w["start"][i], w["end"][i]] for i in table["p_word"]], dtype=np.float32)


def seg_sets(feats: np.ndarray, table: dict) -> dict:
    """Dialogue-segment features -> {"seg": participant subset, "dseg": all}."""
    spans = seg_spans(table)
    p = table["p_seg"]
    return {"seg": (feats[p], spans[p]), "dseg": (feats, spans)}


# ── hand-made features ────────────────────────────────────────────────────────

def frames(a: float, b: float, n: int) -> tuple[int, int]:
    s = min(int(round(a / HOP)), n - 1)
    return s, max(min(int(round(b / HOP)), n), s + 1)


def build(pid: str, audio_path: str, rule: str, egemaps: bool) -> str:
    """Table, prosody-seg, role-seg, wordpros (and egemaps-seg) for one recording. Returns a log line."""
    audio = read_audio(audio_path)
    table = build_table(pid, len(audio) / SAMPLE_RATE, rule)
    segs = table["segments"]
    db = frame_db(audio)
    n = len(db)
    lo, hi = np.percentile(db, [10, 95])
    loud_m = db > lo + SILENCE_FRAC * (hi - lo)
    f0, loud = lld_frames(audio, n)

    # Prosody per segment, speech masks per role as in utterances.segment: a frame is
    # this speaker's speech when it is loud, inside their segments and in nobody else's.
    pros = np.zeros((len(segs), len(PROSODY_NAMES)), dtype=np.float32)
    p_speech = None
    for r in range(len(ROLES)):
        idx = [i for i, s in enumerate(segs) if s["role"] == r]
        if not idx:
            continue
        mine = [(segs[i]["start"], segs[i]["end"], segs[i]["text"]) for i in idx]
        others = [(s["start"], s["end"]) for s in segs if s["role"] != r]
        other = span_mask(others, n)
        speech = loud_m & span_mask(mine, n) & ~other
        if r == 0:
            p_speech = speech
        pros[idx] = prosody([frames(a, b, n) for a, b, _ in mine], speech, other, mine, f0, loud)
    onehot = np.eye(len(ROLES), dtype=np.float32)[[s["role"] for s in segs]]

    w = table["words"]
    rows = []
    for k, i in enumerate(table["p_word"]):
        s, e = frames(w["start"][i], w["end"][i], n)
        seg = segs[w["dseg"][i]]
        sp = p_speech[s:e]
        voiced = sp & (f0[s:e] > 0)
        p = f0[s:e][voiced]
        rows.append([w["end"][i] - w["start"][i], w["syllables"][i],
                     sum(map(syllables, seg["text"].split())) / max(seg["end"] - seg["start"], 1e-3),
                     w["position"][i], p.mean() if len(p) else 0.0, p.std() if len(p) else 0.0,
                     loud[s:e][sp].mean() if sp.any() else 0.0, voiced.sum() / max(sp.sum(), 1)])
    wpros = np.asarray(rows, dtype=np.float32)

    info = {"rule": rule, "fallback": table["fallback"], "has_utterances": table["has_utterances"],
            "n_seg": len(table["p_seg"]), "n_dseg": len(segs), "n_word": len(table["p_word"]),
            "total_sec": len(audio) / SAMPLE_RATE}
    WORD_DIR.mkdir(parents=True, exist_ok=True)
    (WORD_DIR / f"{pid}.json").write_text(json.dumps({**info, **table}, ensure_ascii=False))
    save("prosody-seg", pid, seg_sets(pros[:, None, :], table),
         {**info, "names": PROSODY_NAMES}, full_key="seg")
    save("role-seg", pid, seg_sets(onehot[:, None, :], table), {**info, "names": ROLES}, full_key="seg")
    save("wordpros", pid, {"word": (wpros[:, None, :], word_spans(table))},
         {**info, "names": WORDPROS_NAMES}, full_key="word")
    if egemaps:
        import opensmile
        smile = opensmile.Smile(opensmile.FeatureSet.eGeMAPSv02, opensmile.FeatureLevel.Functionals)
        feats = np.stack([smile.process_signal(audio[int(s["start"] * SAMPLE_RATE):int(s["end"] * SAMPLE_RATE)],
                                               SAMPLE_RATE).to_numpy()[0] for s in segs])
        save("egemaps-seg", pid, seg_sets(feats[:, None, :], table), info, full_key="seg")
    return (f"{pid}  {info['n_seg']}/{info['n_dseg']} participant/dialogue segments, {info['n_word']} words"
            + (f"  fallback: {info['fallback']}" if info["fallback"] else ""))


# ── encoder states ────────────────────────────────────────────────────────────

def encode(rows, encoder_name: str) -> None:
    """Encoder hidden states over the full recording, mean-pooled over every segment and word."""
    import torch
    from minns.extract import FRAME_RATE, load_encoder, pick_device, pool_windows
    device = pick_device()
    enc = load_encoder(encoder_name, device)
    for i, (pid, path) in enumerate(rows, 1):
        table = json.loads((WORD_DIR / f"{pid}.json").read_text())
        hs = enc(read_audio(path))
        n_frames = hs.shape[1]

        def pooled(spans):
            bounds = []
            for a, b in spans:
                s = min(int(a * FRAME_RATE), n_frames - 1)
                bounds.append((s, max(min(int(round(b * FRAME_RATE)), n_frames), s + 1)))
            return pool_windows(hs, bounds)

        info = {k: table[k] for k in ("rule", "fallback", "n_seg", "n_dseg", "n_word")}
        save(f"{encoder_name}-seg", pid, seg_sets(pooled(seg_spans(table)), table), info,
             dtype=np.float16, full_key="seg")
        ws = word_spans(table)
        save(f"{encoder_name}-word", pid, {"word": (pooled(ws), ws)}, info, dtype=np.float16, full_key="word")
        if device == "mps":
            torch.mps.empty_cache()
        print(f"[{i}/{len(rows)}] {pid}  {table['n_dseg']} segments, {table['n_word']} words", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", nargs="+", default=["tables"],
                    help="tables (segments, words, prosody-seg, role-seg, wordpros) and/or an encoder "
                         "from extract.ENCODERS (needs tables first)")
    ap.add_argument("--egemaps", action="store_true", help="also eGeMAPS functionals per segment (with tables)")
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

    if "tables" in args.features:
        target = OUTPUTS / "features/windows" / ("egemaps-seg" if args.egemaps else "wordpros")
        todo = [r for r in rows if args.overwrite or not (target / f"{r[0]}.npz").exists()]
        print(f"tables for {len(todo)}/{len(rows)} participants (rule={args.rule}, egemaps={args.egemaps})",
              flush=True)
        old = [pid for pid, _ in todo if (UTT_DIR / f"{pid}.json").exists()
               and "dlg_spans" not in json.loads((UTT_DIR / f"{pid}.json").read_text())]
        if old:
            print(f"warning: {len(old)} stage-D utterance files predate role tagging (no dlg_spans); their words "
                  "get no utterance link and minns.text writes no -utt set for them. To get one, re-run "
                  "`minns.utterances --features prosody --overwrite`, then these tables with --overwrite.",
                  flush=True)
        for line in Parallel(n_jobs=args.jobs, return_as="generator")(
                delayed(build)(pid, path, args.rule, args.egemaps) for pid, path in todo):
            print(line, flush=True)

    for enc in [f for f in args.features if f != "tables"]:
        missing = [pid for pid, _ in rows if not (WORD_DIR / f"{pid}.json").exists()]
        if missing:
            raise SystemExit(f"{len(missing)} participants have no tables yet (run --features tables first)")
        todo = [r for r in rows if args.overwrite or not (OUTPUTS / "features/windows" / f"{enc}-word"
                                                          / f"{r[0]}.npz").exists()]
        print(f"{enc}: encoding {len(todo)}/{len(rows)} participants", flush=True)
        encode(todo, enc)

    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
