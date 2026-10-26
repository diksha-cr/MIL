"""Participants, labels, splits and audio, read from mh-ML-pipeline.

Nothing here re-derives a label or a split: the fine-tune bucket CSV written by
the pipeline's scripts/build_buckets.py carries `label`, the frozen `split`
(train/test) and the composite `cv_stratum` the pipeline folds on, so every
number produced in this repo is computed on exactly the participants and strata
the pipeline uses.

Set MH_PIPELINE_ROOT to point at a pipeline checkout other than ~/mh-ML-pipeline.
"""
from __future__ import annotations

import collections
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

PIPELINE_ROOT = Path(os.environ.get("MH_PIPELINE_ROOT", Path.home() / "mh-ML-pipeline"))
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = Path(os.environ.get("MINNS_OUTPUTS", REPO_ROOT / "outputs"))

BUCKETS = {
    "finetune": PIPELINE_ROOT / "outputs/features/finetune_bucket.csv",
    "pretrain": PIPELINE_ROOT / "outputs/features/pretrain_bucket.csv",
}
TRANSCRIPT_DIR = PIPELINE_ROOT / "outputs/features/text/participant_jsons"
SAMPLE_RATE = 16000
OPENER_WORDS = 3          # speaker_turns(rule="role"): shortest segment that counts as the prompt
MONOLOGUE_SHARE = 0.75


def load_bucket(bucket: str = "finetune") -> pd.DataFrame:
    """One row per participant, sorted by participant_id.

    The sort is what makes fold assignment reproducible: CV folds are drawn over
    row positions, so the row order has to be fixed independently of how the CSV
    happens to be written.
    """
    df = pd.read_csv(BUCKETS[bucket])
    df["audio_path"] = df["audio_path"].map(lambda p: str(PIPELINE_ROOT / p))
    return df.sort_values("participant_id").reset_index(drop=True)


ROLES = ("participant", "interviewer", "other")


def speaker_roles(participant_id: str, rule: str = "dominant") -> tuple[list[tuple[float, float, str, str]],
                                                                      dict[str, str]]:
    """(segments as (start, end, tier, text) in time order, {tier: role}) for one recording.

    The TextGrid tiers carry no role, so the participant has to be inferred:

      dominant  the tier with the most speech. Same rule as mh-ML-pipeline's
                src/speech/extract_features.participant_segments, so "participant
                audio" means the same thing in both repos.
      role      the tier of the first segment of at least OPENER_WORDS words reads
                the prompt, so it is the interviewer and the participant is the
                largest other tier — unless that opener holds MONOLOGUE_SHARE of all
                speech, i.e. the participant read the prompt themselves and kept
                talking. Differs from `dominant` on about a third of recordings,
                mostly ones where the interviewer keeps asking follow-up questions and
                ends up with the most speech. (Counting questions per tier was tried
                and rejected: diarisation often puts a question in the wrong tier.)

    The interviewer is the opener when that is not the participant, otherwise the
    largest remaining tier (none in a single-speaker recording). Every other tier —
    third speakers and diarisation's unsure "S?" bucket, which is never the
    participant — is "other". Returns ([], {}) without a usable transcript.
    """
    path = TRANSCRIPT_DIR / f"{participant_id}.json"
    if not path.exists():
        return [], {}
    try:
        segments = (json.loads(path.read_text()).get("transcript") or {}).get("segments") or []
    except json.JSONDecodeError:
        return [], {}
    segs = sorted(((float(s.get("start_sec") or 0.0), float(s.get("end_sec") or 0.0),
                    str(s.get("speaker_id") or ""), str(s.get("text") or "")) for s in segments),
                  key=lambda s: s[0])
    segs = [s for s in segs if s[1] > s[0] and s[2]]

    durations: dict[str, float] = collections.defaultdict(float)
    for a, b, tier, text in segs:
        if not tier.endswith("?"):
            durations[tier] += b - a
    if not durations:
        return [], {}

    opener = next((t for _, _, t, text in segs if t in durations and len(text.split()) >= OPENER_WORDS),
                  next(t for _, _, t, _ in segs if t in durations))
    if rule == "dominant" or len(durations) == 1:
        participant = max(durations, key=durations.get)
    elif rule == "role":
        if durations[opener] >= MONOLOGUE_SHARE * sum(durations.values()):
            participant = opener             # read the prompt themselves and kept talking
        else:
            participant = max((t for t in durations if t != opener), key=durations.get)
    else:
        raise ValueError(f"unknown speaker rule {rule!r}")
    rest = [t for t in durations if t != participant]
    interviewer = opener if opener != participant else max(rest, key=durations.get, default=None)
    roles = {t: "participant" if t == participant else "interviewer" if t == interviewer else "other"
             for t in {s[2] for s in segs}}
    return segs, roles


def speaker_turns(participant_id: str, rule: str = "dominant", role: str = "participant"
                  ) -> tuple[list[tuple[float, float, str]], list[tuple[float, float]], str | None]:
    """(segments of `role` as (start, end, text), every other speaker's spans, the role's tier).

    See speaker_roles for how roles are assigned. For "other" the tier is None (it can
    be several). Returns ([], [], None) without a usable transcript.
    """
    segs, roles = speaker_roles(participant_id, rule)
    mine = [(a, b, text) for a, b, t, text in segs if roles[t] == role]
    others = [(a, b) for a, b, t, _ in segs if roles[t] != role]
    tier = next((t for t, r in roles.items() if r == role), None) if role != "other" else None
    return mine, others, tier


def participant_spans(participant_id: str, rule: str = "dominant") -> tuple[list[tuple[float, float]], str | None]:
    """The participant's own speech spans under `rule` (see speaker_turns). ([], None) without a transcript."""
    mine, _, tier = speaker_turns(participant_id, rule)
    return [(a, b) for a, b, _ in mine], tier


def load_audio(row, diarise: bool = False) -> tuple[np.ndarray, dict]:
    """16 kHz mono float32 audio for one participant, plus how it was obtained.

    With diarise=True the participant's turns are cut out and concatenated (not
    muted in place, which would inject silence). Falls back to the full recording
    when there is no transcript, and says so in the returned info.
    """
    audio, sr = sf.read(row.audio_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{row.participant_id}: expected {SAMPLE_RATE} Hz, got {sr}")

    info = {"total_sec": len(audio) / sr, "tier": None, "diarised": False}
    if diarise:
        spans, tier = participant_spans(row.participant_id)
        pieces = [audio[int(a * sr):int(b * sr)] for a, b in spans]
        pieces = [p for p in pieces if len(p)]
        if pieces:
            audio = np.concatenate(pieces)
            info.update(tier=tier, diarised=True)
    info["kept_sec"] = len(audio) / sr
    return audio, info
