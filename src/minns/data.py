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


def load_bucket(bucket: str = "finetune") -> pd.DataFrame:
    """One row per participant, sorted by participant_id.

    The sort is what makes fold assignment reproducible: CV folds are drawn over
    row positions, so the row order has to be fixed independently of how the CSV
    happens to be written.
    """
    df = pd.read_csv(BUCKETS[bucket])
    df["audio_path"] = df["audio_path"].map(lambda p: str(PIPELINE_ROOT / p))
    return df.sort_values("participant_id").reset_index(drop=True)


def participant_spans(participant_id: str) -> tuple[list[tuple[float, float]], str | None]:
    """The participant's own speech spans: the diarisation tier with the most speech.

    Same rule as mh-ML-pipeline's src/speech/extract_features.participant_segments,
    so "participant audio" means the same thing in both repos. Tiers ending in "?"
    are diarisation's unsure bucket and are ignored. Returns ([], None) when there
    is no usable transcript.
    """
    path = TRANSCRIPT_DIR / f"{participant_id}.json"
    if not path.exists():
        return [], None
    try:
        segments = (json.loads(path.read_text()).get("transcript") or {}).get("segments") or []
    except json.JSONDecodeError:
        return [], None

    durations: dict[str, float] = collections.defaultdict(float)
    for seg in segments:
        tier = str(seg.get("speaker_id") or "")
        if tier and not tier.endswith("?"):
            durations[tier] += float(seg.get("duration_sec") or 0.0)
    if not durations:
        return [], None

    tier = max(durations, key=durations.get)
    spans = [(float(s.get("start_sec") or 0.0), float(s.get("end_sec") or 0.0))
             for s in segments if str(s.get("speaker_id") or "") == tier]
    return sorted((a, b) for a, b in spans if b > a), tier


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
