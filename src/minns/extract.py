"""Window-level speech-foundation-model embeddings, every layer.

For each participant the encoder runs ONCE over the full interview recording
(interviewer and participant, as in both papers) in 30 s chunks, and its
frame-level hidden states (50 frames/s for every encoder here) are
mean-pooled into windows of each requested length with a hop of half the
window, as in de Gennes et al. 2024. Every window therefore carries up to 30 s of
encoder context; "window size" here is the instance granularity seen by the
probe / MIL model, not the receptive field of the encoder. Running the encoder
per padded window instead would cost ~50x more for Whisper, which always
consumes a 30 s input.

Layer indexing: 0 is the input to the first transformer block (after the conv
front end and positional embedding), k is the output of block k. mh-ML-pipeline
numbers Whisper layers by block index, so its "layer 23" is layer 24 here.

Output, one file per participant:
    outputs/features/windows/<encoder>[-diarised]/<participant_id>.npz
        w<sec>       (n_windows, n_layers, dim) float16
        w<sec>_span  (n_windows, 2) start/end seconds within the audio used
        full         (n_layers, dim) float16 — mean over every frame
        info         JSON: total/kept seconds, diarised, tier

    python -m minns.extract --encoder whisper-medium
    python -m minns.extract --encoder mhubert-147 --windows 2 5 10 20 --limit 3
    python -m minns.extract --encoder xlsr-300m --diarise    # participant turns only
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from minns.data import OUTPUTS, SAMPLE_RATE, load_audio, load_bucket

CHUNK_SEC = 30
FRAME_RATE = 50          # encoder frames per second, both encoders
ENCODERS = ("whisper-medium", "xlsr-300m", "mhubert-147", "hubert-large")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class WhisperEncoder:
    def __init__(self, size: str, device: str):
        import whisper
        self._whisper = whisper
        # Loaded on CPU first: the full model registers a sparse alignment-heads
        # buffer that MPS cannot hold. Only the encoder is moved to the device.
        model = whisper.load_model(size, device="cpu")
        self.enc = model.encoder.to(device).eval()
        self.n_mels = model.dims.n_mels
        self.device = device

    @torch.no_grad()
    def __call__(self, audio: np.ndarray) -> torch.Tensor:
        """(n_layers, n_frames, dim) for the whole of `audio`, padding frames dropped."""
        w = self._whisper
        step = CHUNK_SEC * SAMPLE_RATE
        chunks = [audio[i:i + step] for i in range(0, len(audio), step)]
        valid = [int(np.ceil(len(c) / (SAMPLE_RATE / FRAME_RATE))) for c in chunks]
        mel = torch.stack([w.log_mel_spectrogram(w.pad_or_trim(torch.from_numpy(c)), n_mels=self.n_mels)
                           for c in chunks]).to(self.device)

        enc = self.enc
        x = F.gelu(enc.conv1(mel))
        x = F.gelu(enc.conv2(x)).permute(0, 2, 1)
        x = (x + enc.positional_embedding).to(x.dtype)
        states = [x]
        for block in enc.blocks:
            x = block(x)
            states.append(x)
        hs = torch.stack(states)                                  # (L, B, 1500, D)
        return torch.cat([hs[:, b, :n] for b, n in enumerate(valid)], dim=1)


class HFEncoder:
    """wav2vec2 / HuBERT family: raw waveform in, hidden_states[0] is the input to block 1."""

    def __init__(self, name: str, device: str):
        from transformers import AutoModel
        # A copy under outputs/models/<repo name> wins over the hub cache — for
        # when the hub client cannot download and the files were fetched by hand.
        local = OUTPUTS / "models" / name.split("/")[-1]
        self.model = AutoModel.from_pretrained(local if local.exists() else name).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, audio: np.ndarray) -> torch.Tensor:
        step = CHUNK_SEC * SAMPLE_RATE
        out = []
        for i in range(0, len(audio), step):
            c = audio[i:i + step]
            if len(c) < 400:                                      # shorter than one conv receptive field
                continue
            c = (c - c.mean()) / np.sqrt(c.var() + 1e-7)          # Wav2Vec2FeatureExtractor's do_normalize
            x = torch.from_numpy(c).to(self.device)[None]
            hs = self.model(x, output_hidden_states=True).hidden_states
            out.append(torch.stack(hs)[:, 0])                     # (L, T, D)
        return torch.cat(out, dim=1)


def load_encoder(name: str, device: str):
    if name == "whisper-medium":
        return WhisperEncoder("medium", device)
    if name == "xlsr-300m":
        return HFEncoder("facebook/wav2vec2-xls-r-300m", device)
    if name == "mhubert-147":
        return HFEncoder("utter-project/mHuBERT-147", device)
    if name == "hubert-large":          # the HuBERT-L checkpoint de Gennes et al. probe (English, ASR-fine-tuned)
        return HFEncoder("facebook/hubert-large-ls960-ft", device)
    raise ValueError(f"unknown encoder {name!r}; choose from {ENCODERS}")


def window_bounds(n_frames: int, win_sec: float) -> list[tuple[int, int]]:
    """Frame ranges of win_sec windows at half-window hop; a final window is
    aligned to the end so the tail is never dropped. Audio shorter than one
    window becomes a single window."""
    win = int(round(win_sec * FRAME_RATE))
    if n_frames <= win:
        return [(0, n_frames)]
    hop = max(win // 2, 1)
    starts = list(range(0, n_frames - win + 1, hop))
    if starts[-1] + win < n_frames:
        starts.append(n_frames - win)
    return [(s, s + win) for s in starts]


def pool_windows(hs: torch.Tensor, bounds: list[tuple[int, int]]) -> np.ndarray:
    """Mean of hs (L, T, D) over each frame range -> (n_windows, L, D) float16."""
    cs = F.pad(hs.float().cumsum(dim=1), (0, 0, 1, 0))          # (L, T+1, D)
    s = torch.tensor([b[0] for b in bounds], device=hs.device)
    e = torch.tensor([b[1] for b in bounds], device=hs.device)
    means = (cs[:, e] - cs[:, s]) / (e - s).to(cs.dtype)[None, :, None]
    return means.permute(1, 0, 2).to(torch.float16).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", choices=ENCODERS, required=True)
    ap.add_argument("--bucket", default="finetune", choices=["finetune", "pretrain"])
    ap.add_argument("--windows", type=float, nargs="+", default=[2, 5, 10, 20])
    ap.add_argument("--diarise", action="store_true",
                    help="participant's own turns only (dominant diarisation tier) instead of the full recording")
    ap.add_argument("--limit", type=int, default=None, help="first N participants only (smoke test)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    tag = args.encoder + ("-diarised" if args.diarise else "")
    out_dir = OUTPUTS / "features/windows" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_bucket(args.bucket)
    if args.limit:
        df = df.head(args.limit)
    todo = [r for r in df.itertuples() if args.overwrite or not (out_dir / f"{r.participant_id}.npz").exists()]
    print(f"{args.encoder}: {len(df)} participants, {len(df) - len(todo)} already done, {len(todo)} to go -> {out_dir}")
    if not todo:
        return

    device = pick_device()
    encoder = load_encoder(args.encoder, device)
    print(f"device={device}  windows={args.windows}s")

    t_start, failed = time.time(), []
    for i, row in enumerate(todo, 1):
        t0 = time.time()
        try:
            audio, info = load_audio(row, diarise=args.diarise)
            hs = encoder(audio)
            arrays = {"full": hs.float().mean(dim=1).to(torch.float16).cpu().numpy()}
            for w in args.windows:
                bounds = window_bounds(hs.shape[1], w)
                key = f"w{w:g}"
                arrays[key] = pool_windows(hs, bounds)
                arrays[f"{key}_span"] = np.asarray(bounds, dtype=np.float32) / FRAME_RATE
            info["n_frames"] = int(hs.shape[1])
            tmp = out_dir / f"{row.participant_id}.tmp.npz"
            np.savez(tmp, info=json.dumps(info), **arrays)
            tmp.rename(out_dir / f"{row.participant_id}.npz")    # never leave a half-written file behind
        except Exception as exc:                                  # noqa: BLE001 — log and keep going
            failed.append(row.participant_id)
            print(f"  FAILED {row.participant_id}: {exc!r}")
            continue
        finally:
            if device == "mps":
                torch.mps.empty_cache()
        elapsed = time.time() - t_start
        print(f"[{i}/{len(todo)}] {row.participant_id}  {info['kept_sec']:.0f}s audio  "
              f"{time.time() - t0:.1f}s  eta {elapsed / i * (len(todo) - i) / 60:.0f} min", flush=True)

    print(f"done in {(time.time() - t_start) / 60:.1f} min; {len(failed)} failed: {failed}")


if __name__ == "__main__":
    main()
