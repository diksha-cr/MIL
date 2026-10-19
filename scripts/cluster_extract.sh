#!/usr/bin/env bash
# Window-level embedding extraction on a GPU cluster (full recordings, all layers).
#
# Works under SLURM (the #SBATCH lines) or as plain bash on a GPU node. Resumable:
# participants that already have an .npz are skipped, so just resubmit after a timeout.
#
#   1. On the LOGIN node (needs internet), once:
#        MH_PIPELINE_ROOT=/path/to/mh-data bash scripts/cluster_extract.sh setup
#   2. One job per encoder:
#        sbatch --export=ALL,MH_PIPELINE_ROOT=/path/to/mh-data scripts/cluster_extract.sh whisper-medium
#        sbatch ... scripts/cluster_extract.sh xlsr-300m
#        sbatch ... scripts/cluster_extract.sh mhubert-147
#        sbatch ... scripts/cluster_extract.sh hubert-large
#   3. Verify (any node):
#        MH_PIPELINE_ROOT=/path/to/mh-data bash scripts/cluster_extract.sh check whisper-medium
#
# MH_PIPELINE_ROOT must contain (copy from the Mac, see the rsync in the README/notes):
#   outputs/features/finetune_bucket.csv  (and pretrain_bucket.csv for BUCKET=pretrain)
#   outputs/data/cleaned/<participant_id>/spontaneous.wav
# Optional env: BUCKET=finetune|pretrain (default finetune), MINNS_OUTPUTS (default ./outputs),
#               WINDOWS="2 5 10 20", LIMIT=3 (first N participants only: a quick test job).
#
#SBATCH --job-name=minns-extract
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs
MODE="${1:?usage: cluster_extract.sh setup | check <encoder> | <encoder>}"
BUCKET="${BUCKET:-finetune}"
WINDOWS="${WINDOWS:-2 5 10 20}"
ENCODERS="whisper-medium xlsr-300m mhubert-147 hubert-large"

: "${MH_PIPELINE_ROOT:?set MH_PIPELINE_ROOT to the directory holding outputs/features/*_bucket.csv and the audio}"
export MH_PIPELINE_ROOT
[ -f "$MH_PIPELINE_ROOT/outputs/features/${BUCKET}_bucket.csv" ] \
  || { echo "missing $MH_PIPELINE_ROOT/outputs/features/${BUCKET}_bucket.csv"; exit 1; }

export PATH="$HOME/.local/bin:$PATH"

case "$MODE" in
  setup)
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    uv sync --frozen
    uv run python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
EOF
    # Download every encoder now so compute nodes do not need internet.
    for enc in $ENCODERS; do
      echo "== fetching $enc"
      uv run python -c "from minns.extract import load_encoder; load_encoder('$enc', 'cpu'); print('ok')"
    done
    # Every audio file in the bucket must be readable before submitting jobs.
    uv run python - <<'EOF'
import os, soundfile as sf
from minns.data import load_bucket
df = load_bucket(os.environ.get("BUCKET", "finetune"))
bad = [p for p in df.audio_path if not os.path.exists(p)]
print(f"{len(df)} participants, {len(bad)} audio files missing", bad[:5])
rates = {sf.info(p).samplerate for p in df.audio_path if os.path.exists(p)}
print("sample rates:", rates)
EOF
    ;;

  check)
    ENC="${2:?usage: cluster_extract.sh check <encoder>}"
    ENC="$ENC" uv run python - <<'EOF'
import json, os
import numpy as np
from minns.data import OUTPUTS, load_bucket
enc, bucket = os.environ["ENC"], os.environ.get("BUCKET", "finetune")
d = OUTPUTS / "features/windows" / enc
df = load_bucket(bucket)
if os.environ.get("LIMIT"):
    df = df.head(int(os.environ["LIMIT"]))
missing, bad, shapes = [], [], set()
for pid in df.participant_id:
    f = d / f"{pid}.npz"
    if not f.exists():
        missing.append(pid); continue
    z = np.load(f)
    full = z["full"].astype(np.float32)
    shapes.add(full.shape)
    if not np.isfinite(full).all() or any(not np.isfinite(z[k].astype(np.float32)).all()
                                          for k in z.files if k.startswith("w") and not k.endswith("_span")):
        bad.append(pid)
    if json.loads(str(z["info"]))["diarised"]:
        bad.append(pid + " (diarised?)")
print(f"{enc}/{bucket}: {len(df) - len(missing)}/{len(df)} done, {len(bad)} with NaN/inf, layer x dim {shapes}")
print("missing:", missing[:10], "bad:", bad[:10])
raise SystemExit(1 if missing or bad else 0)
EOF
    ;;

  *)
    [[ " $ENCODERS " == *" $MODE "* ]] || { echo "unknown encoder $MODE; choose from: $ENCODERS"; exit 1; }
    # Compute nodes are often offline: use the caches filled by `setup`.
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    nvidia-smi -L || true
    # shellcheck disable=SC2086
    uv run --frozen python -m minns.extract --encoder "$MODE" --bucket "$BUCKET" --windows $WINDOWS ${LIMIT:+--limit $LIMIT}
    bash "$0" check "$MODE"
    ;;
esac
