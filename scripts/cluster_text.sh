#!/usr/bin/env bash
# Stage E: audio + text instances (TextGrid segments and words) and the fusion MIL models.
# Plan and experiment table: docs/text_audio_mil_plan.md.
#
#   0. (stage D, if not done yet; needed only for the utterance-level text sets)
#        sbatch scripts/cluster_utterances.sh segment
#   1. Segment / word tables + prosody-seg, role-seg, wordpros, CPU:
#        sbatch scripts/cluster_text.sh tables
#   2. Audio states pooled over segments and words, GPU (after 1):
#        sbatch --gres=gpu:1 --mem=32G --job-name=seg-whisper scripts/cluster_text.sh audio whisper-medium
#   3. Text embeddings of the same instances, GPU (after 1; also writes <enc>-utt if stage D ran first):
#        sbatch --gres=gpu:1 --mem=32G --job-name=text-xlmr scripts/cluster_text.sh text xlmr-large
#   4. Verify:   bash scripts/cluster_text.sh check xlmr-large-seg seg
#   5. Models (GPU), FUSION_ARGS passed to minns.fusion:
#        FUSION_ARGS="--audio whisper-medium-seg --text xlmr-large-seg@16 --extra prosody-seg --windows seg --layers 23 24" \
#          sbatch --gres=gpu:1 --job-name=fusion-seg scripts/cluster_text.sh fusion
#      Text-layer probe first (picks the @ layer):
#        PROBE_ARGS="--windows seg" sbatch scripts/cluster_probe.sh xlmr-large-seg random
#
# Resumable: participants that already have an .npz are skipped. Rebuilding the tables
# (TEXT_ARGS=--overwrite on step 1) needs --overwrite on steps 2 and 3 too, so every
# set keeps the same instances.
#
# Optional env: TEXT_ARGS (extra flags for minns.segments / minns.text, e.g. "--limit 3"),
#               DATA_DIR / MH_PIPELINE_ROOT and MINNS_OUTPUTS as in cluster_extract.sh.
#
#SBATCH --job-name=minns-text
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs
MODE="${1:?usage: cluster_text.sh tables | audio <encoder> | text <encoder> | check <set> <key> | fusion}"

DATA_DIR="${DATA_DIR:-/data/mh/ML_data}"
export MH_PIPELINE_ROOT="${MH_PIPELINE_ROOT:-$DATA_DIR}"
[ -f "$MH_PIPELINE_ROOT/outputs/features/finetune_bucket.csv" ] \
  || { echo "missing $MH_PIPELINE_ROOT/outputs/features/finetune_bucket.csv; set DATA_DIR or MH_PIPELINE_ROOT"; exit 1; }
[ -d "$MH_PIPELINE_ROOT/outputs/features/text/participant_jsons" ] \
  || { echo "missing $MH_PIPELINE_ROOT/outputs/features/text/participant_jsons (the TextGrid transcripts)"; exit 1; }
export PATH="$HOME/.local/bin:$PATH"

case "$MODE" in
  tables)
    JOBS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
    # shellcheck disable=SC2086
    uv run --frozen python -m minns.segments --features tables --jobs "$JOBS" ${TEXT_ARGS:-}
    bash "$0" check prosody-seg seg && bash "$0" check wordpros word
    ;;

  audio)
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    nvidia-smi -L || true
    # shellcheck disable=SC2086
    uv run --frozen python -m minns.segments --features "${2:?encoder}" ${TEXT_ARGS:-}
    bash "$0" check "$2-seg" seg && bash "$0" check "$2-word" word
    ;;

  text)
    # Text models are fetched from the hub on the first run; on an offline node, copy the
    # repo into outputs/models/<repo name> (as for the speech encoders) and set HF_HUB_OFFLINE=1.
    nvidia-smi -L || true
    # shellcheck disable=SC2086
    uv run --frozen python -m minns.text --encoder "${2:?encoder}" ${TEXT_ARGS:-}
    bash "$0" check "$2-seg" seg && bash "$0" check "$2-word" word
    ;;

  check)
    SET="${2:?usage: cluster_text.sh check <set> <key>}" KEY="${3:?key: seg dseg word utt dlg}" \
      uv run --frozen python - <<'EOF'
import json, os
import numpy as np
from minns.data import OUTPUTS, load_bucket
d, key = OUTPUTS / "features/windows" / os.environ["SET"], os.environ["KEY"]
df = load_bucket("finetune")
missing, bad, n, zero, fallback, shapes = [], [], [], 0, 0, set()
for pid in df.participant_id:
    f = d / f"{pid}.npz"
    if not f.exists():
        missing.append(pid); continue
    z = np.load(f)
    x = z[key].astype(np.float32)
    shapes.add(x.shape[1:])
    n.append(len(x))
    if len(x) == 0 or not np.isfinite(x).all():
        bad.append(pid)
    zero += int((np.abs(x).sum((1, 2)) == 0).sum())
    fallback += bool(json.loads(str(z["info"])).get("fallback"))
print(f"{d.name}/{key}: {len(df) - len(missing)}/{len(df)} done, {len(bad)} empty or NaN/inf, "
      f"{fallback} bags with a fallback, {zero} all-zero instances, (layers, dim) {shapes}")
if n:
    print("instances per bag p10/50/90:", np.percentile(n, [10, 50, 90]).tolist(), " max", max(n))
print("missing:", missing[:10], "bad:", bad[:10])
raise SystemExit(1 if missing or bad else 0)
EOF
    ;;

  fusion)
    nvidia-smi -L || true
    # shellcheck disable=SC2086
    uv run --frozen python -m minns.fusion ${FUSION_ARGS:?set FUSION_ARGS}
    ;;

  *)
    echo "unknown mode $MODE"; exit 1
    ;;
esac
