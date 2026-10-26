#!/usr/bin/env bash
# Stage D: utterance instances (participant speech cut at pauses) and their features.
#
#   1. Segment + prosody + eGeMAPS, CPU (writes the utterance spans everything else reuses):
#        sbatch scripts/cluster_utterances.sh segment
#   2. Encoder states pooled over the same utterances, GPU (after 1 has finished):
#        sbatch --gres=gpu:1 --mem=32G --job-name=utt-whisper scripts/cluster_utterances.sh whisper-medium
#   3. Verify (any node):
#        bash scripts/cluster_utterances.sh check prosody      # or egemaps, whisper-medium-utt
#
# Resumable: participants that already have an .npz are skipped. Re-segmenting
# (e.g. after changing a constant in minns/utterances.py) needs UTT_ARGS=--overwrite
# on step 1 AND step 2, so every feature set keeps the same utterances.
#
# Two instance sets: utt (participant only) and dlg (whole dialogue, every utterance
# tagged participant / interviewer / other in the `role` set). Probe / MIL on them like
# any other encoder; with dlg add +role so the model knows who is speaking:
#   PROBE_ARGS="--windows dlg utt" sbatch scripts/cluster_probe.sh egemaps+prosody+role random
#   uv run python -m minns.mil --encoder whisper-medium-utt+prosody+role --windows dlg --layers 23 24
#
# Optional env: UTT_ARGS (extra minns.utterances flags, e.g. "--limit 3" or "--rule dominant"),
#               DATA_DIR / MH_PIPELINE_ROOT and MINNS_OUTPUTS as in cluster_extract.sh.
#
#SBATCH --job-name=minns-utterances
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs
MODE="${1:?usage: cluster_utterances.sh segment | <encoder> | check <feature set>}"

DATA_DIR="${DATA_DIR:-/data/mh/ML_data}"
export MH_PIPELINE_ROOT="${MH_PIPELINE_ROOT:-$DATA_DIR}"
[ -f "$MH_PIPELINE_ROOT/outputs/features/finetune_bucket.csv" ] \
  || { echo "missing $MH_PIPELINE_ROOT/outputs/features/finetune_bucket.csv; set DATA_DIR or MH_PIPELINE_ROOT"; exit 1; }
[ -d "$MH_PIPELINE_ROOT/outputs/features/text/participant_jsons" ] \
  || { echo "missing $MH_PIPELINE_ROOT/outputs/features/text/participant_jsons (the TextGrid transcripts)"; exit 1; }
export PATH="$HOME/.local/bin:$PATH"

case "$MODE" in
  segment)
    JOBS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
    # shellcheck disable=SC2086
    uv run --frozen python -m minns.utterances --features prosody egemaps --jobs "$JOBS" ${UTT_ARGS:-}
    bash "$0" check prosody && bash "$0" check egemaps && bash "$0" check role
    ;;

  check)
    SET="${2:?usage: cluster_utterances.sh check <prosody | egemaps | <encoder>-utt>}"
    SET="$SET" uv run --frozen python - <<'EOF'
import json, os
import numpy as np
from minns.data import OUTPUTS, load_bucket
d = OUTPUTS / "features/windows" / os.environ["SET"]
df = load_bucket("finetune")
missing, bad, n_utts, n_dlg, fallback, shapes = [], [], [], [], 0, set()
for pid in df.participant_id:
    f = d / f"{pid}.npz"
    if not f.exists():
        missing.append(pid); continue
    z = np.load(f)
    u = z["utt"].astype(np.float32)
    shapes.add(u.shape[1:])
    n_utts.append(len(u))
    if "dlg" not in z.files:
        bad.append(pid + " (no dlg: segmented before role tagging)"); continue
    n_dlg.append(len(z["dlg"]))
    if not np.isfinite(u).all() or not np.isfinite(z["dlg"].astype(np.float32)).all():
        bad.append(pid)
    fallback += bool(json.loads(str(z["info"])).get("fallback"))
print(f"{d.name}: {len(df) - len(missing)}/{len(df)} done, {len(bad)} with NaN/inf, "
      f"{fallback} fallback bags, (layers, dim) {shapes}")
if n_utts:
    print("participant utterances p10/50/90:", np.percentile(n_utts, [10, 50, 90]).tolist(),
          "  dialogue utterances p10/50/90:", np.percentile(n_dlg, [10, 50, 90]).tolist() if n_dlg else "-")
print("missing:", missing[:10], "bad:", bad[:10])
raise SystemExit(1 if missing or bad else 0)
EOF
    ;;

  *)
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    nvidia-smi -L || true
    # shellcheck disable=SC2086
    uv run --frozen python -m minns.utterances --features "$MODE" ${UTT_ARGS:-}
    bash "$0" check "$MODE-utt"
    ;;
esac
