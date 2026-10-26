#!/usr/bin/env bash
# Stage 0 baselines and stage C multiple-instance networks on the cluster.
#
#   sbatch --job-name=stage0 scripts/cluster_mil.sh stage0
#   sbatch --job-name=mil-whisper scripts/cluster_mil.sh whisper-medium "2 5" "23 24"
#   sbatch --job-name=mil-hubert  scripts/cluster_mil.sh hubert-large   "20"  "23"
#
# MIL arguments: <encoder> "<window seconds>" "<layers>" [random|site|both, default both].
# Needs the packed features the probe jobs wrote (outputs/features/packed/<encoder>/).
# Optional env: MIL_ARGS (extra minns.mil flags, e.g. "--models mi-max attention --seeds 1"),
#               DATA_DIR / MH_PIPELINE_ROOT and MINNS_OUTPUTS as in cluster_extract.sh.
# Results: outputs/results/stage0/*.csv, outputs/results/stage_c/mil_<encoder>_<cv>.csv (+ _oof.npz).
#
#SBATCH --job-name=minns-mil
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs
MODE="${1:?usage: cluster_mil.sh stage0 | <encoder> \"<windows>\" \"<layers>\" [random|site|both]}"

DATA_DIR="${DATA_DIR:-/data/mh/ML_data}"
export MH_PIPELINE_ROOT="${MH_PIPELINE_ROOT:-$DATA_DIR}"
[ -f "$MH_PIPELINE_ROOT/outputs/features/finetune_bucket.csv" ] \
  || { echo "missing $MH_PIPELINE_ROOT/outputs/features/finetune_bucket.csv; set DATA_DIR or MH_PIPELINE_ROOT"; exit 1; }
export PATH="$HOME/.local/bin:$PATH"

if [ "$MODE" = stage0 ]; then
  uv run --frozen python -m minns.baseline
  exit 0
fi

WINDOWS="${2:?give the window sizes, e.g. \"2 5\"}"
LAYERS="${3:?give the layers, e.g. \"23 24\"}"
CV="${4:-both}"
case "$CV" in both) CVS="random site" ;; random|site) CVS="$CV" ;; *) echo "cv must be random, site or both"; exit 1 ;; esac

nvidia-smi -L || true
for cv in $CVS; do
  echo "== $MODE $cv CV  $(date)"
  # shellcheck disable=SC2086
  uv run --frozen python -m minns.mil --encoder "$MODE" --windows $WINDOWS --layers $LAYERS --cv "$cv" ${MIL_ARGS:-}
done
echo "== done $(date)"
