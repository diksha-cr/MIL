#!/usr/bin/env bash
# Nested CV with Optuna (minns.tune): instance type, audio layer, text layer and the MIL
# network settings chosen inside every outer fold. One array task per outer fold, then merge.
#
#   TUNE_ARGS="--audio whisper-medium --text xlmr-large --windows seg word" \
#     sbatch --array=0-4 --job-name=tune-xlmr scripts/cluster_tune.sh
#   # after all tasks finish (same TUNE_ARGS; CPU is enough):
#   TUNE_ARGS="..." sbatch --gres=none --job-name=tune-xlmr-merge scripts/cluster_tune.sh merge
#
#   Text encoder comparison: the same with --text mbert-base / sbert-mpnet / muril-large.
#   Repeated outer CV: add --repeats R to TUNE_ARGS and use --array=0-$((5*R-1)).
#   Smoke run first: TUNE_ARGS="... --n-trials 3 --inner-seeds 1" sbatch --array=0 ...
#
# Killed or timed-out tasks resume (the Optuna study is in outputs/results/tune/<stem>/fold<k>.db):
# resubmit the same array index. Needs the -seg / -word sets of both encoders (cluster_text.sh).
# Optional env: TUNE_ARGS (flags for minns.tune), DATA_DIR / MH_PIPELINE_ROOT and MINNS_OUTPUTS
#               as in cluster_extract.sh.
#
#SBATCH --job-name=minns-tune
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs
: "${TUNE_ARGS:?set TUNE_ARGS, e.g. \"--audio whisper-medium --text xlmr-large --windows seg word\"}"

DATA_DIR="${DATA_DIR:-/data/mh/ML_data}"
export MH_PIPELINE_ROOT="${MH_PIPELINE_ROOT:-$DATA_DIR}"
[ -f "$MH_PIPELINE_ROOT/outputs/features/finetune_bucket.csv" ] \
  || { echo "missing $MH_PIPELINE_ROOT/outputs/features/finetune_bucket.csv; set DATA_DIR or MH_PIPELINE_ROOT"; exit 1; }
export PATH="$HOME/.local/bin:$PATH"

if [ "${1:-}" = merge ]; then
  # shellcheck disable=SC2086
  uv run --frozen python -m minns.tune $TUNE_ARGS --merge
  exit 0
fi

FOLD="${SLURM_ARRAY_TASK_ID:?submit with --array=0-4 (one task per outer fold)}"
nvidia-smi -L || true
echo "== outer fold $FOLD  $(date)"
# shellcheck disable=SC2086
uv run --frozen python -m minns.tune $TUNE_ARGS --fold "$FOLD" --jobs "${SLURM_CPUS_PER_TASK:-16}"
echo "== done $(date)"
