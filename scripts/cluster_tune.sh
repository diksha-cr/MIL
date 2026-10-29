#!/usr/bin/env bash
# Nested CV with Optuna (minns.tune): instance type, audio layer, text layer and the MIL
# network settings chosen inside every outer fold. Array tasks = outer folds x workers.
#
#   One worker per outer fold (5 GPUs):
#     TUNE_ARGS="--audio whisper-medium --text xlmr-large --windows seg word" \
#       sbatch --array=0-4 --job-name=tune-xlmr scripts/cluster_tune.sh
#   W workers per outer fold, all searching the same studies (5 x W GPUs, ~W x faster):
#     WORKERS=4 TUNE_ARGS="..." sbatch --array=0-19 --job-name=tune-xlmr scripts/cluster_tune.sh
#     (task t runs fold t / W as worker t % W; "%8" after the range caps GPUs in use: --array=0-19%8)
#   After every fold x model result exists (same TUNE_ARGS; CPU is enough):
#     TUNE_ARGS="..." sbatch --gres=none --job-name=tune-xlmr-merge scripts/cluster_tune.sh merge
#
#   Text encoder comparison: the same with --text mbert-base / sbert-mpnet / muril-large.
#   Repeated outer CV: add --repeats R to TUNE_ARGS; tasks become 0..5*R*W-1.
#   Smoke run first: TUNE_ARGS="... --n-trials 3 --inner-seeds 1" sbatch --array=0 ...
#
# 24 h limit: every task stops starting trials after HOURS (default 23) and exits; a study
# that is not finished says "resubmit to continue" in the log. Resubmit the same array
# (finished studies are skipped, unfinished ones continue; stage 1 is cached). To chain
# that automatically:  sbatch --dependency=afterany:<jobid> --array=... (same command).
# Needs the -seg / -word sets of both encoders (cluster_text.sh).
# Optional env: TUNE_ARGS (flags for minns.tune), WORKERS (default 1), HOURS (default 23),
#               DATA_DIR / MH_PIPELINE_ROOT and MINNS_OUTPUTS as in cluster_extract.sh.
#
#SBATCH --job-name=minns-tune
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
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

TASK="${SLURM_ARRAY_TASK_ID:?submit with --array (one task per outer fold x worker)}"
WORKERS="${WORKERS:-1}"
FOLD=$((TASK / WORKERS)) WORKER=$((TASK % WORKERS))
nvidia-smi -L || true
echo "== outer fold $FOLD, worker $WORKER of $WORKERS  $(date)"
# shellcheck disable=SC2086
uv run --frozen python -m minns.tune $TUNE_ARGS --fold "$FOLD" --worker "$WORKER" --hours "${HOURS:-23}" \
  --jobs "${SLURM_CPUS_PER_TASK:-16}"
echo "== done $(date)"
