#!/usr/bin/env bash
# Optuna tuning of the audio + text MIL (minns.tune), one stage per job.
#
#   sbatch --job-name=tune1 scripts/cluster_tune.sh stage1
#   sbatch --job-name=tune2 scripts/cluster_tune.sh stage2       # after stage1 has finished
#   sbatch --job-name=tuneb scripts/cluster_tune.sh baselines    # any time; CPU is enough
#   sbatch --job-name=tunes scripts/cluster_tune.sh select       # after stage2
#   sbatch --job-name=tunet scripts/cluster_tune.sh test         # after select, ONCE
#   bash scripts/cluster_tune.sh plots                           # login node, no GPU needed
#
# Or chain them:  j=$(sbatch --parsable scripts/cluster_tune.sh stage1)
#                 sbatch --dependency=afterok:$j scripts/cluster_tune.sh stage2
#
# A stage stops after TUNE_HOURS (default 23, under the 24 h limit); resubmit the same
# command and it resumes the study until it holds its trial count. Run one job per
# study at a time (one SQLite file). Optional env: TUNE_ARGS (e.g. "--trials 50"),
# DATA_DIR / MH_PIPELINE_ROOT and MINNS_OUTPUTS as in cluster_extract.sh.
# Results: outputs/results/tune/ (optuna.db, best.json, *.html).
#
#SBATCH --job-name=minns-tune
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs
STAGE="${1:?usage: cluster_tune.sh stage1 | stage2 | baselines | select | test | plots}"

DATA_DIR="${DATA_DIR:-/data/mh/ML_data}"
export MH_PIPELINE_ROOT="${MH_PIPELINE_ROOT:-$DATA_DIR}"
[ -f "$MH_PIPELINE_ROOT/outputs/features/finetune_bucket.csv" ] \
  || { echo "missing $MH_PIPELINE_ROOT/outputs/features/finetune_bucket.csv; set DATA_DIR or MH_PIPELINE_ROOT"; exit 1; }
export PATH="$HOME/.local/bin:$PATH"

nvidia-smi -L || true
echo "== tune $STAGE  $(date)"
# shellcheck disable=SC2086
uv run --frozen python -m minns.tune "$STAGE" --hours "${TUNE_HOURS:-23}" ${TUNE_ARGS:-}
echo "== done $(date)"
