#!/usr/bin/env bash
# Stage B probe (layer x window x pooling) on the cluster, on features written by
# cluster_extract.sh. CPU only: one logistic regression per (layer, window, fold),
# run in parallel over the job's cores.
#
#   sbatch --job-name=probe-whisper-medium scripts/cluster_probe.sh whisper-medium
#   sbatch --job-name=probe-xlsr-300m      scripts/cluster_probe.sh xlsr-300m
#   sbatch --job-name=probe-mhubert-147    scripts/cluster_probe.sh mhubert-147
#   sbatch --job-name=probe-hubert-large   scripts/cluster_probe.sh hubert-large
#
# Second argument: random | site | both (default both — random-fold CV, then
# site-grouped CV, in the same job). Run one job per encoder, not two jobs on the
# same encoder at once: the first run of an encoder repacks its features into
# outputs/features/packed/<encoder>/ (~ the size of its windows/ folder) and two jobs
# would write the same files.
#
# Optional env: PROBE_ARGS (extra minns.probe flags, e.g. "--layers 6 12 --windows full 5"; mhubert-147 has layers 0..12, the others 0..24),
#               DATA_DIR / MH_PIPELINE_ROOT and MINNS_OUTPUTS as in cluster_extract.sh.
# Results: outputs/results/stage_b/probe_<encoder>_<cv>_C0.01.csv (+ _oof.npz).
#
#SBATCH --job-name=minns-probe
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs
ENC="${1:?usage: cluster_probe.sh <encoder> [random|site|both]}"
CV="${2:-both}"
case "$CV" in both) CVS="random site" ;; random|site) CVS="$CV" ;; *) echo "cv must be random, site or both"; exit 1 ;; esac

DATA_DIR="${DATA_DIR:-/data/mh/ML_data}"
export MH_PIPELINE_ROOT="${MH_PIPELINE_ROOT:-$DATA_DIR}"
[ -f "$MH_PIPELINE_ROOT/outputs/features/finetune_bucket.csv" ] \
  || { echo "missing $MH_PIPELINE_ROOT/outputs/features/finetune_bucket.csv; set DATA_DIR or MH_PIPELINE_ROOT"; exit 1; }

export PATH="$HOME/.local/bin:$PATH"
JOBS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
echo "probe $ENC: cv=$CVS jobs=$JOBS data=$MH_PIPELINE_ROOT extra=${PROBE_ARGS:-none}"

for cv in $CVS; do
  echo "== $cv CV  $(date)"
  # shellcheck disable=SC2086
  uv run --frozen python -m minns.probe --encoder "$ENC" --cv "$cv" --jobs "$JOBS" ${PROBE_ARGS:-}
done
echo "== done $(date)"
