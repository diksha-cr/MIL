#!/usr/bin/env bash
# Stage E smoke test on a small subset: every step of scripts/cluster_text.sh, end to end,
# with short training. Checks that the code runs on this machine's data. The AUCs
# are meaningless at ~3 positives per fold.
#
#   bash scripts/smoke_text.sh                       # 60 train participants (15 positive)
#   N_POS=10 N_NEG=30 bash scripts/smoke_text.sh
#   BUCKET=/path/to/finetune_bucket.csv bash scripts/smoke_text.sh
#
# Writes nothing into the repo's outputs/: the subset bucket and every feature / result
# go under $WORK (default /tmp/minns-smoke-text). Needs the pipeline's audio and
# participant_jsons under $PIPELINE (default ~/mh-ML-pipeline), whisper medium.pt, and the
# hub (or a cache) for FacebookAI/xlm-roberta-large.
# PY selects the interpreter (default: uv run --frozen python; PY=.venv/bin/python works too).
set -euo pipefail

cd "$(dirname "$0")/.."
PIPELINE="${PIPELINE:-$HOME/mh-ML-pipeline}"
BUCKET="${BUCKET:-$PIPELINE/outputs/features/finetune_bucket.csv}"
WORK="${WORK:-/tmp/minns-smoke-text}"
N_POS="${N_POS:-15}" N_NEG="${N_NEG:-45}"
read -r -a PY <<< "${PY:-uv run --frozen python}"
[ -f "$BUCKET" ] || { echo "no bucket at $BUCKET; set BUCKET= (e.g. outputs/data/buckets/<date>/finetune_bucket.csv)"; exit 1; }

# A pipeline root whose bucket is the subset; audio and transcripts are the real ones.
ROOT="$WORK/pipe"
mkdir -p "$ROOT/outputs/features" "$WORK/out"
ln -sfn "$PIPELINE/outputs/data" "$ROOT/outputs/data"
ln -sfn "$PIPELINE/outputs/features/text" "$ROOT/outputs/features/text"
export MH_PIPELINE_ROOT="$ROOT" MINNS_OUTPUTS="$WORK/out"
"${PY[@]}" - "$BUCKET" "$ROOT/outputs/features/finetune_bucket.csv" "$N_POS" "$N_NEG" <<'EOF'
import sys
import pandas as pd
src, dst, n_pos, n_neg = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
tr = pd.read_csv(src).query("split == 'train'")
sub = pd.concat([tr[tr.label == 1].sample(n_pos, random_state=0), tr[tr.label == 0].sample(n_neg, random_state=0)])
sub["cv_stratum"] = sub["label"].astype(str)        # the composite strata are too fine for a subset
sub.to_csv(dst, index=False)
print(f"subset bucket: {len(sub)} participants ({n_pos} positive), languages {sub.meta_language.value_counts().to_dict()}")
EOF

QUICK=(--seeds 1 --epochs 5)
step() { echo; echo "=== $*"; }

step "stage D utterances (for the -utt text sets)"
"${PY[@]}" -m minns.utterances --features prosody --jobs 8 | tail -1
step "segment / word tables"
"${PY[@]}" -m minns.segments --features tables --jobs 8 | tail -1
step "whisper-medium over segments, words, utterances"
"${PY[@]}" -m minns.segments --features whisper-medium | tail -1
"${PY[@]}" -m minns.utterances --features whisper-medium | tail -1
step "xlmr-large text embeddings"
"${PY[@]}" -m minns.text --encoder xlmr-large | tail -1
step "text-layer probe"
"${PY[@]}" -m minns.probe --encoder xlmr-large-seg --windows seg --layers 8 16 24 --jobs 4 | tail -4
step "Design 2 on segments (E2-E5)"
"${PY[@]}" -m minns.fusion --audio whisper-medium-seg --text xlmr-large-seg@16 --extra prosody-seg \
  --windows seg --layers 23 "${QUICK[@]}" | tail -6
step "Design 2a through mil.py (dialogue segments, with roles)"
"${PY[@]}" -m minns.mil --encoder whisper-medium-seg+xlmr-large-seg@16+prosody-seg+role-seg --windows dseg \
  --layers 23 --models attention MI-max "${QUICK[@]}" | tail -3
step "Design 2 on stage-D utterances"
"${PY[@]}" -m minns.fusion --audio whisper-medium-utt --text xlmr-large-utt@16 --extra prosody+role \
  --windows dlg --layers 23 --models audio-attn concat-attn gate-attn "${QUICK[@]}" | tail -4
step "Design 1 and cross-attention on words (E6-E11)"
"${PY[@]}" -m minns.fusion --audio whisper-medium-word --text xlmr-large-word@16 --extra wordpros \
  --windows word --layers 23 --models audio-attn concat-attn gate-attn concat-hier gate-hier xattn-attn xattn-hier "${QUICK[@]}" | tail -11
step "paired delta vs audio-only, confound check"
R="$WORK/out/results/stage_e"
"${PY[@]}" -m minns.compare delta --ref "$R/fusion_whisper-medium-seg_xlmr-large-segL16_seg_random_oof.npz:audio-attn" \
  --against "$R/fusion_whisper-medium-seg_xlmr-large-segL16_seg_random_oof.npz"
"${PY[@]}" -m minns.compare confound --set xlmr-large-seg@16 --windows seg

echo; echo "smoke test passed; outputs under $WORK/out"
