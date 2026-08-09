#!/usr/bin/env bash
# Train R1-E3 (DoRA) on autodl server
# Usage: bash scripts/lab-train-e3-autodl.sh
set -euo pipefail

if [[ "${ALLOW_LEGACY_KISAKI_R1:-false}" != "true" ]]; then
  echo "blocked_legacy_r1=true; complete the KISAKI V4 human-review gate first" >&2
  exit 2
fi

export QQCHAT_LAB_ROOT=/root/autodl-tmp
PROJECT=$QQCHAT_LAB_ROOT/qqchat-enhanced
PYTHON=${QQCHAT_PYTHON:-$(command -v python)}
SEED=${1:-42}

cd "$PROJECT"

echo "=== R1-E3 (DoRA) Training Start ==="
echo "LAB_ROOT=$QQCHAT_LAB_ROOT"
echo "PYTHON=$PYTHON"
echo "SEED=$SEED"
echo "Time: $(date --iso-8601=seconds)"

# Verify prerequisites
for f in \
  "$QQCHAT_LAB_ROOT/runtime/models/Qwen3-8B-Instruct/config.json" \
  "backend/data/character_dialogues/experiments/configs/kisaki_e3_canonical.json" \
  "backend/data/character_dialogues/experiments/canonical_dataset_manifest.json" \
  "backend/data/character_dialogues/experiments/tsukiyashiro_kisaki_train.json" \
  "backend/data/character_dialogues/experiments/tsukiyashiro_kisaki_eval.json"; do
  [[ -f "$f" ]] || { echo "required_file_missing=$f" >&2; exit 2; }
done

# Check GPU availability
GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
echo "GPU memory used: ${GPU_MEM}MB"
if (( GPU_MEM > 2000 )); then
  echo "ERROR: GPU memory in use (${GPU_MEM}MB); refusing to share the GPU" >&2
  exit 2
fi

# Check disk space
DISK_AVAIL=$(df --output=avail -B1 /root/autodl-tmp | tail -1)
DISK_GB=$((DISK_AVAIL / 1024 / 1024 / 1024))
echo "Disk available: ${DISK_GB}GB"
if (( DISK_GB < 15 )); then
  echo "ERROR: insufficient disk space (need >=15GB, got ${DISK_GB}GB)" >&2
  exit 2
fi

# Run training
echo "=== Starting E3 Training ==="
CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$PROJECT/scripts/run_kisaki_experiment.py" \
  --experiment e3 --seed "$SEED"

echo "=== R1-E3 Training Complete ==="
echo "Time: $(date --iso-8601=seconds)"
