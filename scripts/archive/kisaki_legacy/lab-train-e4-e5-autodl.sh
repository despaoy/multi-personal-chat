#!/usr/bin/env bash
# Train R1-E4 (RSLoRA) and R1-E5 (Sequence Packing) sequentially on autodl server
# Usage: bash scripts/lab-train-e4-e5-autodl.sh
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

for EXPERIMENT in e4 e5; do
  echo ""
  echo "=== R1-${EXPERIMENT^^} Training Start ==="
  echo "Time: $(date --iso-8601=seconds)"

  # Check GPU availability
  GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  echo "GPU memory used: ${GPU_MEM}MB"
  if (( GPU_MEM > 2000 )); then
    echo "ERROR: GPU memory in use (${GPU_MEM}MB). Waiting for GPU to be free..." >&2
    # Bound the wait so a stale job cannot hold this runner forever.
    WAITED=0
    while (( GPU_MEM > 2000 && WAITED < 21600 )); do
      sleep 30
      WAITED=$((WAITED + 30))
      GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
      echo "  waiting... GPU mem=${GPU_MEM}MB $(date --iso-8601=seconds)"
    done
    if (( GPU_MEM > 2000 )); then
      echo "ERROR: GPU remained busy for 6 hours" >&2
      exit 2
    fi
    echo "GPU is now free"
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
  echo "=== Starting ${EXPERIMENT^^} Training ==="
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$PROJECT/scripts/run_kisaki_experiment.py" \
    --experiment "$EXPERIMENT" --seed "$SEED"

  echo "=== R1-${EXPERIMENT^^} Training Complete ==="
  echo "Time: $(date --iso-8601=seconds)"
done

echo ""
echo "=== All E4/E5 Training Complete ==="
