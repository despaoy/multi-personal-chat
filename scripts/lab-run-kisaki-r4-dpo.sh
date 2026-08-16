#!/usr/bin/env bash
set -euo pipefail

ROOT="${QQCHAT_LAB_ROOT:?set QQCHAT_LAB_ROOT to the lab root}"
PROJECT=$ROOT/qqchat-enhanced
PYTHON="${QQCHAT_LAB_PYTHON:-$ROOT/envs/qqchat-gpu-qwen3/bin/python}"
GPU=${1:?usage: lab-run-kisaki-r4-dpo.sh GPU BEST_R1_ADAPTER PREFERENCE_SOURCE}
BEST_ADAPTER=${2:?usage: lab-run-kisaki-r4-dpo.sh GPU BEST_R1_ADAPTER PREFERENCE_SOURCE}
PREFERENCE_SOURCE=${3:?usage: lab-run-kisaki-r4-dpo.sh GPU BEST_R1_ADAPTER PREFERENCE_SOURCE}
BASE_MODEL=$ROOT/runtime/models/Qwen3-8B-Instruct
OUTPUT=$ROOT/runtime/experiments/kisaki/r4
FROZEN=$OUTPUT/frozen_preference_v1
ADAPTER_OUTPUT=$OUTPUT/kisaki-dpo-pilot
LOCK=$ROOT/runtime/locks/kisaki-r4-dpo.lock
LOG=$ROOT/runtime/logs/kisaki-r4-dpo.log

for path in "$BASE_MODEL/config.json" "$BEST_ADAPTER/adapter_config.json" "$PREFERENCE_SOURCE"; do
  [[ -f "$path" ]] || { echo "required_file_missing=$path" >&2; exit 2; }
done
for path in "$BEST_ADAPTER" "$PREFERENCE_SOURCE"; do
  [[ "$path" == "$ROOT/"* ]] || { echo "path_outside_allowed_root=$path" >&2; exit 2; }
done
mkdir -p "$OUTPUT" "$ROOT/runtime/locks" "$ROOT/runtime/logs"
if ! mkdir "$LOCK" 2>/dev/null; then echo "r4_already_locked=$LOCK" >&2; exit 2; fi
cleanup(){ rmdir "$LOCK" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

if [[ ! -f "$FROZEN/manifest.json" ]]; then
  "$PYTHON" "$PROJECT/scripts/prepare_kisaki_dpo_v3.py" \
    --input "$PREFERENCE_SOURCE" --output-dir "$FROZEN" --minimum 100 --seed 42
fi
"$PYTHON" - "$FROZEN/manifest.json" "$FROZEN/kisaki_dpo_train.jsonl" <<'PY'
import hashlib, json, pathlib, sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
train = pathlib.Path(sys.argv[2])
actual = hashlib.sha256(train.read_bytes()).hexdigest()
assert manifest["status"] == "frozen", "preference manifest is not frozen"
assert manifest["human_approved_count"] >= 100, "fewer than 100 approved pairs"
assert manifest["train"]["sha256"] == actual, "frozen train hash mismatch"
PY

if [[ -e "$ADAPTER_OUTPUT" ]]; then echo "refusing_to_overwrite=$ADAPTER_OUTPUT" >&2; exit 2; fi
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$PROJECT/backend" "$PYTHON" -m training.preference_trainer \
  --data "$FROZEN/kisaki_dpo_train.jsonl" --base-model "$BASE_MODEL" \
  --adapter "$BEST_ADAPTER" --method dpo --epochs 1 --beta 0.1 --learning-rate 5e-7 \
  --output-dir "$ADAPTER_OUTPUT" 2>&1 | tee "$LOG"

echo "r4_dpo_training_complete=$ADAPTER_OUTPUT"
echo "heldout_not_trained=$FROZEN/kisaki_dpo_heldout.jsonl"
