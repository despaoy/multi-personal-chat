#!/usr/bin/env bash
set -euo pipefail

ROOT=${QQCHAT_LAB_ROOT:-/home/szw/lhm2}
PROJECT=$ROOT/qqchat-enhanced
PYTHON=${QQCHAT_PYTHON:-$ROOT/envs/qqchat-gpu-qwen3/bin/python}
RUNTIME=$ROOT/runtime
LOCK_DIR=$RUNTIME/locks/kisaki-r1-extension.lock
LOG=$RUNTIME/logs/kisaki-r1-extension.log
STATE=$RUNTIME/experiments/kisaki/r1-queue-state.json
SEED=${1:-42}
POLL_SECONDS=${KISAKI_GPU_POLL_SECONDS:-30}
REQUIRED_IDLE_CHECKS=${KISAKI_GPU_IDLE_CHECKS:-20}
MAX_MEMORY_MB=${KISAKI_GPU_MAX_MEMORY_MB:-2048}
MAX_UTIL_PERCENT=${KISAKI_GPU_MAX_UTIL_PERCENT:-10}

[[ "$SEED" =~ ^(42|43|44)$ ]] || { echo "seed must be 42, 43, or 44" >&2; exit 2; }
[[ "$PROJECT" == "$ROOT/"* ]] || { echo "project_path_outside_allowed_root=$PROJECT" >&2; exit 2; }
mkdir -p "$RUNTIME/locks" "$RUNTIME/logs" "$(dirname "$STATE")"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  if [[ -f "$LOCK_DIR/pid" ]] && kill -0 "$(cat "$LOCK_DIR/pid")" 2>/dev/null; then
    echo "queue_already_running pid=$(cat "$LOCK_DIR/pid")"
    exit 0
  fi
  rm -rf "$LOCK_DIR"
  mkdir "$LOCK_DIR"
fi
echo $$ > "$LOCK_DIR/pid"
CURRENT_STAGE=initialization
write_state(){ printf '{"schema_version":1,"program":"KISAKI-R1","seed":%s,"status":"%s","detail":"%s","updated_at":"%s"}\n' "$SEED" "$1" "$2" "$(date --iso-8601=seconds)" > "$STATE"; }
log(){ printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$LOG"; }
cleanup(){ code=$?; if (( code != 0 )); then write_state failed "stage=$CURRENT_STAGE exit_code=$code"; fi; rm -rf "$LOCK_DIR"; }
trap cleanup EXIT
trap 'exit 130' INT TERM

wait_for_gpu(){
  local candidate='' checks=0
  while true; do
    local selected=''
    while IFS=',' read -r index memory utilization; do
      index=$(echo "$index"|xargs); memory=$(echo "$memory"|xargs); utilization=$(echo "$utilization"|xargs)
      if (( memory < MAX_MEMORY_MB && utilization < MAX_UTIL_PERCENT )); then selected=$index; break; fi
    done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
    if [[ -n "$selected" && "$selected" == "$candidate" ]]; then checks=$((checks+1)); elif [[ -n "$selected" ]]; then candidate=$selected; checks=1; else candidate=''; checks=0; fi
    write_state waiting_for_gpu "candidate=$candidate checks=$checks/$REQUIRED_IDLE_CHECKS"
    if (( checks >= REQUIRED_IDLE_CHECKS )); then echo "$candidate"; return 0; fi
    sleep "$POLL_SECONDS"
  done
}

is_complete(){
  local manifest=$RUNTIME/experiments/kisaki/$1/seed$SEED/run_manifest.json
  [[ -f "$manifest" ]] && "$PYTHON" -c 'import json,sys;sys.exit(0 if json.load(open(sys.argv[1],encoding="utf-8")).get("status")=="training_complete" else 1)' "$manifest"
}
run_variant(){
  local variant=$1 gpu
  if is_complete "$variant"; then log "skip_completed variant=$variant seed=$SEED"; return 0; fi
  gpu=$(wait_for_gpu)
  CURRENT_STAGE=${variant}_seed_${SEED}
  write_state training "variant=$variant gpu=$gpu"
  log "training_start variant=$variant seed=$SEED gpu=$gpu"
  local output=$RUNTIME/loras/kisaki/canonical/$variant/seed$SEED
  if [[ -d "$output" ]] && find "$output" -maxdepth 1 -type d -name 'checkpoint-*' -print -quit | grep -q .; then
    CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" "$PROJECT/scripts/run_kisaki_experiment.py" --experiment "$variant" --seed "$SEED" --resume
  else
    CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" "$PROJECT/scripts/run_kisaki_experiment.py" --experiment "$variant" --seed "$SEED"
  fi
}

cd "$PROJECT"
write_state preflight 'validating source, dependencies, data, model, and Gold v2'
"$PYTHON" -c 'import tensorboard,torch,transformers,peft'
"$PYTHON" "$PROJECT/scripts/validate_kisaki_experiments.py" --require-model --formal-eval --write-registry --registry-output "$RUNTIME/experiments/kisaki/r1-preflight.json"
if ! git diff --quiet || ! git diff --cached --quiet || [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
  echo 'refusing_dirty_source=true' >&2
  exit 2
fi
# 训练 E1/E2（baseline + NEFTune），供 prompt-v2 baseline 评估使用
# run_variant 内置 is_complete 幂等检查，已完成的 variant 会自动跳过
for variant in e1 e2; do run_variant "$variant"; done

gpu=$(wait_for_gpu)
CURRENT_STAGE=baseline_prompt_v2_seed_${SEED}
write_state evaluating "variants=e1,e2 gpu=$gpu prompt=v2 scope=baseline"
log "baseline_prompt_v2_start seed=$SEED gpu=$gpu variants=e1,e2"
bash "$PROJECT/scripts/lab-evaluate-kisaki-r1-seed.sh" "$SEED" "$gpu" prompt-v2-baseline e1,e2
for variant in e3 e4 e5; do run_variant "$variant"; done
gpu=$(wait_for_gpu)
CURRENT_STAGE=evaluation_seed_${SEED}
write_state evaluating "variants=e1,e2,e3,e4,e5 gpu=$gpu prompt=v2"
bash "$PROJECT/scripts/lab-evaluate-kisaki-r1-seed.sh" "$SEED" "$gpu"
write_state pilot_complete 'R1 E1-E5 prompt-v2 evaluation complete; independent blind review pending'
log 'r1_extension_complete'
