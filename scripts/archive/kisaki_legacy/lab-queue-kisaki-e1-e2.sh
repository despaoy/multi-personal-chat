#!/usr/bin/env bash
# Historical artifact only. Do not call from active experiment workflows.
set -euo pipefail

ROOT=${QQCHAT_LAB_ROOT:-/home/szw/lhm2}
PROJECT=$ROOT/qqchat-enhanced
PYTHON=${QQCHAT_PYTHON:-$ROOT/envs/qqchat-gpu-qwen3/bin/python}
RUNTIME=$ROOT/runtime
LOCK_DIR=$RUNTIME/locks/kisaki-e1-e2.lock
LOG=$RUNTIME/logs/kisaki-e1-e2-queue.log
STATE=$RUNTIME/experiments/kisaki/queue-state.json
PHASE=${1:-pilot}
POLL_SECONDS=${KISAKI_GPU_POLL_SECONDS:-30}
REQUIRED_IDLE_CHECKS=${KISAKI_GPU_IDLE_CHECKS:-20}
MAX_MEMORY_MB=${KISAKI_GPU_MAX_MEMORY_MB:-2048}
MAX_UTIL_PERCENT=${KISAKI_GPU_MAX_UTIL_PERCENT:-10}

if [[ "$PHASE" != "pilot" && "$PHASE" != "replicate" ]]; then
  echo "usage: $0 [pilot|replicate]" >&2
  exit 2
fi
if [[ "$PROJECT" != "$ROOT/"* ]]; then
  echo "project_path_outside_allowed_root=$PROJECT" >&2
  exit 2
fi

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
CURRENT_STAGE="initialization"
cleanup() {
  local exit_code=$?
  if (( exit_code != 0 )); then
    write_state "failed" "stage=$CURRENT_STAGE exit_code=$exit_code"
    log "queue_failed stage=$CURRENT_STAGE exit_code=$exit_code"
  fi
  rm -rf "$LOCK_DIR"
}

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$LOG"
}

write_state() {
  local status=$1
  local detail=$2
  printf '{"schema_version":1,"phase":"%s","status":"%s","detail":"%s","updated_at":"%s"}\n' "$PHASE" "$status" "$detail" "$(date --iso-8601=seconds)" > "$STATE"
}

trap cleanup EXIT
trap 'exit 130' INT TERM

gpu_snapshot() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
}

wait_for_gpu() {
  local candidate=""
  local checks=0
  while true; do
    local selected=""
    while IFS=',' read -r index memory utilization; do
      index=$(echo "$index" | xargs)
      memory=$(echo "$memory" | xargs)
      utilization=$(echo "$utilization" | xargs)
      if (( memory < MAX_MEMORY_MB && utilization < MAX_UTIL_PERCENT )); then
        selected=$index
        break
      fi
    done < <(gpu_snapshot)
    if [[ -n "$selected" && "$selected" == "$candidate" ]]; then
      checks=$((checks + 1))
    elif [[ -n "$selected" ]]; then
      candidate=$selected
      checks=1
    else
      candidate=""
      checks=0
    fi
    write_state "waiting_for_gpu" "candidate=$candidate checks=$checks/$REQUIRED_IDLE_CHECKS"
    if (( checks >= REQUIRED_IDLE_CHECKS )); then
      local memory utilization
      IFS=',' read -r memory utilization < <(nvidia-smi --id="$candidate" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
      memory=$(echo "$memory" | xargs)
      utilization=$(echo "$utilization" | xargs)
      if (( memory < MAX_MEMORY_MB && utilization < MAX_UTIL_PERCENT )); then
        echo "$candidate"
        return 0
      fi
      candidate=""
      checks=0
    fi
    sleep "$POLL_SECONDS"
  done
}

run_seed() {
  local seed=$1
  local gpu
  gpu=$(wait_for_gpu)
  log "selected_gpu=$gpu stage=e1 seed=$seed"
  CURRENT_STAGE="e1_seed_$seed"
  write_state "training_e1" "seed=$seed gpu=$gpu"
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" "$PROJECT/scripts/run_kisaki_experiment.py" --experiment e1 --seed "$seed"

  gpu=$(wait_for_gpu)
  log "selected_gpu=$gpu stage=e2 seed=$seed"
  CURRENT_STAGE="e2_seed_$seed"
  write_state "training_e2" "seed=$seed gpu=$gpu"
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" "$PROJECT/scripts/run_kisaki_experiment.py" --experiment e2 --seed "$seed"

  gpu=$(wait_for_gpu)
  log "selected_gpu=$gpu stage=evaluation seed=$seed"
  CURRENT_STAGE="evaluation_seed_$seed"
  write_state "evaluating" "seed=$seed gpu=$gpu"
  bash "$PROJECT/scripts/lab-evaluate-kisaki-seed.sh" "$seed" "$gpu"
}

wait_for_frozen_gold() {
  while ! "$PYTHON" "$PROJECT/scripts/validate_kisaki_experiments.py" --require-model --formal-eval --write-registry --registry-output "$RUNTIME/experiments/kisaki/canonical_experiment_registry.json" >>"$LOG" 2>&1; do
    write_state "waiting_for_gold" "Gold v2 must pass semantic audit and manual review before training"
    sleep "${KISAKI_GOLD_POLL_SECONDS:-300}"
  done
}

wait_for_clean_git() {
  while true; do
    if git diff --quiet && git diff --cached --quiet && [[ -z "$(git ls-files --others --exclude-standard)" ]]; then
      return 0
    fi
    write_state "waiting_for_clean_source" "Commit and synchronize the experiment source before training"
    sleep "${KISAKI_SOURCE_POLL_SECONDS:-300}"
  done
}

cd "$PROJECT"
CURRENT_STAGE="dependency_preflight"
write_state "preflight" "validating canonical training contracts"
"$PYTHON" -c 'import tensorboard' >/dev/null
CURRENT_STAGE="experiment_preflight"
"$PYTHON" "$PROJECT/scripts/validate_kisaki_experiments.py" --require-model --write-registry --registry-output "$RUNTIME/experiments/kisaki/canonical_experiment_registry.json"
wait_for_frozen_gold
wait_for_clean_git

if [[ "$PHASE" == "pilot" ]]; then
  run_seed 42
  write_state "pilot_complete" "seed42 trained and evaluated; review quality gate before replicate phase"
  log "pilot_complete"
  exit 0
fi

PILOT_GATE=$RUNTIME/experiments/kisaki/seed42-quality-gate.json
if [[ ! -f "$PILOT_GATE" ]]; then
  echo "pilot_quality_gate_missing=$PILOT_GATE" >&2
  exit 2
fi
"$PYTHON" -c 'import json,sys;sys.exit(0 if json.load(open(sys.argv[1],encoding="utf-8")).get("passed") else 2)' "$PILOT_GATE"
run_seed 43
run_seed 44
"$PYTHON" "$PROJECT/scripts/aggregate_kisaki_repetitions.py" --results-root "$RUNTIME/experiments/kisaki" --output "$RUNTIME/experiments/kisaki/three-seed-summary.json"
write_state "replications_complete" "seeds42,43,44 aggregated"
log "replications_complete"
