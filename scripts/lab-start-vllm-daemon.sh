#!/usr/bin/env bash
set -euo pipefail

ROOT="${MULTIPERSONAL_LAB_ROOT:-${QQCHAT_LAB_ROOT:?set MULTIPERSONAL_LAB_ROOT or QQCHAT_LAB_ROOT to the lab root}}"
PROJECT=$ROOT/multi-personal-chat
LAUNCHER=$PROJECT/scripts/lab-start-vllm.sh
PIDFILE=$ROOT/runtime/vllm.pid
LOGFILE=$ROOT/runtime/logs/vllm.log

mkdir -p "$ROOT/runtime/logs"

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "vllm_already_running pid=$(cat "$PIDFILE")"
  exit 0
fi

if pgrep -u "$(id -u)" -f 'training\.trainer' >/dev/null; then
  echo "refusing_to_start: a training process is still using this user's GPU" >&2
  exit 2
fi

if pgrep -u "$(id -u)" -f 'vllm serve.*Qwen3-8B-Instruct-AWQ' >/dev/null; then
  echo "refusing_to_start: an untracked vLLM process is already running" >&2
  exit 3
fi

cd "$PROJECT"
nohup "$LAUNCHER" >"$LOGFILE" 2>&1 </dev/null &
VLLM_PID=$!
echo "$VLLM_PID" >"$PIDFILE"

sleep 3
if ! kill -0 "$VLLM_PID" 2>/dev/null; then
  echo "vllm_failed_to_start; inspect $LOGFILE" >&2
  tail -n 40 "$LOGFILE" >&2 || true
  exit 4
fi

echo "vllm_started pid=$VLLM_PID log=$LOGFILE"
