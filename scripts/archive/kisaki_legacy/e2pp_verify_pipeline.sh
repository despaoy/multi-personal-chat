#!/bin/bash
# Historical E2'' artifact only. It is not part of the active research workflow.
# E2'' System Prompt 验证脚本（在服务器上 nohup 运行）
set -e

cd /home/szw/lhm2/qqchat-enhanced
source /home/szw/lhm2/envs/qqchat-gpu-qwen3/bin/activate
export CUDA_VISIBLE_DEVICES=1
export C_INCLUDE_PATH=/home/szw/lhm2/envs/qqchat-gpu-qwen3/include/python3.11
export CPLUS_INCLUDE_PATH=/home/szw/lhm2/envs/qqchat-gpu-qwen3/include/python3.11
export PYTHONPATH=backend:$PYTHONPATH

LOG=/home/szw/lhm2/runtime/logs/e2pp_verify_pipeline.log
echo "=== $(date) 开始 E2'' 验证流程 ===" | tee $LOG

# 1. 清理旧 vLLM
echo "[$(date)] 1. 清理旧 vLLM..." | tee -a $LOG
pkill -f vllm.entrypoints 2>/dev/null || true
sleep 5

# 2. 启动 vLLM
echo "[$(date)] 2. 启动 vLLM (gpu-mem-util=0.7, max-lora-rank=32)..." | tee -a $LOG
nohup python -m vllm.entrypoints.openai.api_server \
  --model /home/szw/lhm2/runtime/models/Qwen3-8B-Instruct \
  --served-model-name qwen3-8b-instruct \
  --port 8002 --enable-lora --max-lora-rank 32 \
  --lora-modules kisaki=/home/szw/lhm2/runtime/loras/kisaki/e2pp_rag_r32/final/ \
  --max-model-len 4096 --gpu-memory-utilization 0.88 \
  --trust-remote-code --dtype float16 \
  > /home/szw/lhm2/runtime/logs/vllm_verify.log 2>&1 &

VLLM_PID=$!
echo "[$(date)] vLLM PID: $VLLM_PID" | tee -a $LOG

# 3. 等待 vLLM 就绪（最多 4 分钟）
echo "[$(date)] 3. 等待 vLLM 就绪..." | tee -a $LOG
for i in $(seq 1 48); do
    if curl -s -m 5 http://127.0.0.1:8002/health 2>/dev/null; then
        echo "" | tee -a $LOG
        echo "[$(date)] vLLM 就绪! (等待 ${i}x5=${i}0 秒)" | tee -a $LOG
        break
    fi
    sleep 5
    if [ $i -eq 48 ]; then
        echo "[$(date)] vLLM 4分钟内未就绪，退出" | tee -a $LOG
        exit 1
    fi
done

# 4. 运行评估
echo "[$(date)] 4. 运行 E2'' 评估（带 System Prompt）..." | tee -a $LOG
python backend/evaluation/character_benchmark.py \
  --dataset backend/evaluation/kisaki_gold_set_v1.json \
  --output backend/data/character_dialogues/experiments/results/kisaki_e2pp_verify_sysprompt.json \
  --base-url http://127.0.0.1:8002 \
  --model kisaki \
  --repetition-penalty 1.15 \
  --frequency-penalty 0.3 \
  --max-tokens 256 \
  --system-prompt-file backend/data/character_dialogues/experiments/archive/prompts/kisaki_system_prompt_v1.txt \
  2>&1 | tee -a $LOG

echo "[$(date)] === E2'' 验证流程完成 ===" | tee -a $LOG

# 5. 关闭 vLLM
echo "[$(date)] 5. 关闭 vLLM..." | tee -a $LOG
pkill -f vllm.entrypoints 2>/dev/null || true

echo "[$(date)] 完成" | tee -a $LOG
