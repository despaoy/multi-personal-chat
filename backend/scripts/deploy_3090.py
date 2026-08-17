# -*- coding: utf-8 -*-
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MultiPersonal Chat System - RTX 3090 双卡部署脚本
  目标: 2×NVIDIA GeForce RTX 3090 24GB + E5-2680
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

部署架构:
  ┌──────────────────────────────────────────────┐
  │  GPU 0 (RTX 3090 24GB)                       │
  │  ├── vLLM Worker (TP rank 0)                 │
  │  │   ├── Qwen3-8B 模型权重 (分片1)          │
  │  │   └── KV-Cache 分页 (~12GB)               │
  │  └── 剩余显存: LoRA 适配器池                  │
  ├──────────────────────────────────────────────┤
  │  GPU 1 (RTX 3090 24GB)                       │
  │  ├── vLLM Worker (TP rank 1)                 │
  │  │   ├── Qwen3-8B 模型权重 (分片2)          │
  │  │   └── KV-Cache 分页 (~12GB)               │
  │  └── 剩余显存: LoRA 适配器池                  │
  └──────────────────────────────────────────────┘

预期性能（vs RTX 4060 8GB 单卡）:
  - 吞吐量: 0.2 req/s → 10-20 req/s (50-100x)
  - P95 延迟: 73s → 2-5s
  - 并发: 5群 → 50+群
  - GPU 利用率: 40% → 90%+
"""

import os
import subprocess
import sys
from pathlib import Path

# ══════════════════════════════════════════════════════════════════
# 环境适配
# ══════════════════════════════════════════════════════════════════

# CUDA 12.4 环境变量 (RTX 3090 Ada Lovelace 架构, SM 8.6)
CUDA_ENV = {
    "CUDA_VISIBLE_DEVICES": "0,1",
    "CUDA_LAUNCH_BLOCKING": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
    # FlashInfer backend (更好的 attention kernel)
    "VLLM_ATTENTION_BACKEND": "FLASHINFER",
}

# vLLM 配置
VLLM_CONFIG = {
    # 模型
    "model": "Qwen/Qwen3-8B",
    # 显存优化
    "gpu_memory_utilization": 0.92,  # 3090 可设更高
    "max_model_len": 4096,           # QQ 聊天不需要过长 context
    # Continuous Batching
    "max_num_seqs": 64,              # 最多同时处理 64 个序列
    "max_num_batched_tokens": 8192,  # 每批次最大 token 数
    # Prefix Caching (system prompt / RAG context 复用)
    "enable_prefix_caching": True,
    # Tensor Parallelism (双卡)
    "tensor_parallel_size": 2,
    # LoRA 支持
    "enable_lora": True,
    "max_lora_rank": 64,
    "max_loras": 10,                 # 最多 10 个 LoRA 适配器
    "max_cpu_loras": 20,
    "fully_sharded_loras": True,     # LoRA 分片到双卡
    # API 服务
    "port": 8001,
    "host": "0.0.0.0",
    # 调度
    "scheduler_delay_factor": 0.5,   # 调度延迟因子
    "max_waiting_sequences": 100,    # 等待队列
}

# 启动命令模板
VLLM_CMD = """
python -m vllm.entrypoints.openai.api_server \\
  --model {model} \\
  --tensor-parallel-size {tensor_parallel_size} \\
  --gpu-memory-utilization {gpu_memory_utilization} \\
  --max-model-len {max_model_len} \\
  --max-num-seqs {max_num_seqs} \\
  --max-num-batched-tokens {max_num_batched_tokens} \\
  --enable-lora \\
  --max-lora-rank {max_lora_rank} \\
  --max-loras {max_loras} \\
  --fully-sharded-loras \\
  --port {port} \\
  --host {host}
"""


def print_config():
    """打印部署配置"""
    print("=" * 60)
    print("  MultiPersonal Chat System - RTX 3090 双卡部署方案")
    print("=" * 60)

    print(f"\n[硬件]")
    print(f"  GPU: 2× NVIDIA RTX 3090 24GB (SM 8.6)")
    print(f"  CPU: Intel Xeon E5-2680")
    print(f"  CUDA: 12.4+")

    print(f"\n[vLLM 推理引擎]")
    for k, v in VLLM_CONFIG.items():
        if k == "model":
            print(f"  模型: {v}")
        elif k == "port":
            print(f"  端口: {v}")
        else:
            print(f"  {k}: {v}")

    print(f"\n[显存分配 (每卡)]")
    print(f"  总显存:      24.0 GB")
    print(f"  模型权重:     ~3.5 GB (7B × FP16 ÷ 2卡)")
    print(f"  KV-Cache:     ~12 GB (PagedAttention 分页)")
    print(f"  LoRA:         ~2 GB (10个适配器)")
    print(f"  系统预留:     ~6.5 GB")
    print(f"  利用率:       92%")

    print(f"\n[预期性能]")
    print(f"  Throughput:   10-20 req/s (真实推理)")
    print(f"  Cache 命中:   30-50 req/s (缓存 + 合并)")
    print(f"  P95 Latency:  < 5s")
    print(f"  GPU 利用率:   90%+ (Continuous Batching)")
    print(f"  最大并发群:   50+")

    print(f"\n[启动命令]")
    cmd = VLLM_CMD.format(**VLLM_CONFIG)
    print(cmd)


def deploy():
    """部署 vLLM 服务器"""
    import os
    for k, v in CUDA_ENV.items():
        os.environ[k] = v
    
    print("[部署] 启动 vLLM 服务器...")
    cmd = VLLM_CMD.format(**VLLM_CONFIG)
    subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    print_config()


# ══════════════════════════════════════════════════════════════════
# FastAPI 后端配置（服务器端）
# ══════════════════════════════════════════════════════════════════

# .env 配置示例
DOTENV_EXAMPLE = """
# ── 模型提供商配置 ──
# 本地推理使用 vLLM（推荐）
MODEL_PROVIDER=vllm
VLLM_BASE_URL=http://localhost:8001/v1
VLLM_MODEL=Qwen3-8B-Instruct
VLLM_TIMEOUT=120

# ── 并发配置 ──
# vLLM 模式下不需要本地 semaphore，由 vLLM 内部调度
# 但保留 pipeline 层面的保护
PIPELINE_MAX_CONCURRENT=50
PIPELINE_MAX_QUEUE=500
PIPELINE_GROUP_RATE=5

# ── 安全配置 ──
JWT_SECRET=PLEASE_SET_A_SECURE_SECRET_HERE

# ── 数据库 ──
DB_PATH=./qq_assistant.db
"""

# uvicorn 启动配置（服务器端，多 worker）
UVICORN_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 1,  # 进程内幂等/会话状态要求单 worker
    "limit_concurrency": 500,
    "timeout_keep_alive": 30,
    "timeout_graceful_shutdown": 30,
    "log_level": "info",
}


def print_uvicorn_config():
    print("\n" + "=" * 60)
    print("  FastAPI 服务器配置")
    print("=" * 60)
    print(f"  启动命令:")
    args = " ".join(f"--{k} {v}" for k, v in UVICORN_CONFIG.items())
    print(f"  uvicorn app.main:app {args}")
    print(f"\n  环境变量 (.env):")
    print(DOTENV_EXAMPLE)
