#!/usr/bin/env python3
"""
MultiPersonal Chat System - vLLM 推理服务启动脚本

在 RTX 3090 (24GB) 上使用 vLLM 启动 Qwen3-8B 推理服务。

硬件适配:
  - 单卡 24GB: Qwen3-8B-Instruct FP16 可直接运行
  - 双卡 24GB×2: 可使用张量并行 (tensor-parallel-size=2) 提升吞吐
  - AWQ 4bit: 显存降至 ~6GB，适合多 LoRA 并存

启动:
  python scripts/launch_vllm.py                              # FP16 默认
  python scripts/launch_vllm.py --quant awq                 # AWQ 4bit 量化
  python scripts/launch_vllm.py --model Qwen/Qwen3-8B
  python scripts/launch_vllm.py --lora-path ./loras/hutao_lora_7b/final
  python scripts/launch_vllm.py --tensor-parallel 1         # 单卡模式

要求:
  pip install vllm==0.7.2
"""
import argparse
import subprocess
import sys
from pathlib import Path


def build_args(args) -> list:
    """构建 vLLM 启动参数"""
    _BACKEND = Path(__file__).parent.parent
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--host", args.host,
        "--port", str(args.port),
        "--model", args.model,
        "--served-model-name", args.served_name or args.model.split("/")[-1],
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_mem_util),
        "--max-num-seqs", str(args.max_num_seqs),
        "--max-num-batched-tokens", str(args.max_batched_tokens),
    ]

    # 张量并行
    if args.tensor_parallel > 1:
        cmd += ["--tensor-parallel-size", str(args.tensor_parallel)]

    # 量化（仅指定量化方法时添加）
    if args.quant:
        cmd += ["--quantization", args.quant]

    # LoRA 适配器
    if args.lora_path:
        lora_path = str(Path(args.lora_path).resolve())
        cmd += [
            "--enable-lora",
            "--max-lora-rank", str(args.max_lora_rank),
            "--lora-modules", f"hutao={lora_path}",
        ]

    # 数据类型
    if args.dtype:
        cmd += ["--dtype", args.dtype]

    # 禁用前缀缓存（vLLM 0.7.x 默认已启用，此选项用于关闭）
    if args.disable_prefix_caching:
        cmd += ["--no-prefix-caching"]

    return cmd


def main():
    parser = argparse.ArgumentParser(description="vLLM 推理服务启动器 (MultiPersonal Chat System)")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8001, help="监听端口 (不要让 FastAPI 占用)")
    parser.add_argument("--model", default="Qwen/Qwen3-8B",
                        help="模型名称或本地路径")
    parser.add_argument("--served-name", default="qwen3-8b",
                        help="对外暴露的模型名")
    parser.add_argument("--max-model-len", type=int, default=24576,
                        help="最大上下文长度（Qwen3-8B-AWQ / RTX 3090 24GB 默认 24576）")
    parser.add_argument("--gpu-mem-util", type=float, default=0.90,
                        help="GPU 显存利用率 (0-1)")
    parser.add_argument("--max-num-seqs", type=int, default=8,
                        help="最大并发序列数（长上下文模式默认 8）")
    parser.add_argument("--max-batched-tokens", type=int, default=8192,
                        help="单次 batch 最大 token 数")
    parser.add_argument("--tensor-parallel", type=int, default=1,
                        help="张量并行数 (单卡=1, 双卡=2)")
    parser.add_argument("--quant", choices=["awq", "gptq", "fp8"],
                        default="", help="量化方法 (留空=不量化; awq/gptq 需对应量化模型)")
    parser.add_argument("--dtype", default="auto",
                        help="数据类型 (auto/float16/bfloat16)")
    parser.add_argument("--lora-path", help="LoRA 适配器路径")
    parser.add_argument("--max-lora-rank", type=int, default=16,
                        help="LoRA 最大秩")
    parser.add_argument("--disable-prefix-caching", action="store_true",
                        default=False, help="禁用前缀缓存 (vLLM 0.7+ 默认已启用)")

    args = parser.parse_args()

    cmd = build_args(args)
    print(f"[vLLM] 启动命令: {' '.join(cmd)}")
    print(f"[vLLM] API 地址: http://{args.host}:{args.port}/v1")
    print(f"[vLLM] 模型: {args.model}")
    print(f"[vLLM] 量化: {args.quant or 'FP16 (不量化)'}")
    print(f"[vLLM] 张量并行: TP={args.tensor_parallel}")
    print(f"[vLLM] 最大并发序列: {args.max_num_seqs}")
    print("-" * 60)

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
