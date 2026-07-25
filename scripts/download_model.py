#!/usr/bin/env python3
"""下载基座模型到指定目录。

支持 modelscope（国内首选）和 huggingface（通过 hf-mirror 镜像）。

用法:
  python scripts/download_model.py --model Qwen3-8B-Instruct
  python scripts/download_model.py --model Qwen2.5-7B-Instruct --source modelscope
  python scripts/download_model.py --model Qwen3-8B --source hf
  python scripts/download_model.py --model Qwen3-8B-Instruct --output /custom/path

注意:
  Qwen3-8B 本身即为 Post-training 完成的 Instruct 版本（混合 thinking/non-thinking），
  Qwen 官方未发布独立的 `-Instruct` 后缀变体。`Qwen3-8B-Instruct` 选项会下载
  modelscope/HF 上的 `Qwen/Qwen3-8B` 权重，并放入名为 `Qwen3-8B-Instruct` 的目录，
  以保持与项目训练配置中的 `base_model_path` 约定一致。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

LAB_ROOT = Path(os.getenv("QQCHAT_LAB_ROOT", "/root/autodl-tmp"))
MODELS_DIR = LAB_ROOT / "runtime" / "models"

# 模型 ID 映射
# - modelscope/HF 上均不存在独立的 `Qwen/Qwen3-8B-Instruct`，因为 Qwen3-8B 本身已是
#   Post-training 完成的 Instruct 版本（支持 thinking/non-thinking 切换）。
# - "Qwen3-8B-Instruct" 在本项目里仅用作目录名，下载时映射到 `Qwen/Qwen3-8B` 权重。
MODEL_IDS: dict[str, dict[str, str | None]] = {
    "Qwen3-8B-Instruct": {
        "modelscope": "Qwen/Qwen3-8B",
        "huggingface": "Qwen/Qwen3-8B",
    },
    "Qwen3-8B": {
        "modelscope": "Qwen/Qwen3-8B",
        "huggingface": "Qwen/Qwen3-8B",
    },
    "Qwen2.5-7B-Instruct": {
        "modelscope": "Qwen/Qwen2.5-7B-Instruct",
        "huggingface": "Qwen/Qwen2.5-7B-Instruct",
    },
    "bge-small-zh-v1.5": {
        "modelscope": "BAAI/bge-small-zh-v1.5",
        "huggingface": "BAAI/bge-small-zh-v1.5",
    },
}


def download_modelscope(model_id: str, target: Path) -> None:
    """通过 modelscope 下载模型。"""
    from modelscope import snapshot_download
    print(f"[modelscope] 下载 {model_id} -> {target}")
    snapshot_download(model_id, local_dir=str(target))
    print(f"[modelscope] 完成: {target}")


def download_hf(model_id: str, target: Path) -> None:
    """通过 huggingface (hf-mirror) 下载模型。"""
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    print(f"[huggingface] 下载 {model_id} -> {target} (via hf-mirror)")
    from huggingface_hub import snapshot_download
    snapshot_download(
        model_id,
        local_dir=str(target),
        endpoint=os.environ["HF_ENDPOINT"],
    )
    print(f"[huggingface] 完成: {target}")


def main() -> int:
    parser = argparse.ArgumentParser(description="下载基座模型")
    parser.add_argument("--model", required=True, choices=list(MODEL_IDS.keys()))
    parser.add_argument("--source", default="auto", choices=["auto", "modelscope", "hf"])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    target = args.output or (MODELS_DIR / args.model)
    target.mkdir(parents=True, exist_ok=True)

    if (target / "config.json").exists():
        print(f"模型已存在: {target}")
        return 0

    ids = MODEL_IDS[args.model]
    source = args.source

    if source == "auto":
        # 优先 modelscope，失败则 hf
        if ids["modelscope"]:
            try:
                download_modelscope(ids["modelscope"], target)
                return 0
            except Exception as e:
                print(f"[modelscope] 失败: {e}", file=sys.stderr)
        if ids["huggingface"]:
            try:
                download_hf(ids["huggingface"], target)
                return 0
            except Exception as e:
                print(f"[huggingface] 失败: {e}", file=sys.stderr)
        print("所有下载源均失败", file=sys.stderr)
        return 1
    elif source == "modelscope":
        if not ids["modelscope"]:
            print(f"modelscope 上无 {args.model}", file=sys.stderr)
            return 1
        download_modelscope(ids["modelscope"], target)
    elif source == "hf":
        if not ids["huggingface"]:
            print(f"huggingface 上无 {args.model}", file=sys.stderr)
            return 1
        download_hf(ids["huggingface"], target)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
