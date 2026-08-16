#!/usr/bin/env python3
"""Merge one PEFT adapter into the BF16 base model for comparable R1 evaluation."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND = PROJECT_ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from evaluation.experiment_contracts import git_commit, sha256_file


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allowed-root", type=Path, default=Path(os.getenv("QQCHAT_LAB_ROOT", str(PROJECT_ROOT / "runtime"))))
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args()

    for source in (args.base_model, args.adapter):
        if not source.exists():
            raise SystemExit(f"required source does not exist: {source}")
    if not _inside(args.output, args.allowed_root):
        raise SystemExit(f"output must remain below {args.allowed_root}")

    marker = args.output / "merge_manifest.json"
    if marker.exists():
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "complete"
            and existing.get("experiment_id") == args.experiment_id
            and existing.get("adapter") == str(args.adapter)
        ):
            print(marker)
            return 0
        raise SystemExit(f"refusing to overwrite incomplete or mismatched merge: {args.output}")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.output}")

    free_bytes = shutil.disk_usage(args.output.parent if args.output.parent.exists() else args.allowed_root).free
    if free_bytes < 24 * 1024**3:
        raise SystemExit("at least 24 GiB free disk is required for a merged Qwen3-8B model")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_config = json.loads((args.adapter / "adapter_config.json").read_text(encoding="utf-8"))
    base_config = json.loads((args.base_model / "config.json").read_text(encoding="utf-8"))
    if base_config.get("quantization_config"):
        raise SystemExit("R1 merge requires a non-quantized BF16 base model")

    args.output.mkdir(parents=True, exist_ok=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
        merged = model.merge_and_unload(safe_merge=True)
        merged.save_pretrained(args.output, safe_serialization=True, max_shard_size="4GB")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=False)
        tokenizer.save_pretrained(args.output)
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "experiment_id": args.experiment_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(PROJECT_ROOT),
            "base_model": str(args.base_model),
            "base_config_sha256": sha256_file(args.base_model / "config.json"),
            "adapter": str(args.adapter),
            "adapter_method": {
                "use_dora": bool(adapter_config.get("use_dora", False)),
                "use_rslora": bool(adapter_config.get("use_rslora", False)),
            },
            "dtype": "bfloat16",
            "safe_merge": True,
        }
        marker.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(marker)
        return 0
    except Exception:
        failure = args.output / "merge_failed.json"
        failure.write_text(json.dumps({"status": "failed", "experiment_id": args.experiment_id}, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
