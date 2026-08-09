# -*- coding: utf-8 -*-
"""服务器端 loss token 口径校验（上传到训练服务器运行）。

用途: 用与训练完全相同的预处理管线, 对 combined_eval/combined_train 计算
      actual_loss_tokens = (labels != -100).sum(), 与本地 token_stats_qwen3.json 对齐。

用法:
    python verify_loss_tokens_server.py <data.jsonl>   # 输出每条总token/loss token
依赖: transformers + 官方 Qwen/Qwen3-8B 权重（可使用服务器本地别名目录）

注意: 请将 enable_thinking / loss mask 范围与训练脚本保持一致。
"""
import json
import argparse
import os
from pathlib import Path

from transformers import AutoTokenizer

DEFAULT_MODEL = os.getenv(
    "QWEN3_BASE_MODEL_PATH",
    "/home/szw/lhm2/runtime/models/Qwen3-8B-Instruct",
)
ENABLE_THINKING = False  # 与训练预处理保持一致


def main():
    parser = argparse.ArgumentParser(description="Verify supervised-token counts with the real Qwen3 tokenizer")
    parser.add_argument("data", type=Path)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Local Qwen3 path or official Qwen/Qwen3-8B repository ID",
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    data_path = args.data
    recs = [json.loads(l) for l in data_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    tok = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    kw = {"enable_thinking": ENABLE_THINKING}

    total_t, loss_t = 0, 0
    for r in recs:
        text = tok.apply_chat_template(r["messages"], tokenize=False, chat_template_kwargs=kw)
        enc = tok(text, return_offsets_mapping=True)
        ids = enc["input_ids"]
        # 定位 assistant 段字符区间 (整体 tokenize 后按偏移标记 labels)
        ranges = []
        for i, m in enumerate(r["messages"]):
            if m["role"] == "assistant":
                s = len(tok.apply_chat_template(r["messages"][:i], tokenize=False, chat_template_kwargs=kw))
                e = len(tok.apply_chat_template(r["messages"][: i + 1], tokenize=False, chat_template_kwargs=kw))
                ranges.append((s, e))
        labels = [-100] * len(ids)
        for s, e in ranges:
            for idx, (a, b) in enumerate(enc["offset_mapping"]):
                if a >= s and a < e and b > a:
                    labels[idx] = ids[idx]
        n_total = len(ids)
        n_loss = sum(1 for x in labels if x != -100)
        total_t += n_total
        loss_t += n_loss
        print(f"{r['id']}: total={n_total} loss={n_loss}")
    print(f"\n合计: {len(recs)}条 total={total_t} loss={loss_t}")


if __name__ == "__main__":
    main()
