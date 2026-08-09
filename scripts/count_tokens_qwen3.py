# -*- coding: utf-8 -*-
"""Qwen3 tokenizer 精确统计: 总 token + assistant loss token。

- tokenizer: Qwen3-4B (与 Qwen3-8B-Instruct 同一 Qwen3 tokenizer, vocab 151643)
- 配置: enable_thinking=False (角色扮演标准, 不插入 <think> 块)
- assistant loss token = 每个 assistant 回合内容 + <|im_end|> (标准 SFT mask)
- 统计对象: combined_merged/train/eval + 分模块/state
"""
import json
import argparse
import os
from collections import Counter, OrderedDict
from pathlib import Path

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = PROJECT_ROOT / "backend" / "data" / "character_dialogues" / "experiments"
DEFAULT_TOKENIZER = os.getenv(
    "QWEN3_TOKENIZER_PATH",
    os.getenv("QWEN3_BASE_MODEL_PATH", "Qwen/Qwen3-8B"),
)
KWARGS = {"enable_thinking": False}


def n_tokens(tok, msgs):
    out = tok.apply_chat_template(msgs, tokenize=True, chat_template_kwargs=KWARGS)
    # transformers 5.x: 返回 BatchEncoding (键: input_ids/attention_mask) 或单样本 ids 列表
    if hasattr(out, "input_ids"):
        ids = out["input_ids"]
        return len(ids[0]) if ids and isinstance(ids[0], (list, tuple)) else len(ids)
    if isinstance(out, list) and len(out) == 1:
        elem = out[0]
        if hasattr(elem, "ids"):
            return len(elem.ids)
        if isinstance(elem, list):
            return len(elem)
    return len(out)


def count_sample(tok, msgs):
    total = n_tokens(tok, msgs)
    loss = 0
    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            n1 = n_tokens(tok, msgs[:i])
            n2 = n_tokens(tok, msgs[: i + 1])
            loss += n2 - n1
    return total, loss


def load(name):
    p = BASE / name
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    parser = argparse.ArgumentParser(description="Count Qwen3 chat and supervised tokens")
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help="Local Qwen3 model/tokenizer path or Hugging Face repository ID",
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    tok = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    print(f"tokenizer: Qwen3 (vocab={tok.vocab_size}), enable_thinking={KWARGS['enable_thinking']}\n")

    files = {"merged": "combined_merged.jsonl", "train": "combined_train.jsonl", "eval": "combined_eval.jsonl"}
    result = {}
    for label, fname in files.items():
        recs = load(fname)
        total_t, loss_t = 0, 0
        by_module = Counter()
        by_state = Counter()
        for r in recs:
            t, l = count_sample(tok, r["messages"])
            total_t += t
            loss_t += l
            md = r["metadata"]
            by_module[f"{md.get('module','?')}({md.get('state','?')})"] += 1
        result[label] = {"samples": len(recs), "total_tokens": total_t, "assistant_loss_tokens": loss_t}
        print(f"{label}: {len(recs)}条 | 总token {total_t} | assistant loss token {loss_t} | 每样本均loss {loss_t/len(recs):.1f}")
        print(f"    模块/state: {dict(by_module)}")

    # 分模块精确统计 (merged 口径)
    print("\n=== 分模块 (merged) ===")
    mods = OrderedDict()
    for r in load("combined_merged.jsonl"):
        md = r["metadata"]
        key = f"{md.get('module','?')}|{md.get('state','?')}"
        mods.setdefault(key, []).append(r)
    for key, recs in mods.items():
        t = sum(count_sample(tok, r["messages"])[0] for r in recs)
        l = sum(count_sample(tok, r["messages"])[1] for r in recs)
        print(f"  {key}: {len(recs)}条 | 总token {t} | loss token {l} ({l/t*100:.1f}% of total)")

    out = BASE / "token_stats_qwen3.json"
    out.write_text(json.dumps({"tokenizer": "Qwen3 (Qwen3-4B 同源)", "enable_thinking": False,
                               "loss_mask": "assistant内容+<|im_end|>", **result},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写出: {out.name}")


if __name__ == "__main__":
    main()
