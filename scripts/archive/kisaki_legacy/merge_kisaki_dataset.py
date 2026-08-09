"""合并游戏提取数据 + LLM v3 生成数据，重新切分 train/eval。

输入：
  - tsukiyashiro_kisaki_sft.json (801条游戏提取，from=assistant)
  - kisaki_llm_generated_v3.json (119条 LLM v3，from=gpt)

输出：
  - tsukiyashiro_kisaki_train.json (合并后 90% 训练集)
  - tsukiyashiro_kisaki_eval.json  (合并后 10% 验证集)
  - tsukiyashiro_kisaki_merged.json (合并全集，含来源标记)
  - kisaki_merged_stats.json (统计信息，输出至 archive/legacy_v3_superseded/，已归档)

切分策略：
  - 按 prompt 归一化文本去重
  - 分层抽样：游戏提取 vs LLM 生成 按比例分到 train/eval
  - 固定随机种子 (42) 保证可复现
"""
import json
import re
import unicodedata
import hashlib
import random
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = PROJECT_ROOT / "backend" / "data" / "character_dialogues"
GOLD_PATH = PROJECT_ROOT / "backend" / "evaluation" / "kisaki_gold_set_v1.json"
SFT_PATH = BASE / "tsukiyashiro_kisaki_sft.json"
LLM_V3_PATH = BASE / "kisaki_llm_generated_v3.json"
TRAIN_PATH = BASE / "tsukiyashiro_kisaki_train.json"
EVAL_PATH = BASE / "tsukiyashiro_kisaki_eval.json"
MERGED_PATH = BASE / "tsukiyashiro_kisaki_merged.json"
STATS_PATH = BASE / "archive" / "legacy_v3_superseded" / "kisaki_merged_stats.json"

SEED = 42
EVAL_RATIO = 0.10


def norm_key(text):
    """归一化文本用于去重 key。"""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("……", "…").replace("--", "—")
    text = re.sub(r"\s+", "", text)
    return text.strip().lower()


def pair_key(item):
    """生成 (prompt, reply) 归一化 key 用于去重。"""
    convs = item["conversations"]
    if len(convs) < 2:
        return None
    prompt = convs[0]["value"] if convs[0]["from"] in ("human", "user") else convs[1]["value"]
    reply = convs[1]["value"] if convs[1]["from"] in ("gpt", "assistant") else convs[0]["value"]
    return (norm_key(prompt), norm_key(reply))


def normalize_from(item):
    """统一 from 字段：gpt → assistant, user → human。"""
    role_map = {"gpt": "assistant", "user": "human"}
    new_convs = []
    for msg in item["conversations"]:
        new_msg = dict(msg)
        new_msg["from"] = role_map.get(msg["from"], msg["from"])
        new_convs.append(new_msg)
    item["conversations"] = new_convs
    return item


def load_sft():
    """加载游戏提取数据。"""
    data = json.loads(SFT_PATH.read_text(encoding="utf-8"))
    for item in data:
        item["metadata"] = item.get("metadata", {})
        item["metadata"]["data_source"] = "game_extraction"
    print(f"游戏提取: {len(data)} 条")
    return data


def load_llm_v3():
    """加载 LLM v3 数据。"""
    data = json.loads(LLM_V3_PATH.read_text(encoding="utf-8"))
    for item in data:
        item = normalize_from(item)
        item["metadata"] = item.get("metadata", {})
        item["metadata"]["data_source"] = "llm_v3_deepseek"
    print(f"LLM v3:   {len(data)} 条")
    return data


def load_gold_prompt_keys():
    """Load held-out prompts that must never enter a training split."""
    if not GOLD_PATH.exists():
        return set()
    data = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    prompts = data.get("prompts", data) if isinstance(data, dict) else data
    return {norm_key(item.get("prompt", "")) for item in prompts if item.get("prompt")}


def exclude_gold_prompts(items, gold_keys):
    kept = []
    excluded = []
    for item in items:
        key = pair_key(item)
        prompt_key = key[0] if key else ""
        (excluded if prompt_key in gold_keys else kept).append(item)
    return kept, excluded


def dedup(items):
    """按 (prompt, reply) 归一化去重，保留第一条。"""
    seen = set()
    kept = []
    dropped = 0
    for item in items:
        k = pair_key(item)
        if k is None or k in seen:
            dropped += 1
            continue
        seen.add(k)
        kept.append(item)
    print(f"去重: 丢弃 {dropped} 条重复，保留 {len(kept)} 条")
    return kept


def stratified_split(items, eval_ratio=0.10, seed=42):
    """按 data_source 分层抽样切分 train/eval。"""
    rng = random.Random(seed)

    # 按来源分组
    by_source = {}
    for item in items:
        src = item["metadata"].get("data_source", "unknown")
        by_source.setdefault(src, []).append(item)

    train, eval_set = [], []
    for src, group in by_source.items():
        group_sorted = sorted(group, key=lambda x: x["id"])  # 稳定排序
        rng.shuffle(group_sorted)
        n_eval = max(1, int(len(group_sorted) * eval_ratio)) if len(group_sorted) >= 10 else 0
        eval_set.extend(group_sorted[:n_eval])
        train.extend(group_sorted[n_eval:])
        print(f"  {src}: {len(group_sorted)} → train {len(group_sorted)-n_eval} / eval {n_eval}")

    rng.shuffle(train)
    rng.shuffle(eval_set)
    return train, eval_set


def stats(items, label=""):
    """计算并打印统计。"""
    if not items:
        return {}
    assistant_msgs = [m["value"] for d in items for m in d["conversations"] if m["from"] == "assistant"]
    lengths = [len(m) for m in assistant_msgs]

    src_counts = Counter(d["metadata"].get("data_source", "unknown") for d in items)

    # 长度分布
    len_dist = {"short(<=25)": 0, "mid(26-50)": 0, "long(51-80)": 0, "over80": 0}
    for n in lengths:
        if n <= 25: len_dist["short(<=25)"] += 1
        elif n <= 50: len_dist["mid(26-50)"] += 1
        elif n <= 80: len_dist["long(51-80)"] += 1
        else: len_dist["over80"] += 1

    # 身份特征（仅对 LLM 来源检查）
    char_names = sum(1 for m in assistant_msgs if any(w in m for w in ["琉璃", "彼方", "夜子", "理央"]))
    meta_narrative = sum(1 for m in assistant_msgs if any(w in m for w in ["故事", "书", "文字", "作者", "规则", "出场人物", "展开", "情节"]))

    result = {
        "label": label,
        "total_dialogues": len(items),
        "total_assistant_msgs": len(assistant_msgs),
        "avg_length": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "length_distribution": {k: {"count": v, "pct": round(v/len(lengths)*100, 1)} for k, v in len_dist.items()},
        "source_distribution": dict(src_counts),
        "identity_features": {
            "char_name_mentions": char_names,
            "meta_narrative_mentions": meta_narrative,
        },
    }

    print(f"\n=== {label} ===")
    print(f"对话数: {result['total_dialogues']}, 妃回复数: {result['total_assistant_msgs']}")
    print(f"平均长度: {result['avg_length']} 字")
    print(f"长度分布:")
    for k, v in result["length_distribution"].items():
        print(f"  {k}: {v['count']} ({v['pct']}%)")
    print(f"来源分布: {dict(src_counts)}")
    if any(s == "llm_v3_deepseek" for s in src_counts):
        print(f"身份特征(LLM部分): 提角色名 {char_names}, 元叙事 {meta_narrative}")
    return result


def main():
    print("=" * 60)
    print("合并游戏提取 + LLM v3 数据，重新切分 train/eval")
    print("=" * 60)

    # 1. 加载
    print("\n[1] 加载数据源")
    sft_data = load_sft()
    llm_data = load_llm_v3()

    # 2. 合并
    print("\n[2] 合并并去重")
    merged = sft_data + llm_data
    print(f"合并前总数: {len(merged)}")
    merged = dedup(merged)
    merged, gold_overlaps = exclude_gold_prompts(merged, load_gold_prompt_keys())
    if gold_overlaps:
        print(f"Gold Set isolation: excluded {len(gold_overlaps)} leaked evaluation prompts")
    print(f"合并后总数: {len(merged)}")

    # 3. 切分
    print("\n[3] 分层抽样切分 (90/10)")
    train, eval_set = stratified_split(merged, EVAL_RATIO, SEED)
    print(f"train: {len(train)}, eval: {len(eval_set)}")

    # 4. 统计
    merged_stats = stats(merged, "merged (全集)")
    train_stats = stats(train, "train (训练集)")
    eval_stats = stats(eval_set, "eval (验证集)")

    # 5. 写入
    print("\n[4] 写入文件")
    TRAIN_PATH.write_text(json.dumps(train, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"  {TRAIN_PATH.name}: {len(train)} 条")
    EVAL_PATH.write_text(json.dumps(eval_set, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"  {EVAL_PATH.name}: {len(eval_set)} 条")
    MERGED_PATH.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"  {MERGED_PATH.name}: {len(merged)} 条")

    all_stats = {
        "seed": SEED,
        "eval_ratio": EVAL_RATIO,
        "sft_input_count": len(sft_data),
        "llm_v3_input_count": len(llm_data),
        "merged_count": len(merged),
        "train_count": len(train),
        "eval_count": len(eval_set),
        "merged_stats": merged_stats,
        "train_stats": train_stats,
        "eval_stats": eval_stats,
    }
    STATS_PATH.write_text(json.dumps(all_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"  {STATS_PATH.name}")

    print("\n" + "=" * 60)
    print("完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
