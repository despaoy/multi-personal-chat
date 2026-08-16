"""训练数据清洗流水线，保留通用的八步清洗实现。

用法:
    python -m evaluation.clean_training_data --input <raw_data.json> --output <clean_dir>
    python -m evaluation.clean_training_data --input backend/training/data/raw/ --output backend/training/data/clean/ --seed 42

功能:
    1. 格式校验：确保 ShareGPT/Qwen/Alpaca 格式合法
    2. 去重：精确去重 + MinHash 近似去重（Jaccard ≥ 0.9）
    3. 长度过滤：< 20 字丢弃，> 1024 字截断，< 2 轮丢弃
    4. 模板标记清理：移除 {character}、<|im_start|> 等残留
    5. 对话级分割：train 80% / val 10% / test 10%，seed=42
    6. 角色一致性预检：contradiction_rate > 0.3 移入审核队列
    7. 安全反向检测：对 safety prompt 未拒绝的样本移入审核队列
    8. 统计报告：输出 cleaning_report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_SEED = 42
_MIN_TOTAL_LEN = 20
_MAX_RESPONSE_LEN = 1024
_MIN_TURNS = 2
_DEDUP_JACCARD_THRESHOLD = 0.9
_CONTRADICTION_THRESHOLD = 0.3

_TEMPLATE_PATTERNS = [
    r"\{character\}",
    r"\{persona\}",
    r"\{user\}",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"<\|system\|>",
    r"<\|user\|>",
    r"<\|assistant\|>",
]
_TEMPLATE_RE = re.compile("|".join(_TEMPLATE_PATTERNS))


@dataclass
class CleaningStats:
    """清洗统计。"""
    total_input: int = 0
    format_valid: int = 0
    format_invalid: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    too_short: int = 0
    too_long_truncated: int = 0
    too_few_turns: int = 0
    template_cleaned: int = 0
    persona_review: int = 0
    safety_review: int = 0
    final_kept: int = 0
    split_counts: Dict[str, int] = field(default_factory=lambda: {"train": 0, "val": 0, "test": 0})

    def to_dict(self) -> Dict[str, Any]:
        return {k: (v if not isinstance(v, dict) else dict(v)) for k, v in self.__dict__.items()}


def _normalize_conversation(item: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    """将各种格式统一为 [{"role": "user"/"assistant", "content": "..."}] 列表。"""
    if "conversations" in item:
        convs = item["conversations"]
        normalized = []
        for msg in convs:
            if "from" in msg and "value" in msg:
                role = "user" if msg["from"] in ("human", "user") else "assistant"
                normalized.append({"role": role, "content": msg["value"]})
            elif "role" in msg and "content" in msg:
                if msg["role"] == "system":
                    continue
                normalized.append({"role": msg["role"], "content": msg["content"]})
            else:
                return None
        return normalized if len(normalized) >= 2 else None
    if "instruction" in item and "output" in item:
        return [{"role": "user", "content": item["instruction"]}, {"role": "assistant", "content": item["output"]}]
    if "prompt" in item and "response" in item:
        return [{"role": "user", "content": item["prompt"]}, {"role": "assistant", "content": item["response"]}]
    return None


def _extract_system(item: Dict[str, Any]) -> str:
    """提取 system prompt。"""
    if "system" in item and isinstance(item["system"], str):
        return item["system"]
    if "conversations" in item:
        for msg in item["conversations"]:
            if msg.get("role") == "system" or msg.get("from") == "system":
                return msg.get("content", "") or msg.get("value", "")
    return ""


def _clean_template_markers(text: str) -> Tuple[str, bool]:
    """移除残留的 prompt 模板标记，返回 (cleaned_text, was_modified)。"""
    cleaned = _TEMPLATE_RE.sub("", text)
    cleaned = re.sub(r"  +", " ", cleaned).strip()
    return cleaned, cleaned != text


def _exact_dedup_key(item: Dict[str, Any]) -> str:
    """生成精确去重 key：(system, first_user_msg)。"""
    system = item.get("system", "")
    convs = item.get("conversations", [])
    first_user = ""
    for c in convs:
        role = c.get("role") or c.get("from", "")
        if role in ("user", "human"):
            first_user = c.get("content", "") or c.get("value", "")
            break
    return hashlib.sha256(f"{system}||{first_user}".encode("utf-8")).hexdigest()


def _minhash_signature(text: str, num_perm: int = 64, k: int = 3) -> List[int]:
    """简单的 MinHash 签名（基于 k-shingle）。"""
    if len(text) < k:
        text = text.ljust(k, "\0")
    shingles = {text[i:i + k] for i in range(len(text) - k + 1)}
    if not shingles:
        return [0] * num_perm
    signatures = []
    for i in range(num_perm):
        h = hashlib.md5(f"{i}|".encode("utf-8"))
        min_hash = min(
            int.from_bytes(h.update(s.encode("utf-8")) or h.digest(), "big") if False
            else int.from_bytes(hashlib.md5(f"{i}:{s}".encode("utf-8")).digest(), "big")
            for s in shingles
        )
        signatures.append(min_hash)
    return signatures


def _jaccard_from_signatures(sig_a: List[int], sig_b: List[int]) -> float:
    """根据 MinHash 签名估算 Jaccard 相似度。"""
    if len(sig_a) != len(sig_b) or not sig_a:
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def _build_near_dedup_index(items: List[Dict[str, Any]], threshold: float) -> Set[int]:
    """返回应移除的近似重复项索引（保留首次出现的）。"""
    signatures: List[Tuple[int, List[int]]] = []
    for idx, item in enumerate(items):
        convs = item.get("conversations", [])
        all_text = " ".join(c.get("content", "") or c.get("value", "") for c in convs)
        signatures.append((idx, _minhash_signature(all_text)))

    to_remove: Set[int] = set()
    for i in range(len(signatures)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(signatures)):
            if j in to_remove:
                continue
            sim = _jaccard_from_signatures(signatures[i][1], signatures[j][1])
            if sim >= threshold:
                to_remove.add(j)
                logger.debug(f"近似去重: item[{j}] 与 item[{i}] 相似度={sim:.3f}")
    return to_remove


def _persona_consistency_check(item: Dict[str, Any], persona: Optional[str]) -> Tuple[bool, float]:
    """角色一致性预检，返回 (pass, contradiction_rate)。"""
    if not persona:
        return True, 0.0
    try:
        from evaluation.persona_metrics import PersonaMetrics, PERSONA_FACTS
        if persona not in PERSONA_FACTS:
            return True, 0.0
        convs = item.get("conversations", [])
        responses = [c.get("content", "") or c.get("value", "") for c in convs if c.get("role") == "assistant" or c.get("from") == "gpt"]
        if not responses:
            return True, 0.0
        metrics = PersonaMetrics()
        rate = metrics.contradiction_rate(responses, PERSONA_FACTS[persona])
        return rate <= _CONTRADICTION_THRESHOLD, rate
    except Exception as e:
        logger.warning(f"角色一致性预检失败: {e}")
        return True, 0.0


def _safety_reverse_check(item: Dict[str, Any]) -> bool:
    """安全反向检测：若 user 内容匹配安全测试集 prompt，检查 assistant 是否拒绝。

    返回 True 表示通过（未触发或已拒绝），False 表示需移入审核队列。
    """
    try:
        safety_path = Path(__file__).parent / "safety_test_set.json"
        if not safety_path.exists():
            return True
        with open(safety_path, "r", encoding="utf-8") as f:
            safety_set = json.load(f)
        safety_prompts: Set[str] = set()
        for category_prompts in safety_set.values():
            if isinstance(category_prompts, list):
                for p in category_prompts:
                    if isinstance(p, dict) and "prompt" in p:
                        safety_prompts.add(p["prompt"].strip())

        convs = item.get("conversations", [])
        for c in convs:
            role = c.get("role") or c.get("from", "")
            if role in ("user", "human"):
                content = (c.get("content", "") or c.get("value", "")).strip()
                if content in safety_prompts:
                    assistant_replies = [
                        (cc.get("content", "") or cc.get("value", ""))
                        for cc in convs
                        if cc.get("role") == "assistant" or cc.get("from") == "gpt"
                    ]
                    if assistant_replies:
                        reply = assistant_replies[0].lower()
                        refusal_markers = ["不能", "拒绝", "不可以", "无法", "抱歉", "不会", "作为", "我是", "坚持"]
                        if not any(m in reply for m in refusal_markers):
                            return False
        return True
    except Exception as e:
        logger.warning(f"安全反向检测失败: {e}")
        return True


def _split_conversations(items: List[Dict[str, Any]], seed: int = _SEED) -> Dict[str, List[Dict[str, Any]]]:
    """对话级 train/val/test 分割（80/10/10）。"""
    rng = random.Random(seed)
    indices = list(range(len(items)))
    rng.shuffle(indices)
    n = len(indices)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    return {
        "train": [items[i] for i in indices[:train_end]],
        "val": [items[i] for i in indices[train_end:val_end]],
        "test": [items[i] for i in indices[val_end:]],
    }


def _load_input(input_path: Path) -> List[Dict[str, Any]]:
    """加载输入数据（单文件或目录）。"""
    if input_path.is_dir():
        data = []
        for ext in ("*.json", "*.jsonl"):
            for f in input_path.glob(ext):
                data.extend(_load_single_file(f))
        return data
    return _load_single_file(input_path)


def _load_single_file(file_path: Path) -> List[Dict[str, Any]]:
    """加载单个 JSON/JSONL 文件。"""
    with open(file_path, "r", encoding="utf-8") as f:
        if file_path.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        content = json.load(f)
        if isinstance(content, list):
            return content
        if isinstance(content, dict) and "data" in content:
            return content["data"]
        return [content]


def clean_training_data(
    input_path: Path,
    output_dir: Path,
    seed: int = _SEED,
    dedup_threshold: float = _DEDUP_JACCARD_THRESHOLD,
) -> CleaningStats:
    """执行完整清洗流水线。"""
    stats = CleaningStats()
    output_dir.mkdir(parents=True, exist_ok=True)
    review_dir = output_dir / "review_queue"
    review_dir.mkdir(exist_ok=True)

    # Step 0: 加载
    raw_data = _load_input(input_path)
    stats.total_input = len(raw_data)
    logger.info(f"步骤 0: 加载 {len(raw_data)} 条原始数据")

    # Step 1: 格式校验 + 归一化
    normalized: List[Dict[str, Any]] = []
    for item in raw_data:
        convs = _normalize_conversation(item)
        if convs is None or len(convs) < _MIN_TURNS:
            stats.format_invalid += 1
            continue
        system = _extract_system(item)
        persona = item.get("persona")
        normalized.append({
            "conversations": convs,
            "system": system,
            "persona": persona,
            "source": item.get("source", "unknown"),
            "scene": item.get("scene", ""),
            "turns": len(convs),
        })
        stats.format_valid += 1
    logger.info(f"步骤 1: 格式校验通过 {stats.format_valid}，失败 {stats.format_invalid}")

    # Step 2: 精确去重
    seen_keys: Set[str] = set()
    exact_deduped: List[Dict[str, Any]] = []
    for item in normalized:
        key = _exact_dedup_key(item)
        if key in seen_keys:
            stats.exact_duplicates += 1
            continue
        seen_keys.add(key)
        exact_deduped.append(item)
    logger.info(f"步骤 2: 精确去重移除 {stats.exact_duplicates} 条")

    # Step 3: 近似去重（MinHash）
    near_dup_indices = _build_near_dedup_index(exact_deduped, dedup_threshold)
    near_deduped = [item for idx, item in enumerate(exact_deduped) if idx not in near_dup_indices]
    stats.near_duplicates = len(near_dup_indices)
    logger.info(f"步骤 3: 近似去重移除 {stats.near_duplicates} 条")

    # Step 4: 长度过滤 + 模板标记清理
    cleaned: List[Dict[str, Any]] = []
    for item in near_deduped:
        total_len = sum(len(c["content"]) for c in item["conversations"])
        if total_len < _MIN_TOTAL_LEN:
            stats.too_short += 1
            continue
        modified = False
        for c in item["conversations"]:
            new_content, was_modified = _clean_template_markers(c["content"])
            if was_modified:
                modified = True
                c["content"] = new_content
            if len(c["content"]) > _MAX_RESPONSE_LEN and c["role"] == "assistant":
                c["content"] = c["content"][:_MAX_RESPONSE_LEN]
                stats.too_long_truncated += 1
        if modified:
            stats.template_cleaned += 1
        if len(item["conversations"]) < _MIN_TURNS:
            stats.too_few_turns += 1
            continue
        cleaned.append(item)
    logger.info(f"步骤 4: 长度过滤（短={stats.too_short}, 截断={stats.too_long_truncated}, 轮数不足={stats.too_few_turns}），模板清理={stats.template_cleaned}")

    # Step 5: 角色一致性预检
    persona_passed: List[Dict[str, Any]] = []
    persona_review: List[Dict[str, Any]] = []
    for item in cleaned:
        passed, rate = _persona_consistency_check(item, item.get("persona"))
        item["_contradiction_rate"] = rate
        if passed:
            persona_passed.append(item)
        else:
            persona_review.append(item)
            stats.persona_review += 1
    logger.info(f"步骤 5: 角色一致性预检移入审核 {stats.persona_review} 条")

    # Step 6: 安全反向检测
    safety_passed: List[Dict[str, Any]] = []
    safety_review: List[Dict[str, Any]] = []
    for item in persona_passed:
        if _safety_reverse_check(item):
            safety_passed.append(item)
        else:
            safety_review.append(item)
            stats.safety_review += 1
    logger.info(f"步骤 6: 安全反向检测移入审核 {stats.safety_review} 条")

    # Step 7: 对话级分割
    splits = _split_conversations(safety_passed, seed=seed)
    stats.split_counts = {k: len(v) for k, v in splits.items()}
    stats.final_kept = len(safety_passed)
    logger.info(f"步骤 7: 分割 train={stats.split_counts['train']}, val={stats.split_counts['val']}, test={stats.split_counts['test']}")

    # Step 8: 写出
    for split_name, split_data in splits.items():
        out_path = output_dir / f"{split_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)

    if persona_review:
        with open(review_dir / "persona_review.json", "w", encoding="utf-8") as f:
            json.dump(persona_review, f, indent=2, ensure_ascii=False)
    if safety_review:
        with open(review_dir / "safety_review.json", "w", encoding="utf-8") as f:
            json.dump(safety_review, f, indent=2, ensure_ascii=False)

    # 写出 split manifest
    manifest = {
        "seed": seed,
        "split_ratio": {"train": 0.8, "val": 0.1, "test": 0.1},
        "conversation_level": True,
        "split_counts": stats.split_counts,
        "conversation_ids": {
            split: [item.get("source", str(i)) for i, item in enumerate(data)]
            for split, data in splits.items()
        },
    }
    with open(output_dir / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # 写出统计报告
    report = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "seed": seed,
        "dedup_threshold": dedup_threshold,
        "stats": stats.to_dict(),
        "config": {
            "min_total_len": _MIN_TOTAL_LEN,
            "max_response_len": _MAX_RESPONSE_LEN,
            "min_turns": _MIN_TURNS,
            "contradiction_threshold": _CONTRADICTION_THRESHOLD,
        },
    }
    with open(output_dir / "cleaning_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"步骤 8: 写出完成，最终保留 {stats.final_kept} 条")
    logger.info(f"清洗报告: {output_dir / 'cleaning_report.json'}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="训练数据清洗流水线")
    parser.add_argument("--input", required=True, help="输入数据文件或目录")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--seed", type=int, default=_SEED, help="随机种子")
    parser.add_argument("--dedup-threshold", type=float, default=_DEDUP_JACCARD_THRESHOLD, help="近似去重 Jaccard 阈值")
    parser.add_argument("--verbose", action="store_true", help="详细日志")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    stats = clean_training_data(input_path, output_dir, seed=args.seed, dedup_threshold=args.dedup_threshold)

    print("\n" + "=" * 60)
    print("清洗完成")
    print("=" * 60)
    print(f"输入:           {stats.total_input}")
    print(f"格式无效:       {stats.format_invalid}")
    print(f"精确去重:       {stats.exact_duplicates}")
    print(f"近似去重:       {stats.near_duplicates}")
    print(f"过短丢弃:       {stats.too_short}")
    print(f"超长截断:       {stats.too_long_truncated}")
    print(f"轮数不足:       {stats.too_few_turns}")
    print(f"模板清理:       {stats.template_cleaned}")
    print(f"角色审核:       {stats.persona_review}")
    print(f"安全审核:       {stats.safety_review}")
    print(f"最终保留:       {stats.final_kept}")
    print(f"分割:           train={stats.split_counts['train']}, val={stats.split_counts['val']}, test={stats.split_counts['test']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
