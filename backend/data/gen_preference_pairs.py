"""偏好对生成器 - 从 gold_prompts.json 和 SFT 数据生成 chosen/rejected 偏好对。

chosen = SFT 生成的角色风格回复
rejected = 轮换 4 种降质模式（通用AI语气/角色崩坏/冗长模板/拒绝回答）
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _BACKEND_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))


REJECTION_PATTERNS = [
    "您好，我是AI助手。关于您的问题，我查询了相关信息。根据我的知识库，这个问题的答案涉及到多个方面，需要综合考虑。希望这个回答对您有帮助，如果您还有其他问题，请随时提问。",
    "好的，我来回答你的问题。这个问题其实很简单，答案就是你所问的内容。如果你还有其他问题，随时可以问我哦，我会尽力帮助你解答的。",
    "抱歉，我无法回答这个问题。建议您查阅相关资料或咨询专业人士。作为一个AI语言模型，我的知识有限，可能无法提供您需要的全部信息。",
    "这个问题嘛...怎么说呢...其实说起来比较复杂。简单来说就是你需要去了解一下相关的内容，然后就会明白了。具体细节我可能不太清楚，但大概就是这个意思吧。",
]


def generate_mock_chosen(item: Dict[str, Any]) -> str:
    persona = item.get("persona", "hutao")
    prompt = item.get("prompt", "")
    expected = item.get("expected_behavior", "")
    templates = {
        "hutao": f"嘿嘿！你问的「{prompt[:20]}...」嘛，本堂主当然知道！{expected}。还有什么想问的尽管开口～",
        "zhongli": f"此事说来话长。{expected}。以契约之名，我已为你详细说明。",
        "qiqi": f"嗯...{expected}...七七...记下来了...",
        "xiao": f"...{expected}。此事我已告知与你。",
    }
    return templates.get(persona, templates["hutao"])


def select_prompts(gold_prompts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_category: Dict[str, List[Dict[str, Any]]] = {}
    for item in gold_prompts:
        cat = item.get("category", "other")
        by_category.setdefault(cat, []).append(item)

    selected: List[Dict[str, Any]] = []
    quota = {
        "persona": 15,
        "safety": 5,
        "rag_grounded": 8,
        "factual": 4,
        "multiturn": 3,
    }
    for cat, limit in quota.items():
        items = by_category.get(cat, [])
        selected.extend(items[:limit])

    return selected


def load_sft_responses(sft_path: Path) -> Dict[str, str]:
    if not sft_path.exists():
        return {}
    with open(sft_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result: Dict[str, str] = {}
    for item in data:
        item_id = item.get("id", "")
        convs = item.get("conversations", [])
        for conv in convs:
            if conv.get("from") == "assistant" and conv.get("value", "") != "（继续）":
                result[item_id] = conv["value"]
                break
    return result


def generate_pairs(
    gold_prompts_path: Path,
    sft_data_path: Path,
    output_path: Path,
    mock: bool = False,
) -> int:
    if not mock:
        raise ValueError(
            "Legacy Gold/template preference generation is disabled for real training; "
            "use python -m training.persona_alignment sample with reviewed train scenes"
        )
    from training.preference_data_schema import PreferencePair, save_jsonl

    with open(gold_prompts_path, "r", encoding="utf-8") as f:
        gold_data = json.load(f)
    prompts = gold_data.get("prompts", [])

    sft_responses: Dict[str, str] = {}
    if not mock:
        sft_responses = load_sft_responses(sft_data_path)

    selected = select_prompts(prompts)
    logger.info(f"选中 {len(selected)} 条 prompt 生成偏好对")

    pairs: List[PreferencePair] = []
    for i, item in enumerate(selected):
        prompt_id = item.get("id", f"prompt_{i}")
        prompt_text = item.get("prompt", "")
        category = item.get("category", "other")

        chosen = sft_responses.get(prompt_id, "")
        if not chosen:
            chosen = generate_mock_chosen(item)

        rejected = REJECTION_PATTERNS[i % len(REJECTION_PATTERNS)]

        pair = PreferencePair(
            prompt=prompt_text,
            chosen=chosen,
            rejected=rejected,
            rubric={"persona_consistency": 1.0, "safety": 0.5},
            annotator="script_mock" if mock else "sft_generated_unreviewed",
            metadata={
                "source": "gold_prompts",
                "category": category,
                "gold_prompt_id": prompt_id,
                "mock": True,
            },
            review_status="pending",
        )
        pairs.append(pair)

    save_jsonl(pairs, output_path)
    logger.info(f"已保存 {len(pairs)} 条偏好对到 {output_path}")
    return len(pairs)


def main():
    parser = argparse.ArgumentParser(description="生成偏好对数据")
    parser.add_argument(
        "--gold-set",
        type=str,
        default="",
        help="gold_prompts.json 路径",
    )
    parser.add_argument(
        "--sft-data",
        type=str,
        default="backend/hutao_dialogues.json",
        help="SFT 训练数据路径（用于 chosen 回复）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="backend/data/preference_pairs.jsonl",
    )
    parser.add_argument("--mock", action="store_true", help="Mock 模式（不读 SFT 数据）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    gold_path = Path(args.gold_set) if args.gold_set else (
        _BACKEND_DIR / "evaluation" / "gold_prompts.json"
    )
    sft_path = Path(args.sft_data)
    if not sft_path.is_absolute():
        sft_path = _PROJECT_ROOT / sft_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = _PROJECT_ROOT / output_path

    count = generate_pairs(gold_path, sft_path, output_path, mock=args.mock)
    print(f"\n生成完成: {count} 条偏好对 → {output_path}")


if __name__ == "__main__":
    main()
