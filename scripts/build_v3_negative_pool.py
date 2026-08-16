"""Build the V3 negative-example pool for Judge B same-question comparison (Task B.2).

Reads the 111 llm_v3_deepseek samples from V2 train, tags each with its
scene and problem categories (meta-narrative overload / '正因如此' overload /
laughter missing / sharp-expression missing), and assigns a stable
``sample_spec_id`` derived from (scene, idx, "v3neg") so that the V4
generator can produce a candidate answering the **same human dialogue**
for fair A/B comparison in Judge B.

Outputs:
  - ``archive/v3_pipeline/llm_v4_judged/v3_negative_pool.jsonl``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND = PROJECT_ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

V2_TRAIN_PATH = (
    BACKEND / "data" / "character_dialogues" / "experiments" / "archive" / "v2_canonical" / "tsukiyashiro_kisaki_train.json"
)
OUTPUT_DIR = (
    BACKEND / "data" / "character_dialogues" / "experiments" / "archive" / "v3_pipeline" / "llm_v4_judged"
)
NEGATIVE_POOL_PATH = OUTPUT_DIR / "v3_negative_pool.jsonl"

# Reuse the same SCENE_RULES as build_few_shot_pool.py (kept in sync manually
# to avoid an inter-module import cycle).
SCENE_RULES: list[tuple[str, str, tuple[str, ...]]] = [
    ("日常问候", "打招呼、问好、早晚安、天气",
     ("早上好", "早啊", "早安", "晚安", "你好", "嗨", "好久不见", "中午好", "晚上好")),
    ("书籍讨论", "讨论书籍、但不主动引用书名",
     ("书", "阅读", "读书", "翻页", "章节", "图书馆", "封面", "书架", "文字")),
    ("回忆故事", "讲述过去的经历、但简洁不冗长",
     ("记得", "以前", "过去", "那时", "回忆", "曾经", "小时候", "那年", "之前")),
    ("情感倾诉", "开心/难过/烦恼/压力、求安慰",
     ("难过", "开心", "烦恼", "压力", "累", "心痛", "伤心", "高兴", "不安", "寂寞", "孤独")),
    ("求助建议", "问问题、求建议、人际关系问题",
     ("怎么办", "建议", "帮我", "教我", "求", "如何", "怎样", "该不该", "能不能")),
    ("观点讨论", "对某件事的看法、立场表达",
     ("觉得", "认为", "看法", "立场", "观点", "以为", "感觉", "看来")),
    ("深度关怀", "一方状态不好时另一方关心、但克制",
     ("没事吧", "还好吗", "怎么了", "不舒服", "别勉强", "注意", "保重")),
    ("幽默互怼", "朋友之间开玩笑、冷幽默、反讽",
     ("呼呼呼", "噗", "呵呵", "哎呀", "玩笑", "逗", "有趣", "搞笑", "无聊")),
    ("突发奇想", "突然想到的点子、简短回应",
     ("突然", "忽然", "如果说", "假如", "假设", "要是", "如果", "万一")),
    ("角色设定", "关于妃的过去、喜好、简洁回答",
     ("琉璃", "彼方", "夜子", "理央", "魔法使", "妃", "我", "自己")),
    ("闲聊吐槽", "今天发生的事、遇到的人、看到的趣闻",
     ("今天", "刚才", "遇到", "看见", "听说", "真是的", "受不了", "奇怪")),
    ("日常场景", "吃饭、散步、休息等日常对话",
     ("吃", "睡", "走", "休息", "散步", "坐", "茶", "饭", "床")),
]

# Problem tag definitions
META_NARRATIVE_WORDS = ("故事", "作者", "剧本", "出场人物", "规则", "书", "文字", "章节", "页")
SHARP_EXPRESSIONS = ("恕我拒绝", "你疯了吗", "你有病", "太危险", "这可不行", "不会写", "没有那个必要")
LAUGHTER_VARIANTS = ("呼呼呼", "噗噗", "呵呵", "哈哈", "嘿嘿")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _tag_scene(conversations: list[dict[str, Any]]) -> tuple[str, str]:
    human_turns = [m.get("value", "") for m in conversations if m.get("from") == "human"]
    assistant_turns = [m.get("value", "") for m in conversations if m.get("from") == "assistant"]
    text = " ".join(human_turns[:1] + assistant_turns[:1])
    for scene, desc, keywords in SCENE_RULES:
        if any(kw in text for kw in keywords):
            return scene, desc
    return "日常场景", "吃饭、散步、休息等日常对话"


def _detect_problems(assistant_text: str) -> list[str]:
    problems: list[str] = []
    meta_hits = sum(1 for w in META_NARRATIVE_WORDS if w in assistant_text)
    if meta_hits >= 2:
        problems.append("meta_narrative_overload")
    if "正因如此" in assistant_text:
        problems.append("zheng_yin_ci_overuse")
    if not any(laugh in assistant_text for laugh in LAUGHTER_VARIANTS):
        problems.append("laughter_missing")
    if not any(sharp in assistant_text for sharp in SHARP_EXPRESSIONS):
        problems.append("sharp_expression_missing")
    if not problems:
        problems.append("other_style_deviation")
    return problems


def _stable_spec_id(scene: str, idx_in_scene: int) -> str:
    """Stable ID for a v3 negative: kisaki_v3neg_<scene>_<idx>.

    The V4 candidate that answers the same human dialogue will share this ID.
    """
    scene_tag = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]", "_", scene)[:20]
    return f"kisaki_v3neg_{scene_tag}_{idx_in_scene:03d}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build V3 negative pool for Judge B same-question compare")
    parser.add_argument("--v2-train", type=Path, default=V2_TRAIN_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    v2_train = _read_json(args.v2_train)
    v3_negatives = [
        s for s in v2_train
        if (s.get("metadata") or {}).get("data_source") == "llm_v3_deepseek"
    ]
    if len(v3_negatives) != 111:
        print(json.dumps(
            {"error": f"expected 111 llm_v3_deepseek samples, got {len(v3_negatives)}"},
            ensure_ascii=False,
        ))
        return 2

    # Group by scene, then assign stable IDs
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for sample in v3_negatives:
        scene, _ = _tag_scene(sample.get("conversations", []))
        by_scene.setdefault(scene, []).append(sample)

    records: list[dict[str, Any]] = []
    scene_counts: dict[str, int] = {}
    problem_counts: dict[str, int] = {}
    for scene, samples in sorted(by_scene.items()):
        for idx, sample in enumerate(samples):
            conversations = sample.get("conversations", [])
            human_dialogue = [m.get("value", "") for m in conversations if m.get("from") == "human"]
            assistant_text = " ".join(
                m.get("value", "") for m in conversations if m.get("from") == "assistant"
            )
            problems = _detect_problems(assistant_text)
            sample_spec_id = _stable_spec_id(scene, idx)
            record = {
                "sample_spec_id": sample_spec_id,
                "v3_sample_id": sample.get("id"),
                "scene": scene,
                "human_dialogue": human_dialogue,
                "v3_assistant_response": assistant_text,
                "conversations": conversations,
                "problem_tags": problems,
                "meta_narrative_word_hits": sum(
                    1 for w in META_NARRATIVE_WORDS if w in assistant_text
                ),
                "has_zheng_yin_ci": "正因如此" in assistant_text,
                "laughter_variants_found": [
                    laught for laught in LAUGHTER_VARIANTS if laught in assistant_text
                ],
                "sharp_expressions_found": [
                    sharp for sharp in SHARP_EXPRESSIONS if sharp in assistant_text
                ],
            }
            records.append(record)
            scene_counts[scene] = scene_counts.get(scene, 0) + 1
            for p in problems:
                problem_counts[p] = problem_counts.get(p, 0) + 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with NEGATIVE_POOL_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")

    # Verify: all sample_spec_ids are unique
    ids = [rec["sample_spec_id"] for rec in records]
    duplicates = [sid for sid in set(ids) if ids.count(sid) > 1]

    print(json.dumps({
        "built": not duplicates,
        "output_path": str(NEGATIVE_POOL_PATH),
        "total_negatives": len(records),
        "scene_distribution": scene_counts,
        "problem_tag_distribution": problem_counts,
        "duplicate_spec_ids": duplicates,
    }, ensure_ascii=False, indent=2))
    return 0 if not duplicates else 1


if __name__ == "__main__":
    raise SystemExit(main())
