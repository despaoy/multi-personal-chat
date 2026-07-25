"""Build the few-shot pool for the Kisaki V4 generator (Task B.1).

Reads the 715 game_extraction training samples from V3 train, tags each
with a scene label via keyword matching, and writes:
  - ``v3/llm_v4_judged/train_source_whitelist.json``  (precise source-line whitelist)
  - ``v3/llm_v4_judged/few_shot_pool.jsonl``           (per-sample few-shot records)

Verifies:
  - every few-shot record's ``source_file + line_range`` is within the 715
    training-sample whitelist
  - no record's sample_id appears in V3 validation (84 samples)
  - no record's source_file+line appears in V3 validation (line-level check,
    not just chapter-level — addresses V2.1 Critical #5)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND = PROJECT_ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

V3_TRAIN_PATH = BACKEND / "data" / "character_dialogues" / "experiments" / "v3" / "tsukiyashiro_kisaki_train.json"
V3_EVAL_PATH = BACKEND / "data" / "character_dialogues" / "experiments" / "v3" / "tsukiyashiro_kisaki_eval.json"
OUTPUT_DIR = BACKEND / "data" / "character_dialogues" / "experiments" / "v3" / "llm_v4_judged"
WHITELIST_PATH = OUTPUT_DIR / "train_source_whitelist.json"
FEW_SHOT_POOL_PATH = OUTPUT_DIR / "few_shot_pool.jsonl"

# 12 scenes from generate_kisaki_llm_dialogues_v3.py + their keyword signals.
# Order matters: earlier patterns win (e.g. "书" is more specific than "故事").
SCENE_RULES: list[tuple[str, str, tuple[str, ...]]] = [
    ("日常问候", "打招呼、问好、早晚安、天气",
     ("早上好", "早啊", "早安", "晚安", "你好", "嗨", "好久不见", "中午好", "晚上好")),
    ("书籍讨论", "讨论书籍、但不主动引用书名",
     ("书", "阅读", "读书", "翻页", "章节", "图书馆", "封面", "书架", "文字")),
    ("回忆故事", "讲述过去的经历、但简洁不冗长",
     ("记得", "以前", "过去", "那时", "回忆", "曾经", "小时候", "那年", "之前", "记得")),
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
     ("今天", "刚才", "刚才", "遇到", "看见", "听说", "真是的", "受不了", "奇怪")),
    ("日常场景", "吃饭、散步、休息等日常对话",
     ("吃", "睡", "走", "休息", "散步", "坐", "茶", "饭", "床")),
]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_turns(conversations: list[dict[str, Any]]) -> dict[str, Any]:
    human_turns = [m.get("value", "") for m in conversations if m.get("from") == "human"]
    assistant_turns = [m.get("value", "") for m in conversations if m.get("from") == "assistant"]
    return {
        "human": human_turns,
        "assistant": assistant_turns,
        "turn_count": len(assistant_turns),
    }


def _tag_scene(conversations: list[dict[str, Any]]) -> tuple[str, str]:
    """Return (scene, scene_desc) based on first human + first assistant turn."""
    turns = _extract_turns(conversations)
    text = " ".join(turns["human"][:1] + turns["assistant"][:1])
    for scene, desc, keywords in SCENE_RULES:
        if any(kw in text for kw in keywords):
            return scene, desc
    # Fallback: classify as 日常场景 (the most generic bucket)
    return "日常场景", "吃饭、散步、休息等日常对话"


def _build_whitelist_entry(sample: dict[str, Any]) -> dict[str, Any]:
    metadata = sample.get("metadata") or {}
    return {
        "sample_id": sample.get("id"),
        "source_file": metadata.get("source_file"),
        "source_line_start": metadata.get("source_line_start"),
        "source_line_end": metadata.get("source_line_end"),
        "source": metadata.get("source"),
    }


def _build_few_shot_record(sample: dict[str, Any], scene: str, scene_desc: str) -> dict[str, Any]:
    metadata = sample.get("metadata") or {}
    turns = _extract_turns(sample.get("conversations", []))
    return {
        "sample_id": sample.get("id"),
        "scene_tag": scene,
        "scene_desc": scene_desc,
        "turns": turns,
        "source_file": metadata.get("source_file"),
        "source_line_start": metadata.get("source_line_start"),
        "source_line_end": metadata.get("source_line_end"),
        "source": metadata.get("source"),
        "conversations": sample.get("conversations", []),
    }


def _verify_no_validation_leak(
    train_records: list[dict[str, Any]],
    eval_samples: list[dict[str, Any]],
) -> list[str]:
    """Return list of leakage errors (empty = OK).

    Checks both sample_id overlap AND source_file+line overlap.
    The line-level check is critical because the same chapter can contain
    both training and validation fragments (V2.1 Critical #5).
    """
    errors: list[str] = []
    eval_ids = {s.get("id") for s in eval_samples}
    eval_sources = {
        (m.get("source_file"), m.get("source_line_start"), m.get("source_line_end"))
        for s in eval_samples
        for m in [s.get("metadata") or {}]
        if m.get("source_file") and m.get("source_line_start") is not None
    }
    for rec in train_records:
        if rec["sample_id"] in eval_ids:
            errors.append(f"sample_id overlap with validation: {rec['sample_id']}")
        source_key = (rec["source_file"], rec["source_line_start"], rec["source_line_end"])
        if source_key in eval_sources:
            errors.append(
                f"source line overlap with validation: {rec['sample_id']} "
                f"({rec['source_file']}:{rec['source_line_start']})"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Build few-shot pool from V3 train (715 samples)")
    parser.add_argument("--v3-train", type=Path, default=V3_TRAIN_PATH)
    parser.add_argument("--v3-eval", type=Path, default=V3_EVAL_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    train_samples = _read_json(args.v3_train)
    eval_samples = _read_json(args.v3_eval)
    if len(train_samples) != 715:
        print(json.dumps({"error": f"expected 715 train samples, got {len(train_samples)}"},
                         ensure_ascii=False))
        return 2
    if len(eval_samples) != 84:
        print(json.dumps({"error": f"expected 84 eval samples, got {len(eval_samples)}"},
                         ensure_ascii=False))
        return 2

    # 1. Build precise whitelist (715 entries)
    whitelist = [_build_whitelist_entry(s) for s in train_samples]

    # 2. Tag scenes and build few-shot records
    few_shot_records: list[dict[str, Any]] = []
    scene_counts: dict[str, int] = {}
    for sample in train_samples:
        scene, desc = _tag_scene(sample.get("conversations", []))
        scene_counts[scene] = scene_counts.get(scene, 0) + 1
        few_shot_records.append(_build_few_shot_record(sample, scene, desc))

    # 3. Verify no validation leakage (sample_id AND source line)
    leakage_errors = _verify_no_validation_leak(few_shot_records, eval_samples)

    # 4. Verify every record's source is in the 715 whitelist (trivially true here,
    #    but we re-check to catch future regressions if the pool is extended)
    whitelist_source_keys = {
        (w["source_file"], w["source_line_start"], w["source_line_end"])
        for w in whitelist
    }
    whitelist_violations = [
        rec["sample_id"] for rec in few_shot_records
        if (rec["source_file"], rec["source_line_start"], rec["source_line_end"])
        not in whitelist_source_keys
    ]

    # 5. Write outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train_source_whitelist.json").write_text(
        json.dumps({
            "whitelist_version": 1,
            "count": len(whitelist),
            "source_files": sorted({w["source_file"] for w in whitelist if w["source_file"]}),
            "entries": whitelist,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with (args.output_dir / "few_shot_pool.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for rec in few_shot_records:
            handle.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")

    # 6. Summary
    ok = not leakage_errors and not whitelist_violations
    print(json.dumps({
        "built": ok,
        "whitelist_path": str(args.output_dir / "train_source_whitelist.json"),
        "few_shot_pool_path": str(args.output_dir / "few_shot_pool.jsonl"),
        "whitelist_count": len(whitelist),
        "few_shot_count": len(few_shot_records),
        "scene_distribution": scene_counts,
        "source_files_count": len({w["source_file"] for w in whitelist if w["source_file"]}),
        "validation_leakage_errors": leakage_errors[:5],
        "validation_leakage_error_count": len(leakage_errors),
        "whitelist_violations": whitelist_violations[:5],
    }, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
