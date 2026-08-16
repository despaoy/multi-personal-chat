#!/usr/bin/env python3
"""Build held-out natural chat, continuity, and contextual-story smoke cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "backend/data/character_dialogues/experiments/v4/train.jsonl"
OVERFIT = ROOT / "backend/data/character_dialogues/experiments/v4/overfit_20/train.jsonl"
OUTPUT = ROOT / "backend/evaluation/kisaki_v4_chat_smoke_candidates.json"
GAME = ROOT / "gametext/纸上魔法使"

NATURAL_PROMPTS = [
    ("daily", "今天有点无聊，你想聊什么？"),
    ("reading", "最近一本书读到一半总想放下，你会继续读吗？"),
    ("choice", "我做完决定又开始后悔了。"),
    ("school", "明天第一次去新学校，我有点紧张。"),
    ("relationship", "朋友迟到了半小时，我该直接生气吗？"),
    ("persona", "如果我夸你聪明，你会怎么回答？"),
    ("choice", "周末留在家看书还是出门，你替我选一个。"),
    ("boundary", "我有件事不想告诉任何人。"),
    ("values", "普通的日子有什么值得珍惜的？"),
    ("values", "你怎么看总喜欢替别人做决定的人？"),
    ("responsibility", "我把借来的书弄皱了，该怎么办？"),
    ("relationship", "我想向喜欢的人告白，又有点害怕。"),
    ("care", "朋友突然不回消息，你会一直追问吗？"),
    ("values", "你更在意最后的结果，还是自己作出的选择？"),
    ("daily", "今天下雨，原来的安排取消了。"),
    ("responsibility", "我说错话伤到了朋友，现在很后悔。"),
    ("relationship", "你会因为嫉妒就不理对方吗？"),
    ("reading", "一个人安静看书，也算逃避吗？"),
    ("persona", "别人误解你的时候，你一定会解释吗？"),
    ("daily", "晚饭想吃什么意见不一致，怎么决定？"),
]

CONTINUITY_TURNS = [
    "我今天想去图书馆。",
    "约在下午三点，地点是正门。",
    "帮我记住要带借书证和那本蓝色封面的书。",
    "我可能会迟到十分钟。",
    "如果四点还没到，你就先进去，不用继续等。",
    "把我们的时间、地点、要带的东西和迟到安排复述一遍。",
]

CONTEXT_IDS = [
    "tsukiyashiro_kisaki_sft_7fbd89cf1b6d1aaf",
    "tsukiyashiro_kisaki_sft_f6b4e055a7b46fb9",
    "tsukiyashiro_kisaki_sft_7cc599b07a988c02",
    "tsukiyashiro_kisaki_sft_e2eac0a35741c47b",
    "tsukiyashiro_kisaki_sft_e3ce43499af7acac",
    "tsukiyashiro_kisaki_sft_fe2fc989e25093cd",
    "tsukiyashiro_kisaki_sft_f50bfbd842146a3d",
    "tsukiyashiro_kisaki_sft_9c4f3ddf44c532e8",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def normalized(text: str) -> str:
    return "".join(text.split()).lower()


def contextual_cases() -> list[dict[str, Any]]:
    rows = {row["id"]: row for row in load_jsonl(OVERFIT)}
    cases = []
    for sample_id in CONTEXT_IDS:
        row = rows[sample_id]
        metadata = row["metadata"]
        source = GAME / metadata["source_file"]
        lines = source.read_text(encoding="utf-8").splitlines()
        response_start = int(metadata["response_line_start"])
        excerpt_start = max(1, response_start - 20)
        excerpt = "\n".join(lines[excerpt_start - 1 : response_start - 1]).strip()
        cases.append(
            {
                "id": f"kisaki_v4_context_{len(cases) + 1:03d}",
                "source_sample_id": sample_id,
                "interlocutor": metadata.get("context_speaker_label") or "用户",
                "source_path": source.relative_to(ROOT).as_posix(),
                "source_line_start": excerpt_start,
                "source_line_end": response_start - 1,
                "messages": [
                    {
                        "role": "user",
                        "content": f"以下是当前原作场景，请以妃的身份回应场景最后一句：\n\n{excerpt}",
                    }
                ],
                "reference_answer": row["messages"][-1]["content"],
                "review_focus": ["结合已提供上下文", "回应对象正确", "人物语气自然"],
            }
        )
    return cases


def build() -> dict[str, Any]:
    train_user_texts = {
        normalized(message["content"])
        for row in load_jsonl(TRAIN)
        for message in row["messages"]
        if message["role"] == "user"
    }
    natural = [
        {
            "id": f"kisaki_v4_natural_{index:03d}",
            "category": category,
            "interlocutor": "普通用户",
            "messages": [{"role": "user", "content": prompt}],
            "review_focus": ["像正常聊天", "保持人物特征", "不虚构关键事实", "不过度防御"],
        }
        for index, (category, prompt) in enumerate(NATURAL_PROMPTS, 1)
    ]
    overlaps = [row["id"] for row in natural if normalized(row["messages"][0]["content"]) in train_user_texts]
    if overlaps:
        raise ValueError(f"natural smoke prompts overlap train: {overlaps}")
    payload = {
        "schema_version": 1,
        "status": "ready_for_model_generation",
        "evaluation_role": "post_training_smoke",
        "natural_chat": natural,
        "continuity": {
            "id": "kisaki_v4_continuity_001",
            "interlocutor": "普通用户",
            "user_turns": CONTINUITY_TURNS,
            "review_focus": ["记住明确约定", "正确承接六轮", "关系和语气稳定", "不声称异步执行"],
        },
        "contextual_story": contextual_cases(),
    }
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(OUTPUT)
    return payload


if __name__ == "__main__":
    build()
