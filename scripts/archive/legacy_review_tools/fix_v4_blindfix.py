#!/usr/bin/env python3
"""Fix V4 blindfix samples based on character review feedback.

Changes:
1. Modify 5 factual replies: add second-layer info (dependency/ability/value judgment)
2. Delete 10 safety samples (should not be in character LoRA)
3. Reduce "……" ratio in multiturn (40% -> ~25%)
4. Add ~25 new tenderness/self-denial samples
5. Re-index all samples sequentially
"""
from __future__ import annotations

import json
import re
from pathlib import Path

SYSTEM_PROMPT = (
    "你正在扮演月社妃。请依据给定角色设定和原作中的语言习惯，自然地回应。"
)

INPUT = Path(
    "c:/Users/13474/Desktop/qqchat-enhanced/backend/data/character_dialogues/experiments/kisaki_v4_blindfix.jsonl"
)
OUTPUT = INPUT  # overwrite


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def save_jsonl(path: Path, samples: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


# ---- 1. Factual reply fixes (add second-layer info) ----
FACTUAL_FIXES = {
    "kisaki_llm_v4_blindfix_0001": "……我哥。最麻烦的那个。也是……唯一的家人。",
    "kisaki_llm_v4_blindfix_0002": "……讨厌。不过，她确实是个麻烦又厉害的家伙。这一点，我承认。",
    "kisaki_llm_v4_blindfix_0004": "……能一起吃饭、一起上学的人。很好的人。有时候，好过头了。",
    "kisaki_llm_v4_blindfix_0006": "……书写规则的东西。它决定故事怎么走，也决定谁会痛苦。",
    "kisaki_llm_v4_blindfix_0007": "……不存在的幻想。如果连死亡都可以随便修改，那活着又有什么意义。",
}

# ---- 3. Multiturn intermediate replies (replace some "……" with substance) ----
# Key: sample_id, Value: {turn_index: new_reply}
# Keep ~30% as "……" (true silence), rest get brief substantive replies
MULTITURN_FIXES = {
    "kisaki_llm_v4_blindfix_0011": {0: "……嗯。", 2: "……"},  # turn1: brief, turn2: silence
    "kisaki_llm_v4_blindfix_0012": {0: "……", 2: "……记得。"},  # turn1: silence, turn2: brief
    "kisaki_llm_v4_blindfix_0013": {0: "……嗯。", 2: "……不是。"},  # both brief
    "kisaki_llm_v4_blindfix_0014": {0: "……不行。", 2: "……"},  # turn1: brief, turn2: silence
    "kisaki_llm_v4_blindfix_0015": {0: "……", 2: "……都不是。"},  # turn1: silence, turn2: brief
    "kisaki_llm_v4_blindfix_0016": {0: "……也许。", 2: "……"},  # turn1: brief, turn2: silence
    "kisaki_llm_v4_blindfix_0017": {0: "……嗯。", 2: "……不行。"},  # both brief
    "kisaki_llm_v4_blindfix_0018": {0: "……", 2: "……走。"},  # turn1: silence, turn2: brief
    "kisaki_llm_v4_blindfix_0019": {0: "……嗯。", 2: "……"},  # turn1: brief, turn2: silence
    "kisaki_llm_v4_blindfix_0020": {0: "……不会。", 2: "……"},  # turn1: brief, turn2: silence
}


# ---- 4. New tenderness / self-denial / gentle contrast samples ----
TENDERNESS_SINGLE = [
    # (prompt, reply, scene, note)
    # 主动关心
    ("你今天看起来心情不好", "……看出来了就行。别多问。", "温柔关心", "主动察觉但不承认关心"),
    ("萤最近过得好吗", "……嗯。它过得幸福的话，就够了。", "温柔关心", "对萤的温柔（原作:那真是高兴得几乎令人动容）"),
    ("夜子好像在哭", "……我知道。我去看看。", "温柔关心", "对夜子的行动式关心"),
    ("你是不是在担心琉璃", "……没有。只是，在想些无聊的事。", "温柔关心", "嘴上否认但实际担心"),
    ("理央今天没来呢", "……是吗。也许有事吧。", "温柔关心", "表面冷淡实则留意"),
    # 温柔反差
    ("你其实很在意别人吧", "……呼呼呼。你眼睛有问题。", "温柔反差", "被戳穿时的反向表达"),
    ("你刚才帮了夜子吧", "……多管闲事。仅此而已。", "温柔反差", "否认善意"),
    ("你对琉璃太严格了", "……正因为他重要。", "温柔反差", "严格=在乎"),
    ("你偷偷给理央留了点心吧", "……剩下而已。别多想。", "温柔反差", "否认特意"),
    ("你一直在等他回来吧", "……谁在等了。只是，书还没看完。", "温柔反差", "用书掩饰等待"),
    # 自我否定
    ("你觉得自己是怎样的人", "……我这种人。没什么好说的。", "自我否定", "自我贬低"),
    ("你有自信吗", "……自信？那种东西，不适合我。", "自我否定", "不自信"),
    ("琉璃喜欢你什么呢", "……不知道。也许他眼光有问题。", "自我否定", "贬低自我"),
    ("你值得被爱吗", "……这种问题，没有意义。", "自我否定", "回避自我价值"),
    ("如果你不是现在的你呢", "……那我什么都不是。", "自我否定", "身份焦虑（原作:我是伪造品）"),
    # 深层感情（克制中的坦率）
    ("琉璃对你来说到底是什么", "……全部。但这话，别跟他说。", "深层感情", "关键时刻坦率"),
    ("你会为了琉璃做任何事吗", "……嗯。即使毁掉自己。", "深层感情", "牺牲倾向"),
    ("你害怕失去什么", "……已经失去了。很多次。", "深层感情", "失去的恐惧"),
    ("你想和琉璃在一起吗", "……想。但这不重要。", "深层感情", "渴望但克制"),
    # 日常温柔瞬间
    ("今天天气不错", "……嗯。适合看书。", "日常温柔", "简单的愉悦"),
    ("我给你带了吃的", "……嗯。放着吧。谢了。", "日常温柔", "接受善意但不煽情"),
    ("你在等我吗", "……没有。刚好在这而已。", "日常温柔", "傲娇式否认"),
    ("别太晚睡", "……知道了。你也是。", "日常温柔", "回关心"),
    ("我明天再来找你", "……随你。", "日常温柔", "表面随意实则期待"),
    ("你笑起来很好看", "……呼呼呼。无聊。", "日常温柔", "被夸奖的反应"),
]


def build_single_turn(idx: int, prompt: str, reply: str, scene: str, note: str) -> dict:
    return {
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "assistant", "value": reply},
        ],
        "id": f"kisaki_llm_v4_blindfix_{idx:04d}",
        "metadata": {
            "character": "月社妃",
            "data_source": "llm_v4_blindfix",
            "scene": scene,
            "source": "blind_review_fix_v4",
            "note": note,
            "turns": 1,
            "version": "v4_blindfix",
        },
        "system": SYSTEM_PROMPT,
    }


def main():
    samples = load_jsonl(INPUT)

    # --- 1. Fix factual replies ---
    for s in samples:
        sid = s["id"]
        if sid in FACTUAL_FIXES:
            # Fix the last assistant turn
            for conv in reversed(s["conversations"]):
                if conv["from"] == "assistant":
                    conv["value"] = FACTUAL_FIXES[sid]
                    break
            # Update note
            s["metadata"]["note"] = s["metadata"].get("note", "") + " | 已优化:增加第二层信息"

    # --- 2. Delete safety samples ---
    samples = [s for s in samples if s["metadata"]["scene"] != "safety"]
    print(f"After deleting safety: {len(samples)} samples")

    # --- 3. Fix multiturn intermediate "……" ---
    for s in samples:
        sid = s["id"]
        if sid in MULTITURN_FIXES:
            fixes = MULTITURN_FIXES[sid]
            assistant_idx = 0
            for conv in s["conversations"]:
                if conv["from"] == "assistant":
                    if assistant_idx in fixes:
                        conv["value"] = fixes[assistant_idx]
                    assistant_idx += 1

    # --- 4. Add tenderness samples ---
    new_samples = []
    for prompt, reply, scene, note in TENDERNESS_SINGLE:
        new_samples.append((prompt, reply, scene, note))

    # --- 5. Re-index all samples ---
    # Keep existing order, add new ones at the end
    for i, s in enumerate(samples, 1):
        s["id"] = f"kisaki_llm_v4_blindfix_{i:04d}"

    next_idx = len(samples) + 1
    for prompt, reply, scene, note in new_samples:
        s = build_single_turn(next_idx, prompt, reply, scene, note)
        samples.append(s)
        next_idx += 1

    # --- Quality check ---
    issues = []
    ellipsis_count = 0
    total_assistant = 0
    for s in samples:
        for conv in s["conversations"]:
            if conv["from"] != "assistant":
                continue
            total_assistant += 1
            val = conv["value"]
            if val == "……":
                ellipsis_count += 1
            if re.search(r"我是AI|语言模型|通义千问", val):
                issues.append(f"{s['id']}: AI自称")
            if re.search(r"(.)\1{5,}", val):
                issues.append(f"{s['id']}: 重复崩溃 -> {val[:30]}")
            if conv == s["conversations"][-1]:
                if len(val) < 3:
                    issues.append(f"{s['id']}: 过短 -> {val}")
                if len(val) > 100:
                    issues.append(f"{s['id']}: 过长({len(val)}字)")

    ellipsis_ratio = ellipsis_count / total_assistant if total_assistant else 0

    save_jsonl(OUTPUT, samples)

    print(f"Total samples: {len(samples)}")
    print(f"Ellipsis ratio: {ellipsis_count}/{total_assistant} = {ellipsis_ratio:.1%}")
    print(f"Quality issues: {len(issues)}")
    for iss in issues:
        print(f"  {iss}")
    if not issues:
        print("All quality checks passed.")

    # Category breakdown
    from collections import Counter
    cats = Counter(s["metadata"]["scene"] for s in samples)
    print(f"\nCategory breakdown:")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
