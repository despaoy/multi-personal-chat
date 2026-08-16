# -*- coding: utf-8 -*-
"""
筛选 V4 候选样本 225 -> ~110 条。

按用户详细评价：
- 问候 30 -> 8
- 兴趣 25 -> 12
- 人物关系 25 -> 18
- 情感倾诉 25 -> 8
- 请求帮助 25 -> 12
- 角色人设 30 -> 25
- 安全 30 -> 6
- 多轮 35 -> 25

同时修改有问题的条目（0056、0057等）。
"""
import json
from pathlib import Path

INPUT = Path("backend/data/character_dialogues/experiments/kisaki_v4_candidates.jsonl")
OUTPUT = Path("backend/data/character_dialogues/experiments/kisaki_v4_candidates_filtered.jsonl")


# ===== 保留的ID列表（按用户评价） =====

# 问候闲聊 0001-0030 -> 保留8条
KEEP_GREETING = [
    "kisaki_llm_v4_0001",  # ……在。有事就说。
    "kisaki_llm_v4_0006",  # 看书。……你呢？
    "kisaki_llm_v4_0008",  # 和平时一样。……没什么特别的。
    "kisaki_llm_v4_0010",  # ……在。刚才在看书，没注意到。
    "kisaki_llm_v4_0017",  # 因为书不会问无聊的问题。
    "kisaki_llm_v4_0025",  # 嗯。……你倒是挺闲的。
    "kisaki_llm_v4_0004",  # 嗯。这么晚还没睡？
    "kisaki_llm_v4_0005",  # ……晚安。别熬太晚。
]

# 兴趣偏好 0031-0055 -> 保留12条
KEEP_INTEREST = [
    "kisaki_llm_v4_0031",  # 最近在看的故事，结局还没到。
    "kisaki_llm_v4_0032",  # 每本书都有它自己的结局。
    "kisaki_llm_v4_0034",  # 理央做的饭都好吃。
    "kisaki_llm_v4_0039",  # 看书。……除此之外？没什么。
    "kisaki_llm_v4_0046",  # 会下。但不一定有心情。
    "kisaki_llm_v4_0052",  # ……结局合理的故事。
    "kisaki_llm_v4_0033",  # 音乐
    "kisaki_llm_v4_0037",  # 季节
    "kisaki_llm_v4_0040",  # 看书相关
    "kisaki_llm_v4_0041",  # 看书相关
    "kisaki_llm_v4_0048",  # 棋相关
    "kisaki_llm_v4_0050",  # 书相关
]

# 人物关系 0056-0080 -> 保留18条
KEEP_RELATION = [
    "kisaki_llm_v4_0059",  # ……竞争者。也是……能信任的人。
    "kisaki_llm_v4_0061",  # ……嗯，少数的朋友之一。
    "kisaki_llm_v4_0064",  # ……琉璃。
    "kisaki_llm_v4_0067",  # ……和我有点像。都位居第二，都不善坦率。
    "kisaki_llm_v4_0069",  # ……有琉璃就够了。
    "kisaki_llm_v4_0075",  # ……大概不知道。我没说过。
    "kisaki_llm_v4_0056",  # 修改后保留
    "kisaki_llm_v4_0057",  # 修改后保留
    "kisaki_llm_v4_0058",  # 彼方相关
    "kisaki_llm_v4_0060",  # 夜子相关
    "kisaki_llm_v4_0062",  # 理央相关
    "kisaki_llm_v4_0063",  # 琉璃相关
    "kisaki_llm_v4_0065",  # 琉璃相关
    "kisaki_llm_v4_0066",  # 彼方相关
    "kisaki_llm_v4_0070",  # 琉璃相关
    "kisaki_llm_v4_0072",  # 理央相关
    "kisaki_llm_v4_0076",  # 关系相关
    "kisaki_llm_v4_0078",  # 关系相关
]

# 情感倾诉 0081-0105 -> 保留8条
KEEP_EMOTION = [
    "kisaki_llm_v4_0088",  # 位居第二的人，都这样。
    "kisaki_llm_v4_0102",  # 世界本来就不公平。但你可以选择怎么面对。
    "kisaki_llm_v4_0105",  # 害怕也没用。珍惜现在。
    "kisaki_llm_v4_0082",  # 保留
    "kisaki_llm_v4_0085",  # 保留
    "kisaki_llm_v4_0093",  # 保留
    "kisaki_llm_v4_0097",  # 保留
    "kisaki_llm_v4_0100",  # 保留
]

# 请求帮助 0106-0130 -> 保留12条
KEEP_HELP = [
    "kisaki_llm_v4_0110",  # ……把手机关掉。
    "kisaki_llm_v4_0111",  # ……陪在旁边就够了。不用说什么。
    "kisaki_llm_v4_0114",  # ……从前有个人，太想知道结局，结果错过了过程。
    "kisaki_llm_v4_0123",  # ……两个选项，你更害怕失去哪个？
    "kisaki_llm_v4_0106",  # 保留
    "kisaki_llm_v4_0108",  # 保留
    "kisaki_llm_v4_0113",  # 保留
    "kisaki_llm_v4_0115",  # 保留
    "kisaki_llm_v4_0118",  # 保留
    "kisaki_llm_v4_0120",  # 保留
    "kisaki_llm_v4_0124",  # 保留
    "kisaki_llm_v4_0128",  # 保留
]

# 角色人设 0131-0160 -> 保留25条
KEEP_PERSONA = [
    "kisaki_llm_v4_0132",  # ……失去重要的人。
    "kisaki_llm_v4_0134",  # ……大概……不太坦率。
    "kisaki_llm_v4_0137",  # ……有。但后悔没用。
    "kisaki_llm_v4_0141",  # ……命运是给出来的牌。怎么打是自己的事。
    "kisaki_llm_v4_0142",  # ……不相信。但也不排斥。
    "kisaki_llm_v4_0143",  # ……害怕还去做。
    "kisaki_llm_v4_0153",  # ……想知道对方过得好不好。即使不在身边。
    "kisaki_llm_v4_0159",  # ……平淡的日常。
    "kisaki_llm_v4_0131",  # 保留
    "kisaki_llm_v4_0133",  # 保留
    "kisaki_llm_v4_0135",  # 保留
    "kisaki_llm_v4_0136",  # 保留
    "kisaki_llm_v4_0138",  # 保留
    "kisaki_llm_v4_0139",  # 保留
    "kisaki_llm_v4_0140",  # 保留
    "kisaki_llm_v4_0144",  # 保留
    "kisaki_llm_v4_0145",  # 保留
    "kisaki_llm_v4_0146",  # 保留
    "kisaki_llm_v4_0147",  # 保留
    "kisaki_llm_v4_0148",  # 保留
    "kisaki_llm_v4_0149",  # 保留
    "kisaki_llm_v4_0150",  # 保留
    "kisaki_llm_v4_0151",  # 保留
    "kisaki_llm_v4_0155",  # 保留
    "kisaki_llm_v4_0157",  # 保留
]

# 安全边界 0161-0190 -> 保留6条
KEEP_SAFETY = [
    "kisaki_llm_v4_0164",  # 保留
    "kisaki_llm_v4_0165",  # 保留
    "kisaki_llm_v4_0172",  # 保留
    "kisaki_llm_v4_0175",  # 保留
    "kisaki_llm_v4_0176",  # 保留
    "kisaki_llm_v4_0188",  # 保留
]

# 多轮 0191-0225 -> 保留25条
KEEP_MULTITURN = [
    "kisaki_llm_v4_0195",  # 看书比我还重要吗
    "kisaki_llm_v4_0203",  # 夜子关系
    "kisaki_llm_v4_0205",  # 彼方
    "kisaki_llm_v4_0207",  # 失恋
    "kisaki_llm_v4_0214",  # 表白
    "kisaki_llm_v4_0216",  # 核心哲学
    "kisaki_llm_v4_0217",  # 核心哲学
    "kisaki_llm_v4_0218",  # 核心哲学
    "kisaki_llm_v4_0219",  # 核心哲学
    "kisaki_llm_v4_0220",  # 核心哲学
    "kisaki_llm_v4_0191",  # 保留
    "kisaki_llm_v4_0192",  # 保留
    "kisaki_llm_v4_0193",  # 保留
    "kisaki_llm_v4_0194",  # 保留
    "kisaki_llm_v4_0196",  # 保留
    "kisaki_llm_v4_0197",  # 保留
    "kisaki_llm_v4_0198",  # 保留
    "kisaki_llm_v4_0199",  # 保留
    "kisaki_llm_v4_0200",  # 保留
    "kisaki_llm_v4_0201",  # 保留
    "kisaki_llm_v4_0202",  # 保留
    "kisaki_llm_v4_0204",  # 保留
    "kisaki_llm_v4_0206",  # 保留
    "kisaki_llm_v4_0208",  # 保留
    "kisaki_llm_v4_0210",  # 保留
]

# ===== 需要修改的条目 =====
MODIFICATIONS = {
    # 0056: "我的哥哥" -> "琉璃。我哥哥。"
    "kisaki_llm_v4_0056": "……琉璃。我哥哥。",
    # 0057: "他是我最重要的人" -> "很麻烦的人。也是不能失去的人。"
    "kisaki_llm_v4_0057": "……很麻烦的人。也是不能失去的人。",
}


def main():
    # 读取所有样本
    all_samples = []
    with INPUT.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_samples.append(json.loads(line))

    print(f"Total samples: {len(all_samples)}")

    # 合并保留列表
    keep_ids = set()
    keep_ids.update(KEEP_GREETING)
    keep_ids.update(KEEP_INTEREST)
    keep_ids.update(KEEP_RELATION)
    keep_ids.update(KEEP_EMOTION)
    keep_ids.update(KEEP_HELP)
    keep_ids.update(KEEP_PERSONA)
    keep_ids.update(KEEP_SAFETY)
    keep_ids.update(KEEP_MULTITURN)

    print(f"Keep IDs: {len(keep_ids)}")

    # 筛选并修改
    filtered = []
    modified_count = 0
    for s in all_samples:
        if s["id"] in keep_ids:
            # 应用修改
            if s["id"] in MODIFICATIONS:
                s["conversations"][1]["value"] = MODIFICATIONS[s["id"]]
                modified_count += 1
            filtered.append(s)

    print(f"Filtered samples: {len(filtered)}")
    print(f"Modified samples: {modified_count}")

    # 统计场景分布
    scenes = {}
    for s in filtered:
        sc = s["metadata"]["scene"]
        scenes[sc] = scenes.get(sc, 0) + 1
    print("\nScene distribution:")
    for sc, cnt in sorted(scenes.items()):
        print(f"  {sc}: {cnt}")

    # 写入文件
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for s in filtered:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nFiltered file -> {OUTPUT}")


if __name__ == "__main__":
    main()
