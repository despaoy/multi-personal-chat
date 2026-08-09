# -*- coding: utf-8 -*-
"""游戏数据修复 v4 (硬问题 2/3/4)。

修复项:
1. riou 四条问答: 0004 user 补语境 / 0008 道谢句 / 0009 全程第三人称 / 0012 首轮语义错配
2. riou split_group 统一: 0001+0002->3蓝宝石_甜点_166_180, 0004+0005->5磷灰石_晚餐_497_510,
   0011+0012->8萤石时空_理央深层_1310_1330
3. 状态分类(module 与 state 分离): riou_0009-0012 与 rikata 全部 state=paper + 纸上存在专用 system
4. rikata 死亡观/死局/自我否定(0001/0002/0004)移入 archive, 避免 8B LoRA 过度学习纸上状态
"""
import json
import os
from pathlib import Path

BASE = Path(r"c:\Users\13474\Desktop\qqchat-enhanced\backend\data\character_dialogues\experiments")
RIOU = BASE / "game_riou_candidates.jsonl"
RIKATA = BASE / "game_rikata_candidates.jsonl"
PAPER = BASE / "game_famous_paper_state.jsonl"
FAMOUS = BASE / "game_famous_candidates.jsonl"
ARCHIVE_RK = BASE / "archive" / "game_rikata_deathview_archive.jsonl"

SYS_GENERAL = "你正在扮演月社妃。请依据给定角色设定和原作中的语言习惯，自然地回应。"
SYS_PAPER = "你正在扮演由魔法之书创造、拥有月社妃记忆与人格的纸上存在。她清楚原本的月社妃已经死亡，自己只是与其相似的存在。请依据这一时期的角色状态自然回应。"


def load(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def dump(path, recs):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n", encoding="utf-8")
    os.replace(tmp, path)  # Windows 下 replace 可覆盖已有文件


def set_system(rec, system):
    rec["messages"][0]["content"] = system
    rec["metadata"]["system_state"] = system


# ── riou ────────────────────────────────────────────────
riou = load(RIOU)
by_id = {r["id"]: r for r in riou}

# 0004: user 补语境
m = by_id["kisaki_game_riou_0004"]["messages"]
assert m[1]["role"] == "user" and m[1]["content"] == "夜子怎么办"
m[1]["content"] = "如果晚餐做日式料理，夜子怎么办？"

# 0008: 道谢句
m = by_id["kisaki_game_riou_0008"]["messages"]
m[3]["content"] = "……可我还是向理央道谢了。在琉璃面前，连逞强也做不到呢。"

# 0009: 全程第三人称
m = by_id["kisaki_game_riou_0009"]["messages"]
m[1]["content"] = "首先是理央。能请她别再畏怯了吗？"

# 0012: 首轮语义错配修正
m = by_id["kisaki_game_riou_0012"]["messages"]
m[1]["content"] = "你现在也喜欢琉璃吗？"
m[3]["content"] = "那你怎么看同样喜欢琉璃的理央？"

# split_group 统一 + 状态
SG_FIX = {
    "kisaki_game_riou_0001": "3蓝宝石_甜点_166_180",
    "kisaki_game_riou_0002": "3蓝宝石_甜点_166_180",
    "kisaki_game_riou_0004": "5磷灰石_晚餐_497_510",
    "kisaki_game_riou_0005": "5磷灰石_晚餐_497_510",
    "kisaki_game_riou_0011": "8萤石时空_理央深层_1310_1330",
    "kisaki_game_riou_0012": "8萤石时空_理央深层_1310_1330",
}
PAPER_RIOU = {"kisaki_game_riou_0009", "kisaki_game_riou_0010", "kisaki_game_riou_0011", "kisaki_game_riou_0012"}

for r in riou:
    rid = r["id"]
    if rid in SG_FIX:
        r["metadata"]["split_group"] = SG_FIX[rid]
    r["metadata"]["state"] = "paper" if rid in PAPER_RIOU else "general"
    if rid in PAPER_RIOU:
        set_system(r, SYS_PAPER)
    r["metadata"]["version"] = "v4_game_riou_clean_v4"

# ── rikata: 0001/0002/0004 移入 archive, 其余 state=paper ──
rikata = load(RIKATA)
ARCHIVE_IDS = {"kisaki_game_rikata_0001", "kisaki_game_rikata_0002", "kisaki_game_rikata_0004"}
arch_rk = [r for r in rikata if r["id"] in ARCHIVE_IDS]
keep_rk = [r for r in rikata if r["id"] not in ARCHIVE_IDS]
for r in keep_rk:
    r["metadata"]["state"] = "paper"
    set_system(r, SYS_PAPER)
    r["metadata"]["version"] = "v4_game_rikata_clean_v4"
for r in arch_rk:
    r["metadata"]["state"] = "paper"
    set_system(r, SYS_PAPER)
    r["metadata"]["note"] = (r["metadata"].get("note", "") + " [归档] 死亡观/死局/自我否定, 避免过度学习纸上状态").strip()

# ── paper: 补 state ──
paper = load(PAPER)
for r in paper:
    r["metadata"]["state"] = "paper"

# ── famous 通用: 补 state ──
famous = load(FAMOUS)
for r in famous:
    r["metadata"]["state"] = "general"

# 写回
dump(RIOU, riou)
dump(RIKATA, keep_rk)
dump(PAPER, paper)
dump(FAMOUS, famous)
dump(ARCHIVE_RK, arch_rk)

print(f"riou: {len(riou)} 条 (0009-0012 state=paper)")
print(f"rikata: 保留 {len(keep_rk)} 条, 移入 archive {len(arch_rk)} 条")
print(f"paper: {len(paper)} 条 state=paper")
print(f"famous: {len(famous)} 条 state=general")
print(f"archive: {ARCHIVE_RK.name}")
# 校验
for r in riou:
    if r["id"] in PAPER_RIOU:
        assert r["metadata"]["state"] == "paper"
        assert SYS_PAPER in r["messages"][0]["content"]
print("校验通过")
