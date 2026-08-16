# -*- coding: utf-8 -*-
"""修复 riou_0008 / riou_0009 的问答位置错位。"""
import json
import os
from pathlib import Path

RIOU = Path(r"c:\Users\13474\Desktop\qqchat-enhanced\backend\data\character_dialogues\experiments\game_riou_candidates.jsonl")
recs = [json.loads(l) for l in RIOU.read_text(encoding="utf-8").splitlines() if l.strip()]
by_id = {r["id"]: r for r in recs}

# riou_0008: 2 回合 => [sys, user, asst, user, asst]
r8 = by_id["kisaki_game_riou_0008"]
assert r8["messages"][3]["role"] == "user" and "道谢" in r8["messages"][3]["content"], r8["messages"][3]
r8["messages"][3]["content"] = "真的吗"
assert r8["messages"][4]["role"] == "assistant"
r8["messages"][4]["content"] = "……可我还是向理央道谢了。在琉璃面前，连逞强也做不到呢。"

# riou_0009: 3 回合 => [sys, user, asst, user, asst, user, asst]
r9 = by_id["kisaki_game_riou_0009"]
assert r9["messages"][1]["role"] == "user" and "畏怯" in r9["messages"][1]["content"], r9["messages"][1]
r9["messages"][1]["content"] = "理央在畏畏缩缩的"
assert r9["messages"][2]["role"] == "assistant"
r9["messages"][2]["content"] = "首先是理央。能请她别再畏怯了吗？"

tmp = RIOU.with_suffix(".tmp")
tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n", encoding="utf-8")
os.replace(tmp, RIOU)

for rid in ["kisaki_game_riou_0008", "kisaki_game_riou_0009"]:
    r = by_id[rid]
    print(f"=== {rid} ===")
    for m in r["messages"]:
        print(f"  [{m['role']}] {m['content']}")
print("修复完成")
