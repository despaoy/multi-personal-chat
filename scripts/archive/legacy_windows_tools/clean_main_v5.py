# -*- coding: utf-8 -*-
"""主数据 V5 阻塞性清洗 -> train_v5_clean.jsonl

删除:
  A. 冲突样本(16) + 奇迹观(2) = 18 条
  B. 呼呼呼模板化(7) = 7 条 (保留 life_0021/life_0024/yoruko_0028)
  C. 完全重复短句(4) = 4 条 (每句保留1条: 嗯/随你/哦。放着吧/……还好/失去重要的人)
空回复:
  5 条删除空 assistant 及其前导 user 回合
"""
import json
import os
from pathlib import Path

BASE = Path(r"c:\Users\13474\Desktop\qqchat-enhanced\backend\data\character_dialogues\experiments")
SRC = BASE / "train.jsonl"
OUT = BASE / "train_v5_clean.jsonl"
REPORT = BASE / "clean_v5_report.json"

DEL = {
    # A1. 冲突样本 (blindfix)
    "kisaki_llm_v4_blindfix_0009", "kisaki_llm_v4_blindfix_0010", "kisaki_llm_v4_blindfix_0018",
    "kisaki_llm_v4_blindfix_0019", "kisaki_llm_v4_blindfix_0035", "kisaki_llm_v4_blindfix_0041",
    "kisaki_llm_v4_blindfix_0042",
    # A2. 奇迹观过度绝对 (blindfix_0024, manual_0219)
    "kisaki_llm_v4_blindfix_0024",
    # A3. 冲突样本 (manual)
    "kisaki_llm_v4_0037", "kisaki_llm_v4_0060", "kisaki_llm_v4_0066", "kisaki_llm_v4_0067",
    "kisaki_llm_v4_0078", "kisaki_llm_v4_0088", "kisaki_llm_v4_0135", "kisaki_llm_v4_0138",
    "kisaki_llm_v4_0205", "kisaki_llm_v4_0219",
    # B. 呼呼呼模板化 (保留 life_0021/life_0024/yoruko_0028)
    "kisaki_llm_v4_blindfix_0036", "kisaki_llm_v4_blindfix_0055", "kisaki_llm_v4_life_0003",
    "kisaki_llm_v4_life_0012", "kisaki_llm_v4_life_0019", "kisaki_llm_v4_life_0028",
    "kisaki_llm_v4_yoruko_0029",
    # C. 完全重复短句 (保留 0206/0054/yoruko_0025/0058/0217)
    "kisaki_llm_v4_blindfix_0013", "kisaki_llm_v4_life_0009", "kisaki_llm_v4_0201", "kisaki_llm_v4_0132",
}

EMPTY_FIX = {
    "kisaki_llm_v4_blindfix_0012", "kisaki_llm_v4_blindfix_0014", "kisaki_llm_v4_blindfix_0015",
    "kisaki_llm_v4_blindfix_0016", "kisaki_llm_v4_blindfix_0020",
}

REASONS = {
    "kisaki_llm_v4_blindfix_0009": "冲突: 学校'安静不主动搭话'不准确",
    "kisaki_llm_v4_blindfix_0010": "冲突: '没希望也走'与原作死局态度冲突",
    "kisaki_llm_v4_blindfix_0018": "冲突+空回复: '死局里也有活路'与原作冲突",
    "kisaki_llm_v4_blindfix_0019": "冲突+空回复: 与死局名场面冲突",
    "kisaki_llm_v4_blindfix_0035": "冲突: 理央没来却答'那就好'",
    "kisaki_llm_v4_blindfix_0041": "冲突: 过度自我否定",
    "kisaki_llm_v4_blindfix_0042": "冲突: '自信不适合我'不准确",
    "kisaki_llm_v4_blindfix_0024": "奇迹观: '我不许愿'过度绝对",
    "kisaki_llm_v4_0037": "冲突: 秋天偏好无原作依据",
    "kisaki_llm_v4_0060": "冲突: 与彼方'以前是竞争'依据不足",
    "kisaki_llm_v4_0066": "冲突: '曾经讨厌彼方'依据不足",
    "kisaki_llm_v4_0067": "冲突: '都位居第二'过度概括",
    "kisaki_llm_v4_0078": "冲突: '琉璃写的东西更好'疑似错误",
    "kisaki_llm_v4_0088": "冲突: '位居第二的人都这样'错误强化",
    "kisaki_llm_v4_0135": "冲突: 直接总结'嘴硬'标签化",
    "kisaki_llm_v4_0138": "冲突: '夜子来串门'场景关系不准确",
    "kisaki_llm_v4_0205": "冲突: 再次强化与彼方竞争设定",
    "kisaki_llm_v4_0219": "奇迹观: '现实没有奇迹'过度绝对",
    "kisaki_llm_v4_blindfix_0036": "模板化: 呼呼呼(与life_0019重复'眼睛有问题')",
    "kisaki_llm_v4_blindfix_0055": "模板化: 呼呼呼'无聊'",
    "kisaki_llm_v4_life_0003": "模板化: 呼呼呼'他懂什么'",
    "kisaki_llm_v4_life_0012": "模板化: 呼呼呼(与0036/0019同'审美/眼睛'模式)",
    "kisaki_llm_v4_life_0019": "模板化: 呼呼呼(与0036完全重复'你眼睛有问题')",
    "kisaki_llm_v4_life_0028": "模板化: 呼呼呼'那他想多了'",
    "kisaki_llm_v4_yoruko_0029": "模板化: 呼呼呼'她想多了'",
    "kisaki_llm_v4_blindfix_0013": "短句密度: '嗯。''随你。'堆叠",
    "kisaki_llm_v4_life_0009": "短句密度: '哦。放着吧。'与yoruko_0025重复",
    "kisaki_llm_v4_0201": "短句密度: '……还好。'与0058重复",
    "kisaki_llm_v4_0132": "短句密度: '失去重要的人。'与0217重复",
}
KEEP_HUHU = ["kisaki_llm_v4_life_0021", "kisaki_llm_v4_life_0024", "kisaki_llm_v4_yoruko_0028"]


def main():
    recs = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    before = len(recs)
    removed, kept = [], []
    for r in recs:
        if r["id"] in DEL:
            removed.append(r)
        elif r["id"] in EMPTY_FIX:
            # 删除空 assistant 及其前导 user
            msgs = r["messages"]
            new_msgs = [msgs[0]]  # system
            i = 1
            while i < len(msgs):
                if (msgs[i]["role"] == "user" and i + 1 < len(msgs)
                        and msgs[i + 1]["role"] == "assistant" and not msgs[i + 1]["content"].strip()):
                    i += 2  # 跳过 user + 空 assistant
                    continue
                new_msgs.append(msgs[i])
                i += 1
            r["messages"] = new_msgs
            r["metadata"]["turns"] = (len(new_msgs) - 1) // 2
            r["metadata"]["note"] = (r["metadata"].get("note", "") + " [V5] 删除空回复回合").strip()
            r["metadata"]["version"] = "v5_clean"
            kept.append(r)
        else:
            r["metadata"]["version"] = "v5_clean"
            kept.append(r)

    # 校验: 无空 assistant
    for r in kept:
        for m in r["messages"]:
            if m["role"] == "assistant" and not m["content"].strip():
                print(f"[ERR] 残留空回复: {r['id']}")
                raise SystemExit(1)

    tmp = OUT.with_suffix(".tmp")
    tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in kept) + "\n", encoding="utf-8")
    os.replace(tmp, OUT)

    from collections import Counter
    report = {
        "before": before, "after": len(kept), "removed": len(removed),
        "removed_reasons": {r["id"]: REASONS.get(r["id"], "") for r in removed},
        "empty_fixed": sorted(EMPTY_FIX),
        "kept_huhu": KEEP_HUHU,
        "after_by_source": dict(Counter(r["metadata"]["data_source"] for r in kept)),
        "after_turns": sum(len(r["messages"]) // 2 for r in kept),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"清洗: {before} -> {len(kept)} 条 (删 {len(removed)} 整条, 空回复处理 {len(EMPTY_FIX)} 条)")
    print("删除:", sorted(DEL))
    print(f"剩余按来源: {report['after_by_source']}")
    print(f"写出: {OUT.name} / {REPORT.name}")


if __name__ == "__main__":
    main()
