# -*- coding: utf-8 -*-
"""扫描月社妃「名场面」候选段落(完整版, 输出中间文件)。

主题: 魔法之书、温柔世界、命运、结局、奇迹等体现月社妃核心思想的场景。
策略: 以主题关键词所在行为锚点, 展开到周围完整对话场景,
      要求场景内含妃发言, 输出完整段落文本到 JSONL 便于人工筛选。
"""
import json
import re
from pathlib import Path

GAME_DIR = Path(r"c:\Users\13474\Desktop\qqchat-enhanced\gametext\纸上魔法使")
OUT = Path(r"c:\Users\13474\Desktop\qqchat-enhanced\backend\data\character_dialogues\experiments\famous_scene_segments.jsonl")

# 核心思想关键词(名场面锚点)
THEME_KEYWORDS = [
    "魔法之书", "温柔的世界", "温柔世界", "命运", "结局", "奇迹", "复活",
    "理想乡", "世界遗忘", "幸福", "纸上", "冒牌书", "最后的", "最后一刻",
]

SPEAKER_RE = re.compile(r"^\[(.+?)\] 「(.+)」$")
KISAKI = "妃"
WINDOW = 45  # 关键词前后展开窗口(行)


def main():
    out_records = []
    for path in sorted(GAME_DIR.glob("*.txt")):
        lines = path.read_text(encoding="utf-8").splitlines()
        # 关键词锚点
        anchors = set()
        for i, ln in enumerate(lines):
            for kw in THEME_KEYWORDS:
                if kw in ln:
                    anchors.add(i)
        if not anchors:
            continue
        # 合并相邻锚点
        anchors = sorted(anchors)
        merged = []
        for a in anchors:
            if merged and a - merged[-1][1] <= WINDOW * 2:
                merged[-1][1] = a
            else:
                merged.append([a, a])
        for a0, a1 in merged:
            # 场景窗口
            start = max(0, a0 - WINDOW)
            end = min(len(lines), a1 + WINDOW)
            # 场景内妃发言
            kisaki = []
            for i in range(start, end):
                m = SPEAKER_RE.match(lines[i].strip())
                if m and m.group(1) == KISAKI:
                    kisaki.append((i + 1, m.group(2)))
            if not kisaki:
                continue
            hits = []
            for i in range(start, end):
                for kw in THEME_KEYWORDS:
                    if kw in lines[i]:
                        hits.append((i + 1, kw))
            # 去掉与已有段落重叠的
            rec = {
                "file": path.name,
                "start": start + 1,
                "end": end,
                "n_kisaki": len(kisaki),
                "keywords": [(ln, kw) for ln, kw in hits],
                "kisaki_lines": [{"line": ln, "text": t} for ln, t in kisaki],
                "text": "\n".join(f"{i+1}: {lines[i]}" for i in range(start, end) if lines[i].strip()),
            }
            out_records.append(rec)

    # 去重: 同文件场景重叠超过 60% 只保留妃发言更多者
    def overlap(a, b):
        if a["file"] != b["file"]:
            return 0
        s = max(a["start"], b["start"])
        e = min(a["end"], b["end"])
        if s >= e:
            return 0
        return min((e - s) / (a["end"] - a["start"]), (e - s) / (b["end"] - b["start"]))

    dedup = []
    for rec in out_records:
        if any(overlap(rec, d) > 0.6 for d in dedup):
            continue
        dedup.append(rec)

    dedup.sort(key=lambda r: (-r["n_kisaki"], -len(r["keywords"])))
    with OUT.open("w", encoding="utf-8") as f:
        for rec in dedup:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"共 {len(dedup)} 个候选场景 -> {OUT}")
    for r in dedup:
        kws = ",".join(sorted({kw for _, kw in r['keywords']}))
        print(f"{r['file']} | L{r['start']}-L{r['end']} | 妃{r['n_kisaki']}句 | {kws}")


if __name__ == "__main__":
    main()
