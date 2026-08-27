"""扩充边界case的样本量 - 用模板批量生成同模式变体（历史工具）。"""
import json
from pathlib import Path

BASE = Path(__file__).parent.parent
CACHE_PATH = BASE / "intent_training_cache.json"

# 原神角色名列表
CHARACTERS = [
    "胡桃", "钟离", "刻晴", "七七", "甘雨", "夜兰", "魈", "温迪",
    "纳西妲", "雷电将军", "神里绫华", "八重神子", "公子", "散兵",
    "凝光", "荒泷一斗", "行秋", "班尼特", "万叶", "宵宫", "迪卢克",
    "琴", "香菱", "砂糖", "迪奥娜", "可莉", "莫娜", "菲谢尔",
]

augmented = []

# 模式1: XX是谁 → NEEDS_RAG
for name in CHARACTERS:
    augmented.append((f"{name}是谁", 1))
    augmented.append((f"{name}是什么角色", 1))
    augmented.append((f"{name}强吗", 1))

# 模式2: 介绍一下XX → NEEDS_RAG
for name in CHARACTERS:
    augmented.append((f"介绍一下{name}", 1))
    augmented.append((f"给我介绍一下{name}", 1))

# 模式3: 要不要抽XX → NEEDS_RAG
for name in CHARACTERS:
    augmented.append((f"要不要抽{name}？", 1))
    augmented.append((f"{name}值得抽吗", 1))

# 模式4: XX和XX什么关系 → NEEDS_RAG
pairs = [("胡桃", "钟离"), ("刻晴", "凝光"), ("魈", "钟离"), ("温迪", "钟离"), ("雷电将军", "八重神子")]
for a, b in pairs:
    augmented.append((f"{a}和{b}什么关系", 1))

# 模式5: XX的YY是什么 → NEEDS_RAG
for name in CHARACTERS[:10]:
    for attr in ["命座效果", "天赋", "最佳武器", "毕业面板", "圣遗物推荐"]:
        augmented.append((f"{name}的{attr}是什么", 1))

# 模式6: 明确非游戏 → NO_RAG (补充分布)
negatives = [
    "今天好累啊", "周末什么安排", "你睡了吗", "帮我查一下天气",
    "讲个笑话听听", "你知道附近有什么好吃的吗", "明天有空吗",
    "最近有什么好看的电影", "你觉得我穿什么颜色好看",
    "帮我算个数", "推荐一本好书", "心情不好怎么办",
    "怎么做红烧肉", "这道题怎么解", "帮我翻译一下",
    "能给我唱首歌吗", "你相信星座吗", "今天是什么日子",
    "有什么好听的歌推荐", "你觉得人生的意义是什么",
    "能帮我写封邮件吗", "叫什么名字好听", "奶茶喝什么好",
    "你有喜欢的人吗", "你害怕什么",
]
for t in negatives:
    augmented.append((t, 0))

# 加载缓存
samples = []
if CACHE_PATH.exists():
    with open(CACHE_PATH, encoding="utf-8") as f:
        samples = json.load(f)

existing = {t for t, _ in samples}
added = 0
for text, label in augmented:
    if text not in existing:
        samples.append([text, label])
        added += 1

print(f"新增模板样本: {added} 条  (模式1~5: NEEDS_RAG, 模式6: NO_RAG)")

with open(CACHE_PATH, "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)

print(f"缓存总计: {len(samples)} 条 → {CACHE_PATH}")
print("运行 train_intent_classifier.py 重新训练")
