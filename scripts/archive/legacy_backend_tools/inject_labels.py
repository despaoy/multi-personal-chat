"""将人工标注的边界 case 注入旧训练缓存并重新训练（历史工具）。"""
import json, sys
from pathlib import Path
from collections import OrderedDict

BASE = Path(__file__).parent.parent
CACHE_PATH = BASE / "intent_training_cache.json"

# 人工标注 (text, label): 1=NEEDS_RAG, 0=NO_RAG
CORRECTED = [
    # 模式A: XX是谁/介绍一下XX → 全部是 Y
    ("刻晴是谁", 1), ("七七是谁", 1), ("风神是谁", 1),
    ("给我介绍一下钟离", 1), ("介绍一下夜兰", 1), ("给我介绍一下七七", 1),
    ("给我说说风神的故事", 1), ("刻晴的料理是什么", 1), ("你和钟离什么关系", 1),
    # 模式B: 要不要抽XX → 全Y，只有提灯是N
    ("要不要抽温迪？", 1), ("要不要抽香菱？", 1), ("要不要抽琴？", 1),
    ("要不要抽班尼特？", 1), ("要不要抽万叶？", 1), ("要不要抽迪卢克？", 1),
    ("要不要抽宵宫？", 1), ("要不要抽提灯？", 0),
    # 模式C: 游戏术语 → 全部是 Y
    ("寒天之钉是做什么的", 1), ("寒天之钉有什么用？", 1),
    ("如何快速获得经验书？", 1), ("经验书主要刷哪个关卡比较好？", 1),
    ("有哪些方法可以提高探索度？", 1), ("龙脊雪山材料在哪刷？", 1),
    ("钓鱼点在哪里？", 1),
    # 模式D: 明确非游戏 → 全部是 N
    ("帮忙找找好看的猫咪照片。", 0), ("炫耀小事：给花浇水后，它们都更绿了。", 0),
    ("听说最近有个很火的小品剧。", 0), ("听说有个好去处，但地址找不到。", 0),
]

# 原神真实角色名/地名，用于过滤LLM幻觉
REAL_GENSHIN_NAMES = {
    '胡桃','钟离','七七','甘雨','刻晴','夜兰','云堇','魈','温迪','行秋','班尼特',
    '雷电将军','纳西妲','神里绫华','八重神子','公子','散兵','女士','达达利亚',
    '凝光','荒泷一斗','托马','艾尔海森','赛诺','迪希雅','万叶','宵宫','迪卢克',
    '琴','香菱','砂糖','迪奥娜','可莉','莫娜','菲谢尔','北斗','诺艾尔',
    '蒙德','璃月','稻妻','须弥','枫丹','纳塔','至冬','天空岛','层岩巨渊',
    '原神','旅行者','派蒙','深渊','教团','愚人众','七神','魔神','西风骑士团',
    '元素','圣遗物','命座','天赋','世界任务','传说任务','周本','秘境',
    '抽卡','祈愿','保底','往生堂','堂主','岩王帝君','摩拉克斯','巴巴托斯',
    '经验书','摩拉','成就','宝箱','神瞳','钓鱼','特产','探索度','突破',
    '尘歌壶','七圣召唤','深境螺旋','寒天之钉',
}

# 中文常用词，含这些词可能是正常中文而非幻觉
COMMON_CHINESE = {'你','我','他','她','它','们','的','了','是','在','不','有','和','就','都','要','会','可以','能','去','来','想','说','看','做','吃','喝','玩','什么','怎么','哪里','为什么','如何','怎样','谁','哪','几','吗','呢','吧','啊','呀','哦','嗯'}

def is_hallucination(text: str) -> bool:
    """检查是否是LLM生成的幻觉样本（虚构角色名/地名）"""
    has_genshin = any(t in text for t in REAL_GENSHIN_NAMES)
    has_common = any(t in text for t in COMMON_CHINESE)
    if has_genshin:
        return False
    if not has_common:
        return True  # 既没有原神词也没有中文常用词 → 幻觉
    return False

# 加载缓存
samples = []
if CACHE_PATH.exists():
    with open(CACHE_PATH, encoding="utf-8") as f:
        samples = json.load(f)

# 去幻觉
before = len(samples)
samples = [[t, l] for t, l in samples if not is_hallucination(t)]
hallucinated = before - len(samples)
if hallucinated:
    print(f"清除 LLM 幻觉样本: {hallucinated} 条")

# 注入人工标注
existing = {t for t, _ in samples}
added = 0
for text, label in CORRECTED:
    if text not in existing:
        samples.append([text, label])
        added += 1
print(f"注入人工标注: {added} 条")

# 去重保存
seen = OrderedDict()
for t, l in samples:
    seen[t] = l
samples = [[t, l] for t, l in seen.items()]

with open(CACHE_PATH, "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)
print(f"缓存已更新: {len(samples)} 条 → {CACHE_PATH}")
print("现在可以运行 train_intent_classifier.py 重新训练")
