"""筛选真实的边界 case（排除 LLM 幻觉样本），供人工标注（历史工具）。"""
import sys, json, sqlite3
sys.path.insert(0, "d:/projects/backend")
from pathlib import Path

from rag_intent_detector import _load_ml_model, get_intent_detector
import rag_intent_detector as r

_load_ml_model()
encoder = r._ml_encoder
clf = r._ml_classifier
detector = get_intent_detector()

# 真实原神关键词 (来自规则引擎的 genshin_terms)
REAL_GENSHIN = {
    '原神','角色','武器','圣遗物','天赋','命座','命之座',
    '蒙德','璃月','稻妻','须弥','枫丹','纳塔','至冬','天空岛',
    '胡桃','钟离','七七','甘雨','刻晴','夜兰','云堇','魈','温迪',
    '雷电将军','纳西妲','神里绫华','八重神子','公子','散兵','女士',
    '凝光','荒泷一斗','托马','艾尔海森','赛诺','迪希雅','行秋','班尼特',
    '万叶','宵宫','迪卢克','琴','香菱','提灯','砂糖','迪奥娜',
    '深渊','教团','愚人众','七神','魔神','执政','眷属',
    '元素','反应','火','水','风','雷','草','冰','岩',
    '副本','秘境','世界任务','传说任务','每日委托','活动','周本',
    '抽卡','祈愿','保底','定轨','武器池','角色池',
    '往生堂','堂主','岩王帝君','摩拉克斯','巴巴托斯',
    '旅行者','派蒙','千岩军','西风骑士团','群玉阁',
    '经验书','摩拉','摩卡','BOSS','突破','材料','升级',
    '探索度','成就','宝箱','神瞳','钓鱼','特产','采集',
    '尘歌壶','七圣召唤','深境螺旋',
}

# 真实社交/中文常用词 - 这些不应被误判
REAL_SOCIAL = {
    '你好','谢谢','在吗','晚安','早安','再见','拜拜',
    '吃了','天气','睡觉','上班','下班','学习','工作',
    '感觉','想','不','好','是','有','什么',
}

all_samples = set()
cache_path = Path("d:/projects/backend/intent_training_cache.json")
if cache_path.exists():
    with open(cache_path, encoding="utf-8") as f:
        for t, _ in json.load(f):
            if 3 < len(t) < 200:
                all_samples.add(t)

try:
    conn = sqlite3.connect("d:/projects/backend/qq_assistant.db", check_same_thread=False)
    rows = conn.execute("SELECT DISTINCT message FROM messages ORDER BY createdAt DESC LIMIT 1000").fetchall()
    conn.close()
    for (m,) in rows:
        if 3 < len(m.strip()) < 200:
            all_samples.add(m.strip())
except Exception:
    pass

results = []
for text in all_samples:
    emb = encoder.encode([text])
    proba = clf.predict_proba(emb)[0]
    ml_pred = proba[1] >= 0.5
    rule_pred, rule_reason = detector.needs_rag(text)
    if ml_pred == rule_pred:
        continue

    # 检查是否含真实原神词或中文常用词
    has_genshin = any(t in text for t in REAL_GENSHIN)
    has_social = any(t in text for t in REAL_SOCIAL) and not has_genshin

    # 排除明显幻觉：不含任何已知原神词且不含中文常用词的
    if not has_genshin and not has_social:
        continue

    results.append((text, ml_pred, rule_pred, proba[1], rule_reason))

results.sort(key=lambda x: abs(x[3] - 0.5))

print(f"真实边界 case (排除LLM幻觉后): {len(results)} 条\n")
for i, (text, ml, rule, pos, reason) in enumerate(results, 1):
    confidence = f"{pos:.1%}"
    print(f"{i:3d}. [{confidence:>6s}] ML={'Y' if ml else 'N'} 规则={'Y' if rule else 'N'}  |  {text}")
    print(f"        规则原因: {reason}\n")
