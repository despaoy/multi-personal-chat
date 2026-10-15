#!/usr/bin/env python3
"""
RAG 意图分类器训练数据生成 + 模型训练脚本
生成高质量、领域对齐的训练数据，训练轻量级意图分类模型。
"""

import json
import logging
import random
import sys
from itertools import product
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BACKEND_DIR
sys.path.insert(0, str(BACKEND_DIR))
from knowledge.intent_trainer import MODEL_DIR, _load_embed_model  # noqa: E402 -- direct script bootstrap

GENSHIN_REGIONS = [
    "蒙德", "蒙德城", "璃月", "璃月港", "稻妻", "须弥", "枫丹", "纳塔", "至冬",
    "风起地", "星落湖", "晨曦酒庄", "千风神殿", "风龙废墟", "奔狼领",
    "绝云间", "轻策庄", "荻花洲", "望舒客栈", "归离原", "翠玦坡",
    "鸣神岛", "神无冢", "八酝岛", "海祇岛", "清籁岛",
    "化城郭", "维摩庄", "护世林", "禅那园", "阿如村",
    "枫丹廷", "苍晶区", "白露区", "黎翡区",
    "龙脊雪山", "金苹果群岛", "渊下宫", "层岩巨渊",
]

GENSHIN_CHARACTERS = [
    "胡桃", "钟离", "刻晴", "甘雨", "魈", "温迪", "可莉", "迪卢克", "琴",
    "莫娜", "迪奥娜", "北斗", "凝光", "香菱", "班尼特", "砂糖", "重云",
    "诺艾尔", "芭芭拉", "雷泽", "菲谢尔", "安柏", "凯亚", "丽莎",
    "行秋", "烟绯", "辛焱", "云堇", "申鹤", "夜兰", "瑶瑶",
    "雷电将军", "神里绫华", "神里绫人", "宵宫", "八重神子", "珊瑚宫心海",
    "荒�的�的一斗", "九条裟罗", "早柚", "托马", "五郎",
    "纳西妲", "提纳里", "赛诺", "妮露", "艾尔海森", "迪希雅", "卡维",
    "流浪者", "珐露珊", "莱依拉",
    "芙宁娜", "那维莱特", "林尼", "琳妮特", "菲米尼", "夏洛蒂",
    "娜维娅", "莱欧斯利", "克洛琳德",
    "阿贝多", "优菈", "万叶", "宵宫", "早柚",
    "罗莎莉亚", "辛焱", "达达利亚", "枫原万叶",
    "七七", "白术", "瑶瑶",
]

GENSHIN_ORGS = [
    "西风骑士团", "骑士团", "千岩军", "往生堂", "璃月七星", "月海亭",
    "社奉行", "勘定奉行", "天守阁", "幕府军", "海祇岛军",
    "教令院", "须弥教令院", "镀金旅团",
    "执律庭", "巡轨庭", "特巡队",
    "愚人众", "深渊教团", "魔女会",
]

GENSHIN_ITEMS = [
    "护摩之杖", "和璞鸢", "狼的末路", "天空之翼", "四风原典", "天空之卷",
    "风鹰剑", "天空之刃", "斫峰之刃", "磐岩结绿",
    "魔女套", "宗室套", "绝缘套", "追忆套", "如雷套", "逆飞套",
    "千岩套", "磐岩套", "角斗套", "乐团套", "流浪者套", "战狂套",
    "冰风套", "水套", "风套", "岩套", "雷套", "草套", "火套",
    "原石", "纠缠之缘", "相遇之缘", "摩拉", "树脂", "脆弱树脂",
    "经验书", "大英雄的经验", "冒险家的经验", "流浪者的经验",
    "突破材料", "天赋书", "武器突破材料",
    "圣遗物", "五星圣遗物", "四星圣遗物",
    "尘歌壶", "七圣召唤",
]

GENSHIN_CONCEPTS = [
    "元素反应", "蒸发", "融化", "超导", "感电", "扩散", "结晶", "燃烧",
    "激化", "超激化", "蔓激化", "绽放", "超绽放", "烈绽放",
    "增幅反应", "剧变反应", "元素精通", "暴击率", "暴击伤害",
    "双爆", "锁面板", "压血", "附魔", "染色", "快照",
    "命座", "零命", "满命", "一命", "二命", "六命",
    "深境螺旋", "深渊", "大世界", "秘境", "周本",
    "冒险等级", "世界等级", "好感度", "命之座",
    "七天神像", "传送锚点", "风神瞳", "岩神瞳", "雷神瞳",
    "风之印", "岩之印", "雷之印",
    "双火共鸣", "双冰共鸣", "双雷共鸣", "双风共鸣", "双水共鸣", "双岩共鸣",
    "种门", "国家队", "胡行钟", "万达国际", "雷国", "宵宫队",
    "魔神任务", "传说任务", "世界任务", "每日委托",
    "海灯节", "风花节", "羽球节",
]

GENSHIN_LORE = [
    "风神", "岩神", "雷神", "草神", "水神", "火神", "冰神",
    "巴巴托斯", "摩拉克斯", "巴尔泽布", "布耶尔", "芙卡洛斯",
    "温迪", "钟离", "雷电将军", "纳西妲", "芙宁娜",
    "四风守护", "璃月七星", "三奉行",
    "坎瑞亚", "戴因斯雷布", "天理", "深渊",
    "尘世七执政", "神之心", "神之眼",
    "提瓦特", "天空岛",
]


def generate_positive_examples() -> list:
    examples = []

    # Pattern 1: 角色 + 属性/能力查询
    char_attrs = [
        "的命座效果是什么", "的命座有什么用", "的命座值得抽吗",
        "用什么武器最好", "的武器推荐", "适合什么武器",
        "的圣遗物怎么配", "的圣遗物推荐", "带什么圣遗物",
        "的天赋怎么点", "的天赋升级顺序", "的天赋优先级",
        "怎么配队", "的配队推荐", "适合什么队伍",
        "的输出手法", "怎么玩", "的玩法攻略",
        "需要堆什么属性", "的毕业面板", "的面板要求",
        "的突破材料", "的突破材料哪里刷", "升级需要什么材料",
        "几命才能用", "几命质变", "零命能玩吗",
        "的定位是什么", "是主C吗", "是辅助吗", "是副C吗",
        "值得抽吗", "值得培养吗", "强度怎么样",
    ]
    for char in GENSHIN_CHARACTERS:
        for attr in random.sample(char_attrs, min(8, len(char_attrs))):
            examples.append(f"{char}{attr}")

    # Pattern 2: 地区 + 泛查询（核心覆盖）
    region_queries = [
        "有什么好玩的", "有什么特产", "有什么节日", "有什么名胜",
        "有什么角色", "有什么NPC", "有什么商店", "有什么美食",
        "有什么副本", "有什么秘境", "有什么BOSS", "有什么周本",
        "有什么宝箱", "有什么神瞳", "有什么传送点",
        "有什么隐藏任务", "有什么世界任务", "有什么成就",
        "有什么采集物", "有什么矿石", "有什么钓鱼点",
        "有什么食谱", "有什么武器", "有什么圣遗物",
        "有什么剧情", "有什么故事", "有什么背景设定",
        "有什么特色", "有什么新内容", "有什么活动",
        "有什么攻略", "有什么推荐", "有什么值得去的",
        "有什么好玩的吗", "有啥好玩的", "有啥特产",
        "有什么有名的节日", "有什么传统庆典", "有什么庆典活动",
        "有什么历史", "有什么传说", "有什么故事背景",
        "有什么声望任务", "有什么每日委托",
        "有什么采集路线", "有什么矿点", "有什么精英怪",
        "有什么隐藏宝箱", "有什么彩蛋", "有什么解密",
    ]
    for region in GENSHIN_REGIONS:
        for query in random.sample(region_queries, min(10, len(region_queries))):
            examples.append(f"{region}{query}")

    # Pattern 3: 组织/概念 + 查询
    org_queries = ["是什么", "是干什么的", "的成员有哪些", "的历史", "的任务", "的故事"]
    for org in GENSHIN_ORGS:
        for q in random.sample(org_queries, min(3, len(org_queries))):
            examples.append(f"{org}{q}")

    concept_queries = [
        "是什么意思", "怎么触发", "怎么算", "有什么用",
        "的机制是什么", "怎么利用", "和什么搭配",
    ]
    for concept in GENSHIN_CONCEPTS:
        for q in random.sample(concept_queries, min(2, len(concept_queries))):
            examples.append(f"{concept}{q}")

    # Pattern 4: 物品 + 查询
    item_queries = [
        "怎么获得", "在哪里刷", "适合谁用", "效果是什么",
        "和什么搭配", "值得抽吗", "怎么用",
    ]
    for item in GENSHIN_ITEMS:
        for q in random.sample(item_queries, min(2, len(item_queries))):
            examples.append(f"{item}{q}")

    # Pattern 5: 剧情/设定查询
    lore_queries = [
        "的故事", "的背景", "的设定", "是谁", "和什么有关",
    ]
    for lore in GENSHIN_LORE:
        for q in random.sample(lore_queries, min(2, len(lore_queries))):
            examples.append(f"{lore}{q}")

    # Pattern 6: 口语化/自然表达
    colloquial = [
        "蒙德城有什么有名的节日，说几个",
        "璃月港有什么好玩的吗",
        "稻妻有啥特产啊",
        "胡桃怎么玩啊",
        "钟离的护盾有多厚",
        "深渊12层怎么打",
        "帮我看看胡桃怎么配队",
        "告诉我蒙德有什么节日",
        "说说璃月的故事",
        "介绍一下稻妻的设定",
        "聊聊枫丹有什么新东西",
        "蒙德有什么好玩的，推荐一下",
        "璃月有什么值得去的地方",
        "须弥有什么新角色",
        "枫丹有什么新玩法",
        "说几个蒙德的节日",
        "推荐几个璃月的特产",
        "蒙德有啥好玩的",
        "璃月有啥好吃的",
        "稻妻有啥角色",
        "原神里蒙德有什么节日",
        "原神璃月港有什么特色",
        "胡桃的输出手法是什么",
        "怎么打深渊",
        "怎么快速升级",
        "原石怎么攒",
        "圣遗物怎么刷",
        "这个角色值得培养吗",
        "帮我配个队",
        "这个武器给谁用好",
        "这个圣遗物主词条对不对",
        "怎么解锁这个秘境",
        "这个任务怎么做",
        "这个BOSS怎么打",
        "这个成就怎么完成",
        "这个食谱在哪里获得",
        "这个材料在哪里采集",
        "这个角色的故事是什么",
        "这个地区有什么隐藏内容",
        "蒙德的风神瞳都在哪",
        "璃月的岩神瞳位置",
        "稻妻的雷神瞳分布",
        "怎么提升冒险等级",
        "世界等级有什么影响",
        "元素反应怎么搭配",
        "配队有什么技巧",
        "深渊满星怎么打",
        "零氪怎么玩",
        "新手应该练什么角色",
        "萌新入坑有什么建议",
        "这个版本有什么活动",
        "卡池顺序是什么",
        "保底机制是什么",
        "武器池怎么抽",
        "原神什么角色最强",
        "哪个角色值得抽",
        "现在入坑来得及吗",
        "回归玩家有什么福利",
    ]
    examples.extend(colloquial)

    # Pattern 7: 知识型疑问词模式
    knowledge_patterns = [
        "什么是{c}", "{c}是什么", "{c}怎么用", "{c}怎么获得",
        "{c}在哪里", "{c}有什么用", "{c}的效果", "{c}怎么触发",
        "怎么解锁{c}", "怎么获得{c}", "哪里可以刷{c}",
    ]
    for pattern in knowledge_patterns:
        for c in random.sample(GENSHIN_CONCEPTS[:20], 5):
            examples.append(pattern.format(c=c))

    # 去重
    examples = list(dict.fromkeys(examples))
    return examples


def generate_negative_examples() -> list:
    examples = []

    # 闲聊 - 日常生活
    daily_chat = [
        "你好啊", "嗨", "早上好", "晚安", "在吗",
        "今天天气真好", "好无聊啊", "好累", "好开心",
        "吃饭了吗", "你在干嘛", "最近怎么样",
        "哈哈", "嘿嘿", "嗯嗯", "好的", "知道了",
        "谢谢", "不客气", "没关系", "对不起",
        "我回来了", "我走了", "拜拜", "下次见",
        "好热啊", "好冷啊", "下雨了", "出太阳了",
        "我饿了", "我想睡觉", "好困", "不想动",
    ]
    examples.extend(daily_chat)

    # 闲聊 - 情感表达（不含知识查询词）
    emotional = [
        "我好想你", "今天心情不太好", "有点烦", "开心死了",
        "太棒了", "好烦啊", "气死我了", "好感动",
        "我好开心", "好难过", "有点焦虑", "压力好大",
        "想出去玩", "想看电影", "想听音乐", "想打游戏",
        "好无聊", "好寂寞", "好孤独", "好害怕",
    ]
    examples.extend(emotional)

    # 闲聊 - 观点/感受（不含知识查询意图）
    opinions = [
        "我觉得今天挺不错的", "这个游戏真好玩", "这首歌好好听",
        "这部电影太精彩了", "这个动漫超好看", "那家餐厅味道不错",
        "我觉得你说的有道理", "我也这么觉得", "说得对",
        "这个角色设计得真好看", "这个剧情太感人了",
        "这个活动挺有意思的", "这个玩法还不错",
        "这个皮肤真好看", "这个特效太帅了",
    ]
    examples.extend(opinions)

    # 闲聊 - 社交互动（不含知识查询）
    social = [
        "来玩吧", "一起打游戏", "组队吗", "加个好友",
        "你在玩什么", "你几级了", "你是什么服务器",
        "你叫什么名字", "你多大了", "你是哪里人",
        "你也在玩原神吗", "你最喜欢哪个角色",
        "你抽到什么了", "你氪了多少", "你是零氪吗",
        "你玩多久了", "你深渊满星了吗",
    ]
    examples.extend(social)

    # 闲聊 - 角色喜好（主观偏好，不需要知识库）
    preference = [
        "我最喜欢胡桃了", "钟离好帅", "刻晴好可爱",
        "我不喜欢这个角色", "这个角色不好看",
        "我想要胡桃", "我想抽钟离", "我歪了",
        "我出金了", "我十连双金", "我又歪了",
        "保底了", "小保底歪了", "大保底出了",
        "我攒了160抽", "我只有20抽了",
    ]
    examples.extend(preference)

    # 闲聊 - 游戏状态分享（不需要知识查询）
    game_status = [
        "我刚上线", "我正在做任务", "我在打深渊",
        "我刚抽完卡", "我正在探索", "我在钓鱼",
        "我刚打完周本", "我正在刷圣遗物",
        "我今天没体力了", "我树脂用完了",
        "我冒险等级50了", "我刚升到45级",
        "我终于满星了", "我打不过去",
    ]
    examples.extend(game_status)

    # 闲聊 - 模糊闲聊（不指向具体知识需求）
    vague = [
        "随便聊聊", "说说话", "陪我聊会天",
        "好无聊说点啥", "讲个笑话", "来段相声",
        "唱首歌", "跳个舞", "卖个萌",
        "你真有趣", "你真可爱", "你好厉害",
        "你是什么人", "你叫什么", "你在哪",
        "你喜欢吃什么", "你有什么爱好",
        "给我讲个故事", "说点有趣的",
        "今天过得怎么样", "周末有什么安排",
    ]
    examples.extend(vague)

    # 闲聊 - 非原神领域知识查询
    non_genshin = [
        "今天的新闻有什么", "帮我查一下天气",
        "教我做菜", "推荐一部电影",
        "帮我写个作文", "这道数学题怎么做",
        "英语怎么学", "怎么减肥",
        "怎么护肤", "推荐个游戏",
        "怎么修电脑", "手机卡怎么办",
        "怎么学编程", "Python怎么学",
    ]
    examples.extend(non_genshin)

    # 闲聊 - 短句/表情式
    short = [
        "嗯", "啊", "哦", "诶", "哇",
        "？", "？？", "。。", "...",
        "好的呢", "知道了啦", "行吧",
        "真的吗", "不会吧", "怎么可能",
        "太离谱了", "绝了", "牛啊",
        "666", "nb", "yyds",
    ]
    examples.extend(short)

    # 去重
    examples = list(dict.fromkeys(examples))
    return examples


def generate_training_data() -> list:
    logger.info("生成训练数据...")
    positive = generate_positive_examples()
    negative = generate_negative_examples()

    logger.info(f"正例(需要RAG): {len(positive)} 条")
    logger.info(f"负例(不需要RAG): {len(negative)} 条")

    data = [[text, 1] for text in positive] + [[text, 0] for text in negative]
    random.shuffle(data)

    cache_path = OUTPUT_DIR / "intent_training_cache.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"训练数据已保存: {cache_path}")

    return data


def train_model(data: list):
    logger.info("开始训练意图分类模型...")

    import joblib
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    embed_model, embedding_source = _load_embed_model()

    texts = [d[0] for d in data]
    labels = [d[1] for d in data]

    logger.info(f"编码 {len(texts)} 条文本...")
    X = embed_model.encode(texts, show_progress_bar=True, batch_size=64)
    y = np.array(labels)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info("交叉验证中...")
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
    scores = cross_val_score(clf, X_scaled, y, cv=5, scoring="f1", error_score="raise")
    logger.info(f"5折交叉验证 F1: {scores.mean():.4f} (+/- {scores.std():.4f})")

    logger.info("训练最终模型...")
    clf.fit(X_scaled, y)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(clf, MODEL_DIR / "classifier.joblib")
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")

    config = {
        "model_type": "logistic_regression",
        "embedding_model": embedding_source,
        "threshold": 0.5,
        "confidence_fallback": 0.3,
        "version": "2.0",
        "training_samples": len(data),
        "positive_samples": int(sum(labels)),
        "negative_samples": int(len(labels) - sum(labels)),
        "cv_f1_mean": float(scores.mean()),
        "cv_f1_std": float(scores.std()),
    }
    with open(MODEL_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info(f"模型已保存: {MODEL_DIR}")

    # 测试关键用例
    test_cases = [
        ("蒙德城有什么有名的节日，说几个", True),
        ("蒙德有什么节日", True),
        ("璃月港有什么特产", True),
        ("胡桃怎么配队", True),
        ("钟离的护盾有多厚", True),
        ("原神里蒸发反应怎么触发", True),
        ("深渊12层怎么打", True),
        ("你好啊", False),
        ("好无聊啊", False),
        ("我最喜欢胡桃了", False),
        ("我刚上线", False),
        ("随便聊聊", False),
        ("太棒了", False),
        ("来玩吧", False),
    ]

    logger.info("\n===== 关键用例测试 =====")
    X_test = embed_model.encode([t[0] for t in test_cases])
    X_test_scaled = scaler.transform(X_test)
    probs = clf.predict_proba(X_test_scaled)
    preds = clf.predict(X_test_scaled)

    all_correct = True
    for (text, expected), pred, prob in zip(test_cases, preds, probs):
        rag_prob = prob[1] if len(prob) > 1 else prob[0]
        correct = (pred == 1) == expected
        status = "✅" if correct else "❌"
        if not correct:
            all_correct = False
        logger.info(f"  {status} [{rag_prob:.2%}] {text} (期望={'需要RAG' if expected else '不需要'})")

    if all_correct:
        logger.info("所有关键用例通过！")
    else:
        logger.warning("部分用例未通过，可能需要调整训练数据")

    return True


def main():
    data = generate_training_data()
    success = train_model(data)
    if success:
        logger.info("训练完成！模型已保存到 %s", MODEL_DIR)
        logger.info("请重启后端服务以加载新模型")
    else:
        logger.error("训练失败")


if __name__ == "__main__":
    main()
