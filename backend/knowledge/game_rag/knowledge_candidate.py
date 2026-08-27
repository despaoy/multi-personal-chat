"""P5 知识卡候选生成（事实卡/关系卡/事件卡）基础设施。

职责：在 P4E approved enriched 场景包（enriched_scenes.jsonl + enriched_manifest.json，
双文件摘要绑定）之上，通过可注入的模型客户端为每个场景生成**待人工审核的知识卡候选**
（FactDocument / RelationDocument / EventDocument 的事实内容），维护可断点续跑的
候选运行状态，并确定性地产出候选文档（稳定 ID、去重、冲突分组、排序）。

明确不做：不初始化模型服务器、不下载模型、不读取环境密钥（模型客户端由调用方按
KnowledgeModelClient 协议注入）；不生成 embedding / 向量 / 索引 / 数据库 / API；
候选文档 review_status 至多为 needs_review，绝不自动 approved；不修改 P3/P4 的
任何审核决定或产物（enriched 输入只读）。

五类产物的区别（不得混淆）：
- 模型候选载荷（FactCard/RelationCard/EventCard）：模型输出经严格解析后的结构化候选，
  存于候选运行状态（断点续跑的载体）；
- 候选文档（FactDocument/RelationDocument/EventDocument）：finalize 从运行状态确定性
  构建的知识卡（稳定 ID + 原文 evidence_text），review_status ∈ {draft, needs_review}；
- 人工审核状态（KnowledgeReviewDocument）：逐卡的人工审核追踪（全部 draft 起步）；
- 运行 manifest（KnowledgeRunManifest）：唯一允许携带时间戳/调用统计等非确定性信息
  的产物；运行状态与候选文档保持确定性（无时间戳/随机数）；
- 质量报告（build_knowledge_quality_report）：确定性统计（去重/冲突/分布/失败）。

确定性：同一 enriched bundle + 同一模型输出重复运行，产物（ID、顺序、内容）完全一致。
详见 docs/research/KISAKI_GAME_RAG_SCHEMA.md 与 P5 阶段说明。
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from knowledge.game_rag.models import (
    EventDocument,
    FactDocument,
    NonEmptyStr,
    RealityStatus,
    RelationDocument,
    ReviewStatus,
    SceneDocument,
    SourceSpan,
)
from knowledge.game_rag.scene_metadata_review import (
    SourceManifestRef,
    _atomic_write_pair,
    _atomic_write_text,
)

KNOWLEDGE_CANDIDATE_SCHEMA_VERSION = 1
KNOWLEDGE_RUN_MANIFEST_SCHEMA_VERSION = 1
KNOWLEDGE_REVIEW_SCHEMA_VERSION = 1
GENERATOR_ID = "knowledge.game_rag.knowledge_candidate"

# 长场景按字符分片阈值：超过该字符数的场景分片调用（按行边界切，不重切场景）。
# 依据：qwen2.5:7b num_ctx=16384 下，6000 中文字符 + 提示 + 输出仍留有余量。
KNOWLEDGE_CHUNK_MAX_CHARS = 6000

# 单场景候选数量上限（质量优先于数量；允许 0 条）。
MAX_FACTS_PER_SCENE = 8
MAX_RELATIONS_PER_SCENE = 5
MAX_EVENTS_PER_SCENE = 3

# evidence 必须是能直接支撑卡片的局部原文，不能用整个场景代替引用。
MAX_FACT_RELATION_EVIDENCE_LINES = 24
MAX_EVENT_EVIDENCE_LINES = 60

# 明显不属于“死因”的模型值直接丢弃；死亡状态本身应由「状态」表达。
DEATH_CAUSE_SIGNAL_RE = re.compile(r"死亡|去世|身亡|自杀|自缢|事故|被杀|杀害|枪杀|烧死|溺死|病逝|毒杀|撕毁")
DEATH_CAUSE_NON_CAUSE_RE = re.compile(r"拒绝命令|附身|幽灵故事中的角色|祈祷|消失$|^死亡$|^死$|撕毁")
PREFERENCE_VALUE_RE = re.compile(r"^(喜欢|爱上|爱着|憧憬|偏爱|讨厌)")
GOAL_VALUE_RE = re.compile(r"^(想|希望|决定|打算|寻求|愿意|目标)")
BEHAVIOR_VALUE_RE = re.compile(r"^(经常|正在|继续|开始|尝试|寻找|跟随|接受|拒绝|赶走|破坏|打开|回忆)")
APPEARANCE_VALUE_RE = re.compile(r"(头发|发色|眼睛|红眼|白发|长发|短发|外貌|容貌)")
PERSONALITY_VALUE_RE = re.compile(r"^(内向|开朗|温柔|勇敢|活泼|豁达|冷静|脆弱|坚强|自卑|任性|有魅力)")
PERSISTENT_STATUS_RE = re.compile(
    r"^(在世|已故|死亡|存在|失忆|记忆障碍|失踪|不在|闭门不出|纸上存在|虚构存在|幽灵|"
    r"被囚禁|被诅咒|受伤|虚弱|生病|被排挤|被欺负|被迫害|无家可归|住在)"
)

# 亲属/恋爱/主从等是关系卡的语义，不应再以“某人的身份是某人的妹妹”
# 这种形式进入事实卡。旧模型尤其容易把同一证据窗口里的称谓误挂到当前
# subject 上；这里只处理身份字段，行为和经历字段仍可保留相应剧情事实。
RELATIONAL_IDENTITY_VALUE_RE = re.compile(
    r"(?:哥哥|姐姐|弟弟|妹妹|兄长|姐妹|兄妹|父亲|母亲|女儿|儿子|丈夫|妻子|"
    r"恋人|前恋人|男朋友|女朋友|朋友|同伴|室友|创造者|照顾者|依赖对象|主人)$"
)
RELATIONAL_IDENTITY_PREFIX_RE = re.compile(
    r"^(?:琉璃|妃|夜子|汀|理央|彼方|暗子|萤|克丽索贝莉露).*(?:哥哥|姐姐|弟弟|妹妹|"
    r"父亲|母亲|女儿|儿子|丈夫|妻子|恋人|前恋人|男朋友|女朋友|朋友|同伴|室友|"
    r"创造者|照顾者|依赖对象|主人)$"
)

# 这些是由 7B 在关系型上下文中生成的确定性误标，证据中的称谓并非该
# subject 的身份。使用结构条件而非卡片 ID，使候选重新 finalize 时仍稳定。
FALSE_IDENTITY_VALUE_RE = re.compile(r"^(?:母亲|父亲|哥哥|姐姐|弟弟|妹妹|丈夫|妻子|馆长)$")

# 已对照原文确认的字段错置。身份字段只保留人物/故事中的稳定角色；
# 行为、偏好、状态和关系性描述改回对应谓词，无法作为事实保留的直接丢弃。
# 键包含原始证据起点，避免影响同名词在其他故事层中的合法用法。
FACT_CONTENT_REWRITES: dict[tuple[str, int, str, str], dict[str, Any] | None] = {
    ("vol01_1翡翠的排挤原理", 7, "夜子", "图书馆管理员"): None,
    ("vol01_1翡翠的排挤原理", 801, "暗子", "图书馆管理员"): None,
    ("vol01_1翡翠的排挤原理", 1147, "琉璃", "夜子的路人"): None,
    ("vol01_1翡翠的排挤原理", 1907, "琉璃", "救世主"): None,
    ("vol01_1翡翠的排挤原理", 1896, "琉璃", "抱起彼方并带她去医院"): {
        "predicate": "行为",
        "value": "抱起彼方并带她去处理伤口",
        "title": "琉璃抱起彼方",
        "summary": "琉璃抱起脚部受伤的彼方，带她去可以处理伤口的地方。",
    },
    ("vol01_1翡翠的排挤原理", 2133, "琉璃", "幻想图书馆的管理员"): None,
    ("vol01_1翡翠的排挤原理", 2235, "彼方", "被欺负的对象"): None,
    ("vol01_1翡翠的排挤原理", 2595, "夜子", "琉璃的命令者"): None,
    ("vol01_1翡翠的排挤原理", 2830, "妃", "《翡翠的排挤原理》的读者"): {
        "predicate": "行为",
        "value": "阅读《翡翠的排挤原理》",
        "title": "妃阅读《翡翠的排挤原理》",
        "summary": "妃阅读并准确复述了《翡翠的排挤原理》的梗概。",
    },
    ("vol01_1翡翠的排挤原理", 3262, "琉璃", "彼方的青梅竹马"): {
        "predicate": "经历",
        "value": "与彼方是青梅竹马",
        "title": "琉璃与彼方的儿时关系",
        "summary": "琉璃称彼方是自己的青梅竹马。",
    },
    ("vol01_1翡翠的排挤原理", 3538, "妃", "推理小说爱好者"): {
        "predicate": "偏好",
        "value": "阅读推理小说",
        "title": "妃喜欢推理小说",
        "summary": "妃喜欢阅读推理小说。",
    },
    ("vol01_1翡翠的排挤原理", 1836, "魔法使", "谣言"): None,
    ("vol01_1翡翠的排挤原理", 3379, "彼方", "少女"): None,
    ("vol01_1翡翠的排挤原理", 3424, "琉璃", "少女"): None,
    ("vol02_2红宝石的天作之合", 32, "琉璃", "魔法使"): None,
    ("vol02_2红宝石的天作之合", 1158, "琉璃", "旧时好友"): None,
    ("vol02_2红宝石的天作之合", 2262, "琉璃", "帮助夜子借活动室"): {
        "predicate": "行为",
        "value": "为夜子借用活动室",
        "title": "琉璃为夜子借用活动室",
        "summary": "琉璃为了让夜子用餐，设法借到了活动室。",
    },
    ("vol02_2红宝石的天作之合", 21, "琉璃", "忏悔"): {
        "subject": "暗子",
        "predicate": "行为",
        "value": "向夜子道歉",
        "title": "暗子向夜子道歉",
        "summary": "暗子反复向女儿夜子道歉，为让她出生并遭遇不幸而悔恨。",
    },
    ("vol02_2红宝石的天作之合", 2335, "夜子", "喜欢琉璃"): None,
    ("vol02_2红宝石的天作之合", 2505, "夜子", "喜欢琉璃"): None,
    ("vol02_2红宝石的天作之合", 2635, "夜子", "喜欢理央"): None,
    ("vol02_2红宝石的天作之合", 2574, "夜子", "图书馆管理员"): None,
    ("vol02_2红宝石的天作之合", 2825, "《红宝石的天作之合》", "有效果"): {
        "value": "可能产生效果",
        "title": "《红宝石的天作之合》效果的推测",
        "summary": "叙述者推测《红宝石的天作之合》可能产生了让人物关系连接起来的效果。",
        "reality_status": "character_claim",
    },
    ("vol02_2红宝石的天作之合", 2710, "琉璃", "夜子的共犯"): {
        "predicate": "经历",
        "value": "与夜子共同隐瞒魔法之书的影响",
        "title": "琉璃与夜子共同隐瞒真相",
        "summary": "琉璃与夜子共同隐瞒魔法之书造成的影响，并被妃称为共犯。",
    },
    ("vol03_3蓝宝石的存在证明", 1659, "妃", "失忆"): None,
    ("vol03_3蓝宝石的存在证明", 1871, "理央", "夜子的老师"): {
        "predicate": "行为",
        "value": "教夜子制作糕点",
        "title": "理央教夜子制作糕点",
        "summary": "理央指导夜子学习制作糕点，并非夜子的老师身份。",
    },
    ("vol03_3蓝宝石的存在证明", 2982, "妃", "与父亲交往"): {
        "predicate": "行为",
        "value": "以琉璃女朋友身份向父亲介绍自己",
        "title": "妃向琉璃的父亲介绍自己",
        "summary": "妃以琉璃女朋友的身份向琉璃的父亲介绍自己，并借机提及自己的父亲。",
    },
    ("vol03_3蓝宝石的存在证明", 4278, "《缟玛瑙的不在证明》", "最恶劣的书"): {
        "predicate": "设定",
        "title": "《缟玛瑙的不在证明》的评价",
        "summary": "《缟玛瑙的不在证明》被描述为充满恶意的故事。",
    },
    ("vol04_4紫水晶的怪异传说", 375, "琉璃", "有恋妹情结"): {
        "predicate": "性格",
        "title": "琉璃的恋妹倾向",
        "summary": "琉璃自称对妹妹妃有强烈的恋慕。",
    },
    ("vol04_4紫水晶的怪异传说", 1988, "妃", "事故"): {
        "value": "可能是事故",
        "title": "妃死因的猜测",
        "summary": "琉璃怀疑妃的死可能只是事故，汀没有确认这一说法。",
        "reality_status": "character_claim",
    },
    ("vol04_4紫水晶的怪异传说", 2530, "夜子", "魔法之书管理员"): None,
    ("vol04_4紫水晶的怪异传说", 1139, "夜子", "讨厌死动物"): None,
    ("vol04_4紫水晶的怪异传说", 3331, "琉璃", "日向彼方"): None,
    ("vol04_4紫水晶的怪异传说", 2509, "琉璃", "紫水晶附身者"): None,
    ("vol05_5磷灰石的怠惰现象", 786, "妃", "创作了《磷灰石的怠惰现象》"): {
        "predicate": "行为",
        "value": "写作《磷灰石的怠惰现象》",
        "title": "妃写作《磷灰石的怠惰现象》",
        "summary": "妃写下了记录幻想图书馆成员幸福生活的《磷灰石的怠惰现象》。",
    },
    ("vol05_5磷灰石的怠惰现象", 826, "妃", "记录了关于幻想图书馆成员的妄想"): {
        "predicate": "行为",
        "value": "记录幻想图书馆成员的妄想",
        "title": "妃记录幻想图书馆成员的妄想",
        "summary": "妃在日记本中记录了关于幻想图书馆成员的幸福妄想。",
    },
    ("vol05_5磷灰石的怠惰现象", 949, "夜子", "喜欢上琉璃"): None,
    ("vol05_5磷灰石的怠惰现象", 1388, "魔法之书", "与失踪事件有关"): {
        "value": "被推测可能与失踪事件有关",
        "title": "魔法之书与失踪事件的关联猜测",
        "summary": "彼方推测失踪事件可能与魔法之书有关，但只给出约百分之五的可能性。",
        "reality_status": "character_claim",
    },
    ("vol06_6芙蓉石的长年隔绝", 658, "夜子", "图书馆管理员"): None,
    ("vol06_6芙蓉石的长年隔绝", 1059, "琉璃", "抽签顺序第三"): None,
    ("vol06_6芙蓉石的终焉轮回", 203, "理央", "琉璃的眷属"): None,
    ("vol06_6芙蓉石的终焉轮回", 374, "夜子", "理央的替代者"): None,
    ("vol06_6芙蓉石的终焉轮回", 793, "琉璃", "眷属"): None,
    ("vol06_6芙蓉石的终焉轮回", 1465, "琉璃", "送给理央日记本"): {
        "predicate": "行为",
        "value": "赠送日记本给理央",
        "title": "琉璃送给理央日记本",
        "summary": "琉璃把未使用的日记本作为礼物送给理央。",
    },
    ("vol06_6芙蓉石的终焉轮回", 1577, "琉璃", "帮助理央克服记忆障碍"): {
        "predicate": "行为",
        "value": "帮助理央面对记忆障碍",
        "title": "琉璃帮助理央面对记忆障碍",
        "summary": "琉璃尝试帮助理央面对反复发生的记忆障碍。",
    },
    ("vol06_6芙蓉石的终焉轮回", 1405, "琉璃", "记忆障碍"): {
        "subject": "理央",
        "title": "理央的记忆障碍",
        "summary": "彼方指出理央存在记忆障碍，琉璃试图像平常一样对待理央。",
    },
    ("vol06_6芙蓉石的长年隔绝", 1581, "琉璃", "吸血鬼最后幸存者"): None,
    ("vol06_6芙蓉石的长年隔绝", 2604, "琉璃", "吸血鬼幸存者"): None,
    ("vol06_6芙蓉石的长年隔绝", 3078, "暗子", "大小姐的母亲安排的侍女"): None,
    ("vol07_7黑珍珠的求爱信号", 1047, "暗子", "琉璃的资助者"): None,
    ("vol07_7黑珍珠的求爱信号", 549, "岬", "与夜子有关系"): None,
    ("vol07_7黑珍珠的求爱信号", 1353, "克丽索贝莉露", "能复活死人的书"): None,
    ("vol07_7黑珍珠的求爱信号", 2417, "妃", "魔法使"): None,
    ("vol08_8萤石的怠惰现象", 53, "妃", "是伪造品"): {
        "predicate": "状态",
        "title": "妃是伪造品",
        "summary": "妃意识到自己是以原本的月社妃为原作制造的伪造存在。",
    },
    ("vol08_8萤石的怠惰现象", 567, "琉璃", "月社的现任成员"): None,
    ("vol08_8萤石的怠惰现象", 567, "彼方", "月社的现任成员"): None,
    # 这里的“女孩子”是妃自称，模型把同一窗口中的叙述者琉璃错配成主体。
    ("vol08_8萤石的怠惰现象", 808, "琉璃", "女孩子"): None,
    ("vol08_8萤石的怠惰现象", 1301, "妃", "自杀"): {
        "predicate": "设定",
        "value": "完成职责后自杀",
        "title": "《月社妃》对妃的结局设定",
        "summary": "《月社妃》给妃的设定是完成让琉璃失恋的职责后自杀；这是故事中的设定，不是现实人物的行为判断。",
        "line_start": 1290,
        "line_end": 1297,
    },
    ("vol08_8萤石的怠惰现象", 1390, "妃", "失恋"): None,
    ("vol08_8萤石的怠惰现象", 1402, "妃", "失忆"): None,
    ("vol08_8萤石的怠惰现象", 189, "妃", "失忆"): {
        "predicate": "经历",
        "value": "与萤分离有关的记忆变得模糊",
        "title": "妃关于萤的记忆变得模糊",
        "summary": "妃回忆起被要求扔掉萤后，关于这段经历的记忆变得模糊。",
    },
    ("vol08_8萤石的时空残影", 19, "琉璃", "琉璃对小说有娱乐态度"): {
        "predicate": "偏好",
        "value": "以娱乐方式阅读小说",
        "title": "琉璃以娱乐方式阅读小说",
        "summary": "琉璃认为小说应更多作为娱乐来享受。",
    },
    ("vol08_8萤石的时空残影", 223, "妃", "魔法之书的作者"): None,
    ("vol08_8萤石的时空残影", 617, "妃", "伏见理央仿制品"): None,
    ("vol08_8萤石的时空残影", 691, "妃", "魔法之书创造"): {
        "predicate": "状态",
        "value": "由魔法之书创造的存在",
        "title": "妃由魔法之书创造",
        "summary": "妃说明自己是由魔法之书以月社妃为原作创造的虚构存在。",
    },
    ("vol08_8萤石的时空残影", 931, "妃", "由遊行寺家创造"): {
        "predicate": "状态",
        "value": "被遊行寺家变为纸上存在",
        "title": "妃被遊行寺家变为纸上存在",
        "summary": "原文说妃因遊行寺家的需要而被利己地创造出来，强调家族需求而非具体创造者。",
    },
    ("vol08_8萤石的时空残影", 957, "琉璃", "跑腿"): {
        "predicate": "行为",
        "value": "替妃跑腿",
        "title": "琉璃替妃跑腿",
        "summary": "琉璃在妃的要求下替她处理跑腿事务。",
    },
    ("vol08_8萤石的时空残影", 1794, "妃", "冒牌货"): {
        "predicate": "状态",
        "title": "妃是冒牌货",
        "summary": "妃承认自己是与真正月社妃不同的冒牌存在。",
    },
    ("vol08_8萤石的时空残影", 1600, "汀", "喜欢上了琉璃"): {
        "predicate": "偏好",
        "value": "喜欢妃",
        "title": "汀喜欢妃",
        "summary": "汀对妃抱有恋爱感情。",
    },
    ("vol08_8萤石的时空残影", 165, "暗子", "魔法之书的作者"): {
        "predicate": "经历",
        "value": "创作《月社妃》",
        "title": "暗子创作《月社妃》",
        "summary": "夜子认为《月社妃》的作者是暗子，并指出其他魔法之书多由遊行寺家的先人写成。",
    },
    ("vol09_9白珍珠的泡沫爱慕", 22, "妃", "魔法之书的作者"): None,
    ("vol09_9白珍珠的泡沫爱慕", 94, "暗子", "魔法之书的作者"): None,
    ("vol09_9白珍珠的泡沫爱慕", 1623, "克丽索贝莉露", "可能是魔法使"): {
        "title": "克丽索贝莉露的身份猜测",
        "summary": "角色推测克丽索贝莉露可能是魔法使，也可能只是纸上存在。",
        "reality_status": "character_claim",
    },
    ("vol09_9白珍珠的泡沫爱慕", 1690, "琉璃", "月社成员"): None,
    ("vol09_9绿幽灵水晶的命运连锁", 1690, "琉璃", "月社成员"): None,
    ("vol09_9绿幽灵水晶的命运连锁", 241, "琉璃", "遊行寺家的一员"): None,
    # L519 只是决定告白；本段随后因夜子不在书房而寻找她，尚未告白。
    ("vol09_9绿幽灵水晶的命运连锁", 519, "琉璃", "告白失败"): None,
    # L508 叙述的是馆长让夜子忘记他人的告白，琉璃保留了被扭曲的记忆，
    # 不能把“本该忘记”概括成琉璃实际失忆。
    ("vol10_10黑曜石的因果目录", 508, "琉璃", "失忆"): None,
    # L449 只是奏提醒“现实与记忆不协调”，琉璃当场否认自己会忘记岬。
    ("vol11_11黑玛瑙的不在证明", 449, "琉璃", "失忆"): None,
    ("vol10_10黑曜石的因果目录", 776, "夜子", "妃的好友"): None,
    ("vol10_10黑曜石的因果目录", 39, "暗子", "遊行寺家前当家之子"): None,
    ("vol11_11黑玛瑙的不在证明", 1211, "琉璃", "收到彼方的信息"): {
        "predicate": "行为",
        "value": "收到彼方传达的信息",
        "title": "琉璃收到彼方的信息",
        "summary": "琉璃打开了由彼方传达、邀请他前往教堂的信息。",
    },
    ("vol11_11黑玛瑙的不在证明", 1246, "琉璃", "暗子的仿制品"): {
        "predicate": "状态",
        "value": "暗子创造的仿制品",
        "title": "琉璃是暗子创造的仿制品",
        "summary": "琉璃说明自己是遊行寺暗子创造的仿制品。",
    },
    ("vol11_11黑玛瑙的不在证明", 1574, "夜子", "失忆"): None,
    ("vol11_11黑玛瑙的不在证明", 714, "夜子", "开始对彼方产生好感"): None,
    ("vol11_11黑玛瑙的不在证明", 1299, "琉璃", "被告白对象"): {
        "predicate": "经历",
        "value": "被彼方告白",
        "title": "琉璃被彼方告白",
        "summary": "琉璃得知彼方准备在教堂向自己告白。",
    },
    ("vol11_11黑玛瑙的不在证明", 1042, "理央", "想和琉璃共度时光"): {
        "predicate": "行为",
        "value": "提议与琉璃玩投接球",
        "title": "理央提议与琉璃玩投接球",
        "summary": "理央向琉璃提议玩投接球，因找不到球而改为在阳光下互拍手掌。",
    },
    ("vol11_11黑玛瑙的不在证明", 1091, "理央", "想增加吸血鬼眷属"): None,
    ("vol11_11黑玛瑙的不在证明", 1115, "理央", "想守望琉璃的幸福"): {
        "predicate": "目标",
        "value": "在一旁守望琉璃的幸福",
        "title": "理央决定守望琉璃的幸福",
        "summary": "理央决定不成为琉璃的恋人，而是在一旁守望琉璃的幸福。",
    },
    ("vol11_11黑玛瑙的不在证明", 1132, "彼方", "拜托理央把信交给琉璃"): {
        "subject": "理央",
        "predicate": "行为",
        "value": "代彼方把信交给琉璃",
        "title": "理央代彼方把信交给琉璃",
        "summary": "理央按照彼方的请求，把彼方写给琉璃的信交给了琉璃。",
    },
    ("vol12_12青金石的幻想图书馆", 1671, "汀", "拯救者"): None,
    ("vol12_12青金石的幻想图书馆", 1688, "克丽索贝莉露", "加害者"): {
        "predicate": "目标",
        "value": "以自己的方式讲述夜子的幸福并加害夜子",
        "title": "克丽索贝莉露决心加害夜子",
        "summary": "克丽索贝莉露决定以自己的方式讲述夜子的幸福，并准备加害夜子。",
    },
    ("vol05_5磷灰石的怠惰现象", 2033, "本城岬", "拿走钥匙扣"): {
        "subject": "岬",
        "value": "拿走钥匙扣",
        "title": "岬拿走钥匙扣",
        "summary": "岬从琉璃手中拿走了钥匙扣。",
    },
    ("vol11_11黑玛瑙的不在证明", 429, "奏", "琉璃的老师"): None,
    ("vol12_12青金石的幻想图书馆", 519, "琉璃", "爱彼方的一切"): None,
    ("vol12_12青金石的幻想图书馆", 1626, "夜子", "失忆"): None,
    ("vol12_12青金石的幻想图书馆", 434, "克丽索贝莉露", "创造魔法之书"): None,
    ("epilogue_bonus", 100, "萤", "《纸上魔法使》的作者"): None,
}

# 事件卡的少数高置信语义修正。它们只改写模型摘要，不改 evidence 行号；
# 目的在于去掉“推测=事实”“拒绝=恋爱”“火海终局=决定自杀”等过度结论。
EVENT_CONTENT_REWRITES: dict[tuple[str, int], dict[str, Any]] = {
    ("vol01_1翡翠的排挤原理", 1866): {
        "title": "琉璃救下彼方",
        "summary": "琉璃在十字路口救下险些被卡车撞到的彼方，并抱她去可以处理伤口的地方。",
        "participants": ["琉璃", "彼方"],
        "causes": ["彼方没有注意到驶来的卡车"],
        "outcomes": ["琉璃救下彼方", "彼方因跌倒受到擦伤和扭伤"],
    },
    ("vol01_1翡翠的排挤原理", 1965): {
        "title": "琉璃将彼方带回图书馆",
        "summary": "琉璃在车祸后救助受伤的彼方，将她带回图书馆，并请理央为她处理伤口。",
        "participants": ["琉璃", "彼方", "理央", "夜子"],
        "causes": ["琉璃救下险些被卡车撞到的彼方"],
        "outcomes": ["琉璃将彼方带回图书馆", "理央为彼方处理伤口", "夜子允许彼方暂时留在一楼"],
    },
    ("vol02_2红宝石的天作之合", 21): {
        "title": "暗子向夜子道歉",
        "summary": "暗子回忆夜子出生后遭遇的不幸，并反复向女儿道歉。",
        "participants": ["暗子", "夜子"],
        "causes": ["暗子为让夜子出生并遭遇不幸而悔恨"],
        "outcomes": ["暗子向夜子道歉"],
    },
    ("vol07_7黑珍珠的求爱信号", 1262): {
        "title": "彼方邀请夜子共进晚餐",
        "summary": "彼方叫夜子一起吃晚餐；夜子起初想读完小说再吃，最后接受邀请前往用餐。",
        "participants": ["彼方", "夜子"],
        "causes": ["彼方希望与夜子一起吃饭"],
        "outcomes": ["夜子接受彼方的晚餐邀请"],
    },
    ("vol09_9绿幽灵水晶的命运连锁", 519): {
        "title": "琉璃寻找夜子并准备告白",
        "summary": "琉璃决定向夜子告白，发现她不在书房后在图书馆寻找，最终在外沿找到夜子并开始交谈；该证据段尚未进入正式告白。",
        "participants": ["琉璃", "夜子"],
        "causes": ["琉璃决定向夜子传达自己的感情"],
        "outcomes": ["琉璃找到夜子并开始交谈", "告白尚未在该证据段发生"],
    },
    ("vol09_9绿幽灵水晶的命运连锁", 387): {
        "title": "琉璃确认夜子未曾喜欢自己",
        "summary": "琉璃询问夜子是否有过初吻以及是否曾把自己当作异性；夜子否认曾对他产生恋爱感情，琉璃说明自己以朋友身份珍惜她。",
        "participants": ["琉璃", "夜子"],
        "causes": ["琉璃想确认夜子对自己的感情"],
        "outcomes": ["夜子否认曾喜欢琉璃", "琉璃表示以朋友身份珍惜夜子"],
    },
    ("vol10_10黑曜石的因果目录", 508): {
        "title": "暗子改写告白相关记忆",
        "summary": "暗子为缓解夜子目睹琉璃被告白后的痛苦，使用不完整的魔法之书让夜子忘记该告白；琉璃保留了被扭曲的记忆，将对象误认为妃。",
        "participants": ["暗子", "夜子", "琉璃", "妃"],
        "causes": ["夜子因目睹琉璃被告白而濒临崩溃", "暗子想让夜子忘记这次告白"],
        "outcomes": ["夜子忘记该告白", "琉璃保留并扭曲了相关记忆", "琉璃误以为妃曾向自己告白"],
    },
    ("vol09_9白珍珠的泡沫爱慕", 486): {
        "title": "暗子回忆夜子的出生与家族排斥",
        "summary": "暗子回忆夜子出生后因白发红眼被遊行寺家视为禁忌，以及自己和丈夫保护夜子的经历。",
        "participants": ["暗子", "夜子"],
        "causes": ["遊行寺家对魔法使传说和夜子外表的恐惧"],
        "outcomes": ["暗子和丈夫保护夜子", "夜子被限制在宅院内生活"],
    },
    ("vol09_9白珍珠的泡沫爱慕", 493): {
        "title": "暗子使用魔法之书保护夜子",
        "summary": "夜子遭到迫害后，暗子打开魔法之书使施害者消失，并反复以此保护夜子，结果使诅咒谣言加剧。",
        "participants": ["暗子", "夜子"],
        "causes": ["夜子遭受佣人和家族成员迫害", "暗子想保护夜子"],
        "outcomes": ["伤害夜子的人消失", "诅咒谣言加剧"],
    },
    ("vol09_9白珍珠的泡沫爱慕", 538): {
        "title": "夜子童年遭受囚禁与迫害",
        "summary": "夜子因家族排斥被限制在宅院内，后来又遭佣人和家族成员长期辱骂、拉扯头发并公开羞辱。",
        "participants": ["暗子", "夜子"],
        "causes": ["遊行寺家对夜子的排斥", "佣人和家族成员对夜子的迫害"],
        "outcomes": ["夜子被囚禁并与外界隔离", "夜子的心理受到严重伤害"],
    },
    ("vol09_9白珍珠的泡沫爱慕", 1202): {
        "title": "暗子设计替代性恋情",
        "summary": "暗子为了不让夜子爱上琉璃，计划抹去两人的恋慕并把琉璃与妃的回忆替换为恋人关系。",
        "participants": ["暗子", "琉璃", "妃", "夜子"],
        "causes": ["暗子想独占夜子", "暗子想拆散夜子与琉璃的关系"],
        "outcomes": ["暗子计划用妃作为琉璃记忆中的替代对象", "琉璃与妃被设定为恋人关系"],
    },
    ("vol08_8萤石的时空残影", 259): {
        "title": "妃以纸上存在形式再次出现",
        "summary": "妃作为与已故月社妃相似的纸上存在再次出现，众人围绕这是否算复生及其与原本妃的差异展开讨论。",
        "participants": ["妃", "理央", "夜子", "琉璃", "汀", "彼方"],
        "outcomes": ["妃以纸上存在形式回到幻想图书馆", "众人意识到她并非原本的月社妃"],
    },
    ("vol10_10黑曜石的因果目录", 774): {
        "title": "妃自杀",
        "summary": "克丽索贝莉露认为妃继续作为夜子的好友会阻碍夜子编写幸福，妃随后自杀。",
        "participants": ["妃", "克丽索贝莉露", "夜子"],
        "causes": ["克丽索贝莉露认为妃会阻碍夜子的幸福"],
        "outcomes": ["妃自杀"],
    },
    ("vol11_11黑玛瑙的不在证明", 225): {
        "title": "夜子送书给琉璃",
        "summary": "夜子在夜间把《蓝宝石的存在证明》借给琉璃。",
        "participants": ["夜子", "琉璃"],
        "outcomes": ["琉璃从夜子手中接过《蓝宝石的存在证明》"],
    },
    ("vol03_3蓝宝石的存在证明", 1012): {
        "title": "妃与琉璃讨论魔法之书",
        "summary": "妃与琉璃谈论魔法之书及其引发的事件；妃承认自己对与魔法之书有关的一切都感兴趣。",
        "participants": ["妃", "琉璃"],
        "causes": ["妃对魔法之书相关事件感兴趣"],
        "outcomes": ["妃表示自己对魔法之书的一切都感兴趣", "琉璃继续处理夜子的委托"],
    },
    ("vol05_5磷灰石的怠惰现象", 2718): {
        "title": "琉璃与奏协商汀和魔法之书",
        "summary": "琉璃与因汀杀书而愤怒的奏交谈，试图协商如何处理汀的复仇和魔法之书；原文没有岬介入调解。",
        "participants": ["琉璃", "奏"],
        "causes": ["奏因汀的行动而愤怒并担心他的安危", "琉璃希望平息奏的怒火"],
        "outcomes": ["奏坚持要求汀放弃危险行动", "琉璃表示不会抛弃汀，但不承担阻止他的职责"],
    },
    ("vol03_3蓝宝石的存在证明", 2827): {
        "title": "蓝宝石影响下妃与琉璃的亲密互动",
        "summary": "蓝宝石影响下，妃对琉璃的爱意被放大并主动要求亲吻等亲密互动；琉璃意识到行为受到蓝宝石影响并试图制止。",
        "participants": ["妃", "琉璃", "理央", "夜子"],
        "causes": ["蓝宝石的影响"],
        "outcomes": ["妃与琉璃继续以恋人相处并讨论如何表达感情"],
    },
    ("vol03_3蓝宝石的存在证明", 3155): {
        "title": "汀追问妃的恋爱关系",
        "summary": "汀回忆与一名女子饮酒的片段，怀疑对方可能是妃，并追问妃是否曾与人交往；原文没有确认汀与妃曾是恋人。",
        "participants": ["汀", "妃", "琉璃"],
        "causes": ["蓝宝石造成的记忆缺口"],
        "outcomes": ["汀未能确认饮酒对象的身份，琉璃以自己与妃交往相告"],
    },
    ("vol08_8萤石的怠惰现象", 1168): {
        "title": "克丽索贝莉露对妃的选择产生不安",
        "summary": "克丽索贝莉露想起妃曾作出出乎意料的选择而产生不安，决定确认妃对重要事物的选择。",
        "participants": ["克丽索贝莉露", "妃"],
        "causes": ["克丽索贝莉露的不安和对妃往事的回想"],
        "outcomes": ["克丽索贝莉露决定确认妃的选择"],
    },
    ("vol08_8萤石的怠惰现象", 1366): {
        "title": "妃与琉璃在火海中迎来终局",
        "summary": "妃与琉璃在无法逃离的火海中互诉心意，并在故事安排的悲剧终局一同死去。",
        "participants": ["妃", "琉璃"],
        "causes": ["无法逃离的火海", "故事安排的悲剧终局"],
        "outcomes": ["妃与琉璃死亡"],
    },
    ("vol11_11黑玛瑙的不在证明", 714): {
        "title": "夜子谈及恋爱话题",
        "summary": "彼方与夜子散步时谈论喜欢的人；夜子否认自己有喜欢的人，原文未确认她喜欢彼方。",
        "participants": ["夜子", "彼方"],
        "causes": ["彼方发起恋爱话题"],
        "outcomes": ["夜子没有承认喜欢的对象"],
    },
    ("vol13_13璀璨的紫翠玉", 20): {
        "title": "克丽索贝莉露接受夜子的选择",
        "summary": "克丽索贝莉露意识到夜子可能在现实中获得幸福，接受自己作为阻碍被撕毁并向夜子告别。",
        "participants": ["克丽索贝莉露", "夜子"],
        "causes": ["克丽索贝莉露意识到夜子拥有现实中的幸福可能"],
        "outcomes": ["克丽索贝莉露接受被夜子撕毁并告别"],
    },
    ("vol13_13璀璨的紫翠玉", 1045): {
        "title": "汀将克丽索贝莉露带入日常",
        "summary": "汀把旁观的克丽索贝莉露推入大家的日常，理央和彼方随后接待她；她与夜子因外貌相似而被拿来比较。",
        "participants": ["汀", "克丽索贝莉露", "理央", "彼方", "夜子"],
        "causes": ["汀把克丽索贝莉露强行带入大家的日常"],
        "outcomes": ["理央和彼方接待克丽索贝莉露", "克丽索贝莉露与夜子因外貌相似而被比较"],
    },
    ("vol09_9白珍珠的泡沫爱慕", 760): {
        "title": "琉璃与夜子的初次见面",
        "summary": "琉璃在幻想图书馆初次见到夜子，被她的外貌和读书姿态强烈吸引；两人开始互相认识。",
        "participants": ["琉璃", "夜子"],
        "causes": ["汀邀请琉璃到幻想图书馆做客"],
        "outcomes": ["琉璃对夜子留下强烈的第一印象，两人开始交流"],
    },
    ("vol11_11黑玛瑙的不在证明", 1152): {
        "title": "理央接受未能实现的恋情",
        "summary": "理央确认自己对琉璃的感情无法得到回应，并为未能直接告白而感到失恋；她接受结果，决定继续作为幻想图书馆的勤杂工。",
        "participants": ["理央", "琉璃"],
        "causes": ["理央确认琉璃不会回应自己的感情"],
        "outcomes": ["理央未能直接告白，接受恋情结束并继续为大家带来笑容"],
    },
    ("vol12_12青金石的幻想图书馆", 332): {
        "title": "琉璃邀请夜子回到大家身边",
        "summary": "琉璃在夜子房间外表示，与夜子一起度过的时间并非虚假，并邀请她回去继续和大家生活；克丽索贝莉露随后揭示琉璃并不属于现实。",
        "participants": ["琉璃", "夜子", "克丽索贝莉露"],
        "causes": ["夜子因彼方的告白和幻想图书馆的真相而动摇"],
        "outcomes": ["琉璃邀请夜子回到大家身边", "克丽索贝莉露揭示琉璃是纸上存在"],
    },
    ("vol12_12青金石的幻想图书馆", 1516): {
        "title": "妃向琉璃说明过去的感情",
        "summary": "妃向琉璃说明自己曾经喜欢他、接受无法成为第一位，并劝他回到幻想图书馆；原文没有说妃爱彼方。",
        "participants": ["妃", "琉璃"],
        "causes": ["妃决定在这段关系即将结束前传达自己的感情"],
        "outcomes": ["妃说明自己曾经喜欢琉璃", "妃将琉璃引回幻想图书馆"],
    },
    ("epilogue_bonus", 698): {
        "title": "妃将萤交给彼方饲养",
        "summary": "由于妃的家庭原因和母亲的命令，妃将萤交给彼方饲养；彼方成为萤的新饲主，妃与萤的日常相处因此疏远。",
        "participants": ["妃", "萤", "彼方"],
        "causes": ["妃的家庭原因", "妃的母亲下令扔掉萤"],
        "outcomes": ["妃将萤交给彼方", "彼方成为萤的饲主", "妃与萤的日常相处疏远"],
    },
}

# 关系卡必须有与标签相符的原文信号；普通“提到/影响/照顾/同住”不够构成创造者或恋人关系。
RELATION_SIGNAL_PATTERNS: dict[str, tuple[str, ...]] = {
    "创造者": ("创造", "制造", "再造", "塑造", "诞生", "改写", "写下"),
    "恋人": ("恋人", "爱上", "相爱", "恋爱", "交往", "告白", "女朋友", "男朋友", "喜欢"),
    "前恋人": ("前恋人", "曾经相爱", "分手", "旧情人"),
    "暗恋对象": ("暗恋", "单恋", "喜欢", "爱上", "心仪"),
    "初恋对象": ("初恋",),
    "朋友": ("朋友", "好友", "同伴", "友人"),
    "儿时好友": ("儿时好友", "儿时玩伴", "青梅竹马"),
    "敌对": ("敌对", "憎恨", "仇恨", "讨厌", "迫害", "敌人"),
    "憎恨对象": ("憎恨", "仇恨", "讨厌"),
    "迫害者": ("迫害", "欺负", "排挤", "加害"),
    "父亲": ("父亲", "爸爸", "父母"),
    "母亲": ("母亲", "妈妈", "母亲", "父母"),
    "哥哥": ("哥哥", "兄长", "兄妹"),
    "姐姐": ("姐姐", "姐妹"),
    "弟弟": ("弟弟", "兄弟"),
    "妹妹": ("妹妹", "兄妹", "姐妹"),
    "丈夫": ("丈夫", "老公", "夫妻"),
    "妻子": ("妻子", "老婆", "夫妻"),
    "主人": ("主人", "饲主"),
    "照顾对象": ("照顾", "照料", "女仆", "招待"),
    "依赖对象": ("依赖", "依靠", "依存"),
    "搭档": ("搭档", "合作", "同伴"),
}
RELATION_STRONG_SIGNAL_PATTERNS: dict[str, tuple[str, ...]] = {
    "创造者": ("创造了", "创造出的", "创造的", "创造者", "写下"),
    "恋人": ("恋人关系", "是恋人", "成为恋人", "开始交往", "男朋友", "女朋友", "两人相爱", "相爱", "恋爱关系", "恋情"),
    "前恋人": ("前恋人", "曾有过恋情", "分手", "旧情人"),
    "暗恋对象": ("暗恋", "单恋", "心仪"),
    "初恋对象": ("初恋",),
}

# 这些卡片的证据位置已人工对照原文，关系标签是 7B 将同一段中的称谓、
# 行为或故事设定误拼成关系。按单元、原始证据起点和三元组限定，避免误伤
# 同一人物对在其他故事段中确实成立的关系。
FALSE_RELATION_SPECS: frozenset[tuple[str, int, str, str, str]] = frozenset(
    {
        ("vol01_1翡翠的排挤原理", 1000, "彼方", "敌对", "琉璃"),
        ("vol01_1翡翠的排挤原理", 875, "彼方", "敌对", "琉璃"),
        ("vol01_1翡翠的排挤原理", 1743, "岬", "敌对", "彼方"),
        ("vol02_2红宝石的天作之合", 2505, "夜子", "朋友", "琉璃"),
        ("vol02_2红宝石的天作之合", 2411, "夜子", "朋友", "琉璃"),
        ("vol02_2红宝石的天作之合", 778, "夜子", "朋友", "汀"),
        ("vol06_6芙蓉石的长年隔绝", 3078, "理央", "母亲", "大小姐"),
        ("vol06_6芙蓉石的长年隔绝", 3073, "大小姐", "母亲", "理央"),
        ("vol06_6芙蓉石的终焉轮回", 1316, "理央", "母亲", "夜子"),
        ("vol06_6芙蓉石的终焉轮回", 1264, "夜子", "母亲", "理央"),
        ("vol07_7黑珍珠的求爱信号", 2491, "理央", "创造者", "克丽索贝莉露"),
        ("vol07_7黑珍珠的求爱信号", 2412, "克丽索贝莉露", "创造者", "理央"),
        ("vol07_7黑珍珠的求爱信号", 2412, "理央", "创造者", "克丽索贝莉露"),
        ("vol08_8萤石的怠惰现象", 325, "夜子", "母亲", "理央"),
        ("vol10_10黑曜石的因果目录", 351, "夜子", "母亲", "理央"),
        ("vol10_10黑曜石的因果目录", 345, "夜子", "母亲", "理央"),
        # 这些证据窗口分别是阶段总结、恋情建议或无人物名的叙述，
        # 不能支撑对应的长期关系标签。
        ("vol05_5磷灰石的怠惰现象", 854, "夜子", "朋友", "琉璃"),
        ("vol06_6芙蓉石的终焉轮回", 1018, "彼方", "恋人", "琉璃"),
        ("vol06_6芙蓉石的终焉轮回", 1396, "彼方", "照顾对象", "理央"),
        ("vol11_11黑玛瑙的不在证明", 60, "琉璃", "依赖对象", "夜子"),
        ("epilogue_bonus", 311, "岬", "朋友", "理央"),
        ("vol07_7黑珍珠的求爱信号", 1020, "奏", "朋友", "琉璃"),
        ("vol07_7黑珍珠的求爱信号", 460, "岬", "敌对", "彼方"),
        ("vol07_7黑珍珠的求爱信号", 364, "彼方", "依赖对象", "汀"),
        ("vol09_9白珍珠的泡沫爱慕", 141, "夜子", "敌对", "妃"),
        ("vol09_9绿幽灵水晶的命运连锁", 1735, "夜子", "敌对", "琉璃"),
        ("vol06_6芙蓉石的终焉轮回", 369, "夜子", "敌对", "琉璃"),
        ("vol11_11黑玛瑙的不在证明", 201, "克丽索贝莉露", "敌对", "琉璃"),
        ("vol11_11黑玛瑙的不在证明", 1656, "夜子", "敌对", "琉璃"),
        ("vol12_12青金石的幻想图书馆", 779, "妃", "朋友", "琉璃"),
        ("vol12_12青金石的幻想图书馆", 779, "理央", "朋友", "琉璃"),
        ("vol12_12青金石的幻想图书馆", 779, "汀", "朋友", "琉璃"),
        ("vol12_12青金石的幻想图书馆", 786, "夜子", "朋友", "琉璃"),
    }
)

# 已按原文确认的非对称关系方向。关系标签的含义是“subject 的 relation 是
# target”，因此“暗子创造了妃”应规范为“妃的创造者是暗子”。
RELATION_DIRECTION_REWRITES: dict[tuple[str, int, str, str, str], tuple[str, str, str, str]] = {
    ("vol08_8萤石的时空残影", 706, "妃", "创造者", "暗子"): (
        "妃",
        "暗子",
        "妃的创造者是暗子",
        "暗子创造了妃这个仿制存在，因此妃的创造者是暗子。",
    ),
    ("vol11_11黑玛瑙的不在证明", 36, "琉璃", "创造者", "暗子"): (
        "琉璃",
        "暗子",
        "琉璃的创造者是暗子",
        "暗子创造了琉璃这个仿制品，因此琉璃的创造者是暗子。",
    ),
    ("vol03_3蓝宝石的存在证明", 190, "夜子", "主人", "理央"): (
        "理央",
        "夜子",
        "夜子是理央的主人",
        "夜子是理央的主人，理央负责照顾夜子的生活。",
    ),
    ("vol09_9绿幽灵水晶的命运连锁", 1026, "夜子", "母亲", "妈妈"): (
        "夜子",
        "暗子",
        "夜子的母亲",
        "暗子是夜子的母亲，夜子在内心称她为妈妈。",
    ),
}

# 关系标签或方向需要同时修正时使用。这里不依赖模型摘要中的叙述顺序，
# 只对已核对的故事单元、证据锚点和原始三元组生效。
RELATION_CONTENT_REWRITES: dict[tuple[str, int, str, str, str], tuple[str, str, str, str, str]] = {
    ("vol05_5磷灰石的怠惰现象", 919, "夜子", "敌对", "琉璃"): (
        "夜子",
        "憎恨对象",
        "琉璃",
        "夜子憎恨琉璃",
        "夜子在这一阶段仍将琉璃视为讨厌的对象。",
    ),
    ("vol09_9绿幽灵水晶的命运连锁", 291, "夜子", "朋友", "琉璃"): (
        "琉璃",
        "儿时好友",
        "夜子",
        "琉璃和夜子是儿时好友",
        "琉璃称夜子是自己重要的儿时好友。",
    ),
}

# 失败摘要 detail 上限：不得把完整 prompt、模型全文或密钥写进运行状态。
FAILURE_DETAIL_MAX_CHARS = 200

# ---------- 词汇规范表（避免同义词无限分裂） ----------

# fact.predicate 规范表：subject 的 {predicate} 相关事实。
FACT_PREDICATES: tuple[str, ...] = (
    "身份",
    "经历",
    "行为",
    "能力",
    "偏好",
    "状态",
    "归属",
    "设定",
    "目标",
    "死因",
    "外貌",
    "性格",
)

# relation 规范表：语义为「subject 的 {relation} 是 target」。
RELATION_LABELS: tuple[str, ...] = (
    "父亲",
    "母亲",
    "哥哥",
    "姐姐",
    "弟弟",
    "妹妹",
    "儿子",
    "女儿",
    "丈夫",
    "妻子",
    "恋人",
    "前恋人",
    "暗恋对象",
    "初恋对象",
    "朋友",
    "儿时好友",
    "敌对",
    "憎恨对象",
    "迫害者",
    "主人",
    "创造者",
    "照顾对象",
    "依赖对象",
    "搭档",
)

FAMILY_RELATIONS: frozenset[str] = frozenset({"哥哥", "姐姐", "弟弟", "妹妹"})

# 原文明确确认的兄妹方向。模型常把「妹妹」一词附近的叙述者、被提及人物
# 互换；对已知人物对按原文固定方向，避免把称谓误写成另一种关系。
KNOWN_SIBLING_DIRECTIONS: dict[frozenset[str], dict[str, tuple[str, str]]] = {
    frozenset({"琉璃", "妃"}): {
        "琉璃": ("妹妹", "妃"),
        "妃": ("哥哥", "琉璃"),
    },
    frozenset({"夜子", "汀"}): {
        "夜子": ("哥哥", "汀"),
        "汀": ("妹妹", "夜子"),
    },
}

# 对称关系：canonical 化时按人物名排序固定方向，避免「琉璃-恋人-夜子」与
# 「夜子-恋人-琉璃」随机混用。
SYMMETRIC_RELATIONS: frozenset[str] = frozenset({"恋人", "前恋人", "朋友", "儿时好友", "敌对", "搭档"})

# 无名角色称谓：允许出现在 relation 的 subject/target 与 event 的 participants。
# 依据语料核对（出现次数）：加害者/同学/老师/读者/教员/兄长/部长/信徒/水手/
# 班主任/前当家/岛民/校方/医生/警察等书中故事与背景叙述中的通用角色。
ROLE_NOUNS: tuple[str, ...] = (
    "母亲",
    "父亲",
    "妈妈",
    "医者",
    "医生",
    "学生",
    "同学",
    "老师",
    "教员",
    "班主任",
    "校方",
    "加害者",
    "佣人",
    "勤杂工",
    "大小姐",
    "双亲",
    "少女",
    "少年",
    "男孩子",
    "少年的母亲",
    "读者",
    "兄长",
    "部长",
    "信徒",
    "岛民",
    "水手",
    "前当家",
    "警察",
)

# 妃拾养的猫（追加剧本叙述者）：按人物实体参与关系与事件，但非 canonical 人物表成员。
EXTRA_PERSON_NAMES: tuple[str, ...] = ("萤",)

# 冲突检测敏感的 fact predicate：同一 (subject, predicate) 出现多个不同 value 视为潜在冲突。
# 只有同一人物的不同死因天然值得作为潜在冲突分组；“状态”包含多个可同时成立、
# 随时间变化的属性，不能仅因取值不同就判冲突。
CONFLICT_PRONE_FACT_PREDICATES: frozenset[str] = frozenset({"死因"})

# 朋友、敌对、依赖、恋人等关系可能有多个对象，也可能随剧情变化；仅按
# (subject, relation) 多值分组会制造大量伪冲突。这里仅保留通常单值的直系亲属角色。
CONFLICT_PRONE_RELATIONS: frozenset[str] = frozenset({"父亲", "母亲"})

# 仅这些不会随剧情自然变化的知识允许在同一内容层、时间层和现实属性内
# 跨场景合并。状态、行为、经历、目标、死因和事件始终保留。
STABLE_FACT_PREDICATES: frozenset[str] = frozenset({"身份", "外貌", "设定"})
STABLE_RELATIONS: frozenset[str] = frozenset(
    {"父亲", "母亲", "哥哥", "姐姐", "弟弟", "妹妹", "儿子", "女儿", "丈夫", "妻子", "创造者"}
)

DEFAULT_GENERATION_PARAMS: dict[str, int] = {
    "chunk_max_chars": KNOWLEDGE_CHUNK_MAX_CHARS,
    "max_facts_per_scene": MAX_FACTS_PER_SCENE,
    "max_relations_per_scene": MAX_RELATIONS_PER_SCENE,
    "max_events_per_scene": MAX_EVENTS_PER_SCENE,
}

DEFAULT_RUN_NOTES = (
    "P5 知识卡候选运行状态。候选由模型生成，仅供人工审核参考；候选文档至多"
    " needs_review，绝不自动 approved。运行状态绑定 enriched 双文件摘要"
    "（enriched_manifest_sha256/enriched_scenes_sha256/enriched_bundle_sha256），"
    "跨 bundle 恢复或合并会被拒绝。scene_id 创建后不得更换。"
)

DEFAULT_MANIFEST_NOTE = (
    "P5 knowledge candidate run manifest: 记录模型标识、参数、输入 enriched 摘要与计数；"
    "时间戳与调用统计等非确定性信息只记录在本 manifest，不进入候选运行状态或候选文档。"
)


class KnowledgeParseError(ValueError):
    """模型输出解析失败（markdown fence / JSON 非法 / 契约违反 / evidence 越界）。

    error_kind 用于失败摘要分类；异常消息不含 prompt 或模型全文。
    """

    def __init__(self, message: str, *, error_kind: str) -> None:
        super().__init__(message)
        self.error_kind = error_kind


@runtime_checkable
class KnowledgeModelClient(Protocol):
    """模型客户端最小协议：输入 prompt 文本，返回模型原始输出文本。

    领域模块不初始化服务器、不下载模型、不读取环境密钥；
    超时/连接错误由客户端以异常形式抛出（TimeoutError 或其他异常），
    本模块负责把异常收敛为可重试的失败摘要。
    """

    def __call__(self, prompt: str) -> str: ...


# ---------- 人物别名配置 ----------


class AliasConfig(BaseModel):
    """人物 canonical 别名配置（来自 character_aliases.json 的确定性投影）。

    - canonical_names + role_nouns + extra_person_names 构成 relation/event 的
      合法人物名集合（经 aliases 归一后校验）；
    - fact.subject 允许实体名（如「魔法之书」的设定事实），仅要求非空；
    - non_person_terms 为书名/家族名等非人物条目，出现在 relation/event 人物位时拒绝。
    """

    model_config = ConfigDict(extra="forbid")

    canonical_names: list[str] = Field(min_length=1)
    aliases: dict[str, str] = Field(default_factory=dict)
    role_nouns: list[str] = Field(default_factory=lambda: list(ROLE_NOUNS))
    extra_person_names: list[str] = Field(default_factory=lambda: list(EXTRA_PERSON_NAMES))
    non_person_terms: list[str] = Field(default_factory=list)

    @property
    def person_names(self) -> frozenset[str]:
        return frozenset(set(self.canonical_names) | set(self.role_nouns) | set(self.extra_person_names))

    def normalize(self, name: str) -> str:
        return self.aliases.get(name, name)

    def is_person(self, name: str) -> bool:
        return self.normalize(name) in self.person_names


def load_alias_config(path: Path | str) -> AliasConfig:
    """从 character_aliases.json 读取别名配置（文件由 P4D 维护，本阶段只读）。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return AliasConfig(
        canonical_names=list(data["canonical_names"]),
        aliases={str(k): str(v) for k, v in data["aliases"].items()},
    )


# ---------- enriched 场景包输入门禁 ----------


class EnrichedManifestDoc(BaseModel):
    """enriched_manifest.json 契约（P4A write_enriched_scenes 的输出结构）。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    generator: NonEmptyStr
    source_boundary_manifest: SourceManifestRef
    total_scenes: int = Field(ge=1)
    scene_review_status: Literal["approved"]
    note: str = ""


class EnrichedSourceRef(BaseModel):
    """P5 运行状态对 enriched 双文件的稳定绑定（三摘要）。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    scene_review_status: Literal["approved"]
    total_scenes: int = Field(ge=1)
    boundary_bundle_sha256: NonEmptyStr
    enriched_manifest_sha256: NonEmptyStr
    enriched_scenes_sha256: NonEmptyStr
    enriched_bundle_sha256: NonEmptyStr


class EnrichedSceneBundle:
    """加载并通过门禁的 approved enriched 场景包（只读使用）。"""

    def __init__(
        self,
        scenes: list[SceneDocument],
        manifest: EnrichedManifestDoc,
        manifest_digest: str,
        scenes_digest: str,
        bundle_digest: str,
    ) -> None:
        self.scenes = scenes
        self.manifest = manifest
        self.manifest_digest = manifest_digest
        self.scenes_digest = scenes_digest
        self.bundle_digest = bundle_digest
        # (story_unit_id, source_path) -> [(line_start, line_end, scene_id)]：
        # 候选文档反查所属场景（文档自身不含 scene_id 字段；卡片 span 落在场景
        # span 内，需按范围匹配而非起点相等）。
        self._scene_span_index: dict[tuple[str, str], list[tuple[int, int, str]]] = {}
        for scene in scenes:
            self._scene_span_index.setdefault((scene.story.story_unit_id, scene.source.source_path), []).append(
                (scene.source.line_start, scene.source.line_end, scene.id)
            )

    @property
    def source_ref(self) -> EnrichedSourceRef:
        return EnrichedSourceRef(
            schema_version=self.manifest.schema_version,
            scene_review_status=self.manifest.scene_review_status,
            total_scenes=self.manifest.total_scenes,
            boundary_bundle_sha256=self.manifest.source_boundary_manifest.bundle_sha256,
            enriched_manifest_sha256=self.manifest_digest,
            enriched_scenes_sha256=self.scenes_digest,
            enriched_bundle_sha256=self.bundle_digest,
        )


def _enriched_manifest_digest(manifest: EnrichedManifestDoc) -> str:
    canonical = json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _enriched_scenes_digest(scenes: list[SceneDocument]) -> str:
    digest = hashlib.sha256()
    for scene in scenes:
        canonical = json.dumps(scene.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest.update(canonical.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _enriched_bundle_digest(manifest_digest: str, scenes_digest: str) -> str:
    return hashlib.sha256(f"{manifest_digest}:{scenes_digest}".encode()).hexdigest()


def enriched_bundle_integrity_errors(bundle: EnrichedSceneBundle) -> list[str]:
    """重算摘要，拒绝加载后被原地修改或手工拼装的 bundle。"""
    errors: list[str] = []
    if _enriched_manifest_digest(bundle.manifest) != bundle.manifest_digest:
        errors.append("EnrichedSceneBundle.manifest 在加载后被修改，缓存摘要已失效")
    if _enriched_scenes_digest(list(bundle.scenes)) != bundle.scenes_digest:
        errors.append("EnrichedSceneBundle.scenes 在加载后被修改，缓存摘要已失效")
    if _enriched_bundle_digest(bundle.manifest_digest, bundle.scenes_digest) != bundle.bundle_digest:
        errors.append("EnrichedSceneBundle 双文件组合摘要无效")
    return errors


def load_enriched_scene_bundle(scenes_path: Path | str, manifest_path: Path | str) -> EnrichedSceneBundle:
    """approved enriched 场景包输入门禁（本阶段唯一一次加载校验）。

    校验（任何候选生成之前全部完成，任一失败抛 ValueError 且零模型调用）：
    1. 两文件均存在；2. manifest 是合法 JSON 且通过 EnrichedManifestDoc 校验；
    3. scene_review_status == approved；4. scenes 数量 = manifest.total_scenes；
    5. 每个非空行可解析且通过 SceneDocument 校验；6. scene id 唯一；
    7. 全部场景 review_status == approved；
    8. story.viewpoint / story.temporal_scope / reality_status 均已填写（approved 丰富化）；
    9. text 行数与 source span 行数一致（evidence 提取的前提）。
    """
    scenes_path = Path(scenes_path)
    manifest_path = Path(manifest_path)
    if not scenes_path.is_file():
        raise ValueError(f"enriched scenes 文件不存在: {scenes_path}")
    if not manifest_path.is_file():
        raise ValueError(f"enriched manifest 文件不存在: {manifest_path}")
    try:
        manifest = EnrichedManifestDoc.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"enriched manifest 非法: {exc}") from exc
    if manifest.scene_review_status != "approved":
        raise ValueError("enriched manifest scene_review_status 必须为 approved")

    scenes: list[SceneDocument] = []
    seen_ids: set[str] = set()
    for line_no, line in enumerate(scenes_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            scene = SceneDocument.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"enriched scenes 第 {line_no} 行非法: {exc}") from exc
        if scene.id in seen_ids:
            raise ValueError(f"enriched scenes 含重复 scene id: {scene.id}")
        seen_ids.add(scene.id)
        if scene.review_status is not ReviewStatus.approved:
            raise ValueError(f"场景 {scene.id} review_status 非 approved: {scene.review_status}")
        if not scene.story.viewpoint:
            raise ValueError(f"场景 {scene.id} story.viewpoint 未填写")
        if scene.story.temporal_scope is None:
            raise ValueError(f"场景 {scene.id} story.temporal_scope 未填写")
        if scene.reality_status is None:
            raise ValueError(f"场景 {scene.id} reality_status 未填写")
        span_lines = scene.source.line_end - scene.source.line_start + 1
        if len(scene.text.splitlines()) != span_lines:
            raise ValueError(
                f"场景 {scene.id} text 行数({len(scene.text.splitlines())})与 span 行数({span_lines})不一致"
            )
        scenes.append(scene)
    if len(scenes) != manifest.total_scenes:
        raise ValueError(f"enriched scenes 数量({len(scenes)})与 manifest.total_scenes({manifest.total_scenes})不一致")
    manifest_digest = _enriched_manifest_digest(manifest)
    scenes_digest = _enriched_scenes_digest(scenes)
    bundle_digest = _enriched_bundle_digest(manifest_digest, scenes_digest)
    return EnrichedSceneBundle(scenes, manifest, manifest_digest, scenes_digest, bundle_digest)


# ---------- 候选载荷契约（模型输出 JSON 的严格结构） ----------


class _CardSpan(BaseModel):
    """模型输出的 evidence 行号范围（source_path 由系统补齐，原文由系统截取）。"""

    model_config = ConfigDict(extra="forbid")

    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)

    @model_validator(mode="after")
    def _check_line_order(self) -> _CardSpan:
        if self.line_end < self.line_start:
            raise ValueError(f"line_end({self.line_end}) 不得小于 line_start({self.line_start})")
        return self


class _FactItem(_CardSpan):
    model_config = ConfigDict(extra="forbid")

    subject: NonEmptyStr
    predicate: NonEmptyStr
    value: NonEmptyStr
    title: NonEmptyStr
    summary: NonEmptyStr
    reality_status: RealityStatus


class _RelationItem(_CardSpan):
    model_config = ConfigDict(extra="forbid")

    subject: NonEmptyStr
    relation: NonEmptyStr
    target: NonEmptyStr
    title: NonEmptyStr
    summary: NonEmptyStr
    reality_status: RealityStatus


class _EventItem(_CardSpan):
    model_config = ConfigDict(extra="forbid")

    title: NonEmptyStr
    summary: NonEmptyStr
    participants: list[str]
    causes: list[str] = Field(default_factory=list)
    outcomes: list[str] = Field(default_factory=list)
    reality_status: RealityStatus


# 顶层契约键（缺一或多出均整体拒绝）。
_PAYLOAD_KEYS = frozenset({"scene_id", "facts", "relations", "events"})


# ---------- 解析后的规范化候选卡（存于运行状态） ----------


class FactCard(BaseModel):
    """规范化事实卡候选：subject 已按别名归一；span 已限制在分片范围内。"""

    model_config = ConfigDict(extra="forbid")

    subject: NonEmptyStr
    predicate: NonEmptyStr
    value: NonEmptyStr
    title: NonEmptyStr
    summary: NonEmptyStr
    reality_status: RealityStatus
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)

    @property
    def payload_key(self) -> str:
        return f"{self.subject}|{self.predicate}|{self.value}"


class RelationCard(BaseModel):
    """规范化关系卡候选：subject/target 已归一为人物名；对称关系方向已固定。"""

    model_config = ConfigDict(extra="forbid")

    subject: NonEmptyStr
    relation: NonEmptyStr
    target: NonEmptyStr
    title: NonEmptyStr
    summary: NonEmptyStr
    reality_status: RealityStatus
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)

    @property
    def payload_key(self) -> str:
        return f"{self.subject}|{self.relation}|{self.target}"


class EventCard(BaseModel):
    """规范化事件卡候选：participants 已归一为人物名并去重。"""

    model_config = ConfigDict(extra="forbid")

    title: NonEmptyStr
    summary: NonEmptyStr
    participants: list[str]
    causes: list[str] = Field(default_factory=list)
    outcomes: list[str] = Field(default_factory=list)
    reality_status: RealityStatus
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)

    @property
    def payload_key(self) -> str:
        return f"{self.title}|{self.summary}|{','.join(self.participants)}"


class SceneKnowledgeCandidates(BaseModel):
    """单场景的分片归并候选（存于运行状态；排序与去重由 finalize 确定性完成）。

    dropped_invalid：卡片级校验失败被丢弃的数量（如非法 predicate/relation、
    人物位出现实体名或长短语、evidence 越界、卡片额外字段）。结构性错误
    （围栏/非法 JSON/重复键/顶层字段缺失或多余/scene_id 不匹配）仍整体拒绝。
    """

    model_config = ConfigDict(extra="forbid")

    scene_id: NonEmptyStr
    facts: list[FactCard] = Field(default_factory=list)
    relations: list[RelationCard] = Field(default_factory=list)
    events: list[EventCard] = Field(default_factory=list)
    dropped_invalid: int = Field(default=0, ge=0)


# ---------- 运行状态 ----------


class KnowledgeFailureSummary(BaseModel):
    """失败摘要：detail 只保留异常类型名（可能含密钥/URL/prompt 回显的消息不落盘）。"""

    model_config = ConfigDict(extra="forbid")

    error_kind: NonEmptyStr
    detail: str
    attempts: int = Field(ge=1)


class KnowledgeGenerationStatus(str, Enum):  # noqa: UP042  # 沿用项目 (str, Enum) 惯例（Python 3.10 兼容）
    pending = "pending"
    success = "success"
    failed = "failed"


class KnowledgeSceneState(BaseModel):
    """单场景运行状态：success 时 candidates 必须非 None，否则必须为 None。"""

    model_config = ConfigDict(extra="forbid")

    scene_id: NonEmptyStr
    status: KnowledgeGenerationStatus
    attempts: int = Field(default=0, ge=0)
    chunk_count: int = Field(default=1, ge=1)
    candidates: SceneKnowledgeCandidates | None = None
    last_failure: KnowledgeFailureSummary | None = None

    @model_validator(mode="after")
    def _check_state_consistency(self) -> KnowledgeSceneState:
        if self.status is KnowledgeGenerationStatus.success and self.candidates is None:
            raise ValueError(f"场景 {self.scene_id} 状态为 success 但缺少候选")
        if self.status is not KnowledgeGenerationStatus.success and self.candidates is not None:
            raise ValueError(f"场景 {self.scene_id} 非 success 状态不得携带候选")
        return self


class KnowledgeCandidateRunState(BaseModel):
    """P5 候选运行状态（确定性：无时间戳、无随机数；断点续跑的载体）。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    source_enriched: EnrichedSourceRef
    total_source_scenes: int = Field(ge=1)
    model_id: NonEmptyStr
    generation_params: dict[str, int]
    scene_states: list[KnowledgeSceneState]
    notes: str = DEFAULT_RUN_NOTES
    created_by: NonEmptyStr = GENERATOR_ID

    @model_validator(mode="after")
    def _check_state_counts(self) -> KnowledgeCandidateRunState:
        if len(self.scene_states) != self.total_source_scenes:
            raise ValueError(
                f"scene_states 数量({len(self.scene_states)})必须等于 total_source_scenes({self.total_source_scenes})"
            )
        ids = [item.scene_id for item in self.scene_states]
        if len(set(ids)) != len(ids):
            raise ValueError("scene_states 含重复 scene_id")
        return self


class KnowledgeRunResult(BaseModel):
    """一次 generate 调用的结果汇总。"""

    model_config = ConfigDict(extra="forbid")

    new_state: KnowledgeCandidateRunState
    attempted_scene_ids: list[str]
    succeeded_scene_ids: list[str]
    failed_scene_ids: list[str]
    skipped_scene_ids: list[str]
    total_attempts: int = Field(ge=0)


# ---------- prompt 构建 ----------


def chunk_limits(total_chunks: int) -> tuple[int, int, int]:
    """分片提示数量上限：单片段场景用整场景上限，多片时按片均摊（下限保护）。"""
    if total_chunks <= 1:
        return MAX_FACTS_PER_SCENE, MAX_RELATIONS_PER_SCENE, MAX_EVENTS_PER_SCENE
    import math

    facts = max(2, math.ceil(MAX_FACTS_PER_SCENE / total_chunks))
    relations = max(1, math.ceil(MAX_RELATIONS_PER_SCENE / total_chunks))
    events = max(1, math.ceil(MAX_EVENTS_PER_SCENE / total_chunks))
    return facts, relations, events


def _chunk_spans(scene: SceneDocument, *, chunk_max_chars: int) -> list[SourceSpan]:
    """把长场景按字符阈值切为连续分片（按行边界切，至少一片；不重切场景）。

    分片只影响模型调用；候选卡归并回原 scene_id，span 均在场景范围内。
    """
    lines = scene.text.splitlines()
    if not lines:
        return [scene.source]
    spans: list[SourceSpan] = []
    start_idx = 0
    while start_idx < len(lines):
        end_idx = start_idx
        used = 0
        while end_idx < len(lines):
            line_len = len(lines[end_idx]) + 1
            if end_idx > start_idx and used + line_len > chunk_max_chars:
                break
            used += line_len
            end_idx += 1
        spans.append(
            SourceSpan(
                source_path=scene.source.source_path,
                line_start=scene.source.line_start + start_idx,
                line_end=scene.source.line_start + end_idx - 1,
            )
        )
        start_idx = end_idx
    return spans


def build_knowledge_prompt(
    scene: SceneDocument,
    span: SourceSpan | None = None,
    *,
    chunk_index: int | None = None,
    total_chunks: int | None = None,
    alias_config: AliasConfig,
    fact_limit: int = MAX_FACTS_PER_SCENE,
    relation_limit: int = MAX_RELATIONS_PER_SCENE,
    event_limit: int = MAX_EVENTS_PER_SCENE,
) -> str:
    """为场景（或其分片）构建确定性知识卡抽取 prompt。

    - 原文按**绝对行号**逐行展示（`L<n>: <line>`）；
    - 人物名称表、predicate/relation 规范表、数量上限、reality_status 规则显式给出；
    - 输出契约：只输出一个 JSON 对象、禁止 markdown 围栏、evidence 只给行号
      （原文由系统按行号截取，保证 evidence_text 与 source span 一致）。
    """
    target = span if span is not None else scene.source
    lines = scene.text.splitlines()
    offset = target.line_start - scene.source.line_start
    body = "\n".join(
        f"L{target.line_start + i}: {lines[offset + i]}" for i in range(target.line_end - target.line_start + 1)
    )
    canonical = "、".join(alias_config.canonical_names + list(alias_config.extra_person_names))
    roles = "、".join(alias_config.role_nouns)
    alias_pairs = "、".join(f"{k}={v}" for k, v in sorted(alias_config.aliases.items()))
    predicates = " / ".join(FACT_PREDICATES)
    relations = " / ".join(RELATION_LABELS)
    chunk_note = (
        f"- 本片段为场景的第 {chunk_index}/{total_chunks} 片（L{target.line_start}-L{target.line_end}），"
        "只基于本片段原文抽取\n"
        if total_chunks is not None and total_chunks > 1
        else ""
    )
    return f"""你是游戏文本知识卡抽取器。从给定场景片段中抽取对问答和角色记忆有价值的知识卡候选。

## 场景信息
- scene_id: {scene.id}
- 故事单元: {scene.story.story_unit_id}（{scene.story.story_title}）
- 原文行号范围: L{target.line_start}-L{target.line_end}
{chunk_note}- 叙述视角: {scene.story.viewpoint}
- 时间层: {scene.story.temporal_scope.value}
- 场景现实层: {scene.reality_status.value}
- 当前叙事层在场人物: {"、".join(scene.present_characters) or "（无）"}
- 被提及人物: {"、".join(scene.mentioned_characters) or "（无）"}

## 人物名称表（relations 的 subject/target 与 events 的 participants 只能使用这些名称）
{canonical}
无名角色（含书中故事/回忆里的无名人物）可用称谓：{roles}
（全名=别名：{alias_pairs}）
**禁止**：书籍名（如《磷灰石的怠惰现象》）、地点、组织、概念（魔法之书、幻想图书馆、藤壶学园、遊行寺家等）以及长短语描述不得出现在 relations 的 subject/target 与 events 的 participants；「某人写了/创造了某书」这类内容改用 fact 表达（subject=作者，predicate=经历或设定，value=写了《书名》）。

## 抽取范围（只抽对问答和角色记忆有价值的）
- 人物身份与背景；家庭/恋爱/友情/主从/冲突关系
- 重要承诺、欺骗、秘密、创作和魔法设定
- 死亡、失忆、重现、关系变化等关键事件
- 能解释人物动机或后续剧情的事实
- 反复出现且对检索有价值的世界观规则（此时 fact.subject 可用「魔法之书」等实体名）

## 不要抽取
- 无意义动作、普通寒暄、重复修辞、纯气氛描写
- 原文无法支持的推测；同一事实的措辞变体
- 为凑数量而拆出的低价值卡片；没有有价值内容就输出空数组

## 数量上限（本片段）
facts ≤ {fact_limit}，relations ≤ {relation_limit}，events ≤ {event_limit}

## predicate 规范表（fact.predicate 只能从中选一个）
{predicates}
（「状态」只用于人物存在状态——如 死亡/在世/失忆/失踪/闭门不出/化为纸上存在——
不用于心情、愿望或行为；fact.value 保持简短精确（一般 ≤20 字），是信息本身而非整句）

## relation 规范表（语义为「subject 的 {{relation}} 是 target」，方向必须准确；只能用下表的值）
{relations}
（示例：琉璃-妹妹-妃 表示「琉璃的妹妹是妃」；理央-创造者-暗子 表示「理央的创造者是暗子」。
「创造者」只能用于 target 确实创造/再造了 subject 的剧情事实，不能表示提到、影响、
照顾、命令或同住；「恋人」只能用于原文明示成立的双向恋爱关系，单向喜欢或告白应使用
「暗恋对象」，普通亲密、朋友、敌意和兄妹关系都不能标为恋人。亲属标签同样表示
subject 的该亲属是 target，例如「夜子-母亲-暗子」。没有精确匹配的关系就不要输出关系卡。
注意：「行为」「状态」「身份」等是 fact.predicate 表的值，不能作 relation）

## reality_status 规则（只能是 objective / character_claim / inferred / fictional / conflicted / unknown）
- 默认直接填场景现实层的值：{scene.reality_status.value}（没有特别理由不要改）
- 场景叙述中引用的梦境、书中故事、魔法重现里的内容 → fictional
- 人物单方面主张且叙述未确认（如日记、口头断言）→ character_claim
- 场景内多种说法互相冲突 → conflicted

## 原文（绝对行号）
{body}

## evidence 规则
- 只引用直接支持该卡片的最小连续原文范围，不得用整个场景或整个分片充当证据
- fact/relation 最多 {MAX_FACT_RELATION_EVIDENCE_LINES} 行，event 最多 {MAX_EVENT_EVIDENCE_LINES} 行
- 若无法在限制内找到直接证据，就不要输出该卡片

## 输出格式（只输出一个 JSON 对象，禁止 markdown 围栏，禁止注释；evidence 只给行号范围，原文由系统截取）
{{
  "scene_id": "{scene.id}",
  "facts": [
    {{"subject": "人物或实体名", "predicate": "规范表之一", "value": "事实内容", "title": "短标题", "summary": "一句话概括", "reality_status": "objective", "line_start": {target.line_start}, "line_end": {target.line_start}}}
  ],
  "relations": [
    {{"subject": "人物名", "relation": "规范表之一", "target": "人物名", "title": "短标题", "summary": "一句话概括", "reality_status": "objective", "line_start": {target.line_start}, "line_end": {target.line_start}}}
  ],
  "events": [
    {{"title": "事件短标题", "summary": "起因经过结果概括", "participants": ["人物名"], "causes": ["起因"], "outcomes": ["结果"], "reality_status": "objective", "line_start": {target.line_start}, "line_end": {target.line_start}}}
  ]
}}"""


# ---------- 严格 JSON 解析 ----------


def _normalize_person(name: str, alias_config: AliasConfig) -> str:
    normalized = alias_config.normalize(name.strip())
    if not normalized:
        raise KnowledgeParseError("人物名不得为空白字符串", error_kind="schema_violation")
    if normalized in alias_config.non_person_terms:
        raise KnowledgeParseError(f"非人物条目不得出现在人物位: {normalized!r}", error_kind="schema_violation")
    if not alias_config.is_person(normalized):
        raise KnowledgeParseError(f"人物名 {normalized!r} 不在人物名称表中", error_kind="schema_violation")
    return normalized


def _check_span(item: _CardSpan, allowed: SourceSpan, *, max_lines: int) -> None:
    if item.line_start < allowed.line_start or item.line_end > allowed.line_end:
        raise KnowledgeParseError(
            f"evidence L{item.line_start}-{item.line_end} 超出允许范围 L{allowed.line_start}-{allowed.line_end}",
            error_kind="evidence_out_of_range",
        )
    if item.line_end - item.line_start + 1 > max_lines:
        raise KnowledgeParseError(
            f"evidence L{item.line_start}-{item.line_end} 超过 {max_lines} 行上限",
            error_kind="evidence_too_broad",
        )


def _hint_fragments(values: list[str]) -> list[str]:
    """提取用于定位原文的短语；不把整句摘要当作证据。"""
    fragments: set[str] = set()
    for value in values:
        text = value.strip()
        if not text:
            continue
        if len(text) <= 20:
            fragments.add(text)
        for match in re.findall(r"[\u4e00-\u9fff]{2,}", text):
            for size in (4, 3, 2):
                fragments.update(match[index : index + size] for index in range(len(match) - size + 1))
    return sorted(fragments, key=lambda item: (-len(item), item))


def _compact_evidence_span(scene: SceneDocument, card: Any, *, max_lines: int) -> tuple[int, int]:
    """把旧运行状态中的整片 evidence 压到命中卡片线索的局部窗口。"""
    original_start = card.line_start
    original_end = card.line_end
    if original_end - original_start + 1 <= max_lines:
        return original_start, original_end
    hints = [card.title, card.summary]
    if isinstance(card, FactCard):
        hints.extend([card.subject, card.value, card.predicate])
    elif isinstance(card, RelationCard):
        hints.extend([card.subject, card.target, card.relation])
    else:
        hints.extend([*card.participants, *card.causes, *card.outcomes])
    fragments = _hint_fragments(hints)
    lines = scene.text.splitlines()
    best_line = original_start
    best_score = 0
    for absolute_line in range(original_start, original_end + 1):
        index = absolute_line - scene.source.line_start
        if index < 0 or index >= len(lines):
            continue
        line = lines[index]
        score = sum((len(fragment) + 1) for fragment in fragments if fragment in line)
        if score > best_score:
            best_line, best_score = absolute_line, score
    if best_score == 0:
        return original_start, min(original_end, original_start + max_lines - 1)
    context = min(2, max_lines // 4)
    start = max(original_start, best_line - context)
    end = min(original_end, start + max_lines - 1)
    start = max(original_start, end - max_lines + 1)
    return start, end


def _optimize_fact_card(scene: SceneDocument, card: FactCard) -> FactCard | None:
    rewrite: dict[str, Any] | None = None
    rewrite_found = False
    rewrite_key = (scene.story.story_unit_id, card.line_start, card.subject, card.value)
    if rewrite_key in FACT_CONTENT_REWRITES:
        rewrite = FACT_CONTENT_REWRITES[rewrite_key]
        rewrite_found = True
    else:
        # 旧运行状态经常把 evidence 起点写成整段场景的起点，而不是模型
        # 实际引用的行。已人工核验的修正应按“锚点落在原始范围内”匹配，
        # 同时保留 subject/value 条件，避免把同一场景中的其他事实误改。
        for (unit, anchor, subject, value), candidate in FACT_CONTENT_REWRITES.items():
            if (
                unit == scene.story.story_unit_id
                and card.line_start <= anchor <= card.line_end
                and card.subject == subject
                and card.value == value
            ):
                rewrite = candidate
                rewrite_found = True
                break
    if rewrite_found:
        if rewrite is None:
            return None
        card = card.model_copy(update=rewrite)
    evidence_text = "\n".join(
        scene.text.splitlines()[card.line_start - scene.source.line_start : card.line_end - scene.source.line_start + 1]
    )
    if card.subject == "妃" and card.value == "兄妹关系":
        # 兄妹语义统一进入有方向的关系卡，避免与「妃的哥哥是琉璃」
        # 及其反向卡重复，且避免丢失年龄方向。
        return None
    if card.subject == "夜子" and card.value == "闭门不出，说话恶毒，爱板着脸的妹妹":
        card = card.model_copy(
            update={
                "value": "闭门不出，说话恶毒，爱板着脸",
                "title": "夜子的性格",
                "summary": "夜子被介绍为闭门不出、说话恶毒且爱板着脸。",
            }
        )
    if card.subject == "克丽索贝莉露" and card.value == "被父亲欺骗":
        card = card.model_copy(
            update={
                "predicate": "经历",
                "title": "克丽索贝莉露被父亲欺骗",
                "summary": "克丽索贝莉露年幼时被父亲利用，按父母安排冒充天神转世并参与欺骗信徒。",
            }
        )
    if card.predicate == "身份":
        # 这些值来自模型把关系、行为或主观判断误放进身份字段；原文证据
        # 不足以把它们当作稳定身份，直接丢弃比制造新的关系事实更安全。
        if card.subject == "琉璃" and card.value in {
            "夜子的女仆",
            "夜子的初恋",
            "认为搜索妃的房间没有收获",
            "与彼方有复杂关系",
        }:
            return None
        if card.subject == "萤" and card.value == "喜欢彼方":
            return None
        # 亲属、恋人、朋友、主从和创造者属于关系图谱；若留在身份事实中，
        # 同一人物会同时得到互相矛盾的“哥哥/妹妹/女儿”等属性。
        if RELATIONAL_IDENTITY_VALUE_RE.search(card.value) or RELATIONAL_IDENTITY_PREFIX_RE.search(card.value):
            return None
        if FALSE_IDENTITY_VALUE_RE.fullmatch(card.value) and card.subject not in {"暗子", "夜子"}:
            return None
        if card.subject == "萤" and "拥抱的魔法使" in card.value:
            return None
        # 两个全名候选本身可以保留，但不能沿用模型误写的性别/亲属摘要。
        if card.subject == "琉璃" and card.value == "四条琉璃":
            card = card.model_copy(update={"summary": "琉璃的姓名是四条琉璃"})
        elif card.subject == "妃" and card.value == "月社妃":
            card = card.model_copy(update={"summary": "妃的姓名是月社妃"})
    if card.predicate == "死因":
        if DEATH_CAUSE_NON_CAUSE_RE.search(card.value):
            return None
        if card.value == "不明":
            if not re.search(r"死亡|去世|身亡|遇害|自杀|自缢", evidence_text):
                return None
        elif not DEATH_CAUSE_SIGNAL_RE.search(card.value):
            return None
    if card.predicate == "状态":
        replacement: str | None = None
        if PREFERENCE_VALUE_RE.search(card.value):
            replacement = "偏好"
        elif GOAL_VALUE_RE.search(card.value):
            replacement = "目标"
        elif BEHAVIOR_VALUE_RE.search(card.value):
            replacement = "行为"
        elif APPEARANCE_VALUE_RE.search(card.value):
            replacement = "外貌"
        elif PERSONALITY_VALUE_RE.search(card.value):
            replacement = "性格"
        elif not PERSISTENT_STATUS_RE.search(card.value):
            return None
        if replacement is not None:
            card = card.model_copy(update={"predicate": replacement})
    start, end = _compact_evidence_span(scene, card, max_lines=MAX_FACT_RELATION_EVIDENCE_LINES)
    return card.model_copy(update={"line_start": start, "line_end": end})


def _explicit_family_link(evidence_text: str, subject: str, target: str) -> bool:
    """要求亲属称谓与两个人物在同一局部证据中形成直接联系。"""
    names = (re.escape(subject), re.escape(target))
    kinship = r"哥哥|姐姐|弟弟|妹妹|兄长|姐妹|兄妹|父亲|母亲|女儿|儿子"
    for left, right in (names, names[::-1]):
        if re.search(rf"{left}.{{0,18}}(?:{kinship}).{{0,18}}{right}", evidence_text):
            return True
    return False


def _normalize_known_family_relation(card: RelationCard, *, evidence_text: str = "") -> RelationCard | None:
    """纠正主线中已核实的兄妹方向，并丢弃明显的反向误抽取。

    关系卡的 subject/target 不能只依赖模型摘要推断：同一段原文常同时出现
    多个「哥哥/妹妹」称谓。对已核实的人物对固定方向；其余主要角色之间若
    没有明确的「看作/当作」语义，则不保留为亲属关系卡。
    """
    if card.relation not in FAMILY_RELATIONS:
        return card

    pair = frozenset({card.subject, card.target})
    known_directions = KNOWN_SIBLING_DIRECTIONS.get(pair)
    if known_directions is not None:
        # 汀/夜子在旧候选中出现过“把妃当作妹妹”被误挂到夜子的情况；
        # 这一对需要证据级确认。琉璃/妃的方向和称谓在主线多处明确出现，
        # 仍由固定方向表统一，避免让同一条明确事实因证据摘录过短而丢失。
        if (
            pair == frozenset({"夜子", "汀"})
            and evidence_text
            and not _explicit_family_link(evidence_text, card.subject, card.target)
        ):
            return None
        relation, target = known_directions[card.subject]
        return card.model_copy(
            update={
                "relation": relation,
                "target": target,
                "summary": f"{target}是{card.subject}的{relation}",
            }
        )

    audited_names = frozenset({"琉璃", "妃", "夜子", "汀", "理央", "彼方", "暗子", "萤"})
    if pair <= audited_names and not _explicit_family_link(evidence_text, card.subject, card.target):
        return None
    return card


def _optimize_relation_card(scene: SceneDocument, card: RelationCard) -> RelationCard | None:
    if any(
        unit == scene.story.story_unit_id
        and card.line_start <= anchor <= card.line_end
        and card.subject == subject
        and card.relation == relation
        and card.target == target
        for unit, anchor, subject, relation, target in FALSE_RELATION_SPECS
    ):
        return None
    direction_rewrite = next(
        (
            rewrite
            for (unit, anchor, subject, relation, target), rewrite in RELATION_DIRECTION_REWRITES.items()
            if unit == scene.story.story_unit_id
            and card.line_start <= anchor <= card.line_end
            and frozenset({card.subject, card.target}) == frozenset({subject, target})
            and card.relation == relation
        ),
        None,
    )
    content_rewrite = next(
        (
            rewrite
            for (unit, anchor, subject, relation, target), rewrite in RELATION_CONTENT_REWRITES.items()
            if unit == scene.story.story_unit_id
            and card.line_start <= anchor <= card.line_end
            and (card.subject, card.relation, card.target) == (subject, relation, target)
        ),
        None,
    )
    if direction_rewrite is not None:
        subject, target, title, rewritten_summary = direction_rewrite
        card = card.model_copy(
            update={
                "subject": subject,
                "target": target,
                "title": title,
                "summary": rewritten_summary,
            }
        )
    if content_rewrite is not None:
        subject, relation, target, title, rewritten_summary = content_rewrite
        card = card.model_copy(
            update={
                "subject": subject,
                "relation": relation,
                "target": target,
                "title": title,
                "summary": rewritten_summary,
            }
        )
    summary = re.sub(r"[\s，。、“”‘’：:；;（）()、]", "", card.summary)
    directional_terms = {
        "父亲": ("父亲", "爸爸"),
        "母亲": ("母亲", "妈妈"),
        "哥哥": ("哥哥", "兄长"),
        "姐姐": ("姐姐",),
        "弟弟": ("弟弟",),
        "妹妹": ("妹妹",),
        "丈夫": ("丈夫", "老公"),
        "妻子": ("妻子", "老婆"),
    }
    if card.relation == "创造者":
        subject_creates_target = re.search(
            rf"{re.escape(card.subject)}.*(?:创造|制造|再造|塑造|诞生|改写|写下).*{re.escape(card.target)}",
            summary,
        )
        subject_is_target_creator = re.search(
            rf"{re.escape(card.subject)}.*是.*{re.escape(card.target)}.*创造者", summary
        )
        target_is_subject_creator = re.search(
            rf"{re.escape(card.target)}.*是.*{re.escape(card.subject)}.*创造者", summary
        )
        if subject_creates_target or subject_is_target_creator or target_is_subject_creator:
            card = card.model_copy(update={"subject": card.target, "target": card.subject})
    elif card.relation in directional_terms:
        all_roles = tuple(dict.fromkeys(role for roles in directional_terms.values() for role in roles))
        for role in all_roles:
            role_pattern = re.escape(role)
            subject_role_is_target = re.search(
                rf"{re.escape(card.subject)}.*(?:的)?{role_pattern}.*是.*{re.escape(card.target)}", summary
            )
            target_role_is_subject = re.search(
                rf"{re.escape(card.target)}.*(?:的)?{role_pattern}.*是.*{re.escape(card.subject)}", summary
            )
            subject_is_target_role = re.search(
                rf"{re.escape(card.subject)}.*是.*{re.escape(card.target)}.*(?:的)?{role_pattern}", summary
            )
            target_is_subject_role = re.search(
                rf"{re.escape(card.target)}.*是.*{re.escape(card.subject)}.*(?:的)?{role_pattern}", summary
            )
            if subject_role_is_target or target_is_subject_role:
                card = card.model_copy(update={"relation": role})
                break
            if target_role_is_subject or subject_is_target_role:
                card = card.model_copy(update={"subject": card.target, "relation": role, "target": card.subject})
                break
    if (
        card.relation == "主人"
        and any(term in summary for term in ("女仆", "照顾"))
        and re.search(rf"{re.escape(card.subject)}.*(?:女仆|照顾).*{re.escape(card.target)}", summary)
    ):
        card = card.model_copy(update={"subject": card.target, "target": card.subject})
    elif card.relation == "主人" and any(term in summary for term in ("女仆", "照顾")):
        # 模型常写成“理央是夜子的女仆”，但关系标签的方向是“谁的主人”。
        # 识别反向陈述后统一为：理央 --主人--> 夜子。
        reverse = re.search(rf"{re.escape(card.target)}.*(?:女仆|照顾).*{re.escape(card.subject)}", summary)
        if reverse:
            card = card.model_copy(
                update={
                    "subject": card.target,
                    "target": card.subject,
                    "summary": f"{card.subject}是{card.target}的主人",
                }
            )
    if card.relation in {"父亲", "母亲", "哥哥", "姐姐", "弟弟", "妹妹", "儿子", "女儿"} and (
        card.subject == card.relation or card.target == card.relation
    ):
        return None
    if card.subject == card.target:
        return None
    start, end = _compact_evidence_span(scene, card, max_lines=MAX_FACT_RELATION_EVIDENCE_LINES)
    # 旧模型经常把整片场景作为证据。对这类卡片要求局部窗口中出现关系信号，
    # 防止“提到/照顾/同住”被升级成创造者或恋人关系。
    signal_patterns = RELATION_STRONG_SIGNAL_PATTERNS.get(card.relation) or RELATION_SIGNAL_PATTERNS.get(
        card.relation, ()
    )
    must_have_signal = card.relation in RELATION_STRONG_SIGNAL_PATTERNS
    lines = scene.text.splitlines()
    evidence_text = "\n".join(lines[start - scene.source.line_start : end - scene.source.line_start + 1])
    if card.relation in FAMILY_RELATIONS:
        card = _normalize_known_family_relation(card, evidence_text=evidence_text)
        if card is None:
            return None
    if (must_have_signal or card.line_end - card.line_start + 1 > MAX_FACT_RELATION_EVIDENCE_LINES) and not any(
        signal in evidence_text for signal in signal_patterns
    ):
        return None
    relation_needs_both_names = (
        card.relation
        in {
            "创造者",
            "恋人",
            "前恋人",
            "暗恋对象",
            "初恋对象",
        }
        or card.line_end - card.line_start + 1 > MAX_FACT_RELATION_EVIDENCE_LINES
    )
    if relation_needs_both_names and not (card.subject in evidence_text and card.target in evidence_text):
        return None
    if card.relation == "恋人":
        weak_summary = re.compile(r"希望|打算|想成为|喜欢|爱意|感情|复杂|亲密|声称|认为自己|有关系|关心")
        strong_summary = re.compile(r"恋人关系|是恋人|成为恋人|开始交往|恋爱关系|曾有过恋情|相爱")
        negative_evidence = re.search(r"(?:没有|没|无|不).{0,10}(?:兴趣|恋爱|交往|恋情|关系)", evidence_text)
        pair_signal = re.search(
            rf"(?:{re.escape(card.subject)}.{{0,28}}(?:恋人|交往|恋情|相爱|恋爱|告白).{{0,28}}{re.escape(card.target)}|"
            rf"{re.escape(card.target)}.{{0,28}}(?:恋人|交往|恋情|相爱|恋爱|告白).{{0,28}}{re.escape(card.subject)})",
            evidence_text,
        )
        if weak_summary.search(summary) or not strong_summary.search(summary) or negative_evidence or not pair_signal:
            return None
    if card.relation == "创造者" and not (card.subject in evidence_text or card.target in evidence_text):
        return None
    if card.relation == "创造者":
        creator_signal = re.search(
            rf"(?:{re.escape(card.target)}.{{0,24}}(?:创造|制造|再造|塑造|写下).{{0,24}}{re.escape(card.subject)}|"
            rf"{re.escape(card.subject)}.{{0,24}}(?:创造|制造|再造|塑造|写下).{{0,24}}{re.escape(card.target)})",
            evidence_text,
        )
        if not creator_signal:
            evidence_lines = evidence_text.splitlines()
            for index, line in enumerate(evidence_lines):
                if not re.search(r"创造|制造|再造|塑造|写下", line) or card.target not in line:
                    continue
                nearby = evidence_lines[max(0, index - 2) : index + 3]
                if any(card.subject in nearby_line for nearby_line in nearby):
                    creator_signal = True
                    break
        if not creator_signal:
            return None
    if card.relation == "朋友" and "搭档" in card.summary and "朋友" not in card.summary:
        card = card.model_copy(update={"relation": "搭档"})
    if direction_rewrite is not None:
        subject, target, title, rewritten_summary = direction_rewrite
        card = card.model_copy(
            update={"subject": subject, "target": target, "title": title, "summary": rewritten_summary}
        )
    return card.model_copy(update={"line_start": start, "line_end": end})


def _optimize_event_card(scene: SceneDocument, card: EventCard) -> EventCard | None:
    rewrite = EVENT_CONTENT_REWRITES.get((scene.story.story_unit_id, card.line_start))
    if rewrite is None and scene.story.story_unit_id == "vol09_9白珍珠的泡沫爱慕":
        # 这三张卡都把整段回忆报成 L483 起点，不能仅用同一个行号区分。
        # 按模型原始标题选择对应事件，避免把“出生”和“流放”合并成一张卡。
        if card.line_start <= 486 <= card.line_end and card.title == "夜子的诞生":
            rewrite = EVENT_CONTENT_REWRITES[(scene.story.story_unit_id, 486)]
        elif card.line_start <= 486 <= card.line_end and card.title == "夜子的流放":
            rewrite = {
                "title": "暗子与夜子被流放到小岛",
                "summary": "暗子反复使用魔法之书保护夜子，导致诅咒谣言加剧；遊行寺家随后将暗子与夜子流放到小岛。",
                "participants": ["暗子", "夜子"],
                "causes": ["暗子使用魔法之书保护夜子", "遊行寺家害怕诅咒谣言"],
                "outcomes": ["暗子与夜子被迫离开遊行寺家并迁往小岛"],
            }
        elif card.line_start <= 493 <= card.line_end and card.title == "魔法之书的使用":
            rewrite = EVENT_CONTENT_REWRITES[(scene.story.story_unit_id, 493)]
    if rewrite is None:
        # 7B 常以整个场景作为事件 evidence，导致候选起点早于事件锚点。
        # 这里仅对已人工核实且标题明确的卡片补做匹配，不按宽泛关键词改写
        # 同一场景中的其他事件。
        fallback_specs: tuple[tuple[str, int, int, str], ...] = (
            ("vol03_3蓝宝石的存在证明", 2822, 2827, "蓝宝石影响下的恋爱"),
            ("vol03_3蓝宝石的存在证明", 3098, 3155, "汀和妃的恋爱"),
            ("vol08_8萤石的怠惰现象", 1185, 1366, "妃和琉璃的告别"),
            ("vol11_11黑玛瑙的不在证明", 649, 714, "夜子对彼方的好感变化"),
            ("vol13_13璀璨的紫翠玉", 1, 20, "克丽索贝莉露与夜子的决裂"),
            ("vol09_9白珍珠的泡沫爱慕", 741, 760, "琉璃与夜子的初次见面"),
            ("vol12_12青金石的幻想图书馆", 288, 332, "告白与拒绝"),
            ("vol12_12青金石的幻想图书馆", 1502, 1516, "告白失败"),
            ("epilogue_bonus", 632, 698, "妃和彼方的关系结束"),
        )
        for unit, raw_start, anchor, title in fallback_specs:
            if scene.story.story_unit_id == unit and card.line_start == raw_start and card.title == title:
                rewrite = EVENT_CONTENT_REWRITES.get((unit, anchor))
                break
    if rewrite is None:
        # 同样兼容旧模型把 evidence 起点扩成场景起点的情况。标题是
        # 第二重约束：同一场景可能有多个事件，不能只按范围命中。
        event_titles = {
            ("vol07_7黑珍珠的求爱信号", 1262): "彼方邀请夜子",
            ("vol09_9绿幽灵水晶的命运连锁", 519): "琉璃告白",
            ("vol10_10黑曜石的因果目录", 508): "馆长施魔法",
            ("vol09_9白珍珠的泡沫爱慕", 493): "魔法之书的使用",
            ("vol03_3蓝宝石的存在证明", 1012): "奏警告琉璃和妃",
            ("vol05_5磷灰石的怠惰现象", 2718): "奏与琉璃的冲突",
            ("vol13_13璀璨的紫翠玉", 1045): "克丽索贝莉露的恋妹情结",
        }
        for (unit, anchor), old_title in event_titles.items():
            if (
                unit == scene.story.story_unit_id
                and card.line_start <= anchor <= card.line_end
                and card.title == old_title
            ):
                rewrite = EVENT_CONTENT_REWRITES.get((unit, anchor))
                break
    if rewrite is not None:
        card = card.model_copy(update=rewrite)
    start, end = _compact_evidence_span(scene, card, max_lines=MAX_EVENT_EVIDENCE_LINES)
    return card.model_copy(update={"line_start": start, "line_end": end})


def parse_knowledge_candidates(
    raw: str,
    scene: SceneDocument,
    *,
    span: SourceSpan | None = None,
    alias_config: AliasConfig,
) -> SceneKnowledgeCandidates:
    """严格解析模型 JSON 输出为单场景（分片）知识卡候选。

    结构性错误整体拒绝（抛 KnowledgeParseError，触发重试）：markdown 围栏、
    非字符串输出、非法 JSON、非对象 JSON、重复键、顶层字段缺失/额外/类型错误、
    scene_id 不匹配。

    卡片级错误丢弃该卡并计入 dropped_invalid（smoke 实测模型会偶发输出非法
    predicate/relation、人物位长短语或实体名、越界行号；丢弃不污染其余合法卡片，
    计数进入质量报告供人工复核）：非法 predicate/relation、非法枚举、卡片额外
    字段、空白必填字段、人物位出现非人物条目/未知名称、evidence 越界、行号倒置。
    """
    if not isinstance(raw, str):
        raise KnowledgeParseError(f"模型输出必须是字符串，得到 {type(raw).__name__}", error_kind="invalid_output")
    text = raw.strip()
    if text.startswith("```"):
        raise KnowledgeParseError("模型输出包含 markdown 代码围栏（```），要求纯 JSON", error_kind="markdown_fence")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise KnowledgeParseError(f"模型 JSON 含重复键: {key!r}", error_kind="duplicate_json_key")
            result[key] = value
        return result

    try:
        data = json.loads(text, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise KnowledgeParseError(f"模型输出不是合法 JSON: {exc}", error_kind="invalid_json") from exc
    if not isinstance(data, dict):
        raise KnowledgeParseError(f"模型输出必须是 JSON 对象，得到 {type(data).__name__}", error_kind="invalid_json")
    if set(data) != _PAYLOAD_KEYS:
        missing = sorted(_PAYLOAD_KEYS - set(data))
        extra = sorted(set(data) - _PAYLOAD_KEYS)
        raise KnowledgeParseError(f"顶层字段非法（缺失: {missing}；额外: {extra}）", error_kind="schema_violation")
    if not isinstance(data["scene_id"], str) or not data["scene_id"].strip():
        raise KnowledgeParseError("scene_id 必须为非空字符串", error_kind="schema_violation")
    for key in ("facts", "relations", "events"):
        if not isinstance(data[key], list):
            raise KnowledgeParseError(f"{key} 必须为数组", error_kind="schema_violation")
    if data["scene_id"] != scene.id:
        raise KnowledgeParseError(
            f"模型输出 scene_id {data['scene_id']!r} 与请求场景 {scene.id!r} 不一致",
            error_kind="scene_id_mismatch",
        )
    allowed = span if span is not None else scene.source

    facts: list[FactCard] = []
    relations: list[RelationCard] = []
    events: list[EventCard] = []
    dropped = 0

    for item in data["facts"]:
        try:
            entry = _FactItem.model_validate(item)
            if entry.predicate not in FACT_PREDICATES:
                raise KnowledgeParseError(
                    f"fact.predicate {entry.predicate!r} 不在规范表内", error_kind="schema_violation"
                )
            _check_span(entry, allowed, max_lines=MAX_FACT_RELATION_EVIDENCE_LINES)
            facts.append(
                FactCard(
                    subject=alias_config.normalize(entry.subject.strip()),
                    predicate=entry.predicate,
                    value=entry.value,
                    title=entry.title,
                    summary=entry.summary,
                    reality_status=entry.reality_status,
                    line_start=entry.line_start,
                    line_end=entry.line_end,
                )
            )
        except (ValidationError, KnowledgeParseError):
            dropped += 1

    for item in data["relations"]:
        try:
            entry = _RelationItem.model_validate(item)
            if entry.relation not in RELATION_LABELS:
                raise KnowledgeParseError(f"relation {entry.relation!r} 不在规范表内", error_kind="schema_violation")
            _check_span(entry, allowed, max_lines=MAX_FACT_RELATION_EVIDENCE_LINES)
            subject = _normalize_person(entry.subject, alias_config)
            target = _normalize_person(entry.target, alias_config)
            if subject == target:
                raise KnowledgeParseError("relation 的 subject 与 target 不得相同", error_kind="self_relation")
            if entry.relation in SYMMETRIC_RELATIONS and target < subject:
                # 对称关系固定方向（按人物名排序），避免同一对人物的两个方向随机混用。
                subject, target = target, subject
            relations.append(
                RelationCard(
                    subject=subject,
                    relation=entry.relation,
                    target=target,
                    title=entry.title,
                    summary=entry.summary,
                    reality_status=entry.reality_status,
                    line_start=entry.line_start,
                    line_end=entry.line_end,
                )
            )
        except (ValidationError, KnowledgeParseError):
            dropped += 1

    for item in data["events"]:
        try:
            entry = _EventItem.model_validate(item)
            _check_span(entry, allowed, max_lines=MAX_EVENT_EVIDENCE_LINES)
            participants = list(dict.fromkeys(_normalize_person(name, alias_config) for name in entry.participants))
            events.append(
                EventCard(
                    title=entry.title,
                    summary=entry.summary,
                    participants=participants,
                    causes=[cause for cause in entry.causes if cause.strip()],
                    outcomes=[outcome for outcome in entry.outcomes if outcome.strip()],
                    reality_status=entry.reality_status,
                    line_start=entry.line_start,
                    line_end=entry.line_end,
                )
            )
        except (ValidationError, KnowledgeParseError):
            dropped += 1

    return SceneKnowledgeCandidates(
        scene_id=scene.id, facts=facts, relations=relations, events=events, dropped_invalid=dropped
    )


# ---------- 候选生成（分片 + 断点续跑） ----------


def _truncate_detail(detail: str) -> str:
    text = " ".join(str(detail or "").split())
    if len(text) <= FAILURE_DETAIL_MAX_CHARS:
        return text
    return text[:FAILURE_DETAIL_MAX_CHARS] + "…"


def _failure_summary(error_kind: str, detail: str, attempts: int) -> KnowledgeFailureSummary:
    return KnowledgeFailureSummary(error_kind=error_kind, detail=_truncate_detail(detail), attempts=attempts)


def _merge_chunk_candidates(chunks: list[SceneKnowledgeCandidates]) -> SceneKnowledgeCandidates:
    """分片候选归并回原 scene_id：卡片按类型拼接，丢弃计数累计。"""
    return SceneKnowledgeCandidates(
        scene_id=chunks[0].scene_id,
        facts=[card for chunk in chunks for card in chunk.facts],
        relations=[card for chunk in chunks for card in chunk.relations],
        events=[card for chunk in chunks for card in chunk.events],
        dropped_invalid=sum(chunk.dropped_invalid for chunk in chunks),
    )


def _generate_for_scene(
    scene: SceneDocument,
    model_client: KnowledgeModelClient,
    *,
    max_attempts: int,
    chunk_max_chars: int,
    alias_config: AliasConfig,
) -> tuple[SceneKnowledgeCandidates | None, KnowledgeFailureSummary | None, int, int]:
    """为单个场景生成候选：按字符分片 → 逐片重试 → 归并。

    返回 (候选, 失败摘要, 模型调用次数, 分片数)；候选与失败摘要至多一个非 None。
    单个分片重试耗尽 → 整场景判失败，不产出部分归并候选。
    """
    spans = _chunk_spans(scene, chunk_max_chars=chunk_max_chars)
    fact_limit, relation_limit, event_limit = chunk_limits(len(spans))
    chunk_candidates: list[SceneKnowledgeCandidates] = []
    attempts = 0
    failure: KnowledgeFailureSummary | None = None
    for index, span in enumerate(spans, start=1):
        chunked = len(spans) > 1
        prompt = build_knowledge_prompt(
            scene,
            span=span if chunked else None,
            chunk_index=index if chunked else None,
            total_chunks=len(spans) if chunked else None,
            alias_config=alias_config,
            fact_limit=fact_limit,
            relation_limit=relation_limit,
            event_limit=event_limit,
        )
        for _ in range(max_attempts):
            attempts += 1
            try:
                raw = model_client(prompt)
                parsed = parse_knowledge_candidates(
                    raw, scene, span=span if chunked else None, alias_config=alias_config
                )
            except TimeoutError as exc:
                # 异常消息可能携带 URL：只落盘类型名，原文不进运行状态。
                failure = _failure_summary("timeout", type(exc).__name__, attempts)
                continue
            except KnowledgeParseError as exc:
                # 自定义校验消息可能包含模型提供的非法值；持久化时仅保留错误分类。
                failure = _failure_summary(exc.error_kind, type(exc).__name__, attempts)
                continue
            except Exception as exc:  # 模型客户端任意异常都收敛为可重试失败
                # 异常消息可能携带密钥、URL 或回显 prompt：只落盘类型名。
                failure = _failure_summary("model_error", type(exc).__name__, attempts)
                continue
            kept = len(parsed.facts) + len(parsed.relations) + len(parsed.events)
            if kept == 0 and parsed.dropped_invalid > 0:
                # 整片卡片全部非法：通常意味着模型对该片段的系统性误解，值得重试；
                # 重试耗尽则整场景失败（不产出「全部丢弃」的空壳结果）。
                failure = _failure_summary("schema_violation", "KnowledgeParseError", attempts)
                continue
            chunk_candidates.append(parsed)
            failure = None
            break
        else:
            return None, failure, attempts, len(spans)
    merged = chunk_candidates[0] if len(chunk_candidates) == 1 else _merge_chunk_candidates(chunk_candidates)
    return merged, None, attempts, len(spans)


def create_knowledge_run(
    bundle: EnrichedSceneBundle,
    *,
    model_id: str,
    generation_params: dict[str, int] | None = None,
) -> KnowledgeCandidateRunState:
    """创建确定性初始运行状态：每个 enriched 场景恰好一条 pending 记录。"""
    if not model_id or not model_id.strip():
        raise ValueError("model_id 不得为空白")
    params = dict(DEFAULT_GENERATION_PARAMS)
    if generation_params is not None:
        params.update({key: int(value) for key, value in generation_params.items()})
    if params["chunk_max_chars"] < 1:
        raise ValueError("chunk_max_chars 必须 >= 1")
    return KnowledgeCandidateRunState(
        schema_version=KNOWLEDGE_CANDIDATE_SCHEMA_VERSION,
        source_enriched=bundle.source_ref,
        total_source_scenes=len(bundle.scenes),
        model_id=model_id,
        generation_params=params,
        scene_states=[
            KnowledgeSceneState(scene_id=scene.id, status=KnowledgeGenerationStatus.pending) for scene in bundle.scenes
        ],
    )


def _coerce_run_state(
    run_state: KnowledgeCandidateRunState | dict[str, Any],
) -> tuple[KnowledgeCandidateRunState | None, list[str]]:
    if isinstance(run_state, KnowledgeCandidateRunState):
        return run_state, []
    try:
        return KnowledgeCandidateRunState.model_validate(run_state), []
    except ValidationError as exc:
        return None, [str(exc)]


def validate_knowledge_run(
    run_state: KnowledgeCandidateRunState | dict[str, Any], bundle: EnrichedSceneBundle
) -> list[str]:
    """运行状态校验：结构合法 + enriched 三摘要绑定 + 场景集合一致（返回错误列表）。"""
    state, coerce_errors = _coerce_run_state(run_state)
    errors = list(coerce_errors)
    if state is None:
        return errors
    errors.extend(enriched_bundle_integrity_errors(bundle))
    ref = bundle.source_ref
    if state.source_enriched.enriched_manifest_sha256 != ref.enriched_manifest_sha256:
        errors.append("source_enriched.enriched_manifest_sha256 与 bundle 不一致（跨 bundle 恢复被拒绝）")
    if state.source_enriched.enriched_scenes_sha256 != ref.enriched_scenes_sha256:
        errors.append("source_enriched.enriched_scenes_sha256 与 bundle 不一致（跨 bundle 恢复被拒绝）")
    if state.source_enriched.enriched_bundle_sha256 != ref.enriched_bundle_sha256:
        errors.append("source_enriched.enriched_bundle_sha256 与 bundle 不一致（跨 bundle 恢复被拒绝）")
    if state.schema_version != KNOWLEDGE_CANDIDATE_SCHEMA_VERSION:
        errors.append(f"schema_version 必须为 {KNOWLEDGE_CANDIDATE_SCHEMA_VERSION}")
    if state.total_source_scenes != len(bundle.scenes):
        errors.append("total_source_scenes 与 bundle 场景数不一致")
    bundle_ids = [scene.id for scene in bundle.scenes]
    state_ids = [item.scene_id for item in state.scene_states]
    if state_ids != bundle_ids:
        errors.append("scene_states 的 scene_id 集合/顺序与 bundle 不一致（scene_id 不得更换）")
    return errors


def select_knowledge_scenes(
    run_state: KnowledgeCandidateRunState, *, scene_ids: list[str] | tuple[str, ...] | None = None
) -> list[str]:
    """场景选择：默认全部 pending + failed（失败自动重试）；指定 id 时精确选择。

    已成功场景默认跳过（成功结果不被无意覆盖）；指定不存在的 scene id 直接拒绝。
    """
    if scene_ids is None:
        return [
            item.scene_id for item in run_state.scene_states if item.status is not KnowledgeGenerationStatus.success
        ]
    known = {item.scene_id for item in run_state.scene_states}
    unknown = sorted(set(scene_ids) - known)
    if unknown:
        raise ValueError(f"选择包含未知 scene id: {unknown[:5]}…")
    return list(scene_ids)


def generate_knowledge_candidates(
    bundle: EnrichedSceneBundle,
    run_state: KnowledgeCandidateRunState | dict[str, Any],
    model_client: KnowledgeModelClient,
    *,
    scene_ids: list[str] | tuple[str, ...] | None = None,
    max_attempts: int = 3,
    state_path: Path | str | None = None,
    alias_config: AliasConfig,
) -> KnowledgeRunResult:
    """对选中场景生成知识卡候选（断点可续：提供 state_path 时逐场景原子保存进度）。

    门禁（任何模型调用之前完成，任一失败抛 ValueError 且零模型调用）：
    - bundle 完整性与 enriched 三摘要绑定校验（跨 bundle 的运行状态一律拒绝）；
    - 运行状态结构/一致性校验；max_attempts >= 1。

    处理规则：
    - 按 enriched 场景顺序处理选中场景；已成功场景跳过（成功结果不被覆盖）；
    - 单场景（分片）最多 max_attempts 次尝试，耗尽置 failed 并保留失败摘要
      （detail 只含异常类型名）；
    - 失败不影响其他场景；state_path 提供时每个已处理场景之后原子保存一次。
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts 必须 >= 1，当前为 {max_attempts}")
    state, coerce_errors = _coerce_run_state(run_state)
    if state is None:
        raise ValueError("候选运行状态非法:\n- " + "\n- ".join(coerce_errors))
    errors = validate_knowledge_run(state, bundle)
    if errors:
        raise ValueError("候选运行状态未通过校验（拒绝在任何模型调用之前）:\n- " + "\n- ".join(errors))

    selected = select_knowledge_scenes(state, scene_ids=scene_ids)
    selected_set = set(selected)

    new_state = state.model_copy(deep=True)
    new_states_by_id = {item.scene_id: item for item in new_state.scene_states}
    attempted: list[str] = []
    succeeded: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []
    total_attempts = 0
    for scene in bundle.scenes:
        scene_id = scene.id
        if scene_id not in selected_set:
            continue
        item = new_states_by_id[scene_id]
        attempted.append(scene_id)
        if item.status is KnowledgeGenerationStatus.success:
            skipped.append(scene_id)
            continue
        candidates, failure, attempts, chunk_count = _generate_for_scene(
            scene,
            model_client,
            max_attempts=max_attempts,
            chunk_max_chars=int(new_state.generation_params["chunk_max_chars"]),
            alias_config=alias_config,
        )
        total_attempts += attempts
        item.attempts += attempts
        item.chunk_count = chunk_count
        if candidates is not None:
            item.candidates = candidates
            item.status = KnowledgeGenerationStatus.success
            item.last_failure = None
            succeeded.append(scene_id)
        else:
            item.status = KnowledgeGenerationStatus.failed
            item.candidates = None
            item.last_failure = failure
            failed.append(scene_id)
        if state_path is not None:
            save_knowledge_run(state_path, new_state)
    return KnowledgeRunResult(
        new_state=new_state,
        attempted_scene_ids=attempted,
        succeeded_scene_ids=succeeded,
        failed_scene_ids=failed,
        skipped_scene_ids=skipped,
        total_attempts=total_attempts,
    )


def save_knowledge_run(path: Path | str, run_state: KnowledgeCandidateRunState | dict[str, Any]) -> None:
    """运行状态原子保存：结构复查，非法状态拒绝落盘。"""
    state, errors = _coerce_run_state(run_state)
    if state is None:
        raise ValueError("候选运行状态非法，拒绝写出:\n- " + "\n- ".join(errors))
    payload = json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    _atomic_write_text(Path(path), payload)


def load_knowledge_run(path: Path | str) -> KnowledgeCandidateRunState:
    return KnowledgeCandidateRunState.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------- 稳定 ID 与候选文档构建 ----------


def _card_id(document_type: str, prefix: str, scene_id: str, span: tuple[int, int], payload_key: str) -> str:
    payload = f"{document_type}|{scene_id}|{span[0]}-{span[1]}|{payload_key}"
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _evidence_text(scene: SceneDocument, line_start: int, line_end: int) -> str:
    lines = scene.text.splitlines()
    offset = line_start - scene.source.line_start
    excerpt = lines[offset : offset + (line_end - line_start + 1)]
    return "\n".join(excerpt)


class DuplicateGroup(BaseModel):
    """清理后仍保留的跨场景重复组，供人工判断动态或跨叙事层知识。"""

    model_config = ConfigDict(extra="forbid")

    card_type: Literal["fact", "relation"]
    payload_key: NonEmptyStr
    card_ids: list[str] = Field(min_length=2)
    scene_ids: list[str] = Field(min_length=2)


class ConflictGroup(BaseModel):
    """潜在冲突组：同一 (subject, predicate)/(subject, relation) 出现多个不同值/对象。

    不擅自择真；相关卡片置为 needs_review，由人工裁决。
    """

    model_config = ConfigDict(extra="forbid")

    card_type: Literal["fact", "relation"]
    group_key: NonEmptyStr
    card_ids: list[str] = Field(min_length=2)
    values: list[str] = Field(min_length=2)
    description: NonEmptyStr


class SceneFinalStats(BaseModel):
    """单场景 finalize 统计。"""

    model_config = ConfigDict(extra="forbid")

    scene_id: NonEmptyStr
    story_unit_id: NonEmptyStr
    facts: int = Field(ge=0)
    relations: int = Field(ge=0)
    events: int = Field(ge=0)
    dropped_by_cap: int = Field(default=0, ge=0)
    deduped_in_scene: int = Field(default=0, ge=0)
    deduped_cross_scene: int = Field(default=0, ge=0)
    dropped_invalid: int = Field(default=0, ge=0)


class KnowledgeFinalization(BaseModel):
    """finalize 结果：候选文档（确定性排序）+ 重复组 + 冲突组 + 逐场景统计。"""

    model_config = ConfigDict(extra="forbid")

    fact_documents: list[FactDocument]
    relation_documents: list[RelationDocument]
    event_documents: list[EventDocument]
    duplicate_groups: list[DuplicateGroup]
    conflict_groups: list[ConflictGroup]
    scene_stats: list[SceneFinalStats]
    deduped_cross_scene: int = Field(default=0, ge=0)

    @property
    def total_documents(self) -> int:
        return len(self.fact_documents) + len(self.relation_documents) + len(self.event_documents)


def finalize_knowledge_candidates(
    bundle: EnrichedSceneBundle,
    run_state: KnowledgeCandidateRunState | dict[str, Any],
    *,
    alias_config: AliasConfig,
) -> KnowledgeFinalization:
    """从运行状态确定性构建候选文档（稳定 ID / 去重 / 上限 / 排序 / 冲突标记）。

    - 仅消费 success 场景的候选；failed/pending 场景不出文档（不伪造）；
    - 场景内相同 canonical payload 去重（保留行号最靠前的证据）；
    - 超出单场景上限的卡片按 (line_start, line_end, payload_key) 确定性截断并计数；
    - 同一叙事域内的稳定身份、外貌、设定和亲属/创造关系跨场景去重；
    - evidence_text 由系统按行号从场景原文截取（绝不改写）；
    - relation/event 的人物名做防御性复核（别名表变更时拒绝而非静默产出）；
    - 冲突组内卡片 review_status=needs_review，其余为 draft；绝无 approved；
    - 全局排序：(场景冻结顺序, 类型 fact<relation<event, line_start, line_end, id)。
    """
    state, coerce_errors = _coerce_run_state(run_state)
    if state is None:
        raise ValueError("候选运行状态非法:\n- " + "\n- ".join(coerce_errors))
    errors = validate_knowledge_run(state, bundle)
    if errors:
        raise ValueError("候选运行状态未通过校验:\n- " + "\n- ".join(errors))

    fact_specs: list[tuple[int, FactCard, str]] = []  # (scene_order, card, scene_id)
    relation_specs: list[tuple[int, RelationCard, str]] = []
    event_specs: list[tuple[int, EventCard, str]] = []
    scene_stats: list[SceneFinalStats] = []
    scene_order_by_id = {scene.id: order for order, scene in enumerate(bundle.scenes)}

    for order, scene in enumerate(bundle.scenes):
        item = next(s for s in state.scene_states if s.scene_id == scene.id)
        if item.status is not KnowledgeGenerationStatus.success or item.candidates is None:
            scene_stats.append(
                SceneFinalStats(
                    scene_id=scene.id, story_unit_id=scene.story.story_unit_id, facts=0, relations=0, events=0
                )
            )
            continue
        candidates = item.candidates

        # 防御性复核：解析期归一后，人物位仍须落在当前别名表的合法人物集合内。
        for card in [*candidates.relations, *candidates.events]:
            names = [card.subject, card.target] if isinstance(card, RelationCard) else list(card.participants)
            for name in names:
                if not alias_config.is_person(name):
                    raise ValueError(f"场景 {scene.id} 候选含非法人物名 {name!r}（别名表校验失败）")

        def unique_keep_first(cards: list[Any]) -> tuple[list[Any], int]:
            """场景内 canonical payload 去重：按 (line_start, line_end, payload_key) 排序保留首个。"""
            seen: set[str] = set()
            kept: list[Any] = []
            for card in sorted(cards, key=lambda c: (c.line_start, c.line_end, c.payload_key)):
                if card.payload_key in seen:
                    continue
                seen.add(card.payload_key)
                kept.append(card)
            return kept, len(cards) - len(kept)

        optimized_facts = [_optimize_fact_card(scene, card) for card in candidates.facts]
        optimized_relations = [_optimize_relation_card(scene, card) for card in candidates.relations]
        optimized_events = [_optimize_event_card(scene, card) for card in candidates.events]
        facts = [card for card in optimized_facts if card is not None]
        relations = [card for card in optimized_relations if card is not None]
        events = [card for card in optimized_events if card is not None]
        optimized_drops = (
            len(candidates.facts)
            - len(facts)
            + len(candidates.relations)
            - len(relations)
            + len(candidates.events)
            - len(events)
        )
        facts, deduped_f = unique_keep_first(facts)
        relations, deduped_r = unique_keep_first(relations)
        events, deduped_e = unique_keep_first(events)
        facts = facts[:MAX_FACTS_PER_SCENE]
        relations = relations[:MAX_RELATIONS_PER_SCENE]
        events = events[:MAX_EVENTS_PER_SCENE]
        scene_stats.append(
            SceneFinalStats(
                scene_id=scene.id,
                story_unit_id=scene.story.story_unit_id,
                facts=len(facts),
                relations=len(relations),
                events=len(events),
                dropped_by_cap=(
                    len(candidates.facts)
                    + len(candidates.relations)
                    + len(candidates.events)
                    - optimized_drops
                    - deduped_f
                    - deduped_r
                    - deduped_e
                    - len(facts)
                    - len(relations)
                    - len(events)
                ),
                deduped_in_scene=deduped_f + deduped_r + deduped_e,
                dropped_invalid=candidates.dropped_invalid + optimized_drops,
            )
        )
        fact_specs.extend((order, card, scene.id) for card in facts)
        relation_specs.extend((order, card, scene.id) for card in relations)
        event_specs.extend((order, card, scene.id) for card in events)

    scene_by_id = {scene.id: scene for scene in bundle.scenes}

    def stable_dedup_domain(card: FactCard | RelationCard, scene_id: str) -> tuple[str, ...] | None:
        """返回保守的跨场景去重域；动态知识和事件不进入此流程。"""
        scene = scene_by_id[scene_id]
        if isinstance(card, FactCard):
            if card.predicate not in STABLE_FACT_PREDICATES:
                return None
            card_type = "fact"
        else:
            if card.relation not in STABLE_RELATIONS:
                return None
            card_type = "relation"
        temporal = (
            getattr(scene.story.temporal_scope, "value", scene.story.temporal_scope)
            if scene.story.temporal_scope is not None
            else "none"
        )
        return (
            card_type,
            card.payload_key,
            getattr(scene.story.content_scope, "value", scene.story.content_scope),
            temporal,
            getattr(card.reality_status, "value", card.reality_status),
        )

    def dedupe_stable_specs(specs: list[tuple[int, Any, str]]) -> tuple[list[tuple[int, Any, str]], list[str]]:
        grouped: dict[tuple[str, ...], list[tuple[int, Any, str]]] = {}
        passthrough: list[tuple[int, Any, str]] = []
        for spec in specs:
            domain = stable_dedup_domain(spec[1], spec[2])
            if domain is None:
                passthrough.append(spec)
            else:
                grouped.setdefault(domain, []).append(spec)

        kept = list(passthrough)
        removed_scene_ids: list[str] = []
        for members in grouped.values():
            ranked = sorted(
                members,
                key=lambda spec: (
                    spec[1].line_end - spec[1].line_start + 1,
                    spec[0],
                    spec[1].line_start,
                    spec[1].line_end,
                    spec[2],
                ),
            )
            kept.append(ranked[0])
            removed_scene_ids.extend(spec[2] for spec in ranked[1:])
        kept.sort(key=lambda spec: (spec[0], spec[1].line_start, spec[1].line_end, spec[1].payload_key))
        return kept, removed_scene_ids

    fact_specs, removed_fact_scenes = dedupe_stable_specs(fact_specs)
    relation_specs, removed_relation_scenes = dedupe_stable_specs(relation_specs)
    removed_by_scene = Counter([*removed_fact_scenes, *removed_relation_scenes])
    for stat in scene_stats:
        removed = removed_by_scene.get(stat.scene_id, 0)
        if not removed:
            continue
        stat.facts -= removed_fact_scenes.count(stat.scene_id)
        stat.relations -= removed_relation_scenes.count(stat.scene_id)
        stat.deduped_cross_scene = removed

    def build_fact(card: FactCard, scene_id: str) -> FactDocument:
        scene = scene_by_id[scene_id]
        return FactDocument(
            id=_card_id("fact", "fact", scene_id, (card.line_start, card.line_end), card.payload_key),
            title=card.title,
            subject=card.subject,
            predicate=card.predicate,
            value=card.value,
            summary=card.summary,
            evidence_text=_evidence_text(scene, card.line_start, card.line_end),
            story=scene.story.model_copy(deep=True),
            source=SourceSpan(source_path=scene.source.source_path, line_start=card.line_start, line_end=card.line_end),
            reality_status=card.reality_status,
            review_status=ReviewStatus.draft,
        )

    def build_relation(card: RelationCard, scene_id: str) -> RelationDocument:
        scene = scene_by_id[scene_id]
        return RelationDocument(
            id=_card_id("relation", "rel", scene_id, (card.line_start, card.line_end), card.payload_key),
            title=card.title,
            subject=card.subject,
            relation=card.relation,
            target=card.target,
            summary=card.summary,
            evidence_text=_evidence_text(scene, card.line_start, card.line_end),
            story=scene.story.model_copy(deep=True),
            source=SourceSpan(source_path=scene.source.source_path, line_start=card.line_start, line_end=card.line_end),
            reality_status=card.reality_status,
            review_status=ReviewStatus.draft,
        )

    def build_event(card: EventCard, scene_id: str) -> EventDocument:
        scene = scene_by_id[scene_id]
        return EventDocument(
            id=_card_id("event", "event", scene_id, (card.line_start, card.line_end), card.payload_key),
            title=card.title,
            summary=card.summary,
            participants=list(card.participants),
            causes=list(card.causes),
            outcomes=list(card.outcomes),
            evidence_text=_evidence_text(scene, card.line_start, card.line_end),
            story=scene.story.model_copy(deep=True),
            source=SourceSpan(source_path=scene.source.source_path, line_start=card.line_start, line_end=card.line_end),
            reality_status=card.reality_status,
            review_status=ReviewStatus.draft,
        )

    fact_docs = [build_fact(card, scene_id) for _, card, scene_id in fact_specs]
    relation_docs = [build_relation(card, scene_id) for _, card, scene_id in relation_specs]
    event_docs = [build_event(card, scene_id) for _, card, scene_id in event_specs]

    # ---- 清理后仍存在的跨场景重复组（动态知识或不同叙事域，仅报告） ----
    duplicate_groups: list[DuplicateGroup] = []
    for docs, card_type in ((fact_docs, "fact"), (relation_docs, "relation")):
        groups: dict[str, list[Any]] = {}
        for doc in docs:
            key = (
                f"{doc.subject}|{doc.predicate}|{doc.value}"
                if card_type == "fact"
                else f"{doc.subject}|{doc.relation}|{doc.target}"
            )
            groups.setdefault(key, []).append(doc)
        for key, members in sorted(groups.items()):
            if len(members) >= 2:
                duplicate_groups.append(
                    DuplicateGroup(
                        card_type=card_type,
                        payload_key=key,
                        card_ids=[m.id for m in members],
                        scene_ids=sorted({_scene_id_of(m, bundle) for m in members}),
                    )
                )

    # ---- 冲突组（相关卡片置 needs_review，不擅自择真） ----
    conflict_groups: list[ConflictGroup] = []
    conflicted_ids: set[str] = set()
    fact_conflict_index: dict[str, list[FactDocument]] = {}
    for doc in fact_docs:
        if doc.predicate in CONFLICT_PRONE_FACT_PREDICATES:
            fact_conflict_index.setdefault(f"{doc.subject}|{doc.predicate}", []).append(doc)
    for key, members in sorted(fact_conflict_index.items()):
        values = sorted({doc.value for doc in members})
        if len(values) >= 2:
            conflict_groups.append(
                ConflictGroup(
                    card_type="fact",
                    group_key=key,
                    card_ids=[doc.id for doc in members],
                    values=values,
                    description=f"{key} 存在多个不同取值: {' vs '.join(values)}",
                )
            )
            conflicted_ids.update(doc.id for doc in members)
    relation_conflict_index: dict[str, list[RelationDocument]] = {}
    relation_conflict_values: dict[str, set[str]] = {}
    for doc in relation_docs:
        if doc.relation not in CONFLICT_PRONE_RELATIONS:
            continue
        key = f"{doc.subject}|{doc.relation}"
        relation_conflict_index.setdefault(key, []).append(doc)
        relation_conflict_values.setdefault(key, set()).add(doc.target)
    for key, members in sorted(relation_conflict_index.items()):
        values = sorted(relation_conflict_values[key])
        if len(set(m.id for m in members)) >= 2 and len(values) >= 2:
            conflict_groups.append(
                ConflictGroup(
                    card_type="relation",
                    group_key=key,
                    card_ids=list(dict.fromkeys(doc.id for doc in members)),
                    values=values,
                    description=f"{key} 存在多个不同对象: {' vs '.join(values)}",
                )
            )
            conflicted_ids.update(doc.id for doc in members)

    for doc in [*fact_docs, *relation_docs, *event_docs]:
        if doc.id in conflicted_ids:
            doc.review_status = ReviewStatus.needs_review

    def sort_key(doc: Any) -> tuple[int, int, int, int, str]:
        scene_order = scene_order_by_id[_scene_id_of(doc, bundle)]
        type_rank = {"fact": 0, "relation": 1, "event": 2}[doc.document_type.value]
        return (scene_order, type_rank, doc.source.line_start, doc.source.line_end, doc.id)

    fact_docs.sort(key=sort_key)
    relation_docs.sort(key=sort_key)
    event_docs.sort(key=sort_key)

    return KnowledgeFinalization(
        fact_documents=fact_docs,
        relation_documents=relation_docs,
        event_documents=event_docs,
        duplicate_groups=duplicate_groups,
        conflict_groups=conflict_groups,
        scene_stats=scene_stats,
        deduped_cross_scene=len(removed_fact_scenes) + len(removed_relation_scenes),
    )


def _scene_id_of(doc: Any, bundle: EnrichedSceneBundle) -> str:
    """由 (story_unit + source span) 反查所属 scene_id（按范围匹配；键唯一）。"""
    spans = bundle._scene_span_index.get((doc.story.story_unit_id, doc.source.source_path), [])
    for line_start, line_end, scene_id in spans:
        if line_start <= doc.source.line_start <= line_end:
            return scene_id
    raise KeyError(
        f"无法定位卡片所属场景: unit={doc.story.story_unit_id} "
        f"path={doc.source.source_path} line_start={doc.source.line_start}"
    )


# ---------- 候选文档原子写出（JSONL） ----------


def save_documents_jsonl(path: Path | str, documents: list[Any]) -> None:
    """候选文档 JSONL 原子保存：一行一个 JSON 对象（键序为模型字段序，确定性）。"""
    lines = [json.dumps(doc.model_dump(mode="json"), ensure_ascii=False) for doc in documents]
    payload = ("\n".join(lines) + "\n") if lines else ""
    _atomic_write_text(Path(path), payload)


def load_documents_jsonl(path: Path | str) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            docs.append(json.loads(line))
    return docs


# ---------- 人工审核状态 ----------


class KnowledgeCardReview(BaseModel):
    """单卡人工审核追踪记录。"""

    model_config = ConfigDict(extra="forbid")

    card_id: NonEmptyStr
    document_type: Literal["fact", "relation", "event"]
    scene_id: NonEmptyStr
    review_status: ReviewStatus
    reviewer: str = ""
    notes: str = ""


class KnowledgeReviewDocument(BaseModel):
    """P5 人工审核状态：候选从 draft 起步，人工定稿后可整体 approved。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    source_enriched: EnrichedSourceRef
    total_candidates: int = Field(ge=0)
    reviewer: str
    review_status: Literal["draft", "approved"]
    card_reviews: list[KnowledgeCardReview]
    notes: str = ""
    created_by: NonEmptyStr = GENERATOR_ID

    @model_validator(mode="after")
    def _check_review_consistency(self) -> KnowledgeReviewDocument:
        if len(self.card_reviews) != self.total_candidates:
            raise ValueError("card_reviews 数量必须等于 total_candidates")
        ids = [item.card_id for item in self.card_reviews]
        if len(set(ids)) != len(ids):
            raise ValueError("card_reviews 含重复 card_id")
        if self.review_status == "approved":
            if not self.reviewer.strip():
                raise ValueError("approved 审核文档必须填写 reviewer")
            incomplete = [
                item.card_id
                for item in self.card_reviews
                if item.review_status not in {ReviewStatus.approved, ReviewStatus.rejected}
            ]
            if incomplete:
                raise ValueError(f"approved 审核文档仍有 {len(incomplete)} 张卡片未定稿")
            if any(not item.reviewer.strip() for item in self.card_reviews):
                raise ValueError("approved/rejected 卡片必须填写 reviewer")
        return self


def create_knowledge_review(
    bundle: EnrichedSceneBundle,
    finalization: KnowledgeFinalization,
    *,
    reviewer: str = "",
) -> KnowledgeReviewDocument:
    """从 finalize 结果创建人工审核状态（全部 draft；冲突卡 needs_review + 冲突说明）。"""
    conflict_note: dict[str, str] = {}
    for group in finalization.conflict_groups:
        for card_id in group.card_ids:
            conflict_note[card_id] = f"冲突组 {group.group_key}: 相关卡片 {'/'.join(group.card_ids)}"
    reviews: list[KnowledgeCardReview] = []
    for doc, doc_type in [
        *[(doc, "fact") for doc in finalization.fact_documents],
        *[(doc, "relation") for doc in finalization.relation_documents],
        *[(doc, "event") for doc in finalization.event_documents],
    ]:
        status = doc.review_status
        reviews.append(
            KnowledgeCardReview(
                card_id=doc.id,
                document_type=doc_type,
                scene_id=_scene_id_of(doc, bundle),
                review_status=status,
                reviewer=reviewer,
                notes=conflict_note.get(doc.id, ""),
            )
        )
    return KnowledgeReviewDocument(
        schema_version=KNOWLEDGE_REVIEW_SCHEMA_VERSION,
        source_enriched=bundle.source_ref,
        total_candidates=len(reviews),
        reviewer=reviewer,
        review_status="draft",
        card_reviews=reviews,
        notes=(
            "P5 知识卡人工审核状态：全部候选卡为 draft（冲突组卡片为 needs_review），"
            "等待人工逐卡复核；本阶段不产生 approved。"
        ),
    )


def save_knowledge_review(path: Path | str, review_doc: KnowledgeReviewDocument | dict[str, Any]) -> None:
    doc = (
        review_doc
        if isinstance(review_doc, KnowledgeReviewDocument)
        else KnowledgeReviewDocument.model_validate(review_doc)
    )
    payload = json.dumps(doc.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    _atomic_write_text(Path(path), payload)


def load_knowledge_review(path: Path | str) -> KnowledgeReviewDocument:
    return KnowledgeReviewDocument.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------- 运行 manifest（唯一非确定性产物） ----------


class KnowledgeRunManifest(BaseModel):
    """P5 运行 manifest：时间戳与模型调用统计只记录在这里。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    generator: NonEmptyStr
    model_id: NonEmptyStr
    generation_params: dict[str, int | float | str | bool]
    source_enriched: EnrichedSourceRef
    total_scenes: int = Field(ge=1)
    scene_status_counts: dict[str, int]
    card_counts: dict[str, int]
    attempted_scene_ids: list[str]
    model_call_stats: dict[str, int | float] = Field(default_factory=dict)
    started_at: str = ""
    completed_at: str = ""
    notes: str = DEFAULT_MANIFEST_NOTE


def build_knowledge_run_manifest(
    bundle: EnrichedSceneBundle,
    run_state: KnowledgeCandidateRunState,
    finalization: KnowledgeFinalization,
    *,
    attempted_scene_ids: list[str],
    model_call_stats: dict[str, int | float] | None = None,
    started_at: str = "",
    completed_at: str = "",
) -> KnowledgeRunManifest:
    """构建运行 manifest（先重校验状态，时间戳与调用统计由调用方注入）。"""
    errors = validate_knowledge_run(run_state, bundle)
    if errors:
        raise ValueError("候选运行状态未通过校验:\n- " + "\n- ".join(errors))
    status_counts = Counter(item.status.value for item in run_state.scene_states)
    return KnowledgeRunManifest(
        schema_version=KNOWLEDGE_RUN_MANIFEST_SCHEMA_VERSION,
        generator=GENERATOR_ID,
        model_id=run_state.model_id,
        generation_params=dict(run_state.generation_params),
        source_enriched=bundle.source_ref,
        total_scenes=len(bundle.scenes),
        scene_status_counts={key: int(value) for key, value in sorted(status_counts.items())},
        card_counts={
            "fact": len(finalization.fact_documents),
            "relation": len(finalization.relation_documents),
            "event": len(finalization.event_documents),
            "total": finalization.total_documents,
        },
        attempted_scene_ids=list(attempted_scene_ids),
        model_call_stats={str(k): v for k, v in (model_call_stats or {}).items()},
        started_at=started_at,
        completed_at=completed_at,
    )


def save_knowledge_run_with_manifest(
    state_path: Path | str,
    manifest_path: Path | str,
    run_state: KnowledgeCandidateRunState,
    manifest: KnowledgeRunManifest,
) -> None:
    """状态 + manifest 两文件整体原子提交（primary=状态，secondary=manifest 完成标志）。

    复用 P3/P4A 冻结对提交协议：primary 先备份旧版再替换，secondary 最后替换；
    secondary 失败时回滚 primary；发现既有 `.tmp.old` 恢复副本时拒绝覆盖。
    """
    state_path = Path(state_path)
    manifest_path = Path(manifest_path)
    if state_path.parent != manifest_path.parent:
        raise ValueError("状态文件与 manifest 必须位于同一目录（两文件整体提交协议要求）")
    if state_path == manifest_path:
        raise ValueError("状态文件与 manifest 必须使用两个不同路径")
    state, coerce_errors = _coerce_run_state(run_state)
    if state is None:
        raise ValueError("候选运行状态非法，拒绝写出:\n- " + "\n- ".join(coerce_errors))
    _atomic_write_pair(
        state_path.parent,
        primary_name=state_path.name,
        primary_payload=json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        secondary_name=manifest_path.name,
        secondary_payload=json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
    )


# ---------- 质量报告 ----------


def build_knowledge_quality_report(
    bundle: EnrichedSceneBundle,
    run_state: KnowledgeCandidateRunState,
    finalization: KnowledgeFinalization,
) -> dict[str, Any]:
    """确定性质量报告：状态计数、卡片分布、重复/冲突组、失败场景与门禁。"""
    errors = validate_knowledge_run(run_state, bundle)
    if errors:
        raise ValueError("候选运行状态未通过校验:\n- " + "\n- ".join(errors))
    status_counts = Counter(item.status.value for item in run_state.scene_states)
    unit_stats: dict[str, dict[str, int]] = {}
    for stat in finalization.scene_stats:
        unit = unit_stats.setdefault(stat.story_unit_id, {"scenes": 0, "facts": 0, "relations": 0, "events": 0})
        unit["scenes"] += 1
        unit["facts"] += stat.facts
        unit["relations"] += stat.relations
        unit["events"] += stat.events
    failed_scenes = [
        {
            "scene_id": item.scene_id,
            "error_kind": item.last_failure.error_kind if item.last_failure else "unknown",
            "attempts": item.attempts,
        }
        for item in run_state.scene_states
        if item.status is KnowledgeGenerationStatus.failed
    ]
    empty_scenes = [
        stat.scene_id for stat in finalization.scene_stats if stat.facts + stat.relations + stat.events == 0
    ]
    total_attempts = sum(item.attempts for item in run_state.scene_states)
    min_calls = sum(
        item.chunk_count for item in run_state.scene_states if item.status is KnowledgeGenerationStatus.success
    )
    all_docs = [*finalization.fact_documents, *finalization.relation_documents, *finalization.event_documents]
    broad_evidence_counts = {
        "fact": sum(
            doc.source.line_end - doc.source.line_start + 1 > MAX_FACT_RELATION_EVIDENCE_LINES
            for doc in finalization.fact_documents
        ),
        "relation": sum(
            doc.source.line_end - doc.source.line_start + 1 > MAX_FACT_RELATION_EVIDENCE_LINES
            for doc in finalization.relation_documents
        ),
        "event": sum(
            doc.source.line_end - doc.source.line_start + 1 > MAX_EVENT_EVIDENCE_LINES
            for doc in finalization.event_documents
        ),
    }
    broad_evidence_counts["total"] = sum(broad_evidence_counts.values())
    return {
        "schema_version": 1,
        "generator": GENERATOR_ID,
        "model_id": run_state.model_id,
        "generation_params": dict(run_state.generation_params),
        "source_enriched": bundle.source_ref.model_dump(mode="json"),
        "scene_status_counts": {key: int(value) for key, value in sorted(status_counts.items())},
        "model_calls": {
            "total_attempts": total_attempts,
            "minimum_required_calls": min_calls,
            "note": "total_attempts 为状态中逐场景累计模型调用（含分片与重试）；minimum_required_calls 为成功场景分片各一次调用的理论下限。",
        },
        "card_counts": {
            "fact": len(finalization.fact_documents),
            "relation": len(finalization.relation_documents),
            "event": len(finalization.event_documents),
            "total": finalization.total_documents,
        },
        "per_unit": dict(sorted(unit_stats.items())),
        "empty_scenes": empty_scenes,
        "capped_scenes": [stat.model_dump(mode="json") for stat in finalization.scene_stats if stat.dropped_by_cap > 0],
        "deduped_in_scene_total": sum(stat.deduped_in_scene for stat in finalization.scene_stats),
        "deduped_cross_scene_total": finalization.deduped_cross_scene,
        "dropped_invalid_total": sum(stat.dropped_invalid for stat in finalization.scene_stats),
        "scenes_with_dropped_invalid": [
            stat.model_dump(mode="json") for stat in finalization.scene_stats if stat.dropped_invalid > 0
        ],
        "duplicate_groups": [group.model_dump(mode="json") for group in finalization.duplicate_groups],
        "duplicate_group_count": len(finalization.duplicate_groups),
        "conflict_groups": [group.model_dump(mode="json") for group in finalization.conflict_groups],
        "conflict_group_count": len(finalization.conflict_groups),
        "failed_scenes": failed_scenes,
        "evidence_quality": {
            "fact_relation_max_lines": MAX_FACT_RELATION_EVIDENCE_LINES,
            "event_max_lines": MAX_EVENT_EVIDENCE_LINES,
            "overbroad_counts": broad_evidence_counts,
            "note": "旧模型运行使用整段范围示例，过宽 evidence 仅作风险标记；候选不得据此直接批准。",
        },
        "reality_status_distribution": dict(Counter(doc.reality_status.value for doc in all_docs)),
        "predicate_distribution": dict(Counter(doc.predicate for doc in finalization.fact_documents)),
        "relation_distribution": dict(Counter(doc.relation for doc in finalization.relation_documents)),
        "gates": {
            "no_approved_cards": not any(doc.review_status is ReviewStatus.approved for doc in all_docs),
            "review_statuses": dict(Counter(doc.review_status.value for doc in all_docs)),
            "embedding_generated": False,
        },
    }
