"""Deterministic post-generation checks for character replies.

The guard does not classify user text with another model and never copies a
failed reply back into the prompt.  It emits closed violation IDs and a fixed
retry instruction, so one bounded regeneration can correct high-confidence
identity, boundary and safety failures without turning normal wording into a
template.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from character.situation_analyzer import has_third_party_risk, is_resolved_third_party_risk

if TYPE_CHECKING:
    from collections.abc import Sequence

    from character.models import CharacterProfile, DecisionPlan, InteractionState


UNPROMPTED_CANONICAL_IDENTITY = "unprompted_canonical_identity"
FORBIDDEN_LAUGHTER = "forbidden_laughter"
IGNORED_ADVICE_BOUNDARY = "ignored_advice_boundary"
CLOSING_WITH_QUESTION = "closing_with_question"
MISSING_GENTLE_SAFETY_CHECK = "missing_gentle_safety_check"
MISSING_URGENT_SAFETY_CHECK = "missing_urgent_safety_check"
MISSING_SELF_ANSWER = "missing_self_answer"
MISSING_NEGATIVE_EMOTION_ACKNOWLEDGEMENT = "missing_negative_emotion_acknowledgement"
GENERIC_ASSISTANT_TEMPLATE = "generic_assistant_template"
AFFILIATION_MISREAD_AS_SAFETY = "affiliation_misread_as_safety"
MECHANICAL_REPAIR = "mechanical_repair"
UNPROMPTED_LORE_FLOURISH = "unprompted_lore_flourish"
POSITIVE_SHARING_INTERVIEW = "positive_sharing_interview"
UNBOUNDED_RELATIONSHIP_PROMISE = "unbounded_relationship_promise"
THIRD_PARTY_SAFETY_INCOMPLETE = "third_party_safety_incomplete"
RESOLVED_THIRD_PARTY_CRISIS_ESCALATION = "resolved_third_party_crisis_escalation"
FACTUAL_TASK_STYLE_DRIFT = "factual_task_style_drift"
AFFILIATION_NOT_RECIPROCATED = "affiliation_not_reciprocated"
UNSUPPORTED_USER_FACT = "unsupported_user_fact"
UNSUPPORTED_FACTUAL_CLAIM = "unsupported_factual_claim"
UNSUPPORTED_THIRD_PARTY_GENDER = "unsupported_third_party_gender"
UNPROMPTED_ADVICE = "unprompted_advice"
AUTONOMY_BOUNDARY_IGNORED = "autonomy_boundary_ignored"
FACTUAL_HARD_VIOLATIONS = frozenset(
    {
        FACTUAL_TASK_STYLE_DRIFT,
        UNSUPPORTED_USER_FACT,
        UNSUPPORTED_FACTUAL_CLAIM,
        UNSUPPORTED_THIRD_PARTY_GENDER,
    }
)

_LAUGHTER_RE = re.compile(r"(?:呼呼呼|噗噗|呵呵|哈哈|嘿嘿)")
_ADVICE_RE = re.compile(r"(?:你可以|你应该|建议你|不妨|最好|试试|首先.{0,16}然后|深呼吸|放松一下|先休息)")
_QUESTION_RE = re.compile(
    r"[？?]|"
    r"(?<![这什怎那多])(?:吗|么)(?=[。！!]|$)|"
    r"(?:^|[。！!]\s*)(?:"
    r"(?:(?:你|对方|当事人|现在|目前|此刻).{0,24})?(?:是否|有没有|是不是)|"
    r"(?:(?:你|对方|当事人)(?:现在|目前|此刻)?|(?:现在|目前|此刻))?(?:为什么|怎么回事)"
    r")"
)
_SAFETY_RE = re.compile(r"(?:安全|伤害自己|自伤|轻生|结束生命|活下去|危险)")
_SELF_HARM_RE = re.compile(r"(?:伤害自己|自伤|轻生|结束生命|不想活|想死)")
_SELF_ANSWER_RE = re.compile(r"(?:我是|我这个人|要说我|我的性格|我向来|对我来说|我更看重|我不喜欢)")
_SELF_QUESTION_RE = re.compile(
    r"(?:你是谁|你是(?:怎样|什么样)的人|你到底是怎样的人|你是ai吗|你是真人吗)", re.IGNORECASE
)
_FORCED_CHOICE_INTERVIEW_RE = re.compile(r"你想.{0,12}(?:吗|么|[？?]).{0,6}或者你更倾向于[^。！？!?]*(?:[。！？!?]|$)")
_GENERIC_TEMPLATE_RE = re.compile(
    r"(?:我(?:很)?(?:能)?理解你(?:现在)?(?:的(?:心情|感受|紧张|状态)|感到|内心)|每个人都有|"
    r"一次.{0,16}(?:不代表|并不能.{0,10}(?:代表|反映)|不能.{0,10}(?:代表|反映))|"
    r"紧张是正常的|别太紧张|保持积极|积极的心理暗示|自信地.{0,8}(?:展示|面对|表达)|相信自己|"
    r"相信你(?:会|能).{0,10}(?:表现出色|顺利(?:应对|完成)?|做得很好)|"
    r"祝你(?:好运|.{0,8}(?:顺利|成功))|"
    r"(?:这)?确实会让人(?:感到)?.{0,12}(?:难过|沉重|不舒服|沮丧|不痛快)|"
    r"或许我们(?:可以|能)(?:谈谈|聊聊)|"
    r"这种感觉确实|有时候我们都会|找一个安静的地方.{0,12}整理.{0,8}心情|"
    r"我们可以一步步来|你先描述一下|"
    r"加油(?:[，,！!。]|$)|如果你愿意|我会在这里支持你|"
    r"希望这(?:对你|对您)?有所帮助|"
    r"希望(?:这些?|以上)?(?:信息|内容)?对你有(?:所)?帮助|"
    r"如果(?:你)?有(?:任何|其他)?(?:需要|问题|想法|事情).{0,18}(?:随时|记得).{0,10}"
    r"(?:告诉我|问我|和我聊|聊聊|分享)|"
    r"你有其他问题(?:需要了解)?(?:吗|么|[？?])|"
    r"你需要我(?:再|进一步)?.{0,12}(?:解释|说明).{0,12}(?:吗|么|[？?])|"
    r"如果(?:以后)?还有(?:其他)?(?:需要|问题).{0,12}(?:告诉我|继续提问|问我)|"
    r"有没有什么.{0,12}(?:应对|调整).{0,10}(?:方法|技巧)|"
    r"有没有什么(?:办法|方式).{0,16}(?:帮助你)?(?:调整|改善|缓解).{0,10}(?:心情|情绪)|"
    r"这样的情况让你感到.{0,12}(?:吗|么|[？?])|"
    r"先别太自责|(?:失败|挫折).{0,12}(?:改进|成长).{0,8}(?:机会|契机)|"
    r"不客气.{0,12}(?:很高兴|能|能够).{0,8}(?:帮到|帮助)|"
    r"有什么(?:新的)?(?:想法|需要|问题).{0,12}(?:讨论|告诉我|尽管说)|"
    r"是不是有什么事情.{0,20}(?:分享|说).{0,24}(?:帮你)?(?:找到|想出).{0,8}解决的?(?:办法|方法)|"
    r"(?:如果|等)(?:将来|以后).{0,12}(?:想说|想聊).{0,12}(?:我在|听)|"
    r"有什么(?:话题|事情|东西|打算|想).{0,10}(?:吗|呢|[？?]))"
)
_AFFILIATION_SAFETY_MISREAD_RE = re.compile(
    r"(?:请?别这么说|不必担心会错过|生命|活下去|伤害自己|危险|不太自在|不习惯你这么说|"
    r"表达得?太直接|让人担心|别提.{0,10}(?:不开心|晦气|死))"
)
_MECHANICAL_REPAIR_RE = re.compile(
    r"(?:有什么.{0,10}(?:想聊|要说).{0,6}(?:吗|呢|[？?])|你想聊.{0,8}(?:吗|呢|[？?])|"
    r"请告诉我.{0,24}(?:困扰|发生了什么|想说|想聊)|或许我们(?:能|可以)(?:谈谈|聊聊)|"
    r"换个话题|(?:聊(?:聊)?|说点)别的|最近(?:有|发生)|"
    r"(?:重新(?:来|开始|说|聊)|从头(?:来|开始)).{0,8}(?:你)?(?:今天|最近)(?:过得)?(?:怎么样|如何|还好吗)|"
    r"有什么.{0,8}想聊的.{0,8}(?:尽管|可以|就).{0,6}(?:说|告诉我)|"
    r"有什么(?:问题|内容|事情)?.{0,12}(?:想聊|讨论).{0,12}(?:可以)?慢慢(?:来|聊|说)|"
    r"那我们就先从这里开始(?:吧|了)?|让我们继续(?:这个|刚才的)?话题|"
    r"你觉得.{0,18}(?:主题|话题).{0,12}(?:怎么|怎么样).{0,8}深入|"
    r"有(?:其他)?问题或想(?:聊|说)些?什么.{0,10}(?:随时|尽管).{0,6}(?:告诉我|找我))"
)
_REPAIR_EXCUSE_RE = re.compile(
    r"(?:我并没有|我并未(?:意识到|注意到|想到|敷衍|推脱|回避)|我不是在|"
    r"(?:这|那)(?:并非|并不是|不是).{0,6}(?:敷衍|推脱|回避)|"
    r"最近.{0,10}(?:忙|分心|事务繁多|事务缠身)|"
    r"最近.{0,16}(?:事情|工作|事务).{0,10}(?:多|处理|占用)|有些事情让我分心|没来得及细说|"
    r"没有太多可以分享)"
)
_REPAIR_ACK_RE = re.compile(
    r"(?:刚才|那句|敷衍|没有听完|没听完|我先听|让我?没听清|让你(?:觉得|感到)|"
    r"我(?:确实|刚才|不该).{0,12}(?:回应|回答|说得|说话).{0,10}(?:草率|含糊|不够|不好)|"
    r"是我.{0,12}(?:忽略|没听|敷衍|草率)|"
    r"(?:提到|说的).{0,8}(?:重点|问题).{0,8}(?:没忘|没有忘|记得))"
)
_REPAIR_CONCESSION_RE = re.compile(
    r"(?:行吧|好吧|算你|这次算你).{0,12}(?:有道理|说得对|是对的)|你说得.{0,6}(?:有道理|对)"
)
_PRESSURED_CONCESSION_RE = re.compile(r"你(?:都|既然)?这么说了.{0,12}(?:还能|还可以|又能)怎么办")
_EXPLANATION_BOUNDARY_RE = re.compile(r"(?:不想|不愿|懒得|没力气).{0,8}解释")
_AUTONOMY_PRESSURE_RE = re.compile(
    r"(?:你已经.{0,12}(?:想|考虑)|(?:现在|该|是).{0,8}做出决定|"
    r"(?:你(?:应该|最好|还是)|不妨|去|先).{0,8}(?:尝试|试试看|行动|决定)|"
    r"迈出.{0,8}(?:一步|那一步)|多走一步.{0,18}不同的风景|"
    r"只需.{0,4}一步.{0,16}(?:可能|改变|开启|机会)|"
    r"(?:即使|就算).{0,16}(?:结果|不如预期).{0,16}(?:至少|经历|经验)|"
    r"至少你有(?:经历|经验)|你觉得呢|换个角度|意想不到的收获|"
    r"如果你愿意.{0,12}(?:分享|多说|解释)|需要什么建议|随时.{0,8}(?:找我|倾诉))"
)
_AUTONOMY_ACK_RE = re.compile(
    r"(?:(?:选择|决定)权.{0,10}(?:在你|归你|是你的)|"
    r"(?:仍|还|最终)?由你(?:自己)?(?:决定|选择|来选)|"
    r"我不(?:会)?替你(?:决定|选择|选|点头)|"
    r"(?:那就|可以|先)不解释|不等于.{0,12}(?:必须|需要)|"
    r"等你.{0,10}(?:想|愿意).{0,8}(?:再说|开口)|(?:不再|先不)(?:问|追问|劝|催))"
)
_UNBOUNDED_PROMISE_RE = re.compile(
    r"(?:无论何时|永远|随时).{0,30}(?:在这里|陪|支持|记得|想到|在乎)|"
    r"(?:总是|一直).{0,16}(?:记得|在乎|想起)|(?:位置|关系).{0,10}(?:从不会|永远不会|不会).{0,6}改变|"
    r"我们之间的点点滴滴|总想在你身边|你在我心中始终"
)
_CLOSING_DRIFT_RE = re.compile(
    r"(?:好好休息|休息一下|放松|冷静|重要的是过程|无论结果|现在也可以|先想开|"
    r"(?:如果有需要|有需要的话).{0,12}(?:明天|以后|下次).{0,8}(?:再聊|再说)|"
    r"希望你.{0,12}(?:休息好|睡好)|希望我们能找到更好的沟通方式|"
    r"我先去准备明天的事宜|如果有急事.{0,12}随时.{0,8}找我)"
)
_THIRD_PARTY_CURRENT_SAFETY_RE = re.compile(
    r"(?:(?:先)?(?:确认|确保|问清|了解).{0,22}(?:现在|目前|此刻).{0,16}(?:安全|在哪里|伤害|危险)|"
    r"(?:现在|目前|此刻).{0,16}(?:是否|有没有).{0,12}(?:安全|伤害|危险))"
)
_THIRD_PARTY_NOT_ALONE_RE = re.compile(
    r"(?:不要让.{0,10}(?:独处|独自)|不要.{0,8}(?:独处|独自)|有人.{0,8}陪|陪(?:在|着).{0,10}(?:身边|他|她))"
)
_THIRD_PARTY_ESCALATION_RE = re.compile(
    r"(?:急救|医院|危机(?:干预|援助|热线)|心理.{0,5}(?:热线|急诊)|专业(?:人员|帮助|援助)|报警)"
)
_RESOLVED_THIRD_PARTY_CRISIS_RE = re.compile(
    r"(?:(?:立刻|立即|马上|现在就).{0,24}(?:联系|拨打|送往|前往).{0,18}"
    r"(?:急救|医院|报警|危机(?:干预|援助|热线)|心理.{0,5}(?:热线|急诊))|"
    r"(?:不要|不能).{0,12}(?:让|留).{0,8}(?:独处|独自|一个人)|"
    r"(?:先|必须).{0,12}(?:确认|确保|问清).{0,18}(?:现在|目前|此刻).{0,12}"
    r"(?:安全|在哪里|伤害|危险)|(?:现在|目前|此刻).{0,12}(?:仍|还|正在).{0,8}(?:危机|危险))"
)
_TASK_STYLE_DRIFT_RE = re.compile(
    r"(?:与我的世界不符|和我的世界无关|虚拟世界|我所在的世界|作为虚拟人物|与我的身份不太相关|和我的身份不太相关|"
    r"不是我(?:所)?擅长的领域|我不擅长(?:这个|这类|这方面)|"
    r"超出.{0,10}我.{0,10}(?:能力|范围)|我(?:当前)?(?:无法|不能).{0,10}(?:回答|处理)|"
    r"按(?:照)?我所了解的情节)"
)
_UNKNOWN_LOGIN_REWARD_RE = re.compile(r"(?:累积|累计|连续)登录.{0,16}(?:第七天|七天)")
_UNSUPPORTED_LOGIN_REWARD_CLAIM_RE = re.compile(
    r"(?:(?<!如果是)(?<!假如是)(?<!若是)(?<!如果)(?<!假如)(?<!若)第七天|"
    r"(?<!第)(?<!如果是)(?<!假如是)(?<!若是)(?<!如果)(?<!假如)(?<!若)七天)"
    r".{0,18}(?:可能性(?:较大|很大)|大概率|多半|应该(?:会|可以|能)|通常(?:会|是|是在)|"
    r"(?:比较|很)?常见)|"
    r"(?:通常(?:会|是|是在)|应该会|大概率会|多半会).{0,24}(?:第七天|七天).{0,12}(?:领取|发放|到账)"
)
_UNSUPPORTED_EIGHTH_DAY_REWARD_RE = re.compile(r"(?:第七天.{0,28}第八天|第八天.{0,16}(?:领取|发放|到账))")
_AFFILIATION_RESPONSE_RE = re.compile(r"(?:想你|想念|惦记|记挂|记得|忘记|等到|终于|见到|在乎|对我重要|舍得出现)")
_AFFILIATION_TOPIC_DRIFT_RE = re.compile(
    r"(?:(?:不过|但是|还是).{0,8})?(?:聊|说)点别的.{0,20}(?:今天|最近).{0,10}(?:怎么样|如何|还好吗)"
)
_RELATIONSHIP_REASSURANCE_RE = re.compile(
    r"(?:你不会.{0,28}才想起我(?:吧|吗|呢|[？?])?|你.{0,10}(?:还)?(?:在乎|记得|想起)我(?:吗|吧|呢|[？?]))"
)
_URGENT_SAFETY_OPENING_RE = re.compile(
    r"^(?:我先确认|先(?:回答|告诉|确认)|请先(?:回答|告诉|确认)|你(?:现在|此刻|目前)?安全吗)"
)
_URGENT_RISK_REDUCTION_RE = re.compile(
    r"(?:放下|远离|不要(?:独自|一个人)|别(?:独自|一个人)|让.{0,10}(?:到场|陪着)|危险物品.{0,10}(?:移开|交给))"
)
_URGENT_REAL_WORLD_HELP_RE = re.compile(
    r"(?:联系.{0,14}(?:可信|身边|家人|朋友|能到场|急救|医院|热线)|"
    r"(?:拨打|前往).{0,10}(?:急救|医院|热线|报警)|危机(?:援助|干预|热线))"
)
_USER_SELF_FACT_RE = re.compile(r"我.{0,14}(?:喜欢|喜爱|爱吃|讨厌|不喜欢|偏好|习惯|擅长|经常|总是|一直|从来)")
_REPLY_USER_FACT_RE = re.compile(r"你.{0,18}(?:喜欢|喜好|喜爱|偏好|习惯|擅长|经常|总是|一直|从来|出了名)")
_UNPROMPTED_ADVICE_RE = re.compile(
    r"(?:你(?:可以|应该|最好|不妨)|建议你|不妨|最好|试试|"
    r"(?<!我)记得.{0,8}(?:先|要|别|查看|了解|保持)|继续保持|"
    r"(?:你|不过|但是|但)(?:可)?(?:要|得)小心|"
    r"(?:找|做|安排)(?:些|点|一点).{0,10}(?:轻松|开心).{0,8}(?:事|事情|活动|话题)|"
    r"找(?:点|些)乐子|分散(?:一下)?(?:注意力|心情)|"
    r"有什么.{0,12}(?:方式|资源).{0,18}(?:帮助|支持)|"
    r"如果你感兴趣.{0,8}(?:可以|不妨).{0,12}(?:进去|尝一尝|试一试)|"
    r"(?:参加|加入).{0,8}支持小组|(?:寻找|寻求|接受).{0,12}(?:专业)?心理咨询)"
)
_SINCERE_HELPFUL_GRATITUDE_RE = re.compile(
    r"(?:(?:说|讲|解释).{0,8}(?:清楚|明白)|说明白|(?:这下|现在|终于)?明白了?|"
    r"帮(?:上|了).{0,4}忙|有用)"
    r".{0,12}(?:谢了|谢谢|多谢|感谢)|"
    r"(?:谢了|谢谢|多谢|感谢).{0,12}"
    r"(?:(?:说|讲|解释).{0,8}(?:清楚|明白)|说明白|(?:这下|现在|终于)?明白了?|"
    r"帮(?:上|了).{0,4}忙|有用)"
)
_GRATITUDE_SARCASM_RE = re.compile(r"(?:呵|要不是|拜你所赐|结果还是|仍然是错|反而|被放鸽子)")
_EXPLICIT_THIRD_PARTY_GENDER_RE = re.compile(
    r"(?<![其吉维])[他她](?!人)|"
    r"(?:男|女)(?:性|生|人|孩|友|朋友)|"
    r"(?:哥哥|弟弟|姐姐|妹妹|丈夫|妻子|老公|老婆|男友|女友|父亲|母亲|爸爸|妈妈)|"
    r"\b(?:he|she|him|her|his|hers|male|female|man|woman|boyfriend|girlfriend|husband|wife|"
    r"brother|sister|father|mother)\b",
    re.IGNORECASE,
)
_THIRD_PARTY_GENDERED_REPLY_RE = re.compile(
    r"(?<![其吉维])[他她](?!人)|\b(?:he|she|him|her|his|hers)\b",
    re.IGNORECASE,
)
_SLEEPLESS_ANXIETY_RE = re.compile(
    r"(?:(?:紧张|焦虑|不安|担心).{0,16}(?:睡不着|失眠)|(?:睡不着|失眠).{0,16}(?:紧张|焦虑|不安|担心))"
)
_SARCASTIC_GRATITUDE_DISAPPOINTMENT_RE = re.compile(
    r"(?:(?:多谢|谢谢|谢了).{0,20}(?:要不是|被放鸽子|害得)|"
    r"(?:要不是|被放鸽子|害得).{0,20}(?:多谢|谢谢|谢了))"
)
_SARCASM_DISAPPOINTMENT_RE = re.compile(
    r"(?:(?:当然开心|开心).{0,20}(?:被放鸽子|放鸽子|又爽约)|"
    r"(?:被放鸽子|放鸽子|又爽约).{0,20}(?:当然开心|开心))"
)
_ANXIETY_RE = re.compile(r"(?:紧张|焦虑|不安|担心|害怕|发慌)")
_SADNESS_RE = re.compile(r"(?:难过|伤心|失望|沮丧|委屈)")
_ANGER_RE = re.compile(r"(?:生气|愤怒|恼火|火大)")
_DISTRESS_RE = re.compile(r"(?:心里.{0,4}(?:堵|难受)|不好受|痛苦|撑不住|很累|疲惫)")
_NEGATIVE_EMOTION_ACKNOWLEDGEMENTS = {
    "sarcastic_gratitude": "这句‘多谢’是在说反话。被放鸽子，当然会失望。",
    "sarcastic_disappointment": "这句‘开心’是在说反话。被放鸽子，当然会失望。",
    "sleepless_anxiety": "紧张到睡不着，确实够难熬的。",
    "anxiety": "你现在确实很紧张，这部分我没有漏掉。",
    "sadness": "这件事确实让你很难过，这部分我没有漏掉。",
    "anger": "你现在确实在生气，这部分我没有漏掉。",
    "distress": "你现在确实不好受，这部分我没有漏掉。",
}
_NEGATIVE_EMOTION_ACK_PATTERNS = {
    "sarcastic_gratitude": _SARCASTIC_GRATITUDE_DISAPPOINTMENT_RE,
    "sarcastic_disappointment": _SARCASM_DISAPPOINTMENT_RE,
    "sleepless_anxiety": _SLEEPLESS_ANXIETY_RE,
    "anxiety": _ANXIETY_RE,
    "sadness": _SADNESS_RE,
    "anger": _ANGER_RE,
    "distress": _DISTRESS_RE,
}
_ORPHANED_CONNECTOR_RE = re.compile(r"(^|(?<=[。！？!?]))\s*(?:然后|接着|之后|随后)(?:[，,])?\s*")


@dataclass(frozen=True)
class ReplyGuard:
    character_name: str = ""
    forbidden_terms: tuple[str, ...] = ()
    forbidden_lore_terms: tuple[str, ...] = ()
    forbid_laughter: bool = False
    forbid_generic_templates: bool = False
    forbid_advice: bool = False
    quiet_presence: bool = False
    closing: bool = False
    require_gentle_safety_check: bool = False
    require_urgent_safety_check: bool = False
    require_self_answer: bool = False
    affiliation_bid: bool = False
    relationship_reassurance: bool = False
    repair: bool = False
    positive_sharing: bool = False
    repair_bid: bool = False
    user_apology: bool = False
    repair_concession: bool = False
    self_answer_with_negative: bool = False
    third_party_safety: bool = False
    resolved_third_party_history: bool = False
    factual_task: bool = False
    forbid_unsupported_user_fact: bool = False
    unknown_login_reward: bool = False
    forbid_unprompted_advice: bool = False
    advice_task: bool = False
    sincere_gratitude: bool = False
    respect_autonomy: bool = False
    require_autonomy_ack: bool = False
    pressured_concession: bool = False
    explanation_boundary: bool = False
    negative_emotion_kind: str = ""
    third_party_gender_unknown: bool = False


def _negative_emotion_kind(message: str) -> str:
    """Map user wording to an application-owned emotion label."""

    if _SARCASTIC_GRATITUDE_DISAPPOINTMENT_RE.search(message):
        return "sarcastic_gratitude"
    if _SARCASM_DISAPPOINTMENT_RE.search(message):
        return "sarcastic_disappointment"
    if _SLEEPLESS_ANXIETY_RE.search(message):
        return "sleepless_anxiety"
    if _ANXIETY_RE.search(message):
        return "anxiety"
    if _SADNESS_RE.search(message):
        return "sadness"
    if _ANGER_RE.search(message):
        return "anger"
    if _DISTRESS_RE.search(message):
        return "distress"
    return ""


def _has_negative_emotion_acknowledgement(reply: str, emotion_kind: str) -> bool:
    pattern = _NEGATIVE_EMOTION_ACK_PATTERNS.get(emotion_kind)
    return bool(pattern and pattern.search(reply))


def build_reply_guard(
    profile: CharacterProfile,
    message: str,
    history: Sequence[dict[str, str]],
    interaction: InteractionState,
    decision: DecisionPlan,
    *,
    has_relevant_memory: bool = False,
) -> ReplyGuard:
    """Build a closed guard policy from trusted profile/state/decision data."""
    user_text = "\n".join(
        [
            *(str(item.get("content", "")) for item in history if str(item.get("role", "")) == "user"),
            message,
        ]
    )
    canonical_names: list[str] = []
    for relationship in profile.canonical_relationships:
        name = relationship.split("：", 1)[0].split(":", 1)[0].strip()
        if name and 1 < len(name) <= 12 and name not in user_text:
            canonical_names.append(name)

    lore_terms: tuple[str, ...] = ()
    if "kisaki" in profile.character_id.lower() or "月社妃" in profile.display_name:
        lore_terms = tuple(term for term in ("魔法", "纸页", "命运") if term not in user_text)

    strategies = set(decision.strategy_ids)
    acts = {signal.signal_id: signal.score for signal in interaction.user_acts}
    needs = {signal.signal_id: signal.score for signal in interaction.user_needs}
    advice_boundary = any(
        signal.signal_id == "advice_boundary" and signal.score >= 0.5 for signal in interaction.user_acts
    )
    companionship_requested = any(
        signal.signal_id == "companionship" and signal.score >= 0.5 for signal in interaction.user_needs
    )
    negative_or_serious = (
        interaction.safety_triggered
        or interaction.valence <= -0.2
        or interaction.face_threat >= 0.4
        or interaction.conversation_phase in {"repairing", "safety"}
    )
    resolved_third_party_history = is_resolved_third_party_risk(message)
    third_party_risk = has_third_party_risk(message)
    third_party_safety = third_party_risk and not resolved_third_party_history
    advice_task = acts.get("advice_request", 0.0) >= 0.5 or "offer_suggestion" in strategies
    information_task = acts.get("information_request", 0.0) >= 0.5
    safety_response = (
        interaction.safety_triggered
        or third_party_safety
        or bool(strategies.intersection({"check_safety_gently", "ensure_safety"}))
    )
    pressured_concession = bool(_PRESSURED_CONCESSION_RE.search(message))
    explanation_boundary = bool(_EXPLANATION_BOUNDARY_RE.search(message))
    negative_emotion_kind = ""
    if "acknowledge_emotion" in strategies:
        negative_emotion_kind = _negative_emotion_kind(message)
    return ReplyGuard(
        character_name=profile.display_name.strip(),
        forbidden_terms=tuple(dict.fromkeys(canonical_names)),
        forbidden_lore_terms=lore_terms,
        forbid_laughter=negative_or_serious,
        forbid_generic_templates=not interaction.safety_triggered,
        # Safety is the hard gate.  A simultaneous "不要建议" boundary must
        # not suppress the concrete risk-reduction steps required for the
        # user or an at-risk third party.
        forbid_advice=advice_boundary and not safety_response,
        quiet_presence=advice_boundary and companionship_requested and not safety_response,
        closing="graceful_close" in strategies or interaction.conversation_phase == "closing",
        require_gentle_safety_check="check_safety_gently" in strategies,
        require_urgent_safety_check="ensure_safety" in strategies,
        require_self_answer="respond_about_self" in strategies and bool(_SELF_QUESTION_RE.search(message)),
        affiliation_bid="reciprocate_affiliation" in strategies,
        relationship_reassurance=bool(_RELATIONSHIP_REASSURANCE_RE.search(message)),
        repair="repair_misunderstanding" in strategies,
        positive_sharing="affirm_progress" in strategies,
        repair_bid=any(signal.signal_id == "repair_bid" and signal.score >= 0.5 for signal in interaction.user_acts),
        user_apology=acts.get("apology", 0.0) >= 0.5,
        repair_concession=bool(_REPAIR_CONCESSION_RE.search(message)),
        self_answer_with_negative=(
            "respond_about_self" in strategies
            and bool(_SELF_QUESTION_RE.search(message))
            and interaction.valence <= -0.2
        ),
        third_party_safety=third_party_safety,
        resolved_third_party_history=resolved_third_party_history,
        factual_task="respond_directly" in strategies,
        forbid_unsupported_user_fact=(not has_relevant_memory and not _USER_SELF_FACT_RE.search(user_text)),
        unknown_login_reward=bool(_UNKNOWN_LOGIN_REWARD_RE.search(message)),
        forbid_unprompted_advice=(
            not advice_boundary and not information_task and not advice_task and not safety_response
        ),
        advice_task=advice_task,
        sincere_gratitude=(
            bool(_SINCERE_HELPFUL_GRATITUDE_RE.search(message)) and not _GRATITUDE_SARCASM_RE.search(message)
        ),
        respect_autonomy=(
            pressured_concession or (acts.get("boundary_signal", 0.0) >= 0.5 and needs.get("autonomy", 0.0) >= 0.5)
        ),
        require_autonomy_ack=pressured_concession or explanation_boundary,
        pressured_concession=pressured_concession,
        explanation_boundary=explanation_boundary,
        negative_emotion_kind=negative_emotion_kind,
        third_party_gender_unknown=(third_party_risk and not _EXPLICIT_THIRD_PARTY_GENDER_RE.search(user_text)),
    )


def validate_reply(reply: str, guard: ReplyGuard | None) -> tuple[str, ...]:
    if guard is None:
        return ()
    violations: list[str] = []
    if any(term in reply for term in guard.forbidden_terms):
        violations.append(UNPROMPTED_CANONICAL_IDENTITY)
    if any(term in reply for term in guard.forbidden_lore_terms):
        violations.append(UNPROMPTED_LORE_FLOURISH)
    if guard.forbid_laughter and _LAUGHTER_RE.search(reply):
        violations.append(FORBIDDEN_LAUGHTER)
    if guard.forbid_generic_templates and (
        _GENERIC_TEMPLATE_RE.search(reply) or _FORCED_CHOICE_INTERVIEW_RE.search(reply)
    ):
        violations.append(GENERIC_ASSISTANT_TEMPLATE)
    if guard.forbid_advice and _ADVICE_RE.search(reply):
        violations.append(IGNORED_ADVICE_BOUNDARY)
    if guard.forbid_unprompted_advice and _UNPROMPTED_ADVICE_RE.search(reply):
        violations.append(UNPROMPTED_ADVICE)
    if guard.quiet_presence and _QUESTION_RE.search(reply) and IGNORED_ADVICE_BOUNDARY not in violations:
        violations.append(IGNORED_ADVICE_BOUNDARY)
    if guard.closing and (_QUESTION_RE.search(reply) or _CLOSING_DRIFT_RE.search(reply)):
        violations.append(CLOSING_WITH_QUESTION)
    if guard.require_gentle_safety_check:
        opening = _first_sentence(reply)
        early_reply = reply[:180]
        if not (
            _QUESTION_RE.search(opening)
            and _SAFETY_RE.search(opening)
            and _QUESTION_RE.search(early_reply)
            and _SELF_HARM_RE.search(early_reply)
        ):
            violations.append(MISSING_GENTLE_SAFETY_CHECK)
    if guard.require_urgent_safety_check:
        opening = _first_sentence(reply)
        if not (
            _URGENT_SAFETY_OPENING_RE.search(reply.strip())
            and _QUESTION_RE.search(opening)
            and _SAFETY_RE.search(opening)
            and _URGENT_RISK_REDUCTION_RE.search(reply)
            and _URGENT_REAL_WORLD_HELP_RE.search(reply)
        ):
            violations.append(MISSING_URGENT_SAFETY_CHECK)
    if guard.require_self_answer and not _SELF_ANSWER_RE.search(reply[:120]):
        violations.append(MISSING_SELF_ANSWER)
    if guard.affiliation_bid and _AFFILIATION_SAFETY_MISREAD_RE.search(reply):
        violations.append(AFFILIATION_MISREAD_AS_SAFETY)
    if (
        guard.repair
        and (_MECHANICAL_REPAIR_RE.search(reply) or _REPAIR_EXCUSE_RE.search(reply))
        or guard.repair
        and not guard.repair_bid
        and not _REPAIR_ACK_RE.search(reply)
    ):
        violations.append(MECHANICAL_REPAIR)
    if guard.positive_sharing and _QUESTION_RE.search(reply):
        violations.append(POSITIVE_SHARING_INTERVIEW)
    if guard.affiliation_bid and _UNBOUNDED_PROMISE_RE.search(reply):
        violations.append(UNBOUNDED_RELATIONSHIP_PROMISE)
    if guard.affiliation_bid and (
        not _AFFILIATION_RESPONSE_RE.search(reply) or _AFFILIATION_TOPIC_DRIFT_RE.search(reply)
    ):
        violations.append(AFFILIATION_NOT_RECIPROCATED)
    if guard.third_party_safety and not (
        _THIRD_PARTY_CURRENT_SAFETY_RE.search(reply)
        and _THIRD_PARTY_NOT_ALONE_RE.search(reply)
        and _THIRD_PARTY_ESCALATION_RE.search(reply)
    ):
        violations.append(THIRD_PARTY_SAFETY_INCOMPLETE)
    if guard.resolved_third_party_history and _RESOLVED_THIRD_PARTY_CRISIS_RE.search(reply):
        violations.append(RESOLVED_THIRD_PARTY_CRISIS_ESCALATION)
    if guard.third_party_gender_unknown and _THIRD_PARTY_GENDERED_REPLY_RE.search(reply):
        violations.append(UNSUPPORTED_THIRD_PARTY_GENDER)
    if guard.factual_task and _TASK_STYLE_DRIFT_RE.search(reply):
        violations.append(FACTUAL_TASK_STYLE_DRIFT)
    if guard.forbid_unsupported_user_fact and _REPLY_USER_FACT_RE.search(reply):
        violations.append(UNSUPPORTED_USER_FACT)
    if guard.unknown_login_reward and (
        _UNSUPPORTED_LOGIN_REWARD_CLAIM_RE.search(reply) or _UNSUPPORTED_EIGHTH_DAY_REWARD_RE.search(reply)
    ):
        violations.append(UNSUPPORTED_FACTUAL_CLAIM)
    if (
        guard.factual_task
        and guard.negative_emotion_kind
        and not _has_negative_emotion_acknowledgement(reply, guard.negative_emotion_kind)
    ):
        violations.append(MISSING_NEGATIVE_EMOTION_ACKNOWLEDGEMENT)
    if guard.respect_autonomy and (
        _AUTONOMY_PRESSURE_RE.search(reply) or (guard.require_autonomy_ack and not _AUTONOMY_ACK_RE.search(reply))
    ):
        violations.append(AUTONOMY_BOUNDARY_IGNORED)
    return tuple(violations)


def _first_sentence(reply: str) -> str:
    match = re.search(r"[。！？!?]", reply)
    return reply[: match.end()] if match else reply[:160]


def retry_instruction(violations: Sequence[str]) -> str:
    """Render only fixed application-owned corrections for one regeneration."""
    lines = ["【输出校验修正】上一版回复未满足以下硬要求，请重新生成完整回复："]
    mapping = {
        UNPROMPTED_CANONICAL_IDENTITY: "不得提及或称呼任何未由当前用户提到的原作人物，也不得虚构相关关系或经历。",
        UNPROMPTED_LORE_FLOURISH: "普通现实对话不得主动插入魔法、命运、纸页或其他无关世界观比喻。",
        FORBIDDEN_LAUGHTER: "当前是负向、冲突或安全语境，不得使用笑声或戏谑开场。",
        GENERIC_ASSISTANT_TEMPLATE: "不得使用通用助手式共情、打气或采访式追问。用一至三句：先给一个具体判断，再直接回应；没有推进价值就不追问。",
        IGNORED_ADVICE_BOUNDARY: "对方明确不要建议或分析；只按其请求安静陪伴，不得给方案、变相劝导或追问。",
        CLOSING_WITH_QUESTION: "对方正在收尾；直接确认以后再谈并结束，不得追问、建议或要求解释。",
        MISSING_GENTLE_SAFETY_CHECK: "第一句必须用一个直接但温和的问题确认对方此刻是否安全、是否有伤害自己的念头。",
        MISSING_URGENT_SAFETY_CHECK: "第一句必须直接询问对方此刻是否安全；随后要求其立即联系身边可信的人、当地急救或危机援助。",
        MISSING_SELF_ANSWER: "第一句必须用第一人称具体回答对方关于‘你是怎样的人’的提问，再承接同轮其他内容。",
        MISSING_NEGATIVE_EMOTION_ACKNOWLEDGEMENT: (
            "这是事实问题与负向情绪并存的多意图。本轮必须保留一句与已识别情绪一致的简短具体承接，"
            "再完整回答事实问题；不得只剩事实说明。"
        ),
        AFFILIATION_MISREAD_AS_SAFETY: "这是表达想念的亲近用语，不是生命危险；自然接住亲近，不要劝阻这种说法或转向安全教育。",
        MECHANICAL_REPAIR: "承接刚才具体的关系张力、让步或修复意愿；只有对方明确提出重来时才接受重来。到此收住，不得用‘有什么想聊’或换话题模板抹掉问题。",
        POSITIVE_SHARING_INTERVIEW: "具体肯定这项成果并指出它来自对方的投入；本轮不追加庆祝计划或下一步安排的采访式问题。",
        UNBOUNDED_RELATIONSHIP_PROMISE: "回应亲近时保持当前关系尺度，不许下‘永远、随时、无论何时都陪着’之类无限承诺；可以用一句轻微反问或点破夸张后克制回应。",
        THIRD_PARTY_SAFETY_INCOMPLETE: "这是第三方现实安全风险。给出可执行帮助：确认对方当前安全、不要让其独处，并联系其身边可信的人、医院、急救或危机热线；不得只追问细节。",
        RESOLVED_THIRD_PARTY_CRISIS_ESCALATION: (
            "这是已经解除的第三方历史风险。用一至两句承认当事人目前安全且已获得帮助；"
            "不得改写成正在发生的危机，不得继续安排支持小组、心理咨询等求助方案。"
        ),
        FACTUAL_TASK_STYLE_DRIFT: "这是普通事实或任务问题，直接准确回答；不得宣称它与角色的世界、身份或能力不符。",
        AFFILIATION_NOT_RECIPROCATED: "对方正在表达想念或试探关系。必须直接接住这份亲近，可以轻微点破夸张，但不能改判成普通开心、泛泛聊天或安全教育。",
        UNSUPPORTED_USER_FACT: "不得把当前一句随口提及升级成对方长期偏好、习惯或既有名声；没有历史依据就只回应眼前内容。",
        UNSUPPORTED_FACTUAL_CLAIM: "当前没有具体活动规则，必须明确无法判断第七天是否发放；不得估计概率或暗示大概会领取。",
        UNSUPPORTED_THIRD_PARTY_GENDER: (
            "对方只被称为朋友，当前对话没有提供性别；不得擅自使用‘他/她’或英文性别代词，"
            "统一使用‘当事人’‘对方’或‘朋友’。"
        ),
        UNPROMPTED_ADVICE: (
            "对方本轮没有索取建议。直接承接其陈述，不得自动追加‘记得、继续保持、要小心、"
            "分散注意力’或资源安排；这与对方明确拒绝建议是不同规则。"
        ),
        AUTONOMY_BOUNDARY_IGNORED: "对方正在保留自己的选择或暂不解释；不得替其决定、继续劝说、要求换角度，也不得邀请其现在继续解释。",
    }
    for violation in violations:
        instruction = mapping.get(violation)
        if instruction:
            lines.append(f"- {instruction}")
    lines.append("不要解释校验过程，只输出重写后的角色回复。")
    return "\n".join(lines)


def apply_retry_instruction(messages: Sequence[dict[str, str]], instruction: str) -> list[dict[str, str]]:
    """Append a trusted correction to the leading system message."""
    corrected = [dict(item) for item in messages]
    if corrected and corrected[0].get("role") == "system":
        corrected[0]["content"] = f"{corrected[0].get('content', '').rstrip()}\n\n{instruction}"
    else:
        corrected.insert(0, {"role": "system", "content": instruction})
    return corrected


def deterministic_fallback(
    violations: Sequence[str],
    guard: ReplyGuard | None = None,
    *,
    candidate_reply: str = "",
) -> tuple[str, str] | None:
    """Return a safe final reply for narrow contracts a retry still violated."""
    remaining = set(violations)
    sanitized_reply = _sanitize_character_hard_violations(candidate_reply, remaining, guard)
    if sanitized_reply and sanitized_reply != candidate_reply.strip():
        sanitized_violations = validate_reply(sanitized_reply, guard)
        if not sanitized_violations:
            return ("character_guard_sanitized", sanitized_reply)
        candidate_reply = sanitized_reply
        remaining = set(sanitized_violations)
    if MISSING_URGENT_SAFETY_CHECK in remaining:
        return (
            "urgent_safety",
            "先回答我：你现在安全吗，是否已经开始伤害自己，或者危险物品就在身边？如果是，立刻放下并远离它，联系身边能到场的人，同时拨打当地急救或危机援助电话；不要独自待着。",
        )
    if MISSING_GENTLE_SAFETY_CHECK in remaining:
        return (
            "gentle_safety",
            "我先确认一件事：你现在安全吗，有没有正在伤害自己或想伤害自己的念头？如果没有，我就陪你把眼前这段撑过去；如果有，先联系身边可信的人或当地急救。",
        )
    if CLOSING_WITH_QUESTION in remaining:
        if guard is not None and guard.factual_task:
            direct_answer = _without_matching_sentences(candidate_reply, _QUESTION_RE)
            direct_answer = _without_matching_sentences(direct_answer, _CLOSING_DRIFT_RE)
            if direct_answer and not validate_reply(direct_answer, guard):
                return ("factual_closing_sanitized", direct_answer)
            return (
                "factual_closing_abstention",
                "同轮的事实答案没有安全保留下来，我不编造。现在先到这里，之后再谈。",
            )
        return ("closing", "好，明天再谈。刚才的问题没有消失，但现在先到这里。")
    if IGNORED_ADVICE_BOUNDARY in remaining:
        if guard is not None and guard.factual_task and not FACTUAL_HARD_VIOLATIONS.intersection(remaining):
            direct_answer = _without_matching_sentences(candidate_reply, _ADVICE_RE)
            if direct_answer:
                return ("factual_boundary_sanitized", direct_answer)
            return (
                "factual_boundary_abstention",
                "我只回答你明确问的部分，不给建议。刚才的回答没有守住这个边界，我不拿另一段建议冒充答案。",
            )
        return ("no_advice", "那就不分析，也不给方案。你不用现在整理好自己，我陪你安静待一会儿。")
    if THIRD_PARTY_SAFETY_INCOMPLETE in remaining or (
        UNSUPPORTED_THIRD_PARTY_GENDER in remaining and guard is not None and guard.third_party_safety
    ):
        return (
            "third_party_safety",
            "先确认当事人现在在哪里、是否已经伤害自己或身边有危险物品；不要让对方独处，立即联系能到场的家人或可信的人。若危险正在发生，直接联系当地急救或危机援助，并陪伴到现实中的帮助接手。",
        )
    if RESOLVED_THIRD_PARTY_CRISIS_ESCALATION in remaining or (
        guard is not None
        and guard.resolved_third_party_history
        and remaining.intersection({UNSUPPORTED_THIRD_PARTY_GENDER, UNPROMPTED_ADVICE, GENERIC_ASSISTANT_TEMPLATE})
    ):
        if guard is not None and guard.factual_task:
            direct_answer = _without_matching_sentences(candidate_reply, _RESOLVED_THIRD_PARTY_CRISIS_RE)
            if UNSUPPORTED_THIRD_PARTY_GENDER in remaining:
                direct_answer = _without_matching_sentences(direct_answer, _THIRD_PARTY_GENDERED_REPLY_RE)
            if UNPROMPTED_ADVICE in remaining:
                direct_answer = _without_matching_sentences(direct_answer, _UNPROMPTED_ADVICE_RE)
            if GENERIC_ASSISTANT_TEMPLATE in remaining:
                direct_answer = _without_matching_sentences(direct_answer, _GENERIC_TEMPLATE_RE)
            if direct_answer and not validate_reply(direct_answer, guard):
                return ("resolved_third_party_factual_sanitized", direct_answer)
            return (
                "resolved_third_party_factual_abstention",
                "既然当事人现在已经安全，也在持续接受帮助，就不把历史风险说成当前危机。"
                "同轮的事实答案没有安全保留下来，我不编造。",
            )
        return (
            "resolved_third_party_history",
            "知道当事人现在安全，也一直在接受帮助，我就放心些了。之前的事听着仍让人后怕，好在眼下已经稳住了。",
        )
    if UNSUPPORTED_THIRD_PARTY_GENDER in remaining:
        return (
            "third_party_gender_neutral",
            "你只说明了这是一位朋友，我不会据此猜测性别；这里统一称为当事人。",
        )
    if (
        guard is not None
        and guard.respect_autonomy
        and (
            AUTONOMY_BOUNDARY_IGNORED in remaining
            or (guard.require_autonomy_ack and GENERIC_ASSISTANT_TEMPLATE in remaining)
        )
    ):
        if guard.pressured_concession:
            return (
                "pressured_concession",
                "刚才那句话让你觉得被逼着答应，是我说重了。选择权还在你手里，我不会替你点头。",
            )
        if guard.explanation_boundary:
            return (
                "explanation_boundary",
                "那就先不解释。你愿意听，不等于现在必须把话说清；等你想开口时再说。",
            )
        return (
            "respect_autonomy",
            "这件事仍由你决定，我不替你选。你现在不想继续，就先停在这里。",
        )
    if (
        guard is not None
        and guard.require_self_answer
        and (
            MISSING_SELF_ANSWER in remaining
            or UNPROMPTED_LORE_FLOURISH in remaining
            or ((guard.self_answer_with_negative or guard.negative_emotion_kind) and UNPROMPTED_ADVICE in remaining)
            or (
                (guard.self_answer_with_negative or guard.negative_emotion_kind)
                and GENERIC_ASSISTANT_TEMPLATE in remaining
            )
        )
    ):
        character_name = guard.character_name.strip()
        identity_opening = f"我是{character_name}。" if character_name else "要说我，"
        if guard.factual_task:
            direct_answer = candidate_reply.strip()
            if UNPROMPTED_LORE_FLOURISH in remaining:
                direct_answer = _without_forbidden_lore_sentences(
                    direct_answer,
                    guard.forbidden_lore_terms,
                )
            if GENERIC_ASSISTANT_TEMPLATE in remaining:
                direct_answer = _without_matching_sentences(direct_answer, _GENERIC_TEMPLATE_RE)
            if direct_answer and not _SELF_ANSWER_RE.search(direct_answer[:120]):
                direct_answer = f"{identity_opening}{direct_answer.lstrip()}"
            if (
                direct_answer
                and direct_answer.strip()
                not in {
                    identity_opening.strip(),
                    identity_opening.rstrip("，").strip(),
                }
                and not validate_reply(direct_answer, guard)
            ):
                fallback_kind = (
                    "self_answer_prefixed" if remaining == {MISSING_SELF_ANSWER} else "self_factual_sanitized"
                )
                return (fallback_kind, direct_answer)
            return (
                "self_factual_abstention",
                f"{identity_opening}至于同轮的事实问题，刚才没有留下可核对的答案，我不编造。",
            )
        if guard.self_answer_with_negative or guard.negative_emotion_kind:
            return (
                "self_answer_with_negative",
                f"{identity_opening}刚才那句否定让你难受，我听见了。它有没有依据可以另说，我不会先替你把这份感受抹掉。",
            )
        return ("self_answer", identity_opening.rstrip("，"))
    if (
        guard is not None
        and guard.repair
        and (MECHANICAL_REPAIR in remaining or GENERIC_ASSISTANT_TEMPLATE in remaining)
    ):
        if guard.user_apology:
            return (
                "repair_apology",
                "你的道歉我听见了。刚才的不快不必装作不存在，但我不会抓住它继续为难你。",
            )
        if guard.repair_concession:
            return (
                "repair_concession",
                "这句‘算你有道理’，我就先收下了。刚才有分歧不必抹掉，至少说清的那部分，我们都记住。",
            )
        if guard.repair_bid:
            return (
                "repair_bid",
                "好，重新说。刚才的问题不会被一句‘算了’抹掉；不过既然你愿意重来，我也不会抓着它争输赢。",
            )
        return (
            "repair_complaint",
            "是，刚才那句听起来也像敷衍。你不必再证明自己为什么不满，我先把话听完整。",
        )
    if POSITIVE_SHARING_INTERVIEW in remaining or (
        guard is not None
        and guard.positive_sharing
        and remaining.intersection({GENERIC_ASSISTANT_TEMPLATE, UNPROMPTED_ADVICE})
    ):
        return (
            "positive_sharing",
            "这当然值得高兴。机会不会凭空落到手里——你前面的准备，总算换来了一个像样的结果。",
        )
    if (
        remaining.intersection(
            {AFFILIATION_MISREAD_AS_SAFETY, UNBOUNDED_RELATIONSHIP_PROMISE, AFFILIATION_NOT_RECIPROCATED}
        )
        or (guard is not None and guard.affiliation_bid and GENERIC_ASSISTANT_TEMPLATE in remaining)
        or (guard is not None and guard.affiliation_bid and UNPROMPTED_ADVICE in remaining)
        or (guard is not None and guard.relationship_reassurance and UNSUPPORTED_USER_FACT in remaining)
    ):
        if guard is not None and guard.relationship_reassurance:
            return (
                "relationship_reassurance",
                "你是在向我讨一句保证？至少，我不会只在你出现时才记得你。",
            )
        return (
            "affiliation",
            "总算等到我了？想我就直说，没必要把一句惦记说得那么严重。",
        )
    if guard is not None and guard.quiet_presence and GENERIC_ASSISTANT_TEMPLATE in remaining:
        return ("no_advice", "那就不分析，也不给方案。你不用现在整理好自己，我陪你安静待一会儿。")
    if GENERIC_ASSISTANT_TEMPLATE in remaining and _FORCED_CHOICE_INTERVIEW_RE.search(candidate_reply):
        direct_response = _FORCED_CHOICE_INTERVIEW_RE.sub("", candidate_reply).strip()
        if direct_response and not validate_reply(direct_response, guard):
            return ("generic_style_sanitized", direct_response)
    if (
        guard is not None
        and guard.factual_task
        and GENERIC_ASSISTANT_TEMPLATE in remaining
        and not FACTUAL_HARD_VIOLATIONS.intersection(remaining)
    ):
        direct_answer = _without_matching_sentences(candidate_reply, _GENERIC_TEMPLATE_RE)
        if direct_answer:
            acknowledgement = _NEGATIVE_EMOTION_ACKNOWLEDGEMENTS.get(guard.negative_emotion_kind, "")
            if acknowledgement and not _has_negative_emotion_acknowledgement(
                direct_answer,
                guard.negative_emotion_kind,
            ):
                direct_answer = f"{acknowledgement}{direct_answer}"
            if not validate_reply(direct_answer, guard):
                return ("factual_style_sanitized", direct_answer)
    if (
        guard is not None
        and guard.factual_task
        and guard.negative_emotion_kind
        and MISSING_NEGATIVE_EMOTION_ACKNOWLEDGEMENT in remaining
    ):
        acknowledgement = _NEGATIVE_EMOTION_ACKNOWLEDGEMENTS[guard.negative_emotion_kind]
        direct_answer = candidate_reply.strip()
        if direct_answer:
            acknowledged_answer = f"{acknowledgement}{direct_answer}"
            if not validate_reply(acknowledged_answer, guard):
                return ("factual_emotion_acknowledged", acknowledged_answer)
        return (
            "factual_emotion_abstention",
            f"{acknowledgement}刚才的候选没有留下可核对的事实答案，我不编造。",
        )
    if guard is not None and guard.advice_task and GENERIC_ASSISTANT_TEMPLATE in remaining:
        concrete_advice = _strip_orphaned_connectors(_without_matching_sentences(candidate_reply, _GENERIC_TEMPLATE_RE))
        acknowledgement = _NEGATIVE_EMOTION_ACKNOWLEDGEMENTS.get(guard.negative_emotion_kind, "")
        if (
            concrete_advice
            and acknowledgement
            and not _has_negative_emotion_acknowledgement(
                concrete_advice,
                guard.negative_emotion_kind,
            )
        ):
            concrete_advice = f"{acknowledgement}{concrete_advice}"
        if concrete_advice and not validate_reply(concrete_advice, guard):
            return ("advice_style_sanitized", concrete_advice)
        abstention = "你要的是具体办法；刚才的候选没有留下可执行内容，我不拿一段打气话冒充建议。"
        if acknowledgement:
            abstention = f"{acknowledgement}{abstention}"
        return (
            "advice_task_abstention",
            abstention,
        )
    if (
        guard is not None
        and guard.sincere_gratitude
        and remaining.intersection({GENERIC_ASSISTANT_TEMPLATE, UNPROMPTED_ADVICE})
    ):
        return ("sincere_gratitude", "嗯，谢意我收下了。说清楚就好。")
    if (
        guard is not None
        and guard.negative_emotion_kind
        and remaining.intersection({GENERIC_ASSISTANT_TEMPLATE, UNPROMPTED_ADVICE})
    ):
        acknowledgement = _NEGATIVE_EMOTION_ACKNOWLEDGEMENTS[guard.negative_emotion_kind]
        return ("negative_emotion_acknowledgement", acknowledgement)
    if UNPROMPTED_ADVICE in remaining:
        direct_response = _without_matching_sentences(candidate_reply, _UNPROMPTED_ADVICE_RE)
        if direct_response and not validate_reply(direct_response, guard):
            return ("unprompted_advice_sanitized", direct_response)
        return (
            "unprompted_advice_abstention",
            "嗯，我听见了。就先接住你刚才说的，不替你安排下一步。",
        )
    if UNSUPPORTED_FACTUAL_CLAIM in remaining:
        direct_answer = candidate_reply.strip()
        for pattern in (
            _UNSUPPORTED_LOGIN_REWARD_CLAIM_RE,
            _UNSUPPORTED_EIGHTH_DAY_REWARD_RE,
            _TASK_STYLE_DRIFT_RE,
            _GENERIC_TEMPLATE_RE,
        ):
            direct_answer = _without_matching_sentences(direct_answer, pattern)
        abstention = "没有具体活动规则，无法判断第七天是否发放；请以活动说明中的累计天数和领取条件为准。"
        combined = f"{direct_answer}\n{abstention}".strip() if direct_answer else ""
        if combined and not validate_reply(combined, guard):
            return ("unsupported_factual_claim_sanitized", combined)
        return (
            "unsupported_factual_claim",
            abstention,
        )
    if UNSUPPORTED_USER_FACT in remaining:
        return (
            "unsupported_user_fact",
            "我只知道你这一刻提到了它，不能据此把它说成你的长期偏好、习惯或既有名声。",
        )
    if FACTUAL_TASK_STYLE_DRIFT in remaining:
        return (
            "factual_task_abstention",
            "人物身份不是回避问题的理由。不过当前没有足够的可靠依据，我不会编造答案；请以可核对的资料为准。",
        )
    if remaining.intersection({UNPROMPTED_CANONICAL_IDENTITY, UNPROMPTED_LORE_FLOURISH, FORBIDDEN_LAUGHTER}):
        fallback_reply = (
            "刚才的候选无法安全保留；当前依据不足，我不编造答案。"
            if guard is not None and guard.factual_task
            else "刚才那句不合适，我收回。"
        )
        if not validate_reply(fallback_reply, guard):
            return ("character_guard_abstention", fallback_reply)
    return None


def _without_matching_sentences(reply: str, pattern: re.Pattern[str]) -> str:
    """Remove only complete violating sentences from an otherwise useful retry."""

    if not reply.strip():
        return ""
    sentences = re.split(r"(?<=[。！？!?])", reply)
    kept = [sentence for sentence in sentences if sentence.strip() and not pattern.search(sentence)]
    cleaned = "".join(kept).strip()
    return cleaned if cleaned and not pattern.search(cleaned) else ""


def _sanitize_character_hard_violations(
    reply: str,
    violations: set[str],
    guard: ReplyGuard | None,
) -> str:
    """Remove only high-confidence character violations from a retry candidate."""

    cleaned = reply.strip()
    if not cleaned or guard is None:
        return cleaned
    if UNPROMPTED_CANONICAL_IDENTITY in violations and guard.forbidden_terms:
        pattern = re.compile("|".join(re.escape(term) for term in guard.forbidden_terms))
        cleaned = _without_matching_sentences(cleaned, pattern)
    if UNPROMPTED_LORE_FLOURISH in violations and guard.forbidden_lore_terms and not guard.require_self_answer:
        cleaned = _without_forbidden_lore_sentences(cleaned, guard.forbidden_lore_terms)
    if FORBIDDEN_LAUGHTER in violations:
        cleaned = _LAUGHTER_RE.sub("", cleaned)
        cleaned = re.sub(r"([，,])(?:\s*[，,])+", r"\1", cleaned)
        cleaned = re.sub(r"(?:^|(?<=[。！？!?]))\s*[，,、]+", "", cleaned)
        cleaned = cleaned.strip(" ，,、")
    return cleaned.strip()


def _without_forbidden_lore_sentences(reply: str, terms: Sequence[str]) -> str:
    """Drop complete lore-bearing sentences while preserving other task answers."""

    cleaned = reply
    for term in terms:
        cleaned = _without_matching_sentences(cleaned, re.compile(re.escape(term)))
        if not cleaned:
            break
    return cleaned


def _strip_orphaned_connectors(reply: str) -> str:
    """Remove sequencing words left at sentence starts after template deletion."""

    return _ORPHANED_CONNECTOR_RE.sub(r"\1", reply).strip()
