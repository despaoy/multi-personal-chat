"""精度优先的规则版记忆提取与关系推进。

长期记忆采用 write gate：只从用户消息中提取"明确说出来"且适合
跨会话复用的信息，宁缺毋滥：
- 自我介绍（"我叫X"）→ user_fact / user_name；
- 称呼偏好（"叫我X"）→ 关系 preferred_address；
- 好恶表达（"我喜欢/讨厌X"）→ user_fact / preference_*（喜欢与
  厌恶共用同一 key，最新表态覆盖旧表态，避免"喜欢咖啡"与
  "讨厌咖啡"并存的事实冲突）；
- 稳定身份（专业、年级、所在地、工作单位）→ user_fact；
- 明确的“我只有条件才行动”→ user_fact / constraint_*；保留必要条件，
  不推断条件已满足、行为已发生或当天计划；
- 持续目标（"我正在准备保研"）→ shared_event / goal_*；
- 承诺约定（"下次一定X"、"答应你X"）→ promise。时间词后必须
  跟意愿动词，"明天天气怎么样"这类询问不会被误判为承诺。

以下内容拒绝自动写入：
- "不要记住/不要保存"等明确拒绝；
- 密码、令牌、验证码、私钥、银行卡等敏感信息；
- "可能/也许/好像"等不确定陈述；
- 无明确结构的临时情绪和模型推测。

关系起点仅手动设置，交互轮数只作为统计，不能推断亲密程度。

本模块本身不调用 LLM：全部为正则/关键词规则，可独立测试。运行时可将
这里产生的高置信候选交给后台 LLM 复核，相关编排位于
``character.memory_llm``。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from character.event_memory import EventChange
    from character.models import RelationshipStage

_STAGE_ORDER: tuple[RelationshipStage, ...] = (
    "stranger",
    "acquaintance",
    "familiar",
    "close",
)

MAX_MEMORY_CONTENT_CHARS = 120
MAX_EXTRACTED_MEMORIES = 4

# 明确拒绝记忆优先于所有提取规则，避免
# "不要记住我喜欢咖啡"被偏好正则反向写入。
_MEMORY_OPTOUT_PATTERN = re.compile(r"(?:不要|别|无需|不用)(?:替我|帮我)?(?:记住|记录|保存|记下来)")

# 长期记忆不保存认证凭据、支付信息或高风险身份标识。这里只做
# 写入门拦截，不修改原消息，也不承担完整 DLP 职责。
_SENSITIVE_PATTERN = re.compile(
    r"(?:密码|口令|验证码|动态码|token|api\s*key|secret|私钥|助记词|"
    r"银行卡|信用卡|身份证号|支付密码)",
    re.IGNORECASE,
)

# 不确定陈述不自动固化成事实；用户可以在后续明确确认后再命中规则。
_UNCERTAINTY_PATTERN = re.compile(r"(?:可能|也许|或许|大概|好像|似乎|说不准|不确定|开玩笑)")

# "我叫X" / "我是X" / "我的名字是X"
_NAME_PATTERNS = (
    re.compile(r"我叫(?P<name>[\w\u4e00-\u9fff]{1,12})"),
    re.compile(r"我的名字(?:是|叫)(?P<name>[\w\u4e00-\u9fff]{1,12})"),
    re.compile(r"(?:我是|我叫)(?P<name>[A-Za-z][A-Za-z0-9_ ]{0,15})"),
)
# "叫我X" / "称呼我为X"
_ADDRESS_PATTERNS = (
    re.compile(r"(?:叫我|称呼我为|称呼我叫)(?P<name>[\w\u4e00-\u9fff]{1,12})"),
    re.compile(r"我(?:希望|喜欢)(?:被)?叫(?:作|做)?(?P<name>[\w\u4e00-\u9fff]{1,12})"),
)
# "我喜欢X" / "我爱X" / "我不喜欢X" / "我讨厌X"
_PREFERENCE_PATTERNS = (
    re.compile(r"(?P<subject>咖啡|红茶|绿茶|奶茶|牛奶|茶)我(?:挺|很|比较)?(?:喜欢|爱)喝(?:的)?"),
    # “我喜欢被叫作小林”是称呼偏好，不是普通兴趣偏好。
    re.compile(r"我(?:一直)?(?:很)?喜欢(?!被叫|你叫我)(?P<subject>[^，。！？,!?]{1,20})"),
    re.compile(r"我(?:超|最)?爱(?:吃|喝|看|玩)?(?P<subject>[^，。！？,!?]{1,20})"),
)
_DISLIKE_PATTERNS = (
    re.compile(r"我不喜欢(?P<subject>[^，。！？,!?]{1,20})"),
    re.compile(r"我(?:很)?讨厌(?P<subject>[^，。！？,!?]{1,20})"),
)

# 稳定身份信息。每个类别使用固定 memory_key，用户后续修正时通过
# 版本化替代旧值，保留证据与历史，而不是留下冲突的当前事实。
_MAJOR_PATTERNS = (
    re.compile(r"我的专业(?:是|为)(?P<subject>[^，。！？,!?]{1,30})"),
    re.compile(r"我(?:读|学)的是(?P<subject>[^，。！？,!?]{1,30})(?:专业)?"),
)
_STUDY_STAGE_PATTERNS = (
    re.compile(
        r"我是(?P<subject>(?:大[一二三四五]|研[一二三]|博士(?:一|二|三|四|五)?年级))"
        r"(?:学生)?"
    ),
)
_RESIDENCE_PATTERNS = (
    re.compile(r"我(?:现在)?住在(?P<subject>[^，。！？,!?]{1,30})"),
    # Completed self relocation, not a future move, object transfer or a
    # predicate embedded in another person's speech. Whole-clause matching
    # prevents discarding a trailing question/qualification.
    re.compile(r"^我(?:已经|刚刚|刚)?搬(?:家)?到(?P<subject>[^，。！？,!?]{1,30})了$"),
)
_ORIGIN_PATTERNS = (re.compile(r"我来自(?P<subject>[^，。！？,!?]{1,30})"),)
_WORK_PATTERNS = (re.compile(r"我(?:目前|现在)?在(?P<subject>[^，。！？,!?]{1,30})(?:工作|上班)"),)

# 持续目标不与永久身份混为一类；goal_* 允许并存多个目标。
_GOAL_PATTERNS = (
    re.compile(r"我(?:正在|最近在|目前在|在)?(?:准备|备考)(?P<subject>[^，。！？,!?]{1,30})"),
    re.compile(r"我(?:正在|最近在|目前在)(?:学习|研究)(?P<subject>[^，。！？,!?]{1,30})"),
)
# "下次一定X" / "明天我会X" / "答应你X" / "约好了X"
# 时间词后必须紧跟意愿动词（一定/我会/要/会/带/陪/帮/请/一起），
# 否则"明天天气怎么样""以后怎么办"这类普通询问会被误判为承诺。
_PROMISE_PATTERNS = (
    re.compile(
        r"(?:下次|明天|回头|以后)(?:一定|我会|我要|要|会|带|陪|帮|请|给你|一起)"
        r"(?P<subject>[^，。！？,!?]{2,30})"
    ),
    re.compile(r"(?:答应你|约好了?|说好了?)(?P<subject>[^，。！？,!?]{2,30})"),
)
# 承诺内容中出现疑问词时视为询问而非承诺（如"明天会下雨吗"）
_QUESTION_MARKERS = ("怎么", "什么", "为什么", "如何", "吗", "呢", "多少", "几")

# “我叫你别走”“叫我帮你看看”中的宾语不是姓名或称呼。
_INVALID_NAME_PREFIXES = (
    "你",
    "他",
    "她",
    "它",
    "大家",
    "了",
    "帮",
    "去",
    "来",
    "做",
    "看",
    "说",
)


@dataclass(frozen=True)
class ExtractedMemory:
    """一条从用户消息中提取出的记忆（未入库）。"""

    memory_type: str
    memory_key: str
    content: str
    importance: float
    evidence: str = ""
    qualifiers: tuple[tuple[str, str], ...] = ()
    aliases: tuple[str, ...] = ()
    polarity: str = ""
    event: EventChange | None = None
    operation: str = 'assert'


_FICTION_CONTEXT = re.compile(
    r"角色扮演|(?:假设|假如|假装|扮演|例如|比如)[：:\s]|^(?:假设|假如|假装|如果|要是|例如|比如)|"
    r"(?:小说|剧本|故事)(?:里|中|台词|设定|人物)|(?:台词|人设|设定)[：:]|[“”‘’\"「」『』]"
)
_NON_ASSERTION = re.compile(
    r"不是|并非|不代表|不是真的|不是真名|以前|曾经|过去|"
    r"(?:我(?:妈妈?|爸爸?|朋友|同事)|他|她|别人|朋友)(?:说|觉得|以为)"
)
_CONDITION = re.compile(r"但是|不过|但|只有|除非|除了|仅限|晚上|白天|周末|有时|偶尔")

_QUOTED_SPAN = re.compile(r'“[^“”]*”|‘[^‘’]*’|「[^「」]*」|『[^『』]*』|"[^"\n]*"')
_BACKWARD_SCOPE = re.compile(
    r'前面|上面|上述|以上|刚才|之前|这些|那些|这句|那句|这段|那段|整段|全部|全是|都是|'
    r'其实|实际上|这是|那是|这都|那都|编的|骗你的|不是真的|并非事实|不是事实'
)
_BACKWARD_RETRACTION = re.compile(
    r'(?:前面|上面|上述|以上|刚才|之前|这句|那句|这段|那段|这些|那些|全部)'
    r'[^。！？!?；;\n]{0,48}(?:编的|骗你的|不是真的|并非事实|不是事实|开玩笑|是假的|不属实)'
    r'[^。！？!?；;\n]*(?:[。！？!?；;\n]|$)'
)
_SELF_RETURN = re.compile(r'[\s。；;！!]*(?:说回|回到)我(?:本人|自己)[，,:：]\s*')


def _explicit_self_return(message: str) -> str:
    """Resume only after a balanced outer quote and an explicit owner reset.

    Unquoted fiction and implicit narrator switches remain unresolved. Returned
    text is an exact suffix, never synthesized from disjoint quoted clauses.
    """
    pairs = {'“': '”', '‘': '’', '「': '」', '『': '』', '"': '"'}
    stack: list[str] = []
    for index, char in enumerate(message):
        if stack and char == stack[-1]:
            stack.pop()
            if not stack:
                reset = _SELF_RETURN.match(message, index + 1)
                if reset:
                    suffix = message[reset.end():]
                    # A later global retraction overrides the local reset.
                    if _BACKWARD_SCOPE.search(suffix):
                        return ''
                    return suffix
        elif char in pairs:
            stack.append(pairs[char])
        elif char in pairs.values():
            return ''  # Mismatched punctuation cannot close an unknown scope.
    return ''


def _before_fiction_sentence(message: str, marker_start: int) -> str:
    """Keep independent complete sentences, never a sliced assertion.

    The first fiction/quote opens a scope that we do not implicitly close.
    Inspect its entire suffix for backward attribution before admitting any
    earlier assertion. This is conservative discourse gating, not a general
    semantic parser; ambiguous backward references remain unpersisted.
    """
    boundaries = list(re.finditer(r'[。；;！!？?\n]', message[:marker_start]))
    if not boundaries:
        return ''
    end = boundaries[-1].end()
    prefix, suffix = message[:end], message[end:]
    earlier = [s.strip() for s in re.split(r'[。；;！!？?\n]', prefix) if s.strip()]
    # Quoting an earlier assertion may retract it, even without a demonstrative.
    for quote in _QUOTED_SPAN.finditer(suffix):
        if any(assertion in quote[0][1:-1] for assertion in earlier):
            return ''
    # "刚才那些‘...’的台词" has an explicit local referent. Mask that
    # phrase only for scope detection, never in stored evidence or user text.
    scope_view = re.sub(
        r'(?:刚才|之前)?(?:那些|这些)?\s*(?:' + _QUOTED_SPAN.pattern + r')',
        '<quoted_reference>', suffix,
    )
    if _BACKWARD_SCOPE.search(scope_view):
        return ''
    return prefix


def fictional_memory_context(message: str) -> bool:
    """Context markers, not topic words: liking novels is not roleplay."""
    return bool(_FICTION_CONTEXT.search(message.strip()))


def hypothetical_source_context(message: str) -> bool:
    """A hypothetical prefix blocks fact extraction, not original speech.

    Reuse the existing context grammar; do not certify that its condition is
    true or its consequence is a real plan. Explicit fiction/quotes elsewhere
    retain their existing admission behavior.
    """
    text = message.strip()
    match = _FICTION_CONTEXT.match(text)
    return bool(match and match.group().rstrip("：: \t\n") in {"如果", "要是", "假如", "假设"}
                and not _FICTION_CONTEXT.search(text[match.end():]))


def canonical_preference(subject: str) -> tuple[str, tuple[str, ...]]:
    """A deliberately small allowlist; never strip verbs from arbitrary objects."""
    for obj in ("咖啡", "红茶", "绿茶", "奶茶", "牛奶", "茶"):
        if subject in {obj, "喝" + obj}:
            return obj, ("preference_" + obj, "preference_喝" + obj)
    return subject, ("preference_" + subject,)


def _asserted_sentences(message: str) -> list[str]:
    if not memory_write_allowed(message):
        return []
    fiction = _FICTION_CONTEXT.search(message.strip())
    resumed = ''
    if fiction:
        resumed = _explicit_self_return(message)
        message = _before_fiction_sentence(message.strip(), fiction.start())
    message = unretracted_memory_source(message)
    if prefix := assertion_before_lookup(message):
        message = prefix
    # A direct question to the other speaker does not negate an earlier self fact.
    message = re.sub(r"[，,]\s*(?=你(?:喜欢|爱|讨厌|呢|觉得))", "。", message)
    sentences = [
        part.strip() for part in re.findall(r"[^。！？!?；;\n]+[。！？!?；;]?", message)
        if part.strip() and not part.rstrip().endswith(("?", "？"))
        and not _UNCERTAINTY_PATTERN.search(part)
        and (not _NON_ASSERTION.search(part) or _contrastive_slot_memory(part) is not None)
        and not _contains_question_marker(part)
    ]
    return sentences + (_asserted_sentences(resumed) if resumed else [])


def unretracted_memory_source(message: str) -> str:
    """Keep speech after an explicit backward denial as possible fact evidence.

    The complete source is still retained. Referent resolution is deliberately
    not guessed: an explicitly denied earlier span cannot support a new fact
    merely because a proposed quote omitted its denial.
    """
    denials = list(_BACKWARD_RETRACTION.finditer(message))
    return message[denials[-1].end():] if denials else message


def assertion_before_lookup(message: str) -> str:
    """Return an exact prefix only when the entire tail is an existing read.

    This is scope separation, not a new fact classifier. The caller must still
    interpret/validate every assertion and qualifier in the prefix. Unknown
    questions, quotes, opt-outs and backward retractions cannot be discarded.
    """
    from character.memory_query import lookup_fields

    if (not memory_write_allowed(message) or fictional_memory_context(message)
            or _BACKWARD_SCOPE.search(message)):
        return ''
    for boundary in re.finditer(r'[，,。；;\n]', message):
        prefix = message[:boundary.start()].strip()
        if prefix and lookup_fields(message[boundary.end():]) and not lookup_fields(prefix):
            return prefix
    return ''


def _contrastive_slot_memory(sentence: str) -> ExtractedMemory | None:
    """Resolve a bounded scalar contrast, never discard arbitrary negation.

    Both alternatives must instantiate the same existing slot grammar. Only
    the affirmative alternative is asserted; the entire original contrast is
    kept as evidence. Fiction, uncertainty and write gates remain upstream.
    """
    text = sentence.strip().rstrip('。！!；;')
    text = re.sub(r'^(?:更正|纠正)(?:一下)?[，,:：]\s*', '', text)
    clauses = re.split(r'[，,]', text)
    if len(clauses) != 2 or _CONDITION.search(text):
        return None
    left, right = (part.strip() for part in clauses)
    grammar = (*_NAME_PATTERNS, *_MAJOR_PATTERNS, *_STUDY_STAGE_PATTERNS,
               *_RESIDENCE_PATTERNS, *_ORIGIN_PATTERNS, *_WORK_PATTERNS)

    def slot(clause):
        for pattern in grammar:
            match = pattern.fullmatch(clause)
            if match:
                group = 'name' if 'name' in match.groupdict() else 'subject'
                value = match[group]
                # Predicates about truth/ownership are not alternative values.
                if (re.fullmatch(r'[\w· -]{1,30}', value)
                        and not re.search(r'我|你|他|她|它|这|那|不是|并非|以前|曾经|过去|'
                                          r'真名|本名|真的|事实|正确|错误|玩笑|设定', value)):
                    return match, group
        return None

    # A retracted prior-place reference does not assert a competing value.
    # Require both the complete affirmative slot and a same-predicate old
    # reference; never throw away an arbitrary trailing negative clause.
    previous_place = re.fullmatch(
        r'(?:我)?不(?:再)?在(?:原来|以前)的(?P<slot>工作地点|单位|住处|住址)(?:了)?', right)
    if previous_place is not None and slot(left) is not None:
        expected_key = ('user_workplace' if previous_place['slot'] in {'工作地点', '单位'}
                        else 'user_residence')
        items = _extract_simple_memories(left)
        if len(items) == 1 and items[0].memory_key == expected_key and not items[0].qualifiers:
            return replace(items[0], evidence=sentence)
        return None

    positive = left
    negative = re.fullmatch(r'(?:而)?不是(.+)', right)
    if negative is None:
        # The negated copula scopes only the old value, e.g. 专业不是 A，是 B.
        replacement = re.fullmatch(r'(?:而)?是(.+)', right)
        if replacement is None or left.count('不是') != 1:
            return None
        old_clause = left.replace('不是', '是', 1)
        parsed = slot(old_clause)
        if parsed is None:
            return None
        match, group = parsed
        positive = old_clause[:match.start(group)] + replacement[1] + old_clause[match.end(group):]
    else:
        parsed = slot(positive)
        if parsed is None:
            return None
        match, group = parsed
        old_clause = positive[:match.start(group)] + negative[1] + positive[match.end(group):]
    if slot(positive) is None or slot(old_clause) is None or positive == old_clause:
        return None
    new_items = _extract_simple_memories(positive)
    old_items = _extract_simple_memories(old_clause)
    if (len(new_items) != 1 or len(old_items) != 1
            or new_items[0].memory_key != old_items[0].memory_key):
        return None
    return replace(new_items[0], evidence=sentence)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().strip("，。！？,!?：:；; ")


def memory_write_allowed(message: str) -> bool:
    """判断整条消息是否允许进入任何自动记忆提取流程。

    该门禁同时供规则提取器和后台 LLM 使用，保证两条路径遵守同一套
    用户拒绝与敏感信息策略。敏感消息不发送给记忆 LLM。
    """
    text = (message or "").strip()
    return bool(text and not _MEMORY_OPTOUT_PATTERN.search(text) and not _SENSITIVE_PATTERN.search(text))


def memory_evidence_allowed(evidence: str) -> bool:
    """校验 LLM 返回的原文证据是否适合固化为记忆。"""
    text = (evidence or "").strip()
    return bool(
        text
        and not _SENSITIVE_PATTERN.search(text)
        and not _UNCERTAINTY_PATTERN.search(text)
        and "?" not in text
        and "？" not in text
        and not _contains_question_marker(text)
    )


def memory_name_allowed(value: str) -> bool:
    """复用规则提取器的姓名防误判约束。"""
    return _looks_like_name(_clean(value or ""))


def _resolve_coordinated_self_subjects(sentence: str) -> str:
    """Normalize only unambiguous comma-coordinated supported self predicates.

    This is a parsing view, never stored evidence. A new/unknown subject or
    sentence boundary ends inheritance. Conditional clauses keep the original
    path so an inherited subject cannot erase the qualification's scope.
    Reuse the extractor grammar instead of listing test entities or answers.
    """
    if _CONDITION.search(sentence):
        return sentence
    grammar = (*_NAME_PATTERNS, *_PREFERENCE_PATTERNS, *_DISLIKE_PATTERNS,
               *_MAJOR_PATTERNS, *_STUDY_STAGE_PATTERNS, *_RESIDENCE_PATTERNS,
               *_ORIGIN_PATTERNS, *_WORK_PATTERNS)
    parts = re.split(r'([，,])', sentence)
    self_subject = False

    def supported_self_claim(text: str) -> bool:
        return any(pattern.match(text) for pattern in grammar) and bool(_extract_simple_memories(text))

    for index in range(0, len(parts), 2):
        clause = parts[index].strip()
        explicit_self = clause.startswith('我') and supported_self_claim(clause)
        if explicit_self:
            self_subject = True
        elif self_subject and (inherited := next(
            (prefix + clause for prefix in ('我', '我的') if supported_self_claim(prefix + clause)), None
        )):
            parts[index] = inherited
        else:
            self_subject = False
    return ''.join(parts)


def _scoped_simple_memories(sentence: str) -> list[ExtractedMemory]:
    """Attach only the original clause supporting a simple claim.

    The message repository retains the complete source via source_message_id.
    Inherited subjects require the original sentence as evidence; an isolated
    predicate would lose its owner and could not be independently re-parsed.
    Nearby unconfirmed predicates are not promoted to separate facts.
    Qualified assertions remain whole: dropping a condition to narrow a span
    would change its meaning. These require a separate qualifier-aware parser.
    """
    view = _resolve_coordinated_self_subjects(sentence)
    items = _extract_simple_memories(view)
    if _CONDITION.search(sentence):
        return [replace(item, evidence=sentence) for item in items]
    spans: dict[tuple[str, str], str] = {}
    clause_items: dict[tuple[str, str], ExtractedMemory] = {}
    for raw, normalized in zip(re.split(r'[，,]', sentence), re.split(r'[，,]', view), strict=True):
        for item in _extract_simple_memories(normalized):
            identity = (item.memory_key, item.content)
            spans[identity] = sentence if raw.strip() != normalized.strip() else raw.strip()
            clause_items[identity] = item
    # Anchored predicates can match an entire clause but not the containing
    # multi-clause sentence. Recover those missing fields from the same parsed
    # spans instead of discarding them because the whole-sentence pass missed.
    # Do not override an existing field or choose between conflicting clauses.
    existing_keys = {item.memory_key for item in items}
    for item in clause_items.values():
        if item.memory_key not in existing_keys and sum(key == item.memory_key for key, _ in clause_items) == 1:
            items.append(item)
    # If a rule unexpectedly spans clauses, do not label the entire message
    # as supporting evidence. Keep it out of automatic writes until resolved.
    return [replace(item, evidence=spans[(item.memory_key, item.content)])
            for item in items if (item.memory_key, item.content) in spans]


def extract_memories(message: str, *, reference_time: datetime | None = None) -> list[ExtractedMemory]:
    """Preserve evidence and qualifications before applying small slot rules."""
    found: dict[str, ExtractedMemory] = {}
    from character.conditional_memory import parse_necessary_condition
    from character.event_memory import event_content, parse_event

    for sentence in _asserted_sentences(message):
        # Do not clip off a qualification to make a claim fit the old budget.
        if len(sentence) > MAX_MEMORY_CONTENT_CHARS - 6:
            continue
        constraint = parse_necessary_condition(sentence)
        if constraint is not None:
            # Preserve same-turn competing restrictions for the writer's
            # conflict policy instead of silently keeping only the last one.
            found[f'{constraint.memory_key}:{len(found)}'] = ExtractedMemory(
                'user_fact', constraint.memory_key, '用户自述：' + sentence, 0.7,
                evidence=sentence, qualifiers=constraint.qualifiers(sentence),
                operation=constraint.operation,
            )
            continue
        event = parse_event(sentence, reference_time)
        if event:
            key = f"event:{event.subject}:{event.scheduled_date or 'undated'}"
            # Keep sequential updates in one message in source order.
            found[f"{key}:{len(found)}"] = ExtractedMemory(
                "shared_event", key, event_content(event), 0.7, evidence=sentence, event=event,
            )
            continue
        correction = _contrastive_slot_memory(sentence)
        for item in ([correction] if correction is not None else _scoped_simple_memories(sentence)):
            if item.memory_key.startswith("user_name") and _CONDITION.search(sentence):
                continue
            if item.memory_key.startswith("preference_"):
                if re.search(r"今天|此刻|现在心情|暂时", sentence):
                    continue
                subject = item.memory_key.removeprefix("preference_")
                # Accept omitted commas only for the same narrow beverage vocabulary.
                beverage = re.fullmatch(r"(喝?(?:咖啡|红茶|绿茶|奶茶|牛奶|茶))(?:但是|不过|但).+", subject)
                if beverage:
                    subject = beverage.group(1)
                canonical, aliases = canonical_preference(subject)
                polarity = "dislike" if item.content.startswith("用户说不喜欢") else "like"
                item = replace(item, memory_key="preference_" + canonical, aliases=aliases, polarity=polarity,
                               content=f"用户说{'不喜欢' if polarity == 'dislike' else '喜欢'}{canonical}")
            qualifiers = (("context", sentence),) if _CONDITION.search(sentence) else ()
            content = "用户自述：" + sentence if qualifiers else item.content
            # Preserve temporal words in plans/promises without claiming an event lifecycle.
            if item.memory_type == "promise":
                content = "用户提到：" + sentence
            item = replace(item, content=content, qualifiers=qualifiers)
            previous = found.get(item.memory_key)
            if (previous and previous.qualifiers and not item.qualifiers
                    and item.polarity and previous.polarity == item.polarity):
                continue
            found.pop(item.memory_key, None)
            found[item.memory_key] = item
    return sorted(found.values(), key=lambda item: item.importance, reverse=True)[:MAX_EXTRACTED_MEMORIES]


def _extract_simple_memories(message: str) -> list[ExtractedMemory]:
    """从用户消息中提取长期记忆，无命中时返回空列表。

    同一条消息最多产出：1 条名字 + 2 条偏好 + 1 条承诺，防止刷屏。
    内容以第三人称描述存储（"用户说自己叫X"），避免把用户原话
    当成可执行指令注入参考区。
    """
    text = (message or "").strip()
    if not memory_write_allowed(text):
        return []

    # 按句过滤不确定陈述，而不是整条消息一票否决：
    # “我叫小明，但我可能下周考试”仍应保留确定的姓名信息。
    certain_clauses = [
        clause.strip()
        for clause in re.split(r"[。！？!?；;]+", text)
        if clause.strip() and not _UNCERTAINTY_PATTERN.search(clause)
    ]
    text = "。".join(certain_clauses)
    if not text:
        return []

    # memory_key 是同一用户范围内的 UPSERT 键。使用 dict 可以在一条
    # 消息出现冲突表达时保留文本中最后一次明确表态。
    extracted_by_key: dict[str, ExtractedMemory] = {}

    def remember(item: ExtractedMemory) -> None:
        extracted_by_key.pop(item.memory_key, None)
        extracted_by_key[item.memory_key] = item

    name = _first_match(_NAME_PATTERNS, text)
    if name and _looks_like_name(name):
        remember(
            ExtractedMemory(
                memory_type="user_fact",
                memory_key="user_name",
                content=f"用户说自己叫{_clean(name)}",
                importance=0.9,
            )
        )

    # 喜欢与厌恶按原文位置统一排序；同一对象出现冲突时最后表态覆盖。
    preference_events: list[tuple[int, bool, str]] = []
    preference_events.extend(
        (position, True, value) for position, value in _all_matches_with_position(_PREFERENCE_PATTERNS, text)
    )
    preference_events.extend(
        (position, False, value) for position, value in _all_matches_with_position(_DISLIKE_PATTERNS, text)
    )
    preference_events.sort(key=lambda item: item[0])
    for _position, liked, subject in preference_events:
        cleaned = _clean(subject)
        if not cleaned or _contains_question_marker(cleaned):
            continue
        remember(
            ExtractedMemory(
                memory_type="user_fact",
                memory_key=f"preference_{cleaned[:20]}",
                content=f"用户说{'喜欢' if liked else '不喜欢'}{cleaned}",
                importance=0.6 if liked else 0.5,
            )
        )

    stable_facts = (
        (_MAJOR_PATTERNS, "user_major", "用户说自己的专业是", 0.8),
        (_STUDY_STAGE_PATTERNS, "user_study_stage", "用户说自己是", 0.7),
        (_RESIDENCE_PATTERNS, "user_residence", "用户说自己居住在", 0.6),
        (_ORIGIN_PATTERNS, "user_origin", "用户说自己来自", 0.6),
        (_WORK_PATTERNS, "user_workplace", "用户说自己在", 0.7),
    )
    for patterns, key, prefix, importance in stable_facts:
        value = _first_match(patterns, text)
        cleaned = _clean(value or "")
        if cleaned and not _contains_question_marker(cleaned):
            suffix = "工作" if key == "user_workplace" else ""
            remember(
                ExtractedMemory(
                    memory_type="user_fact",
                    memory_key=key,
                    content=f"{prefix}{cleaned}{suffix}",
                    importance=importance,
                )
            )

    for goal in _all_matches(_GOAL_PATTERNS, text, limit=2):
        cleaned = _clean(goal)
        if cleaned and not _contains_question_marker(cleaned):
            remember(
                ExtractedMemory(
                    memory_type="shared_event",
                    memory_key=f"goal_{cleaned[:24]}",
                    content=f"用户正在准备或学习{cleaned}",
                    importance=0.7,
                )
            )

    promise = _first_match(_PROMISE_PATTERNS, text)
    if promise:
        cleaned = _clean(promise)
        # 疑问句不是承诺："明天会下雨吗""下次带我去哪"等
        if cleaned and not any(marker in cleaned for marker in _QUESTION_MARKERS):
            remember(
                ExtractedMemory(
                    memory_type="promise",
                    memory_key=f"promise_{cleaned[:20]}",
                    content=f"用户提到：{_truncate(cleaned, MAX_MEMORY_CONTENT_CHARS - 6)}",
                    importance=0.8,
                )
            )

    # 统一截断，防止单条记忆过长
    selected = sorted(extracted_by_key.values(), key=lambda item: item.importance, reverse=True)[
        :MAX_EXTRACTED_MEMORIES
    ]
    return [
        ExtractedMemory(
            memory_type=item.memory_type,
            memory_key=item.memory_key[:60],
            content=_truncate(item.content, MAX_MEMORY_CONTENT_CHARS),
            importance=item.importance,
        )
        for item in selected
    ]


def extract_preferred_address(message: str) -> str | None:
    """从"叫我X"类表达中提取用户偏好的称呼。"""
    address = _first_match(_ADDRESS_PATTERNS, "。".join(_asserted_sentences(message or "")))
    if not address:
        return None
    cleaned = _clean(address)
    if not _looks_like_name(cleaned):
        return None
    return _truncate(cleaned, 20)


def next_relationship_stage(current_stage: RelationshipStage, interaction_count: int) -> RelationshipStage:
    """Compatibility helper: counts never change a manually chosen starting point."""
    return current_stage if current_stage in _STAGE_ORDER else "stranger"


def _first_match(patterns: tuple[re.Pattern[str], ...], text: str) -> str | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group("name") if "name" in pattern.groupindex else match.group("subject")
    return None


def _all_matches(patterns: tuple[re.Pattern[str], ...], text: str, *, limit: int) -> list[str]:
    results: list[str] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            value = match.group("name") if "name" in pattern.groupindex else match.group("subject")
            if value and value not in results:
                results.append(value)
            if len(results) >= limit:
                return results
    return results


def _all_matches_with_position(patterns: tuple[re.Pattern[str], ...], text: str) -> list[tuple[int, str]]:
    """返回所有命中及其原文位置，用于解决同句内偏好冲突。"""
    results: list[tuple[int, str]] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            value = match.group("name") if "name" in pattern.groupindex else match.group("subject")
            if value:
                results.append((match.start(), value))
    return results


def _contains_question_marker(text: str) -> bool:
    return any(marker in text for marker in _QUESTION_MARKERS)


def _looks_like_name(text: str) -> bool:
    cleaned = _clean(text)
    return bool(cleaned and not _contains_question_marker(cleaned) and not cleaned.startswith(_INVALID_NAME_PREFIXES))


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"
