"""Conservative conversational task ownership before topic-based routing.

Topic words are not requests. A negative decision requires every clause to be a
recognized personal task or self-report; an extra unrecognized clause restores
the ordinary router. No generated answer or assistant history is consulted.
"""
from __future__ import annotations

import re

from character.conditional_memory import parse_necessary_condition
from character.memory_query import profile_lookup_fields, storage_fields
from character.memory_request import memory_statement_body
from knowledge.query_tasks import is_dialogue_control_clause, requests_explicit_information, requests_source_text

_PERSONAL = re.compile(
    r'(?:你(?:还)?记得)?我(?:叫(?:什么|什么名字)|(?:不)?喜欢什么(?:饮料|音乐|食物|运动|颜色|书|电影|游戏)?|'
    r'最近要[^，。！？;；]{0,12}什么|之前说过什么|说过什么)'
    r'(?:吗|呢)?$|'
    r'(?:你(?:还)?记得)?我的(?:姓名|名字|专业|工作地点|工作单位|现居地|居住地|籍贯|年级)'
    r'(?:(?:和|与|、)(?:姓名|名字|专业|工作地点|工作单位|现居地|居住地|籍贯|年级))*'
    r'(?:是什么|是哪里|在哪里|有哪些|吗|呢)?$'
)
_REQUEST = re.compile(
    r'请|帮我|给我|告诉我|'
    r'解释|说明|介绍|讲解|查询|搜索|查找|比较|分析|检索|推荐|列举|总结|翻译|说说|讲讲|谈谈|查查|查一查')
_QUESTION = re.compile(r'[？?]|(?:什么|为什么|怎么|如何|怎样|哪里|哪儿|哪本|哪个|哪些|哪位|哪种|谁|多少|多久|几点)|(?:吗|么|呢)$')
_CHOICE_QUESTION = re.compile(r'([\u4e00-\u9fff]{1,2})不\1|有没有')
_WANT_INFORMATION = re.compile(r'(?:想|要|希望|需要)(?:你)?(?:了解|知道|问|弄明白|学习|听|看)')
_VOLITION = re.compile(r'^(?:我(?:们)?|今天|最近|现在|目前)(?:今天|最近|现在|目前|也)?'
                       r'(?:想|要|希望|需要|打算|准备)')
_TIME = r'(?:今天|昨天|明天|最近|目前|现在|刚刚|这几天)'
# An owner/time prefix alone is not a declaration: "我问你...", "今天请..."
# are requests too. Admit only bounded declarative predicates; unknown forms
# defer to the existing router, sacrificing savings rather than retrieval.
_REPORT_START = re.compile(
    rf'^(?:我(?:们)?(?:{_TIME})?(?:刚(?:刚)?|已经|正在|一直|也)?'
    r'(?:叫|来自|住在|在.+工作|的(?:专业|名字|姓名|工作地点|工作单位|现居地|居住地|籍贯|年级)是|'
    r'(?:不怎么|不太|不|很)?喜欢|讨厌|学会|学了|在看|看完|读完|整理好)|'
    rf'{_TIME}(?:没有什么|没什么|没怎么|不怎么|在.+工作))')
_NON_SPECIFIC = re.compile(r'(?:没(?:有)?什么|不怎么|没怎么|没多少|没几)')
_UNDISCLOSED = re.compile(r'我(?:还|一直)?(?:没|没有)(?:告诉(?:过)?你|向你(?:说过|提过|提到))(.+)')


def has_question_form(message: str) -> bool:
    # Explicit speech acts and source requests do not require a question mark.
    # Source intent is shared with query construction and source rendering.
    return bool(_QUESTION.search(message) or _CHOICE_QUESTION.search(message)
                or requests_explicit_information(message) or requests_source_text(message))


def local_context_only(message: str) -> bool:
    """Prove no external lookup dependency within a bounded clause grammar."""
    # Remembering a self-report does not create an external fact lookup.
    # Strip only the leading management wrapper; the complete remainder must
    # still pass the existing dependency checks, including mixed tasks.
    statement = memory_statement_body(message)
    if statement != message.strip():
        if not statement:
            return False
        message = statement
    if profile_lookup_fields(message):
        return True
    if storage_fields(message):
        return True
    clauses = [part.strip() for part in re.split(r'[，,。！？!?；;\n]', message) if part.strip()]
    if not clauses:
        return False
    # Identify complete conversation-management clauses before looking for
    # request verbs. "I want to talk" and "tell me external facts" have
    # different owners/dependencies. Never exempt a whole turn by one match.
    if '?' not in message and '？' not in message:
        clauses = [clause for clause in clauses if not is_dialogue_control_clause(clause)]
    if not clauses:
        return True
    # Do not let a personal prefix swallow a second task without punctuation.
    remaining = '，'.join(clauses)
    if _REQUEST.search(remaining) or _WANT_INFORMATION.search(remaining):
        return False
    # Reuse the writer's complete, self-owned declaration grammar. Parse the
    # whole turn, not clauses: a second task must not disappear behind a rule.
    # The parser validates storage syntax, so question/source intent still
    # needs its own turn-level veto before bypassing external retrieval.
    if not has_question_form(message) and parse_necessary_condition(message):
        return True
    for clause in clauses:
        if _PERSONAL.fullmatch(clause):
            continue
        undisclosed = _UNDISCLOSED.fullmatch(clause)
        if undisclosed and _PERSONAL.fullmatch(undisclosed.group(1)):
            # "what my name is" is the object of a disclosure statement,
            # not a new external question. Compound tasks still fail above.
            continue
        # A question mark is retained as a turn-level veto for reports. Personal
        # queries above may contain one, but unknown interrogatives must defer.
        if ('？' in message or '?' in message or _VOLITION.search(clause) or not _REPORT_START.search(clause)
                or _QUESTION.search(_NON_SPECIFIC.sub('', clause)) or _CHOICE_QUESTION.search(clause)):
            return False
    return True
