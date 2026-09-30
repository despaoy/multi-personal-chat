"""Shared lightweight retrieval tasks used by routing and evidence assembly."""

from __future__ import annotations

import re

_SOURCE = re.compile(r"原文|原作|原著|原书|原句|出处|依据|引用|引文|来源|章节|证据|段落|哪一段")
_REQUEST = re.compile(r"请|能|可否|是否|哪|什么|怎么|给出|指出|列出|找出|出示|提供|[?？]")
_NO_INTERPRETATION = re.compile(
    r"(?:请)?(?:不要|不用|不需要|无需|不必|别)(?:再)?(?:给我)?"
    r"(?:分析|解释|总结|评价|建议|推测)(?:了)?"
)
_EXPLICIT_INFORMATION_REQUEST = re.compile(
    r"^(?:请问|我(?:想)?问你)|"
    r"我(?:想|希望|需要)(?:你)?(?:了解|知道|弄明白)|"
    r"(?:请|麻烦你|能否|能不能)(?:说说|讲讲|谈谈|介绍|解释|分析|查询|检索)|"
    r"^我对[^，。！？!?]{1,80}(?:很好奇|有疑问)[。！？!?]*$"
)
_QUOTED_TEXT = re.compile(r'“[^”]*”|「[^」]*」|『[^』]*』|"[^"\n]*"')
_DIRECT_INFORMATION_REQUEST = re.compile(
    r"(?:^|[，。！？,!?；;\s])"
    r"(?:(?:但(?:是)?|不过|可(?:是)?)[，,\s]*)?"
    r"(?:(?:请你?|麻烦你?|只|就|直接|顺便|现在|快|先|再|接着|随后|最后|你能不能|你可不可以|"
    r"你(?:能|可以)?|能不能|可不可以|能|可以|同时|分别|一并)\s*){0,3}"
    r"(?:告诉我|回答(?:一下)?|说清(?:楚)?|列出|"
    r"(?:查(?:阅|询|找)?|检索|搜索)(?:一下)?(?:知识库|资料库|数据库))"
)
_DIALOGUE_CONTROL = re.compile(
    r"(?:请)?(?:不用|不要|不需要|别)(?:再)?(?:给我|替我)?(?:建议|分析|追问|劝我)(?:了)?|"
    r"我(?:只是|只)想(?:说说|聊聊|讲讲|倾诉)(?:(?:今天|昨天|最近)的事|"
    r"(?:我|自己)的(?:感受|心情|经历|近况))|"
    r"陪我(?:说说话|聊聊天|聊聊|说话)|听我(?:说说|讲讲)(?:就好|就行)"
)


def is_dialogue_control_clause(text: str) -> bool:
    return bool(_DIALOGUE_CONTROL.fullmatch(text))


def independent_content_query(text: str) -> str:
    """Identify one content task alongside complete conversation controls.

    This is a task-classification view, never a replacement user message.
    Unknown clauses, quotation and multiple content tasks retain the original.
    Question punctuation on a control clause prevents treating it as a control.
    """
    if any(mark in text for mark in ("“", "”", "「", "」", "『", "』", '"')):
        return text
    clauses = [part.strip() for part in re.split(r"[，,。；;\n]", text) if part.strip()]
    remaining = [part for part in clauses if not is_dialogue_control_clause(part)]
    if len(remaining) == 1 and len(remaining) < len(clauses):
        return remaining[0]
    return text


def requests_explicit_information(text: str) -> bool:
    """Shared explicit information act, independent of question punctuation.

    This signals an information task, not its external lookup dependency.
    Memory questions and source requests retain their separate handling.
    """
    # Quotes are data, not current speaker requests. Keep offsets stable and
    # require a direct clause prefix rather than searching reported speech.
    visible = _QUOTED_TEXT.sub(lambda match: " " * len(match[0]), text)
    if _DIRECT_INFORMATION_REQUEST.search(visible):
        return True
    for match in _EXPLICIT_INFORMATION_REQUEST.finditer(visible):
        prefix = re.split(r"[，,。！？!?；;\n]", visible[: match.start()])[-1].strip()
        prefix = re.sub(r"^(?:但|不过|可是)\s*", "", prefix)
        if not prefix or _NO_INTERPRETATION.fullmatch(prefix):
            return True
    return False


def requests_source_text(text: str) -> bool:
    """Identify requests for source support, not mere mentions of reading.

    This only chooses evidence presentation. It never certifies that a source
    supports an earlier model claim, nor imports that claim into retrieval.
    """
    return bool(_SOURCE.search(text) and (_REQUEST.search(text) or _SOURCE.fullmatch(text.strip("。！!？? \t"))))


def requests_only_source_excerpt(text: str) -> bool:
    """Narrow display task; mixed interpretation tasks still need generation."""
    # A complete negative presentation instruction is not an extra requested
    # interpretation task. Remove only such clauses, never arbitrary prefixes
    # containing negation or other personal/factual requests.
    clauses = [part.strip() for part in re.split(r"[，,。；;\n]", text) if part.strip()]
    remaining = [part for part in clauses if not _NO_INTERPRETATION.fullmatch(part)]
    if len(remaining) != len(clauses):
        if len(remaining) != 1:
            return False
        text = re.sub(r"^(?:但|不过)\s*", "", remaining[0])
    if not requests_source_text(text):
        return False
    if re.search(
        r"不要|不用|无需|不必|别|分析|解释|翻译|总结|改写|比较|对比|评价|为什么|如何|怎么|"
        r"并且|以及|同时|顺便|然后|另外|还要|而且|[，,；;\n]",
        text,
    ):
        return False
    return bool(
        re.search(r"给出|指出|列出|找出|查找|出示|提供|摘录|引用|展示", text)
        or _SOURCE.fullmatch(text.strip("。！!？? \t"))
    )
