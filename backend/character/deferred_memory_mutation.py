"""Keep a source-dated future assertion from retiring today's target claim.

This only prevents an immediate destructive relation. It neither schedules a
future replacement nor certifies any model-proposed lifetime.
"""

import re
from dataclasses import replace

from character.memory_clock import observation_clock
from character.temporal_expression import resolve_temporal_expression

_TIME = (
    r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2}))?"
    r"|\d{4}年(?:\d{1,2}月(?:\d{1,2}日)?)?"
    r"|今天|今日|明天|后天|昨天|前天|今年|明年|后年|去年|前年"
    r"|(?:上上|下下|上|下|本|这)(?:周|星期|(?:个)?月)"
)
_SELF_START = re.compile(
    rf"(?:^|[。！？!?；;：:\n]|[，,]\s*(?:但|不过|然而|而))\s*"
    rf"(?:从(?P<before>{_TIME})(?:起|开始)[，,]?\s*我"
    rf"|我从(?P<after>{_TIME})(?:起|开始))[^。！？!?；;\n]+"
)


_SELF_CLAUSE = re.compile(
    rf"(?:^|[。！？!?；;：:\n]|[，,]\s*(?:但|不过|然而|而))\s*"
    rf"(?:从(?P<before>{_TIME})(?:起|开始)[，,]?\s*我"
    rf"|我从(?P<after>{_TIME})(?:起|开始))[^，,。！？!?；;\n]+[。！？!?；;\n]?"
)


def deferred_source_start(evidence, *, observed_at):
    """Use a literal first-person start span and the actual message clock.

    Unknown grammar and dates stay unknown. Calendar lower bounds only prove
    that a clearly future bucket has not started; they do not execute a date.
    The caller already validates source ownership and continuous evidence.
    """
    anchor = observation_clock(observed_at)
    for match in _SELF_START.finditer(evidence):
        expression = match["before"] or match["after"]
        resolved = resolve_temporal_expression(expression, observed_at=anchor)
        if resolved is not None and resolved.lower > anchor:
            return resolved
    return None


def deferred_evidence_start(evidence, source_message, *, observed_at):
    """Bind parser-whitespace-equivalent evidence to original source positions.

    Whitespace normalization only locates the admitted span. The grammar view
    preserves original hard line boundaries and quote masks. Dates come from
    that same self clause and the actual clock; sources are never rewritten.
    Repeated normalized occurrences cannot choose a guessed future version.
    """
    if not evidence or not source_message:
        return None
    needle = "".join(evidence.split())
    positions = [index for index, char in enumerate(source_message) if not char.isspace()]
    compact = "".join(source_message[index] for index in positions)
    start = compact.find(needle)
    if not needle or start < 0 or compact.find(needle, start + 1) >= 0:
        return None
    raw_start, raw_end = positions[start], positions[start + len(needle) - 1] + 1
    from character.quoted_erasure_authority import masked_quotes

    try:
        masked = masked_quotes(source_message)[0]
    except ValueError:
        return None
    if masked[raw_start].isspace():
        return None  # A quoted command cannot borrow the reporter's start.
    grammar_positions = [index for index, char in enumerate(source_message) if not char.isspace() or char in "\r\n"]
    grammar = "".join("\n" if source_message[index] in "\r\n" else masked[index] for index in grammar_positions)
    anchor = observation_clock(observed_at)
    for match in _SELF_CLAUSE.finditer(grammar):
        clause_start = grammar_positions[match.start()]
        clause_end = grammar_positions[match.end() - 1] + 1
        if clause_start <= raw_start and raw_end <= clause_end:
            group = "before" if match["before"] else "after"
            resolved = resolve_temporal_expression(match[group], observed_at=anchor)
            if resolved is not None and resolved.lower > anchor:
                a, z = match.span(group)
                literal = source_message[grammar_positions[a] : grammar_positions[z - 1] + 1]
                return replace(resolved, text=literal)
    return None
