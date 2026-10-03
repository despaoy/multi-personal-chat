"""Keep a source-dated future assertion from retiring today's target claim.

This only prevents an immediate destructive relation. It neither schedules a
future replacement nor certifies any model-proposed lifetime.
"""

import re

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
    """Recover a preceding start only for one exact, uniquely bound source span.

    The admitted evidence may omit its own time prefix. That prefix must be
    in the same first-person clause, outside quoted speech. Independent comma
    clauses and repeated evidence never borrow another statement's time.
    This withholds an early mutation; it certifies no future execution.
    """
    if not evidence or not source_message:
        return None
    start = source_message.find(evidence)
    if start < 0 or source_message.find(evidence, start + 1) >= 0:
        return None
    from character.quoted_erasure_authority import masked_quotes

    try:
        masked = masked_quotes(source_message)[0]
    except ValueError:
        return None
    end = start + len(evidence)
    anchor = observation_clock(observed_at)
    for match in _SELF_CLAUSE.finditer(masked):
        if match.start() <= start and end <= match.end():
            expression = match["before"] or match["after"]
            resolved = resolve_temporal_expression(expression, observed_at=anchor)
            if resolved is not None and resolved.lower > anchor:
                return resolved
    return None
