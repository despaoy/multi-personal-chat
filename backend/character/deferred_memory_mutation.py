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
