"""Source-anchored calendar expressions, never inferred claim lifetimes.

The resolver takes an exact expression span and its message timestamp. The
optional source scanner finds lexical candidates only; neither API assigns a
subject or decides whether an expression means an event start/end. A calendar bucket is a range of possible dates, not
proof that a fact holds continuously throughout that range.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, tzinfo
from typing import Literal

from character.memory_clock import DEFAULT_MEMORY_TIMEZONE, observation_clock


@dataclass(frozen=True)
class TemporalExpression:
    text: str
    kind: Literal["calendar_period", "instant"]
    precision: Literal["day", "week", "month", "year", "second", "microsecond"]
    lower: datetime
    upper: datetime | None
    observed_at: datetime


_DAY_OFFSETS = {"前天": -2, "昨天": -1, "今天": 0, "今日": 0, "明天": 1, "后天": 2}
_PERIOD_OFFSETS = {"上上": -2, "上": -1, "本": 0, "这": 0, "下": 1, "下下": 2}
_YEAR_OFFSETS = {"前年": -2, "去年": -1, "今年": 0, "明年": 1, "后年": 2}

# Lexical spans are not semantic roles. Keep unresolved boundary/duration
# phrases too: failure to resolve a deadline must not imply a timeless fact.
_SOURCE_TIME_SPANS = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})?)?"
    r"|\d{4}(?:年(?:\d{1,2}月(?:\d{1,2}日)?)?|-\d{2})"
    r"|(?:上上|下下|上|下|本|这)(?:周|星期|(?:个)?月)"
    r"|" + "|".join(re.escape(word) for word in (*_DAY_OFFSETS, *_YEAR_OFFSETS))
    + r"|(?:[一二三四五六七八九十百两\d]+)(?:个)?(?:小时|星期|个月|天|周|月|年)"
    r"|(?:年|月)(?:初|中|底|末)|(?:直到|截至|截止|到期|之前|之后|期间|暂时|临时)"
)


def source_temporal_spans(text: str) -> tuple[str, ...]:
    """Find time-like source spans without assigning roles or inventing dates.

    This is a conservative lexical signal, not proof of the meaning of a
    sentence. It can also find temporal-looking words inside names/quotations;
    callers must retain that text rather than assert the span's applicability.
    """
    return tuple(match[0] for match in _SOURCE_TIME_SPANS.finditer(text))


def _month_start(year: int, month: int, offset: int, zone: tzinfo) -> datetime:
    shifted_year, shifted_month = divmod(year * 12 + month - 1 + offset, 12)
    return datetime(shifted_year, shifted_month + 1, 1, tzinfo=zone)


def resolve_temporal_expression(
    text: str, *, observed_at: datetime, zone: tzinfo = DEFAULT_MEMORY_TIMEZONE,
) -> TemporalExpression | None:
    """Resolve supported exact spans; leave ambiguous or invalid spans unknown.

    Week starts follow the existing Chinese calendar convention (Monday).
    A timezone-less clock time is intentionally unsupported. Date-only input
    uses the caller's explicit calendar zone (project +08:00 by default).
    """
    anchor = observation_clock(observed_at)
    local = anchor.astimezone(zone)
    cleaned = text.strip()
    if not cleaned:
        return None

    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})", cleaned):
            instant = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
            return TemporalExpression(cleaned, "instant", "microsecond" if "." in cleaned else "second",
                                      instant, None, anchor)

        midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
        precision = None
        lower = upper = None
        if cleaned in _DAY_OFFSETS:
            lower = midnight + timedelta(days=_DAY_OFFSETS[cleaned])
            upper, precision = lower + timedelta(days=1), "day"
        elif match := re.fullmatch(r"(上上|上|本|这|下下|下)(周|星期)", cleaned):
            lower = midnight - timedelta(days=local.weekday()) + timedelta(weeks=_PERIOD_OFFSETS[match[1]])
            upper, precision = lower + timedelta(weeks=1), "week"
        elif match := re.fullmatch(r"(上上|上|本|这|下下|下)(?:个)?月", cleaned):
            offset = _PERIOD_OFFSETS[match[1]]
            lower = _month_start(local.year, local.month, offset, zone)
            upper, precision = _month_start(local.year, local.month, offset + 1, zone), "month"
        elif cleaned in _YEAR_OFFSETS:
            year = local.year + _YEAR_OFFSETS[cleaned]
            lower = datetime(year, 1, 1, tzinfo=zone)
            upper, precision = datetime(year + 1, 1, 1, tzinfo=zone), "year"
        else:
            # Two closed notations; no fuzzy parsing or missing-year guessing.
            match = re.fullmatch(r"(?P<year>\d{4})-(?P<month>\d{2})(?:-(?P<day>\d{2}))?", cleaned)
            if match is None:
                match = re.fullmatch(r"(?P<year>\d{4})年(?:(?P<month>\d{1,2})月(?:(?P<day>\d{1,2})日)?)?", cleaned)
            if match is None:
                return None
            year = int(match["year"])
            month = int(match["month"]) if match["month"] else 1
            day = int(match["day"]) if match["day"] else 1
            lower = datetime(year, month, day, tzinfo=zone)
            if match["day"]:
                upper, precision = lower + timedelta(days=1), "day"
            elif match["month"]:
                upper, precision = _month_start(year, month, 1, zone), "month"
            else:
                upper, precision = datetime(year + 1, 1, 1, tzinfo=zone), "year"
        return TemporalExpression(cleaned, "calendar_period", precision, lower, upper, anchor)
    except (ValueError, OverflowError):
        return None
