"""Time expression resolution retains calendar precision and source anchors."""

import calendar
from datetime import datetime, timedelta, timezone

import pytest

from character.temporal_expression import resolve_temporal_expression

ANCHOR = datetime(2026, 12, 31, 16, 5, tzinfo=timezone.utc)  # Jan 1 2027 in +08.


@pytest.mark.parametrize("text,precision,start,end", [
    ("今天", "day", "2027-01-01", "2027-01-02"),
    ("昨天", "day", "2026-12-31", "2027-01-01"),
    ("明天", "day", "2027-01-02", "2027-01-03"),
    ("这周", "week", "2026-12-28", "2027-01-04"),
    ("下周", "week", "2027-01-04", "2027-01-11"),
    ("上星期", "week", "2026-12-21", "2026-12-28"),
    ("上个月", "month", "2026-12-01", "2027-01-01"),
    ("下个月", "month", "2027-02-01", "2027-03-01"),
    ("今年", "year", "2027-01-01", "2028-01-01"),
    ("2028-02-29", "day", "2028-02-29", "2028-03-01"),
    ("2028年2月", "month", "2028-02-01", "2028-03-01"),
    ("2028年", "year", "2028-01-01", "2029-01-01"),
])
def test_calendar_boundaries(text, precision, start, end):
    result = resolve_temporal_expression(text, observed_at=ANCHOR)
    assert result.kind == "calendar_period" and result.precision == precision
    assert result.lower.isoformat() == start + "T00:00:00+08:00"
    assert result.upper.isoformat() == end + "T00:00:00+08:00"
    assert result.observed_at == ANCHOR


@pytest.mark.parametrize("year", [1900, 2000, 2026, 2028, 2100])
@pytest.mark.parametrize("month", range(1, 13))
def test_all_month_lengths_against_calendar_oracle(year, month):
    span = resolve_temporal_expression(f"{year:04}-{month:02}", observed_at=ANCHOR)
    assert (span.upper - span.lower).days == calendar.monthrange(year, month)[1]
    assert span.lower.day == span.upper.day == 1


def test_explicit_instant_is_not_an_empty_interval():
    point = resolve_temporal_expression("2027-01-01T00:00:00.123456+08:00", observed_at=ANCHOR)
    assert point.kind == "instant" and point.precision == "microsecond"
    assert point.upper is None
    assert point.lower == datetime.fromisoformat("2026-12-31T16:00:00.123456+00:00")


def test_timezone_changes_calendar_day_but_not_instant():
    utc = resolve_temporal_expression("今天", observed_at=ANCHOR, zone=timezone.utc)
    local = resolve_temporal_expression("今天", observed_at=ANCHOR)
    assert utc.lower.date().isoformat() == "2026-12-31"
    assert local.lower.date().isoformat() == "2027-01-01"
    assert local.observed_at == utc.observed_at


def test_delay_does_not_change_source_anchored_period():
    old = resolve_temporal_expression("下周", observed_at=ANCHOR)
    assert resolve_temporal_expression("下周", observed_at=ANCHOR) == old
    assert resolve_temporal_expression("下周", observed_at=ANCHOR + timedelta(days=7)).lower != old.lower


@pytest.mark.parametrize("text", ["下周左右", "月底之前", "2026-02-29", "2026-13", "9999年",
                                 "明天下午", "2026-12-31T15:00:00", "下周回来", "", "2027"])
def test_unknown_or_invalid_expression_is_not_guessed(text):
    assert resolve_temporal_expression(text, observed_at=ANCHOR) is None


def test_source_clock_must_be_aware():
    with pytest.raises(ValueError, match="timezone"):
        resolve_temporal_expression("明天", observed_at=datetime(2026, 1, 1))
