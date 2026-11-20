"""Extract reviewable outreach candidates from a user's own words."""

import re
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

_SENTENCE = re.compile(r"[^\n。！？!?；;]+")
_FOLLOW_UP = re.compile(r"(?:下次|改天|回头|有空|以后)(?:我们|再|继续){0,3}(?:聊|说|讨论)")
_EVENT = re.compile(r"面试|考试|答辩|复试|体检|比赛|演讲")
_CANCEL_OR_HYPOTHETICAL = re.compile(r"取消|推迟|改期|不确定|不一定|还没定|假如|如果|假设|虚构|剧本|不是我|不用问|别问|不要提|不想聊")
_PAST = re.compile(r"昨天|前天|上周|去年|结束了|考完了|面完了|已经")
_EVENT_INTENT = re.compile(r"我(?:要|准备|得|将|会|有|预约)|有(?:一场|个)?(?:面试|考试|答辩|复试|体检|比赛|演讲)|下周|下个月|周[一二三四五六日天]")
_FULL_DATE = re.compile(r"(?<!\d)(\d{4})[-年](\d{1,2})[-月](\d{1,2})(?:日|号)?(?!\d)")
_RELATIVE_DATE = re.compile(r"大后天|后天|明天|今天")
_DAY_OFFSET = {"今天": 0, "明天": 1, "后天": 2, "大后天": 3}


def _event_follow_up_at(quote, created_at, timezone):
    local = datetime.fromisoformat(created_at).astimezone(ZoneInfo(timezone))
    exact = _FULL_DATE.findall(quote)
    relative = _RELATIVE_DATE.findall(quote)
    if len(exact) + len(relative) != 1:
        return None
    try:
        day = (
            datetime(*(int(value) for value in exact[0])).date() if exact
            else local.date() + timedelta(days=_DAY_OFFSET[relative[0]])
        )
        if day < local.date():
            return None
        return datetime.combine(day + timedelta(days=1), time(9), local.tzinfo).timestamp()
    except (ValueError, OverflowError):
        return None


def outreach_candidates(turns, timezone, now):
    """Read newest-first active user turns. Dates only prefill a user-confirmed draft."""
    candidates = []
    seen_events = set()
    seen_topics = set()
    for turn in turns:
        for sentence in reversed(_SENTENCE.findall(turn["message"])):
            quote = sentence.strip()
            events = set(_EVENT.findall(quote))
            excluded = bool(_CANCEL_OR_HYPOTHETICAL.search(quote))
            kind, suggested_at = None, None
            if events:
                unseen = events - seen_events
                seen_events.update(events)
                if not excluded and unseen and not _PAST.search(quote) and (
                    _FULL_DATE.search(quote) or _RELATIVE_DATE.search(quote) or _EVENT_INTENT.search(quote)
                ):
                    kind = "event"
                    suggested_at = _event_follow_up_at(quote, turn["created_at"], timezone)
                    if suggested_at is not None and not now < suggested_at <= now + 366 * 86400:
                        suggested_at = None
            elif not excluded and _FOLLOW_UP.search(quote):
                kind = "follow_up"
            if kind is None or not 4 <= len(quote) <= 300 or (kind, quote) in seen_topics:
                continue
            seen_topics.add((kind, quote))
            candidates.append({"source_turn_id": turn["id"], "kind": kind, "topic": quote, "suggested_at": suggested_at})
    return candidates
