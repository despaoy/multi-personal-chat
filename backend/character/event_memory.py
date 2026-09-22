"""Bounded event lifecycle rules; no inference, timers or model calls."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from character.models import MemoryItem

EVENT_TZ = timezone(timedelta(hours=8))
_TIME = r"今天|明天|后天|\d{4}-\d{2}-\d{2}"
_SUBJECT = r"[\w\u4e00-\u9fff-]{0,24}(?:面试|考试|答辩|会议|体检|旅行|项目|论文|申请|保研)"
_LABELS = {
    "planned": "计划中",
    "ongoing": "进行中",
    "completed": "已完成",
    "cancelled": "已取消",
    "postponed": "已延期",
}


@dataclass(frozen=True)
class EventChange:
    subject: str
    state: str
    scheduled_date: str = ""
    time_text: str = ""


def parse_event(sentence: str, reference_time: datetime | None = None) -> EventChange | None:
    """Only whole, explicit assertions accepted after the common write gate."""
    text = sentence.strip().rstrip("。！!；;")
    patterns = (
        ("planned", rf"(?:我)?(?P<time>{_TIME})(?:我)?(?:要|会|将)(?:去|参加)?(?P<subject>{_SUBJECT})"),
        ("ongoing", rf"我(?:正在|在)(?:准备|进行)(?P<subject>{_SUBJECT})"),
        ("completed", rf"(?:我的)?(?P<subject>{_SUBJECT})(?:已经|已)?(?:结束|完成)了?"),
        ("completed", rf"我(?:已经|已)?完成了?(?P<subject>{_SUBJECT})"),
        ("cancelled", rf"(?:我的)?(?P<subject>{_SUBJECT})(?:已经|已)?取消了?"),
        ("cancelled", rf"我(?:已经|已)?取消了?(?P<subject>{_SUBJECT})"),
        ("postponed", rf"(?:我的)?(?P<subject>{_SUBJECT})(?:已经|已)?(?:延期|推迟)(?:到(?P<time>{_TIME}))?了?"),
    )
    for state, pattern in patterns:
        match = re.fullmatch(pattern, text)
        if not match:
            continue
        subject = match["subject"]
        # Do not resolve pronouns, third-party reports or qualified/negative clauses.
        if re.search(
            r"那个|这个|我|你|他|她|朋友|同事|以为|觉得|认为|不是|没有|没|不|如果|可能|说|但|明天|今天|后天", subject
        ):
            return None
        raw_time = match.groupdict().get("time") or ""
        scheduled = ""
        if raw_time:
            now = reference_time or datetime.now(timezone.utc)
            if now.utcoffset() is None:
                raise ValueError("reference_time must be timezone-aware")
            if raw_time in {"今天", "明天", "后天"}:
                scheduled = (
                    now.astimezone(EVENT_TZ).date() + timedelta(days={"今天": 0, "明天": 1, "后天": 2}[raw_time])
                ).isoformat()
            else:
                try:
                    scheduled = date.fromisoformat(raw_time).isoformat()
                except ValueError:
                    return None
        return EventChange(subject, state, scheduled, raw_time)
    return None


def event_content(event: EventChange) -> str:
    when = f"；计划日期：{event.scheduled_date}" if event.scheduled_date else "；日期未确定"
    if event.time_text:
        when += f"（记录时表述：{event.time_text}）"
    return f"用户事项：{event.subject}；状态：{_LABELS[event.state]}{when}"


def event_reference_content(row: dict, now: datetime) -> str:
    """Render overdue as unknown outcome, never silently complete an event."""
    content = str(row.get("content") or "")
    if row.get("status", "active") not in {"active", "current"}:
        return content  # Historical versions describe what was known then.
    event = (row.get("metadata") or {}).get("event") or {}
    if not isinstance(event, dict):
        return content
    scheduled = event.get("scheduled_date")
    if scheduled and event.get("state") in {"planned", "ongoing", "postponed"}:
        try:
            overdue = date.fromisoformat(scheduled) < now.astimezone(EVENT_TZ).date()
        except (ValueError, TypeError):
            overdue = False
        if overdue:
            return f"用户事项：{event.get('subject', '')}；原计划日期：{scheduled}；日期已过，结果未知（不能当作当前待办或已完成）"
    return content


async def write_event_memory(repo, character_id, scope, item, source_message_id=None) -> bool:
    event = item.event
    if event is None or not item.evidence:
        return False
    for attempt in range(3):
        rows = await repo.list_memory_records(
            character_id,
            scope,
            limit=None,
            scope_levels=("conversation",),
            include_inactive=False,
        )
        matches = []
        for row in rows:
            if row.get("status", "active") != "active":
                continue
            info = (row.get("metadata") or {}).get("event") or {}
            if not isinstance(info, dict):
                continue
            if info.get("subject") == event.subject or row.get("memory_key") in {
                "goal_" + event.subject,
                "promise_" + event.subject,
            }:
                matches.append(row)
        if event.state in {"planned", "ongoing"}:
            # Distinct dated occurrences coexist. Never revive a closed occurrence.
            matches = [
                r
                for r in matches
                if (r.get("metadata") or {}).get("event", {}).get("scheduled_date", "") == event.scheduled_date
            ]
        else:
            matches = [
                r
                for r in matches
                if (r.get("metadata") or {}).get("event", {}).get("state") not in {"completed", "cancelled"}
            ]
            if len(matches) != 1:
                return False  # An ambiguous/orphan update must not change any event.
        if len(matches) > 1:
            return False
        previous = matches[0] if matches else None
        if previous is None and any(row.get("memory_key") == item.memory_key for row in rows):
            # A rescheduled occurrence still owns its original key. Do not guess
            # whether a new mention of the old date reopens or duplicates it.
            return False
        old_event = (previous.get("metadata") or {}).get("event", {}) if previous else {}
        if previous and old_event.get("state") in {"completed", "cancelled"}:
            return False
        scheduled = event.scheduled_date
        if event.state in {"completed", "cancelled"}:
            scheduled = old_event.get("scheduled_date", "")
        updated = EventChange(event.subject, event.state, scheduled, event.time_text)
        info = {
            "subject": updated.subject,
            "state": updated.state,
            "scheduled_date": scheduled,
            "time_text": event.time_text,
            "timezone": "+08:00",
        }
        if previous and all(old_event.get(k) == info[k] for k in ("subject", "state", "scheduled_date")):
            return False
        now = datetime.now(timezone.utc).isoformat()
        try:
            await repo.append_claim(
                character_id,
                scope,
                MemoryItem("", "shared_event", event_content(updated), item.importance),
                memory_key=previous["memory_key"] if previous else item.memory_key,
                relation_type="SUPERSEDE" if previous else "ADD",
                supersedes_memory_id=previous["id"] if previous else None,
                parent_memory_id=previous["id"] if previous else None,
                evidence=(item.evidence,),
                source_message_id=source_message_id,
                source_message_ids=(source_message_id,) if source_message_id else (),
                observed_at=now,
                valid_from=now,
                confidence=0.9,
                metadata={"origin": "rule_v2", "event": info},
            )
            return True
        except ValueError as exc:
            if str(exc) != "rule memory changed concurrently" or attempt == 2:
                raise
    return False
