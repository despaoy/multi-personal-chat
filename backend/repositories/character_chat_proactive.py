"""Durable, consent-fenced in-app messages for a single character session."""

from datetime import UTC, datetime
from time import time
from zoneinfo import ZoneInfo

from db.errors import RevisionConflictError
from repositories.character_chat import CharacterChatRepository
from repositories.character_library import CharacterNotFoundError

_DEFAULT = {
    "revision": 0, "enabled": False, "cadence_hours": 48, "timezone": "UTC",
    "quiet_start": 22 * 60, "quiet_end": 8 * 60, "last_sent_at": None,
}
_TEXT = {
    "follow_up": "上次说到「{topic}」，如果你想继续，我还在听。",
    "event": "你之前提过「{topic}」。如果愿意聊，后来怎么样了？",
    "fiction": "【角色虚构日常】今天翻开旧日记，想到「{topic}」。下次见面时想听听你的想法。",
}


def in_quiet_hours(policy, now):
    start, end = policy["quiet_start"], policy["quiet_end"]
    if start is None or end is None:
        return False
    local = datetime.fromtimestamp(now, UTC).astimezone(ZoneInfo(policy["timezone"]))
    minute = local.hour * 60 + local.minute
    return start <= minute < end if start < end else minute >= start or minute < end


class CharacterChatProactiveRepository:
    def __init__(self, database):
        self.db = database
        self.chats = CharacterChatRepository(database)

    def _policy(self, session_id):
        rows = self.db.execute_sql(
            "SELECT revision,enabled,cadence_hours,timezone,quiet_start,quiet_end,last_sent_at "
            "FROM character_chat_proactive_policies WHERE session_id=:session", {"session": session_id},
        )
        return {**rows[0], "enabled": bool(rows[0]["enabled"])} if rows else dict(_DEFAULT)

    def _public_policy(self, session_id):
        policy = self._policy(session_id)
        del policy["last_sent_at"]
        return policy

    @staticmethod
    def _session_guard(session):
        return (
            "UPDATE character_chat_sessions SET revision=revision "
            "WHERE id=:session AND owner_user_id=:owner AND revision=:session_revision AND mode<>'branch'",
            {"session": session["id"], "owner": session["owner_user_id"], "session_revision": session["revision"]}, 1,
        )

    def view(self, owner, session_id):
        session = self.chats.get(owner, session_id)
        jobs = self.db.execute_sql(
            "SELECT id,kind,topic,due_at,sent_at,content,status FROM character_chat_proactive_jobs "
            "WHERE session_id=:session AND status IN ('pending','sent') "
            "ORDER BY CASE WHEN status='pending' THEN 0 ELSE 1 END, COALESCE(sent_at,due_at) DESC,id LIMIT 70",
            {"session": session_id},
        )
        return {"policy": self._public_policy(session_id), "available": session["mode"] != "branch", "jobs": jobs}

    def recent_sent(self, owner, session_id):
        rows = self.db.execute_sql(
            "SELECT j.kind,j.content FROM character_chat_proactive_jobs j "
            "JOIN character_chat_sessions s ON s.id=j.session_id "
            "WHERE j.session_id=:session AND s.owner_user_id=:owner AND s.mode<>'branch' "
            "AND j.status='sent' ORDER BY j.sent_at DESC,j.id LIMIT 3",
            {"session": session_id, "owner": owner},
        )
        return list(reversed(rows))

    def set_policy(self, owner, session_id, revision, **settings):
        session = self.chats.get(owner, session_id)
        if session["mode"] == "branch":
            raise ValueError("假想分支不发送会话外消息")
        params = {**settings, "session": session_id, "revision": revision, "enabled": int(settings["enabled"])}
        statement = (
            "INSERT INTO character_chat_proactive_policies "
            "(session_id,revision,enabled,cadence_hours,timezone,quiet_start,quiet_end) "
            "VALUES (:session,1,:enabled,:cadence_hours,:timezone,:quiet_start,:quiet_end) "
            "ON CONFLICT (session_id) DO NOTHING"
            if revision == 0 else
            "UPDATE character_chat_proactive_policies SET revision=revision+1,enabled=:enabled,"
            "cadence_hours=:cadence_hours,timezone=:timezone,quiet_start=:quiet_start,quiet_end=:quiet_end "
            "WHERE session_id=:session AND revision=:revision"
        )
        self.db.revision_transaction([
            self._session_guard(session), (statement, params, 1),
            ("UPDATE character_chat_proactive_jobs SET status='cancelled',topic='' "
             "WHERE session_id=:session AND status='pending'", params, None),
        ])
        return self._public_policy(session_id)

    def schedule(self, owner, session_id, revision, request_id, kind, topic, due_at, *, now=None):
        now = time() if now is None else now
        session = self.chats.get(owner, session_id)
        policy = self._policy(session_id)
        if not policy["enabled"] or revision != policy["revision"]:
            raise RevisionConflictError()
        params = {"session": session_id, "id": request_id, "revision": revision,
                  "kind": kind, "topic": topic, "due": due_at}
        query = ("SELECT id,kind,topic,due_at,status,policy_revision FROM character_chat_proactive_jobs "
                 "WHERE session_id=:session AND id=:id")

        def existing():
            rows = self.db.execute_sql(query, params)
            if rows:
                job = rows[0]
                if (job["kind"], job["topic"], job["due_at"], job["policy_revision"]) != (kind, topic, due_at, revision):
                    raise RevisionConflictError()
                return job
            return None

        if saved := existing():
            return saved
        if not now <= due_at <= now + 366 * 86400 or kind not in _TEXT or not topic.strip():
            raise ValueError("请填写有效话题和未来一年内的提醒时间")
        self.db.revision_transaction([
            self._session_guard(session),
            ("UPDATE character_chat_proactive_policies SET revision=revision "
             "WHERE session_id=:session AND revision=:revision AND enabled=1", params, 1),
            ("INSERT INTO character_chat_proactive_jobs "
             "(session_id,id,policy_revision,kind,topic,due_at,next_attempt_at,status) "
             "SELECT :session,:id,:revision,:kind,:topic,:due,:due,'pending' "
             "WHERE (SELECT COUNT(*) FROM character_chat_proactive_jobs WHERE session_id=:session AND status='pending')<20 "
             "AND NOT EXISTS (SELECT 1 FROM character_chat_proactive_jobs WHERE session_id=:session "
             "AND kind=:kind AND topic=:topic AND due_at=:due AND status IN ('pending','sent')) "
             "ON CONFLICT (session_id,id) DO NOTHING", params, None),
        ])
        if saved := existing():
            return saved
        raise ValueError("该话题和时间已安排，或待发送消息已达 20 条")

    def has_due(self, now):
        return bool(self.db.execute_sql(
            "SELECT id FROM character_chat_proactive_jobs "
            "WHERE status='pending' AND next_attempt_at<=:now LIMIT 1", {"now": now},
        ))

    def deliver_due(self, *, now=None):
        """Run a restart-safe worker sweep. No private text leaves the database worker."""
        now = time() if now is None else now
        jobs = self.db.execute_sql(
            "SELECT j.session_id,j.id,j.kind,j.topic,j.policy_revision,s.owner_user_id "
            "FROM character_chat_proactive_jobs j JOIN character_chat_sessions s ON s.id=j.session_id "
            "WHERE j.status='pending' AND j.next_attempt_at<=:now ORDER BY j.next_attempt_at,j.id LIMIT 100",
            {"now": now},
        )
        sent = 0
        for job in jobs:
            try:
                sent += self._deliver(job, now)
            except (RevisionConflictError, CharacterNotFoundError):
                # A newer user action or another worker won the race.
                continue
        return sent

    def _deliver(self, job, now):
        session_id = job["session_id"]
        session = self.chats.get(job["owner_user_id"], session_id)
        policy = self._policy(session_id)
        if not policy["enabled"] or policy["revision"] != job["policy_revision"]:
            return False
        last = policy["last_sent_at"]
        next_allowed = last + policy["cadence_hours"] * 3600 if last is not None else now
        params = {"session": session_id, "id": job["id"], "revision": policy["revision"], "now": now}
        if in_quiet_hours(policy, now) or next_allowed > now:
            self.db.revision_transaction([
                ("UPDATE character_chat_proactive_jobs SET next_attempt_at=:next "
                 "WHERE session_id=:session AND id=:id AND policy_revision=:revision AND status='pending'",
                 {**params, "next": max(now + 60, next_allowed)}, None),
            ])
            return False
        self.db.revision_transaction([
            self._session_guard(session),
            ("UPDATE character_chat_proactive_policies SET last_sent_at=:now "
             "WHERE session_id=:session AND revision=:revision AND enabled=1 "
             "AND (last_sent_at IS NULL OR last_sent_at<=:cutoff)",
             {**params, "cutoff": now - policy["cadence_hours"] * 3600}, 1),
            ("UPDATE character_chat_proactive_jobs SET status='sent',sent_at=:now,content=:content "
             "WHERE session_id=:session AND id=:id AND policy_revision=:revision "
             "AND status='pending' AND next_attempt_at<=:now",
             {**params, "content": _TEXT[job["kind"]].format(topic=job["topic"])}, 1),
        ])
        return True
