"""Durable, consent-fenced in-app messages for a single character session."""

from datetime import UTC, datetime
from time import time
from uuid import uuid4
from zoneinfo import ZoneInfo

from character.proactive_candidates import outreach_candidates
from db.errors import RevisionConflictError
from repositories.character_chat import CharacterChatRepository
from repositories.character_library import CharacterNotFoundError

_DEFAULT = {
    "revision": 0, "enabled": False, "cadence_hours": 48, "timezone": "UTC",
    "quiet_start": 22 * 60, "quiet_end": 8 * 60, "last_sent_at": None,
    "inactivity_hours": None, "inactivity_due_at": None, "inactivity_turn_id": None,
    "inactivity_last_sent_turn_id": None,
}
_INACTIVITY_GREETING = "有一阵子没见了。等你有空，想听听你最近过得怎么样。"
_INACTIVITY_READY = (
    "WHERE p.enabled=1 AND p.inactivity_due_at<=:now "
    "AND NOT EXISTS (SELECT 1 FROM character_chat_generations g "
    "WHERE g.session_id=p.session_id AND g.status='running' AND g.lease_until>:now)"
)
FICTION_LABEL = "【角色虚构日常】"
_TEXT = {
    "follow_up": "上次说到「{topic}」，如果你想继续，我还在听。",
    "event": "你之前提过「{topic}」。如果愿意聊，后来怎么样了？",
    "fiction": FICTION_LABEL + "今天翻开旧日记，想到「{topic}」。下次见面时想听听你的想法。",
}
_FICTION_LEASE_SECONDS = 90


def in_quiet_hours(policy, now):
    start, end = policy["quiet_start"], policy["quiet_end"]
    if start is None or end is None:
        return False
    local = datetime.fromtimestamp(now, UTC).astimezone(ZoneInfo(policy["timezone"]))
    minute = local.hour * 60 + local.minute
    return start <= minute < end if start < end else minute >= start or minute < end


def next_allowed_at(policy, now):
    last = policy["last_sent_at"]
    next_allowed = last + policy["cadence_hours"] * 3600 if last is not None else now
    return max(now + 60, next_allowed) if in_quiet_hours(policy, now) or next_allowed > now else None


class CharacterChatProactiveRepository:
    def __init__(self, database):
        self.db = database
        self.chats = CharacterChatRepository(database)

    def _policy(self, session_id):
        rows = self.db.execute_sql(
            "SELECT revision,enabled,cadence_hours,timezone,quiet_start,quiet_end,last_sent_at,"
            "inactivity_hours,inactivity_due_at,inactivity_turn_id,inactivity_last_sent_turn_id "
            "FROM character_chat_proactive_policies WHERE session_id=:session", {"session": session_id},
        )
        return {**rows[0], "enabled": bool(rows[0]["enabled"])} if rows else dict(_DEFAULT)

    def _public_policy(self, session_id):
        policy = self._policy(session_id)
        return {key: policy[key] for key in ("revision", "enabled", "cadence_hours", "timezone",
                                             "quiet_start", "quiet_end", "inactivity_hours")}

    def _candidates(self, session_id, timezone, now):
        turns = self.db.execute_sql(
            "SELECT id,message,created_at FROM character_chat_turns "
            "WHERE session_id=:session AND active=1 ORDER BY position DESC LIMIT 30",
            {"session": session_id},
        )
        return outreach_candidates(turns, timezone, now)

    def candidates(self, owner, session_id):
        session = self.chats.get(owner, session_id)
        policy = self._policy(session_id)
        if session["mode"] == "branch" or not policy["enabled"]:
            return []
        already_used = {(row["source_turn_id"], row["kind"], row["topic"]) for row in self.db.execute_sql(
            "SELECT source_turn_id,kind,topic FROM character_chat_proactive_jobs "
            "WHERE session_id=:session AND source_turn_id IS NOT NULL AND status IN ('pending','sent')",
            {"session": session_id},
        )}
        return [item for item in self._candidates(session_id, policy["timezone"], time())
                if (item["source_turn_id"], item["kind"], item["topic"]) not in already_used][:8]

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

    def inbox(self, owner):
        return self.db.execute_sql(
            "SELECT s.id AS session_id,s.character_id,s.title,j.id,j.kind,j.content,j.sent_at "
            "FROM character_chat_sessions s JOIN character_chat_proactive_jobs j ON j.session_id=s.id "
            "AND j.id=(SELECT latest.id FROM character_chat_proactive_jobs latest "
            "WHERE latest.session_id=s.id AND latest.status='sent' ORDER BY latest.sent_at DESC,latest.id LIMIT 1) "
            "WHERE s.owner_user_id=:owner ORDER BY j.sent_at DESC,s.id LIMIT 25",
            {"owner": owner},
        )

    def fiction_history(self, owner, session_id, before=None, limit=20):
        session = self.chats.get(owner, session_id)
        if session["mode"] == "branch":
            return {"entries": [], "next_before": None}
        params = {"session": session_id, "limit": limit + 1}
        scope = "WHERE session_id=:session AND status='sent' AND kind='fiction'"
        boundary = ""
        if before is not None:
            cursor = self.db.execute_sql(
                "SELECT sent_at FROM character_chat_proactive_jobs " + scope + " AND id=:before",
                {**params, "before": before},
            )
            if not cursor:
                raise CharacterNotFoundError("剧情记录已变更，请重新载入")
            params.update(before=before, time=cursor[0]["sent_at"])
            boundary = " AND (sent_at<:time OR (sent_at=:time AND id<:before))"
        rows = self.db.execute_sql(
            "SELECT id,content,sent_at FROM character_chat_proactive_jobs " + scope + boundary
            + " ORDER BY sent_at DESC,id DESC LIMIT :limit", params,
        )
        return {"entries": rows[:limit], "next_before": rows[limit - 1]["id"] if len(rows) > limit else None}

    def set_policy(self, owner, session_id, revision, *, inactivity_hours=None, now=None, **settings):
        session = self.chats.get(owner, session_id)
        if session["mode"] == "branch":
            raise ValueError("假想分支不发送会话外消息")
        idle_turn = idle_due = None
        if settings["enabled"] and inactivity_hours is not None:
            turns = self.db.execute_sql(
                "SELECT id FROM character_chat_turns WHERE session_id=:session AND active=1 ORDER BY position DESC LIMIT 1",
                {"session": session_id},
            )
            if turns:
                idle_turn = turns[0]["id"]
                idle_due = (time() if now is None else now) + inactivity_hours * 3600
        params = {**settings, "session": session_id, "revision": revision, "enabled": int(settings["enabled"]),
                  "idle_hours": inactivity_hours, "idle_turn": idle_turn, "idle_due": idle_due}
        statement = (
            "INSERT INTO character_chat_proactive_policies "
            "(session_id,revision,enabled,cadence_hours,timezone,quiet_start,quiet_end,"
            "inactivity_hours,inactivity_turn_id,inactivity_due_at) "
            "VALUES (:session,1,:enabled,:cadence_hours,:timezone,:quiet_start,:quiet_end,:idle_hours,:idle_turn,:idle_due) "
            "ON CONFLICT (session_id) DO NOTHING"
            if revision == 0 else
            "UPDATE character_chat_proactive_policies SET revision=revision+1,enabled=:enabled,"
            "cadence_hours=:cadence_hours,timezone=:timezone,quiet_start=:quiet_start,quiet_end=:quiet_end,"
            "inactivity_hours=:idle_hours,inactivity_turn_id=:idle_turn,inactivity_due_at="
            "CASE WHEN inactivity_last_sent_turn_id=:idle_turn THEN NULL ELSE CAST(:idle_due AS DOUBLE PRECISION) END "
            "WHERE session_id=:session AND revision=:revision"
        )
        statements = [
            self._session_guard(session), (statement, params, 1),
            ("UPDATE character_chat_proactive_jobs SET status='cancelled',topic='' "
             "WHERE session_id=:session AND status='pending'", params, None),
        ]
        if not settings["enabled"]:
            statements.extend([
                ("DELETE FROM character_chat_push_attempts WHERE session_id=:session", params, None),
                ("DELETE FROM character_chat_push_subscriptions WHERE session_id=:session", params, None),
            ])
        self.db.revision_transaction(statements)
        return self._public_policy(session_id)

    def schedule(self, owner, session_id, revision, request_id, kind, topic, due_at, *, source_turn_id=None, now=None):
        now = time() if now is None else now
        session = self.chats.get(owner, session_id)
        policy = self._policy(session_id)
        if not policy["enabled"] or revision != policy["revision"]:
            raise RevisionConflictError()
        params = {"session": session_id, "id": request_id, "revision": revision,
                  "kind": kind, "topic": topic, "due": due_at, "source": source_turn_id}
        query = ("SELECT id,kind,topic,due_at,status,policy_revision,source_turn_id FROM character_chat_proactive_jobs "
                 "WHERE session_id=:session AND id=:id")

        def existing():
            rows = self.db.execute_sql(query, params)
            if rows:
                job = rows[0]
                if (job["kind"], job["topic"], job["due_at"], job["policy_revision"], job["source_turn_id"]) != (
                    kind, topic, due_at, revision, source_turn_id,
                ):
                    raise RevisionConflictError()
                return job
            return None

        if saved := existing():
            return saved
        if not now <= due_at <= now + 366 * 86400 or kind not in _TEXT or not topic.strip():
            raise ValueError("请填写有效话题和未来一年内的提醒时间")
        if source_turn_id is not None and not any(
            (candidate["source_turn_id"], candidate["kind"], candidate["topic"]) == (source_turn_id, kind, topic)
            for candidate in self._candidates(session_id, policy["timezone"], now)
        ):
            raise ValueError("候选话题已过期，请重新读取当前对话")
        self.db.revision_transaction([
            self._session_guard(session),
            ("UPDATE character_chat_proactive_policies SET revision=revision "
             "WHERE session_id=:session AND revision=:revision AND enabled=1", params, 1),
            ("INSERT INTO character_chat_proactive_jobs "
             "(session_id,id,policy_revision,kind,topic,due_at,next_attempt_at,status,source_turn_id) "
             "SELECT :session,:id,:revision,:kind,:topic,:due,:due,'pending',CAST(:source AS TEXT) "
             "WHERE (SELECT COUNT(*) FROM character_chat_proactive_jobs WHERE session_id=:session AND status='pending')<20 "
             "AND NOT EXISTS (SELECT 1 FROM character_chat_proactive_jobs WHERE session_id=:session "
             "AND kind=:kind AND status IN ('pending','sent') "
             "AND topic=:topic AND (due_at=:due OR source_turn_id=CAST(:source AS TEXT))) "
             "ON CONFLICT (session_id,id) DO NOTHING", params, None),
        ])
        if saved := existing():
            return saved
        raise ValueError("该话题和时间已安排，或待发送消息已达 20 条")

    def change_schedule(self, owner, session_id, job_id, expected_due_at, due_at, *, now=None):
        session = self.chats.get(owner, session_id)
        params = {"session": session_id, "id": job_id, "expected_due": expected_due_at, "due": due_at}
        rows = self.db.execute_sql(
            "SELECT status,due_at,policy_revision FROM character_chat_proactive_jobs WHERE session_id=:session AND id=:id",
            params,
        )
        if not rows:
            raise CharacterNotFoundError("安排不存在")
        job = rows[0]
        if due_at is None and job["status"] == "cancelled":
            return {"status": "cancelled"}
        if job["status"] != "pending":
            raise RevisionConflictError()
        if due_at == job["due_at"]:
            return {"status": "pending"}
        if job["due_at"] != expected_due_at:
            raise RevisionConflictError()

        params["revision"] = job["policy_revision"]
        match = " WHERE session_id=:session AND id=:id AND status='pending' AND due_at=:expected_due"
        if due_at is None:
            statement = "UPDATE character_chat_proactive_jobs SET status='cancelled',topic=''" + match
        else:
            now = time() if now is None else now
            if not now <= due_at <= now + 366 * 86400:
                raise ValueError("请填写未来一年内的提醒时间")
            statement = (
                "UPDATE character_chat_proactive_jobs SET due_at=:due,next_attempt_at=:due" + match
                + " AND NOT EXISTS (SELECT 1 FROM character_chat_proactive_jobs other "
                "WHERE other.session_id=:session AND other.id<>:id AND other.status IN ('pending','sent') "
                "AND other.kind=character_chat_proactive_jobs.kind AND other.topic=character_chat_proactive_jobs.topic "
                "AND other.due_at=:due)"
            )
        self.db.revision_transaction([
            self._session_guard(session),
            ("UPDATE character_chat_proactive_policies SET revision=revision "
             "WHERE session_id=:session AND revision=:revision AND enabled=1", params, 1),
            (statement, params, 1),
        ])
        return {"status": "cancelled" if due_at is None else "pending"}

    def has_due(self, now):
        return bool(self.db.execute_sql(
            "SELECT id FROM character_chat_proactive_jobs "
            "WHERE status='pending' AND next_attempt_at<=:now "
            "UNION ALL SELECT p.session_id FROM character_chat_proactive_policies p "
            + _INACTIVITY_READY + " LIMIT 1", {"now": now},
        ))

    def _due_jobs(self, now, *, fiction=None):
        kind = {True: " AND j.kind='fiction'", False: " AND j.kind<>'fiction'", None: ""}
        return self.db.execute_sql(
            "SELECT j.session_id,j.id,j.kind,j.topic,j.policy_revision,j.due_at,j.next_attempt_at,s.owner_user_id "
            "FROM character_chat_proactive_jobs j JOIN character_chat_sessions s ON s.id=j.session_id "
            "WHERE j.status='pending' AND j.next_attempt_at<=:now" + kind[fiction]
            + " ORDER BY j.next_attempt_at,j.id LIMIT 100", {"now": now},
        )

    def deliver_due(self, *, now=None, include_fiction=True):
        """Run a restart-safe worker sweep. No private text leaves the database worker."""
        def clock():
            return time() if now is None else now

        ready_at = clock()
        jobs = self._due_jobs(ready_at, fiction=None if include_fiction else False)
        sent = 0
        for job in jobs:
            try:
                sent += self._deliver(job, clock())
            except (RevisionConflictError, CharacterNotFoundError):
                # A newer user action or another worker won the race.
                continue
        inactivity = self.db.execute_sql(
            "SELECT p.session_id,s.owner_user_id FROM character_chat_proactive_policies p "
            "JOIN character_chat_sessions s ON s.id=p.session_id "
            + _INACTIVITY_READY + " ORDER BY p.inactivity_due_at,p.session_id LIMIT 100",
            {"now": ready_at},
        )
        for entry in inactivity:
            try:
                sent += self._deliver_inactivity(entry, clock())
            except (RevisionConflictError, CharacterNotFoundError):
                continue
        return sent

    def _ready_job(self, job, now, *, session=None):
        session_id = job["session_id"]
        if session is None:
            session = self.chats.get(job["owner_user_id"], session_id)
        policy = self._policy(session_id)
        if not policy["enabled"] or policy["revision"] != job["policy_revision"]:
            return None
        next_allowed = next_allowed_at(policy, now)
        params = {"session": session_id, "id": job["id"], "revision": policy["revision"], "now": now,
                  "expected_attempt": job["next_attempt_at"], "expected_due": job["due_at"]}
        if next_allowed is not None:
            self.db.revision_transaction([
                ("UPDATE character_chat_proactive_jobs SET next_attempt_at=:next "
                 "WHERE session_id=:session AND id=:id AND policy_revision=:revision "
                 "AND status='pending' AND due_at=:expected_due AND next_attempt_at=:expected_attempt",
                 {**params, "next": next_allowed}, None),
            ])
            return None
        return session, policy, params

    def claim_fiction(self, *, now=None):
        now = time() if now is None else now
        for job in self._due_jobs(now, fiction=True):
            try:
                ready = self._ready_job(job, now)
                if ready is None:
                    continue
                session, policy, params = ready
                # The due-time column also fences the current render and makes an expired claim eligible again.
                lease = now + _FICTION_LEASE_SECONDS
                self.db.revision_transaction([
                    self._session_guard(session),
                    ("UPDATE character_chat_proactive_policies SET revision=revision "
                     "WHERE session_id=:session AND revision=:revision AND enabled=1 "
                     "AND (last_sent_at IS NULL OR last_sent_at<=:cutoff)",
                     {**params, "cutoff": now - policy["cadence_hours"] * 3600}, 1),
                    ("UPDATE character_chat_proactive_jobs SET next_attempt_at=:lease "
                     "WHERE session_id=:session AND id=:id AND policy_revision=:revision AND status='pending' "
                     "AND due_at=:expected_due AND next_attempt_at=:expected_attempt", {**params, "lease": lease}, 1),
                ])
                prior = self.fiction_history(job["owner_user_id"], job["session_id"], limit=3)["entries"]
                return {**job, "next_attempt_at": lease}, session, [item["content"] for item in reversed(prior)]
            except (RevisionConflictError, CharacterNotFoundError):
                continue
        return None

    def finish_fiction(self, claim, text, *, now=None):
        now = time() if now is None else now
        job, session, _ = claim
        if now >= job["next_attempt_at"]:
            return False
        content = f"{FICTION_LABEL}{text}" if text else None
        try:
            return self._deliver(job, now, session=session, content=content)
        except (RevisionConflictError, CharacterNotFoundError):
            return False

    def _deliver(self, job, now, *, session=None, content=None):
        ready = self._ready_job(job, now, session=session)
        if ready is None:
            return False
        session, policy, params = ready
        self.db.revision_transaction([
            self._session_guard(session),
            ("UPDATE character_chat_proactive_policies SET last_sent_at=:now "
             "WHERE session_id=:session AND revision=:revision AND enabled=1 "
             "AND (last_sent_at IS NULL OR last_sent_at<=:cutoff)",
             {**params, "cutoff": now - policy["cadence_hours"] * 3600}, 1),
            ("UPDATE character_chat_proactive_jobs SET status='sent',sent_at=:now,content=:content "
             "WHERE session_id=:session AND id=:id AND policy_revision=:revision "
             "AND status='pending' AND due_at=:expected_due AND next_attempt_at=:expected_attempt",
             {**params, "content": content or _TEXT[job["kind"]].format(topic=job["topic"])}, 1),
        ])
        return True

    def _deliver_inactivity(self, entry, now):
        session = self.chats.get(entry["owner_user_id"], entry["session_id"])
        policy = self._policy(session["id"])
        due, turn_id = policy["inactivity_due_at"], policy["inactivity_turn_id"]
        if not policy["enabled"] or due is None or due > now or not turn_id:
            return False
        params = {"session": session["id"], "revision": policy["revision"], "turn": turn_id, "due": due, "now": now,
                  "cutoff": now - policy["cadence_hours"] * 3600}
        guard = (" WHERE session_id=:session AND revision=:revision AND enabled=1 "
                 "AND inactivity_turn_id=:turn AND inactivity_due_at=:due")
        next_allowed = next_allowed_at(policy, now)
        if next_allowed is not None:
            self.db.revision_transaction([
                ("UPDATE character_chat_proactive_policies SET inactivity_due_at=:next" + guard,
                 {**params, "next": next_allowed}, None),
            ])
            return False
        self.db.revision_transaction([
            self._session_guard(session),
            ("UPDATE character_chat_proactive_policies SET last_sent_at=:now,inactivity_due_at=NULL,"
             "inactivity_last_sent_turn_id=:turn" + guard + " AND (last_sent_at IS NULL OR last_sent_at<=:cutoff) "
             "AND NOT EXISTS (SELECT 1 FROM character_chat_generations WHERE session_id=:session "
             "AND status='running' AND lease_until>:now)", params, 1),
            ("INSERT INTO character_chat_proactive_jobs "
             "(session_id,id,policy_revision,kind,topic,due_at,next_attempt_at,status,sent_at,content) "
             "VALUES (:session,:id,:revision,'inactivity','',:due,:due,'sent',:now,:content)",
             {**params, "id": uuid4().hex, "content": _INACTIVITY_GREETING}, 1),
        ])
        return True
