"""Independent, consent-checked delivery of reminders for committed messages."""

import logging
from hashlib import sha256
from itertools import islice
from time import time
from uuid import uuid4

from db.errors import RevisionConflictError
from repositories.character_chat_proactive import CharacterChatProactiveRepository, in_quiet_hours
from repositories.character_library import CharacterNotFoundError
from services.character_web_push import validate_endpoint, validate_keys

logger = logging.getLogger(__name__)
_SEND_BATCH = 12
_SCAN_PAGE = 48
_READY = (
    " FROM character_chat_push_subscriptions d "
    "JOIN character_chat_sessions s ON s.id=d.session_id "
    "JOIN character_chat_proactive_policies p ON p.session_id=s.id "
    "JOIN character_chat_proactive_jobs j ON j.session_id=s.id "
    "LEFT JOIN character_chat_push_attempts a ON a.subscription_id=d.id AND a.message_id=j.id "
    "WHERE p.enabled=1 AND j.policy_revision=p.revision AND j.status='sent' AND j.sent_at>=d.created_at "
    "AND j.sent_at>=:recent AND (a.attempts IS NULL OR (a.terminal=0 AND a.attempts<3 AND a.lease_until<=:now))"
)


def _subscription_id(session, endpoint):
    return sha256(f"{session}\0{endpoint}".encode()).hexdigest()


class CharacterChatPushRepository:
    def __init__(self, database):
        self.db = database
        self.outreach = CharacterChatProactiveRepository(database)

    def register(self, owner, session_id, endpoint, p256dh, auth):
        validate_endpoint(endpoint)
        validate_keys(p256dh, auth)
        session = self.outreach.chats.get(owner, session_id)
        policy = self.outreach._policy(session_id)
        params = {"id": _subscription_id(session_id, endpoint), "session": session_id,
                  "endpoint": endpoint, "p256dh": p256dh, "auth": auth, "now": time(),
                  "revision": policy["revision"]}
        self.db.revision_transaction([
            self.outreach._session_guard(session),
            ("UPDATE character_chat_proactive_policies SET revision=revision "
             "WHERE session_id=:session AND revision=:revision AND enabled=1", params, 1),
            ("INSERT INTO character_chat_push_subscriptions (id,session_id,endpoint,p256dh,auth,created_at) "
             "SELECT :id,:session,:endpoint,:p256dh,:auth,:now "
             "WHERE (SELECT COUNT(*) FROM character_chat_push_subscriptions WHERE session_id=:session)<5 "
             "OR EXISTS (SELECT 1 FROM character_chat_push_subscriptions WHERE id=:id) "
             "ON CONFLICT (id) DO UPDATE SET p256dh=excluded.p256dh,auth=excluded.auth,created_at=excluded.created_at "
             "WHERE character_chat_push_subscriptions.p256dh<>excluded.p256dh "
             "OR character_chat_push_subscriptions.auth<>excluded.auth", params, None),
        ])
        if not self.db.execute_sql("SELECT id FROM character_chat_push_subscriptions WHERE id=:id", params):
            raise ValueError("每个会话最多启用五台浏览器")
        return {"id": params["id"]}

    def registered(self, owner, session_id):
        self.outreach.chats.get(owner, session_id)
        return [row["id"] for row in self.db.execute_sql(
            "SELECT id FROM character_chat_push_subscriptions WHERE session_id=:session", {"session": session_id},
        )]

    def unregister(self, owner, session_id, identity):
        session = self.outreach.chats.get(owner, session_id)
        params = {"id": identity, "session": session_id}
        self.db.revision_transaction([
            self.outreach._session_guard(session),
            ("DELETE FROM character_chat_push_attempts WHERE subscription_id=:id AND session_id=:session", params, None),
            ("DELETE FROM character_chat_push_subscriptions WHERE id=:id AND session_id=:session", params, None),
        ])

    @staticmethod
    def _window(now):
        return {"now": now, "recent": now - 300}

    def has_due(self, now):
        return bool(self.db.execute_sql("SELECT d.id" + _READY + " LIMIT 1", self._window(now)))

    def deliver(self, sender, *, now=None):
        def clock():
            return time() if now is None else now

        pending = (row for row in self._ready_rows(clock()) if not in_quiet_hours(row, clock()))
        for row in islice(pending, _SEND_BATCH):
            try:
                self._deliver_one(sender, row, clock())
            except (RevisionConflictError, CharacterNotFoundError):
                continue
            except Exception as exc:
                # Do not log subscription endpoints, keys or network error bodies.
                logger.warning("Browser reminder failed type=%s", type(exc).__name__)

    def _ready_rows(self, now):
        params = {**self._window(now), "page": _SCAN_PAGE}
        after = ""
        while True:
            rows = self.db.execute_sql(
                "SELECT d.id,d.session_id,d.endpoint,d.p256dh,d.auth,d.created_at,s.owner_user_id,"
                "p.revision,p.timezone,p.quiet_start,p.quiet_end,j.id AS message_id,j.sent_at" + _READY
                + after + " ORDER BY j.sent_at,d.id,j.id LIMIT :page", params,
            )
            yield from rows
            if len(rows) < _SCAN_PAGE:
                return
            last = rows[-1]
            params.update(sent=last["sent_at"], device=last["id"], message=last["message_id"])
            after = " AND (j.sent_at,d.id,j.id)>(:sent,:device,:message)"

    def _deliver_one(self, sender, row, now):
        if in_quiet_hours(row, now) or row["sent_at"] < now - 300:
            return
        session = self.outreach.chats.get(row["owner_user_id"], row["session_id"])
        params = {**row, "now": now, "lease": now + 60, "token": uuid4().hex}
        self.db.revision_transaction([
            self.outreach._session_guard(session),
            ("UPDATE character_chat_proactive_policies SET revision=revision "
             "WHERE session_id=:session_id AND revision=:revision AND enabled=1", params, 1),
            ("UPDATE character_chat_push_subscriptions SET id=id WHERE id=:id AND created_at=:created_at", params, 1),
            ("INSERT INTO character_chat_push_attempts (subscription_id,session_id,message_id,attempts,lease_until,token,terminal) "
             "VALUES (:id,:session_id,:message_id,1,:lease,:token,0) ON CONFLICT (subscription_id,message_id) DO UPDATE "
             "SET attempts=character_chat_push_attempts.attempts+1,lease_until=:lease,token=:token "
             "WHERE character_chat_push_attempts.terminal=0 AND character_chat_push_attempts.attempts<3 "
             "AND character_chat_push_attempts.lease_until<=:now", params, 1),
        ])
        current = self.db.execute_sql(
            "SELECT d.id FROM character_chat_push_subscriptions d JOIN character_chat_proactive_policies p "
            "ON p.session_id=d.session_id JOIN character_chat_push_attempts a ON a.subscription_id=d.id "
            "JOIN character_chat_proactive_jobs j ON j.session_id=d.session_id AND j.id=a.message_id "
            "WHERE d.id=:id AND d.created_at=:created_at AND p.enabled=1 AND p.revision=:revision "
            "AND a.message_id=:message_id AND a.token=:token AND j.status='sent'", params,
        )
        if not current:
            return
        tag = sha256(f"{row['session_id']}\0{row['message_id']}".encode()).hexdigest()[:32]
        status = sender.send(row["endpoint"], row["p256dh"], row["auth"], tag)
        if status in {404, 410}:
            self.db.revision_transaction([
                self.outreach._session_guard(session),
                ("UPDATE character_chat_push_subscriptions SET id=id "
                 "WHERE id=:id AND created_at=:created_at", params, 1),
                ("DELETE FROM character_chat_push_attempts WHERE subscription_id=:id", params, None),
                ("DELETE FROM character_chat_push_subscriptions WHERE id=:id", params, 1),
            ])
        else:
            retry = status in {408, 429} or status >= 500
            self.db.revision_transaction([
                ("UPDATE character_chat_push_attempts SET terminal=:terminal,lease_until=:next "
                 "WHERE subscription_id=:id AND message_id=:message_id AND token=:token",
                 {**params, "terminal": int(not retry), "next": now + 60}, None),
            ])
