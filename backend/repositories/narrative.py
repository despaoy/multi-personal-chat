"""Owner-scoped narrative storage; every mutation commits with a revision CAS."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field


class NarrativeConflict(Exception):
    """A transaction lost its revision or assertion precondition."""


class Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    subject: str = Field(min_length=1, max_length=120)
    predicate: str = Field(min_length=1, max_length=120)
    object: str = Field(min_length=1, max_length=500)
    assertion_kind: Literal["event", "relation", "derived_claim"] = "derived_claim"
    source_assertion_ids: list[str] = Field(default_factory=list, max_length=10)


def now():
    return datetime.now(timezone.utc).isoformat()


def owner_id(user):
    value = (user or {}).get("user_id")
    if value is None or (user or {}).get("auth_type") in {"api_key", "managed_api_key"}:
        raise HTTPException(403, "分支功能需要登录用户账号")
    return str(value)


class NarrativeRepository:
    def __init__(self, database):
        self.db = database

    def rows(self, sql, **params):
        return self.db.execute_sql(sql, params)

    def get(self, owner, branch_id, *, writable=False):
        rows = self.rows(
            "SELECT * FROM narrative_branches WHERE id=:id AND owner_user_id=:owner", id=branch_id, owner=owner
        )
        if not rows:
            raise HTTPException(404, "分支不存在")
        branch = rows[0]
        if writable and branch["status"] != "active":
            raise HTTPException(409, "分支已归档")
        return branch

    def list(self, owner, limit=50, offset=0):
        return self.rows(
            "SELECT * FROM narrative_branches WHERE owner_user_id=:owner "
            "ORDER BY updated_at DESC, id LIMIT :limit OFFSET :offset",
            owner=owner,
            limit=limit,
            offset=offset,
        )

    def transaction(self, statements):
        try:
            self.db.narrative_transaction(statements)
        except NarrativeConflict as exc:
            raise HTTPException(409, "分支状态已变化，请刷新后重试") from exc

    def create(self, owner, character_id, title, hypothesis):
        branch_id, timestamp = uuid.uuid4().hex, now()
        params = dict(
            id=branch_id, owner=owner, character=character_id, title=title, hypothesis=hypothesis, now=timestamp
        )
        self.transaction(
            [
                (
                    "INSERT INTO narrative_branches (id,owner_user_id,character_id,title,initial_hypothesis,"
                    "status,revision,created_at,updated_at) VALUES (:id,:owner,:character,:title,:hypothesis,"
                    "'active',1,:now,:now)",
                    params,
                    1,
                ),
                ("INSERT INTO branch_states (branch_id,created_at,updated_at) VALUES (:id,:now,:now)", params, 1),
                (
                    "INSERT INTO branch_assertions (id,branch_id,object,source_type,assertion_kind,status,created_at,updated_at) "
                    "VALUES (:assertion,:id,:hypothesis,'user_hypothesis','premise','active',:now,:now)",
                    {**params, "assertion": uuid.uuid4().hex},
                    1,
                ),
            ]
        )
        return self.get(owner, branch_id)

    def cas(self, branch, *, archive=False):
        return (
            "UPDATE narrative_branches SET revision=revision+1,updated_at=:now,status=:status "
            "WHERE id=:id AND owner_user_id=:owner AND revision=:revision AND status='active'",
            dict(
                id=branch["id"],
                owner=branch["owner_user_id"],
                revision=branch["revision"],
                now=now(),
                status="archived" if archive else "active",
            ),
            1,
        )

    def assertions(self, owner, branch_id, limit=50, offset=0, status=None):
        self.get(owner, branch_id)
        return self.rows(
            "SELECT * FROM branch_assertions WHERE branch_id=:id "
            + ("AND status=:status " if status else "")
            + "ORDER BY created_at, id LIMIT :limit OFFSET :offset",
            id=branch_id,
            status=status,
            limit=limit,
            offset=offset,
        )

    def archive(self, owner, branch_id, revision):
        branch = self.get(owner, branch_id, writable=True)
        branch["revision"] = revision
        self.transaction([self.cas(branch, archive=True)])
        return self.get(owner, branch_id)

    def decide(self, owner, branch_id, assertion_id, revision, *, confirm, replace_ids=()):
        branch = self.get(owner, branch_id, writable=True)
        branch["revision"] = revision
        targets = self.rows(
            "SELECT * FROM branch_assertions WHERE id=:id AND branch_id=:branch", id=assertion_id, branch=branch_id
        )
        if not targets:
            raise HTTPException(404, "提议不存在")
        target = targets[0]
        statements = [self.cas(branch)]
        if confirm:
            sources = self.rows(
                'SELECT id FROM messages WHERE "sourceMessageId"=:source AND "branchId"=:branch AND "userId"=:owner',
                source=target["source_message_id"],
                branch=branch_id,
                owner=owner,
            )
            if not sources:
                raise HTTPException(409, "来源消息缺失，不能确认该提议")
            active = self.assertions(owner, branch_id, limit=101, status="active")
            if len(active) >= 100:
                raise HTTPException(409, "第一版每个分支最多100条有效事实")
            dependencies = set(json.loads(target["source_assertion_ids"]))
            if not dependencies.issubset({a["id"] for a in active}):
                raise HTTPException(409, "依赖事实已失效，请拒绝该提议并重新生成")

            def normalize(value):
                return " ".join(value.split()).casefold()

            conflicts = {
                a["id"]
                for a in active
                if a["assertion_kind"] != "premise"
                and normalize(a["subject"]) == normalize(target["subject"])
                and normalize(a["predicate"]) == normalize(target["predicate"])
                and normalize(a["object"]) != normalize(target["object"])
            }
            if conflicts != set(replace_ids):
                raise HTTPException(
                    409, {"message": "直接冲突需显式选择替代；语义冲突须人工核对", "conflictIds": sorted(conflicts)}
                )
            # Retract transitive dependants so replacing a premise cannot leave stale claims active.
            retract = set(conflicts)
            while True:
                expanded = retract | {a["id"] for a in active if set(json.loads(a["source_assertion_ids"])) & retract}
                if expanded == retract:
                    break
                retract = expanded
            if dependencies & retract:
                raise HTTPException(409, "新事实依赖即将撤回的事实，请重新提议")
            for old in retract:
                statements.append(
                    (
                        "UPDATE branch_assertions SET status='retracted',updated_at=:now "
                        "WHERE id=:id AND branch_id=:branch AND status='active'",
                        dict(id=old, branch=branch_id, now=now()),
                        1,
                    )
                )
        statements.append(
            (
                "UPDATE branch_assertions SET status=:status,updated_at=:now "
                "WHERE id=:id AND branch_id=:branch AND status='pending'",
                dict(status="active" if confirm else "rejected", now=now(), id=assertion_id, branch=branch_id),
                1,
            )
        )
        self.transaction(statements)
        return self.get(owner, branch_id)

    def history(self, owner, branch_id, limit=20, offset=0):
        self.get(owner, branch_id)
        return list(
            reversed(
                self.rows(
                    'SELECT * FROM messages WHERE "branchId"=:branch AND "userId"=:owner '
                    "ORDER BY id DESC LIMIT :limit OFFSET :offset",
                    branch=branch_id,
                    owner=owner,
                    limit=limit,
                    offset=offset,
                )
            )
        )

    def prior(self, owner, branch_id, trace_id):
        self.get(owner, branch_id)
        rows = self.rows(
            'SELECT * FROM messages WHERE "branchId"=:branch AND "userId"=:owner AND "traceId"=:trace LIMIT 1',
            branch=branch_id,
            owner=owner,
            trace=trace_id,
        )
        return rows[0] if rows else None

    def commit_turn(self, branch, request, response, proposals, stage):
        source = uuid.uuid4().hex
        timestamp = now()
        p = dict(
            branch=branch["id"],
            owner=branch["owner_user_id"],
            source=source,
            trace=request.traceId,
            message=request.message,
            reply=response.reply,
            model=response.model,
            lora=request.loraName,
            cost=response.costTime,
            now=timestamp,
        )
        statements = [
            self.cas(branch),
            (
                'INSERT INTO messages ("sessionType","sessionId",platform,adapter,"conversationId",'
                '"conversationType","senderId","userId","sourceMessageId","traceId",message,reply,'
                '"modelName","loraName","costTime","createdAt","branchId") '
                "VALUES ('private',:branch,'web','narrative',:branch,'private',:owner,:owner,:source,:trace,"
                ":message,:reply,:model,:lora,:cost,:now,:branch)",
                p,
                1,
            ),
            (
                "UPDATE branch_states SET interaction_count=interaction_count+1,relationship_stage=:stage,"
                "updated_at=:now WHERE branch_id=:branch",
                {**p, "stage": stage},
                1,
            ),
        ]
        for proposal in proposals:
            statements.append(
                (
                    "INSERT INTO branch_assertions (id,branch_id,subject,predicate,object,source_type,"
                    "assertion_kind,source_message_id,source_assertion_ids,status,created_at,updated_at) "
                    "VALUES (:id,:branch,:subject,:predicate,:object,'model_proposal',:kind,:source,"
                    ":dependencies,'pending',:now,:now)",
                    dict(
                        id=uuid.uuid4().hex,
                        branch=branch["id"],
                        source=source,
                        now=timestamp,
                        subject=proposal.subject,
                        predicate=proposal.predicate,
                        object=proposal.object,
                        kind=proposal.assertion_kind,
                        dependencies=json.dumps(proposal.source_assertion_ids),
                    ),
                    1,
                )
            )
        self.transaction(statements)
