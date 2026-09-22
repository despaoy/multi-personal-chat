"""Branch generation reuses providers/RAG but owns isolated history and atomic commit."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import replace

from fastapi import HTTPException

from character.memory_extractor import next_relationship_stage
from character.memory_service import CharacterMemoryService
from character.profile_registry import get_default_profile_registry
from db.schemas import GenerateResponse
from repositories.character_memory import DatabaseCharacterMemoryRepository
from repositories.narrative import NarrativeRepository, Proposal, owner_id
from services.character_context import CharacterContextService, TurnInput


def parse_proposals(reply, active_ids):
    """Malformed optional metadata never turns into database instructions."""
    match = re.search(r"\n?```branch_proposals\s*\n(.*?)\n```\s*$", reply, re.S)
    if not match:
        return reply, [], []
    body = reply[: match.start()].strip()
    if not body:
        return "", [], ["proposal_without_reply"]
    try:
        if len(match[1]) > 8000:
            raise ValueError("oversized")
        raw = json.loads(match[1])
        if not isinstance(raw, list) or len(raw) > 3:
            raise ValueError("invalid proposal list")
        proposals = [Proposal.model_validate(item) for item in raw]
        if any(not set(item.source_assertion_ids).issubset(active_ids) for item in proposals):
            raise ValueError("unknown dependency")
        return body, proposals, []
    except (ValueError, TypeError):
        return body, [], ["invalid_branch_proposals"]


class _BranchRelationship:
    def __init__(self, state):
        self.state = state

    async def get_relationship_record(self, *_args):
        return self.state


class _BranchHistory:
    def __init__(self, rows):
        self.rows = rows

    async def list_recent_conversation_history(self, *_args, **_kwargs):
        history = []
        chars = 0
        for row in reversed(self.rows):
            pair = [{"role": "user", "content": row["message"]}, {"role": "assistant", "content": row["reply"]}]
            cost = len(row["message"]) + len(row["reply"])
            if chars + cost > 16000:
                break
            history[0:0] = pair
            chars += cost
        return history


async def prepare_branch(request, owner, repo):
    branch = await asyncio.to_thread(repo.get, owner, request.branchId, writable=True)
    active = await asyncio.to_thread(repo.assertions, owner, branch["id"], 101, 0, "active")
    if len(active) > 100:
        raise HTTPException(409, "有效事实超出第一版上下文容量")
    rows = await asyncio.to_thread(repo.history, owner, branch["id"])
    states = await asyncio.to_thread(repo.rows, "SELECT * FROM branch_states WHERE branch_id=:id", id=branch["id"])
    if not states:
        raise HTTPException(409, "分支关系状态缺失")
    # Re-read revision after snapshot loads; commit CAS catches changes after this point.
    latest = await asyncio.to_thread(repo.get, owner, branch["id"], writable=True)
    if latest["revision"] != branch["revision"]:
        raise HTTPException(409, "分支状态已变化，请重试")
    service = CharacterContextService(
        get_default_profile_registry(),
        _BranchRelationship(states[0]),
        _BranchHistory(rows),
        memory_service=CharacterMemoryService(DatabaseCharacterMemoryRepository(repo.db)),
    )
    # Derive identity server-side. No client-supplied history, sender or conversation is trusted.
    turn = TurnInput(
        request.message,
        "web",
        "narrative:" + branch["character_id"],
        owner,
        "canonical:" + branch["character_id"],
        "private",
    )
    prepared = await service.prepare_turn(turn, branch["character_id"])
    packet = json.dumps(
        {
            "branch_id": branch["id"],
            "facts": [
                {k: a[k] for k in ("id", "subject", "predicate", "object", "source_type", "assertion_kind", "status")}
                for a in active
            ],
        },
        ensure_ascii=False,
    )
    if len(packet) > 16000:
        raise HTTPException(409, "分支事实超过当前上下文预算，请创建较小分支")
    prepared = replace(prepared, compiled=replace(prepared.compiled, branch_context=packet))
    return branch, active, prepared


async def generate_branch_reply(request, current_user, database, *, generate):
    owner = owner_id(current_user)
    repo = NarrativeRepository(database)
    prior = await asyncio.to_thread(repo.prior, owner, request.branchId, request.traceId)
    if prior:
        if prior["message"] != request.message or (prior["loraName"] or "") != request.loraName:
            raise HTTPException(409, "同一请求ID不能用于不同内容")
        return GenerateResponse(
            reply=prior["reply"],
            model=prior["modelName"],
            costTime=prior["costTime"],
            branchId=request.branchId,
            answerMode="counterfactual",
            warnings=["replayed_committed_response"],
        )
    branch, active, prepared = await prepare_branch(request, owner, repo)
    internal = request.model_copy(
        update={
            "history": [],
            "senderId": owner,
            "userId": owner,
            "platform": "web",
            "adapter": "narrative",
            "sessionId": branch["id"],
            "conversationId": branch["id"],
            "sessionType": "private",
            "conversationType": "private",
        }
    )
    response = await generate(
        internal, current_user, persist_message=False, message_db=database, prepared_override=prepared
    )
    if not response.reply.strip() or "character_abstention_fallback" in (response.warnings or []):
        raise HTTPException(503, "本轮未成功生成，分支状态未提交")
    reply, proposals, warnings = parse_proposals(response.reply, {a["id"] for a in active})
    if not reply:
        raise HTTPException(503, "模型仅生成了提议元数据，分支状态未提交")
    response = response.model_copy(
        update={
            "reply": reply,
            "branchId": branch["id"],
            "branchRevision": branch["revision"] + 1,
            "answerMode": "counterfactual",
            "warnings": [*(response.warnings or []), *warnings],
            # These describe supplied evidence, not a semantic verification of generated claims.
            "evidenceSources": [
                {
                    "id": a["id"],
                    "type": "current_hypothesis" if a["assertion_kind"] == "premise" else "confirmed_branch_fact",
                    "scope": branch["id"],
                    "usage": "available_context",
                }
                for a in active
            ]
            + [
                {"type": "canonical_evidence", "citation": c, "usage": "available_context"}
                for c in (response.citations or [])
            ],
        }
    )
    stage = next_relationship_stage(prepared.relationship.stage, prepared.interaction_count + 1)
    try:
        await asyncio.to_thread(repo.commit_turn, branch, internal, response, proposals, stage)
    except HTTPException:
        prior = await asyncio.to_thread(repo.prior, owner, request.branchId, request.traceId)
        if prior and prior["message"] == request.message and (prior["loraName"] or "") == request.loraName:
            return GenerateResponse(
                reply=prior["reply"],
                model=prior["modelName"],
                costTime=prior["costTime"],
                branchId=request.branchId,
                answerMode="counterfactual",
            )
        raise
    return response
