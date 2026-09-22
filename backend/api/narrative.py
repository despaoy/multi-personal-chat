"""Authenticated Web narrative workspace."""

import asyncio
import os
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from app.dependencies import get_current_user
from app.runtime import get_runtime_container
from character.profile_registry import CharacterProfileNotFoundError, get_default_profile_registry
from db.schemas import GenerateResponse, MessageRequest
from repositories.narrative import NarrativeRepository, owner_id


def require_enabled():
    if os.getenv("NARRATIVE_BRANCHES_ENABLED", "false").lower() not in {"true", "1"}:
        raise HTTPException(404, "假想分支功能尚未启用")


router = APIRouter(prefix="/api/narrative-branches", dependencies=[Depends(require_enabled)])


class CreateBranch(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    character_id: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=100)
    initial_hypothesis: str = Field(min_length=1, max_length=2000)


class Revision(BaseModel):
    revision: int = Field(ge=1)


class Decision(Revision):
    replace_ids: list[str] = Field(default_factory=list, max_length=100)


class CanonicalMessage(BaseModel):
    character_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=8000)
    traceId: str = Field(default="", max_length=128)


def repository(request: Request):
    return NarrativeRepository(get_runtime_container(request.app).db)


def validate_character(character_id):
    try:
        get_default_profile_registry().get_profile(character_id)
    except CharacterProfileNotFoundError as exc:
        raise HTTPException(422, "未知角色") from exc


@router.get("/characters")
def characters(user=Depends(get_current_user)):
    owner_id(user)
    return {"characters": list(get_default_profile_registry().list_profiles())}


@router.post("/canonical/generate")
async def canonical_generate(body: CanonicalMessage, request: Request, user=Depends(get_current_user)):
    from api.generate import _generate_reply_impl, _is_high_risk_prompt, _security_policy_response
    from infra.concurrency_control import InferenceQueueFull, RateLimitExceeded, inference_runtime
    from infra.security_utils import strip_control_chars
    from services.character_context import TurnInput, build_character_context_service

    owner = owner_id(user)
    validate_character(body.character_id)
    message = strip_control_chars(body.message).strip()
    if not message:
        raise HTTPException(422, "消息不能为空")
    if _is_high_risk_prompt(message):
        return _security_policy_response()
    container = get_runtime_container(request.app)
    runtime = container.inference_runtime or inference_runtime
    session = "canonical:" + body.character_id

    async def job():
        service = build_character_context_service(container.db)
        adapter = "narrative:" + body.character_id
        if body.traceId:
            prior = await asyncio.to_thread(
                container.db.execute_sql,
                'SELECT message,reply,"modelName","costTime" FROM messages WHERE "branchId" IS NULL '
                'AND platform=\'web\' AND adapter=:adapter AND "senderId"=:owner AND "traceId"=:trace LIMIT 1',
                {"adapter": adapter, "owner": owner, "trace": body.traceId},
            )
            if prior:
                if prior[0]["message"] != message:
                    raise HTTPException(409, "同一请求ID不能用于不同内容")
                return GenerateResponse(
                    reply=prior[0]["reply"], model=prior[0]["modelName"], costTime=prior[0]["costTime"]
                )
        turn = TurnInput(message, "web", adapter, owner, session, "private")
        prepared = await service.prepare_turn(turn, body.character_id)
        req = MessageRequest(
            message=message,
            traceId=body.traceId or uuid.uuid4().hex,
            platform="web",
            adapter=adapter,
            senderId=owner,
            userId=owner,
            sessionId=session,
            conversationId=session,
            conversationType="private",
        )
        return await _generate_reply_impl(
            req, user, message_db=container.db, character_service=service, prepared_override=prepared
        )

    try:
        await runtime.check_rate_limits("web", session, owner)
        return await runtime.submit(
            job, session_id=f"canonical:{owner}:{body.character_id}", priority=runtime.priority_for("admin", "private")
        )
    except RateLimitExceeded as exc:
        raise HTTPException(429, "请求过于频繁", headers={"Retry-After": str(max(1, int(exc.retry_after)))}) from exc
    except (InferenceQueueFull, TimeoutError) as exc:
        raise HTTPException(503, "推理队列繁忙，请稍后重试") from exc


@router.get("/canonical/messages")
def canonical_messages(
    character_id: str = Query(max_length=100),
    limit: int = Query(20, ge=1, le=50),
    offset: int = Query(0, ge=0),
    repo=Depends(repository),
    user=Depends(get_current_user),
):
    owner = owner_id(user)
    validate_character(character_id)
    rows = repo.rows(
        "SELECT message,reply FROM messages WHERE \"branchId\" IS NULL AND platform='web' "
        'AND adapter=:adapter AND "senderId"=:owner ORDER BY id DESC LIMIT :limit OFFSET :offset',
        adapter="narrative:" + character_id,
        owner=owner,
        limit=limit,
        offset=offset,
    )
    return {"messages": list(reversed(rows))}


@router.post("")
def create(body: CreateBranch, repo=Depends(repository), user=Depends(get_current_user)):
    owner = owner_id(user)
    validate_character(body.character_id)
    return repo.create(owner, body.character_id, body.title, body.initial_hypothesis)


@router.get("")
def listing(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    repo=Depends(repository),
    user=Depends(get_current_user),
):
    return {"branches": repo.list(owner_id(user), limit, offset)}


@router.get("/{branch_id}")
def detail(branch_id: str, repo=Depends(repository), user=Depends(get_current_user)):
    return repo.get(owner_id(user), branch_id)


@router.post("/{branch_id}/archive")
def archive(branch_id: str, body: Revision, repo=Depends(repository), user=Depends(get_current_user)):
    return repo.archive(owner_id(user), branch_id, body.revision)


@router.get("/{branch_id}/assertions")
def assertions(
    branch_id: str,
    status: Literal["active", "pending", "rejected", "retracted"] | None = None,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    repo=Depends(repository),
    user=Depends(get_current_user),
):
    return {"assertions": repo.assertions(owner_id(user), branch_id, limit, offset, status)}


@router.get("/{branch_id}/messages")
def messages(
    branch_id: str,
    limit: int = Query(20, ge=1, le=50),
    offset: int = Query(0, ge=0),
    repo=Depends(repository),
    user=Depends(get_current_user),
):
    return {"messages": repo.history(owner_id(user), branch_id, limit, offset)}


@router.post("/{branch_id}/assertions/{assertion_id}/{action}")
def decide(
    branch_id: str,
    assertion_id: str,
    action: Literal["confirm", "reject"],
    body: Decision,
    repo=Depends(repository),
    user=Depends(get_current_user),
):
    return repo.decide(
        owner_id(user),
        branch_id,
        assertion_id,
        body.revision,
        confirm=action == "confirm",
        replace_ids=body.replace_ids,
    )
