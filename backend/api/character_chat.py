"""User-owned version-bound conversations, history edits and branches."""

import asyncio
from functools import partial
from math import ceil
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from api.character_library import account, call
from app.runtime import get_runtime_container
from character.chat_relationship import RelationshipDimensions, RelationshipStage
from db.errors import RevisionConflictError
from db.schemas import MessageText
from infra.environment import read_bool
from repositories.character_chat import CharacterChatRepository
from repositories.character_chat_enrichment import CharacterChatEnrichmentRepository
from repositories.character_chat_generation import (
    CharacterChatGenerationRepository,
    EnrichmentInProgressError,
    GenerationInProgressError,
)
from repositories.character_chat_proactive import CharacterChatProactiveRepository
from repositories.character_chat_push import CharacterChatPushRepository
from repositories.character_library import CharacterNotFoundError
from services.character_chat import CharacterChatService, complete_with_provider
from services.character_chat_events import generation_events
from services.character_chat_memory import enrich_state
from services.character_chat_policy import LocalOnlyPolicyError
from services.character_chat_relationship import enrich_relationship
from services.character_chat_vectors import select_memories
from services.character_source_retrieval import retrieve_configured_sources
from services.character_web_push import WebPushConfig

router = APIRouter(prefix="/api/character-chats", tags=["character-chats"])


class CreateSession(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    character_id: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=100)
    source_session_id: str | None = Field(default=None, min_length=1, max_length=100)
    source_revision: int | None = Field(default=None, ge=1)
    source_turn_id: str | None = Field(default=None, min_length=1, max_length=100)
    hypothesis: str = Field(default="", max_length=8000)
    accept_settings_change: bool = False
    expected_character_revision: int | None = Field(default=None, ge=1)
    excluded_memory_ids: list[str] = Field(default_factory=list, max_length=1000)
    reset_relationship: bool = False
    inherit_history: bool = True
    shared_memory_ids: list[str] = Field(default_factory=list, max_length=50)
    network_policy: Literal["cloud_allowed", "local_only"] | None = None
    source_corpus_id: str | None = Field(default=None, min_length=1, max_length=100)


class SendMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: int = Field(ge=1)
    request_id: str = Field(min_length=1, max_length=128)
    message: MessageText = Field(min_length=1, max_length=8000)
    replace_turn_id: str | None = Field(default=None, min_length=1, max_length=100)


class Revision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: int = Field(ge=1)


class MemoryContent(Revision):
    content: str = Field(min_length=1, max_length=2000)


class DeleteSelection(Revision):
    id: str = Field(min_length=1, max_length=100)


class DeleteSessions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sessions: list[DeleteSelection] = Field(min_length=1, max_length=100)


class RelationshipContent(Revision):
    stage: RelationshipStage
    summary: str = Field(default="", max_length=2000)
    dimensions: RelationshipDimensions | None = None


class RelationshipDecision(Revision):
    proposal_id: str = Field(min_length=1, max_length=128)
    accept: bool


class ProactivePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: int = Field(ge=0)
    enabled: bool
    cadence_hours: int = Field(ge=24, le=168)
    timezone: str = Field(min_length=1, max_length=80)
    quiet_start: int | None = Field(ge=0, le=1439)
    quiet_end: int | None = Field(ge=0, le=1439)
    inactivity_hours: int | None = Field(default=None, ge=72, le=720)

    @model_validator(mode="after")
    def validate_schedule(self):
        try:
            ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError("请选择有效时区") from exc
        if (self.quiet_start is None) != (self.quiet_end is None):
            raise ValueError("免打扰起止时间需同时填写")
        if self.quiet_start is not None and self.quiet_start == self.quiet_end:
            raise ValueError("免打扰起止时间需不同")
        return self


class ProactiveJob(Revision):
    request_id: str = Field(min_length=1, max_length=100)
    kind: Literal["follow_up", "event", "fiction"]
    topic: str = Field(min_length=1, max_length=300)
    due_at: float = Field(gt=0, le=1e10)
    source_turn_id: str | None = Field(default=None, min_length=1, max_length=100)


class ProactiveScheduleChange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_due_at: float = Field(gt=0, le=1e10)
    due_at: float | None = Field(gt=0, le=1e10)


class BrowserSubscription(BaseModel):
    model_config = ConfigDict(extra="forbid")
    endpoint: str = Field(min_length=1, max_length=2048)
    p256dh: str = Field(min_length=1, max_length=100)
    auth: str = Field(min_length=1, max_length=32)


def repository(request: Request):
    return CharacterChatRepository(get_runtime_container(request.app).db)


def service(request: Request):
    container = get_runtime_container(request.app)
    return CharacterChatService(
        container.db,
        partial(complete_with_provider, database=container.db, http_client_pool=container.http_client_pool),
        partial(enrich_state, database=container.db),
        partial(enrich_relationship, database=container.db),
        stream_events=True,
        background=read_bool(container.startup_env, "CHARACTER_CHAT_BACKGROUND_ENABLED", False),
        retrieve=partial(select_memories, database=container.db)
        if read_bool(container.startup_env, "CHARACTER_CHAT_VECTORS_ENABLED", False) else None,
        sources=partial(retrieve_configured_sources, database=container.db)
        if read_bool(container.startup_env, 'CHARACTER_CHAT_SOURCES_ENABLED', False) else None,
        proactive_history=CharacterChatProactiveRepository(container.db).recent_sent,
    )


@router.get("")
def list_sessions(
    character_id: str | None = None,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    owner=Depends(account),
    repo=Depends(repository),
):
    return {"sessions": repo.list(owner, character_id, limit, offset)}


@router.post("", status_code=201)
def create_session(body: CreateSession, owner=Depends(account), repo=Depends(repository)):
    try:
        return call(lambda: repo.create(owner, **body.model_dump()))
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/shared-memories")
def list_shared_memories(
    limit: int = Query(100, ge=1, le=100),
    offset: int = Query(0, ge=0),
    owner=Depends(account),
    repo=Depends(repository),
):
    return {"memories": repo.shared_memories(owner, limit, offset)}


@router.delete("/shared-memories/{shared_id}")
def unshare_memory(shared_id: str, owner=Depends(account), repo=Depends(repository)):
    call(repo.unshare_memory, owner, shared_id)
    return {"removed": True}


@router.post("/{session_id}/memories/{memory_id}/share")
def share_memory(session_id: str, memory_id: str, body: Revision, owner=Depends(account), repo=Depends(repository)):
    try:
        return call(repo.share_memory, owner, session_id, memory_id, body.revision)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/proactive-inbox")
def proactive_inbox(owner=Depends(account), repo=Depends(repository)):
    return {"messages": CharacterChatProactiveRepository(repo.database).inbox(owner)}


@router.get("/{session_id}")
def get_session(session_id: str, owner=Depends(account), repo=Depends(repository)):
    session, turns = call(repo.snapshot, owner, session_id)
    generation = call(CharacterChatGenerationRepository(repo.database).pending, owner, session_id)
    enrichment_pending = CharacterChatEnrichmentRepository(repo.database).is_pending(session)
    enrichment_retryable = CharacterChatEnrichmentRepository(repo.database).retryable(session)
    if repo.get(owner, session_id)["revision"] != session["revision"]:
        raise HTTPException(409, "会话已变化，请刷新后重试")
    return {"session": session, "turns": turns, "generation": generation,
            "enrichment_pending": enrichment_pending, "enrichment_retryable": enrichment_retryable}


@router.post("/{session_id}/enrichment/retry")
def retry_enrichment(session_id: str, body: Revision, owner=Depends(account), repo=Depends(repository)):
    return call(CharacterChatEnrichmentRepository(repo.database).retry, owner, session_id, body.revision)


@router.post("/delete-preview")
def preview_deletion(body: DeleteSessions, owner=Depends(account), repo=Depends(repository)):
    try:
        return call(repo.preview_deletion, owner, [item.model_dump() for item in body.sessions])
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/delete-selected")
def delete_selected(body: DeleteSessions, owner=Depends(account), repo=Depends(repository)):
    try:
        return call(repo.delete_sessions, owner, [item.model_dump() for item in body.sessions])
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/{session_id}/continuation-preview")
def continuation_preview(session_id: str, owner=Depends(account), repo=Depends(repository)):
    return call(repo.continuation_preview, owner, session_id)


@router.get("/{session_id}/proactive")
def proactive_state(session_id: str, owner=Depends(account), repo=Depends(repository)):
    return call(CharacterChatProactiveRepository(repo.database).view, owner, session_id)


@router.get("/{session_id}/proactive/candidates")
def proactive_candidates(session_id: str, owner=Depends(account), repo=Depends(repository)):
    return {"candidates": call(CharacterChatProactiveRepository(repo.database).candidates, owner, session_id)}


@router.get("/{session_id}/proactive/fiction")
def proactive_fiction(session_id: str, before: Annotated[str | None, Query(min_length=1, max_length=100)] = None,
                      limit: Annotated[int, Query(ge=1, le=50)] = 20, owner=Depends(account), repo=Depends(repository)):
    return call(CharacterChatProactiveRepository(repo.database).fiction_history, owner, session_id, before, limit)


def push_configuration():
    try:
        return WebPushConfig.from_env()
    except ValueError as exc:
        raise HTTPException(503, "浏览器通知配置不可用") from exc


@router.get("/{session_id}/browser-push")
def push_state(session_id: str, owner=Depends(account), repo=Depends(repository)):
    registered = call(CharacterChatPushRepository(repo.database).registered, owner, session_id)
    config = push_configuration()
    return {"public_key": config.public_key if config else None, "subscription_ids": registered}


@router.post("/{session_id}/browser-push", status_code=201)
def subscribe_browser(session_id: str, body: BrowserSubscription, owner=Depends(account), repo=Depends(repository)):
    if push_configuration() is None:
        raise HTTPException(503, "服务端未开启浏览器通知")
    try:
        return call(lambda: CharacterChatPushRepository(repo.database).register(owner, session_id, **body.model_dump()))
    except ValueError as exc:
        raise HTTPException(422, "浏览器订阅无效或已达到设备上限") from exc


@router.delete("/{session_id}/browser-push/{subscription_id}")
def unsubscribe_browser(session_id: str, subscription_id: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")],
                        owner=Depends(account), repo=Depends(repository)):
    call(CharacterChatPushRepository(repo.database).unregister, owner, session_id, subscription_id)
    return {"removed": True}


@router.put("/{session_id}/proactive")
def save_proactive_policy(session_id: str, body: ProactivePolicy, owner=Depends(account), repo=Depends(repository)):
    try:
        return call(lambda: CharacterChatProactiveRepository(repo.database).set_policy(owner, session_id, **body.model_dump()))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/{session_id}/proactive/jobs", status_code=201)
def schedule_proactive_job(session_id: str, body: ProactiveJob, owner=Depends(account), repo=Depends(repository)):
    try:
        return call(lambda: CharacterChatProactiveRepository(repo.database).schedule(owner, session_id, **body.model_dump()))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.patch("/{session_id}/proactive/jobs/{job_id}")
def change_proactive_schedule(session_id: str, job_id: Annotated[str, Field(min_length=1, max_length=100)],
                              body: ProactiveScheduleChange, owner=Depends(account), repo=Depends(repository)):
    try:
        return call(CharacterChatProactiveRepository(repo.database).change_schedule,
                    owner, session_id, job_id, body.expected_due_at, body.due_at)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/{session_id}/messages")
async def generate_message(
    session_id: str, body: SendMessage, request: Request, owner=Depends(account), chat=Depends(service)
):
    container = get_runtime_container(request.app)
    runtime = container.inference_runtime
    if runtime is None or container.character_chat_tasks is None:
        raise HTTPException(503, "会话调度器未初始化")
    from infra.concurrency_control import InferenceQueueFull, RateLimitExceeded

    try:
        key = f"character-chat:{owner}:{session_id}"
        # Confirm ownership before queueing work or invoking a model.
        await asyncio.to_thread(chat.repository.get, owner, session_id)
        await runtime.check_rate_limits("web", key, owner)
        return await container.character_chat_tasks.run(
            (owner, session_id, body.request_id),
            body.model_dump(),
            lambda: runtime.submit(
                lambda: chat.generate(owner, session_id, **body.model_dump()),
                session_id=key,
                priority=runtime.priority_for("web", "private"),
            ),
        )
    except CharacterNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except RevisionConflictError as exc:
        raise HTTPException(409, "会话已变化，请刷新后重试") from exc
    except GenerationInProgressError as exc:
        raise HTTPException(409, "这条消息仍在生成，请稍后查询结果") from exc
    except EnrichmentInProgressError as exc:
        raise HTTPException(409, "上一轮记忆和关系仍在整理，请稍后继续发送") from exc
    except LocalOnlyPolicyError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RateLimitExceeded as exc:
        raise HTTPException(
            429, "请求过于频繁，请稍后重试", headers={"Retry-After": str(max(1, ceil(exc.retry_after)))},
        ) from exc
    except (InferenceQueueFull, TimeoutError) as exc:
        raise HTTPException(503, "生成未完成，请刷新会话核对后重试") from exc


@router.get("/{session_id}/generations/{request_id}")
def generation_status(session_id: str, request_id: str, owner=Depends(account), repo=Depends(repository)):
    status = call(CharacterChatGenerationRepository(repo.database).get, owner, session_id, request_id)
    if status is None:
        raise HTTPException(404, "生成请求不存在")
    return status


@router.get("/{session_id}/generations/{request_id}/events")
async def generation_event_stream(session_id: str, request_id: str, owner=Depends(account), repo=Depends(repository)):
    try:
        await asyncio.to_thread(repo.get, owner, session_id)
    except CharacterNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return StreamingResponse(
        generation_events(CharacterChatGenerationRepository(repo.database), owner, session_id, request_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@router.post("/{session_id}/messages/{turn_id}/retract")
def retract_message(session_id: str, turn_id: str, body: Revision, owner=Depends(account), repo=Depends(repository)):
    return call(repo.retract, owner, session_id, turn_id, body.revision)


@router.put("/{session_id}/relationship")
def update_relationship(session_id: str, body: RelationshipContent, owner=Depends(account), repo=Depends(repository)):
    return call(repo.update_relationship, owner, session_id, body.revision, body.stage, body.summary, body.dimensions)


@router.post("/{session_id}/relationship-proposal")
def resolve_relationship_proposal(
    session_id: str, body: RelationshipDecision, owner=Depends(account), repo=Depends(repository)
):
    return call(repo.resolve_relationship_proposal, owner, session_id, body.revision, body.proposal_id, body.accept)


@router.post("/{session_id}/memories", status_code=201)
def add_memory(session_id: str, body: MemoryContent, owner=Depends(account), repo=Depends(repository)):
    return memory_change(repo, owner, session_id, body.revision, content=body.content)


@router.put("/{session_id}/memories/{memory_id}")
def correct_memory(
    session_id: str, memory_id: str, body: MemoryContent, owner=Depends(account), repo=Depends(repository)
):
    return memory_change(repo, owner, session_id, body.revision, memory_id=memory_id, content=body.content)


@router.delete("/{session_id}/memories/{memory_id}")
def delete_memory(session_id: str, memory_id: str, body: Revision, owner=Depends(account), repo=Depends(repository)):
    return memory_change(repo, owner, session_id, body.revision, memory_id=memory_id)


def memory_change(repo, owner, session_id, revision, **kwargs):
    try:
        return call(lambda: repo.manage_memory(owner, session_id, revision, **kwargs))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
