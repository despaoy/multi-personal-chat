"""角色与长期记忆管理API（仅管理员）。

提供画像列表、关系查看/覆盖、记忆的查看/修改/删除能力。
所有端点都要求 admin 角色；所有读写都以完整隔离范围
（平台+适配器+发送者+会话类型+会话ID）为前提，防止跨用户越权。

仓储经 FastAPI 依赖注入从当前应用容器的数据库解析，
create_app(custom_container) 的多实例/测试注入不会串到全局数据库。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.dependencies import get_current_admin
from app.providers import get_character_memory_repository
from character.context_builder import build_user_scope
from character.models import MemoryItem, RelationshipState
from character.natural_relationship import CATEGORIES, NoteCommand, is_note, save_note
from character.profile_registry import get_default_profile_registry
from repositories.character_memory import DatabaseCharacterMemoryRepository

logger = logging.getLogger(__name__)
router = APIRouter()


class RelationshipUpdateRequest(BaseModel):
    """管理员手动覆盖关系状态（回退关系阶段、修正摘要等）。"""

    stage: str = Field(..., pattern="^(stranger|acquaintance|familiar|close)$")
    preferred_address: str = Field(default="", max_length=100)
    summary: str = Field(default="", max_length=500)


class MemoryUpdateRequest(BaseModel):
    """管理员修正单条记忆内容或重要度。"""

    content: str = Field(..., min_length=1, max_length=500)
    importance: float = Field(default=0.0, ge=0.0, le=1.0)
    resolved: bool | None = None


class RelationshipNoteRequest(BaseModel):
    category: str = Field(pattern="^(preference|boundary|shared_event|promise|repair|transient)$")
    content: str = Field(min_length=1, max_length=300)


@router.get("/api/characters/{character_id}/relationship-notes", dependencies=[Depends(get_current_admin)])
async def list_relationship_notes(
    character_id: str, platform: str, adapter: str, sender_id: str,
    conversation_type: str, conversation_id: str,
    repo: DatabaseCharacterMemoryRepository = Depends(get_character_memory_repository),
):
    scope = _build_scope(platform, adapter, sender_id, conversation_type, conversation_id)
    return {"success": True, "memories": await repo.list_relationship_notes(character_id, scope)}


@router.post("/api/characters/{character_id}/relationship-notes", dependencies=[Depends(get_current_admin)])
async def create_relationship_note(
    character_id: str, request: RelationshipNoteRequest,
    platform: str, adapter: str, sender_id: str, conversation_type: str, conversation_id: str,
    repo: DatabaseCharacterMemoryRepository = Depends(get_character_memory_repository),
):
    scope = _build_scope(platform, adapter, sender_id, conversation_type, conversation_id)
    if request.category not in CATEGORIES or not request.content.strip():
        raise HTTPException(status_code=400, detail="备忘录类别或内容无效")
    record = await save_note(repo, character_id, scope, NoteCommand(request.category, request.content.strip()))
    return {"success": True, "memory": record}


def _build_scope(
    platform: str,
    adapter: str,
    sender_id: str,
    conversation_type: str,
    conversation_id: str,
):
    """规范化并校验隔离范围，非法时返回 400。"""
    try:
        return build_user_scope(
            platform=platform,
            adapter=adapter,
            sender_id=sender_id,
            conversation_id=conversation_id,
            conversation_type=conversation_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


@router.get("/api/characters", dependencies=[Depends(get_current_admin)])
async def list_characters():
    """列出已注册的人物画像（不含完整画像内容）。"""
    registry = get_default_profile_registry()
    try:
        profiles = registry.list_profiles()
    except Exception:
        logger.error("加载人物画像列表失败", exc_info=True)
        raise HTTPException(status_code=500, detail="加载人物画像失败") from None
    return {"success": True, "characters": list(profiles)}


@router.get(
    "/api/characters/{character_id}/relationship",
    dependencies=[Depends(get_current_admin)],
)
async def get_character_relationship(
    character_id: str,
    platform: str,
    adapter: str,
    sender_id: str,
    conversation_type: str,
    conversation_id: str,
    repo: DatabaseCharacterMemoryRepository = Depends(get_character_memory_repository),
):
    """查询指定角色+用户范围的关系状态。"""
    user_scope = _build_scope(
        platform, adapter, sender_id, conversation_type, conversation_id
    )
    try:
        record = await repo.get_relationship_record(character_id, user_scope)
    except Exception:
        logger.error("查询角色关系失败 character=%s", character_id, exc_info=True)
        raise HTTPException(status_code=500, detail="查询关系失败") from None
    if record is None:
        return {"success": True, "relationship": None}
    return {"success": True, "relationship": record}


@router.put(
    "/api/characters/{character_id}/relationship",
    dependencies=[Depends(get_current_admin)],
)
async def update_character_relationship(
    character_id: str,
    request: RelationshipUpdateRequest,
    platform: str,
    adapter: str,
    sender_id: str,
    conversation_type: str,
    conversation_id: str,
    repo: DatabaseCharacterMemoryRepository = Depends(get_character_memory_repository),
):
    """管理员手动覆盖关系状态（可回退阶段、修正称呼与摘要）。"""
    user_scope = _build_scope(
        platform, adapter, sender_id, conversation_type, conversation_id
    )
    try:
        record = await repo.upsert_relationship(
            character_id,
            user_scope,
            RelationshipState(
                stage=request.stage,
                preferred_address=request.preferred_address.strip(),
                summary=request.summary.strip(),
            ),
        )
    except Exception:
        logger.error("更新角色关系失败 character=%s", character_id, exc_info=True)
        raise HTTPException(status_code=500, detail="更新关系失败") from None
    return {"success": True, "relationship": record}


@router.get(
    "/api/characters/{character_id}/memories",
    dependencies=[Depends(get_current_admin)],
)
async def list_character_memories(
    character_id: str,
    platform: str,
    adapter: str,
    sender_id: str,
    conversation_type: str,
    conversation_id: str,
    limit: int = 100,
    include_inactive: bool = True,
    scope_level: str | None = None,
    repo: DatabaseCharacterMemoryRepository = Depends(get_character_memory_repository),
):
    """列出三层可见 claim；管理端默认包含版本历史。"""
    user_scope = _build_scope(
        platform, adapter, sender_id, conversation_type, conversation_id
    )
    if scope_level is not None and scope_level not in {
        "user_global", "user_character", "conversation"
    }:
        raise HTTPException(status_code=400, detail="scope_level 必须是 user_global、user_character 或 conversation")
    try:
        records = await repo.list_memory_records(
            character_id,
            user_scope,
            limit=max(1, min(limit, 500)),
            include_inactive=include_inactive,
            scope_levels=(scope_level,) if scope_level else None,
        )
    except Exception:
        logger.error("查询角色记忆失败 character=%s", character_id, exc_info=True)
        raise HTTPException(status_code=500, detail="查询记忆失败") from None
    return {"success": True, "memories": records}


@router.put(
    "/api/characters/{character_id}/memories/{memory_id}",
    dependencies=[Depends(get_current_admin)],
)
async def update_character_memory(
    character_id: str,
    memory_id: int,
    request: MemoryUpdateRequest,
    platform: str,
    adapter: str,
    sender_id: str,
    conversation_type: str,
    conversation_id: str,
    repo: DatabaseCharacterMemoryRepository = Depends(get_character_memory_repository),
):
    """修正记忆并追加 SUPERSEDE claim，保留旧版本与证据链。"""
    user_scope = _build_scope(
        platform, adapter, sender_id, conversation_type, conversation_id
    )
    try:
        target = await repo.get_memory_record(memory_id, character_id, user_scope)
        if target is None:
            raise HTTPException(status_code=404, detail="记忆不存在")
        if not request.content.strip():
            raise HTTPException(status_code=400, detail="内容不能为空")
        if is_note(target) and target.get("status") not in {"active", "current"}:
            raise HTTPException(status_code=409, detail="该备忘录已被更正，请刷新后编辑当前版本")
        if request.resolved is not None and not is_note(target):
            raise HTTPException(status_code=400, detail="只有关系备忘录支持结束状态")
        record = await repo.append_claim(
            character_id,
            user_scope,
            MemoryItem(
                memory_id=str(memory_id),
                memory_type=target.get("memory_type", "user_fact"),
                content=request.content.strip(),
                importance=request.importance,
            ),
            memory_key=str(target.get("memory_key") or f"memory_{memory_id}"),
            relation_type="SUPERSEDE",
            scope_level=str(target.get("scope_level") or "conversation"),
            parent_memory_id=memory_id,
            supersedes_memory_id=memory_id,
            evidence=tuple(target.get("evidence") or ()),
            confidence=float(target.get("confidence") or 1.0),
            attributed_to=str(target.get("attributed_to") or "user"),
            valid_from=target.get("valid_from"),
            valid_to=target.get("valid_to"),
            source_message_id=target.get("source_message_id"),
            source_message_ids=tuple(target.get("source_message_ids") or ()),
            metadata={
                **dict(target.get("metadata") or {}),
                "manual_correction": True,
                "corrected_memory_id": memory_id,
                **({"resolved": request.resolved} if request.resolved is not None else {}),
            },
        )
    except HTTPException:
        raise
    except Exception:
        logger.error("更新角色记忆失败 character=%s memory=%s", character_id, memory_id, exc_info=True)
        raise HTTPException(status_code=500, detail="更新记忆失败") from None
    return {"success": True, "memory": record}


@router.delete(
    "/api/characters/{character_id}/memories/{memory_id}",
    dependencies=[Depends(get_current_admin)],
)
async def delete_character_memory(
    character_id: str,
    memory_id: int,
    platform: str,
    adapter: str,
    sender_id: str,
    conversation_type: str,
    conversation_id: str,
    repo: DatabaseCharacterMemoryRepository = Depends(get_character_memory_repository),
):
    """隐私 ERASE：物理删除目标 claim 及其派生版本，不保留 tombstone。"""
    user_scope = _build_scope(
        platform, adapter, sender_id, conversation_type, conversation_id
    )
    try:
        target = await repo.get_memory_record(memory_id, character_id, user_scope)
        if target is not None and is_note(target):
            deleted = await repo.erase_memory(
                character_id, user_scope, memory_key=target["memory_key"],
                scope_level=target.get("scope_level") or "conversation",
            )
        else:
            deleted = await repo.erase_memory(
                character_id, user_scope, memory_id=memory_id
            )
    except Exception:
        logger.error("删除角色记忆失败 character=%s memory=%s", character_id, memory_id, exc_info=True)
        raise HTTPException(status_code=500, detail="删除记忆失败") from None
    if not deleted:
        raise HTTPException(status_code=404, detail="记忆不存在")
    return {"success": True, "deleted": deleted, "message": "记忆已物理删除"}


@router.delete(
    "/api/characters/{character_id}/memories",
    dependencies=[Depends(get_current_admin)],
)
async def clear_character_memories(
    character_id: str,
    platform: str,
    adapter: str,
    sender_id: str,
    conversation_type: str,
    conversation_id: str,
    repo: DatabaseCharacterMemoryRepository = Depends(get_character_memory_repository),
):
    """清空指定角色+用户范围的全部长期记忆。"""
    user_scope = _build_scope(
        platform, adapter, sender_id, conversation_type, conversation_id
    )
    try:
        deleted = await repo.clear_memories(character_id, user_scope)
    except Exception:
        logger.error("清空角色记忆失败 character=%s", character_id, exc_info=True)
        raise HTTPException(status_code=500, detail="清空记忆失败") from None
    return {"success": True, "deleted": deleted, "message": f"已删除 {deleted} 条记忆"}
