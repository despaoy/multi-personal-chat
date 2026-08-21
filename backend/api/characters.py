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
    repo: DatabaseCharacterMemoryRepository = Depends(get_character_memory_repository),
):
    """列出指定角色+用户范围的长期记忆。"""
    user_scope = _build_scope(
        platform, adapter, sender_id, conversation_type, conversation_id
    )
    try:
        records = await repo.list_memory_records(
            character_id, user_scope, limit=max(1, min(limit, 200))
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
    """修正单条记忆的内容与重要度（key 不变，视为覆盖写）。"""
    user_scope = _build_scope(
        platform, adapter, sender_id, conversation_type, conversation_id
    )
    try:
        records = await repo.list_memory_records(
            character_id, user_scope, limit=200
        )
        target = next((r for r in records if int(r.get("id") or 0) == memory_id), None)
        if target is None:
            raise HTTPException(status_code=404, detail="记忆不存在")
        await repo.add_or_update_memory(
            character_id,
            user_scope,
            MemoryItem(
                memory_id=str(memory_id),
                memory_type=target.get("memory_type", "user_fact"),
                content=request.content.strip(),
                importance=request.importance,
            ),
            memory_key=str(target.get("memory_key") or f"memory_{memory_id}"),
            source_message_id=target.get("source_message_id"),
        )
    except HTTPException:
        raise
    except Exception:
        logger.error("更新角色记忆失败 character=%s memory=%s", character_id, memory_id, exc_info=True)
        raise HTTPException(status_code=500, detail="更新记忆失败") from None
    return {"success": True}


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
    """删除单条记忆（必须匹配完整隔离范围）。"""
    user_scope = _build_scope(
        platform, adapter, sender_id, conversation_type, conversation_id
    )
    try:
        deleted = await repo.delete_memory(memory_id, character_id, user_scope)
    except Exception:
        logger.error("删除角色记忆失败 character=%s memory=%s", character_id, memory_id, exc_info=True)
        raise HTTPException(status_code=500, detail="删除记忆失败") from None
    if not deleted:
        raise HTTPException(status_code=404, detail="记忆不存在")
    return {"success": True, "message": "记忆已删除"}


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
