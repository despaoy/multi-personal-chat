"""消息记录API"""
import logging
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.dependencies import get_current_admin
from app.providers import get_message_repository

from repositories.messages import MessageQuery, MessageRepository

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/messages")
async def get_messages(
    search: Optional[str] = Query(None),
    sessionType: Optional[str] = Query(None),
    lora: Optional[str] = Query(None),
    sessionId: Optional[str] = Query(None),
    sessionName: Optional[str] = Query(None),
    platform: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_admin),
    repository: MessageRepository = Depends(get_message_repository),
):
    """获取消息记录 — 支持 SQL 层多条件组合过滤 + 分页"""
    # C2 fix: 原先 5 个端点均无 try/except，数据库失败时 FastAPI 返回 500
    # 并将原始异常字符串泄露给客户端。现在统一捕获并转换为 HTTPException，
    # 同时记录 traceback 便于运维排查。
    try:
        page = await repository.list_page(
            MessageQuery(
                search=search,
                session_type=sessionType if sessionType and sessionType != "all" else None,
                lora_name=lora if lora and lora != "all" else None,
                session_id=sessionId,
                session_name=sessionName,
                platform=platform if platform and platform != "all" else None,
            ),
            limit=limit,
            offset=offset,
        )

        return {
            "messages": page.messages,
            "total": page.total,
            "total_all": page.total_all,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取消息记录失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="获取消息记录失败")


@router.get("/api/sessions")
async def get_session_summaries(
    current_user: dict = Depends(get_current_admin),
    repository: MessageRepository = Depends(get_message_repository),
):
    """获取所有会话的聚合统计（按sessionId分组）"""
    try:
        sessions = await repository.list_session_summaries()
        return {"sessions": sessions}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取会话统计失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="获取会话统计失败")


class SessionBotToggle(BaseModel):
    sessionId: str
    enabled: bool
    platform: str = "qq"
    conversationId: Optional[str] = None
    conversationType: Literal["private", "group", "channel"] = "private"


@router.put("/api/sessions/bot-toggle")
async def toggle_session_bot(
    req: SessionBotToggle,
    current_user: dict = Depends(get_current_admin),
    repository: MessageRepository = Depends(get_message_repository),
):
    """设置某个会话的机器人开关

    s2 fix: 影响任意会话的机器人开关（IDOR），限定 admin。
    """
    try:
        await repository.set_session_bot_enabled(
            req.sessionId,
            req.enabled,
            platform=req.platform,
            conversation_id=req.conversationId,
            conversation_type=req.conversationType,
        )
        return {"success": True, "sessionId": req.sessionId, "botEnabled": req.enabled}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"设置机器人开关失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="设置机器人开关失败")


class BatchDeleteRequest(BaseModel):
    search: Optional[str] = None
    sessionType: Optional[str] = None
    lora: Optional[str] = None
    sessionName: Optional[str] = None
    platform: Optional[str] = None


@router.delete("/api/messages/batch")
async def delete_messages_batch(
    req: BatchDeleteRequest,
    current_user: dict = Depends(get_current_admin),
    repository: MessageRepository = Depends(get_message_repository),
):
    """批量删除消息（基于筛选条件）

    C-S1 fix: 批量删除不可逆，限定 admin。
    """
    try:
        count = await repository.delete_filtered(
            MessageQuery(
                search=req.search,
                session_type=req.sessionType if req.sessionType != "all" else None,
                lora_name=req.lora if req.lora != "all" else None,
                session_name=req.sessionName,
                platform=req.platform if req.platform != "all" else None,
            ),
        )
        return {"success": True, "deleted": count, "message": f"已删除 {count} 条记录"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量删除消息失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="批量删除消息失败")


@router.delete("/api/messages/{msg_id}")
async def delete_message(
    msg_id: int,
    current_user: dict = Depends(get_current_admin),
    repository: MessageRepository = Depends(get_message_repository),
):
    """删除单条消息记录 — 需 admin

    s2 fix: 消息无所有权字段，批量删除已是 admin，单条删除对齐为 admin。
    """
    try:
        success = await repository.delete(msg_id)
        if not success:
            raise HTTPException(status_code=404, detail="消息不存在")
        return {"success": True, "message": "删除成功"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除消息失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="删除消息失败")
