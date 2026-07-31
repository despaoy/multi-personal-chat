"""偏好数据管理 API - DPO/ORPO 训练数据管理"""
import asyncio
import json
import logging
import secrets
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from app.dependencies import get_current_admin
from db.adapter import db
from db.schemas import (
    PreferenceExportRequest,
    PreferencePairCreate,
    PreferencePairUpdate,
    PreferenceReviewStatus,
    SampleFromHistoryRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter()


async def _execute_sql(query: str, params: Optional[dict] = None):
    return await asyncio.to_thread(db.execute_sql, query, params or {})


async def _execute_sql_insert(query: str, params: Optional[dict] = None):
    return await asyncio.to_thread(db.execute_sql_insert, query, params or {})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@router.get("/api/preferences/")
async def list_preferences(review_status: Optional[PreferenceReviewStatus] = None, limit: int = 50, offset: int = 0,
                           current_user: dict = Depends(get_current_admin)):
    """列出偏好对（分页）"""
    try:
        limit = max(1, min(int(limit), 500))
        offset = max(0, int(offset))
        if review_status:
            rows = await _execute_sql(
                "SELECT * FROM preference_pairs WHERE review_status=:rs ORDER BY created_at DESC LIMIT :lim OFFSET :off",
                {"rs": review_status, "lim": limit, "off": offset},
            )
            count_rows = await _execute_sql(
                "SELECT COUNT(*) as cnt FROM preference_pairs WHERE review_status=:rs",
                {"rs": review_status},
            )
        else:
            rows = await _execute_sql(
                "SELECT * FROM preference_pairs ORDER BY created_at DESC LIMIT :lim OFFSET :off",
                {"lim": limit, "off": offset},
            )
            count_rows = await _execute_sql("SELECT COUNT(*) as cnt FROM preference_pairs")
        total = count_rows[0]["cnt"] if count_rows else 0
        pairs = []
        for r in (rows or []):
            try:
                rubric = json.loads(r["rubric"]) if r["rubric"] else {}
            except Exception:
                rubric = {}
            try:
                metadata = json.loads(r["metadata"]) if r["metadata"] else {}
            except Exception:
                metadata = {}
            pairs.append({
                "id": r["id"], "prompt": r["prompt"], "chosen": r["chosen"],
                "rejected": r["rejected"], "rubric": rubric, "annotator": r["annotator"],
                "metadata": metadata, "review_status": r["review_status"],
                "created_at": r["created_at"],
            })
        return {"success": True, "preferences": pairs, "total": total, "limit": limit, "offset": offset}
    except Exception as e:
        logger.error(f"列出偏好对失败: {e}")
        raise HTTPException(status_code=500, detail="偏好数据操作失败") from e


@router.post("/api/preferences/")
async def create_preference(req: PreferencePairCreate,
                            current_user: dict = Depends(get_current_admin)):
    """创建偏好对"""
    pid = f"pref_{secrets.token_hex(8)}"
    try:
        await _execute_sql_insert(
            "INSERT INTO preference_pairs (id, prompt, chosen, rejected, rubric, annotator, metadata, review_status, created_at) "
            "VALUES (:id, :prompt, :chosen, :rejected, :rubric, :annotator, :metadata, :review_status, :created_at)",
            {
                "id": pid, "prompt": req.prompt, "chosen": req.chosen,
                "rejected": req.rejected,
                "rubric": json.dumps(req.rubric or {}, ensure_ascii=False),
                "annotator": req.annotator,
                "metadata": json.dumps(req.metadata or {}, ensure_ascii=False),
                "review_status": req.review_status, "created_at": _now(),
            },
        )
        return {"success": True, "id": pid}
    except Exception as e:
        logger.error(f"创建偏好对失败: {e}")
        raise HTTPException(status_code=500, detail="偏好数据操作失败") from e


@router.put("/api/preferences/{pid}")
async def update_preference(pid: str, req: PreferencePairUpdate,
                            current_user: dict = Depends(get_current_admin)):
    """更新偏好对（审核状态变更等）"""
    try:
        updates = []
        params: dict = {}
        if req.review_status is not None:
            updates.append("review_status=:review_status")
            params["review_status"] = req.review_status
        if req.rubric is not None:
            updates.append("rubric=:rubric")
            params["rubric"] = json.dumps(req.rubric, ensure_ascii=False)
        if req.annotator is not None:
            updates.append("annotator=:annotator")
            params["annotator"] = req.annotator
        if not updates:
            return {"success": True, "id": pid, "note": "no fields to update"}
        params["id"] = pid
        updated = await _execute_sql(
            f"UPDATE preference_pairs SET {', '.join(updates)} WHERE id=:id",
            params,
        )
        if not updated:
            raise HTTPException(status_code=404, detail="偏好对不存在")
        return {"success": True, "id": pid}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新偏好对失败: {e}")
        raise HTTPException(status_code=500, detail="偏好数据操作失败") from e


@router.delete("/api/preferences/{pid}")
async def delete_preference(pid: str, current_user: dict = Depends(get_current_admin)):
    """删除偏好对"""
    try:
        deleted = await _execute_sql("DELETE FROM preference_pairs WHERE id=:id", {"id": pid})
        if not deleted:
            raise HTTPException(status_code=404, detail="偏好对不存在")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除偏好对失败: {e}")
        raise HTTPException(status_code=500, detail="偏好数据操作失败") from e


@router.post("/api/preferences/export")
async def export_preferences(req: PreferenceExportRequest,
                             current_user: dict = Depends(get_current_admin)):
    """导出偏好对为训练格式（JSONL）"""
    try:
        rows = await _execute_sql(
            "SELECT * FROM preference_pairs WHERE review_status=:rs "
            "ORDER BY created_at LIMIT :lim",
            {"rs": req.review_status, "lim": req.limit},
        )
        pairs = []
        for r in (rows or []):
            pairs.append({
                "prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"],
                "rubric": json.loads(r["rubric"]) if r["rubric"] else {},
            })
        return {"success": True, "format": req.format, "count": len(pairs), "data": pairs}
    except Exception as e:
        logger.error(f"导出偏好对失败: {e}")
        raise HTTPException(status_code=500, detail="偏好数据操作失败") from e


@router.post("/api/preferences/sample-from-history")
async def sample_from_history(req: SampleFromHistoryRequest,
                              current_user: dict = Depends(get_current_admin)):
    """从消息历史采样候选偏好对（待人工标注）"""
    try:
        if req.session_id:
            rows = await _execute_sql(
                "SELECT message, reply, loraName, traceId FROM messages WHERE sessionId=:sid AND reply != '' AND LENGTH(reply) >= :min_len ORDER BY RANDOM() LIMIT :lim",
                {"sid": req.session_id, "min_len": req.min_length, "lim": req.limit},
            )
        else:
            rows = await _execute_sql(
                "SELECT message, reply, loraName, traceId FROM messages WHERE reply != '' AND LENGTH(reply) >= :min_len ORDER BY RANDOM() LIMIT :lim",
                {"min_len": req.min_length, "lim": req.limit},
            )
        candidates = []
        for r in (rows or []):
            candidates.append({
                "prompt": r["message"], "response": r["reply"],
                "lora_name": r["loraName"], "trace_id": r["traceId"],
                "needs_annotation": True,
            })
        return {"success": True, "candidates": candidates, "total": len(candidates)}
    except Exception as e:
        logger.error(f"采样历史失败: {e}")
        raise HTTPException(status_code=500, detail="偏好数据操作失败") from e
