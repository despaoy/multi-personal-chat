"""评估相关 API - Gold 评估集管理与评估运行"""
import json
import logging
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from app.dependencies import get_current_user
from db.adapter import db
from db.schemas import EvalRunRequest

logger = logging.getLogger(__name__)
router = APIRouter()

_EVAL_DIR = Path(__file__).resolve().parent.parent / "evaluation"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@router.get("/api/evaluation/gold-set")
async def get_gold_set(category: Optional[str] = None, split: Optional[str] = None,
                       current_user: dict = Depends(get_current_user)):
    """返回 Gold 评估集（可按 category/split 过滤）"""
    try:
        from evaluation.gold_set_manager import get_gold_set_manager
        mgr = get_gold_set_manager()
        prompts = mgr.load_set()
        if category:
            prompts = [p for p in prompts if p.get("category") == category]
        if split:
            prompts = [p for p in prompts if p.get("split") == split]
        categories = {}
        for p in prompts:
            c = p.get("category", "unknown")
            categories[c] = categories.get(c, 0) + 1
        return {"success": True, "total": len(prompts), "category_breakdown": categories, "prompts": prompts}
    except ImportError:
        return {"success": True, "total": 0, "category_breakdown": {}, "prompts": [], "note": "evaluation module not initialized"}
    except Exception as e:
        logger.error(f"加载 gold set 失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/evaluation/run")
async def run_evaluation(req: EvalRunRequest, current_user: dict = Depends(get_current_user)):
    """触发评估运行（异步），返回 run_id"""
    run_id = f"eval_{secrets.token_hex(8)}"
    run_at = _now()
    try:
        db.execute_sql_insert(
            "INSERT INTO gold_eval_runs (id, run_at, adapter_name, model_label, total_prompts, category_breakdown, metrics, config_snapshot, notes) "
            "VALUES (:id, :run_at, :adapter_name, :model_label, 0, :cb, :metrics, :cfg, :notes)",
            {
                "id": run_id, "run_at": run_at, "adapter_name": req.adapter_name,
                "model_label": req.model_label, "cb": json.dumps({}),
                "metrics": json.dumps({}), "cfg": json.dumps(req.model_dump()), "notes": "",
            },
        )
    except Exception as e:
        logger.warning(f"记录评估运行失败（非致命）: {e}")

    from evaluation.runtime_runner import schedule_generation_evaluation
    queued_metrics = {"status": "queued", "mock": req.mock}
    try:
        db.execute_sql(
            "UPDATE gold_eval_runs SET metrics=:metrics, notes=:notes WHERE id=:id",
            {"metrics": json.dumps(queued_metrics), "notes": "queued", "id": run_id},
        )
    except Exception as exc:
        logger.warning("failed to persist queued evaluation run=%s: %s", run_id, exc)
    schedule_generation_evaluation(run_id, req.model_dump(), db)
    return {"success": True, "run_id": run_id, "status": "queued", "mock": req.mock}

@router.get("/api/evaluation/runs")
async def list_runs(limit: int = 20, offset: int = 0, current_user: dict = Depends(get_current_user)):
    """列出评估运行历史"""
    try:
        # 统一分页边界校验：limit 上限 500，offset 非负
        limit = max(1, min(int(limit), 500))
        offset = max(0, int(offset))
        rows = db.execute_sql(
            "SELECT id, run_at, adapter_name, model_label, total_prompts, metrics, notes "
            "FROM gold_eval_runs ORDER BY run_at DESC LIMIT :lim OFFSET :off",
            {"lim": limit, "off": offset},
        )
        count_rows = db.execute_sql(
            "SELECT COUNT(*) AS cnt FROM gold_eval_runs",
            {},
        )
        runs = []
        for r in (rows or []):
            try:
                metrics = json.loads(r["metrics"]) if r["metrics"] else {}
            except Exception:
                metrics = {}
            runs.append({
                "id": r["id"], "run_at": r["run_at"], "adapter_name": r["adapter_name"],
                "model_label": r["model_label"], "total_prompts": r["total_prompts"],
                "metrics": metrics, "notes": r["notes"],
            })
        return {"success": True, "runs": runs}
    except Exception as e:
        logger.error(f"列出评估运行失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/evaluation/runs/{run_id}")
async def get_run_detail(run_id: str, current_user: dict = Depends(get_current_user)):
    """获取单个评估运行的详细结果"""
    try:
        rows = db.execute_sql(
            "SELECT * FROM gold_eval_runs WHERE id=:id",
            {"id": run_id},
        )
        if not rows:
            raise HTTPException(status_code=404, detail="run not found")
        r = rows[0]
        try:
            metrics = json.loads(r["metrics"]) if r["metrics"] else {}
        except Exception:
            metrics = {}
        try:
            breakdown = json.loads(r["category_breakdown"]) if r["category_breakdown"] else {}
        except Exception:
            breakdown = {}
        try:
            config = json.loads(r["config_snapshot"]) if r["config_snapshot"] else {}
        except Exception:
            config = {}
        return {
            "success": True,
            "run": {
                "id": r["id"], "run_at": r["run_at"], "adapter_name": r["adapter_name"],
                "model_label": r["model_label"], "total_prompts": r["total_prompts"],
                "category_breakdown": breakdown, "metrics": metrics,
                "config_snapshot": config, "notes": r["notes"],
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取评估运行详情失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/feedback")
async def create_feedback(req: dict, current_user: dict = Depends(get_current_user)):
    """创建用户反馈（在线反馈闭环）"""
    from db.schemas import FeedbackCreate
    fb = FeedbackCreate(**req)
    try:
        db.execute_sql_insert(
            "INSERT INTO feedback (trace_id, message_id, rating, reason, adapter_name, kb_revision, prompt_version, detail, created_at) "
            "VALUES (:trace_id, :message_id, :rating, :reason, :adapter_name, :kb_revision, :prompt_version, :detail, :created_at)",
            {
                "trace_id": fb.trace_id, "message_id": fb.message_id,
                "rating": fb.rating, "reason": fb.reason,
                "adapter_name": fb.adapter_name, "kb_revision": fb.kb_revision,
                "prompt_version": fb.prompt_version, "detail": fb.detail,
                "created_at": _now(),
            },
        )
        return {"success": True}
    except Exception as e:
        logger.error(f"创建反馈失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/feedback")
async def list_feedback(limit: int = 50, offset: int = 0, rating: Optional[str] = None,
                        current_user: dict = Depends(get_current_user)):
    """列出用户反馈"""
    try:
        # 统一分页边界校验：limit 上限 500，offset 非负
        limit = max(1, min(int(limit), 500))
        offset = max(0, int(offset))
        # 使用命名参数 :name，SQLite 与 PostgreSQL 均支持
        if rating:
            rows = db.execute_sql(
                "SELECT * FROM feedback WHERE rating=:rating "
                "ORDER BY created_at DESC LIMIT :lim OFFSET :off",
                {"rating": rating, "lim": limit, "off": offset},
            )
            count_rows = db.execute_sql(
                "SELECT COUNT(*) AS cnt FROM feedback WHERE rating=:rating",
                {"rating": rating},
            )
        else:
            rows = db.execute_sql(
                "SELECT * FROM feedback ORDER BY created_at DESC "
                "LIMIT :lim OFFSET :off",
                {"lim": limit, "off": offset},
            )
            count_rows = db.execute_sql(
                "SELECT COUNT(*) AS cnt FROM feedback",
                {},
            )
        feedbacks = [dict(r) for r in (rows or [])]
        total = (count_rows[0]["cnt"] if count_rows else 0) or 0
        return {"success": True, "feedbacks": feedbacks, "total": total}
    except Exception as e:
        logger.error(f"列出反馈失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
