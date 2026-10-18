"""评估相关 API - Gold 评估集管理与评估运行"""
import asyncio
import json
import logging
import secrets
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from app.dependencies import get_current_admin, get_current_user, get_request_database
from db.json_fields import read_json_object
from db.schemas import EvalRunRequest, FeedbackCreate

logger = logging.getLogger(__name__)
router = APIRouter()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@router.get("/api/evaluation/gold-set")
async def get_gold_set(
    dataset_id: str = "kisaki_v21", category: Optional[str] = None, split: Optional[str] = None,
    current_user: dict = Depends(get_current_admin),
):
    """返回 Gold 评估集（可按 category/split 过滤）"""
    try:
        from evaluation.runtime_runner import load_runtime_dataset
        dataset = await asyncio.to_thread(load_runtime_dataset, dataset_id)
        prompts = dataset["prompts"]
        if category:
            prompts = [p for p in prompts if p.get("category") == category]
        if split and any("split" in item for item in prompts):
            prompts = [p for p in prompts if p.get("split", "eval") == split]
        categories = {}
        for p in prompts:
            c = p.get("category", "unknown")
            categories[c] = categories.get(c, 0) + 1
        return {
            "success": True,
            "dataset_id": dataset_id,
            "dataset_status": dataset.get("status"),
            "dataset_role": dataset.get("evaluation_role"),
            "total": len(prompts),
            "category_breakdown": categories,
            "prompts": prompts,
        }
    except LookupError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("加载 Gold Set 失败: %s", type(exc).__name__)
        raise HTTPException(status_code=500, detail="加载 Gold Set 失败") from exc


@router.post("/api/evaluation/run")
async def run_evaluation(
    req: EvalRunRequest, current_user: dict = Depends(get_current_admin),
    database=Depends(get_request_database),
):
    """Persist and queue one bounded evaluation run."""
    run_id = f"eval_{secrets.token_hex(8)}"
    queued_metrics = {"status": "queued", "mock": req.mock}
    try:
        await asyncio.to_thread(
            database.execute_sql_insert,
            "INSERT INTO gold_eval_runs (id, run_at, adapter_name, model_label, total_prompts, category_breakdown, metrics, config_snapshot, notes) "
            "VALUES (:id, :run_at, :adapter_name, :model_label, 0, :cb, :metrics, :cfg, :notes)",
            {
                "id": run_id,
                "run_at": _now(),
                "adapter_name": req.adapter_name,
                "model_label": req.model_label,
                "cb": json.dumps({}),
                "metrics": json.dumps(queued_metrics),
                "cfg": json.dumps(req.model_dump()),
                "notes": "queued",
            },
        )
    except Exception as exc:
        logger.error("failed to create evaluation run=%s type=%s", run_id, type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail="无法创建评估任务，请稍后重试",
            headers={"Retry-After": "1"},
        ) from exc

    from evaluation.runtime_runner import schedule_generation_evaluation

    try:
        schedule_generation_evaluation(run_id, req.model_dump(), database)
    except Exception as exc:
        logger.error("failed to schedule evaluation run=%s type=%s", run_id, type(exc).__name__)
        try:
            await asyncio.to_thread(
                database.execute_sql,
                "UPDATE gold_eval_runs SET metrics=:metrics, notes=:notes WHERE id=:id",
                {
                    "metrics": json.dumps({"status": "failed", "error": "schedule_failed", "mock": req.mock}),
                    "notes": "failed",
                    "id": run_id,
                },
            )
        except Exception as persist_error:
            logger.error("failed to persist scheduling failure run=%s type=%s", run_id, type(persist_error).__name__)
        raise HTTPException(status_code=500, detail="评估任务调度失败") from exc

    return {"success": True, "run_id": run_id, "status": "queued", "mock": req.mock}


@router.get("/api/evaluation/runs")
async def list_runs(
    limit: int = 20, offset: int = 0,
    current_user: dict = Depends(get_current_admin),
    database=Depends(get_request_database),
):
    """列出评估运行历史"""
    try:
        limit = max(1, min(int(limit), 500))
        offset = max(0, int(offset))
        rows, count_rows = await asyncio.gather(
            asyncio.to_thread(
                database.execute_sql,
                "SELECT id, run_at, adapter_name, model_label, total_prompts, metrics, notes "
                "FROM gold_eval_runs ORDER BY run_at DESC LIMIT :lim OFFSET :off",
                {"lim": limit, "off": offset},
            ),
            asyncio.to_thread(
                database.execute_sql,
                "SELECT COUNT(*) AS cnt FROM gold_eval_runs",
                {},
            ),
        )
        runs = []
        for row in rows:
            runs.append({
                "id": row["id"],
                "run_at": row["run_at"],
                "adapter_name": row["adapter_name"],
                "model_label": row["model_label"],
                "total_prompts": row["total_prompts"],
                "metrics": read_json_object(row["metrics"], "metrics"),
                "notes": row["notes"],
            })
        total = count_rows[0]["cnt"]
        return {"success": True, "runs": runs, "total": total}
    except Exception as exc:
        logger.error("列出评估运行失败: %s", type(exc).__name__)
        raise HTTPException(status_code=500, detail="列出评估运行失败") from exc


@router.get("/api/evaluation/runs/{run_id}")
async def get_run_detail(
    run_id: str, current_user: dict = Depends(get_current_admin),
    database=Depends(get_request_database),
):
    """获取单个评估运行的详细结果"""
    try:
        rows = await asyncio.to_thread(
            database.execute_sql,
            "SELECT * FROM gold_eval_runs WHERE id=:id",
            {"id": run_id},
        )
        if not rows:
            raise HTTPException(status_code=404, detail="run not found")
        r = rows[0]
        return {
            "success": True,
            "run": {
                "id": r["id"], "run_at": r["run_at"], "adapter_name": r["adapter_name"],
                "model_label": r["model_label"], "total_prompts": r["total_prompts"],
                "category_breakdown": read_json_object(r["category_breakdown"], "category_breakdown"), "metrics": read_json_object(r["metrics"], "metrics"),
                "config_snapshot": read_json_object(r["config_snapshot"], "config_snapshot"), "notes": r["notes"],
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("获取评估运行详情失败 run=%s type=%s", run_id, type(exc).__name__)
        raise HTTPException(status_code=500, detail="获取评估运行详情失败") from exc


@router.post("/api/feedback")
async def create_feedback(
    req: FeedbackCreate, current_user: dict = Depends(get_current_user),
    database=Depends(get_request_database),
):
    """创建用户反馈（在线反馈闭环）"""
    try:
        await asyncio.to_thread(
            database.execute_sql_insert,
            "INSERT INTO feedback (trace_id, message_id, rating, reason, adapter_name, kb_revision, prompt_version, detail, created_at) "
            "VALUES (:trace_id, :message_id, :rating, :reason, :adapter_name, :kb_revision, :prompt_version, :detail, :created_at)",
            {
                "trace_id": req.trace_id,
                "message_id": req.message_id,
                "rating": req.rating,
                "reason": req.reason,
                "adapter_name": req.adapter_name,
                "kb_revision": req.kb_revision,
                "prompt_version": req.prompt_version,
                "detail": req.detail,
                "created_at": _now(),
            },
        )
        return {"success": True}
    except Exception as exc:
        logger.error("创建反馈失败: %s", type(exc).__name__)
        raise HTTPException(status_code=500, detail="创建反馈失败") from exc


@router.get("/api/feedback")
async def list_feedback(
    limit: int = 50, offset: int = 0, rating: Optional[str] = None,
    current_user: dict = Depends(get_current_admin),
    database=Depends(get_request_database),
):
    """列出用户反馈"""
    try:
        limit = max(1, min(int(limit), 500))
        offset = max(0, int(offset))
        if rating:
            rows_query = (
                "SELECT * FROM feedback WHERE rating=:rating "
                "ORDER BY created_at DESC LIMIT :lim OFFSET :off"
            )
            rows_params = {"rating": rating, "lim": limit, "off": offset}
            count_query = "SELECT COUNT(*) AS cnt FROM feedback WHERE rating=:rating"
            count_params = {"rating": rating}
        else:
            rows_query = "SELECT * FROM feedback ORDER BY created_at DESC LIMIT :lim OFFSET :off"
            rows_params = {"lim": limit, "off": offset}
            count_query = "SELECT COUNT(*) AS cnt FROM feedback"
            count_params = {}

        rows, count_rows = await asyncio.gather(
            asyncio.to_thread(database.execute_sql, rows_query, rows_params),
            asyncio.to_thread(database.execute_sql, count_query, count_params),
        )
        feedbacks = [dict(row) for row in rows]
        total = count_rows[0]["cnt"]
        return {"success": True, "feedbacks": feedbacks, "total": total}
    except Exception as exc:
        logger.error("列出反馈失败: %s", type(exc).__name__)
        raise HTTPException(status_code=500, detail="列出反馈失败") from exc
