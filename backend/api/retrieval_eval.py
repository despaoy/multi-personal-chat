"""检索评估数据集管理 API"""
import asyncio
import json
import logging
import secrets
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from app.dependencies import get_current_admin
from db.adapter import db
from db.schemas import RetrievalEvalQuestionCreate

logger = logging.getLogger(__name__)
router = APIRouter()


async def _execute_sql(query: str, params: Optional[dict] = None):
    return await asyncio.to_thread(db.execute_sql, query, params or {})


async def _execute_sql_insert(query: str, params: Optional[dict] = None):
    return await asyncio.to_thread(db.execute_sql_insert, query, params or {})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@router.get("/api/retrieval-eval/questions")
async def list_questions(category: Optional[str] = None, limit: int = 100, offset: int = 0,
                         current_user: dict = Depends(get_current_admin)):
    """列出检索评估问题"""
    try:
        # 统一分页边界校验：limit 上限 500，offset 非负
        limit = max(1, min(int(limit), 500))
        offset = max(0, int(offset))
        # 使用命名参数 :name，SQLite 与 PostgreSQL 均支持
        if category:
            rows = await _execute_sql(
                "SELECT * FROM retrieval_eval_questions WHERE category=:cat "
                "ORDER BY created_at DESC LIMIT :lim OFFSET :off",
                {"cat": category, "lim": limit, "off": offset},
            )
            count_rows = await _execute_sql(
                "SELECT COUNT(*) AS cnt FROM retrieval_eval_questions WHERE category=:cat",
                {"cat": category},
            )
        else:
            rows = await _execute_sql(
                "SELECT * FROM retrieval_eval_questions ORDER BY created_at DESC "
                "LIMIT :lim OFFSET :off",
                {"lim": limit, "off": offset},
            )
            count_rows = await _execute_sql(
                "SELECT COUNT(*) AS cnt FROM retrieval_eval_questions",
                {},
            )
        questions = []
        for r in (rows or []):
            try:
                doc_ids = json.loads(r["expected_doc_ids"]) if r["expected_doc_ids"] else []
            except Exception:
                doc_ids = []
            try:
                doc_titles = json.loads(r["expected_doc_titles"]) if r["expected_doc_titles"] else []
            except Exception:
                doc_titles = []
            questions.append({
                "id": r["id"], "question": r["question"],
                "expected_doc_ids": doc_ids, "expected_doc_titles": doc_titles,
                "gold_answer": r["gold_answer"], "category": r["category"],
                "created_at": r["created_at"],
            })
        total = (count_rows[0]["cnt"] if count_rows else 0) or 0
        return {"success": True, "questions": questions, "total": total}
    except Exception as e:
        logger.error(f"列出检索评估问题失败: {e}")
        raise HTTPException(status_code=500, detail="检索评估数据操作失败") from e


@router.post("/api/retrieval-eval/questions")
async def create_question(req: RetrievalEvalQuestionCreate,
                          current_user: dict = Depends(get_current_admin)):
    """创建检索评估问题"""
    qid = req.id or f"rq_{secrets.token_hex(6)}"
    try:
        await _execute_sql_insert(
            "INSERT INTO retrieval_eval_questions (id, question, expected_doc_ids, expected_doc_titles, gold_answer, category, created_at) "
            "VALUES (:id, :q, :dids, :dtitles, :ga, :cat, :ts)",
            {
                "id": qid, "q": req.question,
                "dids": json.dumps(req.expected_doc_ids, ensure_ascii=False),
                "dtitles": json.dumps(req.expected_doc_titles, ensure_ascii=False),
                "ga": req.gold_answer, "cat": req.category, "ts": _now(),
            },
        )
        return {"success": True, "id": qid}
    except Exception as e:
        logger.error(f"创建检索评估问题失败: {e}")
        raise HTTPException(status_code=500, detail="检索评估数据操作失败") from e


@router.delete("/api/retrieval-eval/questions/{qid}")
async def delete_question(qid: str, current_user: dict = Depends(get_current_admin)):
    """删除检索评估问题"""
    try:
        deleted = await _execute_sql(
            "DELETE FROM retrieval_eval_questions WHERE id=:id",
            {"id": qid},
        )
        if not deleted:
            raise HTTPException(status_code=404, detail="检索评估问题不存在")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除检索评估问题失败: {e}")
        raise HTTPException(status_code=500, detail="检索评估数据操作失败") from e


@router.post("/api/retrieval-eval/run")
async def run_retrieval_eval(current_user: dict = Depends(get_current_admin)):
    """运行检索评估，计算 recall@k/MRR/nDCG 指标"""
    try:
        from evaluation.retrieval_metrics import RetrievalMetrics
        from knowledge.rag_helper import get_rag_helper
        rows = await asyncio.to_thread(db.execute_sql, "SELECT * FROM retrieval_eval_questions")
        if not rows:
            return {"success": True, "metrics": {}, "total": 0, "note": "no questions in dataset"}
        questions = []
        for r in rows:
            try:
                doc_ids = json.loads(r["expected_doc_ids"]) if r["expected_doc_ids"] else []
            except (TypeError, json.JSONDecodeError):
                doc_ids = []
            try:
                doc_titles = json.loads(r["expected_doc_titles"]) if r["expected_doc_titles"] else []
            except (TypeError, json.JSONDecodeError):
                doc_titles = []
            questions.append({
                "id": r["id"],
                "question": r["question"],
                "expected_doc_ids": doc_ids,
                "expected_doc_titles": doc_titles,
            })

        helper = await asyncio.to_thread(get_rag_helper)
        metrics = RetrievalMetrics()

        def retrieve_fn(query):
            return helper.retrieve_context(query, top_k=10, use_cache=False)

        results = await asyncio.to_thread(
            metrics.evaluate_questions,
            questions,
            retrieve_fn,
        )
        return {"success": True, "metrics": results, "total": len(questions)}
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="检索评估模块不可用",
        ) from exc
    except Exception as exc:
        logger.exception("检索评估失败")
        raise HTTPException(status_code=500, detail="检索评估失败") from exc
