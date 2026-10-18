"""实验管理 API - LoRA 消融/RAG 消融/量化基准"""
import asyncio
import json
import logging
import secrets
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from app.dependencies import get_current_admin, get_request_database
from db.schemas import ExperimentStartRequest

logger = logging.getLogger(__name__)
router = APIRouter()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialize_results(results):
    return [asdict(item) if is_dataclass(item) else item for item in results]


def _read_results(value: str | None):
    if value is None:
        return {}
    decoded = json.loads(value)
    if not isinstance(decoded, (dict, list)):
        raise ValueError("Experiment results must be a JSON object or array")
    return decoded


async def _create_experiment_record(
    database,
    experiment_id: str,
    experiment_type: str,
    hypothesis: str,
) -> None:
    try:
        await asyncio.to_thread(
            database.execute_sql_insert,
            "INSERT INTO experiment_runs (id, experiment_type, hypothesis, status, started_at, results, config_path, report_path) "
            "VALUES (:id, :et, :hyp, 'running', :ts, :r, '', '')",
            {
                "id": experiment_id,
                "et": experiment_type,
                "hyp": hypothesis,
                "ts": _now(),
                "r": json.dumps({}),
            },
        )
    except Exception as exc:
        logger.error("failed to create experiment=%s type=%s", experiment_id, type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail="无法创建实验记录，请稍后重试",
            headers={"Retry-After": "1"},
        ) from exc


@router.get("/api/experiments/")
async def list_experiments(
    experiment_type: Optional[str] = None, limit: int = 50, offset: int = 0,
    current_user: dict = Depends(get_current_admin),
    database=Depends(get_request_database),
):
    """列出实验运行记录"""
    try:
        # 统一分页边界校验：limit 上限 500，offset 非负
        limit = max(1, min(int(limit), 500))
        offset = max(0, int(offset))
        # 使用命名参数 :name，SQLite 与 PostgreSQL 均支持
        if experiment_type:
            rows = await asyncio.to_thread(database.execute_sql,
                "SELECT * FROM experiment_runs WHERE experiment_type=:et "
                "ORDER BY started_at DESC LIMIT :lim OFFSET :off",
                {"et": experiment_type, "lim": limit, "off": offset},
            )
            count_rows = await asyncio.to_thread(database.execute_sql,
                "SELECT COUNT(*) AS cnt FROM experiment_runs WHERE experiment_type=:et",
                {"et": experiment_type},
            )
        else:
            rows = await asyncio.to_thread(database.execute_sql,
                "SELECT * FROM experiment_runs ORDER BY started_at DESC "
                "LIMIT :lim OFFSET :off",
                {"lim": limit, "off": offset},
            )
            count_rows = await asyncio.to_thread(database.execute_sql,
                "SELECT COUNT(*) AS cnt FROM experiment_runs",
                {},
            )
        experiments = []
        for r in rows:
            results = _read_results(r["results"])
            experiments.append({
                "id": r["id"], "experiment_type": r["experiment_type"],
                "hypothesis": r["hypothesis"], "status": r["status"],
                "started_at": r["started_at"], "completed_at": r["completed_at"],
                "results": results, "report_path": r["report_path"],
            })
        total = count_rows[0]["cnt"]
        return {"success": True, "experiments": experiments, "total": total}
    except Exception as e:
        logger.error("列出实验失败: %s", type(e).__name__)
        raise HTTPException(status_code=503, detail="实验记录暂时不可用") from e


@router.get("/api/experiments/{exp_id}")
async def get_experiment(
    exp_id: str, current_user: dict = Depends(get_current_admin),
    database=Depends(get_request_database),
):
    """获取单个实验详情"""
    try:
        rows = await asyncio.to_thread(database.execute_sql,"SELECT * FROM experiment_runs WHERE id=:id", {"id": exp_id})
        if not rows:
            raise HTTPException(status_code=404, detail="experiment not found")
        r = rows[0]
        results = _read_results(r["results"])
        return {
            "success": True,
            "experiment": {
                "id": r["id"], "experiment_type": r["experiment_type"],
                "hypothesis": r["hypothesis"], "status": r["status"],
                "started_at": r["started_at"], "completed_at": r["completed_at"],
                "results": results, "config_path": r["config_path"],
                "report_path": r["report_path"],
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("获取实验详情失败: %s", type(e).__name__)
        raise HTTPException(status_code=503, detail="实验记录暂时不可用") from e


@router.post("/api/experiments/lora-ablation")
async def start_lora_ablation(
    req: ExperimentStartRequest, current_user: dict = Depends(get_current_admin),
    database=Depends(get_request_database),
):
    """启动 LoRA 消融实验"""
    exp_id = f"lora_abl_{secrets.token_hex(6)}"
    await _create_experiment_record(
        database,
        exp_id,
        "lora_ablation",
        req.hypothesis or "LoRA vs DoRA vs RSLoRA ablation",
    )
    if req.mock:
        mock_results = {
            "mock": True,
            "variants": ["lora_baseline", "dora", "rslora", "lora_neftune", "lora_packing"],
            "comparison_table": [
                {"variant": "lora_baseline", "eval_loss": 1.85, "perplexity": 6.36, "adapter_size_mb": 45.2, "trainable_params": 19568128},
                {"variant": "dora", "eval_loss": 1.79, "perplexity": 5.99, "adapter_size_mb": 45.3, "trainable_params": 19568128},
                {"variant": "rslora", "eval_loss": 1.82, "perplexity": 6.17, "adapter_size_mb": 45.2, "trainable_params": 19568128},
            ],
        }
        await asyncio.to_thread(database.execute_sql,
            "UPDATE experiment_runs SET status='completed', completed_at=:ts, results=:r WHERE id=:id",
            {"ts": _now(), "r": json.dumps(mock_results), "id": exp_id},
        )
        return {"success": True, "experiment_id": exp_id, "status": "completed", "mock": True, "results": mock_results}

    error = (
        "Formal R1V4 training is blocked until human review, the V4 dataset, and Gold v3 "
        "are frozen. Run scripts/validate_kisaki_v4_training_gate.py to inspect blockers."
    )
    await asyncio.to_thread(database.execute_sql,
        "UPDATE experiment_runs SET status='failed', completed_at=:ts, results=:r WHERE id=:id",
        {"ts": _now(), "r": json.dumps({"mock": False, "error": error}), "id": exp_id},
    )
    raise HTTPException(status_code=409, detail=error)


@router.post("/api/experiments/rag-ablation")
async def start_rag_ablation(
    req: ExperimentStartRequest, current_user: dict = Depends(get_current_admin),
    database=Depends(get_request_database),
):
    """启动 RAG 消融实验"""
    exp_id = f"rag_abl_{secrets.token_hex(6)}"
    await _create_experiment_record(
        database,
        exp_id,
        "rag_ablation",
        req.hypothesis or "vector vs BM25 vs hybrid vs hybrid+reranker",
    )
    try:
        from experiments.rag_ablation import RAGAblation

        ablation = RAGAblation(formal=not req.mock)
        raw_results = await asyncio.to_thread(
            ablation.run_all_mock if req.mock else ablation.run_all
        )
        serialized = _serialize_results(raw_results)
        payload = {
            "mock": req.mock,
            "formal": not req.mock,
            "results": serialized,
            "comparison_table": ablation.build_comparison_table(raw_results),
        }
        await asyncio.to_thread(database.execute_sql,
            "UPDATE experiment_runs SET status='completed', completed_at=:ts, results=:r WHERE id=:id",
            {"ts": _now(), "r": json.dumps(payload, ensure_ascii=False), "id": exp_id},
        )
        return {
            "success": True,
            "experiment_id": exp_id,
            "status": "completed",
            "mock": req.mock,
            "results": payload,
        }
    except ValueError as exc:
        error = str(exc)
        await asyncio.to_thread(database.execute_sql,
            "UPDATE experiment_runs SET status='failed', completed_at=:ts, results=:r WHERE id=:id",
            {"ts": _now(), "r": json.dumps({"mock": req.mock, "error": error}), "id": exp_id},
        )
        raise HTTPException(status_code=409, detail=error)
    except ImportError:
        error = "RAG ablation module not available"
        await asyncio.to_thread(database.execute_sql,
            "UPDATE experiment_runs SET status='failed', completed_at=:ts, results=:r WHERE id=:id",
            {"ts": _now(), "r": json.dumps({"error": error}), "id": exp_id},
        )
        raise HTTPException(status_code=503, detail=error)
    except Exception as e:
        logger.error("RAG ablation failed experiment_id=%s type=%s", exp_id, type(e).__name__)
        await asyncio.to_thread(database.execute_sql,
            "UPDATE experiment_runs SET status='failed', completed_at=:ts, results=:r WHERE id=:id",
            {"ts": _now(), "r": json.dumps({"error": str(e)}, ensure_ascii=False), "id": exp_id},
        )
        raise HTTPException(status_code=500, detail="RAG ablation failed") from e


@router.post("/api/experiments/quantization-benchmark")
async def start_quantization_benchmark(
    req: ExperimentStartRequest, current_user: dict = Depends(get_current_admin),
    database=Depends(get_request_database),
):
    """启动量化基准实验"""
    exp_id = f"quant_bench_{secrets.token_hex(6)}"
    await _create_experiment_record(
        database,
        exp_id,
        "quantization_benchmark",
        req.hypothesis or "FP16 vs AWQ vs NF4 vs INT8 comparison",
    )
    try:
        from experiments.quantization_benchmark import QuantizationBenchmark
        bench = QuantizationBenchmark()
        if not req.mock:
            error = (
                "A real quantization comparison requires one isolated vLLM process per "
                "quantization variant. Use scripts/lab-run-kisaki-r3.sh on the lab server."
            )
            await asyncio.to_thread(database.execute_sql,
                "UPDATE experiment_runs SET status='failed', completed_at=:ts, results=:r WHERE id=:id",
                {"ts": _now(), "r": json.dumps({"error": error}), "id": exp_id},
            )
            raise HTTPException(status_code=409, detail=error)
        rows = [item.to_dict() for item in await bench.run_comparison(bench.DEFAULT_CONFIGS, mock=True)]
        results = {"mock": True, "formal": False, "results": rows}
        await asyncio.to_thread(database.execute_sql,
            "UPDATE experiment_runs SET status='completed', completed_at=:ts, results=:r WHERE id=:id",
            {"ts": _now(), "r": json.dumps(results), "id": exp_id},
        )
        return {"success": True, "experiment_id": exp_id, "status": "completed", "mock": True, "results": results}
    except HTTPException:
        raise
    except ImportError as exc:
        error = "Quantization benchmark module is unavailable"
        await asyncio.to_thread(database.execute_sql,
            "UPDATE experiment_runs SET status='failed', completed_at=:ts, results=:r WHERE id=:id",
            {"ts": _now(), "r": json.dumps({"mock": req.mock, "error": error}), "id": exp_id},
        )
        raise HTTPException(status_code=503, detail=error) from exc
    except Exception as e:
        logger.error("量化基准实验失败 experiment_id=%s type=%s", exp_id, type(e).__name__)
        await asyncio.to_thread(database.execute_sql,
            "UPDATE experiment_runs SET status='failed', completed_at=:ts, results=:r WHERE id=:id",
            {"ts": _now(), "r": json.dumps({"error": str(e)}), "id": exp_id},
        )
        raise HTTPException(status_code=500, detail="量化基准实验失败") from e


@router.get("/api/experiments/{exp_id}/report")
async def download_report(
    exp_id: str, current_user: dict = Depends(get_current_admin),
    database=Depends(get_request_database),
):
    """获取实验报告（Markdown 格式）"""
    try:
        rows = await asyncio.to_thread(database.execute_sql,"SELECT * FROM experiment_runs WHERE id=:id", {"id": exp_id})
        if not rows:
            raise HTTPException(status_code=404, detail="experiment not found")
        r = rows[0]
        results = _read_results(r["results"])
        lines = [
            f"# 实验报告: {r['id']}",
            f"",
            f"- **类型**: {r['experiment_type']}",
            f"- **状态**: {r['status']}",
            f"- **假设**: {r['hypothesis'] or 'N/A'}",
            f"- **开始时间**: {r['started_at']}",
            f"- **完成时间**: {r['completed_at'] or 'N/A'}",
            f"",
            "## 结果",
            "",
            "```json",
            json.dumps(results, indent=2, ensure_ascii=False),
            "```",
        ]
        report = "\n".join(lines)
        return {"success": True, "report": report, "format": "markdown"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("获取实验报告失败: %s", type(e).__name__)
        raise HTTPException(status_code=503, detail="实验报告暂时不可用") from e
