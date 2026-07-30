"""实验管理 API - LoRA 消融/RAG 消融/量化基准"""
import asyncio
import json
import logging
import secrets
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from app.dependencies import get_current_user
from db.adapter import db
from db.schemas import ExperimentStartRequest

logger = logging.getLogger(__name__)
router = APIRouter()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_background_tasks: set[asyncio.Task] = set()


def _track_background(coro) -> None:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _run_lora_ablation(exp_id: str, overrides: Optional[dict]) -> None:
    """Run GPU training off the API event loop and persist terminal state."""
    try:
        from experiments.ablation_runner import AblationRunner

        runner = AblationRunner.from_default_config(overrides)
        results = await asyncio.to_thread(runner.run_all)
        db.execute_sql(
            "UPDATE experiment_runs SET status='completed', completed_at=:ts, results=:r WHERE id=:id",
            {"ts": _now(), "r": json.dumps(results, ensure_ascii=False), "id": exp_id},
        )
    except Exception as exc:
        logger.exception("LoRA ablation failed experiment_id=%s", exp_id)
        db.execute_sql(
            "UPDATE experiment_runs SET status='failed', completed_at=:ts, results=:r WHERE id=:id",
            {"ts": _now(), "r": json.dumps({"error": str(exc)}, ensure_ascii=False), "id": exp_id},
        )


def _serialize_results(results):
    return [asdict(item) if is_dataclass(item) else item for item in results]


@router.get("/api/experiments/")
async def list_experiments(experiment_type: Optional[str] = None,
                           limit: int = 50,
                           offset: int = 0,
                           current_user: dict = Depends(get_current_user)):
    """列出实验运行记录"""
    try:
        # 统一分页边界校验：limit 上限 500，offset 非负
        limit = max(1, min(int(limit), 500))
        offset = max(0, int(offset))
        # 使用命名参数 :name，SQLite 与 PostgreSQL 均支持
        if experiment_type:
            rows = db.execute_sql(
                "SELECT * FROM experiment_runs WHERE experiment_type=:et "
                "ORDER BY started_at DESC LIMIT :lim OFFSET :off",
                {"et": experiment_type, "lim": limit, "off": offset},
            )
            count_rows = db.execute_sql(
                "SELECT COUNT(*) AS cnt FROM experiment_runs WHERE experiment_type=:et",
                {"et": experiment_type},
            )
        else:
            rows = db.execute_sql(
                "SELECT * FROM experiment_runs ORDER BY started_at DESC "
                "LIMIT :lim OFFSET :off",
                {"lim": limit, "off": offset},
            )
            count_rows = db.execute_sql(
                "SELECT COUNT(*) AS cnt FROM experiment_runs",
                {},
            )
        experiments = []
        for r in (rows or []):
            try:
                results = json.loads(r["results"]) if r["results"] else {}
            except Exception:
                results = {}
            experiments.append({
                "id": r["id"], "experiment_type": r["experiment_type"],
                "hypothesis": r["hypothesis"], "status": r["status"],
                "started_at": r["started_at"], "completed_at": r["completed_at"],
                "results": results, "report_path": r["report_path"],
            })
        total = (count_rows[0]["cnt"] if count_rows else 0) or 0
        return {"success": True, "experiments": experiments, "total": total}
    except Exception as e:
        logger.error(f"列出实验失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/experiments/{exp_id}")
async def get_experiment(exp_id: str, current_user: dict = Depends(get_current_user)):
    """获取单个实验详情"""
    try:
        rows = db.execute_sql("SELECT * FROM experiment_runs WHERE id=:id", {"id": exp_id})
        if not rows:
            raise HTTPException(status_code=404, detail="experiment not found")
        r = rows[0]
        try:
            results = json.loads(r["results"]) if r["results"] else {}
        except Exception:
            results = {}
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
        logger.error(f"获取实验详情失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/experiments/lora-ablation")
async def start_lora_ablation(req: ExperimentStartRequest,
                              current_user: dict = Depends(get_current_user)):
    """启动 LoRA 消融实验"""
    exp_id = f"lora_abl_{secrets.token_hex(6)}"
    try:
        db.execute_sql_insert(
            "INSERT INTO experiment_runs (id, experiment_type, hypothesis, status, started_at, results, config_path, report_path) "
            "VALUES (:id, :et, :hyp, 'running', :ts, :r, '', '')",
            {
                "id": exp_id, "et": "lora_ablation",
                "hyp": req.hypothesis or "LoRA vs DoRA vs RSLoRA ablation",
                "ts": _now(), "r": json.dumps({}),
            },
        )
    except Exception as e:
        logger.warning(f"记录实验失败（非致命）: {e}")

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
        db.execute_sql(
            "UPDATE experiment_runs SET status='completed', completed_at=:ts, results=:r WHERE id=:id",
            {"ts": _now(), "r": json.dumps(mock_results), "id": exp_id},
        )
        return {"success": True, "experiment_id": exp_id, "status": "completed", "mock": True, "results": mock_results}

    error = (
        "Formal R1 training is protected by the reproducibility and GPU lock gates. "
        "Run scripts/lab-queue-kisaki-r1-extension.sh on the experiment server."
    )
    db.execute_sql(
        "UPDATE experiment_runs SET status='failed', completed_at=:ts, results=:r WHERE id=:id",
        {"ts": _now(), "r": json.dumps({"mock": False, "error": error}), "id": exp_id},
    )
    raise HTTPException(status_code=409, detail=error)


@router.post("/api/experiments/rag-ablation")
async def start_rag_ablation(req: ExperimentStartRequest,
                             current_user: dict = Depends(get_current_user)):
    """启动 RAG 消融实验"""
    exp_id = f"rag_abl_{secrets.token_hex(6)}"
    try:
        db.execute_sql_insert(
            "INSERT INTO experiment_runs (id, experiment_type, hypothesis, status, started_at, results, config_path, report_path) "
            "VALUES (:id, :et, :hyp, 'running', :ts, :r, '', '')",
            {
                "id": exp_id, "et": "rag_ablation",
                "hyp": req.hypothesis or "vector vs BM25 vs hybrid vs hybrid+reranker",
                "ts": _now(), "r": json.dumps({}),
            },
        )
    except Exception as e:
        logger.warning(f"记录实验失败（非致命）: {e}")

    try:
        from experiments.rag_ablation import RAGAblation

        ablation = RAGAblation()
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
        db.execute_sql(
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
        db.execute_sql(
            "UPDATE experiment_runs SET status='failed', completed_at=:ts, results=:r WHERE id=:id",
            {"ts": _now(), "r": json.dumps({"mock": req.mock, "error": error}), "id": exp_id},
        )
        raise HTTPException(status_code=409, detail=error)
    except ImportError:
        error = "RAG ablation module not available"
        db.execute_sql(
            "UPDATE experiment_runs SET status='failed', completed_at=:ts, results=:r WHERE id=:id",
            {"ts": _now(), "r": json.dumps({"error": error}), "id": exp_id},
        )
        # C5 fix: 模块不可用是服务端故障，应用 503 而非 200+success=False，
        # 与同函数 except Exception 分支（line 219）的 raise 模式保持一致。
        raise HTTPException(status_code=503, detail=error)
    except Exception as e:
        logger.exception("RAG ablation failed experiment_id=%s", exp_id)
        db.execute_sql(
            "UPDATE experiment_runs SET status='failed', completed_at=:ts, results=:r WHERE id=:id",
            {"ts": _now(), "r": json.dumps({"error": str(e)}, ensure_ascii=False), "id": exp_id},
        )
        raise HTTPException(status_code=500, detail="RAG ablation failed")


@router.post("/api/experiments/quantization-benchmark")
async def start_quantization_benchmark(req: ExperimentStartRequest,
                                       current_user: dict = Depends(get_current_user)):
    """启动量化基准实验"""
    exp_id = f"quant_bench_{secrets.token_hex(6)}"
    try:
        db.execute_sql_insert(
            "INSERT INTO experiment_runs (id, experiment_type, hypothesis, status, started_at, results, config_path, report_path) "
            "VALUES (:id, :et, :hyp, 'running', :ts, :r, '', '')",
            {
                "id": exp_id, "et": "quantization_benchmark",
                "hyp": req.hypothesis or "FP16 vs AWQ vs NF4 vs INT8 comparison",
                "ts": _now(), "r": json.dumps({}),
            },
        )
    except Exception as e:
        logger.warning(f"记录实验失败（非致命）: {e}")

    try:
        from experiments.quantization_benchmark import QuantizationBenchmark
        bench = QuantizationBenchmark()
        if not req.mock:
            error = (
                "A real quantization comparison requires one isolated vLLM process per "
                "quantization variant. Use scripts/lab-run-kisaki-r3.sh on the lab server."
            )
            db.execute_sql(
                "UPDATE experiment_runs SET status='failed', completed_at=:ts, results=:r WHERE id=:id",
                {"ts": _now(), "r": json.dumps({"error": error}), "id": exp_id},
            )
            raise HTTPException(status_code=409, detail=error)
        rows = [item.to_dict() for item in await bench.run_comparison(bench.DEFAULT_CONFIGS, mock=True)]
        results = {"mock": True, "formal": False, "results": rows}
        db.execute_sql(
            "UPDATE experiment_runs SET status='completed', completed_at=:ts, results=:r WHERE id=:id",
            {"ts": _now(), "r": json.dumps(results), "id": exp_id},
        )
        return {"success": True, "experiment_id": exp_id, "status": "completed", "mock": True, "results": results}
    except HTTPException:
        # C4 fix: 必须在 except Exception 之前 re-raise，否则 409 会被改写成 500，
        # 丢失"需在实验室服务器运行"的语义。对比 api/models.py:90 的同模式修复。
        raise
    except ImportError:
        mock_results = {
            "mock": True,
            "configs": ["fp16", "awq", "nf4", "int8"],
            "comparison_table": [
                {"config": "fp16", "load_time_s": 12.5, "vram_mb": 14800, "ttft_ms": 85, "decode_tokens_per_s": 42.3, "quality_score": 0.92},
                {"config": "awq", "load_time_s": 8.2, "vram_mb": 5200, "ttft_ms": 62, "decode_tokens_per_s": 55.1, "quality_score": 0.89},
                {"config": "nf4", "load_time_s": 9.1, "vram_mb": 4800, "ttft_ms": 70, "decode_tokens_per_s": 48.5, "quality_score": 0.87},
                {"config": "int8", "load_time_s": 10.3, "vram_mb": 7600, "ttft_ms": 75, "decode_tokens_per_s": 44.2, "quality_score": 0.90},
            ],
        }
        db.execute_sql(
            "UPDATE experiment_runs SET status='completed', completed_at=:ts, results=:r WHERE id=:id",
            {"ts": _now(), "r": json.dumps(mock_results), "id": exp_id},
        )
        return {"success": True, "experiment_id": exp_id, "status": "completed", "mock": True, "results": mock_results}
    except Exception as e:
        logger.error(f"量化基准实验失败: {e}")
        db.execute_sql(
            "UPDATE experiment_runs SET status='failed', completed_at=:ts, results=:r WHERE id=:id",
            {"ts": _now(), "r": json.dumps({"error": str(e)}), "id": exp_id},
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/experiments/{exp_id}/report")
async def download_report(exp_id: str, current_user: dict = Depends(get_current_user)):
    """获取实验报告（Markdown 格式）"""
    try:
        rows = db.execute_sql("SELECT * FROM experiment_runs WHERE id=:id", {"id": exp_id})
        if not rows:
            raise HTTPException(status_code=404, detail="experiment not found")
        r = rows[0]
        try:
            results = json.loads(r["results"]) if r["results"] else {}
        except Exception:
            results = {}
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
        logger.error(f"获取报告失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
