"""Background executor for bounded Gold Set generation evaluations."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from typing import Any, Mapping

logger = logging.getLogger(__name__)
_evaluation_lock: asyncio.Lock | None = None
_evaluation_lock_loop: asyncio.AbstractEventLoop | None = None
_evaluation_tasks: set[asyncio.Task[None]] = set()


def _get_evaluation_lock() -> asyncio.Lock:
    """Return the evaluator lock owned by the active application loop."""
    global _evaluation_lock, _evaluation_lock_loop
    loop = asyncio.get_running_loop()
    if _evaluation_lock is None or _evaluation_lock_loop is not loop:
        _evaluation_lock = asyncio.Lock()
        _evaluation_lock_loop = loop
    return _evaluation_lock


def _update_run(
    database: Any,
    run_id: str,
    *,
    metrics: Mapping[str, Any],
    total: int,
    breakdown: Mapping[str, int],
    note: str,
) -> None:
    database.execute_sql(
        "UPDATE gold_eval_runs "
        "SET metrics=:metrics, total_prompts=:total, "
        "category_breakdown=:breakdown, notes=:note WHERE id=:run_id",
        {
            "metrics": json.dumps(metrics, ensure_ascii=False),
            "total": total,
            "breakdown": json.dumps(breakdown, ensure_ascii=False),
            "note": note,
            "run_id": run_id,
        },
    )

async def execute_generation_evaluation(run_id: str, options: Mapping[str, Any], database: Any) -> None:
    """Run one evaluation at a time so it cannot starve interactive inference."""
    async with _get_evaluation_lock():
        try:
            from evaluation.generation_metrics import GenerationMetrics
            from evaluation.gold_set_manager import get_gold_set_manager

            prompts = await asyncio.to_thread(get_gold_set_manager().load_set)
            categories = options.get("categories") or []
            split = options.get("split") or "eval"
            if categories:
                prompts = [item for item in prompts if item.get("category") in categories]
            if split:
                prompts = [item for item in prompts if item.get("split") == split]

            requested_limit = options.get("max_prompts")
            limit = min(max(int(requested_limit or 25), 1), 50)
            prompts = prompts[:limit]
            if not prompts:
                raise RuntimeError("no evaluation prompts matched the requested filters")
            breakdown = Counter(str(item.get("category", "unknown")) for item in prompts)
            metric = GenerationMetrics()

            if options.get("mock"):
                result = metric.evaluate_mock([str(item.get("prompt", "")) for item in prompts])
            else:
                from api.generate import get_vllm_client

                client = await get_vllm_client()
                if client is None:
                    raise RuntimeError("vLLM client is unavailable")

                responses: list[str] = []
                samples: list[dict[str, str]] = []
                generation_errors = 0
                adapter_name = options.get("adapter_name") or None
                for item in prompts:
                    prompt = str(item.get("prompt", ""))
                    try:
                        reply = await client.generate(
                            messages=[{"role": "user", "content": prompt}],
                            lora_name=adapter_name,
                            temperature=0.0,
                            max_tokens=256,
                        )
                    except Exception as exc:
                        generation_errors += 1
                        logger.warning("evaluation generation failed run=%s: %s", run_id, exc)
                        reply = f"[GENERATION_ERROR] {type(exc).__name__}"
                    responses.append(reply)
                    samples.append({"prompt": prompt, "response": reply})

                result = {
                    "total_prompts": len(prompts),
                    "distinct_1": metric.distinct_n(responses, 1),
                    "distinct_2": metric.distinct_n(responses, 2),
                    "avg_repetition_rate": round(sum(metric.repetition_rate(reply) for reply in responses) / max(len(responses), 1), 4),
                    "avg_length": metric.avg_length(responses),
                    "max_repetition_ratio": round(sum(metric.max_repetition_ratio(reply) for reply in responses) / max(len(responses), 1), 4),
                    "samples": samples,
                    "mock": False,
                    "generation_errors": generation_errors,
                }

            generation_errors = int(result.get("generation_errors", 0))
            if generation_errors >= len(prompts):
                completion_note = "failed"
            elif generation_errors:
                completion_note = "completed_with_errors"
            else:
                completion_note = "completed"

            await asyncio.to_thread(
                _update_run,
                database,
                run_id,
                metrics=result,
                total=len(prompts),
                breakdown=breakdown,
                note=completion_note,
            )
        except Exception as exc:
            logger.exception("evaluation run failed run=%s", run_id)
            try:
                await asyncio.to_thread(
                    _update_run,
                    database,
                    run_id,
                    metrics={"error": str(exc), "mock": bool(options.get("mock"))},
                    total=0,
                    breakdown={},
                    note="failed",
                )
            except Exception:
                logger.exception("failed to persist evaluation failure run=%s", run_id)


def schedule_generation_evaluation(run_id: str, options: Mapping[str, Any], database: Any) -> asyncio.Task[None]:
    """Schedule the bounded evaluator from a FastAPI request handler."""
    task = asyncio.create_task(
        execute_generation_evaluation(run_id, dict(options), database),
        name=f"gold-eval-{run_id}",
    )
    _evaluation_tasks.add(task)
    task.add_done_callback(_evaluation_tasks.discard)
    return task


async def shutdown_generation_evaluations() -> None:
    """Cancel and join evaluations before database/model resources are closed."""
    tasks = list(_evaluation_tasks)
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    _evaluation_tasks.clear()