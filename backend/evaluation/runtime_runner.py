"""Bounded Gold Set evaluations with one validated request and provider snapshot."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import httpx
from services.model_usage import UsageRecordingError, account_usage

from inference.context_budget import InputBudgetError
from inference.model_manager import get_model_manager
from inference.openai_protocol import ProviderResponseError
from inference.provider_context import get_provider_context_budget
from inference.request_context import detached_request_context, resolve_request_context, use_request_context
from inference.usage import begin_completion
from inference.vllm_client import get_enabled_vllm_client

if TYPE_CHECKING:
    from db.schemas import EvalRunRequest

logger = logging.getLogger(__name__)
_evaluation_lock: asyncio.Lock | None = None
_evaluation_lock_loop: asyncio.AbstractEventLoop | None = None
_evaluation_tasks: set[asyncio.Task[None]] = set()

EVALUATION_DIR = Path(__file__).resolve().parent
RUNTIME_DATASETS = {
    "kisaki_v21": EVALUATION_DIR / "kisaki_gold_set_v21_candidates.json",
    "kisaki_v3": EVALUATION_DIR / "kisaki_gold_set_v3.json",
    "legacy_general": EVALUATION_DIR / "gold_prompts.json",
}


def load_runtime_dataset(dataset_id: str) -> dict[str, Any]:
    path = RUNTIME_DATASETS.get(dataset_id)
    if path is None:
        raise LookupError(f"unknown evaluation dataset: {dataset_id}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("prompts"), list):
        raise ValueError(f"evaluation dataset is invalid: {dataset_id}")
    return value


def conversation_turns(item: Mapping[str, Any]) -> list[str]:
    conversation = item.get("conversation")
    if isinstance(conversation, list):
        turns = [
            str(message.get("content", ""))
            for message in conversation
            if isinstance(message, Mapping)
            and message.get("role") == "user"
            and str(message.get("content", "")).strip()
        ]
        if turns:
            return turns
    turns = item.get("turns")
    if isinstance(turns, list) and turns:
        return [str(turn) for turn in turns if str(turn).strip()]
    prompt = str(item.get("prompt", "")).strip()
    return [prompt] if prompt else []


def _get_evaluation_lock() -> asyncio.Lock:
    """Return the evaluator lock owned by the active application loop."""
    global _evaluation_lock, _evaluation_lock_loop
    loop = asyncio.get_running_loop()
    if _evaluation_lock is None or _evaluation_lock_loop is not loop:
        _evaluation_lock = asyncio.Lock()
        _evaluation_lock_loop = loop
    return _evaluation_lock


def _update_run(
    database: Any, run_id: str, *, metrics: Mapping[str, Any], total: int, breakdown: Mapping[str, int], note: str
) -> None:
    database.execute_sql(
        "UPDATE gold_eval_runs SET metrics=:metrics, total_prompts=:total, "
        "category_breakdown=:breakdown, notes=:note WHERE id=:run_id",
        {
            "metrics": json.dumps(metrics, ensure_ascii=False),
            "total": total,
            "breakdown": json.dumps(breakdown, ensure_ascii=False),
            "note": note,
            "run_id": run_id,
        },
    )


def _error_details(error: BaseException) -> dict:
    if isinstance(error, (ProviderResponseError, InputBudgetError, UsageRecordingError)):
        detail = error.public_detail()
        return {"error": detail["message"], "error_type": type(error).__name__, "detail": detail}
    message = "模型请求失败，请检查连接与服务。" if isinstance(error, httpx.HTTPError) else str(error)
    return {"error": message, "error_type": type(error).__name__}


async def _generate_samples(run_id, options, database, prompts, http_client_pool):
    from evaluation.generation_metrics import GenerationMetrics, validate_generation_response
    from inference.generation_request import GenerationRequest, generate_character_response
    from inference.lora_registry import get_lora_system_prompt

    context = await resolve_request_context(get_model_manager(database=database), http_client_pool=http_client_pool)
    if context.provider_name not in {"openai_compat", "vllm"}:
        raise ValueError(f"Generation evaluation does not support selected provider: {context.provider_name}")
    adapter_name = options.adapter_name or None
    if context.provider_name == "openai_compat" and adapter_name is not None:
        raise ValueError("OpenAI-compatible evaluation does not support LoRA; clear the adapter selection")
    with use_request_context(context):
        client = await get_enabled_vllm_client() if context.provider_name == "vllm" else None
        if context.provider_name == "vllm" and client is None:
            raise RuntimeError("vLLM client is unavailable")
        model_id = context.provider.model
        budget = get_provider_context_budget()
        persona_prompt = get_lora_system_prompt(adapter_name or "kisaki")
        top_p = 1.0 if context.provider_name == "openai_compat" else 0.9

        async def generate(**kwargs):
            if context.provider_name == "openai_compat":
                text, _cost = await context.provider.async_complete(**kwargs)
                return text
            # This client returns text without provider usage. Record unknown
            # token counts, never a character-based estimate.
            begin_completion()
            return await client.generate(**kwargs)

        responses, samples = [], []
        generation_errors = 0
        for index, item in enumerate(prompts):
            turns = conversation_turns(item)
            history, turn_responses = [], []
            error = None
            reply = ""
            stop = False
            try:
                if not turns:
                    raise ValueError("evaluation prompt has no user turns")
                for turn_index, turn in enumerate(turns):
                    metadata = {
                        "traceId": f"{run_id}:{index}:{turn_index}",
                        "platform": "evaluation",
                        "sessionId": run_id,
                        "purpose": "generation_evaluation",
                        "modelName": model_id,
                        "loraName": adapter_name or "default",
                        "usedLora": adapter_name is not None,
                        "usedRag": False,
                    }
                    async with account_usage(database, metadata):
                        generated = await generate_character_response(
                            GenerationRequest(
                                message=turn,
                                persona_prompt=persona_prompt,
                                interlocutor=str(item.get("interlocutor") or "普通用户"),
                                history=history,
                                lora_name=adapter_name,
                                temperature=0.0,
                                max_tokens=256,
                                top_p=top_p,
                                context_window_tokens=budget.window_tokens,
                            ),
                            generate,
                        )
                        reply = validate_generation_response(generated.reply)
                        turn_responses.append(reply)
                        history.extend(({"role": "user", "content": turn}, {"role": "assistant", "content": reply}))
            except Exception as exc:
                generation_errors += 1
                logger.warning("evaluation generation failed run=%s type=%s", run_id, type(exc).__name__)
                error = _error_details(exc)
                reply = ""
                stop = isinstance(exc, (ProviderResponseError, httpx.HTTPError, UsageRecordingError))
            else:
                responses.append(reply)
            samples.append(
                {
                    "id": item.get("id"),
                    "turns": turns,
                    "turn_responses": turn_responses,
                    "response": reply,
                    **(error or {}),
                }
            )
            if stop:
                break
        return {
            "total_prompts": len(samples),
            "planned_prompts": len(prompts),
            "unattempted_prompts": len(prompts) - len(samples),
            **GenerationMetrics().summarize(responses),
            "samples": samples,
            "mock": False,
            "generation_errors": generation_errors,
            "provider": context.provider_name,
            "model": model_id,
        }


async def execute_generation_evaluation(
    run_id: str, options: EvalRunRequest, database: Any, *, http_client_pool=None
) -> None:
    """Own one job's selection, result publication and terminal failure state."""
    async with _get_evaluation_lock():
        result = None
        prompts = []
        try:
            from evaluation.generation_metrics import GenerationMetrics

            dataset = await asyncio.to_thread(load_runtime_dataset, options.dataset_id)
            prompts = [
                item
                for item in dataset["prompts"]
                if item.get("benchmark_suite", "character") == "character"
                and item.get("split", "eval") == options.split
                and (not options.categories or item.get("category") in options.categories)
            ]
            prompts = prompts[: options.max_prompts or 25]
            if not prompts:
                raise RuntimeError("no evaluation prompts matched the requested filters")
            if options.mock:
                result = GenerationMetrics().evaluate_mock([conversation_turns(item)[-1] for item in prompts])
                result.update(planned_prompts=len(prompts), unattempted_prompts=0, provider="mock", model=None)
            else:
                result = await _generate_samples(run_id, options, database, prompts, http_client_pool)
            result.update(
                dataset_id=options.dataset_id,
                dataset_status=dataset.get("status"),
                dataset_role=dataset.get("evaluation_role"),
            )
            errors = result["generation_errors"]
            total = result["total_prompts"]
            note = (
                "failed"
                if result["unattempted_prompts"] or errors >= total
                else "completed_with_errors"
                if errors
                else "completed"
            )
            await asyncio.to_thread(
                _update_run,
                database,
                run_id,
                metrics=result,
                total=total,
                breakdown=Counter(str(item.get("category", "unknown")) for item in prompts[:total]),
                note=note,
            )
        except (Exception, asyncio.CancelledError) as exc:
            logger.warning("evaluation run failed run=%s type=%s", run_id, type(exc).__name__)
            if result is None:
                result = {
                    "mock": options.mock,
                    "total_prompts": 0,
                    "planned_prompts": len(prompts),
                    "unattempted_prompts": len(prompts),
                }
            result.update(_error_details(exc))
            cancelled = isinstance(exc, asyncio.CancelledError)
            try:
                await asyncio.to_thread(
                    _update_run,
                    database,
                    run_id,
                    metrics=result,
                    total=result["total_prompts"],
                    breakdown=Counter(
                        str(item.get("category", "unknown")) for item in prompts[: result["total_prompts"]]
                    ),
                    note="cancelled" if cancelled else "failed",
                )
            except Exception as recording_error:
                if cancelled:
                    raise exc from recording_error
                raise
            if cancelled:
                raise


def _evaluation_finished(task):
    _evaluation_tasks.discard(task)
    if not task.cancelled() and (error := task.exception()) is not None:
        logger.error("evaluation task failed task=%s type=%s", task.get_name(), type(error).__name__)


def schedule_generation_evaluation(
    run_id: str, options: EvalRunRequest, database: Any, *, http_client_pool=None
) -> asyncio.Task[None]:
    """Freeze validated options; the application continues to own its HTTP pool."""
    task = asyncio.create_task(
        execute_generation_evaluation(
            run_id, options.model_copy(deep=True), database, http_client_pool=http_client_pool
        ),
        name=f"gold-eval-{run_id}",
        context=detached_request_context(),
    )
    _evaluation_tasks.add(task)
    task.add_done_callback(_evaluation_finished)
    return task


async def shutdown_generation_evaluations() -> None:
    """Cancel and join evaluations before database/model resources are closed."""
    tasks = list(_evaluation_tasks)
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    _evaluation_tasks.clear()
