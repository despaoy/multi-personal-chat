"""消息生成API - 支持vLLM高并发推理"""

import asyncio
import functools
import hashlib
import json
import logging
import os
import time
import uuid
from contextlib import ExitStack, contextmanager, nullcontext
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import ValidationError

from app.config import (
    circuit_breaker_registry,
    get_llm_semaphore,
    get_vllm_served_model_name,
    response_cache,
)
from app.dependencies import get_current_admin, get_current_user
from app.providers import get_vllm_client
from app.runtime import get_runtime_container
from db.adapter import db
from db.database import get_lora_path_by_id
from db.message_feedback import reply_with_warning
from db.schemas import GenerateResponse, MessageRequest
from inference.context_budget import InputBudgetError
from inference.generation_options import read_generation_options
from inference.generation_request import (
    ContextAuthorityChangedError,
    ReplyValidationError,
)
from inference.history import HistoryFormatError, validate_history
from inference.lora_registry import get_lora_character_id
from inference.lora_utils import resolve_lora_served_name
from inference.openai_protocol import ProviderResponseError, validate_completion_options, with_provider_error
from inference.usage import collect_usage, has_pending_usage, usage_record_started
from infra.circuit_breaker import CircuitOpenError
from infra.concurrency_control import InferenceQueueFull, RateLimitExceeded, inference_runtime
from infra.environment import read_bool
from infra.input_validator import MESSAGE_SCHEMA, InputValidator
from infra.observability import increment, log_event, set_consecutive
from services import chat_answer
from services.chat_generation import ChatGenerationService
from services.model_usage import write_invocation

logger = logging.getLogger(__name__)

_DB_WRITE_TIMEOUT = float(os.getenv("DB_WRITE_TIMEOUT", "3"))
_MODEL_INFERENCE_TIMEOUT = float(os.getenv("MODEL_INFERENCE_TIMEOUT", "180"))
_DETERMINISTIC_MODEL_LABELS = {
    "current_constraint": "statement/constraint",
    "memory_lookup": "memory/lookup",
    "memory_storage_status": "memory/storage_status",
    "memory_mentions": "memory/mentions",
    "memory_field_result": "memory/field_result",
    "memory_operation": "memory/operation",
    "task_composite": "tasks/composite",
    "source_excerpt": "rag/source_excerpt",
    "source_unavailable": "rag/abstained",
    "abstention": "rag/abstained",
}
_local_model_lock: asyncio.Lock | None = None
_local_model_lock_loop: asyncio.AbstractEventLoop | None = None

_HIGH_RISK_PROMPT_PATTERNS = (
    "export config",
    "dump config",
    "show config",
    "read .env",
    "cat .env",
    "read secret",
    "show secret",
    "read token",
    "show token",
    "print env",
    "reveal system prompt",
    "show system prompt",
    "ignore previous instructions and export",
    "\u5bfc\u51fa\u914d\u7f6e",
    "\u8bfb\u53d6\u914d\u7f6e",
    "\u663e\u793a\u914d\u7f6e",
    "\u8bfb\u53d6\u5bc6\u94a5",
    "\u663e\u793a\u5bc6\u94a5",
    "\u8bfb\u53d6token",
    "\u663e\u793atoken",
    "\u8bfb\u53d6.env",
    "\u5ffd\u7565\u4e4b\u524d\u6307\u4ee4\u5e76\u5bfc\u51fa",
)


def _get_local_model_lock() -> asyncio.Lock:
    """Serialize stateful local model access within the active event loop."""
    global _local_model_lock, _local_model_lock_loop
    loop = asyncio.get_running_loop()
    if _local_model_lock is None or _local_model_lock_loop is not loop:
        _local_model_lock = asyncio.Lock()
        _local_model_lock_loop = loop
    return _local_model_lock


def _is_high_risk_prompt(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in _HIGH_RISK_PROMPT_PATTERNS)


def _security_policy_response() -> GenerateResponse:
    return GenerateResponse(
        reply="\u8be5\u8bf7\u6c42\u6d89\u53ca\u7cfb\u7edf\u914d\u7f6e\u3001\u51ed\u636e\u6216\u5185\u90e8\u6307\u4ee4\uff0c\u5df2\u88ab\u5b89\u5168\u7b56\u7565\u62e6\u622a\u3002",
        model="security-policy",
        costTime=0.0,
    )


router = APIRouter()


class _ModelIdentityError(RuntimeError):
    """The invoked provider did not supply a usable model identity."""


def _response_cache_keys(
    request: MessageRequest,
    lora_name: str,
    config: dict[str, Any],
    *,
    enable_rag: bool = True,
    model_identity: dict[str, Any],
) -> tuple[str, str, int]:
    """Include every response-affecting setting in the cache identity.

    注意：带角色上下文（人物记忆/关系）的对话整体绕过响应缓存，
    不会走到本函数——有状态对话每轮都要回写记忆与关系，缓存命中
    会跳过回写导致状态与对话脱节。
    """
    options = read_generation_options(config)
    use_knowledge_base = read_bool(config, "useKnowledgeBase", True) and enable_rag
    interlocutor = request.senderName or request.userName or ""
    prompt_identity = {
        "message": request.message,
        "history": list(request.history or []),
        "interlocutor": interlocutor,
        "enable_rag": enable_rag,
    }
    identity = {
        "model": model_identity,
        "generation_contract": "compiled-native-v1",
        "lora": lora_name,
        "temperature": options["temperature"],
        "max_tokens": options["maxTokens"],
        "top_p": options["topP"],
        "use_knowledge_base": use_knowledge_base,
        "enable_rag": enable_rag,
        "platform": request.platform,
        "conversation_type": request.conversationType or request.sessionType,
        "conversation_id": request.conversationId or request.sessionId,
        "interlocutor": interlocutor,
    }
    prompt_hash = hashlib.sha256(
        json.dumps(prompt_identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    serialized = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    cache_key = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return prompt_hash, cache_key, 60 if use_knowledge_base else 300


async def _generate_reply_impl(
    request: MessageRequest,
    current_user: dict | None = None,
    *,
    persist_message: bool = True,
    enable_rag: bool = True,
    record_invocation: bool = True,
    invocation_purpose: str = "chat",
    character_service=None,
    message_db,
    delivery_context: dict | None = None,
    prepared_override=None,
):
    from inference.model_manager import get_model_manager
    from inference.request_context import get_request_context, resolve_request_context, use_request_context

    context = get_request_context()
    if context is None:
        try:
            context = await resolve_request_context(get_model_manager())
        except Exception as exc:
            raise HTTPException(503, "模型配置读取失败，请检查配置及模型状态后重试") from exc
    started_at = time.time()
    with ExitStack() as turn_scope, use_request_context(context), collect_usage():
        try:
            return await _generate_reply_body(
                request,
                current_user,
                persist_message=persist_message,
                enable_rag=enable_rag,
                record_invocation=record_invocation,
                invocation_purpose=invocation_purpose,
                character_service=character_service,
                message_db=message_db,
                delivery_context=delivery_context,
                prepared_override=prepared_override,
                _turn_scope=turn_scope,
            )
        except (Exception, asyncio.CancelledError) as error:
            if record_invocation and has_pending_usage():
                try:
                    await _record_model_invocation(
                        request,
                        context.provider.model,
                        "default",
                        round(time.time() - started_at, 2),
                        error_type=type(error.__cause__ or error).__name__,
                        database=message_db,
                        purpose=invocation_purpose,
                    )
                except HTTPException as record_error:
                    if isinstance(error, asyncio.CancelledError):
                        raise error from record_error
                    detail = with_provider_error(record_error.detail, error)
                    if isinstance(error, HTTPException):
                        detail = dict(detail, requestError=error.detail)
                    raise HTTPException(503, detail=detail) from error
            raise


@contextmanager
def _report_memory_operation(prepared):
    """Preserve a committed operation receipt on later API failures."""
    try:
        yield
    except HTTPException as error:
        if prepared.memory_operation_receipt is not None:
            from character.memory_operation import operation_receipt_context

            receipt = operation_receipt_context(prepared.memory_operation_receipt)
            detail = error.detail if isinstance(error.detail, dict) else {"message": error.detail}
            error.detail = {
                **detail,
                "memoryOperation": receipt,
                "message": f"{detail['message']}\n请核对本轮记忆操作回执：\n{receipt}",
            }
        raise


async def _invoke_model(generate, **kwargs):
    """Count actual generation outcomes independently of later turn processing."""
    try:
        result = await generate(**kwargs)
    except CircuitOpenError:
        raise
    except Exception:
        increment("model_failures")
        set_consecutive("model_failure", False)
        raise
    set_consecutive("model_failure", True)
    return result


async def _generate_reply_body(
    request: MessageRequest,
    current_user: dict | None = None,
    *,
    persist_message: bool = True,
    enable_rag: bool = True,
    record_invocation: bool = True,
    invocation_purpose: str = "chat",
    character_service=None,
    message_db,
    delivery_context: dict | None = None,
    prepared_override=None,
    _turn_scope: ExitStack,
):
    """Generate using the accepted request's database and provider snapshot.

    Context reads, invocation records and message writes share message_db;
    this business path never selects another database.
    """
    try:
        validate_history(request.history)
    except HistoryFormatError as exc:
        raise HTTPException(503, detail=exc.public_detail()) from exc
    if (
        prepared_override is None
        and not request.branchId
        and (request.adapter == "narrative" or request.adapter.startswith("narrative:"))
    ):
        # This namespace is owned by authenticated workspace routes. Letting
        # generic callers choose its senderId would poison another owner's history.
        raise HTTPException(422, "请通过假想分支工作区访问该会话")
    if request.branchId and prepared_override is None:
        from app.dependencies import require_narrative_enabled

        require_narrative_enabled()
        if delivery_context is not None or not persist_message or request.platform != "web":
            raise HTTPException(422, "假想分支第一版仅支持 Web 交互")
        from services.narrative import generate_branch_reply

        return await generate_branch_reply(
            request,
            current_user,
            message_db,
            generate=_generate_reply_impl,
        )

    # C11 fix: 主聊天端点 mock provider 防护
    # 此前仅 training/generate-dialogues 有此检查，主聊天端点遗漏，
    # 导致生产环境默认 mock 时静默返回罐头回复。现统一拦截。
    from inference.request_context import get_request_context

    request_context = get_request_context()
    request_provider = request_context.provider_name
    if request_provider == "mock":
        raise HTTPException(
            status_code=503,
            detail="当前模型提供商为 mock 模式，无法提供真实推理。"
            "请在设置页面配置有效的模型提供商（如 vLLM / DeepSeek API / Ollama）。",
        )

    # 输入验证
    is_valid, errors = InputValidator.validate({"message": request.message}, MESSAGE_SCHEMA)
    if not is_valid:
        raise HTTPException(status_code=422, detail={"message": "输入验证失败", "errors": errors})

    select_active_lora = not request.loraName and not request.characterId and prepared_override is None
    try:
        if select_active_lora or request.loraName not in {"", "default"}:
            runtime_config, loras = await asyncio.gather(
                asyncio.to_thread(lambda: message_db.config),
                asyncio.to_thread(lambda: list(message_db.loras)),
            )
        else:
            runtime_config = await asyncio.to_thread(lambda: message_db.config)
            loras = []
        runtime_config = {**runtime_config}
    except Exception as exc:
        raise HTTPException(503, "生成配置读取失败，请检查数据库连接后重试") from exc

    if request_provider == "openai_compat":
        runtime_config.setdefault("topP", 1.0)
    try:
        runtime_config.update(read_generation_options(runtime_config))
        if request_provider == "openai_compat":
            validate_completion_options(request_context.provider.base_url, top_p=runtime_config["topP"])
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            503, "生成参数配置无效，请检查 temperature、maxTokens、topP 及当前模型服务要求后重试"
        ) from exc

    selected_lora = None
    if request.loraName and request.loraName != "default":
        selected_lora = next((item for item in loras if item["name"] == request.loraName), None)
        if selected_lora is None:
            raise HTTPException(status_code=422, detail="Specified LoRA does not exist")
    elif select_active_lora:
        selected_lora = next((item for item in loras if item["status"] == "active"), None)
    lora_name = selected_lora["name"] if selected_lora else "default"

    if selected_lora and request_provider not in {"vllm", "transformers_peft"}:
        raise HTTPException(422, "所选模型提供商不支持 LoRA，请选择基础模型或显式选择支持 LoRA 的提供商")
    vllm_lora_name = resolve_lora_served_name(lora_name) if lora_name != "default" else "default"

    # ── 角色上下文准备（仅当 LoRA 显式映射了人物画像时启用） ──
    # 未映射的 LoRA（如 hutao/minamo）与 default 保持原有生成行为。
    prepared_character_turn = prepared_override
    mapped_character_id = get_lora_character_id(lora_name) if lora_name != "default" else None
    if request.characterId:
        from character.profile_registry import CharacterProfileNotFoundError, get_default_profile_registry

        try:
            get_default_profile_registry().get_profile(request.characterId)
        except CharacterProfileNotFoundError:
            raise HTTPException(422, "所选人物不存在") from None
        if lora_name != "default" and mapped_character_id != request.characterId:
            raise HTTPException(422, "LoRA 与所选人物不匹配，请选择基础模型或对应 LoRA")
        if prepared_override is not None and prepared_override.character_id != request.characterId:
            raise HTTPException(422, "人物与当前分支不匹配")
        mapped_character_id = request.characterId
    if prepared_override is not None and request.loraName and mapped_character_id != prepared_override.character_id:
        raise HTTPException(422, "LoRA 与分支角色不匹配")
    completion_slot = None
    if (
        (mapped_character_id or prepared_override is not None)
        and persist_message
        and delivery_context is None
        and not request.branchId
    ):
        from services.turn_completion import CompletionUnavailable, get_turn_completion_runtime

        try:
            completion_slot = get_turn_completion_runtime().reserve()
        except CompletionUnavailable:
            raise HTTPException(
                503,
                detail={
                    "code": "turn_completion_busy",
                    "message": "长期记忆回写繁忙，本轮尚未生成，请稍后重试。",
                },
            ) from None
        _turn_scope.callback(completion_slot.release)

    if mapped_character_id and prepared_override is None:
        if character_service is None:
            from services.character_context import (
                build_character_context_service,
                get_default_character_context_service,
            )

            try:
                character_service = (
                    get_default_character_context_service()
                    if message_db is db
                    else build_character_context_service(message_db)
                )
            except Exception as exc:
                raise HTTPException(503, "角色上下文配置读取失败，请检查上下文及审核配置后重试") from exc
        prepared_character_turn = await _prepare_character_turn(
            request,
            mapped_character_id,
            character_service=character_service,
            execute_memory_operations=persist_message and delivery_context is None,
            defer_memory_operations=persist_message and delivery_context is not None and not request.branchId,
        )
    if prepared_character_turn is not None:
        mapped_character_id = prepared_character_turn.character_id
        _turn_scope.enter_context(_report_memory_operation(prepared_character_turn))

    # 检查vLLM是否实际支持该LoRA，避免404触发熔断
    vllm_effective_lora = vllm_lora_name if lora_name != "default" else None
    if request_provider == "vllm" and vllm_effective_lora:
        try:
            client = await get_vllm_client()
            if client is None:
                raise RuntimeError("Selected vLLM service is unavailable")
            available_loras = await client.list_loras()
            if vllm_effective_lora not in available_loras:
                raise HTTPException(409, "所选 LoRA 尚未加载到 vLLM，请先重新激活")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(503, "所选 LoRA 状态查询失败，请检查 vLLM 服务及适配器后重试") from exc

    memory_write_enabled = None
    if prepared_character_turn is not None:
        from character.memory_llm import get_memory_enrichment_scheduler

        memory_write_enabled = bool(
            persist_message and not request.branchId and get_memory_enrichment_scheduler().enabled
        )

    completion_warning = None
    start_time = time.time()
    prompt_hash = cache_key = ""
    cache_ttl = 300
    # 带角色上下文的对话是有状态流程（每轮回写记忆/关系），整体绕过
    # 响应缓存：缓存命中会跳过回写，导致人物状态与对话脱节。
    use_response_cache = (
        response_cache is not None
        and prepared_character_turn is None
        and selected_lora is None
        and not (read_bool(runtime_config, "useKnowledgeBase", True) and enable_rag)
    )
    provider = request_context.provider
    if use_response_cache:
        try:
            model_identity = {
                "provider": request_provider,
                "status": provider.get_status(),
                "endpoint": getattr(provider, "base_url", None),
                "credential_sha256": hashlib.sha256(getattr(provider, "api_key", "").encode()).hexdigest(),
            }
        except Exception as exc:
            raise HTTPException(503, "模型配置读取失败，请检查配置及模型状态后重试") from exc
        try:
            prompt_hash, cache_key, cache_ttl = _response_cache_keys(
                request,
                lora_name,
                runtime_config,
                enable_rag=enable_rag,
                model_identity=model_identity,
            )
            cached = await response_cache.get(prompt_hash, cache_key)
            response = GenerateResponse.model_validate(cached) if cached is not None else None
        except ValidationError as exc:
            raise HTTPException(
                503,
                detail={
                    "stage": "response_cache",
                    "reason": "invalid_cached_response",
                    "fields": list(
                        dict.fromkeys(
                            str(error["loc"][0]) if error["loc"] else "$" for error in exc.errors(include_input=False)
                        )
                    ),
                    "message": "缓存的生成结果缺少必要字段或内容无效，请清理该条缓存后重试。",
                },
            ) from exc
        except Exception as exc:
            raise HTTPException(503, "回复缓存读取失败，请检查缓存状态后重试") from exc
        if response is not None:
            logger.debug("response cache hit")
            if persist_message:
                stored_model_name = "vllm" if response.model.startswith("vllm/") else response.model
                await _save_message(
                    request,
                    response.reply,
                    stored_model_name,
                    lora_name,
                    round(time.time() - start_time, 2),
                    database=message_db,
                    character_id=mapped_character_id,
                    model_invoked=False,
                )
            return response.model_copy(
                update={
                    "modelInvoked": False,
                    "costTime": round(time.time() - start_time, 2),
                    "warnings": list(dict.fromkeys([*(response.warnings or []), "cached_model_response"])),
                }
            )

    generate_answer = functools.partial(
        chat_answer.generate_chat_answer,
        request,
        runtime_config=runtime_config,
        enable_rag=enable_rag,
        prepared_character_turn=prepared_character_turn,
        message_db=message_db,
        memory_write_enabled=memory_write_enabled,
    )
    is_vllm = request_provider == "vllm"
    model_name = f"vllm/{get_vllm_served_model_name()}" if is_vllm else "model_manager"
    message_saved = False
    model_invoked = None
    try:
        if is_vllm:
            client = await get_vllm_client()
            if client is None:
                raise HTTPException(503, "所选 vLLM 服务不可用，请恢复服务后重试")
            answer = await generate_answer(
                vllm_effective_lora,
                prompt_lora_name=vllm_lora_name,
                model_generate=functools.partial(_invoke_model, client.generate),
            )
        else:
            semaphore = get_llm_semaphore()
            try:
                await asyncio.wait_for(semaphore.acquire(), timeout=30.0)
            except TimeoutError as exc:
                raise HTTPException(503, "服务繁忙，请稍后再试") from exc
            try:
                async with nullcontext() if request_provider == "openai_compat" else _get_local_model_lock():
                    try:
                        lora_path = get_lora_path_by_id(selected_lora["id"], loras=loras) if selected_lora else None
                    except OSError as exc:
                        raise HTTPException(503, "所选 LoRA 目录读取失败，请检查适配器存储后重试") from exc
                    if selected_lora and lora_path is None:
                        raise HTTPException(409, "所选 LoRA 没有可用本地目录，请先重新激活")
                    if request_provider != "openai_compat":
                        provider.set_lora_adapter(lora_path)

                    async def adapter(*, messages, **kwargs):
                        nonlocal model_name
                        breaker = (
                            await circuit_breaker_registry.get_or_create(
                                f"model_generate:{request_context.connection_id}"
                            )
                            if circuit_breaker_registry
                            else None
                        )
                        async with breaker() if breaker is not None else nullcontext():
                            if request_provider == "openai_compat":
                                model_name = provider.get_status()["modelName"]
                            if request_provider == "transformers_peft":
                                # The local adapter was bound above under this same lock.
                                kwargs["lora_name"] = None
                            text, _cost = await _invoke_model(provider.async_complete, messages=messages, **kwargs)
                        return text

                    answer = await asyncio.wait_for(
                        generate_answer(lora_name, model_generate=adapter),
                        timeout=_MODEL_INFERENCE_TIMEOUT,
                    )
            finally:
                semaphore.release()
        model_invoked = answer.model_invoked
        reply, used_rag, rag_meta = answer.reply, answer.used_rag, answer.metadata
        if is_vllm:
            model_label = model_name
            stored_model_name = "vllm"
        else:
            if model_invoked:
                try:
                    name = provider.get_status()["modelName"]
                    if not isinstance(name, str) or not name.strip():
                        raise ValueError("Selected provider must report a nonempty model name")
                    model_name = name
                except Exception as exc:
                    raise _ModelIdentityError("Selected provider model identity is unavailable") from exc
            model_label = f"{model_name} ({request_provider})"
            stored_model_name = model_name

        cost_time = round(time.time() - start_time, 2)
        if not model_invoked:
            try:
                model_name = _DETERMINISTIC_MODEL_LABELS[rag_meta.get("answerMode")]
            except (KeyError, TypeError) as exc:
                raise HTTPException(
                    503,
                    detail={
                        "stage": "generation_result",
                        "reason": "unknown_response_mode",
                        "modelInvoked": False,
                        "messageSaved": False,
                        "message": "生成结果类型缺失或未受支持，本轮结果未保存。请检查生成链路后重试。",
                    },
                ) from exc
            model_label = stored_model_name = model_name
        stored_lora_name = lora_name if model_invoked else "default"
        result = GenerateResponse(
            reply=reply,
            model=model_label,
            costTime=cost_time,
            citations=rag_meta.get("citations"),
            confidence=rag_meta.get("confidence"),
            abstained=rag_meta.get("abstained", False),
            answerMode=rag_meta.get("answerMode"),
            domainId=rag_meta.get("domainId"),
            modelInvoked=model_invoked,
            warnings=rag_meta.get("warnings"),
        )
        if record_invocation and (model_invoked or has_pending_usage()):
            await _record_model_invocation(
                request,
                model_name if model_invoked else provider.model,
                stored_lora_name,
                cost_time,
                used_rag=used_rag,
                error_type="",
                model_invoked=model_invoked,
                database=message_db,
                purpose=invocation_purpose,
            )
        archive_receipt = {}
        if persist_message:
            await _save_message(
                request,
                reply,
                stored_model_name,
                stored_lora_name,
                cost_time,
                database=message_db,
                character_id=mapped_character_id,
                saved_receipt=archive_receipt,
                model_invoked=model_invoked,
            )
            message_saved = True
        # Required archive failure has already stopped this turn. Internal
        # generation does not archive messages or update character memory.
        if delivery_context is not None:
            delivery_context.update(message_saved=message_saved, character_id=mapped_character_id)
            if prepared_character_turn is not None and message_saved:
                from services.delivery_memory import freeze_completion

                delivery_context["completion_snapshot"] = freeze_completion(prepared_character_turn)
        if prepared_character_turn is not None and persist_message and delivery_context is None:
            try:
                completion_warning = await _complete_character_turn(
                    prepared_character_turn,
                    request,
                    character_service=character_service,
                    completion_slot=completion_slot,
                )
            except Exception as exc:
                logger.warning("回写处理失败 traceId=%s type=%s", request.traceId, type(exc).__name__)
                raise HTTPException(
                    503,
                    detail={
                        "stage": "turn_completion",
                        "reason": "completion_failed",
                        "modelInvoked": model_invoked,
                        "messageSaved": message_saved,
                        "message": "回复已存档，但后续回写未能确认完成。请核对聊天与记忆记录后再重试。",
                    },
                ) from exc
            if completion_warning:
                stored = await _persist_completion_feedback(
                    archive_receipt,
                    completion_warning,
                    database=message_db,
                )
                if not stored:
                    completion_warning += " 保存提示未能写入聊天历史，刷新后可能看不到本次保存状态。"
        log_event(
            "message_generated",
            traceId=request.traceId,
            platform=request.platform,
            conversationId=request.conversationId or request.sessionId,
            senderId=request.senderId or request.userId,
            model=model_label,
            costTime=cost_time,
            errorType="",
            usedRag=used_rag,
        )

        if completion_warning:
            result.reply = reply_with_warning(result.reply, completion_warning)
            result.warnings = [*(result.warnings or []), completion_warning]
        if use_response_cache:
            try:
                await response_cache.set(prompt_hash, cache_key, result.model_dump(), ttl=cache_ttl)
            except Exception as exc:
                raise HTTPException(
                    503,
                    detail={
                        "stage": "response_cache",
                        "message": "回复缓存写入失败，请检查缓存状态后重试",
                        "modelInvoked": model_invoked,
                        "messageSaved": message_saved,
                    },
                ) from exc

        return result
    except chat_answer.ChatRetrievalError as exc:
        raise HTTPException(exc.http_status, detail=exc.public_detail()) from exc.__cause__
    except HTTPException:
        raise
    except Exception as e:
        failed_cost = round(time.time() - start_time, 2)
        budget_failure = isinstance(e, InputBudgetError)
        log_event(
            "generation_rejected" if budget_failure else "generation_failed",
            level="warning" if budget_failure else "error",
            traceId=request.traceId,
            platform=request.platform,
            conversationId=request.conversationId or request.sessionId,
            senderId=request.senderId or request.userId,
            model=model_name,
            costTime=failed_cost,
            errorType=type(e).__name__,
        )
        if record_invocation and not usage_record_started() and (model_invoked is not False or has_pending_usage()):
            try:
                await _record_model_invocation(
                    request,
                    provider.model if model_invoked is False else model_name,
                    lora_name,
                    failed_cost,
                    used_rag=False,
                    error_type=type(e).__name__,
                    model_invoked=model_invoked,
                    database=message_db,
                    purpose=invocation_purpose,
                )
            except HTTPException as record_error:
                # Keep the model cause and recording error on separate chains.
                raise HTTPException(503, detail=with_provider_error(record_error.detail, e)) from e
        if isinstance(e, ValidationError):
            raise HTTPException(
                503,
                detail={
                    "stage": "generation_result",
                    "reason": "invalid_generation_result",
                    "messageSaved": message_saved,
                    "fields": sorted({str(error["loc"][0]) for error in e.errors(include_input=False) if error["loc"]}),
                    "message": "生成结果结构无效，本轮结果未保存。请检查生成及检索返回格式后重试。",
                },
            ) from e
        if isinstance(e, (HistoryFormatError, InputBudgetError)):
            raise HTTPException(503, detail=e.public_detail()) from e
        if isinstance(e, _ModelIdentityError):
            raise HTTPException(
                503,
                detail={
                    "stage": "model_status",
                    "reason": "model_identity_unavailable",
                    "modelInvoked": True,
                    "messageSaved": False,
                    "message": "生成后模型标识读取失败，本轮结果未保存。请检查所选模型配置后重试。",
                },
            ) from e
        if isinstance(e, ProviderResponseError):
            raise HTTPException(e.http_status, detail=e.public_detail()) from e
        if isinstance(e, CircuitOpenError):
            raise HTTPException(
                503,
                detail={
                    "stage": "model_circuit",
                    "reason": "circuit_open",
                    "message": "模型服务连续失败，暂时停止调用。请恢复所选模型服务后重试。",
                    "modelInvoked": False,
                },
            ) from e
        if isinstance(e, ContextAuthorityChangedError):
            raise HTTPException(
                503,
                detail={
                    "stage": "context_authority",
                    "reason": "repeated_authority_change",
                    "message": "本轮资料在生成期间持续变化，未返回答案，请稍后重新请求。",
                },
            ) from e
        if isinstance(e, ReplyValidationError):
            raise HTTPException(
                503,
                detail={
                    "stage": "reply_validation",
                    "reason": "model_output_invalid",
                    "violations": list(e.violations),
                    "message": "模型回复经一次修正仍未通过校验，本轮未返回回复。请重试或检查模型与输出策略。",
                },
            ) from e
        # 安全：不把内部异常字符串返回给客户端（信息泄露），
        # 日志记录错误类型；异常链保留内部原因，客户端收到通用消息。
        raise HTTPException(
            503 if is_vllm else 500,
            detail={
                "stage": "generation",
                "messageSaved": message_saved,
                "message": (
                    "回复已存档，但后续处理失败。请核对聊天记录。"
                    if message_saved
                    else "生成回复失败，请检查所选模型服务后重试"
                ),
            },
        ) from e


# ═══════════════════════════════════════════
# 角色上下文辅助函数
# ═══════════════════════════════════════════


def _character_turn_input(request: MessageRequest):
    """Map one API request consistently for context reads and completion writes."""
    from services.character_context import TurnInput

    return TurnInput(
        message=request.message,
        platform=request.platform,
        adapter=request.adapter,
        sender_id=request.senderId or request.userId,
        conversation_id=request.conversationId or request.sessionId,
        conversation_type=request.conversationType or request.sessionType,
        history=tuple(request.history or []),
        received_at=request._source_received_at,
    )


async def _prepare_character_turn(
    request: MessageRequest,
    character_id: str,
    *,
    character_service=None,
    execute_memory_operations: bool = False,
    defer_memory_operations: bool = False,
):
    """Prepare the selected character; failures stop generation at this API boundary.

    HTTP callers inject the application-scoped service. Direct non-HTTP callers
    use the default service when none was supplied.
    """
    from character.context_builder import MemoryContextBudgetError
    from character.event_memory import EventMemoryReadError
    from character.memory_read_authority import MemoryRecordError

    try:
        from services.character_context import (
            ContextAfterMemoryOperationError,
            get_default_character_context_service,
        )

        turn_input = _character_turn_input(request)
        service = character_service or get_default_character_context_service()
        if execute_memory_operations:
            from character.memory_llm import is_memory_erasure_request

            if is_memory_erasure_request(request.message):
                return await service.prepare_interactive_turn(
                    turn_input, character_id, source_message_id=request.sourceMessageId
                )
        prepared = await service.prepare_turn(turn_input, character_id)
        if defer_memory_operations and not execute_memory_operations:
            from dataclasses import replace

            from character.memory_extractor import memory_write_allowed
            from character.memory_llm import is_memory_erasure_request

            if memory_write_allowed(request.message) and is_memory_erasure_request(request.message):
                # A server-owned delivery plan is not an execution receipt.
                # Keep the latter empty so the actual post-delivery writer runs.
                prepared = replace(
                    prepared,
                    compiled=replace(
                        prepared.compiled,
                        memory_operation_deferred=True,
                    ),
                )
        return prepared
    except ContextAfterMemoryOperationError as exc:
        raise HTTPException(
            503,
            detail=with_provider_error(
                {
                    "code": "context_failed_after_memory_operation",
                    "message": "角色上下文准备失败，回复未生成。请先核对本轮记忆操作回执，再检查审核模型和上下文配置。",
                    "memoryOperation": exc.operation_summary,
                },
                exc,
            ),
        ) from exc
    except (HistoryFormatError, InputBudgetError) as exc:
        raise HTTPException(503, detail=exc.public_detail()) from exc
    except MemoryContextBudgetError as exc:
        raise HTTPException(
            503,
            detail={
                "stage": "memory_context",
                "reason": "selected_evidence_budget_exceeded",
                "message": "已选必要记忆无法完整放入当前上下文，本轮未生成。请缩小问题范围或增加上下文预算。",
            },
        ) from exc
    except EventMemoryReadError as exc:
        raise HTTPException(
            503,
            detail={
                "stage": "memory_read",
                "reason": "invalid_event_record",
                "field": exc.field,
                "message": "已保存的事件记录格式错误，本轮未生成。请检查并修复对应记忆字段后重试。",
            },
        ) from exc
    except MemoryRecordError as exc:
        raise HTTPException(
            503,
            detail={
                "stage": "memory_read",
                "reason": "invalid_memory_record",
                "field": exc.field,
                "message": "已保存的记忆记录格式错误，本轮未生成。请检查并修复对应记忆字段后重试。",
            },
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            503,
            with_provider_error(
                "角色上下文准备失败，本轮未生成。请检查数据库、审核模型及上下文配置后重试。",
                exc,
            ),
        ) from exc


async def _persist_completion_feedback(receipt: dict, warning: str, *, database) -> bool:
    """Finalize only the exact row returned by successful message storage."""
    writer = getattr(database, "update_message_feedback", None)
    if not receipt or not callable(writer):
        return False
    try:
        return bool(
            await asyncio.wait_for(
                asyncio.to_thread(writer, receipt, warning=warning),
                timeout=_DB_WRITE_TIMEOUT,
            )
        )
    except Exception:
        # Preserve visible current feedback even when history storage is
        # unknown; never regenerate the model or replace an unrelated row.
        logger.warning("保存提示存档未能确认", exc_info=True)
        return False


async def _complete_character_turn(
    prepared, request: MessageRequest, *, character_service=None, completion_slot=None
) -> str | None:
    """生成成功后回写；保留已生成内容，向调用方返回独立的保存提示。

    character_service 语义同 _prepare_character_turn：必须与准备阶段
    使用同一服务实例，保证读写同一（容器）数据库。
    """
    from services.turn_completion import CompletionWaitTimeout, get_turn_completion_runtime

    try:
        from services.character_context import (
            get_default_character_context_service,
        )

        turn_input = _character_turn_input(request)
        service = character_service or get_default_character_context_service()

        outcome = await get_turn_completion_runtime().run(
            lambda: service.complete_turn(
                prepared,
                turn_input,
                source_message_id=request.sourceMessageId,
            ),
            timeout=_DB_WRITE_TIMEOUT,
            reservation=completion_slot,
        )
        capture = outcome.source_capture
        status = outcome.memory_enrichment_status
        warning = None
        if status == "partial":
            warning = "这条信息的长期记忆仅部分保存，部分写入失败；请检查已保存内容后再重试。"
        elif capture == "recorded" and status == "failed":
            warning = "原文已保存，但长期记忆处理失败；本次回复不代表结构化记忆已写入，请稍后重试。"
        elif capture == "failed" or status == "failed":
            warning = "这条信息的长期记忆保存失败；本次回复不代表已保存，请稍后重试。"
        elif capture in {"revoked", "stale"}:
            warning = "这条信息未保存到长期记忆：相关记忆已删除，本次回复不会恢复它。"
        elif capture == "conflict":
            warning = "这条信息未保存到长期记忆：同一条消息已绑定其他内容，请重新发送。"
        elif status in {"capacity", "closed", "not_scheduled"}:
            warning = (
                "原文已保存，但" if capture == "recorded" else ""
            ) + "长期记忆任务尚未受理；本次回复不代表结构化记忆已写入，请稍后重试。"
        elif status == "disabled":
            warning = "自动长期记忆未启用；本次回复不代表已保存。"
        elif status in {"pending", "cancelled", "unknown"}:
            warning = "本轮长期记忆操作的结果尚未确认；请稍后检查，避免重复提交。"
        elif status == "conflict":
            warning = "长期记忆记录在处理期间发生变化，本轮操作未完成；请检查后重试。"
        elif capture not in {"", "recorded", "erase_request"} or status not in {
            "saved",
            "erased",
            "no_change",
            "source_only",
            "relationship_note_saved",
            "skipped_fiction_or_note",
            "skipped",
            "queued_hot",
            "buffered_idle",
            "write_gate",
        }:
            raise ValueError("Unrecognized memory completion result")
        failures = outcome.failed_writes
        if failures:
            labels = {"interaction_count": "交互统计", "preferred_address": "称呼更新"}
            detail = "、".join(labels[stage] for stage in failures)
            partial = f"部分回写失败（{detail}）；已完成的保存不会撤销，请检查后重试。"
            warning = partial + (warning or "")
        return warning
    except CompletionWaitTimeout:
        # The runtime still owns this same task; elapsed waiting proves no
        # rollback and does not authorize submitting the work again.
        return "这条信息的长期记忆保存状态尚未确认；本次回复不代表已保存，请稍后检查。"


# ═══════════════════════════════════════════
# vLLM 推理辅助函数
# ═══════════════════════════════════════════


async def _validate_web_source_identity(request, current_user, *, database):
    """Reject known immutable source collisions before any model or write.

    The writer's transactional capture guard remains authoritative. This
    admission check uses only the authenticated owner's exact source scope and
    returns state without exposing revoked source text.
    """
    if (
        request.branchId
        or not request.characterId
        or request.platform != "web"
        or request.adapter != "web-character"
        or not request.sourceMessageId
    ):
        return
    from character.models import build_user_scope

    identity = str(
        (current_user or {}).get("id")
        or (current_user or {}).get("user_id")
        or (current_user or {}).get("username")
        or ""
    )
    if not identity:
        raise HTTPException(401, "无法确认当前登录用户")
    try:
        scope = build_user_scope(
            platform="web",
            adapter="web-character",
            sender_id=identity,
            conversation_id=request.conversationId or request.sessionId,
            conversation_type=request.conversationType or request.sessionType,
        )
    except ValueError as exc:
        raise HTTPException(422, "无法确认消息来源范围") from exc
    if not request.message:
        raise HTTPException(422, "消息不能为空")
    try:
        status = await asyncio.wait_for(
            asyncio.to_thread(
                database.memory_source_admission,
                request.characterId,
                scope.platform,
                scope.adapter,
                scope.sender_id,
                scope.conversation_type,
                scope.conversation_id,
                source_message_id=request.sourceMessageId,
                body=request.message,
            ),
            timeout=_DB_WRITE_TIMEOUT,
        )
    except Exception as exc:
        raise HTTPException(503, "暂时无法核对消息来源，请稍后重试") from exc
    if status in {"new", "pending"}:
        from character.memory_extractor import memory_write_allowed
        from character.memory_llm import is_memory_erasure_request

        # Opt-out, credential and explicit erasure messages create no binding.
        if memory_write_allowed(request.message) and not is_memory_erasure_request(request.message):
            try:
                receipt = await asyncio.wait_for(
                    asyncio.to_thread(
                        database.reserve_memory_source,
                        request.characterId,
                        scope.platform,
                        scope.adapter,
                        scope.sender_id,
                        scope.conversation_type,
                        scope.conversation_id,
                        source_message_id=request.sourceMessageId,
                        body=request.message,
                        observed_at=datetime.now(timezone.utc),
                    ),
                    timeout=_DB_WRITE_TIMEOUT,
                )
                status = receipt["status"]
                if status == "pending":
                    observed = datetime.fromisoformat(receipt["observed_at"])
                    if observed.tzinfo is None or observed.utcoffset() is None:
                        raise ValueError("Untrusted source receipt")
                    request._source_received_at = observed
            except Exception as exc:
                raise HTTPException(503, "暂时无法核对消息来源，请稍后重试") from exc
    if status in {"revoked", "stale", "conflict"}:
        message = (
            "该来源标识已用于另一条消息，请为新消息使用新的标识。"
            if status == "conflict"
            else "该消息来源已失效，不能重放；重新发送消息时请使用新的标识。"
        )
        raise HTTPException(409, detail={"code": "source_identity_" + status, "message": message})
    if status not in {"new", "pending", "recorded"}:
        raise HTTPException(503, "暂时无法核对消息来源，请稍后重试")


def _build_chat_generation_service(runtime, character_service=None, *, message_db) -> ChatGenerationService:
    handler = functools.partial(
        _generate_reply_impl,
        character_service=character_service,
        message_db=message_db,
    )

    async def validate_request(request, current_user):
        await _validate_web_source_identity(request, current_user, database=message_db)

    return ChatGenerationService(
        generate_handler=handler,
        inference_runtime=runtime,
        is_high_risk_prompt=_is_high_risk_prompt,
        security_response_factory=_security_policy_response,
        trace_id_factory=lambda: uuid.uuid4().hex,
        validate_request=validate_request,
    )


def get_request_chat_generation_service(request: Request) -> ChatGenerationService:
    """Bind the application runtime and database before queue admission.

    Character context is built after the queued request resolves its provider.
    Memory reads, completion writes and message storage use the same database.
    """
    container = get_runtime_container(request.app)
    runtime = container.inference_runtime
    if container.db is None or runtime is None:
        raise HTTPException(
            503,
            detail={
                "stage": "dependencies",
                "message": "Chat database or scheduler is not configured; check application startup",
            },
        )
    return _build_chat_generation_service(runtime, message_db=container.db)


async def generate_reply_core(
    request: MessageRequest,
    current_user: dict | None = None,
    *,
    persist_message: bool = True,
    enable_rag: bool = True,
    record_invocation: bool = True,
    invocation_purpose: str = "chat",
    character_service=None,
    message_db,
    delivery_context: dict | None = None,
):
    """Generate for a platform caller with its explicitly owned database."""

    service = _build_chat_generation_service(
        inference_runtime,
        character_service,
        message_db=message_db,
    )
    return await service.generate(
        request,
        current_user,
        persist_message=persist_message,
        enable_rag=enable_rag,
        record_invocation=record_invocation,
        invocation_purpose=invocation_purpose,
        delivery_context=delivery_context,
    )


@router.post("/api/generate")
async def generate_reply(
    request: MessageRequest,
    current_user: dict = Depends(get_current_user),
    service: ChatGenerationService = Depends(get_request_chat_generation_service),
):
    """Queue-protected management/test generation endpoint."""
    if not request.branchId and (request.characterId or request.adapter == "web-character"):
        # The new web character entry owns its user scope; never trust a UI senderId.
        identity = str(current_user.get("id") or current_user.get("user_id") or current_user.get("username") or "")
        if not identity:
            raise HTTPException(401, "无法确认当前登录用户")
        request.platform = "web"
        request.adapter = "web-character"
        request.senderId = request.userId = identity
        request.conversationId = request.sessionId or "default"
        # Stateful web chat already persists every completed turn. Its scoped
        # server history owns revocation/delivery grants; a stale browser copy
        # must not override that projection in readers, reviewers or writers.
        # Stateless management and external bot paths retain their own history.
        request.history = []
        # Web clients have no external platform receipt. Allocate one identity
        # before queueing so the saved message and completion writer share it.
        # This is a request identity, not a content hash or retry dedup key.
        if not request.sourceMessageId.strip():
            request.sourceMessageId = "web:" + uuid.uuid4().hex
    try:
        return await service.generate_queued(request, current_user)
    except RateLimitExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail="请求过于频繁，请稍后重试",
            headers={"Retry-After": str(max(1, int(exc.retry_after)))},
        ) from exc
    except InferenceQueueFull as exc:
        raise HTTPException(status_code=503, detail="推理队列已满，请稍后再试") from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=503, detail="推理排队超时，请稍后再试") from exc


async def _record_model_invocation(
    request: MessageRequest,
    model_name: str,
    lora_name: str,
    cost_time: float,
    *,
    used_rag: bool = False,
    error_type: str = "",
    model_invoked: bool | None = None,
    purpose: str = "chat",
    database,
):
    """Record required accounting before archive; timed-out writes may still commit."""
    try:
        await asyncio.wait_for(
            write_invocation(
                database,
                {
                    "traceId": request.traceId,
                    "purpose": purpose,
                    "platform": request.platform,
                    "conversationId": request.conversationId or request.sessionId,
                    "sessionId": request.sessionId,
                    "modelName": model_name,
                    "loraName": lora_name,
                    "costTime": cost_time,
                    "usedRag": used_rag,
                    "usedLora": bool(lora_name and lora_name != "default"),
                    "errorType": error_type,
                },
            ),
            timeout=_DB_WRITE_TIMEOUT,
        )
    except Exception as e:
        increment("db_write_failures")
        log_event(
            "db_write_failed",
            level="warning",
            traceId=request.traceId,
            platform=request.platform,
            conversationId=request.conversationId or request.sessionId,
            senderId=request.senderId or request.userId,
            model=model_name,
            costTime=cost_time,
            errorType=type(e).__name__,
        )
        logger.warning("Failed to save model invocation traceId=%s error=%s", request.traceId, type(e).__name__)
        raise HTTPException(
            503,
            detail={
                "stage": "model_invocation_record",
                "message": (
                    "生成失败，且调用记录保存失败或未确认，请检查数据库和模型服务后重试"
                    if error_type
                    else "模型已完成，但调用记录保存失败或未确认，请检查数据库后重试"
                ),
                "generationErrorType": error_type or None,
                "recordErrorType": type(e).__name__,
                "modelInvoked": model_invoked if model_invoked is not None else (None if error_type else True),
                "invocationRecorded": None,
                "messageSaved": False,
            },
        ) from e


async def _save_message(
    request: MessageRequest,
    reply: str,
    model_name: str,
    lora_name: str,
    cost_time: float,
    *,
    database,
    character_id: str | None = None,
    saved_receipt: dict | None = None,
    model_invoked: bool | None = None,
) -> bool:
    """Save generated replies with platform-aware metadata.

    Return True only with a valid storage receipt; errors stop generation.
    A timed-out thread can still commit, so an error does not prove rollback.
    database is the owning database already bound by the request entry.
    """
    if saved_receipt is not None:
        saved_receipt.clear()
    try:
        stored = await asyncio.wait_for(
            asyncio.to_thread(
                database.add_message,
                {
                    "sessionType": request.sessionType,
                    "sessionId": request.sessionId,
                    "sessionName": request.sessionName or request.userName or request.sessionId or "test-session",
                    "conversationType": request.conversationType or request.sessionType,
                    "platform": request.platform,
                    "adapter": request.adapter,
                    "conversationId": request.conversationId or request.sessionId,
                    "senderId": request.senderId or request.userId,
                    "senderName": request.senderName or request.userName,
                    "sourceMessageId": request.sourceMessageId,
                    "traceId": request.traceId,
                    "userId": request.userId,
                    "userName": request.userName,
                    "message": request.message,
                    "reply": reply,
                    "modelName": model_name,
                    "loraName": lora_name,
                    "characterId": character_id or request.characterId,
                    "costTime": cost_time,
                },
            ),
            timeout=_DB_WRITE_TIMEOUT,
        )
        if not isinstance(stored, dict) or not stored.get("id"):
            raise ValueError("Message storage returned no valid receipt")
        if saved_receipt is not None:
            saved_receipt.update(stored)
        return True
    except Exception as e:
        increment("db_write_failures")
        log_event(
            "db_write_failed",
            level="warning",
            traceId=request.traceId,
            platform=request.platform,
            conversationId=request.conversationId or request.sessionId,
            senderId=request.senderId or request.userId,
            model=model_name,
            costTime=cost_time,
            errorType=type(e).__name__,
        )
        logger.warning(
            "Failed to save message record traceId=%s sessionId=%s error=%s",
            request.traceId,
            request.sessionId,
            type(e).__name__,
        )
        raise HTTPException(
            503,
            detail={
                "stage": "message_archive",
                "message": "消息保存失败或未确认，请检查数据库及该请求的保存状态后重试",
                "modelInvoked": model_invoked,
                "messageSaved": None,
            },
        ) from e


# ═══════════════════════════════════════════
# vLLM 状态查询路由
# ═══════════════════════════════════════════


@router.get("/api/vllm/status")
async def vllm_status(current_user: dict = Depends(get_current_admin)):
    """查询 vLLM 实例状态"""
    client = await get_vllm_client()
    if client is None:
        return {"enabled": False, "instances": []}
    try:
        instances = await client.health_check()
    except Exception as exc:
        raise HTTPException(503, "vLLM 状态查询失败，请检查服务连接后重试") from exc
    return {"enabled": True, "instances": instances}
