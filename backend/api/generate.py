"""消息生成API - 支持vLLM高并发推理"""

import asyncio
import functools
import hashlib
import json
import logging
import os
import time
import uuid
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import ValidationError
from services.chat_generation import ChatGenerationService

from api.error_details import with_provider_error
from app.config import (
    INPUT_VALIDATOR_AVAILABLE,
    circuit_breaker_registry,
    get_llm_semaphore,
    get_vllm_served_model_name,
    response_cache,
)
from app.dependencies import get_current_admin, get_current_user
from app.runtime import get_runtime_container
from db.adapter import db
from db.database import get_lora_path_by_id
from db.schemas import GenerateResponse, MessageRequest
from inference.generation_request import (
    ContextAuthorityChangedError,
    ReplyValidationError,
    RetrievalResult,
    generate_character_response,
)
from inference.generation_request import (
    GenerationRequest as CharacterGenerationRequest,
)
from inference.lora_registry import get_lora_character_id
from inference.lora_registry import get_lora_system_prompt as _get_system_prompt
from inference.lora_utils import resolve_lora_served_name
from inference.openai_protocol import ProviderResponseError
from inference.provider_context import get_provider_context_budget
from infra.circuit_breaker import CircuitOpenError
from infra.concurrency_control import InferenceQueueFull, RateLimitExceeded, inference_runtime
from infra.environment import read_bool
from infra.observability import increment, log_event, set_consecutive
from infra.security_utils import strip_control_chars

if INPUT_VALIDATOR_AVAILABLE:
    from infra.input_validator import MESSAGE_SCHEMA, InputValidator

logger = logging.getLogger(__name__)

# ── vLLM 客户端（延迟初始化） ──
_vllm_client = None
_vllm_initialized = False
_vllm_init_lock: asyncio.Lock | None = None
_vllm_init_lock_loop: asyncio.AbstractEventLoop | None = None

_RAG_TIMEOUT = float(os.getenv("RAG_TIMEOUT", "8"))
_RAG_COLD_START_TIMEOUT = float(os.getenv("RAG_COLD_START_TIMEOUT", "60"))
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


def _loop_local_lock(name: str) -> asyncio.Lock:
    """Return a module lock owned by the active application event loop."""
    global _vllm_init_lock, _vllm_init_lock_loop
    global _local_model_lock, _local_model_lock_loop

    loop = asyncio.get_running_loop()
    if name == "vllm":
        if _vllm_init_lock is None or _vllm_init_lock_loop is not loop:
            _vllm_init_lock = asyncio.Lock()
            _vllm_init_lock_loop = loop
        return _vllm_init_lock
    if name == "local_model":
        if _local_model_lock is None or _local_model_lock_loop is not loop:
            _local_model_lock = asyncio.Lock()
            _local_model_lock_loop = loop
        return _local_model_lock
    raise ValueError(f"unknown lock name: {name}")


def _is_high_risk_prompt(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in _HIGH_RISK_PROMPT_PATTERNS)


def _security_policy_response() -> GenerateResponse:
    return GenerateResponse(
        reply="\u8be5\u8bf7\u6c42\u6d89\u53ca\u7cfb\u7edf\u914d\u7f6e\u3001\u51ed\u636e\u6216\u5185\u90e8\u6307\u4ee4\uff0c\u5df2\u88ab\u5b89\u5168\u7b56\u7565\u62e6\u622a\u3002",
        model="security-policy",
        costTime=0.0,
    )


async def _ensure_vllm():
    """延迟初始化可恢复的共享 vLLM 客户端。"""
    global _vllm_client, _vllm_initialized
    if _vllm_client is not None:
        _vllm_initialized = True
        return True

    async with _loop_local_lock("vllm"):
        if _vllm_client is not None:
            _vllm_initialized = True
            return True

        from app.config import is_vllm_enabled

        if not is_vllm_enabled():
            _vllm_initialized = False
            return False

        try:
            from inference.vllm_client import get_vllm_client as _get_shared

            client = await _get_shared()
            if client is None:
                raise RuntimeError("共享 vLLM 客户端返回空实例")
            _vllm_client = client
            _vllm_initialized = True
            logger.info("vLLM 客户端初始化成功（共享单例）")
            return True
        except Exception as exc:
            # vLLM 是外部服务，短暂初始化失败必须允许后续请求重试。
            _vllm_client = None
            _vllm_initialized = False
            logger.warning("vLLM 客户端初始化失败: %s", exc)
            return False


async def get_vllm_client():
    if not await _ensure_vllm():
        return None
    return _vllm_client


async def close_vllm_client():
    """关闭共享 vLLM 客户端，并允许后续应用生命周期重新初始化。"""
    global _vllm_client, _vllm_initialized
    async with _loop_local_lock("vllm"):
        _vllm_client = None
        _vllm_initialized = False
        try:
            from inference.vllm_client import close_shared_vllm_client

            await close_shared_vllm_client()
            logger.info("vLLM 客户端已关闭（共享单例）")
        except Exception as exc:
            logger.warning("关闭 vLLM 客户端失败: %s", exc)


router = APIRouter()


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
        "temperature": float(config.get("temperature", os.getenv("VLLM_TEMPERATURE", "0.7"))),
        "max_tokens": int(config.get("maxTokens", os.getenv("VLLM_MAX_TOKENS", "2048"))),
        "top_p": float(config.get("topP", os.getenv("VLLM_TOP_P", "0.9"))),
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
    character_service=None,
    message_db=None,
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
    slots = []
    try:
        with use_request_context(context):
            return await _generate_reply_body(
                request, current_user, persist_message=persist_message, enable_rag=enable_rag,
                record_invocation=record_invocation, character_service=character_service,
                message_db=message_db, delivery_context=delivery_context,
                prepared_override=prepared_override, _completion_slots=slots,
            )
    finally:
        for slot in slots:
            slot.release()


async def _generate_reply_body(
    request: MessageRequest,
    current_user: dict | None = None,
    *,
    persist_message: bool = True,
    enable_rag: bool = True,
    record_invocation: bool = True,
    character_service=None,
    message_db=None,
    delivery_context: dict | None = None,
    prepared_override=None,
    _completion_slots=None,
):
    """使用所选提供商执行共享检索和上下文编译。

    character_service / message_db 由 HTTP 层经容器注入（绑定当前
    应用容器的数据库）：前者用于人物上下文（记忆/关系/历史），后者
    用于消息持久化、模型调用记录与生成配置读取。两者为 None 时
    （bot 直连、旧测试等非 HTTP 调用方）回退到全局单例，保持既有
    行为。必须保证消息写入与人物历史读取使用同一数据库，否则上一
    轮消息写入全局库、下一轮从容器库读不到，历史会断裂。
    """
    if (
        prepared_override is None
        and not request.branchId
        and (request.adapter == "narrative" or request.adapter.startswith("narrative:"))
    ):
        # This namespace is owned by authenticated workspace routes. Letting
        # generic callers choose its senderId would poison another owner's history.
        raise HTTPException(422, "请通过假想分支工作区访问该会话")
    if request.branchId and prepared_override is None:
        from api.narrative import require_enabled

        require_enabled()
        if delivery_context is not None or not persist_message or request.platform != "web":
            raise HTTPException(422, "假想分支第一版仅支持 Web 交互")
        from services.narrative import generate_branch_reply

        return await generate_branch_reply(
            request,
            current_user,
            message_db if message_db is not None else db,
            generate=_generate_reply_impl,
        )

    # C11 fix: 主聊天端点 mock provider 防护
    # 此前仅 training/generate-dialogues 有此检查，主聊天端点遗漏，
    # 导致生产环境默认 mock 时静默返回罐头回复。现统一拦截。
    from inference.model_manager import get_model_manager
    from inference.request_context import get_request_context

    _mgr = get_model_manager()
    request_context = get_request_context()
    request_provider = request_context.provider_name
    if request_provider == "mock":
        raise HTTPException(
            status_code=503,
            detail="当前模型提供商为 mock 模式，无法提供真实推理。"
            "请在设置页面配置有效的模型提供商（如 vLLM / DeepSeek API / Ollama）。",
        )

    # 输入验证
    if INPUT_VALIDATOR_AVAILABLE:
        is_valid, errors = InputValidator.validate(request.model_dump(), MESSAGE_SCHEMA)
        if not is_valid:
            raise HTTPException(status_code=422, detail={"message": "输入验证失败", "errors": errors})

    source_db = message_db if message_db is not None else db
    try:
        runtime_config, loras = await asyncio.gather(
            asyncio.to_thread(lambda: source_db.config or {}),
            asyncio.to_thread(lambda: list(source_db.loras)),
        )
    except Exception as exc:
        raise HTTPException(503, "生成配置读取失败，请检查数据库连接后重试") from exc

    if request_provider == "openai_compat":
        runtime_config = dict(runtime_config)
        runtime_config.setdefault("topP", 1.0)

    # 获取LoRA：始终计算 active_lora，优先使用前端指定的，否则使用当前激活的
    active_lora = next((item for item in loras if item["status"] == "active"), None)
    selected_lora = active_lora
    if prepared_override is not None or request.characterId or request.loraName == "default":
        selected_lora = None  # Explicit adapter or base model; never auto-route to another persona.

    if request.loraName and request.loraName != "default":
        selected_lora = next((item for item in loras if item["name"] == request.loraName), None)
        if selected_lora is None:
            raise HTTPException(status_code=422, detail="Specified LoRA does not exist")
        lora_name = selected_lora["name"]
    else:
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
    if (mapped_character_id or prepared_override is not None) and persist_message and delivery_context is None and not request.branchId:
        from services.turn_completion import CompletionUnavailable, get_turn_completion_runtime

        try:
            completion_slot = get_turn_completion_runtime().reserve()
        except CompletionUnavailable:
            raise HTTPException(503, detail={
                "code": "turn_completion_busy",
                "message": "长期记忆回写繁忙，本轮尚未生成，请稍后重试。",
            }) from None
        _completion_slots.append(completion_slot)

    if mapped_character_id and prepared_override is None:
        if character_service is None:
            from services.character_context import (
                build_character_context_service,
                get_default_character_context_service,
            )

            try:
                character_service = (get_default_character_context_service() if source_db is db
                                     else build_character_context_service(source_db))
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

    # 检查vLLM是否实际支持该LoRA，避免404触发熔断
    vllm_effective_lora = vllm_lora_name if lora_name != "default" else None
    if request_provider == "vllm" and vllm_effective_lora:
        try:
            if not await _ensure_vllm() or _vllm_client is None:
                raise RuntimeError("Selected vLLM service is unavailable")
            available_loras = await _vllm_client.list_loras()
            if vllm_effective_lora not in available_loras:
                raise HTTPException(409, "所选 LoRA 尚未加载到 vLLM，请先重新激活")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(503, "所选 LoRA 状态查询失败，请检查 vLLM 服务及适配器后重试") from exc

    completion_warning = None
    start_time = time.time()
    prompt_hash = cache_key = ""
    cache_ttl = 300
    # 带角色上下文的对话是有状态流程（每轮回写记忆/关系），整体绕过
    # 响应缓存：缓存命中会跳过回写，导致人物状态与对话脱节。
    use_response_cache = (
        response_cache is not None and prepared_character_turn is None and selected_lora is None
        and not (read_bool(runtime_config, "useKnowledgeBase", True) and enable_rag)
    )
    cloud_provider = request_context.provider
    if use_response_cache:
        try:
            provider = cloud_provider if cloud_provider is not None else _mgr.get_current_provider()
            model_identity = {
                "provider": request_provider, "status": provider.get_status(),
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
            raise HTTPException(503, detail={
                "stage": "response_cache", "reason": "invalid_cached_response",
                "fields": list(dict.fromkeys(str(error["loc"][0]) if error["loc"] else "$"
                                             for error in exc.errors(include_input=False))),
                "message": "缓存的生成结果缺少必要字段或内容无效，请清理该条缓存后重试。",
            }) from exc
        except Exception as exc:
            raise HTTPException(503, "回复缓存读取失败，请检查缓存状态后重试") from exc
        if response is not None:
            logger.debug("response cache hit")
            if persist_message:
                stored_model_name = "vllm" if response.model.startswith("vllm/") else response.model
                await _save_message(
                    request, response.reply, stored_model_name, lora_name,
                    round(time.time() - start_time, 2), database=message_db, character_id=mapped_character_id,
                    model_invoked=False,
                )
            return response.model_copy(update={
                "modelInvoked": False, "costTime": round(time.time() - start_time, 2),
                "warnings": list(dict.fromkeys([*(response.warnings or []), "cached_model_response"])),
            })

    is_vllm = request_provider == "vllm"
    model_name = f"vllm/{get_vllm_served_model_name()}" if is_vllm else "model_manager"
    try:
        if is_vllm:
            if not await _ensure_vllm() or _vllm_client is None:
                raise HTTPException(503, "所选 vLLM 服务不可用，请恢复服务后重试")
            reply, used_rag, rag_meta = await _generate_with_vllm(
                request, vllm_effective_lora, vllm_lora_name, runtime_config,
                enable_rag=enable_rag, prepared_character_turn=prepared_character_turn,
                message_db=message_db,
            )
            model_label = model_name
            stored_model_name = "vllm"
        else:
            semaphore = get_llm_semaphore()
            try:
                await asyncio.wait_for(semaphore.acquire(), timeout=30.0)
            except TimeoutError as exc:
                raise HTTPException(503, "服务繁忙，请稍后再试") from exc
            try:
                async with (nullcontext() if request_provider == "openai_compat" else _loop_local_lock("local_model")):
                    try:
                        lora_path = get_lora_path_by_id(selected_lora["id"], loras=loras) if selected_lora else None
                    except OSError as exc:
                        raise HTTPException(503, "所选 LoRA 目录读取失败，请检查适配器存储后重试") from exc
                    if selected_lora and lora_path is None:
                        raise HTTPException(409, "所选 LoRA 没有可用本地目录，请先重新激活")
                    if request_provider != "openai_compat":
                        _mgr.set_lora_adapter(lora_path)

                    async def adapter(*, messages, **kwargs):
                        nonlocal model_name
                        if _mgr._current_provider.value != request_provider:
                            raise RuntimeError("Selected provider changed before generation; retry the request")
                        if request_provider == "openai_compat":
                            model_name = cloud_provider.get_status()["modelName"]
                            text, _cost = await cloud_provider.async_complete(messages, **kwargs)
                        else:
                            text, _cost = await _mgr.async_generate(
                                prompt=messages[-1]["content"], session_history=messages[:-1],
                                rag_docs=None, max_tokens_override=kwargs["max_tokens"],
                            )
                        return text

                    async def generate():
                        return await _generate_with_retrieval(
                            request, lora_name, runtime_config=runtime_config,
                            enable_rag=enable_rag, prepared_character_turn=prepared_character_turn,
                            message_db=message_db, model_generate=adapter,
                        )

                    breaker = await circuit_breaker_registry.get_or_create("model_generate") if circuit_breaker_registry else None
                    operation = breaker.call(generate) if breaker else generate()
                    reply, used_rag, rag_meta = await asyncio.wait_for(operation, timeout=_MODEL_INFERENCE_TIMEOUT)
            finally:
                semaphore.release()
            if cloud_provider is not None:
                model_name = cloud_provider.get_status()["modelName"]
            else:
                status = _mgr.get_status()
                model_name = status.get("providers", {}).get(request_provider, {}).get("modelName", "Unknown")
            model_label = f"{model_name} ({request_provider})"
            stored_model_name = model_name

        cost_time = round(time.time() - start_time, 2)
        model_invoked = rag_meta.get("modelInvoked", True) is not False
        if not model_invoked:
            model_name = _DETERMINISTIC_MODEL_LABELS.get(rag_meta.get("answerMode"), "rag/abstained")
            model_label = stored_model_name = model_name
        stored_lora_name = lora_name if model_invoked else "default"
        if model_invoked:
            if record_invocation:
                await _record_model_invocation(
                    request, model_name, lora_name, cost_time, used_rag=used_rag,
                    completion_text=reply, error_type="", database=message_db,
                )
            set_consecutive("model_failure", True)
        message_saved = False
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
            completion_warning = await _complete_character_turn(
                prepared_character_turn,
                request,
                reply,
                character_service=character_service,
                completion_slot=completion_slot,
            )
            if completion_warning:
                stored = await _persist_completion_feedback(
                    archive_receipt, completion_warning, database=message_db,
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

        result = GenerateResponse(
            reply=_reply_with_memory_warning(reply, completion_warning),
            model=model_label,
            costTime=cost_time,
            citations=rag_meta.get("citations"),
            confidence=rag_meta.get("confidence"),
            abstained=rag_meta.get("abstained", False),
            answerMode=rag_meta.get("answerMode"),
            domainId=rag_meta.get("domainId"),
            modelInvoked=model_invoked,
            warnings=([*(rag_meta.get("warnings") or []), completion_warning]
                      if completion_warning else rag_meta.get("warnings")),
        )
        if use_response_cache:
            try:
                await response_cache.set(prompt_hash, cache_key, result.model_dump(), ttl=cache_ttl)
            except Exception as exc:
                raise HTTPException(503, detail={
                    "stage": "response_cache", "message": "回复缓存写入失败，请检查缓存状态后重试",
                    "modelInvoked": model_invoked, "messageSaved": message_saved,
                }) from exc

        return result
    except HTTPException:
        raise
    except Exception as e:
        failed_cost = round(time.time() - start_time, 2)
        increment("model_failures")
        set_consecutive("model_failure", False)
        log_event(
            "model_invocation_failed",
            level="error",
            traceId=request.traceId,
            platform=request.platform,
            conversationId=request.conversationId or request.sessionId,
            senderId=request.senderId or request.userId,
            model=model_name,
            costTime=failed_cost,
            errorType=type(e).__name__,
        )
        logger.error("Generation failed: %s", type(e).__name__)
        if record_invocation:
            try:
                await _record_model_invocation(
                    request,
                    model_name,
                    lora_name,
                    failed_cost,
                    used_rag=False,
                    error_type=type(e).__name__,
                    database=message_db,
                )
            except HTTPException as record_error:
                # Keep the model cause and recording error on separate chains.
                raise HTTPException(503, detail=with_provider_error(record_error.detail, e)) from e
        if isinstance(e, ProviderResponseError):
            raise HTTPException(e.http_status, detail=e.public_detail()) from e
        if isinstance(e, CircuitOpenError):
            raise HTTPException(503, detail={
                "stage": "model_circuit", "reason": "circuit_open",
                "message": "模型服务连续失败，暂时停止调用。请恢复所选模型服务后重试。",
                "modelInvoked": False,
            }) from e
        if isinstance(e, ContextAuthorityChangedError):
            raise HTTPException(503, detail={
                "stage": "context_authority", "reason": "repeated_authority_change",
                "message": "本轮资料在生成期间持续变化，未返回答案，请稍后重新请求。",
            }) from e
        if isinstance(e, ReplyValidationError):
            raise HTTPException(503, detail={
                "stage": "reply_validation",
                "reason": "model_output_invalid",
                "violations": list(e.violations),
                "message": "模型回复经一次修正仍未通过校验，本轮未返回回复。请重试或检查模型与输出策略。",
            }) from e
        # 安全：不把内部异常字符串返回给客户端（信息泄露），
        # 日志记录错误类型；异常链保留内部原因，客户端收到通用消息。
        raise HTTPException(503 if is_vllm else 500, "生成回复失败，请检查所选模型服务后重试") from e


# ═══════════════════════════════════════════
# 角色上下文辅助函数
# ═══════════════════════════════════════════


async def _prepare_character_turn(
    request: MessageRequest, character_id: str, *, character_service=None, execute_memory_operations: bool = False,
    defer_memory_operations: bool = False
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
            TurnInput,
            get_default_character_context_service,
        )

        turn_input = TurnInput(
            message=request.message,
            platform=request.platform,
            adapter=request.adapter,
            sender_id=request.senderId or request.userId,
            conversation_id=request.conversationId or request.sessionId,
            conversation_type=request.conversationType or request.sessionType,
            history=tuple(request.history or []),
            received_at=request._source_received_at,
        )
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
                prepared = replace(prepared, compiled=replace(
                    prepared.compiled, memory_operation_deferred=True,
                ))
        return prepared
    except ContextAfterMemoryOperationError as exc:
        raise HTTPException(503, detail=with_provider_error({
            "code": "context_failed_after_memory_operation",
            "message": "角色上下文准备失败，回复未生成。请先核对本轮记忆操作回执，再检查审核模型和上下文配置。",
            "memoryOperation": exc.operation_summary,
        }, exc)) from exc
    except MemoryContextBudgetError as exc:
        raise HTTPException(503, detail={
            "stage": "memory_context", "reason": "selected_evidence_budget_exceeded",
            "message": "已选必要记忆无法完整放入当前上下文，本轮未生成。请缩小问题范围或增加上下文预算。",
        }) from exc
    except EventMemoryReadError as exc:
        raise HTTPException(503, detail={
            "stage": "memory_read", "reason": "invalid_event_record", "field": exc.field,
            "message": "已保存的事件记录格式错误，本轮未生成。请检查并修复对应记忆字段后重试。",
        }) from exc
    except MemoryRecordError as exc:
        raise HTTPException(503, detail={
            "stage": "memory_read", "reason": "invalid_memory_record", "field": exc.field,
            "message": "已保存的记忆记录格式错误，本轮未生成。请检查并修复对应记忆字段后重试。",
        }) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            503, with_provider_error(
                "角色上下文准备失败，本轮未生成。请检查数据库、审核模型及上下文配置后重试。", exc,
            ),
        ) from exc


def _reply_with_memory_warning(reply: str, warning: str | None) -> str:
    """Keep model content and make persistence feedback visible to reply-only clients."""
    from db.message_feedback import reply_with_warning

    return reply_with_warning(reply, warning)


async def _persist_completion_feedback(receipt: dict, warning: str, *, database=None) -> bool:
    """Finalize only the exact row returned by successful message storage."""
    target_db = database if database is not None else db
    writer = getattr(target_db, "update_message_feedback", None)
    if not receipt or not callable(writer):
        return False
    try:
        return bool(await asyncio.wait_for(
            asyncio.to_thread(writer, receipt, warning=warning), timeout=_DB_WRITE_TIMEOUT,
        ))
    except Exception:
        # Preserve visible current feedback even when history storage is
        # unknown; never regenerate the model or replace an unrelated row.
        logger.warning("保存提示存档未能确认", exc_info=True)
        return False


async def _complete_character_turn(prepared, request: MessageRequest, reply: str, *, character_service=None, completion_slot=None) -> str | None:
    """生成成功后回写；保留已生成内容，向调用方返回独立的保存提示。

    character_service 语义同 _prepare_character_turn：必须与准备阶段
    使用同一服务实例，保证读写同一（容器）数据库。
    """
    try:
        from services.character_context import (
            TurnInput,
            get_default_character_context_service,
        )

        turn_input = TurnInput(
            message=request.message,
            platform=request.platform,
            adapter=request.adapter,
            sender_id=request.senderId or request.userId,
            conversation_id=request.conversationId or request.sessionId,
            conversation_type=request.conversationType or request.sessionType,
            history=tuple(request.history or []),
            received_at=request._source_received_at,
        )
        service = character_service or get_default_character_context_service()
        from services.turn_completion import get_turn_completion_runtime

        outcome = await get_turn_completion_runtime().run(
            lambda: service.complete_turn(
                prepared,
                turn_input,
                reply,
                source_message_id=request.sourceMessageId,
            ),
            timeout=_DB_WRITE_TIMEOUT,
            reservation=completion_slot,
        )
        capture = getattr(outcome, "source_capture", "")
        status = getattr(outcome, "memory_enrichment_status", "")
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
        elif status in {"capacity", "closed"}:
            warning = (
                ("原文已保存，但" if capture == "recorded" else "")
                + "长期记忆任务尚未受理；本次回复不代表结构化记忆已写入，请稍后重试。"
            )
        elif status == "disabled":
            warning = "自动长期记忆未启用；本次回复不代表已保存。"
        failures = getattr(outcome, "failed_writes", ())
        if failures:
            labels = {"interaction_count": "交互统计", "preferred_address": "称呼更新"}
            detail = "、".join(labels[stage] for stage in failures)
            partial = f"部分回写失败（{detail}）；已完成的保存不会撤销，请检查后重试。"
            warning = partial + (warning or "")
        return warning
    except Exception as e:
        logger.warning("角色记忆回写失败 character=%s type=%s", prepared.character_id, type(e).__name__)
        # Cancelling a to_thread write does not prove its transaction rolled
        # back. Never claim failure or successful storage on timeout/unknown.
        return "这条信息的长期记忆保存状态尚未确认；本次回复不代表已保存，请稍后检查。"


# ═══════════════════════════════════════════
# vLLM 推理辅助函数
# ═══════════════════════════════════════════


async def _validate_web_source_identity(request, current_user, *, database=None):
    """Reject known immutable source collisions before any model or write.

    The writer's transactional capture guard remains authoritative. This
    admission check uses only the authenticated owner's exact source scope and
    returns state without exposing revoked source text.
    """
    if request.branchId or not request.characterId or request.platform != "web" or request.adapter != "web-character" or not request.sourceMessageId:
        return
    from character.context_builder import build_user_scope

    identity = str((current_user or {}).get("id") or (current_user or {}).get("user_id") or (current_user or {}).get("username") or "")
    if not identity:
        raise HTTPException(401, "无法确认当前登录用户")
    try:
        scope = build_user_scope(platform="web", adapter="web-character", sender_id=identity,
                                 conversation_id=request.conversationId or request.sessionId,
                                 conversation_type=request.conversationType or request.sessionType)
    except ValueError as exc:
        raise HTTPException(422, "无法确认消息来源范围") from exc
    if not request.message:
        raise HTTPException(422, "消息不能为空")
    target_db = db if database is None else database
    try:
        status = await asyncio.wait_for(asyncio.to_thread(
            target_db.memory_source_admission, request.characterId, scope.platform, scope.adapter,
            scope.sender_id, scope.conversation_type, scope.conversation_id,
            source_message_id=request.sourceMessageId, body=request.message,
        ), timeout=_DB_WRITE_TIMEOUT)
    except Exception as exc:
        raise HTTPException(503, "暂时无法核对消息来源，请稍后重试") from exc
    if status in {"new", "pending"}:
        from character.memory_extractor import memory_write_allowed
        from character.memory_llm import is_memory_erasure_request

        # Opt-out, credential and explicit erasure messages create no binding.
        if memory_write_allowed(request.message) and not is_memory_erasure_request(request.message):
            try:
                receipt = await asyncio.wait_for(asyncio.to_thread(
                    target_db.reserve_memory_source, request.characterId, scope.platform, scope.adapter,
                    scope.sender_id, scope.conversation_type, scope.conversation_id,
                    source_message_id=request.sourceMessageId, body=request.message,
                    observed_at=datetime.now(timezone.utc),
                ), timeout=_DB_WRITE_TIMEOUT)
                status = receipt["status"]
                if status == "pending":
                    observed = datetime.fromisoformat(receipt["observed_at"])
                    if observed.tzinfo is None or observed.utcoffset() is None:
                        raise ValueError("Untrusted source receipt")
                    request._source_received_at = observed
            except Exception as exc:
                raise HTTPException(503, "暂时无法核对消息来源，请稍后重试") from exc
    if status in {"revoked", "stale", "conflict"}:
        message = "该来源标识已用于另一条消息，请为新消息使用新的标识。" if status == "conflict" else "该消息来源已失效，不能重放；重新发送消息时请使用新的标识。"
        raise HTTPException(409, detail={"code": "source_identity_" + status, "message": message})
    if status not in {"new", "pending", "recorded"}:
        raise HTTPException(503, "暂时无法核对消息来源，请稍后重试")


def _build_chat_generation_service(runtime, character_service=None, message_db=None) -> ChatGenerationService:
    handler = _generate_reply_impl
    if character_service is not None or message_db is not None:
        # 角色上下文服务与消息库均绑定当前应用容器的数据库，
        # 随 handler 注入；两者必须同库，历史才不会断裂
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
        sanitize_message=strip_control_chars,
        is_high_risk_prompt=_is_high_risk_prompt,
        security_response_factory=_security_policy_response,
        trace_id_factory=lambda: uuid.uuid4().hex,
        validate_request=validate_request,
    )


_chat_generation_service = _build_chat_generation_service(inference_runtime)


def get_chat_generation_service() -> ChatGenerationService:
    """Return the default service for non-HTTP compatibility callers."""

    return _chat_generation_service


def get_request_chat_generation_service(request: Request) -> ChatGenerationService:
    """Bind the application runtime and database before queue admission.

    Character context is built after the queued request resolves its provider.
    Memory reads, completion writes and message storage use the same database.
    """
    container = get_runtime_container(request.app)
    runtime = inference_runtime if container.inference_runtime is None else container.inference_runtime
    if runtime is inference_runtime and container.db is db:
        return get_chat_generation_service()
    return _build_chat_generation_service(runtime, message_db=container.db)


async def generate_reply_core(
    request: MessageRequest,
    current_user: dict | None = None,
    *,
    persist_message: bool = True,
    enable_rag: bool = True,
    record_invocation: bool = True,
    character_service=None,
    message_db=None,
    delivery_context: dict | None = None,
):
    """Compatibility entry point used by integrations and existing callers.

    character_service / message_db：HTTP 调用方（如 astrbot 集成端点）
    传入绑定当前应用容器数据库的角色服务与数据库；bot 直连等进程内
    调用方留空，回退全局单例。
    """

    return await get_chat_generation_service().generate(
        request,
        current_user,
        persist_message=persist_message,
        enable_rag=enable_rag,
        record_invocation=record_invocation,
        character_service=character_service,
        message_db=message_db,
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


async def _retrieve_rag_bundle(
    query: str, top_k: int, filters: dict[str, Any] | None, *, search_views: tuple[str, ...] = (),
    question_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Retrieve authorized knowledge with fresh source authority."""

    def retrieve() -> dict[str, Any]:
        from knowledge.source_expansion import requested_document_titles

        # An explicit named-document read owns its evidence targets. Incidental
        # character names must not replace those sources with curated lore.
        # Ordinary character questions retain the strict domain availability gate.
        if question_binding is not None:
            from knowledge.multiscale_rag.runtime import get_multiscale_rag_service
            from knowledge.public_domains import domain_plan, requires_independent_domains, retrieve_domains

            character_rag = get_multiscale_rag_service()
            plan = domain_plan(question_binding, character_rag.config)
            if requires_independent_domains(plan):
                return retrieve_domains(plan, query, top_k, curated=character_rag.retrieve_with_citations, generic=retrieve_generic)
        if not filters and not requested_document_titles(query):
            from knowledge.multiscale_rag.runtime import get_multiscale_rag_service
            from knowledge.retrieval_core.query import QueryAnalyzer

            character_rag = get_multiscale_rag_service()
            matched = QueryAnalyzer([character_rag.config]).analyze(query).matched_domains
            if question_binding is not None:
                from knowledge.public_question_binding import all_bound_objects_outside_character_domain

                if all_bound_objects_outside_character_domain(question_binding, character_rag.config):
                    matched = []
                    character_rag = None
            # retrieve_with_citations 内部完成惰性加载和域门控。不能先用
            # is_available() 短路，否则冷启动首条请求无法触发索引加载。
            bundle = character_rag.retrieve_with_citations(query, top_k=top_k, filters=filters) if character_rag is not None else None
            if bundle is not None:
                return bundle
            if matched:
                raise RuntimeError("Requested character knowledge domain is unavailable")

        return retrieve_generic()

    def retrieve_generic() -> dict[str, Any]:
        from api import knowledge as knowledge_api
        from knowledge.rag_helper import get_rag_helper
        from knowledge.source_expansion import expand_source_context
        from knowledge.vector_db import get_vector_db

        # A normal generation does not necessarily follow the search API.
        # Resolve dirty metadata/content before consulting cached RAG
        # lookup; an incomplete rebuild cannot authorize stale source evidence.
        if not knowledge_api._ensure_vector_index():
            raise RuntimeError("Generic knowledge index is not ready")

        vector_db = get_vector_db()
        generation = vector_db.cache_generation
        revision = knowledge_api._vector_index_revision
        views = search_views
        contextual_binding = question_binding is not None and "reference_context" in question_binding["input"]
        if contextual_binding or (not views and question_binding is not None):
            from knowledge.public_question_binding import bound_object_search_views

            views = bound_object_search_views(question_binding, query)
        view_options = {"additional_queries": views} if views else {}
        if contextual_binding:
            view_options["question_binding"] = question_binding
        bundle = get_rag_helper().retrieve_with_citations(query, top_k=top_k, filters=filters, **view_options)
        expanded = expand_source_context(
            bundle, vector_db, expected_generation=generation,
            source_budget_tokens=get_provider_context_budget().window_tokens, filters=filters, query=query,
        )
        if knowledge_api._get_rebuild_revision() != revision:
            raise RuntimeError("Knowledge authority changed during retrieval")
        from knowledge.original_sources import attach_original_sources

        expanded = attach_original_sources(
            expanded, db.get_knowledge_document,
            source_budget_tokens=get_provider_context_budget().window_tokens,
            authority_revision=revision,
        )
        if knowledge_api._get_rebuild_revision() != revision:
            raise RuntimeError("Knowledge authority changed during retrieval")
        if expanded is not bundle:
            expanded["citations"] = get_rag_helper().build_citations(expanded["results"])
        return expanded

    return await asyncio.to_thread(retrieve)


def _rag_retrieval_timeout() -> float:
    """Allow character-index cold start without weakening steady-state limits."""

    from knowledge.multiscale_rag.runtime import get_multiscale_rag_service

    if not get_multiscale_rag_service().is_warm():
        return max(_RAG_TIMEOUT, _RAG_COLD_START_TIMEOUT)
    return _RAG_TIMEOUT


async def _generate_with_vllm(*args, **kwargs):
    """Compatibility entry point for the vLLM transport."""
    return await _generate_with_retrieval(*args, **kwargs)


async def _generate_with_retrieval(
    request: MessageRequest,
    lora_name: str | None,
    prompt_lora_name: str | None = None,
    runtime_config: dict[str, Any] | None = None,
    *,
    enable_rag: bool = True,
    prepared_character_turn=None,
    message_db=None,
    model_generate=None,
) -> tuple[str, bool, dict[str, Any]]:
    """Shared retrieval, abstention and character policy for either transport."""
    # 使用请求开始时读取的配置快照，避免同一次生成多次同步访问数据库。
    _cfg = runtime_config or {}
    _temperature = float(_cfg.get("temperature", os.getenv("VLLM_TEMPERATURE", "0.7")))
    _max_tokens = int(_cfg.get("maxTokens", os.getenv("VLLM_MAX_TOKENS", "2048")))
    _top_p = float(_cfg.get("topP", os.getenv("VLLM_TOP_P", "0.9")))
    _use_kb = read_bool(_cfg, "useKnowledgeBase", True) and enable_rag

    # RAG 检索（受设置页 useKnowledgeBase 开关控制）
    rag_context = ""
    rag_meta: dict[str, Any] = {}
    citations_enabled = read_bool(os.environ, "RAG_CITATIONS_ENABLED", True)
    independent_tasks_enabled = read_bool(os.environ, "CHARACTER_INDEPENDENT_TASKS_ENABLED", False)
    retrieval = RetrievalResult()
    filters = None
    if _use_kb:
        try:
            from knowledge.dialogue_query import contextual_retrieval_query
            from knowledge.retrieval_core.registry import get_default_registry
            from knowledge.retrieval_query_plan import plan_retrieval_views

            effective_history = tuple(request.history or []) or (
                prepared_character_turn.history if prepared_character_turn else ()
            )
            rag_message = request.message
            query_plan = await plan_retrieval_views(
                rag_message, history=effective_history, context_budget=get_provider_context_budget().review,
            )
            if not query_plan.needs_public_retrieval:
                retrieval = RetrievalResult(reason="local_context_dependencies")
                rag_meta = {"citations": [], "confidence": None, "abstained": False,
                            "answerMode": "personal_context" if query_plan.private_context_only else "direct_answer",
                            "modelInvoked": True}
            else:
                retrieval_query = contextual_retrieval_query(
                    rag_message, effective_history, get_default_registry().list_domains()
                )
                from knowledge.public_question_binding import resolve_question_binding

                question_binding = None
                if query_plan.dependencies is not None and dict(query_plan.dependencies.groups)["public_knowledge"]:
                    question_binding = await resolve_question_binding(
                        query_plan.dependencies, rag_message,
                        window_tokens=get_provider_context_budget().window_tokens,
                        public_obligations=query_plan.public_obligations,
                        **({"history": effective_history, "context_budget": get_provider_context_budget().review}
                           if effective_history else {}),
                    )
                view_options = {"search_views": query_plan.views} if query_plan.views else {}
                if question_binding is not None:
                    view_options["question_binding"] = question_binding
                retrieval_timeout = _rag_retrieval_timeout()
                bundle = await asyncio.wait_for(
                    _retrieve_rag_bundle(retrieval_query, 3, filters, **view_options),
                    timeout=retrieval_timeout,
                )
                from knowledge.public_task_evidence import review_public_candidates

                bundle = await review_public_candidates(
                    bundle, query_plan.dependencies, rag_message,
                    window_tokens=get_provider_context_budget().window_tokens,
                    public_obligations=query_plan.public_obligations,
                    question_binding=question_binding,
                )
                bundle["query_plan_status"] = query_plan.status
                rag_meta = {
                    "citations": bundle.get("citations", []) if citations_enabled else [],
                    "confidence": bundle.get("confidence"),
                    "abstained": bundle.get("abstained", False),
                    "answerMode": ("abstention" if bundle.get("abstained", False) else "grounded_answer"),
                    "domainId": next(iter(bundle.get("domains") or []), None),
                    "warnings": bundle.get("warnings") or None,
                    "modelInvoked": True,
                }
                from inference.evidence_coverage import requested_source_lookups
                from knowledge.query_tasks import requests_only_source_excerpt

                source_requests = requested_source_lookups(bundle, rag_message)

                source_lookup = bundle.get(
                    "retrieval_strategy"
                ) == "multi_scale_character" and requests_only_source_excerpt(rag_message)
                if bundle.get("abstained", False):
                    # Generate only the character's expression of uncertainty.
                    # Unreliable candidates and their citations never reach it.
                    rag_meta["citations"] = []
                    retrieval = RetrievalResult(
                        status="character_abstention",
                        confidence=bundle.get("confidence"),
                        reason="insufficient_retrieval_evidence",
                        requested_sources=source_requests,
                        source_references=tuple(bundle.get("source_references") or ()),
                        source_lookup=source_lookup,
                        task_coverage=tuple(bundle.get("task_candidate_coverage") or ()),
                        task_query=retrieval_query,
                        public_curated_review=bundle.get("public_curated_review") or {},
                        public_task_review=bundle.get("public_task_review") or {},
                        public_task_query=rag_message if bundle.get("public_task_review") or bundle.get("public_curated_review") or bundle.get("public_domain_branches") else "",
                        public_dependency_indices=tuple(dict(query_plan.dependencies.groups)["public_knowledge"]) if query_plan.dependencies is not None else (),
                    )

                # 角色知识检索结果自带按粒度组装的 context_text；
                # 通用知识库结果保留整份检索片段，交给最终请求预算裁决。
                else:
                    character_knowledge_context = bundle.get("context_text") or ""
                    evidence_packets = tuple(bundle.get("evidence_packets") or ())
                    if character_knowledge_context or bundle.get("retrieval_strategy") == "multi_scale_character":
                        rag_context = character_knowledge_context
                    else:
                        from knowledge.evidence_packets import document_evidence_packets

                        # Citation display cannot decide which source text reaches the model.
                        evidence_packets = (
                            document_evidence_packets(bundle.get("results", []))
                            + tuple(bundle.get("original_source_packets") or ())
                        )
                        rag_context = "\n\n".join(packet["text"] for packet in evidence_packets)
                    retrieval = RetrievalResult(
                        status="ok" if rag_context else "character_abstention",
                        reason="" if rag_context else "evidence_budget_exhausted",
                        evidence=rag_context,
                        evidence_packets=evidence_packets,
                        public_domain_branches=bundle.get("public_domain_branches") or {},
                        source_coverage=tuple(bundle.get("source_coverage") or ()),
                        task_coverage=tuple(bundle.get("task_candidate_coverage") or ()),
                        task_query=retrieval_query,
                        public_curated_review=bundle.get("public_curated_review") or {},
                        public_task_review=bundle.get("public_task_review") or {},
                        public_task_query=rag_message if bundle.get("public_task_review") or bundle.get("public_curated_review") or bundle.get("public_domain_branches") else "",
                        public_dependency_indices=tuple(dict(query_plan.dependencies.groups)["public_knowledge"]) if query_plan.dependencies is not None else (),
                        identity_task=bundle.get("identity_task") or {},
                        identity_subtask=bundle.get("identity_subtask") or {},
                        documents=tuple(bundle.get("results", [])),
                        citations=tuple(rag_meta.get("citations", [])),
                        confidence=bundle.get("confidence"),
                        requested_sources=source_requests,
                        source_references=tuple(bundle.get("source_references") or ()),
                        source_lookup=source_lookup,
                        source_excerpts=(
                            (bundle["raw_excerpt"],) if isinstance(bundle.get("raw_excerpt"), dict) else ()
                        ),
                    )
        except Exception as e:
            increment("rag_failures")
            log_event(
                "rag_failed",
                level="warning",
                traceId=request.traceId,
                platform=request.platform,
                conversationId=request.conversationId or request.sessionId,
                senderId=request.senderId or request.userId,
                model="rag",
                costTime=0,
                errorType=type(e).__name__,
            )
            raise HTTPException(
                status_code=503,
                detail=with_provider_error("检索或证据审核失败，请检查检索服务及审核模型状态后重试", e),
            ) from e

    model_generate = model_generate or _vllm_client.generate

    async def generate_reply(**kwargs):
        reply = await model_generate(**kwargs)
        if not isinstance(reply, str) or not reply.strip():
            raise ValueError("Model response must be non-empty text")
        return reply

    from inference.private_context_authority import make_private_context_revalidator
    from inference.public_context_authority import make_public_context_revalidator

    generation = await generate_character_response(
        CharacterGenerationRequest(
            message=request.message,
            # 角色上下文已按用户范围从数据库加载完整历史；
            # 调用方现场历史（bot 传入）非空时优先。
            history=(
                tuple(request.history or [])
                if (request.history or [])
                else (prepared_character_turn.history if prepared_character_turn else ())
            ),
            persona_prompt=_get_system_prompt(prompt_lora_name or lora_name),
            interlocutor=request.senderName or request.userName or "普通用户",
            retrieval=retrieval,
            independent_tasks_enabled=independent_tasks_enabled,
            # 编译后的角色上下文：画像/关系/情景/决策进系统提示词，
            # 长期记忆只进用户消息的不可信参考区。
            character_context=(prepared_character_turn.compiled if prepared_character_turn else None),
            reply_guard=getattr(prepared_character_turn, "reply_guard", None),
            private_context_revalidator=make_private_context_revalidator(
                prepared_character_turn, message_db if message_db is not None else db,
            ),
            public_context_revalidator=make_public_context_revalidator(db.get_knowledge_document),
            lora_name=lora_name if lora_name != "default" else None,
            temperature=_temperature,
            max_tokens=_max_tokens,
            top_p=_top_p,
            context_window_tokens=get_provider_context_budget().window_tokens,
            evidence_max_chars=get_provider_context_budget().evidence_max_chars,
        ),
        generate_reply,
    )

    if retrieval.evidence_packets or generation.plan.retrieval.answer_citations_bound:
        rag_meta["citations"] = list(generation.plan.retrieval.citations) if citations_enabled else []
        rag_meta["abstained"] = generation.plan.retrieval.status == "character_abstention"
        rag_meta["answerMode"] = "abstention" if rag_meta["abstained"] else "grounded_answer"
    from inference.evidence_coverage import is_partial_coverage

    if is_partial_coverage(generation.plan.retrieval):
        rag_meta["warnings"] = list(dict.fromkeys([*(rag_meta.get("warnings") or []), "partial_source_context"]))
    if generation.plan.retrieval.answer_citations_bound:
        rag_meta["citations"] = list(generation.response_citations) if citations_enabled else []
    if generation.authority_refreshed:
        rag_meta["warnings"] = list(dict.fromkeys([*(rag_meta.get("warnings") or []), "context_authority_refreshed"]))
    if not getattr(generation, "model_invoked", True):
        rag_meta["modelInvoked"] = False
        rag_meta["answerMode"] = generation.response_mode
        rag_meta["abstained"] = generation.response_mode == "source_unavailable"
        rag_meta["citations"] = list(generation.response_citations) if citations_enabled else []
    elif generation.response_mode == "task_composite":
        rag_meta["answerMode"] = "task_composite"
    from knowledge.public_task_evidence import render_public_tasks

    public_tasks = render_public_tasks(generation.plan.retrieval)
    if public_tasks and any(task["status"] != "related_candidate_admitted" for task in public_tasks):
        rag_meta["warnings"] = list(dict.fromkeys([*(rag_meta.get("warnings") or []), "partial_public_evidence"]))
        rag_meta["abstained"] = True
        context = generation.plan.character_context
        from html import unescape

        reference = getattr(context, "reference_context", "")
        visible_message = unescape(generation.plan.messages[-1]["content"])
        speech = getattr(context, "episodic_reference_context", "")
        private_visible = bool(reference and getattr(context, "used_memory_ids", ()) and reference in visible_message)
        private_visible = private_visible or bool(
            speech and getattr(context, "memory_source_status", "not_checked") == "available"
            and speech in visible_message
        )
        public_visible = any(
            task["status"] == "related_candidate_admitted"
            or any(
                fact["status"] == "fact_evidence_admitted" and fact["admitted_evidence"]
                for fact in task.get("fact_coverage", ())
            )
            for task in public_tasks
        )
        private_visible = bool(
            query_plan.dependencies is not None
            and dict(query_plan.dependencies.groups)["private_memory"]
            and private_visible
        )
        if (generation.model_invoked
                and (public_visible or private_visible)):
            rag_meta["answerMode"] = "partial_answer"
        elif rag_meta.get("answerMode") == "grounded_answer":
            rag_meta["answerMode"] = "abstention"
    return generation.reply, generation.plan.retrieval.has_evidence or bool(rag_meta.get("abstained")), rag_meta


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


async def _record_model_invocation(
    request: MessageRequest,
    model_name: str,
    lora_name: str,
    cost_time: float,
    *,
    used_rag: bool = False,
    error_type: str = "",
    completion_text: str = "",
    database=None,
):
    """Record required accounting before archive; timed-out writes may still commit."""
    target_db = database if database is not None else db
    try:
        prompt_tokens = _estimate_tokens(request.message)
        completion_tokens = 0 if error_type else _estimate_tokens(completion_text)
        await asyncio.wait_for(
            asyncio.to_thread(
                target_db.add_model_invocation,
                {
                    "traceId": request.traceId,
                    "platform": request.platform,
                    "conversationId": request.conversationId or request.sessionId,
                    "sessionId": request.sessionId,
                    "modelName": model_name,
                    "loraName": lora_name,
                    "costTime": cost_time,
                    "promptTokens": prompt_tokens,
                    "completionTokens": completion_tokens,
                    "totalTokens": prompt_tokens + completion_tokens,
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
        raise HTTPException(503, detail={
            "stage": "model_invocation_record",
            "message": ("生成失败，且调用记录保存失败或未确认，请检查数据库和模型服务后重试"
                        if error_type else "模型已完成，但调用记录保存失败或未确认，请检查数据库后重试"),
            "generationErrorType": error_type or None,
            "recordErrorType": type(e).__name__,
            "modelInvoked": None if error_type else True,
            "invocationRecorded": None, "messageSaved": False,
        }) from e


async def _save_message(
    request: MessageRequest,
    reply: str,
    model_name: str,
    lora_name: str,
    cost_time: float,
    *,
    database=None,
    character_id: str | None = None,
    saved_receipt: dict | None = None,
    model_invoked: bool | None = None,
) -> bool:
    """Save generated replies with platform-aware metadata.

    Return True only with a valid storage receipt; errors stop generation.
    A timed-out thread can still commit, so an error does not prove rollback.
    database 为容器注入的数据库（必须与人物历史读取同库），
    None 时用全局单例。
    """
    target_db = database if database is not None else db
    if saved_receipt is not None:
        saved_receipt.clear()
    try:
        stored = await asyncio.wait_for(
            asyncio.to_thread(
                target_db.add_message,
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
        raise HTTPException(503, detail={
            "stage": "message_archive",
            "message": "消息保存失败或未确认，请检查数据库及该请求的保存状态后重试",
            "modelInvoked": model_invoked, "messageSaved": None,
        }) from e


# ═══════════════════════════════════════════
# vLLM 状态查询路由
# ═══════════════════════════════════════════


@router.get("/api/vllm/status")
async def vllm_status(current_user: dict = Depends(get_current_admin)):
    """查询 vLLM 实例状态"""
    if not await _ensure_vllm() or not _vllm_client:
        return {"enabled": False, "instances": []}
    return {
        "enabled": True,
        "instances": await _vllm_client.health_check(),
    }
