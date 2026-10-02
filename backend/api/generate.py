"""消息生成API - 支持vLLM高并发推理"""

import asyncio
import functools
import hashlib
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

# C-F1 fix: failover_mgr 在 lifespan 中通过 app.config.failover_mgr = ...
# 赋值，导入时绑定到 None 会永远看不到实例。改为动态访问模块属性。
from app import config as _app_config
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
    GenerationRequest as CharacterGenerationRequest,
)
from inference.generation_request import (
    RetrievalResult,
    generate_character_response,
)
from inference.lora_registry import get_lora_character_id
from inference.lora_utils import resolve_lora_served_name
from inference.provider_context import get_provider_context_budget
from infra.concurrency_control import InferenceQueueFull, RateLimitExceeded, inference_runtime
from infra.observability import increment, log_event, set_consecutive
from infra.security_utils import strip_control_chars
from services.chat_generation import ChatGenerationService


def _failover_mgr():
    return _app_config.failover_mgr


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
_RAG_ABSTENTION_REPLY = (
    os.getenv("RAG_ABSTENTION_REPLY", "").strip() or "我没有找到足够可靠的信息，暂时无法回答这个问题。"
)
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


def _read_shared_kb_config_mapping() -> dict:
    """读取进程共享的意图模型配置中的知识库名称→ID映射。

    intent_classifier_model/config.json 是全局训练产物，与具体
    应用容器无绑定；读取失败或文件不存在时返回空映射。
    """
    try:
        from pathlib import Path

        config_path = Path(__file__).parent.parent / "intent_classifier_model" / "config.json"
        if config_path.exists():
            import json

            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            mapping = config.get("kb_name_to_id", {})
            return mapping if isinstance(mapping, dict) else {}
    except Exception as e:
        logger.debug("从模型config读取KB映射失败: %s", e)
    return {}


def _resolve_kb_id(kb_name: str, *, database=None):
    """根据知识库名称查询其ID，用于RAG检索过滤

    database 为容器注入的数据库时，只查该数据库：进程共享的
    intent_classifier_model/config.json 是全局训练产物，其中可能
    保存同名知识库的旧 ID，先读它会让容器实例绕过自己的数据库，
    RAG 过滤到错误的知识库。database 为 None 时（全局默认路径）
    优先读共享配置映射（训练时保存，避免每次查库），回退全局数据库。
    """
    if database is not None:
        try:
            bases = database.get_knowledge_bases()
            for b in bases:
                if b["name"] == kb_name:
                    return b["id"]
        except Exception as e:
            logger.warning("数据库查询KB ID失败: %s", e)
        return None

    # 全局路径：优先从模型config读取（训练时保存的映射，避免每次查库）
    kb_name_to_id = _read_shared_kb_config_mapping()
    if kb_name in kb_name_to_id:
        return kb_name_to_id[kb_name]

    # 回退到全局数据库查询
    try:
        bases = db.get_knowledge_bases()
        for b in bases:
            if b["name"] == kb_name:
                return b["id"]
    except Exception as e:
        logger.warning("数据库查询KB ID失败: %s", e)

    return None


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
) -> tuple[str, str, int]:
    """Include every response-affecting setting in the cache identity.

    注意：带角色上下文（人物记忆/关系）的对话整体绕过响应缓存，
    不会走到本函数——有状态对话每轮都要回写记忆与关系，缓存命中
    会跳过回写导致状态与对话脱节。
    """
    model_name = get_vllm_served_model_name()
    use_knowledge_base = bool(config.get("useKnowledgeBase", True))
    interlocutor = request.senderName or request.userName or ""
    prompt_identity = {
        "message": request.message,
        "history": list(request.history or []),
        "interlocutor": interlocutor,
        "enable_rag": enable_rag,
    }
    identity = {
        "model": model_name,
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
    slots = []
    try:
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
    """默认聊天生成实现：优先使用 vLLM，回退到模型管理器。

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

    _mgr = get_model_manager()
    if _mgr._current_provider.value == "mock":
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
    except Exception:
        logger.warning("读取生成配置失败，使用安全默认值", exc_info=True)
        runtime_config, loras = {}, []

    # 获取LoRA：始终计算 active_lora，优先使用前端指定的，否则使用当前激活的
    active_lora = next((item for item in loras if item["status"] == "active"), None)
    selected_lora = active_lora
    if prepared_override is not None or request.characterId or request.loraName == "default":
        selected_lora = None  # Explicit adapter or base model; never auto-route to another persona.

    if not request.loraName and not request.characterId and prepared_override is None:
        try:
            raw_router_config = runtime_config.get("lora_router_config", {"enabled": False})
            router_config = json.loads(raw_router_config) if isinstance(raw_router_config, str) else raw_router_config
            if not isinstance(router_config, dict):
                router_config = {"enabled": False}
            if router_config.get("enabled"):
                from inference.lora_router import RouteTarget, get_lora_router

                lora_router = get_lora_router(router_config)
                decision = lora_router.route(request.message)
                lora_router.log_routing(decision, request.traceId)
                if decision.target == RouteTarget.PERSONA_ADAPTER.value:
                    routed = next(
                        (item for item in loras if item["name"] == decision.adapter_name),
                        None,
                    )
                    if routed is not None:
                        selected_lora = routed
                    else:
                        logger.warning(
                            "LoRA route fallback: adapter is not registered adapter=%s traceId=%s",
                            decision.adapter_name,
                            request.traceId,
                        )
        except Exception:
            logger.warning(
                "LoRA routing failed; using explicit/active adapter traceId=%s", request.traceId, exc_info=True
            )
    if request.loraName and request.loraName != "default":
        selected_lora = next((item for item in loras if item["name"] == request.loraName), None)
        if selected_lora is None:
            raise HTTPException(status_code=422, detail="Specified LoRA does not exist")
        lora_name = selected_lora["name"]
    else:
        lora_name = selected_lora["name"] if selected_lora else "default"

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
        prepared_character_turn = await _prepare_character_turn(
            request,
            mapped_character_id,
            character_service=character_service,
            execute_memory_operations=persist_message and delivery_context is None,
            defer_memory_operations=persist_message and delivery_context is not None and not request.branchId,
        )
        if request.characterId and prepared_character_turn is None:
            raise HTTPException(503, "人物上下文暂时不可用，请稍后重试；本轮未降级为普通聊天")
    if prepared_character_turn is not None:
        mapped_character_id = prepared_character_turn.character_id

    # 检查vLLM是否实际支持该LoRA，避免404触发熔断
    vllm_effective_lora = vllm_lora_name if lora_name != "default" else None
    if _mgr._current_provider.value == "vllm" and vllm_effective_lora and await _ensure_vllm() and _vllm_client:
        try:
            available_loras = await _vllm_client.list_loras()
            if available_loras is not None and vllm_effective_lora not in available_loras:
                logger.warning(
                    "Selected LoRA is not loaded in vLLM name=%s available=%s",
                    vllm_effective_lora,
                    available_loras,
                )
                raise HTTPException(status_code=409, detail="所选 LoRA 尚未加载到 vLLM，请先重新激活")
        except HTTPException:
            raise
        except Exception as e:
            # A failed capability probe should not make the model unavailable.
            logger.warning("failed to query vLLM LoRA inventory: %s", e)

    completion_warning = None
    start_time = time.time()
    prompt_hash = cache_key = ""
    cache_ttl = 300
    # 带角色上下文的对话是有状态流程（每轮回写记忆/关系），整体绕过
    # 响应缓存：缓存命中会跳过回写，导致人物状态与对话脱节。
    use_response_cache = response_cache is not None and prepared_character_turn is None
    if use_response_cache:
        try:
            prompt_hash, cache_key, cache_ttl = _response_cache_keys(
                request,
                lora_name,
                runtime_config,
                enable_rag=enable_rag,
            )
            cached = await response_cache.get(prompt_hash, cache_key)
            if cached:
                logger.debug("response cache hit")
                if persist_message:
                    cached_model = cached.get("model", "response-cache")
                    stored_model_name = "vllm" if cached_model.startswith("vllm/") else cached_model
                    await _save_message(
                        request,
                        cached.get("reply", ""),
                        stored_model_name,
                        lora_name,
                        round(time.time() - start_time, 2),
                        database=message_db,
                        character_id=mapped_character_id,
                    )
                return GenerateResponse(**cached)
        except Exception as e:
            logger.warning("response cache read failed: %s", e)

    # ── 按当前提供方选择 vLLM 高并发推理 ──
    if _mgr._current_provider.value == "vllm" and await _ensure_vllm() and _vllm_client:
        try:
            reply, used_rag, rag_meta = await _generate_with_vllm(
                request,
                vllm_effective_lora,
                vllm_lora_name,
                runtime_config,
                enable_rag=enable_rag,
                prepared_character_turn=prepared_character_turn,
                message_db=message_db,
            )
            cost_time = round(time.time() - start_time, 2)

            model_invoked = rag_meta.get("modelInvoked", True) is not False
            model_label = (
                f"vllm/{get_vllm_served_model_name()}"
                if model_invoked
                else _DETERMINISTIC_MODEL_LABELS.get(rag_meta.get("answerMode"), "rag/abstained")
            )
            stored_model_name = "vllm" if model_invoked else model_label
            stored_lora_name = lora_name if model_invoked else "default"
            generation_error = rag_meta.get("generationError", "")
            if model_invoked and record_invocation:
                await _record_model_invocation(
                    request,
                    model_label,
                    lora_name,
                    cost_time,
                    used_rag=used_rag,
                    completion_text=reply,
                    error_type=generation_error,
                    database=message_db,
                )
                set_consecutive("model_failure", not bool(generation_error))
                if generation_error:
                    increment("model_failures")
            message_saved = False
            archive_receipt = {}
            if persist_message:
                message_saved = await _save_message(
                    request,
                    reply,
                    stored_model_name,
                    stored_lora_name,
                    cost_time,
                    database=message_db,
                    character_id=mapped_character_id,
                    saved_receipt=archive_receipt,
                )
            # 仅当本轮消息确实要持久化且保存成功时才回写人物状态：
            # persist_message=False（如 Claw 内部推理）不得污染人物记忆；
            # 消息保存失败时跳过回写，保持消息记录与人物状态一致。
            if delivery_context is not None:
                delivery_context.update(message_saved=message_saved, character_id=mapped_character_id)
                if prepared_character_turn is not None and message_saved:
                    from services.delivery_memory import freeze_completion

                    delivery_context["completion_snapshot"] = freeze_completion(prepared_character_turn)
            if prepared_character_turn is not None and persist_message and message_saved and delivery_context is None:
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
                errorType=generation_error,
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
                warnings=([*(rag_meta.get("warnings") or []), completion_warning]
                          if completion_warning else rag_meta.get("warnings")),
            )
            if use_response_cache:
                try:
                    await response_cache.set(prompt_hash, cache_key, result.model_dump(), ttl=cache_ttl)
                except Exception as e:
                    logger.warning("vLLM缓存写入失败: %s", e)
                    pass

            return result
        except Exception as e:
            failed_cost = round(time.time() - start_time, 2)
            model_label = f"vllm/{get_vllm_served_model_name()}"
            if record_invocation:
                await _record_model_invocation(
                    request,
                    model_label,
                    lora_name,
                    failed_cost,
                    used_rag=False,
                    error_type=type(e).__name__,
                    database=message_db,
                )
            increment("model_failures")
            log_event(
                "model_invocation_failed",
                level="warning",
                traceId=request.traceId,
                platform=request.platform,
                conversationId=request.conversationId or request.sessionId,
                senderId=request.senderId or request.userId,
                model=model_label,
                costTime=failed_cost,
                errorType=type(e).__name__,
            )
            logger.warning("vLLM inference failed, falling back to model manager: %s", e)
            if vllm_effective_lora:
                raise HTTPException(
                    status_code=503,
                    detail="所选 LoRA 推理失败，请检查 vLLM 适配器状态",
                ) from e

    # ── 回退：使用原有模型管理器，共享检索与提示词策略 ──
    fallback_rag_meta: dict[str, Any] = {}
    fallback_used_rag = False
    try:
        from inference.model_manager import ModelProvider, get_model_manager

        model_manager = get_model_manager()

        semaphore = get_llm_semaphore()
        sem_acquired = False
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=30.0)
            sem_acquired = True
        except TimeoutError as exc:
            raise HTTPException(status_code=503, detail="服务繁忙，请稍后再试") from exc

        try:
            async with _loop_local_lock("local_model"):
                # C-R1 fix: 原先 LORA_PATH_MAP 恒为空 dict，此分支永不进入。
                # 改为调用 get_lora_path_by_id() 动态查找 LoRA 路径。
                # loras 传自当前应用容器数据库（source_db）：默认查全局
                # db.loras 时，容器实例的 LoRA 列表来自容器库、路径却在
                # 全局库查不到，会静默退回基座模型。
                lora_path = None
                if selected_lora:
                    lora_path = get_lora_path_by_id(selected_lora["id"], loras=loras)
                if lora_path:
                    if ModelProvider.TRANSFORMERS_PEFT in model_manager._providers:
                        peft_provider = model_manager._providers[ModelProvider.TRANSFORMERS_PEFT]
                        if hasattr(peft_provider, "set_lora_adapter"):
                            peft_provider.set_lora_adapter(lora_path)
                        model_manager.set_provider(ModelProvider.TRANSFORMERS_PEFT)
                else:
                    model_manager.set_lora_adapter(None)

                async def _do_generate_async():
                    nonlocal fallback_rag_meta, fallback_used_rag

                    # P0-C1 fix: 直接 await 原生 async_generate，禁止跨事件循环
                    # 复用缓存的 httpx.AsyncClient（曾用 asyncio.to_thread → asyncio.run
                    # 创建新循环，第二次请求会报 RuntimeError: Event loop is closed）。
                    # R6 fix: 有角色上下文时与 vLLM 路径共用同一套提示词编译
                    # （画像/关系/情景/决策进系统提示词，长期记忆与对话者
                    # 昵称进不可信参考区，数据库历史兜底），避免两条路径
                    # 对有状态人物对话行为不一致。
                    async def adapter(*, messages, **kwargs):
                        text, _cost = await model_manager.async_generate(
                            prompt=messages[-1]["content"],
                            session_history=messages[:-1],
                            rag_docs=None,
                            max_tokens_override=kwargs["max_tokens"],
                        )
                        return text

                    reply, fallback_used_rag, fallback_rag_meta = await _generate_with_retrieval(
                        request,
                        lora_name,
                        runtime_config=runtime_config,
                        enable_rag=enable_rag,
                        prepared_character_turn=prepared_character_turn,
                        message_db=message_db,
                        model_generate=adapter,
                    )
                    return reply, round(time.time() - start_time, 2)

                if circuit_breaker_registry:
                    cb = await circuit_breaker_registry.get_or_create("model_generate")
                    if cb:
                        reply, cost_time = await asyncio.wait_for(
                            cb.call(_do_generate_async),
                            timeout=_MODEL_INFERENCE_TIMEOUT,
                        )
                    else:
                        reply, cost_time = await asyncio.wait_for(
                            _do_generate_async(),
                            timeout=_MODEL_INFERENCE_TIMEOUT,
                        )
                else:
                    reply, cost_time = await _do_generate_async()
        finally:
            if sem_acquired:
                semaphore.release()

        status = model_manager.get_status()
        current_provider = status.get("currentProvider", "unknown")
        provider_status = status.get("providers", {}).get(current_provider, {})
        model_name = provider_status.get("modelName", "Unknown")
        model_invoked = fallback_rag_meta.get("modelInvoked", True) is not False
        if not model_invoked:
            model_name = _DETERMINISTIC_MODEL_LABELS.get(fallback_rag_meta.get("answerMode"), "rag/abstained")

        if record_invocation and model_invoked:
            await _record_model_invocation(
                request,
                model_name,
                lora_name,
                cost_time,
                used_rag=fallback_used_rag,
                completion_text=reply,
                error_type=fallback_rag_meta.get("generationError", ""),
                database=message_db,
            )
        message_saved = False
        archive_receipt = {}
        if persist_message:
            message_saved = await _save_message(
                request,
                reply,
                model_name,
                lora_name if model_invoked else "default",
                cost_time,
                database=message_db,
                character_id=mapped_character_id,
                saved_receipt=archive_receipt,
            )
        # 同 vLLM 路径：消息保存成功才回写人物状态，persist_message=False 不回写
        if delivery_context is not None:
            delivery_context.update(message_saved=message_saved, character_id=mapped_character_id)
            if prepared_character_turn is not None and message_saved:
                from services.delivery_memory import freeze_completion

                delivery_context["completion_snapshot"] = freeze_completion(prepared_character_turn)
        if prepared_character_turn is not None and persist_message and message_saved and delivery_context is None:
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
        set_consecutive("model_failure", not bool(fallback_rag_meta.get("generationError")))
        if fallback_rag_meta.get("generationError"):
            increment("model_failures")
        log_event(
            "message_generated",
            traceId=request.traceId,
            platform=request.platform,
            conversationId=request.conversationId or request.sessionId,
            senderId=request.senderId or request.userId,
            model=model_name,
            costTime=cost_time,
            errorType="",
            usedRag=fallback_used_rag,
        )

        result = GenerateResponse(
            reply=_reply_with_memory_warning(reply, completion_warning),
            model=f"{model_name} ({current_provider})" if model_invoked else model_name,
            costTime=cost_time,
            citations=fallback_rag_meta.get("citations"),
            confidence=fallback_rag_meta.get("confidence"),
            abstained=fallback_rag_meta.get("abstained", False),
            answerMode=fallback_rag_meta.get("answerMode"),
            warnings=([*(fallback_rag_meta.get("warnings") or []), completion_warning]
                      if completion_warning else fallback_rag_meta.get("warnings")),
            domainId=fallback_rag_meta.get("domainId"),
        )

        if use_response_cache:
            try:
                await response_cache.set(prompt_hash, cache_key, result.model_dump(), ttl=cache_ttl)
            except Exception as e:
                logger.warning("模型管理器缓存写入失败: %s", e)
                pass

        return result

    except HTTPException:
        raise
    except Exception as e:
        failed_cost = round(time.time() - start_time, 2) if "start_time" in locals() else 0.0
        if record_invocation:
            await _record_model_invocation(
                request,
                "model_manager",
                lora_name if "lora_name" in locals() else "default",
                failed_cost,
                used_rag=False,
                error_type=type(e).__name__,
                database=message_db,
            )
        increment("model_failures")
        set_consecutive("model_failure", False)
        log_event(
            "model_invocation_failed",
            level="error",
            traceId=request.traceId,
            platform=request.platform,
            conversationId=request.conversationId or request.sessionId,
            senderId=request.senderId or request.userId,
            model="model_manager",
            costTime=failed_cost,
            errorType=type(e).__name__,
        )
        logger.exception("generate reply failed: %s", e)
        # C-F1 fix: 动态读取 app.config.failover_mgr，而非导入时绑定的 None
        _fmgr = _failover_mgr()
        if _fmgr:
            try:
                fallback_provider = await _fmgr.check_and_failover()
                if fallback_provider:
                    logger.info("故障转移至: %s", fallback_provider)
            except Exception as fe:
                logger.warning("故障转移失败: %s", fe)
        # 安全：不把内部异常字符串返回给客户端（信息泄露），
        # 真实详情已写入日志（含 exc_info=True），客户端只收到通用消息。
        raise HTTPException(status_code=500, detail="生成回复失败，请稍后重试") from e


# ═══════════════════════════════════════════
# 角色上下文辅助函数
# ═══════════════════════════════════════════


async def _prepare_character_turn(
    request: MessageRequest, character_id: str, *, character_service=None, execute_memory_operations: bool = False,
    defer_memory_operations: bool = False
):
    """准备角色上下文；任何失败都降级为无角色上下文的旧行为。

    返回 prepared_turn | None。
    用户范围非法（如管理台测试无 senderId）、画像缺失或数据库
    故障都不应让整条消息失败。

    character_service 由 HTTP 层注入（绑定当前应用容器的数据库）；
    为 None 时回退全局默认服务（bot 直连等非 HTTP 调用方）。
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
    except Exception as e:
        logger.warning("角色上下文准备失败，按无角色上下文继续 character=%s: %s", character_id, e)
        return None


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
        if capture == "failed":
            return "这条信息的长期记忆保存失败；本次回复不代表已保存，请稍后重试。"
        if capture in {"revoked", "stale"}:
            return "这条信息未保存到长期记忆：相关记忆已删除，本次回复不会恢复它。"
        if capture == "conflict":
            return "这条信息未保存到长期记忆：同一条消息已绑定其他内容，请重新发送。"
        return None
    except Exception as e:
        logger.warning("角色记忆回写失败 character=%s: %s", prepared.character_id, e)
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
    """Compose the HTTP service from the owning application's runtime.

    生成链路必须完全绑定当前应用容器的数据库：人物上下文服务
    （记忆/关系/历史）与消息持久化/调用记录/生成配置读取都要走
    容器数据库。只注入人物服务时，上一轮消息仍写入全局库，下一轮
    从容器库读不到，历史会断裂。容器数据库即全局单例时复用
    模块级缓存服务。
    """

    from services.character_context import build_character_context_service

    container = get_runtime_container(request.app)
    runtime = inference_runtime if container.inference_runtime is None else container.inference_runtime
    character_service = None
    message_db = None
    if container.db is not db:
        character_service = build_character_context_service(container.db)
        message_db = container.db
    if runtime is inference_runtime and character_service is None:
        return get_chat_generation_service()
    return _build_chat_generation_service(runtime, character_service, message_db)


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


async def _retrieve_rag_bundle(query: str, top_k: int, filters: dict[str, Any] | None) -> dict[str, Any]:
    """Retrieve curated character knowledge, then use the generic KB fallback."""

    def retrieve() -> dict[str, Any]:
        if not filters:
            from knowledge.multiscale_rag.runtime import get_multiscale_rag_service
            from knowledge.retrieval_core.query import QueryAnalyzer

            character_rag = get_multiscale_rag_service()
            matched = QueryAnalyzer([character_rag.config]).analyze(query).matched_domains
            # retrieve_with_citations 内部完成惰性加载和域门控。不能先用
            # is_available() 短路，否则冷启动首条请求无法触发索引加载。
            bundle = character_rag.retrieve_with_citations(query, top_k=top_k, filters=filters)
            if bundle is not None:
                return bundle
            if matched:
                raise RuntimeError("Requested character knowledge domain is unavailable")

        from api.knowledge import _ensure_vector_index
        from knowledge.rag_helper import get_rag_helper

        # A normal generation does not necessarily follow the search API.
        # Resolve dirty metadata/content before consulting either cached RAG
        # path; an incomplete rebuild cannot authorize stale source evidence.
        if not _ensure_vector_index():
            raise RuntimeError("Generic knowledge index is not ready")

        if os.getenv("CORRECTIVE_RAG_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}:
            from knowledge.corrective_rag import get_corrective_rag

            return get_corrective_rag().retrieve_with_correction(query, top_k=top_k, filters=filters)
        return get_rag_helper().retrieve_with_citations(query, top_k=top_k, filters=filters)

    return await asyncio.to_thread(retrieve)


def _rag_retrieval_timeout() -> float:
    """Allow character-index cold start without weakening steady-state limits."""

    try:
        from knowledge.multiscale_rag.runtime import get_multiscale_rag_service

        if not get_multiscale_rag_service().is_warm():
            return max(_RAG_TIMEOUT, _RAG_COLD_START_TIMEOUT)
    except Exception:  # noqa: BLE001 - retrieval owns the fallback/error handling
        pass
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
    _use_kb = _cfg.get("useKnowledgeBase", True) and enable_rag

    # RAG 检索（受设置页 useKnowledgeBase 开关控制）
    rag_context = ""
    rag_meta: dict[str, Any] = {}
    citations_enabled = os.getenv("RAG_CITATIONS_ENABLED", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    retrieval = RetrievalResult()
    filters = None
    # An executable personal-field read has no external knowledge dependency.
    # Use the same evidence contract as generation, not another keyword router.
    memory_lookup_ready = False
    personal_lookup_only = False
    operation_response_ready = False
    if _use_kb and prepared_character_turn is not None:
        from character.memory_operation import render_operation_response
        from character.memory_query import lookup_fields, storage_fields
        from inference.current_constraint_response import render_current_constraint_response
        from inference.memory_response import render_complete_memory_read, render_memory_response

        operation_response_ready = (
            render_operation_response(
                request.message, getattr(prepared_character_turn.compiled, "memory_operation_receipt", None)
            )
            is not None
        )

        # Dependency routing is independent of whether a saved answer exists.
        # A personal lookup can still need generation over visible history;
        # external story knowledge cannot resolve missing private user data.
        personal_lookup_only = bool(lookup_fields(request.message) or storage_fields(request.message)) and not (
            prepared_character_turn.compiled.branch_context
        )
        memory_lookup_ready = (
            render_current_constraint_response(request.message, prepared_character_turn.compiled) is not None
            or render_memory_response(
                request.message,
                prepared_character_turn.compiled,
                history=tuple(request.history or []) or prepared_character_turn.history,
            )
            is not None
            or render_complete_memory_read(
                request.message,
                prepared_character_turn.compiled,
                tuple(request.history or []) or prepared_character_turn.history,
            )
            is not None
        )
    if _use_kb and not (memory_lookup_ready or personal_lookup_only or operation_response_ready):
        try:
            from knowledge.dialogue_query import contextual_retrieval_query
            from knowledge.intent_detector import needs_rag
            from knowledge.retrieval_core.registry import get_default_registry

            effective_history = tuple(request.history or []) or (
                prepared_character_turn.history if prepared_character_turn else ()
            )
            from character.memory_operation import split_operation_request

            operation_tasks = (
                split_operation_request(request.message)
                if prepared_character_turn is not None
                and getattr(prepared_character_turn, "memory_operation_receipt", None) is not None
                else None
            )
            rag_message = operation_tasks[1] if operation_tasks else request.message
            retrieval_query = contextual_retrieval_query(
                rag_message, effective_history, get_default_registry().list_domains()
            )

            need_rag, _, kb_name = await asyncio.wait_for(
                asyncio.to_thread(needs_rag, retrieval_query),
                timeout=_RAG_TIMEOUT,
            )
            if need_rag:
                if kb_name:
                    kb_id = await asyncio.to_thread(_resolve_kb_id, kb_name, database=message_db)
                    if kb_id is not None:
                        filters = {"knowledge_base_id": kb_id}
                        logger.info("RAG路由: 消息→「%s」(id=%s)", kb_name, kb_id)

                bundle = await asyncio.wait_for(
                    _retrieve_rag_bundle(retrieval_query, 3, filters),
                    timeout=_rag_retrieval_timeout(),
                )
                rag_meta = {
                    "citations": bundle.get("citations", []) if citations_enabled else [],
                    "confidence": bundle.get("confidence"),
                    "abstained": bundle.get("abstained", False),
                    "answerMode": ("abstention" if bundle.get("abstained", False) else "grounded_answer"),
                    "domainId": next(iter(bundle.get("domains") or []), None),
                    "warnings": bundle.get("warnings") or None,
                    "modelInvoked": True,
                }
                from knowledge.query_tasks import requests_only_source_excerpt

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
                        source_lookup=source_lookup,
                    )

                # 角色知识检索结果自带按粒度组装的 context_text；
                # 通用知识库结果继续使用 RAGHelper 的格式化器。
                else:
                    character_knowledge_context = bundle.get("context_text") or ""
                    if character_knowledge_context or bundle.get("retrieval_strategy") == "multi_scale_character":
                        rag_context = character_knowledge_context
                    else:
                        from knowledge.rag_helper import get_rag_helper

                        rag_context = get_rag_helper().format_context_results(bundle.get("results", []))
                    retrieval = RetrievalResult(
                        status="ok" if rag_context else "character_abstention",
                        reason="" if rag_context else "evidence_budget_exhausted",
                        evidence=rag_context,
                        evidence_packets=tuple(bundle.get("evidence_packets", ())),
                        identity_task=bundle.get("identity_task") or {},
                        identity_subtask=bundle.get("identity_subtask") or {},
                        documents=tuple(bundle.get("results", [])),
                        citations=tuple(rag_meta.get("citations", [])),
                        confidence=bundle.get("confidence"),
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
            logger.warning("RAG retrieval failed: %s", e)
            retrieval = RetrievalResult(status="character_abstention", reason="retrieval_unavailable")
            rag_meta = {
                "citations": [],
                "confidence": None,
                "abstained": True,
                "answerMode": "abstention",
                "modelInvoked": True,
                "warnings": ["retrieval_unavailable"],
            }

    model_generate = model_generate or _vllm_client.generate

    async def generate_reply(**kwargs):
        if retrieval.status != "character_abstention":
            return await model_generate(**kwargs)
        try:
            reply = await model_generate(**kwargs)
            if isinstance(reply, str) and reply.strip():
                return reply
            rag_meta["generationError"] = "EmptyModelReply"
        except Exception as exc:
            rag_meta["generationError"] = type(exc).__name__
            logger.warning("Character abstention generation failed; using fallback", exc_info=True)
        rag_meta["warnings"] = [*(rag_meta.get("warnings") or []), "character_abstention_fallback"]
        return _RAG_ABSTENTION_REPLY

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
            independent_tasks_enabled=os.getenv("CHARACTER_INDEPENDENT_TASKS_ENABLED", "false").lower() == "true",
            # 编译后的角色上下文：画像/关系/情景/决策进系统提示词，
            # 长期记忆只进用户消息的不可信参考区。
            character_context=(prepared_character_turn.compiled if prepared_character_turn else None),
            reply_guard=getattr(prepared_character_turn, "reply_guard", None),
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
    if generation.plan.retrieval.answer_citations_bound:
        rag_meta["citations"] = list(generation.response_citations) if citations_enabled else []
    if not getattr(generation, "model_invoked", True):
        rag_meta["modelInvoked"] = False
        rag_meta["answerMode"] = generation.response_mode
        rag_meta["abstained"] = generation.response_mode == "source_unavailable"
        rag_meta["citations"] = list(generation.response_citations) if citations_enabled else []
    elif generation.response_mode == "task_composite":
        rag_meta["answerMode"] = "task_composite"
    return generation.reply, generation.plan.retrieval.has_evidence or bool(rag_meta.get("abstained")), rag_meta


async def _generate_with_model_manager_character(
    request: MessageRequest,
    lora_name: str,
    runtime_config: dict[str, Any] | None,
    *,
    prepared_character_turn,
    model_manager,
) -> tuple[str, float]:
    """回退（模型管理器）路径的角色上下文生成。

    与 vLLM 路径共用 build_generation_request 的同一套编译：
    画像/关系/情景/决策进系统提示词，长期记忆与对话者昵称进
    用户消息的不可信参考区，request.history 为空时使用数据库
    加载的完整历史。model_manager.async_generate 只接受
    prompt + session_history，因此把编译后的消息列表拆成
    session_history（系统提示词+历史）与最后一条用户消息。
    """
    _cfg = runtime_config or {}
    generation_request = CharacterGenerationRequest(
        message=request.message,
        history=(tuple(request.history or []) if (request.history or []) else prepared_character_turn.history),
        persona_prompt=_get_system_prompt(lora_name),
        interlocutor=request.senderName or request.userName or "普通用户",
        character_context=prepared_character_turn.compiled,
        reply_guard=getattr(prepared_character_turn, "reply_guard", None),
        lora_name=lora_name if lora_name != "default" else None,
        temperature=float(_cfg.get("temperature", os.getenv("VLLM_TEMPERATURE", "0.7"))),
        max_tokens=int(_cfg.get("maxTokens", os.getenv("VLLM_MAX_TOKENS", "2048"))),
        top_p=float(_cfg.get("topP", os.getenv("VLLM_TOP_P", "0.9"))),
        context_window_tokens=get_provider_context_budget(model_manager).window_tokens,
    )
    total_cost = 0.0

    async def _model_manager_adapter(*, messages, max_tokens, **_kwargs):
        nonlocal total_cost
        reply, call_cost = await model_manager.async_generate(
            prompt=messages[-1]["content"],
            session_history=messages[:-1],
            rag_docs=None,
            max_tokens_override=max_tokens,
        )
        total_cost += call_cost
        return reply

    generation = await generate_character_response(generation_request, _model_manager_adapter)
    return generation.reply, round(total_cost, 2)


def _get_system_prompt(lora_name: str) -> str:
    """获取 LoRA 对应的系统提示词"""
    try:
        # H1 fix: 此前从 bot.bot 导入 LORA_REGISTRY，造成 API 层反向依赖 bot 层。
        # 现从 inference.lora_registry 中立层导入，依赖方向：api → inference。
        from inference.lora_registry import get_lora_system_prompt

        return get_lora_system_prompt(lora_name)
    except Exception as e:
        logger.warning("获取系统提示词失败: %s", e)
        return ""


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
    """记录模型调用；database 为容器注入的数据库，None 时用全局单例。"""
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
        logger.warning("Failed to save model invocation traceId=%s error=%s", request.traceId, e)


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
) -> bool:
    """Save generated replies with platform-aware metadata.

    返回保存是否成功：调用方以此决定是否回写人物状态（记忆/关系），
    保存失败时不得继续 complete_turn，否则消息记录与人物状态会脱节。
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
        if saved_receipt is not None and isinstance(stored, dict) and stored.get("id"):
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
            e,
        )
        return False


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
