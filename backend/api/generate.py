"""消息生成API - 支持vLLM高并发推理"""
import asyncio
import hashlib
import json
import os
import time
import uuid
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Request
from app.dependencies import get_current_admin, get_current_user
from app.runtime import get_runtime_container
from db.schemas import MessageRequest, GenerateResponse
from infra.concurrency_control import InferenceQueueFull, RateLimitExceeded, inference_runtime
from infra.security_utils import strip_control_chars
from inference.lora_utils import resolve_lora_served_name
from inference.prompt_policy import build_grounded_user_message, compose_system_prompt
from infra.observability import increment, log_event, set_consecutive
from services.chat_generation import ChatGenerationService

from db.adapter import db
from db.database import get_lora_path_by_id
from app.config import (
    get_llm_semaphore, circuit_breaker_registry,
    response_cache,
    INPUT_VALIDATOR_AVAILABLE,
    CIRCUIT_BREAKER_AVAILABLE,
    is_vllm_enabled, get_vllm_served_model_name,
)
# C-F1 fix: failover_mgr 在 lifespan 中通过 app.config.failover_mgr = ...
# 赋值，导入时绑定到 None 会永远看不到实例。改为动态访问模块属性。
from app import config as _app_config
_failover_mgr = lambda: _app_config.failover_mgr

if INPUT_VALIDATOR_AVAILABLE:
    from infra.input_validator import InputValidator, MESSAGE_SCHEMA

logger = logging.getLogger(__name__)

# ── vLLM 客户端（延迟初始化） ──
_vllm_client = None
_vllm_initialized = False
_vllm_init_lock: asyncio.Lock | None = None
_vllm_init_lock_loop: asyncio.AbstractEventLoop | None = None

_RAG_TIMEOUT = float(os.getenv("RAG_TIMEOUT", "8"))
_DB_WRITE_TIMEOUT = float(os.getenv("DB_WRITE_TIMEOUT", "3"))
_MODEL_INFERENCE_TIMEOUT = float(os.getenv("MODEL_INFERENCE_TIMEOUT", "180"))
_RAG_ABSTENTION_REPLY = (
    os.getenv("RAG_ABSTENTION_REPLY", "").strip()
    or "我没有找到足够可靠的信息，暂时无法回答这个问题。"
)
_local_model_lock: asyncio.Lock | None = None
_local_model_lock_loop: asyncio.AbstractEventLoop | None = None

_HIGH_RISK_PROMPT_PATTERNS = (
    "export config", "dump config", "show config", "read .env", "cat .env",
    "read secret", "show secret", "read token", "show token", "print env",
    "reveal system prompt", "show system prompt", "ignore previous instructions and export",
    "\u5bfc\u51fa\u914d\u7f6e", "\u8bfb\u53d6\u914d\u7f6e", "\u663e\u793a\u914d\u7f6e",
    "\u8bfb\u53d6\u5bc6\u94a5", "\u663e\u793a\u5bc6\u94a5",
    "\u8bfb\u53d6token", "\u663e\u793atoken", "\u8bfb\u53d6.env",
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


def _resolve_kb_id(kb_name: str):
    """根据知识库名称查询其ID，用于RAG检索过滤

    优先从意图分类器模型的config中读取映射（训练时保存），
    回退到数据库实时查询。
    """
    # 优先从模型config读取（训练时保存的映射，避免每次查库）
    try:
        from pathlib import Path
        config_path = Path(__file__).parent.parent / "intent_classifier_model" / "config.json"
        if config_path.exists():
            import json
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            kb_name_to_id = config.get("kb_name_to_id", {})
            if kb_name in kb_name_to_id:
                return kb_name_to_id[kb_name]
    except Exception as e:
        logger.debug(f"从模型config读取KB映射失败: {e}")

    # 回退到数据库查询
    try:
        from db.adapter import db
        bases = db.get_knowledge_bases()
        for b in bases:
            if b["name"] == kb_name:
                return b["id"]
    except Exception as e:
        logger.warning(f"数据库查询KB ID失败: {e}")

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


def _response_cache_keys(request: MessageRequest, lora_name: str, config: Dict[str, Any]) -> tuple[str, str, int]:
    """Include every response-affecting setting in the cache identity."""
    model_name = get_vllm_served_model_name()
    use_knowledge_base = bool(config.get("useKnowledgeBase", True))
    identity = {
        "model": model_name,
        "lora": lora_name,
        "temperature": float(config.get("temperature", os.getenv("VLLM_TEMPERATURE", "0.7"))),
        "max_tokens": int(config.get("maxTokens", os.getenv("VLLM_MAX_TOKENS", "2048"))),
        "use_knowledge_base": use_knowledge_base,
        "platform": request.platform,
        "conversation_type": request.conversationType or request.sessionType,
        "conversation_id": request.conversationId or request.sessionId,
    }
    prompt_hash = hashlib.sha256(request.message.encode("utf-8")).hexdigest()
    serialized = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    cache_key = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return prompt_hash, cache_key, 60 if use_knowledge_base else 300


async def _generate_reply_impl(request: MessageRequest, current_user: dict | None = None):
    """默认聊天生成实现：优先使用 vLLM，回退到模型管理器。"""
    # C11 fix: 主聊天端点 mock provider 防护
    # 此前仅 training/generate-dialogues 有此检查，主聊天端点遗漏，
    # 导致生产环境默认 mock 时静默返回罐头回复。现统一拦截。
    from inference.model_manager import get_model_manager
    _mgr = get_model_manager()
    if _mgr._current_provider.value == "mock":
        raise HTTPException(
            status_code=503,
            detail="当前模型提供商为 mock 模式，无法提供真实推理。"
                   "请在设置页面配置有效的模型提供商（如 vLLM / DeepSeek API / Ollama）。"
        )

    # 输入验证
    if INPUT_VALIDATOR_AVAILABLE:
        is_valid, errors = InputValidator.validate(request.model_dump(), MESSAGE_SCHEMA)
        if not is_valid:
            raise HTTPException(status_code=422, detail={"message": "输入验证失败", "errors": errors})

    try:
        runtime_config, loras = await asyncio.gather(
            asyncio.to_thread(lambda: db.config or {}),
            asyncio.to_thread(lambda: list(db.loras)),
        )
    except Exception:
        logger.warning("读取生成配置失败，使用安全默认值", exc_info=True)
        runtime_config, loras = {}, []

    # 获取LoRA：始终计算 active_lora，优先使用前端指定的，否则使用当前激活的
    active_lora = next((l for l in loras if l["status"] == "active"), None)
    selected_lora = active_lora

    if not request.loraName:
        try:
            raw_router_config = runtime_config.get("lora_router_config", {"enabled": False})
            router_config = (
                json.loads(raw_router_config)
                if isinstance(raw_router_config, str)
                else raw_router_config
            )
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
            logger.warning("LoRA routing failed; using explicit/active adapter traceId=%s", request.traceId, exc_info=True)
    if request.loraName:
        selected_lora = next((item for item in loras if item["name"] == request.loraName), None)
        if selected_lora is None:
            raise HTTPException(status_code=422, detail="Specified LoRA does not exist")
        lora_name = selected_lora["name"]
    else:
        lora_name = selected_lora["name"] if selected_lora else "default"

    vllm_lora_name = resolve_lora_served_name(lora_name) if lora_name != "default" else "default"

    # 检查vLLM是否实际支持该LoRA，避免404触发熔断
    vllm_effective_lora = vllm_lora_name if lora_name != "default" else None
    if vllm_effective_lora and await _ensure_vllm() and _vllm_client:
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

    prompt_hash = cache_key = ""
    cache_ttl = 300
    if response_cache:
        try:
            prompt_hash, cache_key, cache_ttl = _response_cache_keys(request, lora_name, runtime_config)
            cached = await response_cache.get(prompt_hash, cache_key)
            if cached:
                logger.debug("response cache hit")
                return GenerateResponse(**cached)
        except Exception as e:
            logger.warning("response cache read failed: %s", e)

    start_time = time.time()

    # ── 优先使用 vLLM 高并发推理 ──
    if await _ensure_vllm() and _vllm_client:
        try:
            reply, used_rag, rag_meta = await _generate_with_vllm(request, vllm_effective_lora, vllm_lora_name, runtime_config)
            cost_time = round(time.time() - start_time, 2)

            model_invoked = rag_meta.get("modelInvoked", True) is not False
            model_label = (
                f"vllm/{get_vllm_served_model_name()}"
                if model_invoked
                else "rag/abstained"
            )
            stored_model_name = "vllm" if model_invoked else model_label
            stored_lora_name = lora_name if model_invoked else "default"
            if model_invoked:
                await _record_model_invocation(
                    request,
                    model_label,
                    lora_name,
                    cost_time,
                    used_rag=used_rag,
                    completion_text=reply,
                )
                set_consecutive("model_failure", True)
            await _save_message(
                request,
                reply,
                stored_model_name,
                stored_lora_name,
                cost_time,
            )
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
                reply=reply,
                model=model_label,
                costTime=cost_time,
                citations=rag_meta.get("citations"),
                confidence=rag_meta.get("confidence"),
                abstained=rag_meta.get("abstained", False),
            )
            if response_cache:
                try:
                    await response_cache.set(
                        prompt_hash, cache_key, result.model_dump(), ttl=cache_ttl
                    )
                except Exception as e:
                    logger.warning(f"vLLM缓存写入失败: {e}")
                    pass

            return result
        except Exception as e:
            failed_cost = round(time.time() - start_time, 2)
            model_label = f"vllm/{get_vllm_served_model_name()}"
            await _record_model_invocation(
                request,
                model_label,
                lora_name,
                failed_cost,
                used_rag=False,
                error_type=type(e).__name__,
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
            logger.warning(f"vLLM inference failed, falling back to model manager: {e}")
            if vllm_effective_lora:
                raise HTTPException(
                    status_code=503,
                    detail="所选 LoRA 推理失败，请检查 vLLM 适配器状态",
                ) from e

    # ── 回退：使用原有模型管理器 ──
    try:
        from inference.model_manager import get_model_manager, ModelProvider
        model_manager = get_model_manager()

        semaphore = get_llm_semaphore()
        sem_acquired = False
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=30.0)
            sem_acquired = True
        except asyncio.TimeoutError:
            raise HTTPException(status_code=503, detail="服务繁忙，请稍后再试")

        try:
            async with _loop_local_lock("local_model"):
                # C-R1 fix: 原先 LORA_PATH_MAP 恒为空 dict，此分支永不进入。
                # 改为调用 get_lora_path_by_id() 动态查找 LoRA 路径。
                lora_path = None
                if selected_lora:
                    lora_path = get_lora_path_by_id(selected_lora["id"])
                if lora_path:
                    if ModelProvider.TRANSFORMERS_PEFT in model_manager._providers:
                        peft_provider = model_manager._providers[ModelProvider.TRANSFORMERS_PEFT]
                        if hasattr(peft_provider, 'set_lora_adapter'):
                            peft_provider.set_lora_adapter(lora_path)
                        model_manager.set_provider(ModelProvider.TRANSFORMERS_PEFT)
                else:
                    model_manager.set_lora_adapter(None)

                async def _do_generate_async():
                    # P0-C1 fix: 直接 await 原生 async_generate，禁止跨事件循环
                    # 复用缓存的 httpx.AsyncClient（曾用 asyncio.to_thread → asyncio.run
                    # 创建新循环，第二次请求会报 RuntimeError: Event loop is closed）。
                    return await model_manager.async_generate(
                        prompt=request.message,
                        session_history=[],
                        rag_docs=None
                    )

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


        await _record_model_invocation(request, model_name, lora_name, cost_time, used_rag=False, completion_text=reply)
        await _save_message(request, reply, model_name, lora_name, cost_time)
        set_consecutive("model_failure", True)
        log_event("message_generated", traceId=request.traceId, platform=request.platform, conversationId=request.conversationId or request.sessionId, senderId=request.senderId or request.userId, model=model_name, costTime=cost_time, errorType="", usedRag=False)

        result = GenerateResponse(
            reply=reply,
            model=f"{model_name} ({current_provider})",
            costTime=cost_time
        )

        if response_cache:
            try:
                await response_cache.set(
                    prompt_hash, cache_key, result.model_dump(), ttl=cache_ttl
                )
            except Exception as e:
                logger.warning(f"模型管理器缓存写入失败: {e}")
                pass

        return result

    except HTTPException:
        raise
    except Exception as e:
        failed_cost = round(time.time() - start_time, 2) if "start_time" in locals() else 0.0
        await _record_model_invocation(
            request,
            "model_manager",
            lora_name if "lora_name" in locals() else "default",
            failed_cost,
            used_rag=False,
            error_type=type(e).__name__,
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
        logger.error(f"generate reply failed: {e}", exc_info=True)
        # C-F1 fix: 动态读取 app.config.failover_mgr，而非导入时绑定的 None
        _fmgr = _failover_mgr()
        if _fmgr:
            try:
                fallback_provider = await _fmgr.check_and_failover()
                if fallback_provider:
                    logger.info(f"故障转移至: {fallback_provider}")
            except Exception as fe:
                logger.warning(f"故障转移失败: {fe}")
        # 安全：不把内部异常字符串返回给客户端（信息泄露），
        # 真实详情已写入日志（含 exc_info=True），客户端只收到通用消息。
        raise HTTPException(status_code=500, detail="生成回复失败，请稍后重试")


# ═══════════════════════════════════════════
# vLLM 推理辅助函数
# ═══════════════════════════════════════════


def _build_chat_generation_service(runtime) -> ChatGenerationService:
    return ChatGenerationService(
        generate_handler=_generate_reply_impl,
        inference_runtime=runtime,
        sanitize_message=strip_control_chars,
        is_high_risk_prompt=_is_high_risk_prompt,
        security_response_factory=_security_policy_response,
        trace_id_factory=lambda: uuid.uuid4().hex,
    )


_chat_generation_service = _build_chat_generation_service(inference_runtime)


def get_chat_generation_service() -> ChatGenerationService:
    """Return the default service for non-HTTP compatibility callers."""

    return _chat_generation_service


def get_request_chat_generation_service(request: Request) -> ChatGenerationService:
    """Compose the HTTP service from the owning application's runtime."""

    container = get_runtime_container(request.app)
    runtime = (
        inference_runtime
        if container.inference_runtime is None
        else container.inference_runtime
    )
    if runtime is inference_runtime:
        return get_chat_generation_service()
    return _build_chat_generation_service(runtime)


async def generate_reply_core(
    request: MessageRequest,
    current_user: dict | None = None,
):
    """Compatibility entry point used by integrations and existing callers."""

    return await get_chat_generation_service().generate(request, current_user)


@router.post("/api/generate")
async def generate_reply(
    request: MessageRequest,
    current_user: dict = Depends(get_current_user),
    service: ChatGenerationService = Depends(get_request_chat_generation_service),
):
    """Queue-protected management/test generation endpoint."""
    try:
        return await service.generate_queued(request, current_user)
    except RateLimitExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail="请求过于频繁，请稍后重试",
            headers={"Retry-After": str(max(1, int(exc.retry_after)))},
        )
    except InferenceQueueFull:
        raise HTTPException(status_code=503, detail="推理队列已满，请稍后再试")
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="推理排队超时，请稍后再试")

async def _retrieve_rag_bundle(query: str, top_k: int, filters: Dict[str, Any] | None) -> Dict[str, Any]:
    """Run one retrieval pass and return context evidence plus confidence metadata."""
    def retrieve() -> Dict[str, Any]:
        from knowledge.rag_helper import get_rag_helper

        if os.getenv("CORRECTIVE_RAG_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}:
            from knowledge.corrective_rag import get_corrective_rag
            return get_corrective_rag().retrieve_with_correction(query, top_k=top_k, filters=filters)
        return get_rag_helper().retrieve_with_citations(query, top_k=top_k, filters=filters)

    return await asyncio.to_thread(retrieve)

async def _generate_with_vllm(request: MessageRequest, lora_name: str | None, prompt_lora_name: str | None = None, runtime_config: Dict[str, Any] | None = None) -> tuple[str, bool, Dict[str, Any]]:
    """使用 vLLM 客户端生成回复，返回 (reply, used_rag, rag_meta)。"""
    # 使用请求开始时读取的配置快照，避免同一次生成多次同步访问数据库。
    _cfg = runtime_config or {}
    _temperature = float(_cfg.get('temperature', os.getenv("VLLM_TEMPERATURE", "0.7")))
    _max_tokens = int(_cfg.get('maxTokens', os.getenv("VLLM_MAX_TOKENS", "2048")))
    _use_kb = _cfg.get('useKnowledgeBase', True)

    # RAG 检索（受设置页 useKnowledgeBase 开关控制）
    rag_context = ""
    rag_meta: Dict[str, Any] = {}
    filters = None
    if _use_kb:
        try:
            from knowledge.intent_detector import needs_rag
            need_rag, _, kb_name = await asyncio.wait_for(
                asyncio.to_thread(needs_rag, request.message),
                timeout=_RAG_TIMEOUT,
            )
            if need_rag:
                if kb_name:
                    kb_id = await asyncio.to_thread(_resolve_kb_id, kb_name)
                    if kb_id is not None:
                        filters = {"knowledge_base_id": kb_id}
                        logger.info("RAG路由: 消息→「%s」(id=%s)", kb_name, kb_id)

                bundle = await asyncio.wait_for(
                    _retrieve_rag_bundle(request.message, 3, filters),
                    timeout=_RAG_TIMEOUT,
                )
                citations_enabled = os.getenv("RAG_CITATIONS_ENABLED", "true").strip().lower() in {
                    "1", "true", "yes", "on"
                }
                rag_meta = {
                    "citations": bundle.get("citations", []) if citations_enabled else [],
                    "confidence": bundle.get("confidence"),
                    "abstained": bundle.get("abstained", False),
                    "modelInvoked": True,
                }
                if bundle.get("abstained", False):
                    # A low-confidence retrieval must not fall through to an
                    # evidence-free model answer while reporting abstained=true.
                    rag_meta["modelInvoked"] = False
                    return _RAG_ABSTENTION_REPLY, True, rag_meta

                from knowledge.rag_helper import get_rag_helper
                rag_context = get_rag_helper().format_context_results(bundle.get("results", []))
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
            rag_meta = {}

    persona_prompt = _get_system_prompt(prompt_lora_name or lora_name)
    system_prompt = compose_system_prompt(persona_prompt, include_rag=bool(rag_context))
    messages = [{"role": "system", "content": system_prompt}]

    # RAG知识注入user消息
    user_content = build_grounded_user_message(request.message, rag_context, max_chars=800)
    messages.append({"role": "user", "content": user_content})

    # RAG命中时适当降低温度以更忠实于检索内容
    _gen_temperature = min(_temperature, 0.5) if rag_context else _temperature

    # 调用 vLLM
    reply = await _vllm_client.generate(
        messages=messages,
        lora_name=lora_name if lora_name != "default" else None,
        temperature=_gen_temperature,
        max_tokens=_max_tokens,
    )

    return reply, bool(rag_context), rag_meta


def _get_system_prompt(lora_name: str) -> str:
    """获取 LoRA 对应的系统提示词"""
    try:
        # H1 fix: 此前从 bot.bot 导入 LORA_REGISTRY，造成 API 层反向依赖 bot 层。
        # 现从 inference.lora_registry 中立层导入，依赖方向：api → inference。
        from inference.lora_registry import get_lora_system_prompt
        return get_lora_system_prompt(lora_name)
    except Exception as e:
        logger.warning(f"获取系统提示词失败: {e}")
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
):
    try:
        prompt_tokens = _estimate_tokens(request.message)
        completion_tokens = 0 if error_type else _estimate_tokens(completion_text)
        await asyncio.wait_for(
            asyncio.to_thread(db.add_model_invocation, {
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
            }),
            timeout=_DB_WRITE_TIMEOUT,
        )
    except Exception as e:
        increment("db_write_failures")
        log_event("db_write_failed", level="warning", traceId=request.traceId, platform=request.platform, conversationId=request.conversationId or request.sessionId, senderId=request.senderId or request.userId, model=model_name, costTime=cost_time, errorType=type(e).__name__)
        logger.warning("Failed to save model invocation traceId=%s error=%s", request.traceId, e)


async def _save_message(request: MessageRequest, reply: str, model_name: str, lora_name: str, cost_time: float):
    """Save generated replies with platform-aware metadata."""
    try:
        await asyncio.wait_for(
            asyncio.to_thread(db.add_message, {
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
                "costTime": cost_time,
            }),
            timeout=_DB_WRITE_TIMEOUT,
        )
    except Exception as e:
        increment("db_write_failures")
        log_event("db_write_failed", level="warning", traceId=request.traceId, platform=request.platform, conversationId=request.conversationId or request.sessionId, senderId=request.senderId or request.userId, model=model_name, costTime=cost_time, errorType=type(e).__name__)
        logger.warning(
            "Failed to save message record traceId=%s sessionId=%s error=%s",
            request.traceId,
            request.sessionId,
            e,
        )


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
