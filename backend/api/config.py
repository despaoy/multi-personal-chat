"""配置与模型状态API"""

from json import JSONDecodeError

from fastapi import APIRouter, Request, HTTPException, Depends
from pydantic import ValidationError
from inference.generation_options import GenerationOptions
from app.dependencies import get_current_admin, get_request_database

from infra.input_validator import CONFIG_SCHEMA, InputValidator
from cache.config_cache import invalidate_config_cache, read_config
from infra.db_executor import run_db

router = APIRouter()

# 敏感字段关键词（匹配任一即脱敏）
_SENSITIVE_KEYWORDS = ("apikey", "api_key", "secret", "password", "token")
# 脱敏标记，用于PUT时识别并跳过未修改的字段
_MASK_MARKER = "****"
_SERVER_ONLY_CONFIG_KEYS = frozenset(
    {
        "astrbotIntegrationToken",
        "astrbotIntegrationTokens",
    }
)
_SERVER_ONLY_CONFIG_KEYS_NORMALIZED = frozenset(key.lower() for key in _SERVER_ONLY_CONFIG_KEYS)


def _is_sensitive_key(key: str) -> bool:
    return key != "maxTokens" and any(word in key.lower() for word in _SENSITIVE_KEYWORDS)


def _mask_value(value: str) -> str:
    """对敏感值脱敏：长度>8显示前4+****+后4，否则整体替换为****"""
    if not value:
        return ""
    if len(value) <= 8:
        return _MASK_MARKER
    return value[:4] + _MASK_MARKER + value[-4:]


def _mask_config(config: dict) -> dict:
    """返回脱敏后的配置副本（不影响缓存中的原始值）"""
    masked = {}
    for k, v in config.items():
        if str(k).lower() in _SERVER_ONLY_CONFIG_KEYS_NORMALIZED:
            continue
        if _is_sensitive_key(k) and v:
            masked[k] = _mask_value(str(v))
        else:
            masked[k] = v
    return masked


# ============================================
# 系统配置
# ============================================


async def _request_json(request: Request):
    try:
        return await request.json()
    except JSONDecodeError as exc:
        raise HTTPException(400, "请求必须是合法 JSON") from exc


@router.get("/api/config")
async def get_config(current_user: dict = Depends(get_current_admin), *, database=Depends(get_request_database)):
    """Read a process-local TTL snapshot, with sensitive values masked."""
    try:
        config, cached = await run_db(read_config, database)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="配置读取失败，请检查配置数据库后重试") from exc
    return {"config": _mask_config(config), "cached": cached}


@router.put("/api/config")
async def update_config(
    request: Request, current_user: dict = Depends(get_current_admin), *, database=Depends(get_request_database)
):
    """更新系统配置（含脱敏标记的字段自动跳过，保留原值）

    C-S1 fix: 系统配置含 API key 等敏感项，限定 admin。
    """
    new_config = await _request_json(request)
    if not isinstance(new_config, dict):
        raise HTTPException(status_code=422, detail="Configuration payload must be an object")
    server_only = {str(key) for key in new_config if str(key).lower() in _SERVER_ONLY_CONFIG_KEYS_NORMALIZED}
    if server_only:
        raise HTTPException(
            status_code=422,
            detail="AstrBot integration tokens are server-only environment variables",
        )
    try:
        numeric_updates = GenerationOptions(
            **{key: new_config[key] for key in GenerationOptions.model_fields if key in new_config}
        ).model_dump(exclude_unset=True)
    except ValidationError as exc:
        raise HTTPException(
            422,
            detail={
                "message": "生成参数配置无效，请修改标出的字段后重试。",
                "fields": sorted({error["loc"][0] for error in exc.errors(include_input=False)}),
            },
        ) from exc
    # 过滤掉含脱敏标记的未修改字段（前端回传的脱敏值不应覆盖真实凭证）
    new_config = {
        k: v for k, v in new_config.items() if not (_is_sensitive_key(k) and isinstance(v, str) and _MASK_MARKER in v)
    }
    new_config.update(numeric_updates)
    # 输入验证 - 逐字段验证键值对
    for key, value in new_config.items():
        entry = {"key": str(key), "value": str(value)}
        is_valid, errors = InputValidator.validate(entry, CONFIG_SCHEMA)
        if not is_valid:
            raise HTTPException(status_code=422, detail={"message": f"配置项 '{key}' 验证失败", "errors": errors})
    provider = _requested_provider(new_config["modelProvider"], 422) if "modelProvider" in new_config else None
    saved_config, _ = await _save_settings(new_config, database=database, provider=provider)
    return {"success": True, "message": "配置已更新", "config": _mask_config(saved_config)}


def _requested_provider(value, status_code):
    from inference.model_manager import validate_provider

    try:
        return validate_provider(value)
    except ValueError as exc:
        raise HTTPException(status_code, "提供商名称无效，或当前环境不允许使用该提供商") from exc


async def _save_settings(updates, *, database, provider=None):
    """Persist once, then publish the selected runtime and confirm saved settings."""
    from inference.model_manager import ModelProvider, get_model_manager

    progress = {"stage": "config_persistence", "configPersisted": None}
    if provider is not None:
        progress["providerSwitched"] = False
    try:
        await run_db(database.update_config, updates)
        progress.update(stage="config_cache", configPersisted=True)
        await run_db(invalidate_config_cache)
        manager = None
        if provider is not None:
            progress["stage"] = "provider_selection"
            manager = get_model_manager(database=database)
            progress["providerSwitched"] = None
            manager.set_provider(provider)
            progress["providerSwitched"] = True
        if (
            provider == ModelProvider.OPENAI_COMPAT
            or {"openaiCompatBaseUrl", "openaiCompatApiKey", "openaiCompatModel"} & updates.keys()
        ):
            progress["stage"] = "runtime_config"
            if manager is None:
                manager = get_model_manager(database=database)
            await run_db(manager._providers[ModelProvider.OPENAI_COMPAT]._refresh_db_config)
        progress["stage"] = "config_readback"
        saved_config, _ = await run_db(read_config, database)
        status = None
        if provider is not None:
            progress["stage"] = "model_status"
            status = await run_db(manager.get_status)
        return saved_config, status
    except Exception as exc:
        raise HTTPException(
            503,
            detail={
                **progress,
                "message": "配置保存未完整完成，请核对已保存配置与模型状态后重试。结果未确认时，请勿假定配置未变更。",
            },
        ) from exc


# ============================================
# 模型状态与提供商
# ============================================


@router.get("/api/model/status")
async def get_model_status(current_user: dict = Depends(get_current_admin), *, database=Depends(get_request_database)):
    """获取模型状态，读取失败保留原因并明确报错。"""
    try:
        from inference.model_manager import get_model_manager

        model_manager = get_model_manager(database=database)
        return {"success": True, "status": await run_db(model_manager.get_status)}
    except Exception as exc:
        raise HTTPException(503, "模型状态读取失败，请检查配置及模型服务后重试") from exc


@router.put("/api/model/provider")
async def set_model_provider(
    request: Request, current_user: dict = Depends(get_current_admin), *, database=Depends(get_request_database)
):
    """设置提供商；失败时报告已确认和未确认的操作状态。"""
    body = await _request_json(request)
    if not isinstance(body, dict):
        raise HTTPException(400, "请求必须是包含 provider 的对象")
    provider = _requested_provider(body.get("provider"), 400)
    _, status = await _save_settings({"modelProvider": provider.value}, database=database, provider=provider)
    return {"success": True, "message": f"已切换到提供商: {provider.value}", "status": status}
