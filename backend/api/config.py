"""配置与模型状态API"""
import logging

from fastapi import APIRouter, Request, HTTPException, Depends
from app.dependencies import get_current_admin

from db.adapter import db
from app.config import (
    INPUT_VALIDATOR_AVAILABLE,
    CONFIG_SCHEMA,
)
from cache.config_cache import get_cached_config, set_cached_config, invalidate_config_cache
from infra.db_executor import run_db

logger = logging.getLogger(__name__)
router = APIRouter()

# 敏感字段关键词（匹配任一即脱敏）
_SENSITIVE_KEYWORDS = ("apikey", "api_key", "secret", "password", "token")
# 脱敏标记，用于PUT时识别并跳过未修改的字段
_MASK_MARKER = "****"
_SERVER_ONLY_CONFIG_KEYS = frozenset({
    "astrbotIntegrationToken",
    "astrbotIntegrationTokens",
})
_SERVER_ONLY_CONFIG_KEYS_NORMALIZED = frozenset(
    key.lower() for key in _SERVER_ONLY_CONFIG_KEYS
)


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
        if any(s in k.lower() for s in _SENSITIVE_KEYWORDS) and v:
            masked[k] = _mask_value(str(v))
        else:
            masked[k] = v
    return masked


# ============================================
# 系统配置
# ============================================

@router.get("/api/config")
async def get_config(current_user: dict = Depends(get_current_admin)):
    """获取系统配置（Redis 缓存优先，TTL 60s；敏感字段脱敏后返回）"""
    cached = await run_db(get_cached_config)
    if cached is not None:
        return {"config": _mask_config(cached), "cached": True}

    config = await run_db(lambda: db.config)
    await run_db(set_cached_config, config)
    return {"config": _mask_config(config)}


@router.put("/api/config")
async def update_config(request: Request, current_user: dict = Depends(get_current_admin)):
    """更新系统配置（含脱敏标记的字段自动跳过，保留原值）

    C-S1 fix: 系统配置含 API key 等敏感项，限定 admin。
    """
    new_config = await request.json()
    if not isinstance(new_config, dict):
        raise HTTPException(status_code=422, detail="Configuration payload must be an object")
    server_only = {
        str(key) for key in new_config
        if str(key).lower() in _SERVER_ONLY_CONFIG_KEYS_NORMALIZED
    }
    if server_only:
        raise HTTPException(
            status_code=422,
            detail="AstrBot integration tokens are server-only environment variables",
        )
    # 过滤掉含脱敏标记的未修改字段（前端回传的脱敏值不应覆盖真实凭证）
    new_config = {k: v for k, v in new_config.items()
                  if not (isinstance(v, str) and _MASK_MARKER in v)}
    # 输入验证 - 逐字段验证键值对
    if INPUT_VALIDATOR_AVAILABLE:
        from infra.input_validator import InputValidator
        for key, value in new_config.items():
            entry = {"key": str(key), "value": str(value)}
            is_valid, errors = InputValidator.validate(entry, CONFIG_SCHEMA)
            if not is_valid:
                raise HTTPException(status_code=422, detail={"message": f"配置项 '{key}' 验证失败", "errors": errors})
    await run_db(db.update_config, new_config)
    # 使缓存失效
    await run_db(invalidate_config_cache)
    # 同步OpenAI兼容提供商配置
    compat_keys = {'openaiCompatBaseUrl', 'openaiCompatApiKey', 'openaiCompatModel'}
    if compat_keys & set(new_config.keys()):
        try:
            from inference.model_manager import get_model_manager
            manager = get_model_manager()
            from inference.model_manager import ModelProvider
            provider = manager._providers.get(ModelProvider.OPENAI_COMPAT)
            if provider:
                if 'openaiCompatBaseUrl' in new_config:
                    provider.base_url = new_config['openaiCompatBaseUrl']
                if 'openaiCompatApiKey' in new_config:
                    provider.api_key = new_config['openaiCompatApiKey']
                if 'openaiCompatModel' in new_config:
                    provider.model = new_config['openaiCompatModel']
                    provider._model_name = new_config['openaiCompatModel']
        except Exception as e:
            logger.warning(f"同步OpenAI兼容配置失败: {e}")
    return {"success": True, "message": "配置已更新", "config": _mask_config(await run_db(lambda: db.config))}


# ============================================
# 模型状态与提供商
# ============================================

@router.get("/api/model/status")
async def get_model_status(current_user: dict = Depends(get_current_admin)):
    """获取模型状态"""
    # C3 fix: 原先失败时返回 HTTP 200 + {"success": False}，破坏 REST 语义，
    # 前端需通过 success 字段而非状态码判断错误。改为 raise HTTPException，
    # 与模块内其他端点（update_config、set_model_provider）的失败处理一致。
    try:
        from inference.model_manager import get_model_manager
        model_manager = get_model_manager()
        return {
            "success": True,
            "status": model_manager.get_status()
        }
    except Exception as e:
        logger.error(f"获取模型状态失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="获取模型状态失败")


@router.put("/api/model/provider")
async def set_model_provider(request: Request, current_user: dict = Depends(get_current_admin)):
    """设置模型提供商

    C-S1 fix: 切换模型 provider 影响全局推理，限定 admin。
    """
    try:
        from inference.model_manager import get_model_manager, ModelProvider
        model_manager = get_model_manager()

        body = await request.json()
        provider_name = body.get("provider", "ollama")

        provider_map = {
            "ollama": ModelProvider.OLLAMA,
            "llama_cpp": ModelProvider.LLAMA_CPP,
            "openai_compat": ModelProvider.OPENAI_COMPAT,
            "transformers_peft": ModelProvider.TRANSFORMERS_PEFT,
            "vllm": ModelProvider.VLLM,
            "mock": ModelProvider.MOCK
        }

        if provider_name not in provider_map:
            raise HTTPException(status_code=400, detail="无效的提供商名称")

        success = model_manager.set_provider(provider_map[provider_name])

        if success:
            await run_db(db.update_config, {"modelProvider": provider_name})
            await run_db(invalidate_config_cache)
            if provider_name == "openai_compat":
                compat_provider = model_manager.get_current_provider()
                cfg = await run_db(lambda: db.config)
                if hasattr(compat_provider, 'api_key') and cfg.get('openaiCompatApiKey'):
                    compat_provider.api_key = cfg['openaiCompatApiKey']
                if hasattr(compat_provider, 'base_url') and cfg.get('openaiCompatBaseUrl'):
                    compat_provider.base_url = cfg['openaiCompatBaseUrl']
                if hasattr(compat_provider, 'model') and cfg.get('openaiCompatModel'):
                    compat_provider.model = cfg['openaiCompatModel']
                    compat_provider._model_name = cfg['openaiCompatModel']
            return {
                "success": True,
                "message": f"已切换到提供商: {provider_name}",
                "status": model_manager.get_status()
            }
        else:
            raise HTTPException(status_code=400, detail="提供商切换失败")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"设置提供商失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
