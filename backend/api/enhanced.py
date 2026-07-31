"""增强功能API"""
import asyncio
import logging
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Depends
from app.dependencies import get_current_admin
from db.schemas import ApiKeyCreateRequest
from infra.concurrency_control import inference_runtime

from app.config import (
    LOAD_BALANCER_AVAILABLE,
    RESOURCE_POOL_AVAILABLE,
    CIRCUIT_BREAKER_AVAILABLE,
    BACKUP_MANAGER_AVAILABLE,
    FAILOVER_AVAILABLE,
    INPUT_VALIDATOR_AVAILABLE,
    ENCRYPTION_AVAILABLE,
    ACCESS_CONTROL_AVAILABLE,
    LLM_OPTIMIZER_AVAILABLE,
    circuit_breaker_registry,
    response_cache,
    encryption_mgr,
)
# C-F1 fix: connection_pool/http_client_pool/backup_mgr/failover_mgr/
# access_control_mgr 在 lifespan 中通过 app.config.xxx = ... 赋值。
# 若在导入时绑定，会永远持有 None。改为动态访问模块属性。
from app import config as _app_config
connection_pool = lambda: _app_config.connection_pool
http_client_pool = lambda: _app_config.http_client_pool
backup_mgr = lambda: _app_config.backup_mgr
failover_mgr = lambda: _app_config.failover_mgr
access_control_mgr = lambda: _app_config.access_control_mgr


async def _get_vllm_load_balancer_stats():
    """Read live balancing/health data from the client that serves requests."""
    try:
        from app.config import is_vllm_enabled
        if not is_vllm_enabled():
            return None
        from inference.vllm_client import get_vllm_client
        client = await get_vllm_client()
        return await client.get_stats()
    except Exception as exc:
        logger.debug("读取 vLLM 负载统计失败: %s", exc)
        return None

logger = logging.getLogger(__name__)
router = APIRouter()


def _validate_path(path_str: str, allowed_base: str = None) -> str:
    """Validate path doesn't contain traversal sequences and is within allowed base."""
    if not path_str:
        raise ValueError("Path cannot be empty")
    # Block path traversal
    if '..' in path_str or '\x00' in path_str:
        raise ValueError("Path contains invalid sequences")
    resolved = Path(path_str).resolve()
    if allowed_base:
        base = Path(allowed_base).resolve()
        if not resolved.is_relative_to(base):
            raise ValueError(f"Path must be within {allowed_base}")
    return str(resolved)


@router.get("/api/enhanced/status")
async def get_enhanced_status(current_user: dict = Depends(get_current_admin)):
    """获取增强功能状态"""
    available = {
        "loadBalancer": LOAD_BALANCER_AVAILABLE,
        "resourcePool": RESOURCE_POOL_AVAILABLE,
        "circuitBreaker": CIRCUIT_BREAKER_AVAILABLE,
        "backupManager": BACKUP_MANAGER_AVAILABLE,
        "failover": FAILOVER_AVAILABLE,
        "inputValidator": INPUT_VALIDATOR_AVAILABLE,
        "encryption": ENCRYPTION_AVAILABLE,
        "accessControl": ACCESS_CONTROL_AVAILABLE,
        "llmOptimizer": LLM_OPTIMIZER_AVAILABLE,
    }
    vllm_stats = await _get_vllm_load_balancer_stats()
    active = {
        **available,
        "loadBalancer": vllm_stats is not None,
        "resourcePool": connection_pool() is not None or http_client_pool() is not None,
        "backupManager": backup_mgr() is not None,
        "failover": failover_mgr() is not None,
        "encryption": encryption_mgr is not None,
        "accessControl": access_control_mgr() is not None,
    }
    return {
        "success": True,
        "enhancedFeatures": active,
        "availableFeatures": available,
    }


@router.get("/api/enhanced/stats")
async def get_enhanced_stats(current_user: dict = Depends(get_current_admin)):
    """获取增强功能统计信息"""
    stats = {}

    vllm_stats = await _get_vllm_load_balancer_stats()
    if vllm_stats is not None:
        stats["loadBalancer"] = vllm_stats
    # C-F1 fix: 动态访问 lifespan 中赋值的单例
    _cp = connection_pool()
    if _cp:
        stats["connectionPool"] = _cp.get_pool_stats()

    _hcp = http_client_pool()
    if _hcp:
        stats["httpClientPool"] = _hcp.get_pool_stats()

    if circuit_breaker_registry:
        stats["circuitBreakers"] = circuit_breaker_registry.get_all_stats()

    _bm = backup_mgr()
    if _bm:
        stats["backup"] = _bm.get_backup_stats()

    _fm = failover_mgr()
    if _fm:
        stats["failover"] = _fm.get_failover_status()

    if response_cache:
        stats["responseCache"] = response_cache.stats

    stats["rateLimiter"] = inference_runtime.rate_limit_stats()

    return {"success": True, "stats": stats}


# --- 负载均衡API ---

@router.get("/api/enhanced/load-balancer/stats")
async def get_load_balancer_stats(current_user: dict = Depends(get_current_admin)):
    stats = await _get_vllm_load_balancer_stats()
    if stats is None:
        raise HTTPException(status_code=503, detail="vLLM 负载均衡器不可用")
    return {"success": True, "stats": stats}

# --- 熔断器API ---

@router.get("/api/enhanced/circuit-breaker/stats")
async def get_circuit_breaker_stats(current_user: dict = Depends(get_current_admin)):
    if not circuit_breaker_registry:
        raise HTTPException(status_code=503, detail="熔断器不可用")
    return {"success": True, "stats": circuit_breaker_registry.get_all_stats()}


@router.post("/api/enhanced/circuit-breaker/{name}/reset")
async def reset_circuit_breaker(name: str, current_user: dict = Depends(get_current_admin)):
    # C-S1 fix: 重置熔断器影响系统稳定性，限定 admin
    if not circuit_breaker_registry:
        raise HTTPException(status_code=503, detail="熔断器不可用")
    cb = await circuit_breaker_registry.get(name)
    if cb:
        await cb.reset()
        return {"success": True, "message": f"熔断器 {name} 已重置"}
    raise HTTPException(status_code=404, detail=f"熔断器 {name} 不存在")


# --- 备份管理API ---

@router.get("/api/enhanced/backups")
async def list_backups(current_user: dict = Depends(get_current_admin)):
    _bm = backup_mgr()
    if not _bm:
        raise HTTPException(status_code=503, detail="备份管理器不可用")
    backups = await asyncio.to_thread(_bm.list_backups)
    return {"success": True, "backups": [item.to_dict() for item in backups]}


@router.post("/api/enhanced/backups/create")
async def create_backup(
    backup_type: Literal["full", "incremental"] = "full",
    current_user: dict = Depends(get_current_admin),
):
    _bm = backup_mgr()
    if not _bm:
        raise HTTPException(status_code=503, detail="备份管理器不可用")
    from infra.backup_manager import BackupType

    result = await _bm.backup(BackupType(backup_type))
    if result is None:
        raise HTTPException(status_code=500, detail="备份创建失败，请检查服务器日志")
    payload = result.to_dict()
    return {
        "success": True,
        "message": "备份创建成功",
        "path": payload["path"],
        "backup": payload,
    }


@router.post("/api/enhanced/backups/{backup_name}/restore")
async def restore_backup(
    backup_name: str,
    current_user: dict = Depends(get_current_admin),
):
    # SQLite connections are thread-local and may still reference the old file.
    # Replacing the database under a running process can split reads and writes
    # across two files, so restoration must be performed while the backend is stopped.
    if not backup_mgr():
        raise HTTPException(status_code=503, detail="备份管理器不可用")
    raise HTTPException(
        status_code=409,
        detail="不支持在线恢复 SQLite。请停止后端后运行 scripts/restore_sqlite_backup.py，并在恢复后重新启动服务。",
    )

# --- 故障转移API ---

@router.get("/api/enhanced/failover/status")
async def get_failover_status(current_user: dict = Depends(get_current_admin)):
    _fm = failover_mgr()
    if not _fm:
        raise HTTPException(status_code=503, detail="故障转移管理器不可用")
    return {"success": True, "status": _fm.get_failover_status()}


# --- 缓存管理API ---

@router.get("/api/enhanced/cache/stats")
async def get_cache_stats(current_user: dict = Depends(get_current_admin)):
    if not response_cache:
        raise HTTPException(status_code=503, detail="响应缓存不可用")
    return {"success": True, "stats": response_cache.stats}


@router.post("/api/enhanced/cache/invalidate")
async def invalidate_cache(pattern: Optional[str] = None, current_user: dict = Depends(get_current_admin)):
    # C-S1 fix: 缓存失效可能引发性能雪崩，限定 admin
    if not response_cache:
        raise HTTPException(status_code=503, detail="响应缓存不可用")
    removed = await response_cache.invalidate(pattern)
    return {"success": True, "message": "缓存已清除", "removed": removed}


# --- 访问控制API ---

@router.get("/api/enhanced/access-control/keys")
async def list_api_keys(current_user: dict = Depends(get_current_admin)):
    _acm = access_control_mgr()
    if not _acm:
        raise HTTPException(status_code=503, detail="访问控制不可用")
    try:
        keys = await asyncio.to_thread(_acm.list_api_keys)
    except Exception as exc:
        logger.exception("列出 API Key 失败")
        raise HTTPException(status_code=503, detail="API Key 存储暂时不可用") from exc
    return {"success": True, "keys": keys}


@router.post("/api/enhanced/access-control/keys")
async def create_api_key(
    request: ApiKeyCreateRequest,
    current_user: dict = Depends(get_current_admin),
):
    _acm = access_control_mgr()
    if not _acm:
        raise HTTPException(status_code=503, detail="访问控制不可用")
    from infra.access_control import Role

    try:
        api_key = await asyncio.to_thread(
            _acm.create_api_key,
            Role(request.role),
            request.description,
            request.rate_limit,
        )
    except Exception as exc:
        logger.exception("创建 API Key 失败")
        raise HTTPException(status_code=503, detail="API Key 存储暂时不可用") from exc
    return {"success": True, "apiKey": api_key}


@router.delete("/api/enhanced/access-control/keys/{key_id}")
async def revoke_api_key(
    key_id: int,
    current_user: dict = Depends(get_current_admin),
):
    _acm = access_control_mgr()
    if not _acm:
        raise HTTPException(status_code=503, detail="访问控制不可用")
    try:
        revoked = await asyncio.to_thread(_acm.revoke_api_key_by_id, key_id)
    except Exception as exc:
        logger.exception("吊销 API Key 失败")
        raise HTTPException(status_code=503, detail="API Key 存储暂时不可用") from exc
    if not revoked:
        raise HTTPException(status_code=404, detail="API Key 不存在或已吊销")
    return {"success": True, "message": "API Key已吊销"}

# --- 限流器API ---

@router.get("/api/enhanced/rate-limiter/stats")
async def get_rate_limiter_stats(current_user: dict = Depends(get_current_admin)):
    return {"success": True, "stats": inference_runtime.rate_limit_stats()}


# --- 加密管理API ---

@router.get("/api/enhanced/encryption/status")
async def get_encryption_status(current_user: dict = Depends(get_current_admin)):
    if not encryption_mgr:
        raise HTTPException(status_code=503, detail="加密管理器不可用")
    return {"success": True, "available": True}
