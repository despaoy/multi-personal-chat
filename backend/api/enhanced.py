"""增强功能API"""
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request, HTTPException, Depends
from app.dependencies import get_current_user, get_current_admin

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
    load_balancer_mgr,
    circuit_breaker_registry,
    response_cache,
    rate_limiter,
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
async def get_enhanced_status(current_user: dict = Depends(get_current_user)):
    """获取增强功能状态"""
    status = {
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
    return {"success": True, "enhancedFeatures": status}


@router.get("/api/enhanced/stats")
async def get_enhanced_stats(current_user: dict = Depends(get_current_user)):
    """获取增强功能统计信息"""
    stats = {}

    if load_balancer_mgr:
        # P1-M3 fix: 实时同步 VLLMClient 统计，避免返回初始化快照
        try:
            from api.generate import get_vllm_client
            vllm_client = await get_vllm_client()
            if vllm_client is not None:
                load_balancer_mgr.sync_from_vllm_client(vllm_client)
        except Exception as exc:
            logging.getLogger(__name__).debug("实时同步 VLLMClient 统计失败: %s", exc)
        stats["loadBalancer"] = load_balancer_mgr.get_stats()

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
        stats["responseCache"] = response_cache.stats()

    if rate_limiter:
        stats["rateLimiter"] = rate_limiter.get_stats()

    return {"success": True, "stats": stats}


# --- 负载均衡API ---

@router.get("/api/enhanced/load-balancer/stats")
async def get_load_balancer_stats(current_user: dict = Depends(get_current_user)):
    if not load_balancer_mgr:
        raise HTTPException(status_code=503, detail="负载均衡器不可用")
    # P1-M3 fix: 每次请求实时从 VLLMClient 同步统计，避免返回初始化快照。
    # 原 sync_from_vllm_client 只在 _ensure_vllm 初始化时调用一次，之后
    # VLLMClient 内部 record_success 更新的数据无法反映到监控接口。
    try:
        from api.generate import get_vllm_client
        vllm_client = await get_vllm_client()
        if vllm_client is not None:
            load_balancer_mgr.sync_from_vllm_client(vllm_client)
    except Exception as exc:
        logging.getLogger(__name__).debug("实时同步 VLLMClient 统计失败: %s", exc)
    return {"success": True, "stats": load_balancer_mgr.get_stats()}


# --- 熔断器API ---

@router.get("/api/enhanced/circuit-breaker/stats")
async def get_circuit_breaker_stats(current_user: dict = Depends(get_current_user)):
    if not circuit_breaker_registry:
        raise HTTPException(status_code=503, detail="熔断器不可用")
    return {"success": True, "stats": circuit_breaker_registry.get_all_stats()}


@router.post("/api/enhanced/circuit-breaker/{name}/reset")
async def reset_circuit_breaker(name: str, current_user: dict = Depends(get_current_admin)):
    # C-S1 fix: 重置熔断器影响系统稳定性，限定 admin
    if not circuit_breaker_registry:
        raise HTTPException(status_code=503, detail="熔断器不可用")
    cb = circuit_breaker_registry.get(name)
    if cb:
        cb.reset()
        return {"success": True, "message": f"熔断器 {name} 已重置"}
    raise HTTPException(status_code=404, detail=f"熔断器 {name} 不存在")


# --- 备份管理API ---

@router.get("/api/enhanced/backups")
async def list_backups(current_user: dict = Depends(get_current_user)):
    _bm = backup_mgr()
    if not _bm:
        raise HTTPException(status_code=503, detail="备份管理器不可用")
    return {"success": True, "backups": _bm.list_backups()}


@router.post("/api/enhanced/backups/create")
async def create_backup(backup_type: str = "full", current_user: dict = Depends(get_current_admin)):
    # C-S1 fix: 备份操作影响数据安全，限定 admin
    _bm = backup_mgr()
    if not _bm:
        raise HTTPException(status_code=503, detail="备份管理器不可用")
    try:
        if backup_type == "full":
            path = _bm.create_full_backup()
        else:
            path = _bm.create_incremental_backup()
        return {"success": True, "message": "备份创建成功", "path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/enhanced/backups/{backup_name}/restore")
async def restore_backup(backup_name: str, current_user: dict = Depends(get_current_admin)):
    # C-S1 fix: 备份恢复会覆盖现有数据，限定 admin
    _bm = backup_mgr()
    if not _bm:
        raise HTTPException(status_code=503, detail="备份管理器不可用")
    try:
        backup_dir = Path(__file__).parent.parent / "backups"
        # Validate backup_name to prevent path traversal
        if '..' in backup_name or '/' in backup_name or '\\' in backup_name or '\x00' in backup_name:
            raise HTTPException(status_code=400, detail="无效的备份名称")
        backup_path = backup_dir / backup_name
        # Ensure resolved path is within backup directory
        if not backup_path.resolve().is_relative_to(backup_dir.resolve()):
            raise HTTPException(status_code=400, detail="无效的备份路径")
        if not backup_path.exists():
            raise HTTPException(status_code=404, detail="备份文件不存在")
        _bm.restore(str(backup_path))
        return {"success": True, "message": "备份恢复成功"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- 故障转移API ---

@router.get("/api/enhanced/failover/status")
async def get_failover_status(current_user: dict = Depends(get_current_user)):
    _fm = failover_mgr()
    if not _fm:
        raise HTTPException(status_code=503, detail="故障转移管理器不可用")
    return {"success": True, "status": _fm.get_failover_status()}


# --- 缓存管理API ---

@router.get("/api/enhanced/cache/stats")
async def get_cache_stats(current_user: dict = Depends(get_current_user)):
    if not response_cache:
        raise HTTPException(status_code=503, detail="响应缓存不可用")
    return {"success": True, "stats": response_cache.stats}


@router.post("/api/enhanced/cache/invalidate")
async def invalidate_cache(pattern: Optional[str] = None, current_user: dict = Depends(get_current_admin)):
    # C-S1 fix: 缓存失效可能引发性能雪崩，限定 admin
    if not response_cache:
        raise HTTPException(status_code=503, detail="响应缓存不可用")
    response_cache.invalidate(pattern)
    return {"success": True, "message": "缓存已清除"}


# --- 访问控制API ---

@router.get("/api/enhanced/access-control/keys")
async def list_api_keys(current_user: dict = Depends(get_current_user)):
    _acm = access_control_mgr()
    if not _acm:
        raise HTTPException(status_code=503, detail="访问控制不可用")
    return {"success": True, "keys": _acm.list_api_keys()}


@router.post("/api/enhanced/access-control/keys")
async def create_api_key(request: Request, current_user: dict = Depends(get_current_admin)):
    # C-S1 fix: 签发 API Key 等同于授权访问，限定 admin
    _acm = access_control_mgr()
    if not _acm:
        raise HTTPException(status_code=503, detail="访问控制不可用")
    body = await request.json()
    role = body.get("role", "viewer")
    description = body.get("description", "")
    api_key = _acm.create_api_key(role, description)
    return {"success": True, "apiKey": api_key}

@router.delete("/api/enhanced/access-control/keys/{key_id}")
async def revoke_api_key(key_id: str, current_user: dict = Depends(get_current_admin)):
    # C-S1 fix: 吊销 API Key 影响服务可用性，限定 admin
    _acm = access_control_mgr()
    if not _acm:
        raise HTTPException(status_code=503, detail="访问控制不可用")
    _acm.revoke_api_key(key_id)
    return {"success": True, "message": "API Key已吊销"}


# --- 限流器API ---

@router.get("/api/enhanced/rate-limiter/stats")
async def get_rate_limiter_stats(current_user: dict = Depends(get_current_user)):
    if not rate_limiter:
        raise HTTPException(status_code=503, detail="限流器不可用")
    return {"success": True, "stats": rate_limiter.get_stats()}


# --- 加密管理API ---

@router.get("/api/enhanced/encryption/status")
async def get_encryption_status(current_user: dict = Depends(get_current_user)):
    if not encryption_mgr:
        raise HTTPException(status_code=503, detail="加密管理器不可用")
    return {"success": True, "available": True}
