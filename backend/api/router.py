"""多 LoRA 路由 API - 路由配置/适配器兼容性/路由日志"""
import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from app.dependencies import get_current_admin
from db.adapter import db
from db.schemas import RouterConfigUpdate

logger = logging.getLogger(__name__)
router = APIRouter()

_router_config_lock: asyncio.Lock | None = None
_router_config_lock_loop: asyncio.AbstractEventLoop | None = None


def _get_router_config_lock() -> asyncio.Lock:
    global _router_config_lock, _router_config_lock_loop
    loop = asyncio.get_running_loop()
    if _router_config_lock is None or _router_config_lock_loop is not loop:
        _router_config_lock = asyncio.Lock()
        _router_config_lock_loop = loop
    return _router_config_lock


async def _execute_sql(query: str, params: Optional[dict] = None):
    return await asyncio.to_thread(db.execute_sql, query, params or {})


async def _execute_sql_insert(query: str, params: Optional[dict] = None):
    return await asyncio.to_thread(db.execute_sql_insert, query, params or {})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_config(config: dict) -> dict:
    """Merge newly supported personas without overwriting administrator choices."""
    from inference.lora_router import DEFAULT_PERSONA_ADAPTERS, DEFAULT_PERSONA_KEYWORDS

    normalized = dict(config)
    normalized.setdefault("enabled", False)
    normalized.setdefault("default_adapter", "default")
    normalized.setdefault("mode", "manual")
    normalized.setdefault("rag_confidence_threshold", 0.3)
    adapters = dict(normalized.get("persona_adapters") or {})
    keywords = dict(normalized.get("persona_keywords") or {})
    for persona, adapter in DEFAULT_PERSONA_ADAPTERS.items():
        adapters.setdefault(persona, adapter)
    for persona, values in DEFAULT_PERSONA_KEYWORDS.items():
        keywords.setdefault(persona, list(values))
    normalized["persona_adapters"] = adapters
    normalized["persona_keywords"] = keywords
    return normalized


@router.get("/api/router/config")
async def get_router_config(current_user: dict = Depends(get_current_admin)):
    """获取路由配置"""
    try:
        rows = await _execute_sql("SELECT value FROM config WHERE key='lora_router_config'")
        if rows:
            config = json.loads(rows[0]["value"])
        else:
            config = {
                "enabled": False,
                "default_adapter": "default",
                "mode": "manual",
                "persona_adapters": {"kisaki": "kisaki", "minamo": "minamo", "hutao": "hutao"},
                "rag_confidence_threshold": 0.3,
                "persona_keywords": {
                    "hutao": ["胡桃", "往生堂", "堂主"],
                    "zhongli": ["钟离", "岩王帝君", "摩拉克斯"],
                    "qiqi": ["七七", "不卜庐"],
                },
            }
        config = _normalize_config(config)
        return {"success": True, "config": config}
    except Exception as e:
        logger.error(f"获取路由配置失败: {e}")
        raise HTTPException(status_code=500, detail="LoRA 路由操作失败") from e


@router.put("/api/router/config")
async def update_router_config(req: RouterConfigUpdate,
                               current_user: dict = Depends(get_current_admin)):
    """更新路由配置并原子化同进程内的读改写。"""
    try:
        async with _get_router_config_lock():
            rows = await _execute_sql(
                "SELECT value FROM config WHERE key='lora_router_config'"
            )
            if rows:
                config = json.loads(rows[0]["value"])
            else:
                config = {
                    "enabled": False,
                    "default_adapter": "default",
                    "mode": "manual",
                    "persona_adapters": {},
                    "rag_confidence_threshold": 0.3,
                    "persona_keywords": {},
                }
            config.update(req.model_dump(exclude_none=True))
            config = _normalize_config(config)
            serialized = json.dumps(config, ensure_ascii=False)
            await _execute_sql(
                "INSERT INTO config (key, value) VALUES ('lora_router_config', :val) "
                "ON CONFLICT(key) DO UPDATE SET value=:val",
                {"val": serialized},
            )
            from inference.lora_router import get_lora_router

            get_lora_router(config)
        return {"success": True, "config": config}
    except Exception as exc:
        logger.exception("更新路由配置失败")
        raise HTTPException(status_code=500, detail="路由配置更新失败") from exc


@router.get("/api/router/adapters")
async def list_adapters(current_user: dict = Depends(get_current_admin)):
    """列出已注册适配器及兼容性状态"""
    try:
        from db.database import refresh_lora_dir_map
        directory_map = await asyncio.to_thread(refresh_lora_dir_map)
        adapters = []
        for name, path in directory_map.items():
            compat_rows = await _execute_sql(
                "SELECT * FROM adapter_compatibility WHERE adapter_name=:name ORDER BY checked_at DESC LIMIT 1",
                {"name": name},
            )
            compatibility = None
            if compat_rows:
                r = compat_rows[0]
                try:
                    checks = json.loads(r["checks"]) if r["checks"] else {}
                except Exception:
                    checks = {}
                try:
                    warnings = json.loads(r["warnings"]) if r["warnings"] else []
                except Exception:
                    warnings = []
                try:
                    errors = json.loads(r["errors"]) if r["errors"] else []
                except Exception:
                    errors = []
                compatibility = {
                    "compatible": bool(r["compatible"]), "checked_at": r["checked_at"],
                    "checks": checks, "warnings": warnings, "errors": errors,
                }
            adapters.append({"name": name, "path": path, "compatibility": compatibility})
        return {"success": True, "adapters": adapters, "total": len(adapters)}
    except Exception as e:
        logger.error(f"列出适配器失败: {e}")
        raise HTTPException(status_code=500, detail="LoRA 路由操作失败") from e


@router.get("/api/router/logs")
async def get_routing_logs(limit: int = 100, current_user: dict = Depends(get_current_admin)):
    """获取路由日志"""
    try:
        limit = max(1, min(int(limit), 500))
        from inference.lora_router import get_lora_router
        r = get_lora_router()
        logs = r.get_routing_logs(limit=limit)
        return {"success": True, "logs": logs, "total": len(logs)}
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="LoRA 路由模块不可用") from exc
    except Exception as e:
        logger.error(f"获取路由日志失败: {e}")
        raise HTTPException(status_code=500, detail="LoRA 路由操作失败") from e


@router.post("/api/router/check/{adapter_name}")
async def check_adapter_compat(adapter_name: str, current_user: dict = Depends(get_current_admin)):
    """手动触发适配器兼容性检查"""
    try:
        from inference.adapter_checker import AdapterChecker
        from db.database import refresh_lora_dir_map
        directory_map = await asyncio.to_thread(refresh_lora_dir_map)
        if adapter_name not in directory_map:
            raise HTTPException(status_code=404, detail=f"adapter {adapter_name} not found")
        checker = AdapterChecker()
        report = await asyncio.to_thread(checker.check_adapter, directory_map[adapter_name])
        try:
            await _execute_sql_insert(
                "INSERT INTO adapter_compatibility (adapter_name, checked_at, compatible, checks, warnings, errors) "
                "VALUES (:adapter_name, :checked_at, :compatible, :checks, :warnings, :errors)",
                {
                    "adapter_name": adapter_name, "checked_at": _now(),
                    "compatible": int(report.compatible),
                    "checks": json.dumps(report.checks, ensure_ascii=False),
                    "warnings": json.dumps(report.warnings, ensure_ascii=False),
                    "errors": json.dumps(report.errors, ensure_ascii=False),
                },
            )
        except Exception as e:
            logger.warning(f"存储兼容性记录失败（非致命）: {e}")
        return {
            "success": True,
            "adapter_name": adapter_name,
            "compatible": report.compatible,
            "checks": report.checks,
            "warnings": report.warnings,
            "errors": report.errors,
        }
    except HTTPException:
        raise
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="适配器兼容性检查模块不可用") from exc
    except Exception as e:
        logger.error(f"兼容性检查失败: {e}")
        raise HTTPException(status_code=500, detail="LoRA 路由操作失败") from e
