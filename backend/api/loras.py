"""LoRA管理API"""

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, Body, Depends, HTTPException

from app.dependencies import get_current_admin, get_request_database
from app.providers import get_vllm_client
from db.database import LORA_ROOT, refresh_lora_dir_map
from db.lora_metadata import read_lora_metadata
from inference.adapter_checker import AdapterChecker
from inference.lora_utils import resolve_lora_served_name

logger = logging.getLogger(__name__)
router = APIRouter()
_lora_status_lock: asyncio.Lock | None = None
_lora_status_lock_loop: asyncio.AbstractEventLoop | None = None


def _get_lora_status_lock() -> asyncio.Lock:
    """Return the LoRA transaction lock for the active application loop."""
    global _lora_status_lock, _lora_status_lock_loop
    loop = asyncio.get_running_loop()
    if _lora_status_lock is None or _lora_status_lock_loop is not loop:
        _lora_status_lock = asyncio.Lock()
        _lora_status_lock_loop = loop
    return _lora_status_lock


async def _rollback_runtime_lora_state(
    client,
    *,
    loaded_adapter: tuple[str, str] | None,
    unloaded_adapter: tuple[str, str] | None,
) -> None:
    """Attempt both compensations; a failed recovery leaves runtime state unknown."""
    if client is None:
        return
    errors = []
    if loaded_adapter is not None:
        try:
            await client.unload_lora_adapter(loaded_adapter[0])
        except Exception as error:
            errors.append(error)
    if unloaded_adapter is not None:
        try:
            await client.load_lora_adapter(*unloaded_adapter)
        except Exception as error:
            errors.append(error)
    if errors:
        from builtins import ExceptionGroup

        raise HTTPException(
            502,
            detail={
                "kind": "lora_recovery_failed",
                "runtimeState": "unknown",
                "message": "LoRA 操作失败且运行时恢复未完成，请检查 vLLM 后重新核对加载状态",
                "errorTypes": [type(error).__name__ for error in errors],
            },
        ) from ExceptionGroup("LoRA runtime recovery failed", errors)


def _allowed_real_roots() -> list[Path]:
    """Configured real roots are trusted; the primary LoRA root is always included."""
    roots = [LORA_ROOT.resolve()]
    extra = os.getenv("LORA_ALLOWED_REAL_ROOTS", "")
    for part in extra.split(","):
        part = part.strip()
        if part:
            roots.append(Path(part).expanduser().resolve())
    # 去重，保持顺序
    seen: set[str] = set()
    unique: list[Path] = []
    for r in roots:
        key = str(r)
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def _is_within_allowed_real(path: Path, allowed_roots: list[Path]) -> bool:
    """判断真实路径是否位于任一受信根目录之下（含根本身）。"""
    norm = str(path)
    for root in allowed_roots:
        root_str = str(root)
        if norm == root_str or norm.startswith(root_str + os.sep):
            return True
    return False


def _resolve_vllm_adapter_path(lora_name: str) -> str:
    """Map a logical adapter to a trusted real directory and the vLLM container path.

    Symlink trust is checked independently from the logical served path."""
    local_root = LORA_ROOT.resolve()
    normalized = os.path.normpath(str(local_root / lora_name))
    normalized_root = os.path.normpath(str(local_root))
    if normalized != normalized_root and not normalized.startswith(normalized_root + os.sep):
        raise ValueError("LoRA path escapes the configured root")

    logical_rel = Path(lora_name)

    real_path = (local_root / lora_name).resolve()
    if not (real_path / "adapter_config.json").exists():
        final_path = real_path / "final"
        if (final_path / "adapter_config.json").exists():
            real_path = final_path
            logical_rel = logical_rel / "final"
        else:
            raise FileNotFoundError("LoRA adapter_config.json was not found")

    allowed_roots = _allowed_real_roots()
    if not _is_within_allowed_real(real_path, allowed_roots):
        logger.warning(
            "LoRA symlink target outside trusted roots: name=%s real=%s allowed=%s",
            lora_name,
            real_path,
            [str(r) for r in allowed_roots],
        )
        raise ValueError("LoRA symlink target escapes the trusted roots")

    vllm_root = Path(os.getenv("VLLM_LORA_ROOT", str(local_root)))
    return str(vllm_root / logical_rel)


@router.get("/api/loras")
async def get_loras(
    status: Optional[str] = None,
    current_user: dict = Depends(get_current_admin),
    database=Depends(get_request_database),
):
    """获取LoRA模型列表"""
    return {"loras": await asyncio.to_thread(database.get_loras, status)}


def _scan_loras_sync(database):
    """扫描 loras/ 目录，自动发现并注册新的 LoRA 适配器，更新已有记录元信息"""
    lora_base = LORA_ROOT
    if not lora_base.exists():
        return {"success": True, "message": "loras 目录不存在", "new_count": 0}

    existing_loras = database.get_loras()
    existing_map = {lora["name"]: lora for lora in existing_loras}
    next_id = max((int(str(item["id"])) for item in existing_loras if str(item.get("id", "")).isdigit()), default=0) + 1

    new_count = 0
    updated_count = 0
    failed_names: list[str] = []
    for d in sorted(lora_base.iterdir()):
        if not d.is_dir():
            continue

        # 检查是否包含 adapter_config.json
        config_path = d / "adapter_config.json"
        adapter_path = d
        if not config_path.exists() and (d / "final" / "adapter_config.json").exists():
            config_path = d / "final" / "adapter_config.json"
            adapter_path = d / "final"

        if not config_path.exists():
            continue

        meta = read_lora_metadata(adapter_path)

        adapter_file = adapter_path / "adapter_model.safetensors"
        size_str = "未知"
        if adapter_file.exists():
            size_mb = adapter_file.stat().st_size / (1024 * 1024)
            size_str = f"{size_mb:.0f}MB"

        trained_steps = meta["trained_steps"]
        total_steps = meta["total_steps"]

        if d.name in existing_map:
            # 更新已有记录
            try:
                database.execute_sql(
                    "UPDATE loras SET size = :size, trainedSteps = :trained_steps, totalSteps = :total_steps WHERE name = :name",
                    {"size": size_str, "trained_steps": trained_steps, "total_steps": total_steps, "name": d.name},
                )
                updated_count += 1
            except Exception as e:
                logger.error("LoRA metadata update failed type=%s", type(e).__name__)
                failed_names.append(d.name)
            continue

        try:
            database.add_lora(
                {
                    "id": str(next_id),
                    "name": d.name,
                    "description": f"LoRA 适配器 - {d.name}",
                    "status": "inactive",
                    "style": "",
                    "size": size_str,
                    "trainedSteps": trained_steps,
                    "totalSteps": total_steps,
                    "createdAt": datetime.now().strftime("%Y-%m-%d"),
                }
            )
            new_count += 1
            next_id += 1
            logger.info(f"自动注册 LoRA: {d.name} (size={size_str})")
        except Exception as e:
            logger.error("LoRA registration failed type=%s", type(e).__name__)
            failed_names.append(d.name)

    msg_parts = []
    if new_count > 0:
        msg_parts.append(f"发现 {new_count} 个新 LoRA")
    if updated_count > 0:
        msg_parts.append(f"更新 {updated_count} 个记录")
    if not msg_parts:
        msg_parts.append("无新增或更新")

    if failed_names:
        msg_parts.append(f"{len(failed_names)} 个目录处理失败")

    refresh_lora_dir_map(lora_base)
    return {
        "success": not failed_names,
        "message": "，".join(msg_parts),
        "new_count": new_count,
        "updated_count": updated_count,
        "failed_count": len(failed_names),
        "failed_names": failed_names,
    }


@router.post("/api/loras/scan")
async def scan_loras(current_user: dict = Depends(get_current_admin), database=Depends(get_request_database)):
    """Scan and register adapters without blocking or racing status changes."""
    async with _get_lora_status_lock():
        return await asyncio.to_thread(_scan_loras_sync, database)


@router.put("/api/loras/{lora_id}/status")
async def update_lora_status(
    lora_id: str,
    status: Literal["active", "inactive"] = Body(embed=True),
    current_user: dict = Depends(get_current_admin),
    database=Depends(get_request_database),
):
    """Apply an explicit adapter status under the runtime/database lock."""
    async with _get_lora_status_lock():
        # Read and update the active adapter under one lock so concurrent
        # activation requests cannot both act on stale database state.
        runtime_client = None
        loaded_adapter: tuple[str, str] | None = None
        unloaded_adapter: tuple[str, str] | None = None
        all_loras = await asyncio.to_thread(database.get_loras)
        existing = next((item for item in all_loras if item["id"] == lora_id), None)
        if existing is None:
            raise HTTPException(status_code=404, detail="LoRA模型不存在")
        if existing.get("status") == status:
            return {
                "success": True,
                "message": f"LoRA状态已经是{status}",
                "lora": existing,
            }
        previous_active = next(
            (item for item in all_loras if item["status"] == "active" and item["id"] != lora_id),
            None,
        )

        if status == "active":
            try:
                # 适配器兼容性检查（激活前验证）
                checker = AdapterChecker(lora_root=str(LORA_ROOT))
                compat_report = await asyncio.to_thread(checker.check_adapter, existing["name"])
                if not compat_report.compatible:
                    # 基座不匹配返回明确的 LORA_BASE_MODEL_MISMATCH，
                    # 避免下游 vLLM 400 被包装成模糊的 502。
                    if compat_report.base_model_mismatch:
                        detail = {
                            "code": "LORA_BASE_MODEL_MISMATCH",
                            "message": "LoRA 基座与当前 vLLM 基座不兼容",
                            "expected": compat_report.expected_base_model,
                            "actual": compat_report.actual_base_model,
                            "errors": compat_report.errors,
                        }
                        raise HTTPException(status_code=409, detail=detail)
                    detail = {
                        "message": "适配器兼容性检查未通过",
                        "errors": compat_report.errors,
                        "warnings": compat_report.warnings,
                    }
                    raise HTTPException(status_code=409, detail=detail)

                runtime_client = await get_vllm_client()
                if runtime_client is None:
                    raise RuntimeError("vLLM client is unavailable")
                served_name = resolve_lora_served_name(existing["name"])
                adapter_path = _resolve_vllm_adapter_path(existing["name"])
                if previous_active is not None:
                    unloaded_adapter = (
                        resolve_lora_served_name(previous_active["name"]),
                        _resolve_vllm_adapter_path(previous_active["name"]),
                    )
                try:
                    if unloaded_adapter is not None:
                        await runtime_client.unload_lora_adapter(unloaded_adapter[0])
                    loaded_adapter = (served_name, adapter_path)
                    await runtime_client.load_lora_adapter(*loaded_adapter)
                except Exception:
                    await _rollback_runtime_lora_state(
                        runtime_client,
                        loaded_adapter=loaded_adapter,
                        unloaded_adapter=unloaded_adapter,
                    )
                    raise
            except HTTPException:
                raise
            except (FileNotFoundError, ValueError) as exc:
                logger.warning("Invalid LoRA adapter id=%s type=%s", lora_id, type(exc).__name__)
                raise HTTPException(status_code=422, detail="LoRA 适配器文件无效或配置不完整") from exc
            except Exception as exc:
                logger.error("LoRA load failed id=%s type=%s", lora_id, type(exc).__name__)
                raise HTTPException(status_code=502, detail="LoRA 无法加载到 vLLM，数据库状态未变更") from exc

        if status == "inactive" and existing.get("status") == "active":
            runtime_client = await get_vllm_client()
            if runtime_client is None:
                raise HTTPException(
                    status_code=503,
                    detail="vLLM 不可用，LoRA 状态保持不变",
                )
            unloaded_adapter = (
                resolve_lora_served_name(existing["name"]),
                _resolve_vllm_adapter_path(existing["name"]),
            )
            try:
                await runtime_client.unload_lora_adapter(unloaded_adapter[0])
            except Exception as exc:
                await _rollback_runtime_lora_state(
                    runtime_client,
                    loaded_adapter=None,
                    unloaded_adapter=unloaded_adapter,
                )
                logger.error("LoRA unload failed id=%s type=%s", lora_id, type(exc).__name__)
                raise HTTPException(
                    status_code=502,
                    detail="LoRA 卸载失败，数据库状态未变更",
                ) from exc

        try:
            lora = await asyncio.to_thread(database.update_lora_status, lora_id, status)
            if lora is None:
                raise RuntimeError("LoRA record disappeared during status update")
        except Exception as exc:
            await _rollback_runtime_lora_state(
                runtime_client,
                loaded_adapter=loaded_adapter,
                unloaded_adapter=unloaded_adapter,
            )
            logger.error("LoRA status write failed id=%s type=%s", lora_id, type(exc).__name__)
            raise HTTPException(
                status_code=500,
                detail="LoRA 状态保存失败，运行时已尝试恢复",
            ) from exc

    return {"success": True, "message": f"LoRA状态已更新为{status}", "lora": lora}


@router.delete("/api/loras/{lora_id}")
async def delete_lora(
    lora_id: str, current_user: dict = Depends(get_current_admin), database=Depends(get_request_database)
):
    """Remove a LoRA registration after reconciling the live vLLM state.

    Adapter files are deliberately retained as research artifacts.
    """
    async with _get_lora_status_lock():
        try:
            loras = await asyncio.to_thread(database.get_loras)
            lora = next((item for item in loras if item["id"] == lora_id), None)
            if lora is None:
                raise HTTPException(status_code=404, detail="LoRA模型不存在")

            runtime_client = None
            unloaded_adapter: tuple[str, str] | None = None
            if lora.get("status") == "active":
                runtime_client = await get_vllm_client()
                if runtime_client is None:
                    raise HTTPException(
                        status_code=503,
                        detail="vLLM 不可用，无法安全卸载当前 LoRA",
                    )
                unloaded_adapter = (
                    resolve_lora_served_name(lora["name"]),
                    _resolve_vllm_adapter_path(lora["name"]),
                )
                try:
                    await runtime_client.unload_lora_adapter(unloaded_adapter[0])
                except Exception as exc:
                    await _rollback_runtime_lora_state(
                        runtime_client,
                        loaded_adapter=None,
                        unloaded_adapter=unloaded_adapter,
                    )
                    logger.error("LoRA unload before delete failed id=%s type=%s", lora_id, type(exc).__name__)
                    raise HTTPException(
                        status_code=502,
                        detail="LoRA 卸载失败，注册记录未删除",
                    ) from exc

            try:
                deleted = await asyncio.to_thread(database.delete_lora, lora_id)
            except Exception as exc:
                await _rollback_runtime_lora_state(
                    runtime_client,
                    loaded_adapter=None,
                    unloaded_adapter=unloaded_adapter,
                )
                raise RuntimeError("LoRA registration delete failed") from exc
            if not deleted:
                await _rollback_runtime_lora_state(
                    runtime_client,
                    loaded_adapter=None,
                    unloaded_adapter=unloaded_adapter,
                )
                raise HTTPException(status_code=404, detail="LoRA模型不存在")
            logger.info("Removed LoRA registration while retaining adapter files: %s", lora_id)
            return {
                "success": True,
                "message": "LoRA 注册记录已删除，适配器文件已保留",
            }
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("LoRA delete failed id=%s type=%s", lora_id, type(exc).__name__)
            raise HTTPException(status_code=500, detail="删除 LoRA 失败") from exc
