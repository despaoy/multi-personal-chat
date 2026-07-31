"""LoRA管理API"""
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
import os
from typing import Optional

from fastapi import APIRouter, Request, HTTPException, Depends
from app.dependencies import get_current_admin

from db.adapter import db
from db.database import LORA_ROOT, refresh_lora_dir_map
from inference.lora_utils import resolve_lora_served_name
from inference.adapter_checker import AdapterChecker

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
    """Best-effort compensation when a cross-system LoRA update fails."""
    if client is None:
        return
    if loaded_adapter is not None:
        try:
            await client.unload_lora_adapter(loaded_adapter[0])
        except Exception:
            logger.exception("Failed to remove newly loaded LoRA during rollback: %s", loaded_adapter[0])
    if unloaded_adapter is not None:
        try:
            await client.load_lora_adapter(*unloaded_adapter)
        except Exception:
            logger.exception("Failed to restore previous LoRA during rollback: %s", unloaded_adapter[0])


def _allowed_real_roots() -> list[Path]:
    """受信任的 LoRA 真实根目录集合。

    LORA_ROOT 本身总是受信。运维可通过 LORA_ALLOWED_REAL_ROOTS（逗号分隔）
    显式追加额外的根目录（例如 qqchat-data/loras），用于容纳指向根外的
    符号链接目标。未列入该集合的根外真实路径将被拒绝，避免任意符号链接
    把适配器指向受信边界之外的位置。
    """
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
    """Map a trusted backend LoRA directory to the path visible by vLLM.

    安全模型分两层：
    1. 逻辑路径检查：用 os.path.normpath 防止 lora_name 含 ../ 穿越，
       但不跟随符号链接，因此 LORA_ROOT 下指向其他目录的合法链接不会被误判。
    2. 真实目标受信边界检查：Path.resolve() 会展开符号链接得到真实路径，
       要求该真实路径位于 LORA_ROOT 或 LORA_ALLOWED_REAL_ROOTS 之一之下，
       避免任意符号链接把适配器指向受信边界之外的位置。
    """
    local_root = LORA_ROOT.resolve()
    # 1) 逻辑路径检查（不跟随符号链接）
    normalized = os.path.normpath(str(local_root / lora_name))
    normalized_root = os.path.normpath(str(local_root))
    if normalized != normalized_root and not normalized.startswith(normalized_root + os.sep):
        raise ValueError("LoRA path escapes the configured root")

    # 逻辑相对路径（vLLM 看到的路径，不跟随符号链接）
    logical_rel = Path(lora_name)

    # 2) resolve 检查文件是否存在（符号链接会被展开）
    real_path = (local_root / lora_name).resolve()
    if not (real_path / "adapter_config.json").exists():
        final_path = real_path / "final"
        if (final_path / "adapter_config.json").exists():
            real_path = final_path
            logical_rel = logical_rel / "final"
        else:
            raise FileNotFoundError("LoRA adapter_config.json was not found")

    # 3) 受信边界检查：真实路径必须位于 LORA_ROOT 或显式配置的允许根之下
    allowed_roots = _allowed_real_roots()
    if not _is_within_allowed_real(real_path, allowed_roots):
        logger.warning(
            "LoRA symlink target outside trusted roots: name=%s real=%s allowed=%s",
            lora_name, real_path, [str(r) for r in allowed_roots],
        )
        raise ValueError("LoRA symlink target escapes the trusted roots")

    vllm_root = Path(os.getenv("VLLM_LORA_ROOT", str(local_root)))
    return str(vllm_root / logical_rel)


def _read_lora_metadata(adapter_path: Path) -> dict:
    """Read LoRA metadata from adapter_config.json and trainer_state.json."""
    meta = {"rank": 0, "alpha": 0, "trained_steps": 0, "total_steps": 0, "train_completed": False}

    config_path = adapter_path / "adapter_config.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            meta["rank"] = cfg.get("r", 0)
            meta["alpha"] = cfg.get("lora_alpha", 0)
        except Exception as e:
            # H3 fix: 此前静默吞噬，adapter_config.json 读取失败时无法排障
            logger.warning("读取 adapter_config.json 失败 (lora=%s): %s", adapter_path, e)

    state_path = adapter_path / "trainer_state.json"
    if not state_path.exists() and adapter_path.name == "final":
        try:
            checkpoint_dirs = [
                d for d in adapter_path.parent.iterdir()
                if d.is_dir() and d.name.startswith("checkpoint-")
            ]
            if checkpoint_dirs:
                max_ckpt = max(checkpoint_dirs, key=lambda d: int(d.name.split("-")[-1]))
                candidate = max_ckpt / "trainer_state.json"
                if candidate.exists():
                    state_path = candidate
        except Exception as e:
            # H3 fix: 此前静默吞噬，checkpoint 目录查找失败时无法排障
            logger.warning("查找 checkpoint 目录失败 (lora=%s): %s", adapter_path, e)

    if state_path and state_path.exists():
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            meta["trained_steps"] = state.get("global_step", 0)
            meta["total_steps"] = state.get("max_steps", 0)
            meta["train_completed"] = state.get("best_metric") is not None or meta["trained_steps"] > 0
        except Exception:
            pass

    adapter_file = adapter_path / "adapter_model.safetensors"
    if adapter_file.exists() and meta["trained_steps"] == 0:
        meta["train_completed"] = True
        meta["trained_steps"] = 1
        meta["total_steps"] = 1

    return meta


@router.get("/api/loras")
async def get_loras(status: Optional[str] = None, current_user: dict = Depends(get_current_admin)):
    """获取LoRA模型列表"""
    return {"loras": await asyncio.to_thread(db.get_loras, status)}


def _scan_loras_sync():
    """扫描 loras/ 目录，自动发现并注册新的 LoRA 适配器，更新已有记录元信息"""
    lora_base = LORA_ROOT
    if not lora_base.exists():
        return {"success": True, "message": "loras 目录不存在", "new_count": 0}

    # 获取数据库中已有的 LoRA
    existing_loras = db.get_loras()
    existing_map = {lora["name"]: lora for lora in existing_loras}
    next_id = max((
        int(str(item["id"]))
        for item in existing_loras
        if str(item.get("id", "")).isdigit()
    ), default=0) + 1

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

        # 读取元信息
        meta = _read_lora_metadata(adapter_path)

        # 计算 adapter 大小
        adapter_file = adapter_path / "adapter_model.safetensors"
        size_str = "未知"
        if adapter_file.exists():
            size_mb = adapter_file.stat().st_size / (1024 * 1024)
            size_str = f"{size_mb:.0f}MB"

        trained_steps = meta["trained_steps"]
        total_steps = meta["total_steps"] if meta["total_steps"] > 0 else trained_steps

        if d.name in existing_map:
            # 更新已有记录
            try:
                db.execute_sql(
                    'UPDATE loras SET size = :size, trainedSteps = :trained_steps, totalSteps = :total_steps WHERE name = :name',
                    {"size": size_str, "trained_steps": trained_steps, "total_steps": total_steps, "name": d.name}
                )
                updated_count += 1
            except Exception as e:
                logger.error(f"更新 LoRA 失败 {d.name}: {e}")
                failed_names.append(d.name)
            continue

        # 新增记录

        try:
            db.add_lora({
                "id": str(next_id),
                "name": d.name,
                "description": f"LoRA 适配器 - {d.name}",
                "status": "inactive",
                "style": "",
                "size": size_str,
                "trainedSteps": trained_steps,
                "totalSteps": total_steps,
                "createdAt": datetime.now().strftime("%Y-%m-%d"),
            })
            new_count += 1
            next_id += 1
            logger.info(f"自动注册 LoRA: {d.name} (size={size_str})")
        except Exception as e:
            logger.error(f"注册 LoRA 失败 {d.name}: {e}")
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
async def scan_loras(current_user: dict = Depends(get_current_admin)):
    """Scan and register adapters without blocking or racing status changes."""
    async with _get_lora_status_lock():
        return await asyncio.to_thread(_scan_loras_sync)


@router.put("/api/loras/{lora_id}/status")
async def update_lora_status(lora_id: str, request: Request, current_user: dict = Depends(get_current_admin)):
    """更新LoRA模型状态

    s2 fix: 激活/停用 LoRA 直接影响全局推理路由和 vLLM 加载状态，限定 admin。
    """
    body = await request.json()
    status = body.get("status", "inactive")
    if status not in {"active", "inactive"}:
        raise HTTPException(status_code=422, detail="LoRA 状态只能是 active 或 inactive")

    async with _get_lora_status_lock():
        # Read and update the active adapter under one lock so concurrent
        # activation requests cannot both act on stale database state.
        runtime_client = None
        loaded_adapter: tuple[str, str] | None = None
        unloaded_adapter: tuple[str, str] | None = None
        all_loras = await asyncio.to_thread(db.get_loras)
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
            (
                item
                for item in all_loras
                if item["status"] == "active" and item["id"] != lora_id
            ),
            None,
        )

        if status == "active":
            try:
                from api.generate import get_vllm_client

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
                    await runtime_client.unload_lora_adapter(unloaded_adapter[0])
                loaded_adapter = (served_name, adapter_path)
                try:
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
                logger.warning("Invalid LoRA adapter id=%s error=%s", lora_id, exc)
                raise HTTPException(status_code=422, detail="LoRA 适配器文件无效或配置不完整") from exc
            except Exception as exc:
                logger.exception("Failed to load LoRA into vLLM id=%s", lora_id)
                raise HTTPException(status_code=502, detail="LoRA 无法加载到 vLLM，数据库状态未变更") from exc

        if status == "inactive" and existing.get("status") == "active":
            from api.generate import get_vllm_client

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
                logger.exception("Failed to unload active LoRA id=%s", lora_id)
                raise HTTPException(
                    status_code=502,
                    detail="LoRA 卸载失败，数据库状态未变更",
                ) from exc

        try:
            lora = await asyncio.to_thread(db.update_lora_status, lora_id, status)
            if lora is None:
                raise RuntimeError("LoRA record disappeared during status update")
        except Exception as exc:
            await _rollback_runtime_lora_state(
                runtime_client,
                loaded_adapter=loaded_adapter,
                unloaded_adapter=unloaded_adapter,
            )
            logger.exception("Failed to persist LoRA status id=%s status=%s", lora_id, status)
            raise HTTPException(
                status_code=500,
                detail="LoRA 状态保存失败，运行时已尝试恢复",
            ) from exc

    return {"success": True, "message": f"LoRA状态已更新为{status}", "lora": lora}


@router.delete("/api/loras/{lora_id}")
async def delete_lora(lora_id: str, current_user: dict = Depends(get_current_admin)):
    """Remove a LoRA registration after reconciling the live vLLM state.

    Adapter files are deliberately retained as research artifacts.
    """
    async with _get_lora_status_lock():
        try:
            loras = await asyncio.to_thread(db.get_loras)
            lora = next((item for item in loras if item["id"] == lora_id), None)
            if lora is None:
                raise HTTPException(status_code=404, detail="LoRA模型不存在")

            runtime_client = None
            unloaded_adapter: tuple[str, str] | None = None
            if lora.get("status") == "active":
                from api.generate import get_vllm_client

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
                    logger.exception("Failed to unload LoRA before deletion id=%s", lora_id)
                    raise HTTPException(
                        status_code=502,
                        detail="LoRA 卸载失败，注册记录未删除",
                    ) from exc

            try:
                deleted = await asyncio.to_thread(db.delete_lora, lora_id)
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
            logger.exception("删除 LoRA 注册记录失败 id=%s", lora_id)
            raise HTTPException(status_code=500, detail="删除 LoRA 失败") from exc
