"""LoRA训练管理API"""
import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
import zipfile
from collections import Counter
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from app.dependencies import get_current_admin, get_request_database
from infra.db_executor import run_db
from infra.environment import read_positive_float, read_positive_int
from inference.openai_protocol import ProviderResponseError
from pydantic import BaseModel
from typing import Optional

from db.adapter import db
from db.schemas import DatasetUploadRequest, TrainingStartRequest, DialogueGenerateRequest, SaveDialoguesRequest
from app.config import generation_state, get_generation_state_lock
from infra.input_validator import TRAINING_SCHEMA, InputValidator

from services.character_research import CharacterResearchError, search_character_info
from training.dialogue_generation import (
    DialogueBatchError, build_dialogue_prompt, parse_dialogue_batch, plan_dialogue_batches,
)
from inference.usage import begin_completion
from services.model_usage import UsageRecordingError, account_usage

logger = logging.getLogger(__name__)
router = APIRouter()

_dialogue_generation_task: asyncio.Task | None = None
_DIALOGUE_CANCELLED = "dialogue_user_cancel"


_DATASET_EXPORT_MAX_CONCURRENCY = read_positive_int(os.environ, "TRAINING_EXPORT_MAX_CONCURRENCY", 2)
_DATASET_EXPORT_ACQUIRE_TIMEOUT = read_positive_float(os.environ, "TRAINING_EXPORT_ACQUIRE_TIMEOUT", 1.0)
_dataset_export_slots: asyncio.BoundedSemaphore | None = None
_dataset_export_slots_loop: asyncio.AbstractEventLoop | None = None


def _get_dataset_export_slots() -> asyncio.BoundedSemaphore:
    """Return export admission control owned by the active event loop."""
    global _dataset_export_slots, _dataset_export_slots_loop
    loop = asyncio.get_running_loop()
    if _dataset_export_slots is None or _dataset_export_slots_loop is not loop:
        _dataset_export_slots = asyncio.BoundedSemaphore(
            _DATASET_EXPORT_MAX_CONCURRENCY
        )
        _dataset_export_slots_loop = loop
    return _dataset_export_slots


def _create_export_temp_path() -> Path:
    """Create a private, collision-safe archive path without holding file data in memory."""
    fd, raw_path = tempfile.mkstemp(prefix="qqchat-dataset-", suffix=".zip")
    os.close(fd)
    return Path(raw_path)


def _write_dataset_archive(dataset_dir: Path, archive_path: Path) -> os.stat_result:
    """Build one dataset archive in a worker thread and reject escaping symlinks."""
    dataset_root = dataset_dir.resolve(strict=True)
    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for file_path in dataset_root.rglob("*"):
            if file_path.is_symlink():
                raise ValueError(f"Unsafe dataset symlink: {file_path.name}")
            if not file_path.is_file():
                continue
            resolved_file = file_path.resolve(strict=True)
            if not resolved_file.is_relative_to(dataset_root):
                raise ValueError(f"Dataset file escapes export root: {file_path.name}")
            archive.write(resolved_file, file_path.relative_to(dataset_root))
    return archive_path.stat()


async def _build_dataset_archive(dataset_dir: Path, archive_path: Path) -> os.stat_result:
    """Run ZIP compression off-loop while retaining ownership on cancellation."""
    worker = asyncio.create_task(
        asyncio.to_thread(_write_dataset_archive, dataset_dir, archive_path)
    )
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        # A client can cancel the ASGI task more than once. Keep ownership until
        # the non-cancellable thread has really stopped touching the archive.
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        if worker.done():
            try:
                worker.result()
            except BaseException:
                pass
        raise


def _remove_export_temp(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.exception("Failed to remove dataset export temp file: %s", path)


class _TemporaryExportFileResponse(FileResponse):
    """FileResponse that releases its archive and admission slot on every exit path."""

    def __init__(
        self,
        path: Path,
        *,
        filename: str,
        stat_result: os.stat_result,
        semaphore: asyncio.BoundedSemaphore,
    ) -> None:
        super().__init__(
            path,
            media_type="application/zip",
            filename=filename,
            stat_result=stat_result,
            headers={"Cache-Control": "private, no-store"},
        )
        self._temporary_path = path
        self._semaphore = semaphore
        self._released = False

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                await asyncio.shield(
                    asyncio.to_thread(_remove_export_temp, self._temporary_path)
                )
            finally:
                if not self._released:
                    self._released = True
                    self._semaphore.release()



def _validate_resource_name(name: str, label: str = "名称") -> str:
    """Require one filesystem-safe path component while allowing Unicode."""
    value = (name or "").strip()
    invalid_chars = {"/", "\\", "\x00", ":", "*", "?", "<", ">", "|", chr(34)}
    if (
        not value
        or len(value) > 100
        or value in {".", ".."}
        or any(char in value for char in invalid_chars)
        or any(ord(char) < 32 for char in value)
    ):
        raise HTTPException(status_code=422, detail=f"{label}包含非法路径字符")
    return value


@router.get("/api/training/datasets")
async def list_datasets(current_user: dict = Depends(get_current_admin)):
    """列出可用数据集"""
    try:
        from training.preprocessor import get_dataset_preprocessor
        preprocessor = get_dataset_preprocessor()

        datasets = []
        if preprocessor.data_dir.exists():
            for dataset_dir in preprocessor.data_dir.iterdir():
                if dataset_dir.is_dir():
                    info_path = dataset_dir / "dataset_info.json"
                    if info_path.exists():
                        with open(info_path, 'r', encoding='utf-8') as f:
                            info = json.load(f)
                        datasets.append({
                            "name": dataset_dir.name,
                            "path": str(dataset_dir),
                            "stats": info.get("stats", {})
                        })

        return {"success": True, "datasets": datasets}
    except Exception as e:
        logger.error(f"列出数据集失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/training/datasets")
async def create_dataset(request: DatasetUploadRequest, current_user: dict = Depends(get_current_admin)):
    """创建新数据集

    s2 fix: 数据集影响全局训练资源，限定 admin。
    """
    try:
        from training.preprocessor import get_dataset_preprocessor
        preprocessor = get_dataset_preprocessor()

        style_config = None
        if request.style:
            style_config = preprocessor.get_style(request.style)

        # 清理数据集名称中的非法文件名字符
        safe_name = _validate_resource_name(request.dataset_name, "数据集名称")

        dataset_dir, stats = preprocessor.prepare_training_data(
            raw_data=request.data,
            style_config=style_config,
            output_name=safe_name,
            custom_prompt=request.custom_prompt
        )

        return {
            "success": True,
            "message": "数据集创建成功",
            "dataset": {
                "name": safe_name,
                "path": str(dataset_dir),
                "stats": stats
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建数据集失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/training/datasets/{dataset_name}/export")
async def export_dataset(dataset_name: str, current_user: dict = Depends(get_current_admin)):
    """导出数据集为 ZIP 文件（用于上传到服务器训练）。"""
    archive_path: Path | None = None
    slot_acquired = False
    response_owns_slot = False
    export_slots = _get_dataset_export_slots()
    try:
        dataset_name = _validate_resource_name(dataset_name, "数据集名称")

        from training.preprocessor import get_dataset_preprocessor

        preprocessor = get_dataset_preprocessor()
        data_root = preprocessor.data_dir.resolve()
        resolved_dataset = (preprocessor.data_dir / dataset_name).resolve()
        if not resolved_dataset.is_relative_to(data_root):
            raise HTTPException(status_code=400, detail="无效的数据集路径")
        if not resolved_dataset.exists() or not resolved_dataset.is_dir():
            raise HTTPException(status_code=404, detail=f"数据集不存在: {dataset_name}")

        try:
            await asyncio.wait_for(
                export_slots.acquire(),
                timeout=_DATASET_EXPORT_ACQUIRE_TIMEOUT,
            )
            slot_acquired = True
        except asyncio.TimeoutError as exc:
            raise HTTPException(
                status_code=503,
                detail="数据集导出任务繁忙，请稍后重试",
                headers={"Retry-After": "1"},
            ) from exc

        archive_path = _create_export_temp_path()
        stat_result = await _build_dataset_archive(resolved_dataset, archive_path)
        response = _TemporaryExportFileResponse(
            archive_path,
            filename=f"{dataset_name}.zip",
            stat_result=stat_result,
            semaphore=export_slots,
        )
        response_owns_slot = True
        return response
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("导出数据集失败")
        raise HTTPException(status_code=500, detail="导出数据集失败") from exc
    finally:
        if not response_owns_slot:
            try:
                if archive_path is not None:
                    await asyncio.shield(
                        asyncio.to_thread(_remove_export_temp, archive_path)
                    )
            finally:
                if slot_acquired:
                    export_slots.release()

@router.get("/api/training/datasets/scan")
async def scan_datasets(folder: Optional[str] = None, current_user: dict = Depends(get_current_admin)):
    """扫描文件夹，发现所有有效的数据集子文件夹"""
    try:
        from training.preprocessor import scan_datasets_folder, DEFAULT_SCAN_DIR
        results = scan_datasets_folder(folder)
        return {
            "success": True,
            "scan_path": folder or str(DEFAULT_SCAN_DIR),
            "datasets": results,
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"扫描数据集文件夹失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ImportDatasetRequest(BaseModel):
    source_path: str
    dataset_name: Optional[str] = None


@router.post("/api/training/datasets/scan/import")
async def import_dataset(req: ImportDatasetRequest, current_user: dict = Depends(get_current_admin)):
    """从扫描结果导入数据集到训练目录

    s2 fix: 可探测服务器目录结构并写入训练目录，限定 admin。
    """
    try:
        from training.preprocessor import import_dataset_from_folder
        dataset_name = _validate_resource_name(req.dataset_name, "数据集名称") if req.dataset_name else None
        return import_dataset_from_folder(req.source_path, dataset_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"导入数据集失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/training/models")
async def list_model_configs(current_user: dict = Depends(get_current_admin)):
    """列出可用的模型配置（支持多种GPU）"""
    try:
        from training.task_manager import ALL_GPU_CONFIGS

        configs = []
        for name, config in ALL_GPU_CONFIGS.items():
            # 根据配置类型判断GPU和描述
            is_3090 = isinstance(config, type) and "3090" in config.__class__.__name__
            if "3090" in name or is_3090:
                gpu_type = "RTX 3090 24GB"
                desc = f"RTX 3090 24GB优化配置 — Qwen3-8B FP16微调 (rank={config.lora_rank}, batch={config.per_device_train_batch_size}×{config.gradient_accumulation_steps}, seq={config.max_seq_length})"
            else:
                gpu_type = "RTX 4060 8GB"
                desc = f"RTX 4060 8GB优化配置 — Qwen3-8B 4bit量化微调 (rank={config.lora_rank}, batch={config.per_device_train_batch_size}×{config.gradient_accumulation_steps}, seq={config.max_seq_length})"

            configs.append({
                "name": name,
                "model_name": config.model_name_or_path,
                "gpu_type": gpu_type,
                "batch_size": config.per_device_train_batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "cutoff_len": config.max_seq_length,
                "lora_rank": config.lora_rank,
                "lora_alpha": config.lora_alpha,
                "lora_dropout": config.lora_dropout,
                "learning_rate": config.learning_rate,
                "num_train_epochs": config.num_train_epochs,
                "warmup_ratio": config.warmup_ratio,
                "weight_decay": config.weight_decay,
                "bf16": getattr(config, 'bf16', False),
                "fp16": getattr(config, 'fp16', True),
                "load_in_4bit": getattr(config, 'load_in_4bit', True),
                "use_gradient_checkpointing": getattr(config, 'use_gradient_checkpointing', True),
                "description": desc
            })

        return {"success": True, "configs": configs}
    except Exception as e:
        logger.error(f"获取模型配置失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/training/start")
async def start_training(request: TrainingStartRequest, current_user: dict = Depends(get_current_admin)):
    """启动LoRA训练

    C-S1 fix: 训练任务消耗大量 GPU/磁盘资源，限定 admin。
    """
    try:
        request.lora_name = _validate_resource_name(request.lora_name, "LoRA 名称")
        request.dataset_name = _validate_resource_name(request.dataset_name, "数据集名称")

        # 输入验证
        is_valid, errors = InputValidator.validate(request.model_dump(), TRAINING_SCHEMA)
        if not is_valid:
            raise HTTPException(status_code=422, detail={"message": "输入验证失败", "errors": errors})

        # 内存检查：训练需要大量GPU/CPU内存
        import psutil
        mem = psutil.virtual_memory()
        gpu_available = False
        gpu_mem_free = 0

        try:
            import torch
            if torch.cuda.is_available():
                gpu_available = True
                gpu_mem_free = torch.cuda.mem_get_info(0)[0] / (1024**3)  # (free, total)
        except (ImportError, RuntimeError):
            pass

        # 7B模型至少需要8GB GPU显存或16GB系统内存
        MIN_GPU_MEM_GB = 8.0
        MIN_SYSTEM_MEM_FREE_GB = 16.0
        system_mem_free_gb = mem.available / (1024**3)

        if gpu_available:
            if gpu_mem_free < MIN_GPU_MEM_GB:
                raise HTTPException(
                    status_code=507,
                    detail=f"GPU显存不足：可用 {gpu_mem_free:.1f}GB，训练至少需要 {MIN_GPU_MEM_GB:.0f}GB。"
                           f"建议将数据集导出后上传到服务器训练。"
                )
        else:
            if system_mem_free_gb < MIN_SYSTEM_MEM_FREE_GB:
                raise HTTPException(
                    status_code=507,
                    detail=f"系统内存不足：可用 {system_mem_free_gb:.1f}GB，CPU训练至少需要 {MIN_SYSTEM_MEM_FREE_GB:.0f}GB。"
                           f"建议将数据集导出后上传到服务器训练。"
                )

        from training.task_manager import get_simple_lora_trainer, ALL_GPU_CONFIGS
        from training.preprocessor import get_dataset_preprocessor

        trainer = get_simple_lora_trainer(db=db)
        preprocessor = get_dataset_preprocessor()

        # 验证数据集是否存在
        dataset_dir = preprocessor.data_dir / request.dataset_name
        if not dataset_dir.exists():
            raise HTTPException(status_code=404, detail=f"数据集不存在: {request.dataset_name}")

        # 获取训练配置（支持多种GPU配置）
        if request.model_type not in ALL_GPU_CONFIGS:
            available = ", ".join(ALL_GPU_CONFIGS.keys())
            raise HTTPException(status_code=400, detail=f"无效的模型配置: {request.model_type}，可选: {available}")

        base_config = ALL_GPU_CONFIGS[request.model_type]

        # 将 base_config 转换为字典
        from dataclasses import asdict
        base_config_dict = asdict(base_config)

        # 合并自定义配置
        custom_config = request.custom_config or {}

        # 合并配置：自定义配置覆盖基础配置
        merged_config = {**base_config_dict, **custom_config}

        # 启动训练
        task_id = await trainer.start_training(
            lora_name=request.lora_name,
            dataset_path=dataset_dir,
            config=merged_config
        )

        return {
            "success": True,
            "message": "训练任务已启动",
            "task_id": task_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"启动训练失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/training/tasks")
async def list_training_tasks(current_user: dict = Depends(get_current_admin)):
    """列出所有训练任务"""
    try:
        from training.task_manager import get_simple_lora_trainer
        trainer = get_simple_lora_trainer(db=db)

        tasks = await trainer.get_all_tasks()
        return {"success": True, "tasks": tasks}
    except Exception as e:
        logger.error(f"获取训练任务列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/training/tasks/{task_id}")
async def get_training_task(task_id: str, current_user: dict = Depends(get_current_admin)):
    """获取训练任务状态"""
    try:
        from training.task_manager import get_simple_lora_trainer
        trainer = get_simple_lora_trainer(db=db)

        task = await trainer.get_task_status(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="训练任务不存在")

        return {"success": True, "task": task}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取训练任务状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/training/tasks/{task_id}/cancel")
async def cancel_training_task(task_id: str, current_user: dict = Depends(get_current_admin)):
    """取消训练任务

    s2 fix: 训练任务影响全局 LoRA 资源，限定 admin。
    """
    try:
        from training.task_manager import get_simple_lora_trainer
        trainer = get_simple_lora_trainer(db=db)

        success = await trainer.cancel_task(task_id)
        if success:
            return {"success": True, "message": "训练任务已取消"}
        else:
            raise HTTPException(status_code=400, detail="无法取消该训练任务")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"取消训练任务失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/training/styles")
async def list_predefined_styles(current_user: dict = Depends(get_current_admin)):
    """列出预定义的人物风格"""
    try:
        from training.preprocessor import get_dataset_preprocessor
        preprocessor = get_dataset_preprocessor()

        styles = []
        user_styles = preprocessor.list_styles()

        for name, config in user_styles.items():
            styles.append({
                "name": name,
                "display_name": config.name,
                "description": config.description
            })

        return {"success": True, "styles": styles}
    except Exception as e:
        logger.error(f"获取预定义风格失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/training/generate-dialogues")
async def generate_dialogues(request: DialogueGenerateRequest, current_user: dict = Depends(get_current_admin),
                             database=Depends(get_request_database)):
    """基于角色描述生成LoRA训练对话数据

    s2 fix: 消耗 LLM token/GPU 资源，限定 admin。
    """

    global _dialogue_generation_task
    async with get_generation_state_lock():
        if _dialogue_generation_task is not None and not _dialogue_generation_task.done():
            raise HTTPException(status_code=409, detail="已有生成任务正在运行")

        total_batches, batch_plan = plan_dialogue_batches(request.num_dialogues)
        # 初始化状态
        generation_state.update({
            "is_generating": True,
            "cancel_requested": False,
            "progress": 0,
            "total": request.num_dialogues,
            "batch_num": 0,
            "total_batches": total_batches,
            "generated_count": 0,
            "new_dialogues": [],
            "all_generated_dialogues": [],
            "started_at": time.time(),
        })

        task = asyncio.create_task(_run_dialogue_generation(request, database, batch_plan), name="dialogue-generation")
        _dialogue_generation_task = task

    try:
        result = await task
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError
        return result
    except asyncio.CancelledError as exc:
        if asyncio.current_task().cancelling() or exc.args != (_DIALOGUE_CANCELLED,):
            raise
        return _dialogue_result([], 0.0, cancelled=True)
    finally:
        async with get_generation_state_lock():
            if _dialogue_generation_task is task:
                generation_state.update({"is_generating": False, "cancel_requested": False})


def _dialogue_result(dialogues, cost, *, cancelled):
    return {
        "success": not cancelled,
        "dialogues": dialogues,
        "total": len(dialogues),
        "cost_time": round(cost, 2),
        "cancelled": cancelled,
    }


def _dialogue_error_detail(detail, dialogues, cost):
    """Return only this task's validated batches alongside its failure."""
    if not dialogues:
        return detail
    return {
        **(detail if isinstance(detail, dict) else {"message": detail}),
        "partial_result": {
            "dialogues": dialogues,
            "total": len(dialogues),
            "cost_time": round(cost, 2),
        },
    }


async def _run_dialogue_generation(request, database, batch_plan):
    all_validated = []
    total_cost = 0.0
    try:
        from inference.model_manager import get_model_manager
        from inference.request_context import resolve_request_context

        context = await resolve_request_context(get_model_manager())
        provider = context.provider
        if context.provider_name == "mock":
            raise HTTPException(status_code=400, detail="当前模型提供商为 mock 模式，无法调用大模型生成对话。请在设置中配置有效的模型提供商（如 DeepSeek API）。")

        research = await search_character_info(request.character_description)
        total_target = request.num_dialogues
        task_id = uuid.uuid4().hex
        model_name = provider.get_status()["modelName"]
        for global_batch_num, batch in enumerate(batch_plan, 1):
            async with get_generation_state_lock():
                if generation_state["cancel_requested"]:
                    logger.info("Dialogue generation cancelled after %s dialogues", len(all_validated))
                    break
                generation_state["batch_num"] = global_batch_num

            system_prompt = build_dialogue_prompt(
                request.character_description, batch, style=request.style,
                custom_prompt=request.custom_prompt, research=research,
            )
            async with account_usage(database, {
                "traceId": f"dialogue:{task_id}:{global_batch_num}", "sessionId": task_id,
                "platform": "web", "modelName": model_name, "purpose": "dialogue_generation",
            }):
                if context.provider_name != "openai_compat":
                    begin_completion()
                result, cost = await provider.async_generate(
                    prompt=system_prompt,
                    session_history=[],
                    max_tokens_override=batch.max_tokens
                )
                total_cost += cost

                try:
                    conversations = parse_dialogue_batch(result, batch.count, batch.turns)
                except DialogueBatchError as exc:
                    logger.warning("Dialogue batch %s failed validation: %s", global_batch_num, exc.reason)
                    raise HTTPException(
                        status_code=502,
                        detail={
                            "error": "dialogue_batch_invalid",
                            "stage": "dialogue_validation",
                            "reason": exc.reason,
                            "batch": global_batch_num,
                            "expected_count": batch.count,
                            "target_turns": batch.turns,
                            "completed_count": len(all_validated),
                            "requested_count": total_target,
                            "message": "模型返回的对话批次不符合约定，请检查模型服务及输出。",
                        },
                    ) from exc
            new_in_batch = [
                {
                    "conversations": messages,
                    "system": f"你是{request.character_description}，请始终保持角色设定和语言风格",
                    "scene": scene,
                    "turns": batch.turns,
                    "tags": ([request.style] if request.style else []) + [f"{batch.turns}轮"],
                }
                for messages, scene in zip(conversations, batch.scenes)
            ]
            all_validated.extend(new_in_batch)
            async with get_generation_state_lock():
                generation_state.update({
                    "new_dialogues": new_in_batch,
                    "generated_count": len(all_validated),
                    "progress": round(len(all_validated) / total_target * 100, 1),
                })
                generation_state["all_generated_dialogues"].extend(new_in_batch)
            logger.info("Dialogue generation progress: %s/%s", len(all_validated), total_target)

        async with get_generation_state_lock():
            was_cancelled = generation_state["cancel_requested"]

        if not was_cancelled and len(all_validated) != total_target:
            raise RuntimeError("Dialogue plan did not produce its requested count")

        return _dialogue_result(all_validated, total_cost, cancelled=was_cancelled)

    except asyncio.CancelledError:
        if not generation_state["cancel_requested"]:
            raise
        return _dialogue_result(all_validated, total_cost, cancelled=True)
    except CharacterResearchError as exc:
        logger.warning("Character research failed: source=%s reason=%s status=%s", exc.source, exc.reason, exc.status_code)
        raise HTTPException(
            status_code=504 if exc.timed_out else 502,
            detail={
                "error": "character_research_failed",
                "stage": "character_research",
                "source": exc.source,
                "reason": exc.reason,
                "upstream_status": exc.status_code,
                "message": "角色资料搜索失败，请恢复搜索服务后重试。",
            },
        ) from exc
    except UsageRecordingError as exc:
        raise HTTPException(
            status_code=exc.http_status,
            detail=_dialogue_error_detail(exc.public_detail(), all_validated, total_cost),
        ) from exc
    except ProviderResponseError as exc:
        logger.warning("Dialogue model request failed: upstream_status=%s", exc.status_code)
        raise HTTPException(
            status_code=exc.http_status,
            detail=_dialogue_error_detail(exc.public_detail(), all_validated, total_cost),
        ) from exc
    except HTTPException as exc:
        exc.detail = _dialogue_error_detail(exc.detail, all_validated, total_cost)
        raise
    except Exception as exc:
        logger.error("Dialogue generation failed: %s", type(exc).__name__)
        raise HTTPException(
            status_code=500,
            detail=_dialogue_error_detail(
                "对话生成失败，请检查模型服务及后端日志。", all_validated, total_cost,
            ),
        ) from exc



@router.post("/api/training/generate-dialogues/cancel")
async def cancel_dialogue_generation(current_user: dict = Depends(get_current_admin)):
    """取消正在进行的对话生成

    s2 fix: 影响全局生成任务，限定 admin。
    """
    async with get_generation_state_lock():
        task = _dialogue_generation_task
        if task is None or task.done():
            # C13 fix: 无任务可取消是冲突状态，应用 409 而非 200+success=False
            raise HTTPException(status_code=409, detail="没有正在进行的生成任务")
        generation_state["cancel_requested"] = True
        task.cancel(_DIALOGUE_CANCELLED)
    return {"success": True, "message": "已发送取消请求"}


@router.get("/api/training/generate-dialogues/progress")
async def get_dialogue_generation_progress(current_user: dict = Depends(get_current_admin)):
    """获取对话生成进度（含新增对话的实时推送 + 所有已生成对话）"""
    async with get_generation_state_lock():
        new_dialogues = generation_state.get("new_dialogues", [])
        generation_state["new_dialogues"] = []
        result = {**generation_state, "new_dialogues": new_dialogues}
    return result


@router.post("/api/training/generate-dialogues/force-reset")
async def force_reset_generation(current_user: dict = Depends(get_current_admin)):
    """强制重置生成状态（用于断线重连后清理残留状态）

    C-S1 fix: 强制重置会丢失正在进行的生成进度，限定 admin。
    """
    async with get_generation_state_lock():
        task = _dialogue_generation_task
        if task is None or task.done():
            # C13 fix: 无任务可重置是冲突状态，应用 409 而非 200+success=False
            raise HTTPException(status_code=409, detail="没有正在进行的生成任务")
        generation_state["cancel_requested"] = True
        task.cancel(_DIALOGUE_CANCELLED)
    await asyncio.gather(task, return_exceptions=True)
    async with get_generation_state_lock():
        if _dialogue_generation_task is task:
            generation_state.update({
                "is_generating": False,
                "cancel_requested": False,
                "progress": 0,
                "total": 0,
                "batch_num": 0,
                "total_batches": 0,
                "generated_count": 0,
                "new_dialogues": [],
                "all_generated_dialogues": [],
                "started_at": None,
            })
    logger.info("Original dialogue generation task terminated")
    return {"success": True, "message": "原生成任务已终止"}


# ═══════════════════════════════════════════════════════════
# 已保存对话管理 API
# ═══════════════════════════════════════════════════════════

@router.get("/api/training/saved-dialogues")
async def list_saved_dialogues(current_user: dict = Depends(get_current_admin), database=Depends(get_request_database)):
    """列出所有已保存的对话"""
    rows = await run_db(
        database.execute_sql,
        """
            SELECT id, name, character_desc, style, dialogue_count,
                   turn_stats, scene_stats, created_at, updated_at
            FROM saved_dialogues ORDER BY updated_at DESC
        """,
    )
    items = []
    for row in rows:
        items.append(
            {
                "id": row["id"],
                "name": row["name"],
                "character_desc": row["character_desc"],
                "style": row["style"],
                "dialogue_count": row["dialogue_count"],
                "turn_stats": json.loads(row["turn_stats"]) if row["turn_stats"] else None,
                "scene_stats": json.loads(row["scene_stats"]) if row["scene_stats"] else None,
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )
    return {"success": True, "items": items}


@router.post("/api/training/saved-dialogues")
async def save_dialogues(
    request: SaveDialoguesRequest,
    current_user: dict = Depends(get_current_admin),
    database=Depends(get_request_database),
):
    """保存对话数据

    s2 fix: saved_dialogues 表无 user_id 字段，无法做所有权校验。
    为防止 IDOR（任意用户删除/修改他人对话），统一限定 admin。
    """
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    result = await run_db(
        database.execute_sql_insert,
        """
            INSERT INTO saved_dialogues
            (name, character_desc, style, dialogue_count, dialogues_json, turn_stats, scene_stats, created_at, updated_at)
            VALUES (:name, :character_desc, :style, :dialogue_count, :dialogues_json, :turn_stats, :scene_stats, :created_at, :updated_at)
        """,
        {
            "name": request.name,
            "character_desc": request.character_desc,
            "style": request.style,
            "dialogue_count": len(request.dialogues),
            "dialogues_json": json.dumps(request.dialogues, ensure_ascii=False),
            "turn_stats": json.dumps(request.turn_stats) if request.turn_stats else None,
            "scene_stats": json.dumps(request.scene_stats) if request.scene_stats else None,
            "created_at": now,
            "updated_at": now,
        },
    )
    return {"success": True, "id": result["lastrowid"], "message": "保存成功"}


@router.get("/api/training/saved-dialogues/{item_id}")
async def get_saved_dialogue(
    item_id: int, current_user: dict = Depends(get_current_admin), database=Depends(get_request_database)
):
    """获取单个已保存对话"""
    rows = await run_db(
        database.execute_sql,
        """
            SELECT id, name, character_desc, style, dialogue_count,
                   dialogues_json, turn_stats, scene_stats, created_at, updated_at
            FROM saved_dialogues WHERE id = :item_id
        """,
        {"item_id": item_id},
    )
    if not rows:
        raise HTTPException(status_code=404, detail="对话不存在")
    row = rows[0]
    return {
        "success": True,
        "id": row["id"],
        "name": row["name"],
        "character_desc": row["character_desc"],
        "style": row["style"],
        "dialogue_count": row["dialogue_count"],
        "dialogues": json.loads(row["dialogues_json"]),
        "turn_stats": json.loads(row["turn_stats"]) if row["turn_stats"] else None,
        "scene_stats": json.loads(row["scene_stats"]) if row["scene_stats"] else None,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


@router.delete("/api/training/saved-dialogues/{item_id}")
async def delete_saved_dialogue(
    item_id: int, current_user: dict = Depends(get_current_admin), database=Depends(get_request_database)
):
    """删除已保存对话

    s2 fix: IDOR 防护，限定 admin。
    """
    rowcount = await run_db(
        database.execute_sql, "DELETE FROM saved_dialogues WHERE id = :item_id", {"item_id": item_id}
    )
    if rowcount == 0:
        raise HTTPException(status_code=404, detail="对话不存在")
    return {"success": True, "message": "删除成功"}


@router.delete("/api/training/saved-dialogues/{item_id}/dialogues/{dialogue_index}")
async def delete_dialogue_from_saved(
    item_id: int,
    dialogue_index: int,
    current_user: dict = Depends(get_current_admin),
    database=Depends(get_request_database),
):
    """从已保存对话中删除单条对话

    s2 fix: IDOR 防护，限定 admin。
    """
    rows = await run_db(
        database.execute_sql, "SELECT dialogues_json FROM saved_dialogues WHERE id = :item_id", {"item_id": item_id}
    )
    if not rows:
        raise HTTPException(status_code=404, detail="对话不存在")

    dialogues = json.loads(rows[0]["dialogues_json"])
    if dialogue_index < 0 or dialogue_index >= len(dialogues):
        raise HTTPException(status_code=400, detail="索引越界")

    dialogues.pop(dialogue_index)
    turn_stats = Counter(str(len(item.get("conversations") or [])) for item in dialogues)
    scene_stats = Counter(item.get("scene") or "未知" for item in dialogues)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    await run_db(
        database.execute_sql,
        """
            UPDATE saved_dialogues
            SET dialogues_json = :dialogues_json, dialogue_count = :dialogue_count,
            turn_stats = :turn_stats, scene_stats = :scene_stats, updated_at = :updated_at
            WHERE id = :item_id
        """,
        {
            "dialogues_json": json.dumps(dialogues, ensure_ascii=False),
            "dialogue_count": len(dialogues),
            "turn_stats": json.dumps(turn_stats),
            "scene_stats": json.dumps(scene_stats),
            "updated_at": now,
            "item_id": item_id,
        },
    )
    return {"success": True, "message": "删除成功", "remaining_count": len(dialogues)}


@router.post("/api/training/saved-dialogues/{item_id}/create-dataset")
async def create_dataset_from_saved(
    item_id: int,
    dataset_name: Optional[str] = None,
    current_user: dict = Depends(get_current_admin),
    database=Depends(get_request_database),
):
    """从已保存对话创建训练数据集

    s2 fix: IDOR 防护，限定 admin。
    """
    rows = await run_db(
        database.execute_sql,
        "SELECT name, dialogues_json FROM saved_dialogues WHERE id = :item_id",
        {"item_id": item_id},
    )
    if not rows:
        raise HTTPException(status_code=404, detail="对话不存在")

    saved_name = rows[0]["name"]
    dialogues = json.loads(rows[0]["dialogues_json"])
    # 清理文件名中的非法字符（Windows不允许 / \ : * ? " < > |）
    raw_name = dataset_name or f"{saved_name}_dataset"
    ds_name = _validate_resource_name(raw_name, "数据集名称")

    # 转换为训练数据格式
    training_data = []
    for d in dialogues:
        convs = d.get("conversations", [])
        if convs:
            training_data.append(
                {
                    "conversations": convs,
                    "system": d.get("system", ""),
                }
            )

    if not training_data:
        raise HTTPException(status_code=400, detail="没有有效的对话数据可创建数据集")

    # 使用 preprocessor 创建标准数据集目录结构（与 list_datasets 兼容）
    try:
        from training.preprocessor import get_dataset_preprocessor

        preprocessor = get_dataset_preprocessor()
        dataset_dir, stats = preprocessor.prepare_training_data(
            raw_data=training_data,
            output_name=ds_name,
            min_samples=1,
        )
    except ValueError as ve:
        # preprocessor 校验失败时给出更友好的错误
        raise HTTPException(status_code=400, detail=str(ve))

    return {
        "success": True,
        "dataset": {
            "name": ds_name,
            "path": str(dataset_dir),
            "count": stats["total_samples"],
            "stats": stats,
        },
    }
