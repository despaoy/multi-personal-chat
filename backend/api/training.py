"""LoRA训练管理API"""
import asyncio
import json
import logging
import math
import os
import re
import tempfile
import time
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from app.dependencies import get_current_admin
from infra.db_executor import run_db
from pydantic import BaseModel
from typing import Optional

from db.adapter import db
from db.schemas import DatasetUploadRequest, TrainingStartRequest, DialogueGenerateRequest
from app.config import INPUT_VALIDATOR_AVAILABLE, TRAINING_SCHEMA, generation_state, get_generation_state_lock, _search_character_info

logger = logging.getLogger(__name__)
router = APIRouter()


_DATASET_EXPORT_MAX_CONCURRENCY = max(
    1, int(os.getenv("TRAINING_EXPORT_MAX_CONCURRENCY", "2"))
)
_DATASET_EXPORT_ACQUIRE_TIMEOUT = max(
    0.05, float(os.getenv("TRAINING_EXPORT_ACQUIRE_TIMEOUT", "1"))
)
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
        # Validate source_path to prevent path traversal
        backend_dir = str(Path(__file__).parent.parent.resolve())
        allowed_dirs = [backend_dir]
        # Allow a configured lab root when present. The default remains the
        # backend directory only, so local deployments are fail-closed.
        lab_root = (os.getenv("MULTIPERSONAL_LAB_ROOT") or os.getenv("QQCHAT_LAB_ROOT", "")).strip()
        if lab_root and Path(lab_root).exists():
            allowed_dirs.append(str(Path(lab_root).resolve()))
        try:
            validated_path = _validate_path(req.source_path)
            resolved = Path(validated_path)
            if not any(resolved.is_relative_to(Path(d)) for d in allowed_dirs):
                raise HTTPException(status_code=400, detail="源路径不在允许的目录范围内")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        from training.preprocessor import import_dataset_from_folder
        dataset_name = _validate_resource_name(req.dataset_name, "数据集名称") if req.dataset_name else None
        result = import_dataset_from_folder(req.source_path, dataset_name)
        return result
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
        if INPUT_VALIDATOR_AVAILABLE:
            from infra.input_validator import InputValidator
            is_valid, errors = InputValidator.validate(request.model_dump(), TRAINING_SCHEMA)
            if not is_valid:
                raise HTTPException(status_code=422, detail={"message": "输入验证失败", "errors": errors})

        # 内存检查：训练需要大量GPU/CPU内存
        import psutil
        mem = psutil.virtual_memory()
        gpu_available = False
        gpu_mem_total = 0
        gpu_mem_free = 0

        try:
            import torch
            if torch.cuda.is_available():
                gpu_available = True
                gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
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
async def generate_dialogues(request: DialogueGenerateRequest, current_user: dict = Depends(get_current_admin)):
    """基于角色描述生成LoRA训练对话数据

    s2 fix: 消耗 LLM token/GPU 资源，限定 admin。
    """

    # 防止重复生成（异步安全检查+初始化）
    async with get_generation_state_lock():
        if generation_state["is_generating"]:
            raise HTTPException(status_code=409, detail="已有生成任务正在运行")

        # 初始化状态
        generation_state.update({
            "is_generating": True,
            "cancel_requested": False,
            "progress": 0,
            "total": request.num_dialogues,
            "batch_num": 0,
            "total_batches": (request.num_dialogues + 19) // 20,
            "generated_count": 0,
            "new_dialogues": [],
            "all_generated_dialogues": [],
            "started_at": time.time(),
        })

    try:
        from inference.model_manager import get_model_manager
        manager = get_model_manager()

        # 检查模型提供商是否可用
        if manager._current_provider.value == "mock":
            raise HTTPException(status_code=400, detail="当前模型提供商为 mock 模式，无法调用大模型生成对话。请在设置中配置有效的模型提供商（如 DeepSeek API）。")

        # 构建生成 prompt - 风格描述
        style_instruction = ""
        if request.style:
            style_instruction = f"角色风格：{request.style}。"
        if request.custom_prompt:
            style_instruction += f"\n额外要求：{request.custom_prompt}"

        # ============================================================
        # 网络搜索角色信息，丰富角色背景
        # ============================================================
        character_context = ""
        try:
            search_results = await asyncio.to_thread(_search_character_info, request.character_description)
            if search_results:
                character_context = f"""

【角色背景参考信息（来自网络搜索）】
以下是从网络上搜索到的关于该角色的真实信息，请在生成对话时严格参考：

{search_results}

请根据以上真实角色信息生成对话，确保角色的言行符合其官方设定和性格特点。"""
                logger.info(f"角色信息搜索成功，获得参考信息 {len(search_results)} 字符")
        except Exception as e:
            logger.warning(f"角色信息搜索失败（不影响生成）: {e}")

        total_target = request.num_dialogues
        all_validated = []
        total_cost = 0.0

        # ============================================================
        # 轮次分布策略 —— 模拟真人聊天中不同长度的对话比例
        # ============================================================
        TURN_WEIGHTS = [
            (1, 0.05),
            (2, 0.10),
            (3, 0.22),
            (4, 0.22),
            (5, 0.18),
            (6, 0.12),
            (7, 0.07),
            (8, 0.04),
        ]

        # 按权重计算每种轮次需要生成的数量
        turn_targets = []
        cumulative = 0
        for turns, weight in TURN_WEIGHTS:
            count = max(1, round(total_target * weight))
            if cumulative + count > total_target:
                count = total_target - cumulative
            if count > 0:
                turn_targets.append((turns, count))
            cumulative += count
        if cumulative < total_target:
            turn_targets[-1] = (turn_targets[-1][0], turn_targets[-1][1] + total_target - cumulative)

        # ============================================================
        # 场景类型
        # ============================================================
        scene_types = [
            "日常问候：打招呼、问好、早晚安、天气、吃了没",
            "闲聊吐槽：今天发生的事、遇到的人、看到的趣闻、抱怨工作/学习",
            "兴趣爱好：游戏/动漫/音乐/电影/书籍讨论、安利、吐槽",
            "情感倾诉：开心/难过/烦恼/压力、求安慰、分享秘密",
            "角色扮演互动：以角色身份与朋友玩乐、耍宝、搞笑",
            "求助与建议：问路/买东西/学东西/人际关系问题，求建议",
            "观点讨论：对某件事的看法、争议话题、立场表达",
            "知识科普：角色用自己熟悉的知识解释/科普某些概念",
            "回忆与故事：讲述角色的经历、过去的事情、有趣的故事",
            "幽默互怼：朋友之间开玩笑、相互调侃、冷笑话",
            "深度关怀：一方状态不好时另一方关心、鼓励、陪伴",
            "突发奇想：突然想到一个点子/问题、天马行空的对话",
        ]

        BATCH_SIZE_PER_TURN = 6
        estimated_batches = sum(max(1, math.ceil(count / BATCH_SIZE_PER_TURN)) for _, count in turn_targets)
        async with get_generation_state_lock():
            generation_state["total_batches"] = estimated_batches
        global_batch_num = 0
        scene_idx = 0

        for target_turns, need_count in turn_targets:
            generated_this_turn = 0
            retry_count = 0
            max_retries_per_turn = need_count * 3  # 防止模型持续生成不足轮次导致死循环

            while generated_this_turn < need_count:
                async with get_generation_state_lock():
                    if generation_state["cancel_requested"]:
                        cancel_flag = True
                    else:
                        cancel_flag = False
                if cancel_flag:
                    logger.info(f"对话生成已取消，已生成 {len(all_validated)} 组")
                    break

                retry_count += 1
                if retry_count > max_retries_per_turn:
                    logger.warning(f"{target_turns}轮对话重试已达上限({max_retries_per_turn})，已生成{generated_this_turn}/{need_count}组，跳过")
                    break

                global_batch_num += 1
                batch_count = min(BATCH_SIZE_PER_TURN, need_count - generated_this_turn)

                async with get_generation_state_lock():
                    generation_state.update({
                        "batch_num": global_batch_num,
                        "generated_count": len(all_validated),
                        "progress": round(len(all_validated) / total_target * 100, 1),
                    })

                batch_scenes = []
                for i in range(batch_count):
                    batch_scenes.append(scene_types[(scene_idx + i) % len(scene_types)])
                scene_idx = (scene_idx + batch_count) % len(scene_types)

                scene_desc = "\n".join(f"  对话{i+1}场景：{s}" for i, s in enumerate(batch_scenes))

                if target_turns == 1:
                    turn_guide = "单轮对话，用户说一句、角色回复一句即可。用户消息可以是打招呼、道别、简短提问、确认信息等。角色回复要简洁自然但有个性。"
                    example_fmt = """[
  {"conversations": [{"role":"user","content":"早上好呀～"},{"role":"assistant","content":"嘿嘿早上好！今天精神不错嘛～"}]},
  {"conversations": [{"role":"user","content":"帮我查下天气"},{"role":"assistant","content":"行啊我看看……呃好像我也查不到，要不你自己看看？"}]}
]"""
                elif target_turns == 2:
                    turn_guide = "两轮对话，用户发起话题→角色回应→用户追问或回应→角色再回应。话题自然切换，可以稍有转折。"
                    example_fmt = """[
  {"conversations": [{"role":"user","content":"你最近在忙啥"},{"role":"assistant","content":"最近忙着给往生堂整理账本呢，快累死我了"},{"role":"user","content":"哈哈那你加油"},{"role":"assistant","content":"谢谢～你要不要来帮忙呀，请你吃饭！"}]}
]"""
                elif target_turns <= 3:
                    turn_guide = "三轮对话，有完整的起承转合：寒暄/引入→展开话题→自然结束或留下悬念。消息长度要有变化，不要每轮都对称。"
                elif target_turns <= 5:
                    turn_guide = "较长的多轮对话，话题可以有1-2次轻微转折。消息长度多样化：有的很简短（几个字）、有的较长（2-3句）。模拟真实朋友聊天，不需要每轮都像问答。"
                else:
                    turn_guide = "长对话，有完整的话题发展弧线。前面铺垫、中间深入、后面自然收敛或转折。要有口语化表达、语气词、甚至打字错误或口语修正，让对话非常像真人在qq上聊天。"

                if target_turns <= 2:
                    format_example = example_fmt
                else:
                    turns_desc = f"(共{target_turns}对 user+assistant)"
                    format_example = f"""[{{
  "conversations": [
    {{"role":"user","content":"口语化的用户消息"}},
    {{"role":"assistant","content":"有人物特点的回复"}},
    {{"role":"user","content":"自然的追问或其他话题"}},
    {{"role":"assistant","content":"自然的回应"}},
    ...
    {turns_desc}
  ]
}}]"""

                system_prompt = f"""你是一个专业的QQ聊天对话数据生成器。请生成{batch_count}组对话数据，每组恰好{target_turns}轮对话。

角色：{request.character_description}
{style_instruction}{character_context}

【每组对话的场景】
{scene_desc}

【本批对话要求】
- 轮次数：每组恰好{target_turns}轮（一轮 = 一条用户消息 + 一条角色回复）
- {turn_guide}

【拟人化聊天要求 - 非常重要】
- 用户消息要像真人在QQ聊天：口语化、有语气词（呢、嘛、吧、呀、哈、嘿嘿、哈哈哈、555、orz、woc等）、可能打字快不带标点、也有正式一点的表达
- 消息长度随机变化：有时一个字（"嗯"、"？」、"草"），有时一句话，有时两三句
- 聊天节奏自然：有时秒回，有时角色可以"正在输入"；对话可以被打断、话题可以跳转
- 角色回复要符合角色性格，但不要每句都像在"表演角色"，要像真人在聊天
- 避免模板化：不要每组对话都"你好→你好→再见→再见"
- 用户可以反悔/改口："算了不说这个了"、"突然想起来…"
- 对话要有真实感而不是"教科书问答"

【输出格式】
严格按以下JSON数组格式输出{batch_count}个对象，每组恰好{target_turns}对 user/assistant 消息：

{format_example}

直接输出JSON数组，不要markdown代码块标记。"""

                # 根据轮次计算合理输出上限（本地模型 context 有限，系统 prompt 已占 ~1000-1500 tokens）
                max_tok = 2048 if target_turns <= 2 else 4096 if target_turns <= 5 else 6144
                result, cost = await asyncio.to_thread(
                    manager.generate,
                    prompt=system_prompt,
                    session_history=[],
                    rag_docs=[],
                    max_tokens_override=max_tok
                )
                total_cost += cost

                cleaned = result.strip()
                cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
                cleaned = re.sub(r'\s*```$', '', cleaned)

                try:
                    dialogues = json.loads(cleaned)
                except json.JSONDecodeError:
                    match = re.search(r'\[.*\]', cleaned, re.DOTALL)
                    if match:
                        try:
                            dialogues = json.loads(match.group())
                        except json.JSONDecodeError as e2:
                            logger.warning(f"批次{global_batch_num}解析失败: {e2}, 内容前500字: {cleaned[:500]}")
                            continue
                    else:
                        logger.warning(f"批次{global_batch_num}格式无效, 内容前500字: {cleaned[:500]}")
                        continue

                if not isinstance(dialogues, list):
                    logger.warning(f"批次{global_batch_num}返回值不是数组")
                    continue

                batch_start_count = len(all_validated)
                scene_iter = iter(batch_scenes)

                for item in dialogues:
                    if not isinstance(item, dict):
                        continue
                    convs = item.get("conversations", [])
                    if not convs or not isinstance(convs, list):
                        continue
                    # 第一阶段：过滤有效消息
                    raw_convs = []
                    for conv in convs:
                        if isinstance(conv, dict) and "role" in conv and "content" in conv:
                            if conv["role"] in ("user", "assistant") and conv["content"].strip():
                                raw_convs.append({
                                    "from": "human" if conv["role"] == "user" else "gpt",
                                    "value": conv["content"].strip()
                                })

                    # 第二阶段：配对验证 —— 确保消息严格交替 human/gpt
                    paired_convs = []
                    i = 0
                    while i + 1 < len(raw_convs):
                        if raw_convs[i]["from"] == "human" and raw_convs[i + 1]["from"] == "gpt":
                            paired_convs.append(raw_convs[i])
                            paired_convs.append(raw_convs[i + 1])
                            i += 2
                        else:
                            # 跳过不匹配的消息，尝试下一对
                            i += 1

                    # 第三阶段：严格要求对话轮数
                    # 每轮 = 1条human + 1条gpt = 2条消息
                    min_required = target_turns * 2
                    if len(paired_convs) >= min_required:
                        # 截断到恰好目标轮数（防止模型多生成）
                        valid_convs = paired_convs[:min_required]
                        scene = next(scene_iter, "闲聊")
                        all_validated.append({
                            "conversations": valid_convs,
                            "system": f"你是{request.character_description}，请始终保持角色设定和语言风格",
                            "scene": scene,
                            "turns": len(valid_convs) // 2,
                            "tags": ([request.style] if request.style else []) + [f"{target_turns}轮"]
                        })

                generated_this_turn += len(all_validated) - batch_start_count

                new_in_batch = all_validated[batch_start_count:]
                if new_in_batch:
                    async with get_generation_state_lock():
                        generation_state["new_dialogues"] = new_in_batch
                        generation_state["all_generated_dialogues"].extend(new_in_batch)

                logger.info(f"对话生成进度: {len(all_validated)}/{total_target} ({target_turns}轮批次, 已生成{generated_this_turn}/{need_count})")

            async with get_generation_state_lock():
                if generation_state["cancel_requested"]:
                    break

        async with get_generation_state_lock():
            was_cancelled = generation_state["cancel_requested"]

        if not all_validated and not was_cancelled:
            raise HTTPException(
                status_code=500,
                detail="生成的对话数据验证失败，请重试"
            )

        all_validated = all_validated[:total_target]

        async with get_generation_state_lock():
            generation_state.update({
                "is_generating": False,
                "progress": 100 if not was_cancelled else round(len(all_validated) / total_target * 100, 1),
                "generated_count": len(all_validated),
            })

        return {
            "success": True,
            "dialogues": all_validated,
            "total": len(all_validated),
            "cost_time": round(total_cost, 2),
            "cancelled": was_cancelled,
        }

    except HTTPException:
        async with get_generation_state_lock():
            generation_state.update({"is_generating": False, "cancel_requested": False})
        raise
    except Exception as e:
        async with get_generation_state_lock():
            generation_state.update({"is_generating": False, "cancel_requested": False})
        logger.error(f"对话生成失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/training/generate-dialogues/cancel")
async def cancel_dialogue_generation(current_user: dict = Depends(get_current_admin)):
    """取消正在进行的对话生成

    s2 fix: 影响全局生成任务，限定 admin。
    """
    async with get_generation_state_lock():
        if not generation_state["is_generating"]:
            # C13 fix: 无任务可取消是冲突状态，应用 409 而非 200+success=False
            raise HTTPException(status_code=409, detail="没有正在进行的生成任务")
        generation_state["cancel_requested"] = True
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
        if not generation_state["is_generating"] and not generation_state["cancel_requested"]:
            # C13 fix: 无任务可重置是冲突状态，应用 409 而非 200+success=False
            raise HTTPException(status_code=409, detail="没有正在进行的生成任务")
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
    logger.info("生成状态已强制重置")
    return {"success": True, "message": "生成状态已强制重置"}


# ═══════════════════════════════════════════════════════════
# 已保存对话管理 API
# ═══════════════════════════════════════════════════════════

class SaveDialoguesRequest(BaseModel):
    name: str
    character_desc: str
    style: Optional[str] = None
    dialogues: list
    turn_stats: Optional[dict] = None
    scene_stats: Optional[dict] = None


@router.get("/api/training/saved-dialogues")
async def list_saved_dialogues(current_user: dict = Depends(get_current_admin)):
    """列出所有已保存的对话"""
    try:
        rows = await run_db(db.execute_sql, '''
            SELECT id, name, character_desc, style, dialogue_count,
                   turn_stats, scene_stats, created_at, updated_at
            FROM saved_dialogues ORDER BY updated_at DESC
        ''')
        items = []
        for row in rows:
            items.append({
                "id": row["id"],
                "name": row["name"],
                "character_desc": row["character_desc"],
                "style": row["style"],
                "dialogue_count": row["dialogue_count"],
                "turn_stats": json.loads(row["turn_stats"]) if row["turn_stats"] else None,
                "scene_stats": json.loads(row["scene_stats"]) if row["scene_stats"] else None,
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            })
        return {"success": True, "items": items}
    except Exception as e:
        logger.error(f"列出已保存对话失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/training/saved-dialogues")
async def save_dialogues(request: SaveDialoguesRequest, current_user: dict = Depends(get_current_admin)):
    """保存对话数据

    s2 fix: saved_dialogues 表无 user_id 字段，无法做所有权校验。
    为防止 IDOR（任意用户删除/修改他人对话），统一限定 admin。
    """
    try:
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        result = await run_db(db.execute_sql_insert, '''
            INSERT INTO saved_dialogues
            (name, character_desc, style, dialogue_count, dialogues_json, turn_stats, scene_stats, created_at, updated_at)
            VALUES (:name, :character_desc, :style, :dialogue_count, :dialogues_json, :turn_stats, :scene_stats, :created_at, :updated_at)
        ''', {
            "name": request.name,
            "character_desc": request.character_desc,
            "style": request.style,
            "dialogue_count": len(request.dialogues),
            "dialogues_json": json.dumps(request.dialogues, ensure_ascii=False),
            "turn_stats": json.dumps(request.turn_stats) if request.turn_stats else None,
            "scene_stats": json.dumps(request.scene_stats) if request.scene_stats else None,
            "created_at": now,
            "updated_at": now,
        })
        return {"success": True, "id": result["lastrowid"], "message": "保存成功"}
    except Exception as e:
        logger.error(f"保存对话失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/training/saved-dialogues/{item_id}")
async def get_saved_dialogue(item_id: int, current_user: dict = Depends(get_current_admin)):
    """获取单个已保存对话"""
    try:
        rows = await run_db(db.execute_sql, '''
            SELECT id, name, character_desc, style, dialogue_count,
                   dialogues_json, turn_stats, scene_stats, created_at, updated_at
            FROM saved_dialogues WHERE id = :item_id
        ''', {"item_id": item_id})
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取已保存对话失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/training/saved-dialogues/{item_id}")
async def delete_saved_dialogue(item_id: int, current_user: dict = Depends(get_current_admin)):
    """删除已保存对话

    s2 fix: IDOR 防护，限定 admin。
    """
    try:
        rowcount = await run_db(db.execute_sql, 'DELETE FROM saved_dialogues WHERE id = :item_id', {"item_id": item_id})
        if rowcount == 0:
            raise HTTPException(status_code=404, detail="对话不存在")
        return {"success": True, "message": "删除成功"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除已保存对话失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/training/saved-dialogues/{item_id}/dialogues/{dialogue_index}")
async def delete_dialogue_from_saved(item_id: int, dialogue_index: int, current_user: dict = Depends(get_current_admin)):
    """从已保存对话中删除单条对话

    s2 fix: IDOR 防护，限定 admin。
    """
    try:
        rows = await run_db(db.execute_sql, 'SELECT dialogues_json FROM saved_dialogues WHERE id = :item_id', {"item_id": item_id})
        if not rows:
            raise HTTPException(status_code=404, detail="对话不存在")

        dialogues = json.loads(rows[0]["dialogues_json"])
        if dialogue_index < 0 or dialogue_index >= len(dialogues):
            raise HTTPException(status_code=400, detail="索引越界")

        dialogues.pop(dialogue_index)
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        await run_db(db.execute_sql, '''
            UPDATE saved_dialogues
            SET dialogues_json = :dialogues_json, dialogue_count = :dialogue_count, updated_at = :updated_at
            WHERE id = :item_id
        ''', {
            "dialogues_json": json.dumps(dialogues, ensure_ascii=False),
            "dialogue_count": len(dialogues),
            "updated_at": now,
            "item_id": item_id,
        })
        return {"success": True, "message": "删除成功", "remaining_count": len(dialogues)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除单条对话失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/training/saved-dialogues/{item_id}/create-dataset")
async def create_dataset_from_saved(item_id: int, dataset_name: Optional[str] = None, current_user: dict = Depends(get_current_admin)):
    """从已保存对话创建训练数据集

    s2 fix: IDOR 防护，限定 admin。
    """
    try:
        rows = await run_db(db.execute_sql, 'SELECT name, dialogues_json FROM saved_dialogues WHERE id = :item_id', {"item_id": item_id})
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
                training_data.append({
                    "conversations": convs,
                    "system": d.get("system", ""),
                })

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
                "count": stats.get("total_samples", len(training_data)),
                "stats": stats,
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建数据集失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
