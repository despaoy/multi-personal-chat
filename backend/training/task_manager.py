"""
LoRA训练器兼容层
桥接main.py的API调用到优化版train_lora.py，提供与旧版SimpleLoRATrainer相同的接口。
负责任务队列管理、异步训练调度和状态跟踪。
"""

import logging
import os
import json
import asyncio
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict

from fastapi import HTTPException

logger = logging.getLogger(__name__)


def _lora_root() -> Path:
    configured = os.getenv("LORA_PATH", "").strip()
    path = Path(configured).expanduser() if configured else Path(__file__).parent.parent / "loras"
    return path if path.is_absolute() else Path(__file__).parent.parent / path



@dataclass
class RTX4060Config:
    """RTX 4060 (8GB) 优化的训练超参数配置。"""
    model_name_or_path: str = ""
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 512
    lora_rank: int = 16
    lora_alpha: int = 32
    learning_rate: float = 3e-4
    num_train_epochs: float = 3.0
    fp16: bool = True
    load_in_4bit: bool = True
    use_gradient_checkpointing: bool = True
    lora_dropout: float = 0.1
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 0.5
    early_stopping_patience: int = 3
    packing: bool = True
    neftune_noise_alpha: float = 5.0
    use_dora: bool = False
    use_rslora: bool = False
    report_to: str = "tensorboard"


@dataclass
class RTX3090Config:
    """RTX 3090 (24GB) 优化的训练超参数配置。

    相比 RTX 4060 (8GB) 的改进：
    - batch_size: 1→4（显存充裕，更大batch训练更稳定）
    - gradient_accumulation: 8→4（等效batch=16，足够收敛）
    - max_seq_length: 512→2048（支持更长多轮对话）
    - lora_rank: 16→32（更高秩，更强的表达能力）
    - lora_alpha: 32→64（与rank等比缩放）
    - load_in_4bit: True→False（显存足够，用FP16精度更高）
    - learning_rate: 3e-4→2e-4（更大batch用更小学习率）
    - lora_dropout: 0.1→0.05（数据量足够，降低正则化）
    - 新增 bf16 支持（Ampere架构原生支持）
    """
    model_name_or_path: str = ""
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    lora_rank: int = 32
    lora_alpha: int = 64
    learning_rate: float = 2e-4
    num_train_epochs: float = 3.0
    fp16: bool = False
    bf16: bool = True
    load_in_4bit: bool = False
    use_gradient_checkpointing: bool = True
    lora_dropout: float = 0.05
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 0.5
    early_stopping_patience: int = 3
    packing: bool = True
    neftune_noise_alpha: float = 5.0
    use_dora: bool = False
    use_rslora: bool = False
    report_to: str = "tensorboard"


def _resolve_model_path(env_var: str, default_rel: str) -> str:
    p = os.getenv(env_var, default_rel)
    if not os.path.isabs(p):
        p = str(Path(__file__).parent / p)
    return p

RTX_4060_CONFIGS = {
    "qwen3-8b": RTX4060Config(
        model_name_or_path=_resolve_model_path("BASE_MODEL_PATH", "models/Qwen3-8B-Instruct"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_seq_length=512,
        lora_rank=16,
        lora_alpha=32,
        learning_rate=3e-4,
        num_train_epochs=3.0,
    ),
}

RTX_3090_CONFIGS = {
    "qwen3-8b-3090": RTX3090Config(
        model_name_or_path=_resolve_model_path("BASE_MODEL_PATH", "models/Qwen3-8B-Instruct"),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_seq_length=2048,
        lora_rank=32,
        lora_alpha=64,
        learning_rate=2e-4,
        num_train_epochs=3.0,
    ),
}

# 合并所有 GPU 配置，供前端选择
ALL_GPU_CONFIGS = {**RTX_4060_CONFIGS, **RTX_3090_CONFIGS}


class SimpleLoRATrainer:
    """LoRA训练器兼容层，负责训练任务的创建、调度和状态管理。

    内部桥接到train_lora.py的LoRATrainer执行实际训练，通过线程池实现异步训练。
    """

    def __init__(self, base_dir: Optional[Path] = None, db=None):
        """初始化训练器兼容层。

        Args:
            base_dir: 基础目录路径，默认为当前文件所在目录
            db: 数据库实例，用于训练任务持久化
        """
        self.base_dir = base_dir or Path(__file__).parent
        self.loras_dir = _lora_root()
        self.loras_dir.mkdir(parents=True, exist_ok=True)
        self.db = db
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self._cancel_events: Dict[str, threading.Event] = {}
        self._runner_tasks: Dict[str, asyncio.Task[None]] = {}
        # Progress callbacks execute in the training worker thread, while API
        # handlers read the same state on the event-loop thread.
        self._lock = threading.RLock()

        # 从数据库恢复未完成的任务到内存
        if self.db:
            self._restore_tasks_from_db()

    async def start_training(self, lora_name: str, dataset_path: Path, config: Dict[str, Any]) -> str:
        """启动异步LoRA训练任务。

        Args:
            lora_name: LoRA模型名称
            dataset_path: 训练数据集文件或目录路径
            config: 训练超参数字典

        Returns:
            str: 训练任务ID，用于后续状态查询
        """
        # 幂等性检查：是否已有同名lora_name的运行中任务
        if self.db:
            active_tasks = self.db.get_active_training_by_lora_name(lora_name)
            if active_tasks:
                raise HTTPException(
                    status_code=409,
                    detail=f"已有运行中的训练任务使用lora_name='{lora_name}'，请等待完成或取消后再试"
                )

        task_id = str(uuid.uuid4())  # 完整 UUID，避免 8 位截断碰撞

        task = {
            "task_id": task_id,
            "lora_name": lora_name,
            "dataset_path": str(dataset_path),
            "config": config,
            "status": "pending",
            "progress": 0,
            "current_step": 0,
            "total_steps": 0,
            "loss": None,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "started_at": None,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "finished_at": None,
            "error_message": None,
            "output_dir": str(self.loras_dir / lora_name),
        }

        with self._lock:
            self.tasks[task_id] = task
            self._cancel_events[task_id] = threading.Event()

        # 持久化到数据库
        if self.db:
            self.db.save_training_task(task_id, task)

        runner = asyncio.create_task(
            self._run_training(task_id, lora_name, dataset_path, config),
            name=f"lora-training-{task_id}",
        )
        with self._lock:
            self._runner_tasks[task_id] = runner
        runner.add_done_callback(
            lambda completed, tracked_id=task_id: self._discard_runner_task(
                tracked_id, completed
            )
        )

        logger.info(f"训练任务已创建: {task_id}, LoRA: {lora_name}")
        return task_id

    async def _run_training(self, task_id: str, lora_name: str, dataset_path: Path, config: Dict[str, Any]):
        cancel_event = self._cancel_events.get(task_id)

        with self._lock:
            if self.tasks[task_id].get("status") == "interrupted":
                return
            self.tasks[task_id]["status"] = "training"
            self.tasks[task_id]["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.tasks[task_id]["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        self._persist_task(task_id)

        try:
            from training.trainer import LoRATrainingConfig, LoRATrainer

            train_config = self._build_config(lora_name, dataset_path, config)
            errors = train_config.validate()
            if errors:
                raise ValueError(f"训练配置错误: {'; '.join(errors)}")

            # 检查是否在配置阶段已被取消
            if cancel_event and cancel_event.is_set():
                with self._lock:
                    if self.tasks[task_id]["status"] not in ("cancelled", "interrupted"):
                        self.tasks[task_id]["status"] = "cancelled"
                        self.tasks[task_id]["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                        self.tasks[task_id]["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                self._persist_task(task_id)
                logger.info(f"训练任务在配置阶段被取消: {task_id}")
                return

            config_path = Path(train_config.output_dir) / "training_config_auto.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            train_config.save(config_path)

            trainer = LoRATrainer(train_config)

            with self._lock:
                if cancel_event and cancel_event.is_set():
                    if self.tasks[task_id]["status"] not in ("cancelled", "interrupted"):
                        self.tasks[task_id]["status"] = "cancelled"
                        self.tasks[task_id]["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                        self.tasks[task_id]["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    self._persist_task(task_id)
                    logger.info(f"训练任务在启动前被取消: {task_id}")
                    return
                self.tasks[task_id]["progress"] = 10
                self.tasks[task_id]["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                self._persist_task(task_id)

            # 训练进度回调：由 trainer 在每个 step 结束时调用，实时更新任务状态（0-100）
            def _progress_callback(progress: int, current_step: int, total_steps: int):
                with self._lock:
                    if task_id in self.tasks:
                        self.tasks[task_id]["progress"] = progress
                        self.tasks[task_id]["current_step"] = current_step
                        self.tasks[task_id]["total_steps"] = total_steps
                        self.tasks[task_id]["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

            # 在线程中执行阻塞的训练操作，传入取消事件和进度回调供训练循环检查
            def _train_with_cancel_check():
                if cancel_event and hasattr(trainer, 'cancel_event'):
                    trainer.cancel_event = cancel_event
                trainer.progress_fn = _progress_callback
                return trainer.train()

            output_path = await asyncio.to_thread(_train_with_cancel_check)

            # 检查取消事件
            if cancel_event and cancel_event.is_set():
                with self._lock:
                    if self.tasks[task_id]["status"] not in ("cancelled", "interrupted"):
                        self.tasks[task_id]["status"] = "cancelled"
                        self.tasks[task_id]["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                        self.tasks[task_id]["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                self._persist_task(task_id)
                logger.info(f"训练任务完成后发现已被取消: {task_id}")
                return

            with self._lock:
                if self.tasks[task_id]["status"] in ("cancelled", "interrupted"):
                    self._persist_task(task_id)
                    logger.info(f"训练任务完成后发现已被取消: {task_id}")
                    return
                self.tasks[task_id]["status"] = "completed"
                self.tasks[task_id]["progress"] = 100
                self.tasks[task_id]["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                self.tasks[task_id]["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                self.tasks[task_id]["output_dir"] = str(output_path)

            self._persist_task(task_id)
            logger.info(f"训练任务完成: {task_id}")

        except Exception as e:
            logger.error(f"训练任务失败: {task_id}, 错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
            with self._lock:
                # 如果已被取消，不覆盖cancelled状态
                if self.tasks[task_id]["status"] in ("cancelled", "interrupted"):
                    self._persist_task(task_id)
                    return
                self.tasks[task_id]["status"] = "failed"
                self.tasks[task_id]["error_message"] = str(e)
                self.tasks[task_id]["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                self.tasks[task_id]["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

            self._persist_task(task_id)

    def _discard_runner_task(
        self,
        task_id: str,
        completed: asyncio.Task[None],
    ) -> None:
        with self._lock:
            if self._runner_tasks.get(task_id) is completed:
                self._runner_tasks.pop(task_id, None)
        if completed.cancelled():
            return
        try:
            error = completed.exception()
        except asyncio.CancelledError:
            return
        if error is not None:
            logger.error("训练任务协程异常 task_id=%s: %s", task_id, error)

    async def shutdown(self, timeout: float | None = None) -> None:
        """Signal active trainers, retain state, and join owned runner tasks."""
        wait_timeout = (
            float(os.getenv("TRAINING_SHUTDOWN_TIMEOUT", "30"))
            if timeout is None
            else max(float(timeout), 0.0)
        )
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            runners = list(self._runner_tasks.values())
            interrupted_ids = []
            for task_id, task in self.tasks.items():
                if task.get("status") not in ("pending", "training"):
                    continue
                interrupted_ids.append(task_id)
                event = self._cancel_events.get(task_id)
                if event is not None:
                    event.set()
                task["status"] = "interrupted"
                task["error_message"] = "服务关闭，训练任务已中断"
                task["updated_at"] = now
                task["finished_at"] = now
            for task_id in interrupted_ids:
                self._persist_task(task_id)

        if not runners:
            return

        _, pending = await asyncio.wait(runners, timeout=wait_timeout)
        if pending:
            logger.warning(
                "训练任务在 %.1f 秒内未结束，取消 %d 个等待协程",
                wait_timeout,
                len(pending),
            )
            for runner in pending:
                runner.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    def _build_config(self, lora_name: str, dataset_path: Path, config: Dict[str, Any]) -> "LoRATrainingConfig":
        from training.trainer import LoRATrainingConfig

        dataset_file = self._find_dataset_file(dataset_path)

        # 已知字段从 config dict 提取（前端 TrainingParamsEditor 发送 + 预设兜底）
        kwargs: Dict[str, Any] = {
            "base_model_path": config.get("model_name_or_path", _resolve_model_path("BASE_MODEL_PATH", "models/Qwen3-8B-Instruct")),
            "train_data_path": str(dataset_file),
            "output_dir": str(self.loras_dir / lora_name),
            "lora_r": config.get("lora_rank", 16),
            "lora_alpha": config.get("lora_alpha", 32),
            "lora_dropout": config.get("lora_dropout", 0.1),
            "learning_rate": config.get("learning_rate", 3e-4),
            "num_train_epochs": int(config.get("num_train_epochs", 3)),
            "per_device_train_batch_size": config.get("per_device_train_batch_size", 2),
            "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 4),
            "max_seq_length": config.get("max_seq_length", 512),
            "fp16": config.get("fp16", True),
            "bf16": config.get("bf16", False),
            "gradient_checkpointing": config.get("use_gradient_checkpointing", True),
            # 以下字段前端可能发送，后端读取时兜底默认值
            "warmup_ratio": float(config.get("warmup_ratio", 0.05)),
            "weight_decay": float(config.get("weight_decay", 0.01)),
            "max_grad_norm": float(config.get("max_grad_norm", 0.5)),
            "early_stopping_patience": int(config.get("early_stopping_patience", 3)),
        }

        # 透传前端可能发送的额外字段（LoRATrainingConfig 可能支持）
        extra_keys = {
            "target_modules", "lr_scheduler_type", "truncation_direction",
            "chat_template", "use_8bit_adam", "use_deepspeed",
            "lora_target_modules", "load_in_4bit", "load_in_8bit",
            "use_cache", "preprocessing_num_workers", "dataloader_num_workers",
            "logging_steps", "save_steps", "eval_steps", "save_total_limit",
            "packing", "neftune_noise_alpha", "use_dora", "use_rslora", "report_to", "logging_dir",
        }
        for key in extra_keys:
            if key in config:
                kwargs[key] = config[key]

        # 透传 config 中任何未被显式提取的字段（未来扩展兼容）
        known = {
            "model_name_or_path", "lora_rank", "lora_alpha", "lora_dropout",
            "learning_rate", "num_train_epochs", "per_device_train_batch_size",
            "gradient_accumulation_steps", "max_seq_length", "fp16", "bf16",
            "use_gradient_checkpointing", "warmup_ratio", "weight_decay",
            "max_grad_norm", "early_stopping_patience",
            # 前端冗余字段，后端已由 fp16/bf16 表达，无需透传
            "mixed_precision",
        } | extra_keys
        for key, value in config.items():
            if key not in known and key not in kwargs:
                kwargs[key] = value

        # 处理 target_modules：前端可能传字符串 "all-linear" 或 "q_proj,v_proj"
        if "target_modules" in kwargs:
            tm = kwargs["target_modules"]
            if isinstance(tm, str):
                if tm.lower() == "all-linear":
                    kwargs["target_modules"] = None  # PEFT 的 inference_mode=False 时会自动推导
                else:
                    kwargs["target_modules"] = [m.strip() for m in tm.split(",") if m.strip()]

        # 处理 use_8bit_adam：切换优化器
        if kwargs.get("use_8bit_adam"):
            kwargs["optim"] = "adamw_8bit"

        # 处理 use_deepspeed：当前仅记录，未配置 ds_config 时保持普通训练
        if kwargs.get("use_deepspeed"):
            logger.warning("use_deepspeed=true 已设置，但当前未提供 DeepSpeed 配置文件，将按普通训练执行")

        train_config = LoRATrainingConfig(**kwargs)

        return train_config

    def _find_dataset_file(self, dataset_path: Path) -> Path:
        if dataset_path.is_file():
            return dataset_path

        for name in ["train.json", "data.json", "dataset.json"]:
            candidate = dataset_path / name
            if candidate.exists():
                return candidate

        json_files = list(dataset_path.glob("*.json"))
        if json_files:
            return json_files[0]

        raise FileNotFoundError(f"数据集目录中没有找到JSON文件: {dataset_path}")

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """查询单个训练任务的状态。

        Args:
            task_id: 任务ID

        Returns:
            任务状态字典，包含status、progress等字段，不存在则返回None
        """
        with self._lock:
            task = self.tasks.get(task_id)
            if task:
                return dict(task)
        # 内存中没有，尝试从数据库读取
        if self.db:
            db_task = self.db.get_training_task(task_id)
            if db_task:
                return db_task
        return None

    async def get_all_tasks(self) -> List[Dict[str, Any]]:
        with self._lock:
            self._cleanup_old_tasks()

        # 优先从数据库读取（重启后也能恢复）
        if self.db:
            try:
                return self.db.get_all_training_tasks()
            except Exception as e:
                logger.warning(f"从数据库读取训练任务失败，回退到内存: {e}")

        with self._lock:
            return [dict(task) for task in self.tasks.values()]

    def _cleanup_old_tasks(self, max_completed: int = 50):
        """清理过多的已完成/失败/取消任务，防止tasks字典无限增长。

        保留最新的 max_completed 个已完成任务，删除更早的。
        注意：调用方需持有 self._lock。
        """
        completed = [
            tid for tid, t in self.tasks.items()
            if t.get("status") in ("completed", "failed", "cancelled", "interrupted")
        ]
        if len(completed) > max_completed:
            # 按创建时间排序，删除最旧的
            completed.sort(key=lambda tid: self.tasks[tid].get("created_at", ""))
            for tid in completed[max_completed:]:
                del self.tasks[tid]
                self._cancel_events.pop(tid, None)

    async def cancel_task(self, task_id: str) -> bool:
        with self._lock:
            task = self.tasks.get(task_id)
            if task and task["status"] in ("pending", "training"):
                task["status"] = "cancelled"
                task["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                task["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                # 设置取消事件，通知训练线程停止
                cancel_event = self._cancel_events.get(task_id)
                if cancel_event:
                    cancel_event.set()
                self._persist_task(task_id)
                return True
        return False

    def _persist_task(self, task_id: str):
        """将内存中的任务状态同步到数据库。"""
        if not self.db:
            return
        task = self.tasks.get(task_id)
        if task:
            try:
                self.db.save_training_task(task_id, task)
            except Exception as e:
                logger.warning(f"持久化训练任务 {task_id} 失败: {e}")

    def _restore_tasks_from_db(self):
        """从数据库恢复任务到内存（服务重启后调用）。"""
        if not self.db:
            return
        try:
            db_tasks = self.db.get_all_training_tasks()
            for db_task in db_tasks:
                task_id = db_task["task_id"]
                status = db_task.get("status", "pending")
                # 将重启前未完成的任务标记为中断状态
                if status in ("pending", "training"):
                    db_task["status"] = "interrupted"
                    db_task["error_message"] = "服务重启，任务被中断"
                    db_task["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    self.db.save_training_task(task_id, db_task)
                self.tasks[task_id] = db_task
            if db_tasks:
                logger.info(f"从数据库恢复了 {len(db_tasks)} 个训练任务到内存")
        except Exception as e:
            logger.warning(f"从数据库恢复训练任务失败: {e}")


_simple_lora_trainer: Optional[SimpleLoRATrainer] = None


def get_simple_lora_trainer(db=None) -> SimpleLoRATrainer:
    global _simple_lora_trainer
    if _simple_lora_trainer is None:
        _simple_lora_trainer = SimpleLoRATrainer(db=db)
    return _simple_lora_trainer


async def shutdown_simple_lora_trainer() -> None:
    """Stop the process-wide trainer and clear it for the next app lifecycle."""
    global _simple_lora_trainer
    trainer = _simple_lora_trainer
    if trainer is None:
        return
    await trainer.shutdown()
    if _simple_lora_trainer is trainer:
        _simple_lora_trainer = None
