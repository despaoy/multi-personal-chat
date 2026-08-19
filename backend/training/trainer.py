"""
胡桃风格LoRA训练核心脚本
基于 Qwen/Qwen3-8B（本地兼容别名 Qwen3-8B-Instruct）的 LoRA 微调，包含完整训练配置、GPU 温度保护和早停机制。
"""
import os as _os
import logging
import inspect
import json
import math
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List

_BACKEND_DIR = Path(__file__).parent.parent.parent  # project root


def _resolve_path(p: str) -> str:
    if _os.path.isabs(p):
        return p
    return str(_BACKEND_DIR / p)

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

# 兼容性补丁：gptqmodel >= 7.x 将 AwqGEMMQuantLinear 重命名为 AwqGEMMLinear，
# 而 peft 仍在尝试导入旧名称。此处建立别名，避免 BF16 模型应用 LoRA 时误触发导入错误。
try:
    from gptqmodel.nn_modules.qlinear.gemm_awq import AwqGEMMLinear
    import gptqmodel.nn_modules.qlinear.gemm_awq as _gemm_awq_module
    if not hasattr(_gemm_awq_module, "AwqGEMMQuantLinear"):
        _gemm_awq_module.AwqGEMMQuantLinear = AwqGEMMLinear
except Exception:
    pass

from datasets import Dataset, load_dataset
from trl import SFTTrainer, SFTConfig
from training.chat_dataset import (
    normalize_chat_record,
    pack_tokenized_records,
    tokenize_assistant_turns,
)
from training.evaluator import write_training_evaluation_report

try:
    from transformers import EarlyStoppingCallback
    HAS_EARLY_STOPPING = True
except ImportError:
    HAS_EARLY_STOPPING = False

import time
from transformers import TrainerCallback

HAS_PYNVML = False
try:
    import pynvml
    pynvml.nvmlInit()
    HAS_PYNVML = True
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GpuTemperatureCallback(TrainerCallback):
    """GPU温度监控回调，训练过程中定期检查GPU温度，超过阈值时触发散热暂停，
    防止笔记本GPU过热导致系统蓝屏(BSOD)。"""

    def __init__(self, max_temp: float = 82.0, cooldown_temp: float = 72.0,
                 cooldown_seconds: int = 30, check_interval_steps: int = 20):
        """初始化GPU温度监控回调。

        Args:
            max_temp: 触发散热的最高温度阈值，默认82°C
            cooldown_temp: 恢复训练的目标冷却温度，默认72°C
            cooldown_seconds: 每次散热暂停秒数，默认30
            check_interval_steps: 每隔多少步检查一次温度，默认20
        """
        self.max_temp = max_temp
        self.cooldown_temp = cooldown_temp
        self.cooldown_seconds = cooldown_seconds
        self.check_interval_steps = check_interval_steps
        self._max_observed = 0.0
        self._cooldown_count = 0
        self._last_check_step = -999

    def _get_gpu_temp(self) -> float:
        if HAS_PYNVML:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                return float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
            except Exception as e:
                # H3 fix: 此前静默吞噬，GPU 温度监控失败时无法排障
                logger.warning("pynvml GPU 温度读取失败: %s", e)
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip().split('\n')[0].strip())
        except Exception as e:
            # H3 fix: nvidia-smi 兜底也失败时记录日志
            logger.warning("nvidia-smi GPU 温度读取失败: %s", e)
        return -1.0

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step - self._last_check_step < self.check_interval_steps:
            return control

        self._last_check_step = step
        temp = self._get_gpu_temp()
        if temp > 0:
            self._max_observed = max(self._max_observed, temp)

        if temp > 0 and temp > self.max_temp:
            self._cooldown_count += 1
            logger.warning(f"GPU温度 ({temp:.0f}°C) 超过限制 ({self.max_temp:.0f}°C)，"
                          f"暂停 {self.cooldown_seconds} 秒散热... (第{self._cooldown_count}次)")

            elapsed = 0
            while elapsed < self.cooldown_seconds:
                time.sleep(5)
                elapsed += 5
                current_temp = self._get_gpu_temp()
                if current_temp > 0:
                    wait_remaining = self.cooldown_seconds - elapsed
                    logger.info(f"散热中... 当前温度: {current_temp:.0f}°C, 还需等待: {wait_remaining}秒")
                if current_temp > 0 and current_temp <= self.cooldown_temp and elapsed >= 15:
                    logger.info(f"GPU已冷却到 {current_temp:.0f}°C，提前恢复训练")
                    break

            if self._cooldown_count >= 10:
                logger.error("散热次数超过10次，可能是散热系统问题。建议：1) 清理风扇灰尘 2) 使用散热底座 3) 降低室温")

        return control

    def on_train_end(self, args, state, control, **kwargs):
        logger.info(f"训练完成。GPU最高温度记录: {self._max_observed:.0f}°C, 散热暂停次数: {self._cooldown_count}")


class ProgressCallback(TrainerCallback):
    """训练进度报告回调，将训练进度实时更新到任务管理器（基于 global_step/max_steps 计算 0-100 百分比）。"""

    def __init__(self, progress_fn=None):
        self.progress_fn = progress_fn

    def on_step_end(self, args, state, control, **kwargs):
        if self.progress_fn and state.max_steps > 0:
            progress = min(100, int(state.global_step / state.max_steps * 100))
            self.progress_fn(progress, state.global_step, state.max_steps)
        return control


@dataclass
class LoRATrainingConfig:
    """LoRA训练配置数据类，包含模型路径、LoRA超参数、训练参数和优化器设置。

    支持从/到JSON文件序列化，提供参数校验功能。
    """
    base_model_path: str = _resolve_path(_os.getenv("BASE_MODEL_PATH", "models/Qwen3-8B-Instruct"))
    train_data_path: str = _resolve_path("backend/hutao_dialogues.json")
    eval_data_path: Optional[str] = None
    output_dir: str = _resolve_path("backend/loras/hutao_lora_7b")

    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    lora_bias: str = "none"
    use_dora: bool = False
    use_rslora: bool = False
    neftune_noise_alpha: float = 5.0
    packing: bool = True

    learning_rate: float = 2e-4
    num_train_epochs: int = 12
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 1024

    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    warmup_steps: int = 0
    max_grad_norm: float = 0.5

    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001

    eval_strategy: str = "steps"
    eval_steps: int = 50
    save_strategy: str = "steps"
    save_steps: int = 50
    save_total_limit: int = 1
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    fp16: bool = True
    bf16: bool = False
    optim: str = "paged_adamw_32bit"
    gradient_checkpointing: bool = True
    use_8bit_adam: bool = False
    use_deepspeed: bool = False
    load_in_4bit: bool = False
    load_in_8bit: bool = False

    logging_steps: int = 10
    report_to: str = "tensorboard"
    logging_dir: Optional[str] = None

    system_prompt: str = "你是胡桃，保持你的风格"
    system_prompt_policy: str = "preserve"
    train_test_split: float = 0.9
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    truncation_direction: str = "left"
    chat_template: bool = True

    def validate(self):
        """校验配置参数合法性。

        Returns:
            List[str]: 错误信息列表，无错误时为空列表
        """
        errors = []
        if not Path(self.base_model_path).exists():
            errors.append(f"基础模型路径不存在: {self.base_model_path}")
        if not Path(self.train_data_path).exists():
            errors.append(f"训练数据路径不存在: {self.train_data_path}")
        if self.eval_data_path and not Path(self.eval_data_path).exists():
            errors.append(f"Evaluation data path does not exist: {self.eval_data_path}")
        if self.lora_r <= 0:
            errors.append(f"lora_r必须大于0: {self.lora_r}")
        if self.learning_rate <= 0:
            errors.append(f"learning_rate必须大于0: {self.learning_rate}")
        if self.num_train_epochs <= 0:
            errors.append(f"num_train_epochs必须大于0: {self.num_train_epochs}")
        if self.fp16 and self.bf16:
            errors.append("不能同时启用fp16和bf16")
        if self.early_stopping_patience < 0:
            errors.append(f"early_stopping_patience必须大于0: {self.early_stopping_patience}")
        if self.neftune_noise_alpha < 0:
            errors.append(f"neftune_noise_alpha cannot be negative: {self.neftune_noise_alpha}")
        if self.packing and self.max_seq_length < 128:
            errors.append(f"max_seq_length must be at least 128 when packing is enabled: {self.max_seq_length}")
        if self.system_prompt_policy not in {"preserve", "replace", "require_match"}:
            errors.append("system_prompt_policy must be preserve, replace, or require_match")
        if self.system_prompt_policy in {"replace", "require_match"} and not self.system_prompt.strip():
            errors.append(f"system_prompt is required when policy is {self.system_prompt_policy}")
        return errors

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LoRATrainingConfig":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        for path_key in ("base_model_path", "train_data_path", "eval_data_path", "output_dir"):
            if filtered.get(path_key) and not _os.path.isabs(filtered[path_key]):
                filtered[path_key] = _resolve_path(filtered[path_key])
        return cls(**filtered)

    @classmethod
    def load(cls, path: Path) -> "LoRATrainingConfig":
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_dict(json.load(f))


class LoRATrainer:
    """LoRA训练器，封装完整的微调流程：加载模型和分词器 -> 预处理数据 -> 配置LoRA -> 训练 -> 保存。

    支持早停、GPU温度保护、检查点恢复、最佳模型自动选择。
    """

    def __init__(self, config: Optional[LoRATrainingConfig] = None, progress_fn=None):
        """初始化训练器。

        Args:
            config: 训练配置，默认使用LoRATrainingConfig()
            progress_fn: 训练进度回调函数，签名 (progress:int, current_step:int, total_steps:int) -> None
        """
        self.config = config or LoRATrainingConfig()
        self._tokenizer = None
        self._model = None
        self.progress_fn = progress_fn

    def print_gpu_memory(self, stage: str = ""):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"[{stage}] GPU显存 - 已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB")

    def cleanup_gpu(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            logger.info("GPU显存已清理")

    def _load_tokenizer(self) -> AutoTokenizer:
        logger.info("加载Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.config.base_model_path),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self._tokenizer = tokenizer
        return tokenizer

    def _load_and_preprocess_data(self, tokenizer: AutoTokenizer) -> Dict[str, Dataset]:
        logger.info("=" * 60)
        logger.info("步骤1: 加载和预处理训练数据")
        logger.info("=" * 60)

        dataset = load_dataset("json", data_files=str(self.config.train_data_path))["train"]
        logger.info(f"训练数据加载完成，样本数: {len(dataset)}")

        def format_and_tokenize(record):
            messages = normalize_chat_record(
                record,
                default_system_prompt=self.config.system_prompt,
                system_prompt_policy=self.config.system_prompt_policy,
            )
            metadata = record.get("metadata")
            assistant_supervision = (
                metadata.get("assistant_supervision", "all")
                if isinstance(metadata, dict)
                else "all"
            )
            return tokenize_assistant_turns(
                tokenizer,
                messages,
                max_length=self.config.max_seq_length,
                truncation_direction=self.config.truncation_direction,
                use_chat_template=self.config.chat_template,
                assistant_supervision=assistant_supervision,
            )

        tokenized_dataset = dataset.map(
            format_and_tokenize,
            batched=False,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset",
        )

        if self.config.eval_data_path:
            eval_raw = load_dataset(
                "json", data_files=str(self.config.eval_data_path)
            )["train"]
            eval_dataset = eval_raw.map(
                format_and_tokenize,
                batched=False,
                remove_columns=eval_raw.column_names,
                desc="Tokenizing fixed evaluation dataset",
            )
            train_dataset = tokenized_dataset
        else:
            split = tokenized_dataset.train_test_split(
                test_size=1 - self.config.train_test_split,
                seed=self.config.seed,
            )
            train_dataset = split["train"]
            eval_dataset = split["test"]

        if self.config.packing:
            unpacked_count = len(train_dataset)
            train_dataset = Dataset.from_list(
                pack_tokenized_records(
                    train_dataset,
                    max_length=self.config.max_seq_length,
                )
            )
            logger.info(
                "Packed %s training samples into %s label-preserving chunks",
                unpacked_count,
                len(train_dataset),
            )

        logger.info(f"训练集: {len(train_dataset)} 样本")
        logger.info(f"验证集: {len(eval_dataset)} 样本")

        return {"train": train_dataset, "eval": eval_dataset}

    def _detect_quantization_type(self) -> Optional[str]:
        """读取模型 config.json 中的 quantization_config，返回量化方式。"""
        config_path = Path(self.config.base_model_path) / "config.json"
        if not config_path.exists():
            return None
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                model_config = json.load(f)
            quant_config = model_config.get("quantization_config")
            if isinstance(quant_config, dict):
                return quant_config.get("quant_method", "unknown")
        except Exception as e:
            logger.warning(f"读取模型 config.json 失败: {e}")
        return None

    def _load_model(self) -> AutoModelForCausalLM:
        logger.info("加载基础模型...")

        torch_dtype = torch.bfloat16 if self.config.bf16 else torch.float16
        quant_method = self._detect_quantization_type()
        is_quantized = quant_method in ("awq", "gptq") or self.config.load_in_4bit or self.config.load_in_8bit
        load_kwargs: Dict[str, Any] = {
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }

        if quant_method == "awq":
            logger.warning(
                "检测到 AWQ 预量化模型。AWQ/GPTQ 模型主要用于推理，对其训练 LoRA "
                "可能遇到兼容性或精度问题。建议训练时使用 FP16/BF16 原始模型。"
            )
            logger.info("检测到 AWQ 预量化模型，使用 AwqConfig 加载")
            try:
                from transformers import AwqConfig
            except ImportError as e:
                raise ImportError(
                    "加载 AWQ 量化模型需要 autoawq。请在服务器执行：pip install autoawq"
                ) from e
            config_path = Path(self.config.base_model_path) / "config.json"
            awq_kwargs: Dict[str, Any] = {
                "bits": 4,
                "group_size": 128,
                "zero_point": True,
            }
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    model_config = json.load(f)
                qcfg = model_config.get("quantization_config", {})
                for key in ("bits", "group_size", "zero_point", "version", "desc_act"):
                    if key in qcfg:
                        awq_kwargs[key] = qcfg[key]
            except Exception:
                pass
            # 明确指定 autoawq 后端，避免 transformers 自动选择 gptqmodel 导致兼容性错误
            awq_kwargs["backend"] = "autoawq"
            # 新版 autoawq 不再接受 version='GEMM'，遇到 TypeError 时剔除
            try:
                load_kwargs["quantization_config"] = AwqConfig(**awq_kwargs)
            except TypeError as e:
                awq_kwargs.pop("version", None)
                load_kwargs["quantization_config"] = AwqConfig(**awq_kwargs)
            # AWQ 权重已是 4bit，BitsAndBytes 与之冲突，关闭
            load_kwargs.pop("torch_dtype", None)
        elif quant_method == "gptq":
            logger.warning(
                "检测到 GPTQ 预量化模型。AWQ/GPTQ 模型主要用于推理，对其训练 LoRA "
                "可能遇到兼容性或精度问题。建议训练时使用 FP16/BF16 原始模型。"
            )
            logger.info("检测到 GPTQ 预量化模型，使用 GPTQConfig 加载")
            try:
                from transformers import GPTQConfig
            except ImportError as e:
                raise ImportError(
                    "加载 GPTQ 量化模型需要 gptqmodel 或 auto-gptq。请在服务器执行：pip install gptqmodel"
                ) from e
            # 从 config.json 读取 bits/group_size，避免硬编码
            bits = 4
            group_size = 128
            config_path = Path(self.config.base_model_path) / "config.json"
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    model_config = json.load(f)
                qcfg = model_config.get("quantization_config", {})
                bits = qcfg.get("bits", bits)
                group_size = qcfg.get("group_size", group_size)
            except Exception:
                pass
            load_kwargs["quantization_config"] = GPTQConfig(
                bits=bits,
                group_size=group_size,
                disable_exllama=False,
            )
            load_kwargs.pop("torch_dtype", None)
        elif self.config.load_in_4bit:
            logger.info("启用 4-bit 量化加载 (NF4)")
            load_kwargs["torch_dtype"] = torch_dtype
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.config.load_in_8bit:
            logger.info("启用 8-bit 量化加载")
            load_kwargs["torch_dtype"] = torch_dtype
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            # transformers 4.x forwards unknown kwargs to the model constructor.
            # Use torch_dtype here; dtype is only accepted by newer releases.
            load_kwargs["torch_dtype"] = torch_dtype

        model = AutoModelForCausalLM.from_pretrained(
            str(self.config.base_model_path),
            **load_kwargs,
        )
        if is_quantized:
            model = prepare_model_for_kbit_training(model)
        self.print_gpu_memory("基础模型加载后")
        self._model = model
        return model

    def _create_lora_config(self) -> LoraConfig:
        target_modules = self.config.target_modules
        if target_modules is None:
            target_modules = "all-linear"
        return LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.config.lora_dropout,
            bias=self.config.lora_bias,
            task_type=TaskType.CAUSAL_LM,
            use_dora=self.config.use_dora,
            use_rslora=self.config.use_rslora,
        )

    def _configure_lora_model(self, model):
        model = get_peft_model(model, self._create_lora_config())
        if self.config.gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            if getattr(model, "config", None) is not None:
                model.config.use_cache = False
        return model
    def train(self) -> Path:
        """执行完整的LoRA训练流程。

        流程：校验配置 -> 清理GPU -> 加载分词器和模型 -> 预处理数据 -> 
        配置LoRA -> 创建SFTTrainer（含早停和温度回调） -> 训练 -> 评估 -> 保存模型。

        Returns:
            Path: 最终模型保存路径（output_dir/final）

        Raises:
            ValueError: 训练配置有错误时抛出
        """
        logger.info("=" * 60)
        logger.info("优化版胡桃LoRA训练开始")
        logger.info("=" * 60)

        config_errors = self.config.validate()
        if config_errors:
            for err in config_errors:
                logger.error(f"配置错误: {err}")
            raise ValueError(f"训练配置有 {len(config_errors)} 个错误")

        try:
            self.cleanup_gpu()

            tokenizer = self._load_tokenizer()
            datasets = self._load_and_preprocess_data(tokenizer)
            model = self._load_model()

            logger.info("配置LoRA...")
            model = self._configure_lora_model(model)
            trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            model.print_trainable_parameters()
            self.print_gpu_memory("LoRA配置后")

            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            if self.config.report_to == "tensorboard" and not self.config.logging_dir:
                self.config.logging_dir = str(output_dir / "runs")

            config_save_path = output_dir / "training_config.json"
            self.config.save(config_save_path)
            logger.info(f"训练配置已保存到: {config_save_path}")

            callbacks = []
            if self.config.early_stopping_patience > 0 and HAS_EARLY_STOPPING:
                early_stopping = EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=self.config.early_stopping_threshold,
                )
                callbacks.append(early_stopping)
                logger.info(f"已启用早停机制: patience={self.config.early_stopping_patience}, "
                          f"threshold={self.config.early_stopping_threshold}")

            gpu_temp_callback = GpuTemperatureCallback(
                max_temp=82.0,
                cooldown_temp=72.0,
                cooldown_seconds=30,
                check_interval_steps=20,
            )
            callbacks.append(gpu_temp_callback)
            logger.info("已启用GPU温度保护: max=82°C, cooldown=72°C, interval=20步")

            if self.progress_fn:
                callbacks.append(ProgressCallback(self.progress_fn))
                logger.info("已启用训练进度报告回调")

            # trl 1.8+ 将 max_seq_length 改名为 max_length，做版本兼容
            sft_config_kwargs = dict(
                output_dir=str(output_dir),
                num_train_epochs=self.config.num_train_epochs,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                per_device_eval_batch_size=self.config.per_device_eval_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                lr_scheduler_type=self.config.lr_scheduler_type,
                warmup_ratio=self.config.warmup_ratio,
                warmup_steps=self.config.warmup_steps,
                max_grad_norm=self.config.max_grad_norm,
                weight_decay=self.config.weight_decay,
                adam_beta1=self.config.adam_beta1,
                adam_beta2=self.config.adam_beta2,
                adam_epsilon=self.config.adam_epsilon,
                logging_steps=self.config.logging_steps,
                logging_dir=self.config.logging_dir,
                eval_strategy=self.config.eval_strategy,
                eval_steps=self.config.eval_steps,
                save_strategy=self.config.save_strategy,
                save_steps=self.config.save_steps,
                save_total_limit=self.config.save_total_limit,
                load_best_model_at_end=self.config.load_best_model_at_end,
                metric_for_best_model=self.config.metric_for_best_model,
                greater_is_better=self.config.greater_is_better,
                fp16=self.config.fp16,
                bf16=self.config.bf16,
                optim=self.config.optim,
                report_to=self.config.report_to,
                gradient_checkpointing=self.config.gradient_checkpointing,
                seed=self.config.seed,
                # Packing is performed above on pre-tokenized records so the
                # response-only labels survive unchanged across sample joins.
                packing=False,
                neftune_noise_alpha=self.config.neftune_noise_alpha or None,
            )
            # 优先用 max_length（trl>=1.8），失败回退 max_seq_length（trl<1.8）
            sft_parameters = inspect.signature(SFTConfig).parameters
            length_key = "max_length" if "max_length" in sft_parameters else "max_seq_length"
            sft_config_kwargs[length_key] = self.config.max_seq_length
            training_args = SFTConfig(**sft_config_kwargs)

            data_collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                model=model,
                padding=True,
                return_tensors="pt",
            )

            trainer_kwargs = dict(
                model=model,
                args=training_args,
                train_dataset=datasets["train"],
                eval_dataset=datasets["eval"],
                callbacks=callbacks,
                processing_class=tokenizer,
            )
            trainer_kwargs["data_collator"] = data_collator
            trainer = SFTTrainer(**trainer_kwargs)

            logger.info("=" * 60)
            logger.info("开始训练")
            logger.info("=" * 60)
            self.print_gpu_memory("训练前")

            checkpoint = self.config.resume_from_checkpoint
            if checkpoint and Path(checkpoint).exists():
                logger.info(f"从检查点恢复训练: {checkpoint}")
                trainer.train(resume_from_checkpoint=checkpoint)
            else:
                trainer.train()

            final_output = output_dir / "final"
            trainer.model.save_pretrained(str(final_output))
            tokenizer.save_pretrained(str(final_output))

            eval_results = trainer.evaluate()
            logger.info(f"最终评估结果: {eval_results}")

            results_path = output_dir / "training_results.json"
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "eval_results": {k: float(v) if isinstance(v, (int, float)) else str(v)
                                     for k, v in eval_results.items()},
                    "trainable_params": trainable_params,
                    "config": self.config.to_dict(),
                    "best_model_checkpoint": str(trainer.state.best_model_checkpoint) if trainer.state.best_model_checkpoint else None,
                    "best_metric": float(trainer.state.best_metric) if trainer.state.best_metric else None,
                }, f, indent=2, ensure_ascii=False)
            evaluation_path = write_training_evaluation_report(
                output_dir=output_dir,
                eval_results=eval_results,
                log_history=trainer.state.log_history,
                config=self.config.to_dict(),
            )
            logger.info(f"Training evaluation report saved to: {evaluation_path}")

            logger.info(f"LoRA模型已保存到: {final_output}")
            logger.info(f"训练结果已保存到: {results_path}")
            self.print_gpu_memory("训练完成")

            logger.info("=" * 60)
            logger.info("训练完成！")
            logger.info("=" * 60)

            return final_output

        except Exception as e:
            logger.error(f"训练失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        finally:
            self.cleanup_gpu()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="LoRA训练脚本")
    parser.add_argument("--config", type=str, default=None, help="训练配置JSON文件路径")
    parser.add_argument("--base-model", type=str, default=None, help="基础模型路径")
    parser.add_argument("--train-data", type=str, default=None, help="训练数据路径")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--lora-r", type=int, default=None, help="LoRA rank")
    parser.add_argument("--resume", type=str, default=None, help="从检查点恢复训练")
    args = parser.parse_args()

    if args.config:
        config = LoRATrainingConfig.load(Path(args.config))
        logger.info(f"从配置文件加载: {args.config}")
    else:
        config = LoRATrainingConfig()

    if args.base_model:
        config.base_model_path = _resolve_path(args.base_model)
    if args.train_data:
        config.train_data_path = _resolve_path(args.train_data)
    if args.output_dir:
        config.output_dir = _resolve_path(args.output_dir)
    if args.epochs:
        config.num_train_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.lora_r:
        config.lora_r = args.lora_r
    if args.resume:
        config.resume_from_checkpoint = args.resume

    trainer = LoRATrainer(config)
    output_path = trainer.train()
    logger.info(f"\n训练成功完成！模型保存在: {output_path}")


if __name__ == "__main__":
    main()
