"""偏好对齐训练管线 - DPO / ORPO 训练。

遵循路线图 guardrail：
- 固定 seed=42 保证可复现
- 复用 trainer.py 的 GpuTemperatureCallback
- 不声称 RLHF（未训练奖励模型，未做策略优化），仅 DPO/ORPO
- 支持 --mock 模式用于 CPU 验证
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))


@dataclass
class PreferenceTrainingConfig:
    """偏好训练配置。"""
    method: str = "dpo"  # dpo | orpo
    beta: float = 0.1
    learning_rate: float = 5e-6
    num_train_epochs: int = 1
    seed: int = 42
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_length: int = 512
    max_prompt_length: int = 256
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    base_model_path: str = ""
    adapter_path: str = ""  # SFT adapter 作为起点
    output_dir: str = "loras/preference_dpo"
    save_total_limit: int = 2
    load_in_4bit: bool = True
    gradient_checkpointing: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PreferenceTrainingConfig":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid}
        return cls(**filtered)

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())


@dataclass
class PreferenceTrainResult:
    """偏好训练结果。"""
    method: str
    output_dir: str
    train_loss: float = 0.0
    eval_accuracy: Optional[float] = None
    train_steps: int = 0
    duration_s: float = 0.0
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    metric_note: str = ""
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PreferenceTrainer:
    """DPO / ORPO 训练器。"""

    def __init__(self, config: Optional[PreferenceTrainingConfig] = None):
        self.config = config or PreferenceTrainingConfig()

    def train(self, pairs: List[Dict[str, Any]]) -> PreferenceTrainResult:
        """执行 DPO 或 ORPO 训练。

        Args:
            pairs: 偏好对列表，每项含 prompt/chosen/rejected

        Returns:
            训练结果
        """
        import time
        result = PreferenceTrainResult(
            method=self.config.method,
            output_dir=self.config.output_dir,
            config_snapshot=self.config.to_dict(),
            timestamp=datetime.now().isoformat(),
        )
        start = time.monotonic()

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import (PeftModel, LoraConfig, get_peft_model,
                              prepare_model_for_kbit_training, TaskType)
            from datasets import Dataset
            from training.trainer import GpuTemperatureCallback
            from training.evidence_dataset import validate_preference_contexts
            from training.preference_compat import resolve_preference_backend, preference_length_kwargs

            trainer_class, config_class = resolve_preference_backend(self.config.method)
            length_kwargs = preference_length_kwargs(
                config_class, max_length=self.config.max_length,
                max_prompt_length=self.config.max_prompt_length,
            )

            logger.info(f"加载基础模型: {self.config.base_model_path}")
            quant_config = None
            if self.config.load_in_4bit:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
            tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            validate_preference_contexts(
                pairs, tokenizer, max_length=self.config.max_length,
                max_prompt_length=self.config.max_prompt_length,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_path,
                quantization_config=quant_config,
                device_map="auto",
            )

            # Quantized parameters must be prepared before loading a trainable adapter.
            is_quantized = (
                self.config.load_in_4bit
                or getattr(model, "is_loaded_in_4bit", False)
                or getattr(model, "is_loaded_in_8bit", False)
            )
            if is_quantized:
                model = prepare_model_for_kbit_training(model)

            # 加载 SFT adapter 作为起点（若指定）
            if self.config.adapter_path and Path(self.config.adapter_path).exists():
                logger.info(f"加载 SFT adapter: {self.config.adapter_path}")
                model = PeftModel.from_pretrained(
                    model, self.config.adapter_path, is_trainable=True
                )
            elif is_quantized:
                lora_config = LoraConfig(
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"],
                )
                model = get_peft_model(model, lora_config)
                logger.info(f"已创建 DPO LoRA adapter: r={self.config.lora_r}, alpha={self.config.lora_alpha}")

            # 构建数据集
            dataset = Dataset.from_list([
                {"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
                for p in pairs
            ])

            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # 配置训练
            if self.config.method == "orpo":
                train_config = config_class(
                    output_dir=str(output_dir),
                    beta=self.config.beta,
                    learning_rate=self.config.learning_rate,
                    num_train_epochs=self.config.num_train_epochs,
                    per_device_train_batch_size=self.config.per_device_train_batch_size,
                    gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                    **length_kwargs,
                    warmup_ratio=self.config.warmup_ratio,
                    lr_scheduler_type=self.config.lr_scheduler_type,
                    seed=self.config.seed,
                    save_total_limit=self.config.save_total_limit,
                    logging_steps=10,
                    report_to="none",
                    gradient_checkpointing=self.config.gradient_checkpointing,
                )
                trainer = trainer_class(
                    model=model,
                    args=train_config,
                    train_dataset=dataset,
                    processing_class=tokenizer,
                )
            else:
                train_config = config_class(
                    output_dir=str(output_dir),
                    beta=self.config.beta,
                    learning_rate=self.config.learning_rate,
                    num_train_epochs=self.config.num_train_epochs,
                    per_device_train_batch_size=self.config.per_device_train_batch_size,
                    gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                    **length_kwargs,
                    warmup_ratio=self.config.warmup_ratio,
                    lr_scheduler_type=self.config.lr_scheduler_type,
                    seed=self.config.seed,
                    save_total_limit=self.config.save_total_limit,
                    logging_steps=10,
                    report_to="none",
                    gradient_checkpointing=self.config.gradient_checkpointing,
                )
                trainer = trainer_class(
                    model=model,
                    args=train_config,
                    train_dataset=dataset,
                    processing_class=tokenizer,
                )

            # 添加 GPU 温度监控回调
            trainer.add_callback(GpuTemperatureCallback())

            logger.info(f"开始 {self.config.method.upper()} 训练，{len(pairs)} 条偏好对")
            train_result = trainer.train()

            # 保存 adapter
            trainer.save_model(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))

            result.train_loss = float(train_result.training_loss)
            result.train_steps = int(train_result.global_step)
            result.duration_s = round(time.monotonic() - start, 2)

            # 评估：计算 chosen vs rejected 的准确率
            # A preference win rate requires held-out pairs and explicit log-prob scoring.
            # Do not fabricate one from training loss.
            result.metric_note = (
                "Preference win rate was not computed: provide a held-out preference "
                "evaluation set and score chosen/rejected log probabilities separately."
            )
            logger.info(f"Training completed: loss={result.train_loss:.4f}; preference win rate not computed")

        except Exception as e:
            result.error = str(e)
            result.duration_s = round(time.monotonic() - start, 2)
            logger.error(f"训练失败: {e}")

        return result

    def train_mock(self, pairs: List[Dict[str, Any]]) -> PreferenceTrainResult:
        """Mock 模式：跳过训练，返回预置指标用于 CPU 验证。"""
        return PreferenceTrainResult(
            method=self.config.method,
            output_dir=self.config.output_dir,
            train_loss=0.42,
            eval_accuracy=None,
            metric_note="Mock run: no preference win rate was computed.",
            train_steps=len(pairs) * self.config.num_train_epochs,
            duration_s=0.1,
            config_snapshot=self.config.to_dict(),
            timestamp=datetime.now().isoformat(),
        )

    def save_report(self, result: PreferenceTrainResult, output_dir: Path) -> Path:
        """保存训练报告。"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = {
            "experiment_type": "preference_alignment",
            "result": result.to_dict(),
            "timestamp": ts,
        }
        report_path = output_dir / f"preference_train_{ts}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"报告已保存: {report_path}")
        return report_path


def main():
    parser = argparse.ArgumentParser(description="DPO/ORPO 偏好对齐训练")
    parser.add_argument("--mock", action="store_true", help="Mock 模式（CPU 验证）")
    parser.add_argument("--data", type=str, default="", help="偏好数据 JSONL 文件（mock 模式可省略）")
    parser.add_argument("--base-model", type=str, default="", help="基础模型路径")
    parser.add_argument("--adapter", type=str, default="", help="SFT adapter 路径")
    parser.add_argument("--method", type=str, default="dpo", choices=["dpo", "orpo"], help="训练方法")
    parser.add_argument("--output-dir", type=str, default="loras/preference_dpo", help="输出目录")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO/ORPO beta")
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    args = parser.parse_args()
    if not 0 < args.max_prompt_length < args.max_length:
        parser.error("require 0 < max-prompt-length < max-length")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    from training.preference_data_schema import PreferencePair
    from training.evidence_dataset import load_preference_training_rows

    pairs = []
    if args.data and Path(args.data).exists():
        pairs = load_preference_training_rows(Path(args.data))
        logger.info(f"加载 {len(pairs)} 条偏好对")
        if not pairs:
            logger.error("No approved preference pairs are available for training")
            raise SystemExit(2)
    elif args.mock:
        pairs = [
            PreferencePair(
                id="mock_1", prompt="测试问题", chosen="优质回复", rejected="劣质回复",
                rubric={}, annotator="mock", metadata={}, review_status="approved",
                created_at="2026-01-01T00:00:00Z",
            ).to_jsonl_dict()
        ]
        logger.info("Mock 模式：使用预置偏好对")
    else:
        logger.error("Non-mock runs require --data")
        raise SystemExit(2)

    config = PreferenceTrainingConfig(
        method=args.method,
        base_model_path=args.base_model,
        adapter_path=args.adapter,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        beta=args.beta,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )
    config.save(Path(args.output_dir) / "preference_config.json")

    trainer = PreferenceTrainer(config)
    if args.mock:
        result = trainer.train_mock(pairs)
    else:
        result = trainer.train(pairs)

    print(f"\nTraining result: method={result.method}, loss={result.train_loss}, preference_accuracy={result.eval_accuracy}")
    trainer.save_report(result, Path(args.output_dir))
    if result.error:
        logger.error(f"Training failed; report was saved: {result.error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
