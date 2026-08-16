"""LoRA 消融实验框架 - 对比 LoRA / DoRA / RSLoRA / NEFTune / Packing 变体。

遵循路线图 guardrail：所有变体必须固定 seed/data/base_model/lr/epochs，否则不称为消融。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _BACKEND_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))


@dataclass
class AblationResult:
    """单个变体实验结果。"""
    variant_name: str
    config_snapshot: Dict[str, Any]
    training_eval: Dict[str, Any] = field(default_factory=dict)
    adapter_size_mb: float = 0.0
    trainable_params: Optional[int] = None
    peak_vram_gb: Optional[float] = None
    generation_metrics: Optional[Dict[str, Any]] = None
    error_cases: List[str] = field(default_factory=list)
    status: str = "completed"
    duration_s: float = 0.0


@dataclass
class AblationExperiment:
    """消融实验定义。"""
    experiment_id: str
    hypothesis: str
    controlled_variables: Dict[str, Any]
    config_variants: Dict[str, Dict[str, Any]]  # variant_name -> config overrides
    metrics_to_collect: List[str] = field(default_factory=lambda: [
        "eval_loss", "perplexity", "adapter_size_mb", "trainable_params",
        "distinct_1", "distinct_2", "repetition_rate",
    ])


class AblationRunner:
    """消融实验运行器。"""

    # 默认消融变体配置覆盖（基于 LoRATrainingConfig 字段）
    DEFAULT_VARIANTS = {
        "lora_baseline": {"use_dora": False, "use_rslora": False, "neftune_noise_alpha": 0.0, "packing": False},
        "lora_neftune": {"use_dora": False, "use_rslora": False, "neftune_noise_alpha": 5.0, "packing": False},
        "lora_packing": {"use_dora": False, "use_rslora": False, "neftune_noise_alpha": 0.0, "packing": True},
        "dora": {"use_dora": True, "use_rslora": False, "neftune_noise_alpha": 0.0, "packing": False},
        "rslora": {"use_dora": False, "use_rslora": True, "neftune_noise_alpha": 0.0, "packing": False},
    }

    def __init__(self, experiment: Optional[AblationExperiment] = None,
                 base_config: Optional[Dict[str, Any]] = None):
        self.experiment = experiment
        self.base_config = base_config or {}

    @classmethod
    def from_default_config(cls, config_overrides: Optional[Dict[str, Any]] = None) -> "AblationRunner":
        """从默认配置创建运行器。"""
        base = {
            "base_model_path": os.getenv("BASE_MODEL_PATH", "models/Qwen3-8B-Instruct"),
            "train_data_path": os.getenv(
                "LORA_ABLATION_TRAIN_DATA",
                "backend/data/character_dialogues/experiments/v4/train.jsonl",
            ),
            "lora_r": 32,
            "lora_alpha": 64,
            "learning_rate": 2e-4,
            "num_train_epochs": 3,
            "seed": 42,
        }
        if config_overrides:
            base.update(config_overrides)
        experiment = AblationExperiment(
            experiment_id=f"lora_abl_{int(time.time())}",
            hypothesis="DoRA 和 RSLoRA 在相同 rank 下能提升角色一致性；NEFTune 提升鲁棒性；packing 提升吞吐",
            controlled_variables={
                "seed": base.get("seed", 42),
                "base_model_path": base.get("base_model_path"),
                "train_data_path": base.get("train_data_path"),
                "learning_rate": base.get("learning_rate"),
                "num_train_epochs": base.get("num_train_epochs"),
                "lora_r": base.get("lora_r"),
            },
            config_variants=cls.DEFAULT_VARIANTS,
        )
        return cls(experiment=experiment, base_config=base)

    def _validate_controlled_variables(self) -> List[str]:
        """校验控制变量一致性。"""
        errors = []
        cv = self.experiment.controlled_variables
        for key in ("seed", "base_model_path", "train_data_path", "learning_rate", "num_train_epochs"):
            if key not in cv:
                errors.append(f"控制变量缺失: {key}")
        return errors

    def _measure_adapter_size(self, output_dir: Path) -> float:
        """测量 adapter 目录大小（MB）。"""
        if not output_dir.exists():
            return 0.0
        # Count adapter tensors only; checkpoints and tokenizer files are not adapter size.
        total = sum(
            f.stat().st_size
            for f in output_dir.rglob("adapter_model.*")
            if f.is_file()
        )
        return round(total / (1024 * 1024), 2)

    def run_single(self, variant_name: str, config_overrides: Dict[str, Any],
                   mock: bool = False) -> AblationResult:
        """运行单个变体。"""
        start_time = time.time()
        logger.info(f"=== 运行变体: {variant_name} ===")

        # 合并配置
        config_dict = dict(self.base_config)
        config_dict.update(config_overrides)
        config_dict["output_dir"] = str(_BACKEND_DIR / "loras" / f"ablation_{variant_name}_{int(time.time())}")

        if mock:
            logger.info(f"[mock] 跳过实际训练: {variant_name}")
            return AblationResult(
                variant_name=variant_name,
                config_snapshot=config_dict,
                training_eval={
                    "eval_loss": 1.85 + hash(variant_name) % 20 / 100,
                    "eval_perplexity": 6.36,
                    "best_eval_loss": 1.79,
                },
                adapter_size_mb=45.2,
                trainable_params=19568128,
                peak_vram_gb=18.5,
                generation_metrics={"distinct_1": 0.42, "distinct_2": 0.68, "repetition_rate": 0.05},
                status="completed",
                duration_s=0.0,
            )

        try:
            from training.trainer import LoRATrainer, LoRATrainingConfig
            config = LoRATrainingConfig.from_dict(config_dict)
            errors = config.validate()
            if errors:
                logger.error(f"配置校验失败: {errors}")
                return AblationResult(
                    variant_name=variant_name, config_snapshot=config_dict,
                    status="failed", error_cases=errors,
                )

            trainer = LoRATrainer(config)
            output_path = trainer.train()

            # 读取 training_evaluation.json
            eval_path = Path(config.output_dir) / "training_evaluation.json"
            training_eval = {}
            if eval_path.exists():
                with open(eval_path, "r", encoding="utf-8") as f:
                    training_eval = json.load(f)

            adapter_size = self._measure_adapter_size(Path(config.output_dir) / "final")
            trainable_params = None
            results_path = Path(config.output_dir) / "training_results.json"
            if results_path.exists():
                with open(results_path, "r", encoding="utf-8") as f:
                    trainable_params = json.load(f).get("trainable_params")

            return AblationResult(
                variant_name=variant_name,
                config_snapshot=config_dict,
                training_eval=training_eval.get("metrics", training_eval),
                adapter_size_mb=adapter_size,
                trainable_params=trainable_params,
                status="completed",
                duration_s=round(time.time() - start_time, 2),
            )
        except Exception as e:
            logger.error(f"变体 {variant_name} 运行失败: {e}")
            return AblationResult(
                variant_name=variant_name, config_snapshot=config_dict,
                status="failed", error_cases=[str(e)],
                duration_s=round(time.time() - start_time, 2),
            )

    def run_all(self, mock: bool = False) -> Dict[str, Any]:
        """运行所有变体。"""
        errors = self._validate_controlled_variables()
        if errors:
            return {"error": "controlled variables validation failed", "details": errors}

        results: Dict[str, AblationResult] = {}
        for variant_name, overrides in self.experiment.config_variants.items():
            result = self.run_single(variant_name, overrides, mock=mock)
            results[variant_name] = asdict(result)

        table = self.build_comparison_table(results)
        report = {
            "experiment_id": self.experiment.experiment_id,
            "hypothesis": self.experiment.hypothesis,
            "controlled_variables": self.experiment.controlled_variables,
            "comparison_table": table,
            "raw_results": results,
            "mock": mock,
        }
        return report

    def build_comparison_table(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成对比表格。"""
        table = []
        for variant_name, result in results.items():
            training_eval = result.get("training_eval", {}) if isinstance(result, dict) else result.training_eval
            gen_metrics = (result.get("generation_metrics") or {}) if isinstance(result, dict) else (result.generation_metrics or {})
            row = {
                "variant": variant_name,
                "status": result.get("status", "unknown") if isinstance(result, dict) else result.status,
                "eval_loss": training_eval.get("final_eval_loss") or training_eval.get("eval_loss"),
                "perplexity": training_eval.get("eval_perplexity"),
                "adapter_size_mb": result.get("adapter_size_mb", 0) if isinstance(result, dict) else result.adapter_size_mb,
                "trainable_params": result.get("trainable_params", 0) if isinstance(result, dict) else result.trainable_params,
                "peak_vram_gb": result.get("peak_vram_gb") if isinstance(result, dict) else result.peak_vram_gb,
                "distinct_1": gen_metrics.get("distinct_1"),
                "distinct_2": gen_metrics.get("distinct_2"),
                "repetition_rate": gen_metrics.get("avg_repetition_rate"),
                "duration_s": result.get("duration_s", 0) if isinstance(result, dict) else result.duration_s,
            }
            table.append(row)
        return table

    def save_report(self, report: Dict[str, Any], output_dir: Path) -> Path:
        """保存实验报告（JSON + Markdown）。"""
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "ablation_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        md_path = output_dir / "ablation_report.md"
        lines = [
            f"# LoRA 消融实验报告",
            f"",
            f"- **实验 ID**: {report.get('experiment_id', 'N/A')}",
            f"- **假设**: {report.get('hypothesis', 'N/A')}",
            f"- **控制变量**: `{json.dumps(report.get('controlled_variables', {}), ensure_ascii=False)}`",
            f"",
            "## 对比表",
            "",
            "| Variant | Status | Eval Loss | PPL | Adapter(MB) | Trainable | Distinct-1 | Distinct-2 | Repetition | Duration(s) |",
            "|---------|--------|-----------|-----|-------------|-----------|------------|------------|------------|-------------|",
        ]
        for row in report.get("comparison_table", []):
            lines.append(
                f"| {row.get('variant','')} | {row.get('status','')} | "
                f"{row.get('eval_loss','N/A')} | {row.get('perplexity','N/A')} | "
                f"{row.get('adapter_size_mb','N/A')} | {row.get('trainable_params','N/A')} | "
                f"{row.get('distinct_1','N/A')} | {row.get('distinct_2','N/A')} | "
                f"{row.get('repetition_rate','N/A')} | {row.get('duration_s','N/A')} |"
            )
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"报告已保存: {json_path} / {md_path}")
        return md_path


def main():
    parser = argparse.ArgumentParser(description="LoRA 消融实验")
    parser.add_argument("--mock", action="store_true", help="mock 模式，跳过实际训练")
    parser.add_argument("--output-dir", type=str, default="reports/ablation", help="报告输出目录")
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--train-data", type=str, default=None)
    parser.add_argument("--variants", type=str, default="",
                        help="逗号分隔的变体名，如 lora_baseline,dora。留空运行全部")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    overrides = {}
    if args.base_model:
        overrides["base_model_path"] = args.base_model
    if args.train_data:
        overrides["train_data_path"] = args.train_data

    runner = AblationRunner.from_default_config(overrides)

    if args.variants:
        selected = [v.strip() for v in args.variants.split(",") if v.strip()]
        runner.experiment.config_variants = {
            k: v for k, v in runner.experiment.config_variants.items() if k in selected
        }
        logger.info(f"仅运行变体: {list(runner.experiment.config_variants.keys())}")

    report = runner.run_all(mock=args.mock)

    output_dir = _PROJECT_ROOT / args.output_dir
    runner.save_report(report, output_dir)
    print(f"\n消融实验完成。报告: {output_dir / 'ablation_report.md'}")


if __name__ == "__main__":
    main()
