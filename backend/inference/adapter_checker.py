"""适配器兼容性检查 - 在激活前验证 adapter_config.json 的兼容性。

遵循路线图 guardrail：
- 验证 base_model_id、tokenizer、target_modules、rank、PEFT version
- 不兼容时阻止激活并降级到 default
- 返回结构化报告供 API 和日志使用
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AdapterCompatibilityReport:
    """适配器兼容性检查报告。"""
    adapter_name: str
    compatible: bool = False
    checks: Dict[str, bool] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    checked_at: str = ""
    # base_model 不匹配是硬性错误：LoRA 只能加载到训练时使用的基座上，
    # 跨基座（如 Qwen2.5-7B 适配器加载到 Qwen3-8B）会触发 vLLM 400。
    # 该标识用于上游返回明确的 409 LORA_BASE_MODEL_MISMATCH，而非模糊的 502。
    base_model_mismatch: bool = False
    expected_base_model: str = ""
    actual_base_model: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "adapter_name": self.adapter_name,
            "compatible": self.compatible,
            "checks": self.checks,
            "warnings": self.warnings,
            "errors": self.errors,
            "checked_at": self.checked_at,
            "base_model_mismatch": self.base_model_mismatch,
            "expected_base_model": self.expected_base_model,
            "actual_base_model": self.actual_base_model,
        }


class AdapterChecker:
    """适配器兼容性检查器。"""

    def __init__(self, expected_base_model: Optional[str] = None, lora_root: Optional[str] = None):
        # 仅在参数为 None（未传值）时回退到环境变量；显式空字符串表示
        # 调用方有意禁用基座兼容性比较，不应被环境变量覆盖。
        if expected_base_model is None:
            self.expected_base_model = os.getenv("BASE_MODEL_PATH", "")
        else:
            self.expected_base_model = expected_base_model
        if lora_root is None:
            self.lora_root = os.getenv("LORA_PATH", "")
        else:
            self.lora_root = lora_root

    def _find_adapter_dir(self, adapter_name: str) -> Optional[Path]:
        """查找适配器目录（支持 final 子目录）。"""
        if not self.lora_root:
            return None
        base = Path(self.lora_root) / adapter_name
        if (base / "adapter_config.json").exists():
            return base
        final = base / "final"
        if (final / "adapter_config.json").exists():
            return final
        return None

    def check_adapter(self, adapter_path: str | Path) -> AdapterCompatibilityReport:
        """检查单个适配器的兼容性。

        Args:
            adapter_path: 适配器目录路径，或适配器名称（自动解析）
        """
        from datetime import datetime
        path = Path(adapter_path)

        # 如果传入的是名称而非路径，尝试解析
        if not path.exists() and self.lora_root:
            resolved = self._find_adapter_dir(str(adapter_path))
            if resolved:
                path = resolved

        # 注册表可能保存适配器根目录，而实际 PEFT 产物位于 final/。
        # 即使根目录本身存在，也必须继续解析 final 子目录；否则路由页
        # 会把已经被 vLLM 成功加载的适配器误报为“不兼容”。
        if not (path / "adapter_config.json").exists():
            final = path / "final"
            if (final / "adapter_config.json").exists():
                path = final

        adapter_name = path.name if path.name != "final" else path.parent.name
        report = AdapterCompatibilityReport(
            adapter_name=adapter_name,
            checked_at=datetime.now().isoformat(),
        )

        config_path = path / "adapter_config.json"
        if not config_path.exists():
            report.errors.append("adapter_config.json 不存在")
            report.checks["config_exists"] = False
            return report

        report.checks["config_exists"] = True

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:
            report.errors.append(f"adapter_config.json 解析失败: {e}")
            return report

        # 1. 检查 base_model_name_or_path（按模型名 basename 比较，跨基座为硬错误）
        base_model = cfg.get("base_model_name_or_path", "")
        report.actual_base_model = base_model
        report.expected_base_model = self.expected_base_model
        if not base_model:
            report.errors.append("base_model_name_or_path 为空")
            report.checks["base_model"] = False
        elif not self.expected_base_model:
            # 未配置期望基座，无法判定，仅警告
            report.warnings.append("未配置 BASE_MODEL_PATH，跳过基座兼容性检查")
            report.checks["base_model"] = True
        else:
            # 比较模型名（basename），避免部署路径不同导致的误判。
            # 例如训练时 /lab/training/models/Qwen3-8B-Instruct 与
            # 部署时 /lab/runtime/models/Qwen3-8B-Instruct 应视为同基座。
            actual_name = Path(base_model).name
            expected_name = Path(self.expected_base_model).name
            if actual_name != expected_name:
                report.base_model_mismatch = True
                report.errors.append(
                    f"base_model 不匹配: adapter={actual_name}, expected={expected_name} "
                    f"(LoRA 只能加载到训练时使用的基座)"
                )
                report.checks["base_model"] = False
            else:
                report.checks["base_model"] = True

        # 2. 检查 target_modules 非空
        target_modules = cfg.get("target_modules", [])
        if not target_modules:
            report.errors.append("target_modules 为空")
            report.checks["target_modules"] = False
        else:
            report.checks["target_modules"] = True

        # 3. 检查 rank (r) > 0
        rank = cfg.get("r", 0)
        if rank <= 0:
            report.errors.append(f"rank (r) 无效: {rank}")
            report.checks["rank"] = False
        else:
            report.checks["rank"] = True

        # 4. 检查 PEFT version
        peft_version = cfg.get("version", cfg.get("peft_version", ""))
        if not peft_version:
            report.warnings.append("PEFT version 未记录")
            report.checks["peft_version"] = True  # 警告但不阻止
        else:
            report.checks["peft_version"] = True

        # 5. 检查 adapter 权重文件存在
        weights_exist = (path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists()
        if not weights_exist:
            report.errors.append("adapter_model.safetensors / adapter_model.bin 不存在")
            report.checks["weights_exist"] = False
        else:
            report.checks["weights_exist"] = True

        # 6. 检查 tokenizer 文件
        tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "vocab.json"]
        has_tokenizer = any((path / tf).exists() for tf in tokenizer_files)
        if not has_tokenizer:
            report.warnings.append("tokenizer 文件不在 adapter 目录（可能复用 base model tokenizer）")
            report.checks["tokenizer"] = True  # 警告但不阻止
        else:
            report.checks["tokenizer"] = True

        # 总体兼容性：无 errors 则兼容
        report.compatible = len(report.errors) == 0
        return report

    def check_all_adapters(self) -> Dict[str, AdapterCompatibilityReport]:
        """检查 lora_root 下所有适配器。"""
        results: Dict[str, AdapterCompatibilityReport] = {}
        if not self.lora_root:
            return results
        root = Path(self.lora_root)
        if not root.exists():
            return results
        for d in root.iterdir():
            if d.is_dir():
                report = self.check_adapter(d)
                results[d.name] = report
        return results


def safe_resolve_lora(name: str, checker: Optional[AdapterChecker] = None) -> str:
    """安全解析 LoRA 名称：检查通过返回原名，否则降级到 default。

    Args:
        name: 请求的 LoRA 名称
        checker: 适配器检查器实例（None 时创建默认）

    Returns:
        兼容则返回 name，不兼容则返回 "default"
    """
    if name == "default" or not name:
        return "default"

    if checker is None:
        checker = AdapterChecker()

    report = checker.check_adapter(name)
    if not report.compatible:
        logger.warning(
            f"适配器 {name} 不兼容，降级到 default。errors={report.errors}"
        )
        return "default"

    return name
