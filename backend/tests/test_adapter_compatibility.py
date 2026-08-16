"""Tests for adapter base_model compatibility checks.

验证 AdapterChecker 能正确识别基座不匹配（如 Qwen2.5-7B 适配器加载到
Qwen3-8B 基座），并在报告中标明 base_model_mismatch，供上游返回明确的
409 LORA_BASE_MODEL_MISMATCH。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def _write_adapter(adapter_dir: Path, base_model: str) -> None:
    """写入指定 base_model 的 adapter_config.json。"""
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
            "base_model_name_or_path": base_model,
            "peft_version": "0.19.1",
        }),
        encoding="utf-8",
    )
    # 写入权重文件占位
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"")


def test_base_model_match_is_compatible(tmp_path):
    """基座模型名一致（路径不同但 basename 相同）应视为兼容。"""
    adapter_dir = tmp_path / "kisaki"
    _write_adapter(adapter_dir, "/root/training/models/Qwen3-8B-Instruct")

    from inference.adapter_checker import AdapterChecker
    checker = AdapterChecker(
        expected_base_model="/lab/runtime/models/Qwen3-8B-Instruct",
        lora_root=str(tmp_path),
    )
    report = checker.check_adapter("kisaki")

    assert report.compatible is True
    assert report.base_model_mismatch is False
    assert report.checks["base_model"] is True


def test_base_model_mismatch_is_incompatible(tmp_path):
    """Qwen2.5-7B 适配器加载到 Qwen3-8B 基座应判为不兼容。"""
    adapter_dir = tmp_path / "hutao"
    _write_adapter(adapter_dir, "/root/hutao-training/models/Qwen2.5-7B-Instruct")

    from inference.adapter_checker import AdapterChecker
    checker = AdapterChecker(
        expected_base_model="/lab/runtime/models/Qwen3-8B-Instruct",
        lora_root=str(tmp_path),
    )
    report = checker.check_adapter("hutao")

    assert report.compatible is False
    assert report.base_model_mismatch is True
    assert report.checks["base_model"] is False
    assert "Qwen2.5-7B-Instruct" in report.actual_base_model
    assert "Qwen3-8B-Instruct" in report.expected_base_model
    assert any("base_model" in e for e in report.errors)


def test_relative_base_model_path_compared_by_basename(tmp_path):
    """相对路径形式的 base_model 应按 basename 比较。"""
    adapter_dir = tmp_path / "minamo"
    _write_adapter(adapter_dir, "./Qwen2.5-7B-Instruct")

    from inference.adapter_checker import AdapterChecker
    checker = AdapterChecker(
        expected_base_model="/lab/runtime/models/Qwen3-8B-Instruct",
        lora_root=str(tmp_path),
    )
    report = checker.check_adapter("minamo")

    assert report.compatible is False
    assert report.base_model_mismatch is True


def test_empty_expected_base_model_skips_check(tmp_path):
    """未配置期望基座时应跳过检查（仅警告）。"""
    adapter_dir = tmp_path / "lora1"
    _write_adapter(adapter_dir, "/some/model")

    from inference.adapter_checker import AdapterChecker
    checker = AdapterChecker(expected_base_model="", lora_root=str(tmp_path))
    report = checker.check_adapter("lora1")

    assert report.base_model_mismatch is False
    assert report.checks["base_model"] is True
    assert any("跳过基座" in w for w in report.warnings)


def test_explicit_empty_string_not_overridden_by_env(monkeypatch, tmp_path):
    """显式传入空字符串必须禁用基座比较，即使 BASE_MODEL_PATH 环境变量已设置。

    回归测试：此前实现用 `expected_base_model or os.getenv(...)`，会把显式
    空字符串当成未传值，导致同一测试在全量运行（其他用例设置了环境变量）
    和单独运行时结果不同。改为 None 判断后，空字符串语义稳定。
    """
    adapter_dir = tmp_path / "lora2"
    _write_adapter(adapter_dir, "/some/Qwen2.5-7B-Instruct")

    # 模拟其他测试设置环境变量的场景
    monkeypatch.setenv("BASE_MODEL_PATH", "/env/Qwen3-8B-Instruct")

    from inference.adapter_checker import AdapterChecker
    # 显式传入空字符串，应禁用比较，而不是用环境变量做比较
    checker = AdapterChecker(expected_base_model="", lora_root=str(tmp_path))
    report = checker.check_adapter("lora2")

    assert report.base_model_mismatch is False
    assert report.checks["base_model"] is True
    assert any("跳过基座" in w for w in report.warnings)


def test_none_falls_back_to_env(monkeypatch, tmp_path):
    """传 None（未传值）应回退到环境变量。"""
    adapter_dir = tmp_path / "lora3"
    _write_adapter(adapter_dir, "/some/Qwen2.5-7B-Instruct")

    monkeypatch.setenv("BASE_MODEL_PATH", "/env/Qwen3-8B-Instruct")

    from inference.adapter_checker import AdapterChecker
    checker = AdapterChecker(expected_base_model=None, lora_root=str(tmp_path))
    report = checker.check_adapter("lora3")

    # 环境变量生效，应检测到不匹配
    assert report.base_model_mismatch is True
    assert report.compatible is False


def test_empty_actual_base_model_is_error(tmp_path):
    """adapter_config.json 中 base_model 为空应为 error。"""
    adapter_dir = tmp_path / "bad"
    _write_adapter(adapter_dir, "")

    from inference.adapter_checker import AdapterChecker
    checker = AdapterChecker(
        expected_base_model="/root/models/Qwen3-8B-Instruct",
        lora_root=str(tmp_path),
    )
    report = checker.check_adapter("bad")

    assert report.compatible is False
    assert report.checks["base_model"] is False
