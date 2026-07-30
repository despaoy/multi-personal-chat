"""Tests for LoRA adapter path resolution and symlink trusted-boundary checks.

覆盖 _resolve_vllm_adapter_path 的安全模型：
1. 逻辑路径穿越（../）被拒绝
2. LORA_ROOT 下的真实目录通过
3. LORA_ALLOWED_REAL_ROOTS 下的符号链接目标通过
4. 未列入允许列表的根外符号链接目标被拒绝

注意：Windows 普通用户无权创建符号链接，因此符号链接场景通过
monkeypatch Path.resolve 模拟，避免依赖文件系统符号链接支持。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def _write_adapter(adapter_dir: Path) -> None:
    """写入一个最小的 adapter_config.json 使该目录被视为合法适配器。"""
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"r": 16, "lora_alpha": 32, "target_modules": ["q_proj"]}),
        encoding="utf-8",
    )


def _setup_lora_root(monkeypatch, tmp_path: Path) -> Path:
    """构造一个临时 LORA_ROOT 并 patch 到 backend.api.loras 模块。"""
    lora_root = tmp_path / "runtime" / "loras"
    lora_root.mkdir(parents=True)
    from db import database as db_module
    from api import loras as loras_module
    monkeypatch.setattr(db_module, "LORA_ROOT", lora_root)
    monkeypatch.setattr(loras_module, "LORA_ROOT", lora_root)
    return lora_root


def _patch_resolve_for_symlink(monkeypatch, logical_to_real: dict[Path, Path]):
    """模拟符号链接：让特定逻辑路径的 resolve() 返回指定的真实路径。

    Args:
        logical_to_real: {逻辑路径: 真实路径} 映射。对于不在映射中的路径，
                        回退到原始 Path.resolve 行为。
    """
    original_resolve = Path.resolve

    def fake_resolve(self, *args, **kwargs):
        # 规范化路径以匹配映射键
        try:
            normalized = Path(os.path.normpath(str(self)))
        except Exception:
            normalized = self
        for logical, real in logical_to_real.items():
            if os.path.normpath(str(self)) == os.path.normpath(str(logical)):
                return real
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", fake_resolve)


def test_real_dir_inside_lora_root_is_accepted(monkeypatch, tmp_path):
    """LORA_ROOT 下的真实目录（非符号链接）应通过。"""
    lora_root = _setup_lora_root(monkeypatch, tmp_path)
    monkeypatch.delenv("LORA_ALLOWED_REAL_ROOTS", raising=False)
    _write_adapter(lora_root / "kisaki")

    from api.loras import _resolve_vllm_adapter_path
    result = _resolve_vllm_adapter_path("kisaki")
    assert result.endswith("kisaki")


def test_traversal_in_lora_name_is_rejected(monkeypatch, tmp_path):
    """lora_name 含 ../ 应被逻辑路径检查拒绝。"""
    _setup_lora_root(monkeypatch, tmp_path)
    monkeypatch.delenv("LORA_ALLOWED_REAL_ROOTS", raising=False)

    from api.loras import _resolve_vllm_adapter_path
    with pytest.raises(ValueError, match="escapes the configured root"):
        _resolve_vllm_adapter_path("../evil")


def test_symlink_to_allowed_real_root_is_accepted(monkeypatch, tmp_path):
    """指向 LORA_ALLOWED_REAL_ROOTS 内的符号链接应通过。"""
    lora_root = _setup_lora_root(monkeypatch, tmp_path)
    allowed_root = tmp_path / "qqchat-data" / "loras"
    _write_adapter(allowed_root / "hutao")
    # 模拟符号链接：lora_root/hutao -> allowed_root/hutao
    logical_link = lora_root / "hutao"
    real_target = allowed_root / "hutao"
    _patch_resolve_for_symlink(monkeypatch, {logical_link: real_target})
    monkeypatch.setenv("LORA_ALLOWED_REAL_ROOTS", str(allowed_root))

    from api.loras import _resolve_vllm_adapter_path
    result = _resolve_vllm_adapter_path("hutao")
    assert result.endswith("hutao")


def test_symlink_to_unlisted_real_root_is_rejected(monkeypatch, tmp_path):
    """指向未列入允许列表的根外符号链接目标应被拒绝。

    这是 P1 修复的核心：即使逻辑路径合法，真实目标位于受信边界外
    也必须拒绝，避免任意符号链接把适配器指向任意位置。
    """
    lora_root = _setup_lora_root(monkeypatch, tmp_path)
    evil_root = tmp_path / "evil_root"
    _write_adapter(evil_root / "stolen")
    # 模拟符号链接：lora_root/stolen -> evil_root/stolen
    logical_link = lora_root / "stolen"
    real_target = evil_root / "stolen"
    _patch_resolve_for_symlink(monkeypatch, {logical_link: real_target})
    monkeypatch.delenv("LORA_ALLOWED_REAL_ROOTS", raising=False)

    from api.loras import _resolve_vllm_adapter_path
    with pytest.raises(ValueError, match="escapes the trusted roots"):
        _resolve_vllm_adapter_path("stolen")


def test_symlink_to_unlisted_real_root_rejected_even_when_other_roots_allowed(monkeypatch, tmp_path):
    """配置了允许列表，但目标不在其中，仍应被拒绝。"""
    lora_root = _setup_lora_root(monkeypatch, tmp_path)
    allowed_root = tmp_path / "qqchat-data" / "loras"
    allowed_root.mkdir(parents=True)
    evil_root = tmp_path / "evil_root"
    _write_adapter(evil_root / "stolen")
    logical_link = lora_root / "stolen"
    real_target = evil_root / "stolen"
    _patch_resolve_for_symlink(monkeypatch, {logical_link: real_target})
    monkeypatch.setenv("LORA_ALLOWED_REAL_ROOTS", str(allowed_root))

    from api.loras import _resolve_vllm_adapter_path
    with pytest.raises(ValueError, match="escapes the trusted roots"):
        _resolve_vllm_adapter_path("stolen")


def test_final_subdir_is_resolved(monkeypatch, tmp_path):
    """适配器在 final/ 子目录下时应正确解析。"""
    lora_root = _setup_lora_root(monkeypatch, tmp_path)
    monkeypatch.delenv("LORA_ALLOWED_REAL_ROOTS", raising=False)
    _write_adapter(lora_root / "kisaki" / "final")

    from api.loras import _resolve_vllm_adapter_path
    result = _resolve_vllm_adapter_path("kisaki")
    assert result.endswith(os.path.join("kisaki", "final"))


def test_missing_adapter_config_raises_file_not_found(monkeypatch, tmp_path):
    """adapter_config.json 不存在时应抛出 FileNotFoundError。"""
    lora_root = _setup_lora_root(monkeypatch, tmp_path)
    monkeypatch.delenv("LORA_ALLOWED_REAL_ROOTS", raising=False)
    (lora_root / "empty").mkdir()

    from api.loras import _resolve_vllm_adapter_path
    with pytest.raises(FileNotFoundError):
        _resolve_vllm_adapter_path("empty")
