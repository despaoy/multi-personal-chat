"""Download source selection tests without network requests or model weights."""

import importlib.util
import sys
from pathlib import Path

import pytest


def load_downloader():
    path = Path(__file__).resolve().parents[2] / "scripts/download_model.py"
    spec = importlib.util.spec_from_file_location("model_download_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("source,expected", [(None, "modelscope"), ("hf", "huggingface")])
def test_model_download_uses_only_selected_source(monkeypatch, tmp_path, source, expected):
    module = load_downloader()
    calls = []
    monkeypatch.setattr(module, "download_modelscope", lambda *args: calls.append("modelscope"))
    monkeypatch.setattr(module, "download_hf", lambda *args: calls.append("huggingface"))
    args = ["download_model.py", "--model", "bge-m3", "--output", str(tmp_path)]
    if source:
        args += ["--source", source]
    monkeypatch.setattr(sys, "argv", args)
    assert module.main() == 0
    assert calls == [expected]


def test_model_download_failure_never_switches_sources(monkeypatch, tmp_path):
    module = load_downloader()
    calls = []

    def fail(*args):
        calls.append("modelscope")
        raise RuntimeError("selected source unavailable")

    monkeypatch.setattr(module, "download_modelscope", fail)
    monkeypatch.setattr(module, "download_hf", lambda *args: calls.append("huggingface"))
    monkeypatch.setattr(sys, "argv", ["download_model.py", "--model", "bge-m3", "--output", str(tmp_path)])
    with pytest.raises(RuntimeError, match="selected source unavailable"):
        module.main()
    assert calls == ["modelscope"]


def test_removed_auto_source_is_rejected_before_download(monkeypatch, tmp_path, capsys):
    module = load_downloader()
    monkeypatch.setattr(
        sys, "argv", ["download_model.py", "--model", "bge-m3", "--source", "auto", "--output", str(tmp_path)]
    )
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "invalid choice: 'auto'" in message
    assert "modelscope" in message and "hf" in message
