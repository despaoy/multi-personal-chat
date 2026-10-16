"""Explicit offline keyword controls preserve inputs and propagate failures."""
import builtins
from unittest.mock import Mock

import jieba
import pytest
from experiments.query_reformulation import append_retrieval_keywords


@pytest.mark.parametrize("failure", [None, "missing", "segmentation"])
def test_reformulation_preserves_inputs_and_reports_failure(monkeypatch, failure):
    row = dict(title="the", content="beta gamma delta epsilon zeta theta", summary="unusedfield")
    original = dict(row)
    def operation():
        return append_retrieval_keywords("ALPHA beta", [f"{row['title']} {row['content']}"])
    if failure == "missing":
        original_import = builtins.__import__
        def load(name, *args, **kwargs):
            if name == "jieba":
                raise ModuleNotFoundError("No module named 'jieba'")
            return original_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", load)
        error = ModuleNotFoundError
    elif failure == "segmentation":
        monkeypatch.setattr(jieba, "cut", Mock(side_effect=OSError("synthetic segmentation failure")))
        error = OSError
    else:
        assert operation() == "ALPHA beta gamma delta epsilon zeta theta"
        assert row == original
        return
    with pytest.raises(error):
        operation()
