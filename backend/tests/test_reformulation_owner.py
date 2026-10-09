"""Shared keyword rules retain adapter-specific source fields."""
import builtins
from unittest.mock import Mock

import jieba
import pytest

from knowledge.corrective_rag import CorrectiveRAG
from knowledge.grounded_answer.corrective import CorrectiveRetrievalAdapter


@pytest.mark.parametrize("kind", ["generic", "grounded"])
@pytest.mark.parametrize("failure", [None, "missing", "segmentation"])
def test_reformulation_preserves_inputs_and_reports_failure(monkeypatch, kind, failure):
    row = dict(title="the", content="beta gamma delta epsilon zeta theta", summary="unusedfield")
    if kind == "grounded":
        row["summary"], row["content"] = row["content"], row["summary"]
    original = dict(row)
    def operation():
        if kind == "generic":
            return CorrectiveRAG(None).reformulate_query("ALPHA beta", [row])
        return CorrectiveRetrievalAdapter(Mock()).reformulate_query("ALPHA beta", {"results": [row]})
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
