"""Segmentation failures abort query scoring and corrective reformulation."""
import builtins
from unittest.mock import Mock

import jieba
import pytest
from experiments.query_reformulation import append_retrieval_keywords

from knowledge.retrieval_core.documents import KnowledgeIndexDocument
from knowledge.retrieval_core.query import QueryAnalysis
from knowledge.retrieval_core.rerank import PipelineReranker
from knowledge.retrieval_core.retrieval import RetrievalCandidate


def rerank():
    rows = [RetrievalCandidate(i, KnowledgeIndexDocument(str(i), "test", "fact", text, text, text, text))
            for i, text in enumerate(["海豚在海里游泳", "进入实验室必须取得书面许可"])]
    result = PipelineReranker(cross_encoder_enabled=False).rerank(QueryAnalysis("书面许可", "书面许可"), rows, 2)
    assert [row.document.id for row in result] == ["1", "0"]
    assert all(row.rerank_method == "deterministic" for row in result)
    assert all(row.rerank_method == "none" for row in rows)


def reformulate():
    result = append_retrieval_keywords("进入需要什么条件", ["实验室 进入实验室必须取得书面许可"])
    assert result.startswith("进入需要什么条件 ") and "许可" in result


@pytest.mark.parametrize("operation", [rerank, reformulate])
@pytest.mark.parametrize("failure", [None, "missing", "segmentation"])
def test_query_segmentation_success_and_failure(monkeypatch, operation, failure):
    if failure is None:
        operation()
        return
    if failure == "missing":
        original = builtins.__import__
        def load(name, *args, **kwargs):
            if name == "jieba":
                raise ModuleNotFoundError("No module named 'jieba'")
            return original(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", load)
        error = ModuleNotFoundError
    else:
        monkeypatch.setattr(jieba, "cut", Mock(side_effect=OSError("synthetic segmentation failure")))
        error = OSError
    with pytest.raises(error):
        operation()
