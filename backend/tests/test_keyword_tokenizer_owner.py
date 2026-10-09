"""Both keyword indexes use real segmentation and propagate dependency failures."""
import builtins
from unittest.mock import Mock

import jieba
import pytest

from knowledge.retrieval_core.documents import KnowledgeIndexDocument
from knowledge.retrieval_core.index import BM25Index
from knowledge.vector_db import BM25Retriever


def build(kind):
    rows = [dict(title="书面许可", content="进入实验室必须取得书面许可"),
            dict(title="海豚", content="海豚在海里游泳")]
    if kind == "persistent":
        index = BM25Retriever()
        index.add_documents(rows)
    else:
        index = BM25Index()
        index.build([KnowledgeIndexDocument(id=str(i), domain_id="test", document_type="rule",
            title=row["title"], summary="", content=row["content"], embedding_text=row["content"]) for i, row in enumerate(rows)])
    return index


@pytest.mark.parametrize("kind", ["persistent", "domain"])
@pytest.mark.parametrize("failure", [None, "missing", "segmentation"])
def test_keyword_indexes_keep_success_and_dependency_failure_distinct(monkeypatch, kind, failure):
    index = build(kind)
    if failure is None:
        assert [i for i, score in index.search("书面许可", top_k=5)] == [0]
        assert index.search("火星", top_k=5) == []
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
        monkeypatch.setattr(jieba, "cut", Mock(side_effect=OSError("synthetic tokenizer failure")))
        error = OSError
    with pytest.raises(error):
        index.search("书面许可", top_k=5)
    with pytest.raises(error):
        build(kind)
