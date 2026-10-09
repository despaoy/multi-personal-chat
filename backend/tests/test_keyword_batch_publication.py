"""Tokenization failure cannot publish half of a keyword index batch."""
import jieba
import pytest

from knowledge.retrieval_core.documents import KnowledgeIndexDocument
from knowledge.retrieval_core.index import BM25Index
from knowledge.vector_db import BM25Retriever


@pytest.mark.parametrize("kind", ["persistent", "domain"])
def test_late_tokenization_failure_keeps_previous_index_and_retry_is_clean(monkeypatch, kind):
    index = BM25Retriever() if kind == "persistent" else BM25Index()
    def write(texts):
        if kind == "persistent":
            index.add_documents([dict(title=text, content=text) for text in texts])
        else:
            index.build([KnowledgeIndexDocument(id=str(i), domain_id="test", document_type="rule",
                title=text, summary="", content=text, embedding_text=text) for i, text in enumerate(texts)])
    stats = index.get_stats if kind == "persistent" else index.stats
    write(["书面许可"])
    before, before_stats = index.search("书面许可", top_k=5), stats()
    original = jieba.cut
    def segment(text):
        if "后条故障" in text:
            raise OSError("synthetic late segmentation failure")
        return original(text)
    with monkeypatch.context() as patch:
        patch.setattr(jieba, "cut", segment)
        with pytest.raises(OSError, match="late segmentation"):
            write(["海豚", "后条故障"])
        assert index.search("书面许可", top_k=5) == before
        assert stats() == before_stats
        assert index.search("海豚", top_k=5) == []
    write(["海豚", "后条故障"])
    assert len(index.search("海豚", top_k=5)) == 1
    assert stats()["total_docs" if kind == "persistent" else "docs"] == (3 if kind == "persistent" else 2)
