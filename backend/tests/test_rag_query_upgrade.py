"""RAG 稳定排序、结果隔离及独立纠正接口验证。

只覆盖本次升级直接相关的行为：
- 稳定 ID 融合正确且不修改 vector_db 返回的原始字典
- 独立纠正接口使用中文分词
- CorrectiveRAG max_retries=0/1 行为正确
- no-op 重写不会触发第二次检索
"""

from __future__ import annotations

import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from knowledge import rag_helper
from knowledge.corrective_rag import CorrectiveRAG
from knowledge.rag_helper import _stable_result_key

# ---------------------------------------------------------------------------
# 稳定 ID 融合
# ---------------------------------------------------------------------------


class _FakeVectorDB:
    """hybrid_search 替身：按调用顺序返回预设结果，并保留返回的原始字典引用。"""

    def __init__(self, rounds: list[list[dict[str, Any]]]):
        self._rounds = list(rounds)
        self.cache_generation = 1
        self.returned: list[dict[str, Any]] = []

    def hybrid_search(self, query, top_k=5, threshold=0.15, keyword_weight=0.3, filters=None) -> list[dict[str, Any]]:
        if not self._rounds:
            return []
        batch = self._rounds.pop(0)
        fresh = [dict(r) for r in batch]
        self.returned.extend(fresh)
        return fresh


def _make_helper(monkeypatch, fake: _FakeVectorDB) -> rag_helper.RAGHelper:
    monkeypatch.setattr(rag_helper, "get_vector_db", lambda: fake, raising=False)
    helper = rag_helper.RAGHelper()
    return helper


class TestFusion:

    def test_fusion_does_not_mutate_vector_db_results(self, monkeypatch):
        doc = {"id": "doc_1_chunk_0", "title": "胡桃攻略", "content": "往生堂堂主", "score": 0.5, "fused_score": 0.5}
        fake = _FakeVectorDB([[dict(doc)]])
        helper = _make_helper(monkeypatch, fake)
        helper.retrieve_context("胡桃怎么配队", top_k=3, use_cache=False)

        assert len(fake.returned) == 1
        for raw in fake.returned:
            assert "query_count" not in raw
            assert "final_score" not in raw
            assert "region_boost" not in raw

    def test_no_id_uses_content_digest_and_avoids_prefix_collision(self, monkeypatch):
        # 同 title 且前 100 字符相同、但整体内容不同的两份文档不应被误融合
        shared_prefix = "胡桃是往生堂第七十七代堂主，掌管往生堂事务，" + "火系角色。" * 20
        doc_a = {"title": "胡桃", "content": shared_prefix + "尾部差异A", "score": 0.6}
        doc_b = {"title": "胡桃", "content": shared_prefix + "尾部差异B", "score": 0.7}
        assert _stable_result_key(doc_a) != _stable_result_key(doc_b)
        assert _stable_result_key(doc_a) == _stable_result_key(dict(doc_a))

        fake = _FakeVectorDB([[doc_a, doc_b]])
        helper = _make_helper(monkeypatch, fake)
        results = helper.retrieve_context("胡桃怎么配队", top_k=5, use_cache=False)

        assert len(results) == 2
        assert {r["content"][-1] for r in results} == {"A", "B"}

    def test_stable_key_prefers_id_fields(self):
        assert _stable_result_key({"id": 7}) == "id:7"
        assert _stable_result_key({"chunk_id": "c1", "document_id": "d1"}) == "chunk_id:c1"
        assert _stable_result_key({"document_id": "d1"}) == "document_id:d1"

    def test_deterministic_ordering_on_ties(self, monkeypatch):
        doc_a = {"id": "b", "title": "B", "content": "内容B", "score": 0.5, "fused_score": 0.5}
        doc_b = {"id": "a", "title": "A", "content": "内容A", "score": 0.5, "fused_score": 0.5}
        fake = _FakeVectorDB([[doc_a, doc_b]])
        helper = _make_helper(monkeypatch, fake)
        results = helper.retrieve_context("任意查询", top_k=5, use_cache=False)

        assert [r["id"] for r in results] == ["a", "b"]


# ---------------------------------------------------------------------------
# CorrectiveRAG
# ---------------------------------------------------------------------------


class _StubRAGHelper:
    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = list(responses)
        self.calls: list[str] = []

    def retrieve_with_citations(
        self, query: str, top_k: int | None = None, threshold: float = 0.3, filters: dict[str, Any] | None = None
    ):
        self.calls.append(query)
        return self.responses.pop(0)


def _abstained_response(results=None, confidence=0.1):
    return {
        "results": results or [],
        "citations": [],
        "confidence": confidence,
        "abstained": True,
    }


def _success_response(confidence=0.8):
    return {
        "results": [{"id": "d1", "title": "文档", "content": "内容", "score": confidence}],
        "citations": [{"source_id": "d1"}],
        "confidence": confidence,
        "abstained": False,
    }


_KEYWORD_RESULTS = [
    {"title": "胡桃攻略", "content": "往生堂 堂主 火属性", "score": 0.2},
]


class TestCorrectiveRAG:
    def test_max_retries_zero_only_first_retrieval(self):
        stub = _StubRAGHelper([_abstained_response(_KEYWORD_RESULTS)])
        rag = CorrectiveRAG(stub, threshold=0.3, max_retries=0)

        result = rag.retrieve_with_correction("胡桃怎么配队")

        assert len(stub.calls) == 1
        assert stub.calls[0] == "胡桃怎么配队"
        assert result["abstained"] is True
        assert result["reformulated"] is False
        assert result["reformulated_query"] is None
        assert len(result["rounds"]) == 1
        assert result["rounds"][0] == {
            "query": "胡桃怎么配队",
            "confidence": 0.1,
            "abstained": True,
        }

    def test_default_max_retries_one_recovers(self):
        stub = _StubRAGHelper(
            [
                _abstained_response(_KEYWORD_RESULTS),
                _success_response(),
            ]
        )
        rag = CorrectiveRAG(stub, threshold=0.3, max_retries=1)

        result = rag.retrieve_with_correction("胡桃怎么配队")

        assert len(stub.calls) == 2
        assert stub.calls[0] == "胡桃怎么配队"
        assert stub.calls[1].startswith("胡桃怎么配队 ")
        # 中文关键词提取生效（不依赖具体分词粒度）
        assert "往生" in stub.calls[1]
        assert result["abstained"] is False
        assert result["reformulated"] is True
        assert result["original_query"] == "胡桃怎么配队"
        assert result["reformulated_query"] == stub.calls[1]
        assert len(result["rounds"]) == 2
        assert result["rounds"][0]["abstained"] is True
        assert result["rounds"][1]["abstained"] is False

    def test_noop_rewrite_skips_second_retrieval(self):
        # 首轮无结果 -> 提取不到关键词 -> 重写等于原查询 -> 不做第二次检索
        stub = _StubRAGHelper([_abstained_response([])])
        rag = CorrectiveRAG(stub, threshold=0.3, max_retries=1)

        result = rag.retrieve_with_correction("胡桃怎么配队")

        assert len(stub.calls) == 1
        assert result["abstained"] is True
        assert result["reformulated"] is False
        assert result["reformulated_query"] is None
        assert len(result["rounds"]) == 1

    def test_existing_keywords_are_not_appended_again(self):
        results = [{"title": "胡桃", "content": "胡桃 配队", "score": 0.1}]
        stub = _StubRAGHelper([_abstained_response(results)])
        rag = CorrectiveRAG(stub, threshold=0.3, max_retries=2)

        result = rag.retrieve_with_correction("胡桃 配队")

        assert len(stub.calls) == 1
        assert result["reformulated"] is False

    def test_retry_exhaustion_abstains(self):
        stub = _StubRAGHelper(
            [
                _abstained_response(_KEYWORD_RESULTS),
                _abstained_response(_KEYWORD_RESULTS, confidence=0.2),
            ]
        )
        rag = CorrectiveRAG(stub, threshold=0.3, max_retries=1)

        result = rag.retrieve_with_correction("胡桃怎么配队")

        assert len(stub.calls) == 2
        assert result["abstained"] is True
        assert result["results"] == []
        assert result["confidence"] == 0.2
        assert len(result["rounds"]) == 2

    def test_high_confidence_returns_directly(self):
        stub = _StubRAGHelper([_success_response()])
        rag = CorrectiveRAG(stub, threshold=0.3, max_retries=1)

        result = rag.retrieve_with_correction("胡桃怎么配队")

        assert len(stub.calls) == 1
        assert result["abstained"] is False
        assert result["reformulated"] is False
        assert result["rounds"][0]["query"] == "胡桃怎么配队"


class TestChineseTokenize:
    def test_jieba_tokenizer_extracts_chinese_words(self):
        from knowledge.retrieval_core.tokenization import segment

        tokens = segment("胡桃是往生堂堂主")
        # 中文被切成词（而非旧实现的单字，单字会被关键词过滤全部丢弃）
        assert any(len(t) >= 2 and any("\u4e00" <= ch <= "\u9fff" for ch in t) for t in tokens)
        assert all(t.strip() for t in tokens)
