"""Complete named-source reads must survive incidental domain vocabulary."""

import json
from pathlib import Path

import pytest

from knowledge import rag_helper
from knowledge.rag_helper import QueryExpander

CASE = json.loads(
    (Path(__file__).parent / "fixtures/deepseek_named_source_binding_case.json").read_text(encoding="utf-8")
)


class ScopedIndex:
    cache_generation = 1

    def __init__(self):
        self.calls = []
        self.documents = [
            {
                **document,
                "id": f"doc_{504 + i}_chunk_0",
                "document_id": 504 + i,
                "chunk_index": 0,
                "knowledge_base_id": 3,
                "score": 0.9,
                "fused_score": 0.9,
            }
            for i, document in enumerate(CASE["documents"])
        ]

    def hybrid_search(self, query, *, filters=None, **kwargs):
        self.calls.append((query, dict(filters or {})))
        return [
            dict(document)
            for document in self.documents
            if all(document.get(key) == value for key, value in (filters or {}).items())
        ]


def helper_for(monkeypatch):
    index = ScopedIndex()
    monkeypatch.setattr(rag_helper, "get_vector_db", lambda: index)
    helper = rag_helper.RAGHelper()
    helper.enable_reranking = False
    return helper, index


def test_complete_failed_native_question_keeps_all_named_originals(monkeypatch):
    helper, index = helper_for(monkeypatch)
    result = helper.retrieve_with_citations(CASE["question"], top_k=4)
    assert not result["abstained"]
    assert {item["content"] for item in result["results"]} == {item["content"] for item in CASE["documents"]}
    assert index.calls[0][0] == CASE["question"]
    assert all(not filters for _, filters in index.calls)
    assert CASE["question"].startswith(CASE["original_question"] + "\n")


@pytest.mark.parametrize(
    "filters, expected",
    [
        ({"knowledge_base_id": 3}, 4),
        ({"knowledge_base_id": 77}, 0),
        ({"knowledge_base_id": 3, "category": "角色"}, 0),
        ({"knowledge_base_id": 3, "category": "办理"}, 4),
    ],
)
def test_explicit_scope_still_controls_named_source_read(monkeypatch, filters, expected):
    helper, index = helper_for(monkeypatch)
    result = helper.retrieve_with_citations(CASE["question"], top_k=4, filters=filters)
    assert len(result["results"]) == expected
    assert all(actual == filters for _, actual in index.calls)
    assert result["abstained"] is (expected == 0)


def test_cache_cannot_reuse_unrestricted_named_read_for_other_scope(monkeypatch):
    helper, index = helper_for(monkeypatch)
    first = helper.retrieve_with_citations(CASE["question"], top_k=4)
    second = helper.retrieve_with_citations(CASE["question"], top_k=4, filters={"knowledge_base_id": 77})
    assert len(first["results"]) == 4
    assert not second["results"] and second["abstained"]
    assert any(filters == {"knowledge_base_id": 77} for _, filters in index.calls)


@pytest.mark.parametrize(
    "query",
    [
        "角色有哪些技能？",
        "背景提到《绿泽窗口受理》，另说明角色有哪些技能。",
        "请查知识库，逐项比较《绿泽窗口受理》这四份说明。另说明角色有哪些技能。",
        "请查知识库，逐项比较《绿泽窗口受理》这份说明。但不要读取原文，另说明角色有哪些技能。",
    ],
)
def test_unresolved_named_read_does_not_grant_a_game_category(query):
    assert QueryExpander().extract_filters(query) == {}


def test_explicit_title_containing_category_word_is_source_data(monkeypatch):
    helper, index = helper_for(monkeypatch)
    index.documents[0]["title"] = "角色自身偏好记录来源"
    query = "请查知识库，逐项比较《角色自身偏好记录来源》这一份说明，分别核对全文。"
    result = helper.retrieve_context(query, top_k=4, use_cache=False)
    assert any(item["id"] == "doc_504_chunk_0" for item in result)
    assert all(not filters for _, filters in index.calls)
