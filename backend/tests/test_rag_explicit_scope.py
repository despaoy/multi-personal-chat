"""Explicit generic scope with real FAISS, BM25, task retention and cache.

Complete synthetic documents and controlled embeddings isolate query/scope
behavior. No paid model, network or production data is used.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from knowledge import rag_helper
from knowledge.vector_db import IndexConfig, VectorDatabase

QUERY = "背景提及原神璃月角色，请比较竹湾登记与枫台续期的材料和费用。"
VIEWS = ("竹湾登记 材料 费用", "枫台续期 材料 费用")
DOCUMENTS = (
    dict(id="bamboo", title="竹湾登记规程", category="规程", region="竹湾", knowledge_base_id=7,
         content="竹湾登记：周二收取12元，须持申请卡原件；节假日不受理，预约不能免除原件。"),
    dict(id="maple", title="枫台续期规程", category="规程", region="枫台", knowledge_base_id=7,
         content="枫台续期：周四收取18元，须持确认函原件；节假日不受理，预约不能免除原件。"),
    dict(id="game", title="原神璃月角色背景", category="角色", region="璃月", knowledge_base_id=9,
         content="这是独立作品角色的背景资料，不提供登记或续期的公共规程。"),
)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.setenv("RERANKER_ENABLED", "false")
    monkeypatch.setattr(VectorDatabase, "_check_gpu_availability", lambda _: False)
    vector = VectorDatabase(str(tmp_path / "vectors"), IndexConfig(index_type="flat"))

    def encode(texts, **_kwargs):
        rows = []
        for text in texts:
            row = np.zeros(384, dtype=np.float32)
            if "竹湾" in text:
                row[0] = 1
            if "枫台" in text:
                row[1] = 1
            if not row.any():
                row[2] = 1
            rows.append(row / np.linalg.norm(row))
        return np.asarray(rows, dtype=np.float32)

    vector._model = SimpleNamespace(encode=encode)
    vector.add_documents([dict(document) for document in DOCUMENTS])
    search = Mock(wraps=vector.hybrid_search)
    monkeypatch.setattr(vector, "hybrid_search", search)
    monkeypatch.setattr(rag_helper, "get_vector_db", lambda: vector)
    return SimpleNamespace(helper=rag_helper.RAGHelper(), search=search)


def test_incidental_game_terms_cannot_rewrite_query_or_filter_public_evidence(rig):
    rows = rig.helper.retrieve_context(QUERY, enable_rerank=False)
    assert {row["id"] for row in rows} >= {"bamboo", "maple"}
    assert {row["content"] for row in rows} >= {document["content"] for document in DOCUMENTS[:2]}
    rig.search.assert_called_once_with(QUERY, top_k=20, threshold=0.05, keyword_weight=0.3, filters=None)
    assert all("query_count" not in row and "region_boost" not in row for row in rows)
    assert all(row["final_score"] == row["fused_score"] for row in rows)


def test_validated_model_views_keep_both_complete_sources_and_scope_in_cache(rig):
    scope = {"knowledge_base_id": 7}
    result = rig.helper.retrieve_with_citations(QUERY, top_k=1, filters=scope, additional_queries=VIEWS)
    assert {row["id"] for row in result["results"]} == {"bamboo", "maple"}
    assert {row["content"] for row in result["results"]} == {document["content"] for document in DOCUMENTS[:2]}
    assert not result["abstained"]
    assert [call.args[0] for call in rig.search.call_args_list] == [QUERY, *VIEWS]
    assert all(call.kwargs["filters"] == scope for call in rig.search.call_args_list)
    assert [bucket["query"] for bucket in result["task_candidate_coverage"]] == [QUERY, *VIEWS]
    result["results"][0]["content"] = "caller mutation"
    cached = rig.helper.retrieve_with_citations(QUERY, top_k=1, filters=scope, additional_queries=VIEWS)
    assert {row["content"] for row in cached["results"]} == {document["content"] for document in DOCUMENTS[:2]}
    assert rig.search.call_count == 3


def test_explicit_region_is_an_index_filter_instead_of_keyword_weight(rig):
    scope = {"knowledge_base_id": 7, "region": "竹湾"}
    rows = rig.helper.retrieve_context(QUERY, filters=scope, enable_rerank=False)
    assert [row["id"] for row in rows] == ["bamboo"]
    assert rows[0]["content"] == DOCUMENTS[0]["content"]
    assert rig.search.call_args.kwargs["filters"] == scope


def test_empty_explicit_scope_is_normal_abstention_and_cannot_reuse_other_scope(rig):
    assert rig.helper.retrieve_with_citations(QUERY)["results"]
    result = rig.helper.retrieve_with_citations(QUERY, filters={"knowledge_base_id": 77})
    assert result["results"] == result["citations"] == [] and result["abstained"]
    assert rig.search.call_count == 2
