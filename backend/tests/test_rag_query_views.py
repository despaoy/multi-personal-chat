"""Additional literal search views preserve scope, original tasks and cache identity."""
import pytest

from knowledge import rag_helper


class Index:
    cache_generation = 1
    def __init__(self):
        self.calls = []
    def hybrid_search(self, query, **kwargs):
        self.calls.append((query, kwargs["filters"]))
        key = "rule" if query == "青川通道 费用 时长" else "context"
        return [dict(id=key, title=key, content="complete-" + key, score=.8, document_id=1 if key == "rule" else 2)]


def helper_for(monkeypatch):
    index = Index()
    monkeypatch.setattr(rag_helper, "get_vector_db", lambda: index)
    helper = rag_helper.RAGHelper()
    helper.enable_query_expansion = False
    return helper, index


def test_original_and_focus_views_both_retrieve_without_new_scope(monkeypatch):
    helper, index = helper_for(monkeypatch)
    query = "青川通道按完整资料比较费用和时长，角色不是用户。"
    rows = helper.retrieve_context(query, additional_queries=("青川通道 费用 时长",), enable_rerank=False)
    assert {x["id"] for x in rows} == {"context", "rule"}
    assert [x[0] for x in index.calls] == [query, "青川通道 费用 时长"]
    assert all(x[1] is None for x in index.calls)


def test_explicit_filters_apply_to_every_view(monkeypatch):
    helper, index = helper_for(monkeypatch)
    helper.retrieve_context("青川通道 费用 时长", additional_queries=("青川通道 时长",), filters={"knowledge_base_id":7}, enable_rerank=False)
    assert index.calls and all(x[1] == {"knowledge_base_id":7} for x in index.calls)


def test_query_plan_is_part_of_cache_identity(monkeypatch):
    helper, index = helper_for(monkeypatch)
    query = "青川通道费用时长"
    assert {x["id"] for x in helper.retrieve_context(query, enable_rerank=False)} == {"context"}
    assert {x["id"] for x in helper.retrieve_context(query, additional_queries=("青川通道 费用 时长",), enable_rerank=False)} == {"context", "rule"}
    count = len(index.calls)
    helper.retrieve_context(query, additional_queries=("青川通道 费用 时长",), enable_rerank=False)
    assert len(index.calls) == count


@pytest.mark.parametrize("bad", [("invented",), (True,), tuple("青川通道" for _ in range(5))])
def test_internal_views_cannot_add_outside_query_text_or_unbounded_work(monkeypatch, bad):
    helper, index = helper_for(monkeypatch)
    with pytest.raises(ValueError):
        helper.retrieve_context("青川通道费用时长", additional_queries=bad)
    assert not index.calls
