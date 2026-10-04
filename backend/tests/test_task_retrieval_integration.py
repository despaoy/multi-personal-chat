"""Bucket integration preserves original scope, rank inputs and cache isolation."""

import pytest

from knowledge import rag_helper

ROOT = "青川规则、蓝溪规则、银叶规则、白石规则和我的原话一起核对"
VIEWS = ("青川规则", "蓝溪规则", "银叶规则", "白石规则")


class Index:
    cache_generation = 1

    def __init__(self):
        self.calls = []

    def hybrid_search(self, q, **kwargs):
        self.calls.append((q, kwargs["filters"]))
        key = q if q in VIEWS else "root"
        return [dict(id=key, title=key, content="complete " + key, score=0.8)]


def prepared(monkeypatch):
    index = Index()
    monkeypatch.setattr(rag_helper, "get_vector_db", lambda: index)
    helper = rag_helper.RAGHelper()
    helper.use_vector_db = True
    helper.enable_query_expansion = False
    return helper, index


def test_each_task_retained_and_all_views_have_independent_receipts(monkeypatch):
    h, i = prepared(monkeypatch)
    b = h.retrieve_with_citations(ROOT, top_k=1, additional_queries=VIEWS)
    assert {x["id"] for x in b["results"]} == {"root", *VIEWS}
    assert len(b["task_candidate_coverage"]) == 5
    assert all(x["semantic_coverage"] == "unverified" for x in b["task_candidate_coverage"])
    assert not b["abstained"]


def test_only_whole_original_question_can_infer_filters(monkeypatch):
    h, i = prepared(monkeypatch)
    h.enable_query_expansion = True
    h.query_expander.expand_query = lambda q: [q]
    seen = []

    def infer(q):
        seen.append(q)
        assert q == ROOT
        return {}

    h.query_expander.extract_filters = infer
    h.retrieve_context(ROOT, top_k=1, additional_queries=VIEWS, enable_rerank=False)
    assert seen == [ROOT] and all(f is None for q, f in i.calls)


def test_explicit_scope_is_identical_in_every_bucket(monkeypatch):
    h, i = prepared(monkeypatch)
    h.retrieve_context(ROOT, top_k=1, additional_queries=VIEWS, filters={"knowledge_base_id": 7}, enable_rerank=False)
    assert len(i.calls) == 5 and all(f == {"knowledge_base_id": 7} for q, f in i.calls)


def test_reranking_is_bound_to_each_task_and_does_not_globally_truncate(monkeypatch):
    h, i = prepared(monkeypatch)
    seen = []
    old = i.hybrid_search

    def search(q, **kw):
        return old(q, **kw) + [dict(id="common", title="common", content="same", score=0.7)]

    i.hybrid_search = search

    class Ranker:
        def rerank(self, q, rows, top_k):
            seen.append(q)
            return rows[:top_k]

    h.reranker = Ranker()
    h.enable_reranking = True
    b = h.retrieve_context(ROOT, top_k=1, additional_queries=VIEWS)
    assert seen == [ROOT, *VIEWS] and {x["id"] for x in b} == {"root", *VIEWS}


def test_changed_task_views_do_not_reuse_a_different_task_plan(monkeypatch):
    h, i = prepared(monkeypatch)
    a = h.retrieve_context(ROOT, top_k=1, additional_queries=VIEWS[:1], enable_rerank=False)
    b = h.retrieve_context(ROOT, top_k=1, additional_queries=VIEWS[1:2], enable_rerank=False)
    assert {x["id"] for x in a} == {"root", VIEWS[0]}
    assert {x["id"] for x in b} == {"root", VIEWS[1]}
    n = len(i.calls)
    h.retrieve_context(ROOT, top_k=1, additional_queries=VIEWS[1:2], enable_rerank=False)
    assert len(i.calls) == n


def test_index_object_replaced_mid_bucket_is_not_same_snapshot(monkeypatch):
    h, i = prepared(monkeypatch)
    original = h.retrieve_context

    def retrieve(q, **kw):
        rows = original(q, **kw)
        if q == VIEWS[0]:
            monkeypatch.setattr(rag_helper, "get_vector_db", lambda: Index())
        return rows

    h.retrieve_context = retrieve
    with pytest.raises(RuntimeError, match="Index changed"):
        h.retrieve_with_citations(ROOT, top_k=1, additional_queries=VIEWS)
