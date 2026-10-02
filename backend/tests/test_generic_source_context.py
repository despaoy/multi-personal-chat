"""Only the source expansion, authority and final packet budget boundaries."""

import copy
from threading import RLock

import pytest

from inference.answer_citations import prepare_answer_citations
from inference.generation_request import GenerationRequest, RetrievalResult, build_generation_request
from knowledge.evidence_packets import document_evidence_packets
from knowledge.source_expansion import expand_source_context
from knowledge.vector_db import VectorDatabase


def row(index, content=None, *, parent=1, kb=7):
    return dict(
        id=f"doc_{parent}_chunk_{index}",
        document_id=parent,
        chunk_index=index,
        knowledge_base_id=kb,
        title=f"完整资料{parent}",
        category="规程",
        content=content or f"第{index}章必要条件",
    )


def vector(records):
    result = VectorDatabase.__new__(VectorDatabase)
    result._lock = RLock()
    result._cache_generation = 11
    result.snapshot_validated = True
    result.metadata = records
    return result


def expand(bundle, database, budget=65536, filters=None):
    return expand_source_context(bundle, database, expected_generation=11, source_budget_tokens=budget, filters=filters)


def test_six_complementary_chapters_keep_ranked_anchors_and_confidence():
    records = [row(i) for i in range(6)]
    anchors = [{**records[i], "score": 0.8, "normalized_score": 1.0} for i in [0, 2, 4]]
    bundle = dict(results=anchors, confidence=0.8, abstained=False)
    before = copy.deepcopy(bundle)
    result = expand(bundle, vector([*records, row(0, parent=2)]))
    assert bundle == before and result["confidence"] == 0.8
    assert result["results"][:3] == anchors
    assert [r["chunk_index"] for r in result["results"]] == [0, 2, 4, 1, 3, 5]
    assert all(
        r["score"] == 0 and r["supporting_document_ids"] == [a["id"] for a in anchors] for r in result["results"][3:]
    )


def test_scope_filter_and_parent_identity_exclude_other_sources():
    anchor = row(0)
    records = [anchor, row(1), row(2, kb=8), row(3, parent=2)]
    result = expand(dict(results=[anchor], abstained=False), vector(records), filters={"knowledge_base_id": 7})
    assert [r["id"] for r in result["results"]] == [anchor["id"], "doc_1_chunk_1"]


@pytest.mark.parametrize("field", ["content", "title"])
def test_stale_anchor_cannot_authorize_siblings(field):
    anchor = row(0)
    stale = {**anchor, field: "过期值"}
    with pytest.raises(RuntimeError, match="no longer matches"):
        expand(dict(results=[stale]), vector([anchor, row(1)]))


@pytest.mark.parametrize("invalid", ["generation", "unvalidated"])
def test_changed_or_unvalidated_snapshot_rejects_source_expansion(invalid):
    anchor = row(0)
    database = vector([anchor, row(1)])
    if invalid == "generation":
        database._cache_generation += 1
    else:
        database.snapshot_validated = False
    with pytest.raises(RuntimeError, match="index changed"):
        expand(dict(results=[anchor]), database)


def test_duplicate_index_identity_is_not_an_expansion_grant():
    anchor = row(0)
    with pytest.raises(RuntimeError, match="Ambiguous"):
        expand(dict(results=[anchor]), vector([anchor, row(1), row(1, "另一版本")]))


def test_oversized_sibling_does_not_remove_anchor_or_later_whole_context():
    anchor = row(0)
    result = expand(dict(results=[anchor]), vector([anchor, row(1, "大" * 10000), row(2, "最后限制")]), budget=100)
    assert result["results"][0] == anchor
    assert [r["chunk_index"] for r in result["results"]] == [0, 2]
    assert result["results"][1]["content"] == "最后限制"


def test_final_budget_never_admits_sibling_without_its_own_source_anchor():
    huge, other = row(0, "完整长章" * 3000), row(0, parent=2)
    result = expand(dict(results=[huge, other]), vector([huge, other, row(1, "同来源补充不能独立证明资料")]))
    packets = document_evidence_packets(result["results"])
    plan = build_generation_request(
        GenerationRequest(
            message="核对条件",
            context_window_tokens=8192,
            evidence_max_chars=0,
            retrieval=RetrievalResult(
                status="ok", evidence="\n".join(p["text"] for p in packets), evidence_packets=packets
            ),
        )
    )
    wire = plan.messages[-1]["content"]
    assert other["content"] in wire and "同来源补充不能独立证明资料" not in wire


def test_expanded_chunks_bind_their_own_citations_and_budget_dependencies():
    records = [row(i) for i in range(6)]
    result = expand(dict(results=[records[0]]), vector(records))
    packets = document_evidence_packets(result["results"])
    assert all(p["kind"] == "background" and p["supporting_document_ids"] == [records[0]["id"]] for p in packets[1:])
    citations = tuple({"source_id": r["id"]} for r in result["results"])
    bound = prepare_answer_citations(
        RetrievalResult(
            status="ok",
            evidence="\n".join(p["text"] for p in packets),
            evidence_packets=packets,
            documents=tuple(result["results"]),
            citations=citations,
        )
    )
    plan = build_generation_request(
        GenerationRequest(message="解释完整资料", context_window_tokens=65536, evidence_max_chars=0, retrieval=bound)
    )
    assert len(plan.retrieval.citations) == 6
    for record in records:
        assert record["content"] in plan.messages[-1]["content"]


def test_abstained_candidates_never_gain_source_support():
    bundle = dict(results=[row(0)], abstained=True, confidence=0.1)
    assert expand(bundle, vector([row(0), row(1)])) is bundle


async def test_database_revision_change_during_retrieval_rejects_expanded_evidence(monkeypatch):
    from api import generate, knowledge
    from knowledge import rag_helper, vector_db

    anchor = row(0)
    database = vector([anchor, row(1)])
    revision = [77]
    monkeypatch.setattr(knowledge, "_ensure_vector_index", lambda: True)
    monkeypatch.setattr(knowledge, "_vector_index_revision", 77)
    monkeypatch.setattr(knowledge, "_get_rebuild_revision", lambda: revision[0])
    monkeypatch.setattr(vector_db, "get_vector_db", lambda: database)

    def retrieve(*args, **kwargs):
        revision[0] += 1
        return dict(results=[anchor], confidence=0.8, abstained=False)

    class Helper:
        retrieve_with_citations = staticmethod(retrieve)

    monkeypatch.setattr(rag_helper, "get_rag_helper", Helper)
    monkeypatch.setenv("CORRECTIVE_RAG_ENABLED", "false")
    with pytest.raises(RuntimeError, match="authority changed"):
        await generate._retrieve_rag_bundle("完整规程", 3, {"knowledge_base_id": 7})
