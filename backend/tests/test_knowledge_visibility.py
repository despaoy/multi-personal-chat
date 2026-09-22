from types import SimpleNamespace

import pytest

from knowledge.multiscale_rag import service
from knowledge.multiscale_rag.visibility import KnowledgeBoundary, visible_to
from knowledge.retrieval_core.documents import KnowledgeIndexDocument
from knowledge.retrieval_core.retrieval import RetrievalCandidate


def _doc(key, *, start=0, end=None, character="kisaki", continuity="canonical", kind="relation"):
    return KnowledgeIndexDocument(
        key,
        "test",
        kind,
        key,
        f"summary-{key}",
        key,
        key,
        entities=["kisaki", "ruri"],
        metadata={
            "knowledge_access": {
                "version": 1,
                "continuity_id": continuity,
                "known_by": {character: {"from_scene": start, "until_scene": end}},
            }
        },
    )


def test_visibility_uses_explicit_knowledge_not_entity_occurrence():
    boundary = KnowledgeBoundary("kisaki", 10)
    assert visible_to(_doc("known", start=10), boundary)
    assert not visible_to(_doc("future", start=11), boundary)
    assert not visible_to(_doc("forgotten", start=0, end=10), boundary)
    assert not visible_to(_doc("other", character="ruri"), boundary)
    assert not visible_to(_doc("branch", continuity="alternate"), boundary)
    doc = _doc("unknown")
    doc.metadata = {}
    assert not visible_to(doc, boundary)
    assert visible_to(doc, None)


@pytest.mark.parametrize("start,end", [(True, None), (-1, None), ("0", None), (0, False), (2, 1)])
def test_malformed_windows_fail_closed(start, end):
    assert not visible_to(_doc("invalid", start=start, end=end), KnowledgeBoundary("kisaki", 10))


def test_pending_annotation_is_not_authority():
    doc = _doc("pending")
    doc.review_status = "pending"
    assert not visible_to(doc, KnowledgeBoundary("kisaki", 10))


def test_boundary_filters_parents_timelines_and_raw_evidence_before_exposure(monkeypatch):
    known, future = _doc("known"), _doc("future", start=50)
    scene = _doc("secret-scene", start=50, kind="scene")
    evidence = _doc("secret-evidence", start=50, kind="evidence")
    known.metadata["scene_id"] = scene.id
    index = SimpleNamespace(documents=[known, future], count=lambda: 100)
    recalls = []

    def search(analysis, *, top_k, recall_k, mode):
        recalls.append(recall_k)
        return [RetrievalCandidate(0, future), RetrievalCandidate(1, known)]

    instance = object.__new__(service.RoutedMultiScaleService)
    instance.config = SimpleNamespace(domain_id="test")
    instance.analyzer = None
    instance.indexes = {frozenset({"relation"}): index}
    instance.retrievers = {frozenset({"relation"}): SimpleNamespace(search=search, index=index)}
    instance.reranker = None
    instance.by_id = {scene.id: scene}
    instance.evidence_by_parent = {known.id: evidence}
    instance.extractor = SimpleNamespace(extract=lambda _: pytest.fail("hidden evidence must not be read"))
    monkeypatch.setattr(service, "analyze_explicit_domain", lambda *args: SimpleNamespace(entities=["kisaki", "ruri"]))
    monkeypatch.setattr(service, "choose_card_types", lambda *args: frozenset({"relation"}))

    def rerank(analysis, candidates, **kwargs):
        assert [item.document.id for item in candidates] == ["known"]
        return candidates

    monkeypatch.setattr(service, "rerank_with_title_frames", rerank)
    result = instance.retrieve("关系和原文", top_k=1, knowledge_boundary=KnowledgeBoundary("kisaki", 10))
    assert recalls == [100]
    assert "secret" not in result["context_text"]
    assert "future" not in str(result)
    assert result["raw_excerpt"] is None
    assert [item["id"] for item in result["relation_timeline"]] == ["known"]
    assert result["knowledge_boundary_applied"]


@pytest.mark.parametrize("position", [True, -1, "10", 1.5])
def test_boundary_positions_are_not_coerced(position):
    with pytest.raises(ValueError):
        KnowledgeBoundary("kisaki", position)
