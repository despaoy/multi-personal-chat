"""Duplicate text shares provenance once and cannot crowd out another condition."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from inference.generation_request import GenerationRequest, RetrievalResult, build_generation_request
from inference.request_context import InferenceRequestContext, use_request_context
from knowledge.multiscale_rag import service
from knowledge.multiscale_rag.source_text import OriginalTextExtractor
from knowledge.retrieval_core.documents import KnowledgeIndexDocument, SourceReference
from knowledge.retrieval_core.registry import KnowledgeDomainConfig
from knowledge.retrieval_core.retrieval import RetrievalCandidate

OPEN = "青屿馆周二09:00至17:00只对预约观众开放，仍需核验原件。"
LIMIT = "暴雨全馆暂停，预约不豁免暂停限制；周一不开放，普通雨天仅关闭室外展区，室内展区照常开放。"
BACKGROUND = "这是同一份虚构布景说明，不能替代开放时段和天气限制。"
QUERY = "核对虚构青屿馆周二的开放时间、预约前提，以及周一和雨天暴雨的范围与例外。"


@pytest.mark.parametrize("mode", ["cloud", "limited", "raw"])
def test_unique_packet_text_preserves_all_sources_and_independent_conditions(tmp_path, monkeypatch, mode):
    docs = []
    bodies = {}
    for identity, title, body, parent in [
        ("a", "开放时间", OPEN, "background-a"),
        ("b", "开放时间", OPEN, "background-b"),
        ("c", "天气限制", LIMIT, None),
        ("background-a", "共同布景", BACKGROUND, None),
        ("background-b", "共同布景", BACKGROUND, None),
    ]:
        filename = identity + ".txt"
        (tmp_path / filename).write_text(body, encoding="utf-8")
        bodies[filename] = body
        docs.append(
            KnowledgeIndexDocument(
                id=identity,
                domain_id="synthetic",
                document_type="scene",
                title=title,
                summary=body,
                content=body,
                embedding_text=body,
                source=SourceReference(filename, 1, 1),
                metadata={"scene_id": parent} if parent else {},
            )
        )
    before = deepcopy([doc.to_dict() for doc in docs])
    candidates = [RetrievalCandidate(row=i, document=doc) for i, doc in enumerate(docs[:3])]
    config = KnowledgeDomainConfig("synthetic", tmp_path, lambda _: [], document_types=["scene"])
    instance = service.RoutedMultiScaleService(
        config,
        SimpleNamespace(documents=docs),
        object(),
        all_documents=docs,
        source_extractor=OriginalTextExtractor(tmp_path),
        reranker=SimpleNamespace(text_view="content"),
        context_max_chars=None,
    )
    # Ranking is fixed to isolate assembly. Source extraction, visibility,
    # service construction, budgeting and final generation construction are real.
    monkeypatch.setattr(instance.retriever, "search", lambda *a, **kw: candidates)
    monkeypatch.setattr(service, "rerank_with_title_frames", lambda *a, **kw: candidates)
    result = instance.retrieve(QUERY, top_k=3, raw_text=mode == "raw")
    if mode == "limited":
        facts = list(dict.fromkeys(p["text"] for p in result["evidence_packets"] if p["kind"] == "evidence"))
        assert len(facts) == 2
        # Enough space for both distinct conditions, independent of formatting.
        # A repeated summary must not consume this allowance a second time.
        instance.context_max_chars = sum(map(len, facts)) + 2
        result = instance.retrieve(QUERY, top_k=3)
    packets = result["evidence_packets"]
    texts = [p["text"] for p in packets]
    assert len(texts) == len(set(texts))
    assert OPEN in result["context_text"] and LIMIT in result["context_text"]
    assert {value for p in packets for value in p["document_ids"]} == {"a", "b", "c"}
    assert [c["id"] for c in result["citations"]] == ["a", "b", "c"]
    assert result["context_budget"]["admitted_ids"] == ["a", "b", "c"]
    catalog = result["curated_source_catalog"]
    assert catalog["unavailable"] == [] and [s["source_id"] for s in catalog["sources"]] == ["a", "b", "c"]
    assert [s["excerpt"]["text"] for s in catalog["sources"]] == [OPEN, OPEN, LIMIT]
    if mode == "limited":
        assert result["context_budget"]["skipped_blocks"] == 1
        assert not any(p["kind"] == "background" for p in packets)
    else:
        (background,) = (p for p in packets if p["kind"] == "background")
        assert background["supporting_document_ids"] == ["a", "b"]
        assert result["context_budget"]["skipped_blocks"] == 0
    if mode == "raw":
        assert texts[0].startswith("【原文摘录】") and result["raw_excerpt"]["source_path"] == "a.txt"
        assert result["raw_excerpt"]["text"] == OPEN and result["raw_source_status"] == "available"
    with use_request_context(InferenceRequestContext("vllm", None, len)):
        plan = build_generation_request(
            GenerationRequest(
                message=QUERY,
                apply_prompt_policy=False,
                context_window_tokens=65536,
                max_tokens=128,
                evidence_max_chars=0,
                retrieval=RetrievalResult(
                    status="ok", evidence=result["context_text"], evidence_packets=tuple(packets)
                ),
            )
        )
    assert OPEN in plan.messages[-1]["content"] and LIMIT in plan.messages[-1]["content"]
    assert list(plan.retrieval.evidence_packets) == packets
    assert [doc.to_dict() for doc in docs] == before
    assert {name: (tmp_path / name).read_text(encoding="utf-8") for name in bodies} == bodies
