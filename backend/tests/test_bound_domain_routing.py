"""A proven source scope cannot be replaced by a second keyword decision."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from test_multiscale_card_recall import ControlledRanking, arrange

from api import generate
from api import knowledge as knowledge_api
from knowledge.multiscale_rag import runtime as runtime_module
from knowledge.multiscale_rag.runtime import MultiScaleRagRuntime, _retrieval_index
from knowledge.multiscale_rag.service import RoutedMultiScaleService
from knowledge.public_question_binding import resolve_question_binding
from knowledge.retrieval_core.query import QueryAnalysis, QueryAnalyzer
from knowledge.turn_dependencies import parse_dependencies, query_segments

QUERY = "请核对原作角色岚的公开登记记录，保留否定和末尾例外。"
BODY = "林栖登记为虚构河湾馆的档案员，只受理已预约的资料复核，不办理新证签发；雨天停办，已预约也不豁免。"


class GenericRouteReached(Exception):
    """A route-only control; it never claims successful generic retrieval."""


def setup_scope(tmp_path, monkeypatch):
    rig = arrange(tmp_path)
    rig.config.aliases["岚"] = "林栖"
    for doc in rig.documents:
        if doc.id in {"fact", "source-fact"}:
            doc.summary = doc.content = doc.embedding_text = BODY
    (tmp_path / "fact.txt").write_text(BODY, encoding="utf-8")
    vectors = np.ones((len(rig.documents), 384), dtype=np.float32) / np.sqrt(384)
    service = RoutedMultiScaleService(
        rig.config,
        _retrieval_index(rig.config.domain_id, rig.documents, vectors),
        rig.embedding,
        all_documents=rig.documents,
        source_extractor=rig.service.extractor,
        reranker=ControlledRanking(["fact", "relation", "event", "scene"]),
    )
    runtime = object.__new__(MultiScaleRagRuntime)
    runtime._base_config = rig.config
    runtime._gate = QueryAnalyzer([rig.config])
    runtime._provider = rig.embedding
    runtime._service = service
    monkeypatch.setattr(runtime_module, "get_multiscale_rag_service", lambda: runtime)
    generic = Mock(side_effect=GenericRouteReached("Generic route selected"))
    monkeypatch.setattr(knowledge_api, "_ensure_vector_index", generic)
    return SimpleNamespace(
        runtime=runtime, service=service, config=rig.config, embedding=rig.embedding, generic=generic
    )


async def bound_scope(query=QUERY, names=("岚",)):
    deps = parse_dependencies(
        dict(
            private_memory=[],
            current_input=[],
            control=[],
            unresolved_source=[],
            public_knowledge=list(range(len(query_segments(query)))),
        ),
        query,
    )

    async def reviewer(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload["query"] == query
        return json.dumps(
            dict(scopes=[dict(task_id=task["id"], objects=list(names)) for task in payload["public_tasks"]])
        )

    return await resolve_question_binding(deps, query, window_tokens=65536, reviewer=reviewer)


async def test_proven_single_character_scope_uses_complete_curated_original(tmp_path, monkeypatch):
    rig = setup_scope(tmp_path, monkeypatch)
    result = await generate._retrieve_rag_bundle(QUERY, 1, None, question_binding=await bound_scope())
    assert result["domains"] == [rig.config.domain_id]
    assert result["curated_source_catalog"]["sources"][0]["excerpt"]["text"] == BODY
    assert result["query_analysis"]["matched_domains"] == [rig.config.domain_id]
    rig.generic.assert_not_called()


@pytest.mark.parametrize("defect", ["vector", "declined"])
async def test_required_curated_failure_never_changes_authority(tmp_path, monkeypatch, defect):
    rig = setup_scope(tmp_path, monkeypatch)
    if defect == "vector":
        failure = OSError("Synthetic required embedding unavailable")
        rig.embedding.embed_query = Mock(side_effect=failure)
        expected = OSError
    else:
        monkeypatch.setattr(rig.runtime, "retrieve_with_citations", lambda *args, **kwargs: None)
        expected = RuntimeError
    with pytest.raises(expected):
        await generate._retrieve_rag_bundle(QUERY, 1, None, question_binding=await bound_scope())
    rig.generic.assert_not_called()


async def test_whole_business_name_is_not_its_embedded_character_alias(tmp_path, monkeypatch):
    rig = setup_scope(tmp_path, monkeypatch)
    monkeypatch.setattr(
        rig.runtime, "retrieve_with_citations", Mock(side_effect=AssertionError("Curated source not requested"))
    )
    query = "核对岚展期的公开办理规则。"
    with pytest.raises(GenericRouteReached):
        await generate._retrieve_rag_bundle(query, 1, None, question_binding=await bound_scope(query, ("岚展期",)))
    rig.runtime.retrieve_with_citations.assert_not_called()
    rig.generic.assert_called_once()


def test_explicit_domain_does_not_need_a_legacy_question_word(tmp_path, monkeypatch):
    rig = setup_scope(tmp_path, monkeypatch)
    analysis = rig.runtime._gate.analyze(QUERY, domain_id=rig.config.domain_id)
    assert analysis.matched_domains == [rig.config.domain_id]
    assert analysis.entities == ["林栖"]
    assert analysis.original_query == QUERY and analysis.normalized_query == QUERY.replace("岚", "林栖")


def test_unknown_explicit_domain_never_searches_other_domains(tmp_path, monkeypatch):
    rig = setup_scope(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="domain"):
        rig.runtime._gate.analyze("林栖的完整登记资料", domain_id="missing-domain")


@pytest.mark.parametrize("defect", ["query", "domain", "multiple_domains"])
def test_precomputed_analysis_must_belong_to_current_request(tmp_path, monkeypatch, defect):
    rig = setup_scope(tmp_path, monkeypatch)
    analysis = QueryAnalysis(
        original_query="changed" if defect == "query" else QUERY,
        normalized_query=QUERY,
        entities=["林栖"],
        matched_domains=(
            ["other"]
            if defect == "domain"
            else [rig.config.domain_id, "other"]
            if defect == "multiple_domains"
            else [rig.config.domain_id]
        ),
    )
    with pytest.raises(ValueError, match="analysis"):
        rig.service.retrieve(QUERY, top_k=1, analysis=analysis)
    assert rig.embedding.calls == []


def test_object_scope_reuses_but_does_not_mutate_literal_analysis(tmp_path, monkeypatch):
    rig = setup_scope(tmp_path, monkeypatch)
    query = QUERY + "不要把穆远当成本次登记对象。"
    analysis = rig.runtime._gate.analyze(query, domain_id=rig.config.domain_id)
    original_entities = list(analysis.entities)
    result = rig.service.retrieve(query, top_k=1, analysis=analysis, object_names=("岚",))
    assert result["curated_source_catalog"]["sources"][0]["excerpt"]["text"] == BODY
    assert analysis.original_query == query and analysis.entities == original_entities
    assert set(original_entities) == {"林栖", "穆远"}


def test_ambiguous_automatic_domains_keep_the_original_query(tmp_path):
    from knowledge.retrieval_core.registry import KnowledgeDomainConfig

    configs = [
        KnowledgeDomainConfig(domain, tmp_path, lambda _: [], aliases={"小林": canonical})
        for domain, canonical in [("first", "林远"), ("second", "林宁")]
    ]
    query = "小林的完整登记资料"
    analysis = QueryAnalyzer(configs).analyze(query)
    assert set(analysis.matched_domains) == {"first", "second"}
    assert analysis.original_query == analysis.normalized_query == query


async def test_invalid_curated_container_never_changes_authority(tmp_path, monkeypatch):
    rig = setup_scope(tmp_path, monkeypatch)
    monkeypatch.setattr(rig.runtime, "retrieve_with_citations", lambda *args, **kwargs: [])
    with pytest.raises(TypeError, match="expected a dictionary"):
        await generate._retrieve_rag_bundle(QUERY, 1, None, question_binding=await bound_scope())
    rig.generic.assert_not_called()


async def test_healthy_empty_curated_index_is_ordinary_abstention(tmp_path, monkeypatch):
    from knowledge.multiscale_rag.runtime import _memory_index

    rig = setup_scope(tmp_path, monkeypatch)
    rig.service.retriever.index = _memory_index(rig.config.domain_id, [], np.zeros((0, 384), dtype=np.float32))
    result = await generate._retrieve_rag_bundle(QUERY, 1, None, question_binding=await bound_scope())
    assert result["abstained"] and result["results"] == []
    assert result["domains"] == [rig.config.domain_id]
    assert rig.embedding.calls == []
    rig.generic.assert_not_called()


async def test_unresolved_mixed_task_searches_available_curated_domain_without_inventing_object(tmp_path, monkeypatch):
    from knowledge.public_domains import domain_plan, retrieve_domains

    rig = setup_scope(tmp_path, monkeypatch)
    query = "有一项原作对象还未指明，先核对登记记录。核对岚展期的办理规则。"
    deps = parse_dependencies(
        dict(private_memory=[], current_input=[], control=[], unresolved_source=[], public_knowledge=[0, 1]), query
    )

    async def review(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload["query"] == query
        return json.dumps(dict(scopes=[dict(task_id=0, objects=[]), dict(task_id=1, objects=["岚展期"])]))

    binding = await resolve_question_binding(deps, query, window_tokens=65536, reviewer=review)
    plan = domain_plan(binding, rig.config)
    result = retrieve_domains(
        plan,
        query,
        1,
        curated=rig.runtime.retrieve_with_citations,
        generic=lambda: dict(results=[dict(id="business", content="岚展期不免原件；雨天停办，预约也不豁免。")]),
    )
    assert result["independent_domains"]["plan"]["tasks"][0]["object_ids"] == []
    assert result["independent_domains"]["branches"]["curated_character"]["bundle"]["domains"] == [rig.config.domain_id]
    assert result["results"][0]["content"] == "岚展期不免原件；雨天停办，预约也不豁免。"


async def test_history_bound_single_character_scope_keeps_current_original(tmp_path, monkeypatch):
    from knowledge.public_question_binding import bound_retrieval_query

    rig = setup_scope(tmp_path, monkeypatch)
    query = "核对该角色的登记记录，保留否定及末尾例外。"
    history = [dict(role="user", content="原作角色岚是本次核对对象，请只使用公开资料。")]
    deps = parse_dependencies(
        dict(
            private_memory=[],
            current_input=[],
            control=[],
            unresolved_source=[],
            public_knowledge=list(range(len(query_segments(query)))),
        ),
        query,
    )

    async def review(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload["query"] == query and "reference_context" in payload
        return json.dumps(dict(scopes=[dict(task_id=task["id"], objects=["岚"]) for task in payload["public_tasks"]]))

    binding = await resolve_question_binding(deps, query, window_tokens=65536, history=history, reviewer=review)
    retrieval_query = bound_retrieval_query(binding, query)
    assert retrieval_query == "岚\n" + query
    result = await generate._retrieve_rag_bundle(retrieval_query, 1, None, question_binding=binding)
    assert result["curated_source_catalog"]["sources"][0]["excerpt"]["text"] == BODY
    assert binding["input"]["query"] == query
    rig.generic.assert_not_called()
