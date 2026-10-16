"""Generic retrieval always retains validated task/object views.

Real SQLite documents/chunks, index publication, CPU FAISS, BM25, RAGHelper,
object-binding parser, source expansion and fresh original reads are used.
Only embeddings/review outputs are controlled; no real model or network.
"""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import test_history_question_binding as history_fixture
from fastapi import HTTPException
from scope_fixtures import open_scope_response

from api import generate, knowledge
from db.database import SQLiteDB
from db.schemas import MessageRequest
from inference.provider_context import get_provider_context_budget
from knowledge import public_question_binding, rag_helper, vector_db
from knowledge.multiscale_rag import runtime
from knowledge.retrieval_core.registry import KnowledgeDomainConfig

pytestmark = pytest.mark.usefixtures("no_live_context_review")
# The current wording has no literal business names. Its complete user history
# supplies both objects through the actual validated reference binding.
QUERY = "查它们的费用与材料"
NAMES = ("竹湾登记", "枫台续期")
BODIES = (
    "竹湾登记：周二09:00至11:00受理，缴款12元；持申请卡正本，节假日不受理，预订不替代正本。",
    "枫台续期：周四14:00至16:00受理，缴款18元；持确认函正本，节假日不受理，预订不替代正本。",
)


async def arrange_generic_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(history_fixture, "QUERY", QUERY)
    binding = await history_fixture.binding()
    database = SQLiteDB(tmp_path / "generic-plan.sqlite")
    base = database.create_knowledge_base("虚构公共规程", "仅隔离检索测试")
    documents = [database.save_knowledge_document(
        dict(title=name, content=body, category="规程", knowledge_base_id=base["id"]), chunks=[body],
    ) for name, body in zip(NAMES, BODIES, strict=True)]
    monkeypatch.setenv("RERANKER_ENABLED", "false")
    monkeypatch.setattr(knowledge, "db", database)
    monkeypatch.setattr(generate, "db", database)
    monkeypatch.setattr(knowledge, "_vector_index_built", False)
    monkeypatch.setattr(knowledge, "_vector_index_revision", None)
    monkeypatch.setattr(vector_db.VectorDatabase, "_check_gpu_availability", lambda _: False)
    vector = vector_db.VectorDatabase(str(tmp_path / "vectors"), vector_db.IndexConfig(index_type="flat"))

    def encode(texts, **_kwargs):
        # The full pronoun question has no object evidence. Validated object
        # views match their own independent document; nothing infers a name.
        vectors = []
        for text in texts:
            axis = next((index for index, name in enumerate(NAMES) if name in text), 2)
            item = np.zeros(384, dtype=np.float32)
            item[axis] = 1.0
            vectors.append(item)
        return np.asarray(vectors, dtype=np.float32)

    monkeypatch.setattr(vector, "_model", object())
    monkeypatch.setattr(vector_db, "sentence_transformer_embeddings", lambda _model, texts, **options: encode(texts, **options))
    monkeypatch.setattr(vector_db, "get_vector_db", lambda: vector)
    monkeypatch.setattr(rag_helper, "get_vector_db", lambda: vector)
    assert knowledge._ensure_vector_index() is True
    helper = rag_helper.RAGHelper()
    monkeypatch.setattr(rag_helper, "get_rag_helper", lambda: helper)
    budget = get_provider_context_budget(
        manager=SimpleNamespace(_current_provider=SimpleNamespace(value="openai_compat")),
        env={"OPENAI_COMPAT_CONTEXT_WINDOW_TOKENS": "65536"},
    )
    monkeypatch.setattr(generate, "get_provider_context_budget", lambda: budget)
    config = KnowledgeDomainConfig("fixture-curated", tmp_path, lambda _: [])
    forbidden_curated = Mock(side_effect=AssertionError("Bound generic objects must not load a curated model"))
    monkeypatch.setattr(runtime, "get_multiscale_rag_service", lambda: SimpleNamespace(
        config=config, retrieve_with_citations=forbidden_curated, is_warm=lambda: True,
    ))
    searches = []
    actual_hybrid = vector.hybrid_search

    def capture_search(question, **kwargs):
        searches.append(question)
        return actual_hybrid(question, **kwargs)

    monkeypatch.setattr(vector, "hybrid_search", capture_search)
    return SimpleNamespace(
        database=database, vector=vector, helper=helper, binding=binding,
        documents=documents, searches=searches, forbidden_curated=forbidden_curated,
    )


async def retrieve_bound(rig):
    return await generate._retrieve_rag_bundle(QUERY, 3, None, question_binding=rig.binding)


async def baseline_preflight(rig, monkeypatch, *, old_flag):
    """Safe observations only; caller can compare false/true before repair."""
    monkeypatch.setenv("CORRECTIVE_RAG_ENABLED", "true" if old_flag else "false")
    rig.helper.clear_cache()
    rig.vector.clear_cache()
    rig.searches.clear()
    bundle = await retrieve_bound(rig)
    packets = bundle.get("original_source_packets", ())
    coverage = bundle.get("task_candidate_coverage", ())
    return {
        "legacy_flag": old_flag, "candidate_count": len(bundle["results"]), "abstained": bundle["abstained"],
        "original_packet_count": len(packets), "searches": list(rig.searches),
        "bound_names_searched": all(name + "\n" + QUERY in rig.searches for name in NAMES),
        "both_complete_originals": {packet["original_body"] for packet in packets} == set(BODIES),
        "bound_task_buckets": sum(row.get("reference_binding") == rig.binding for row in coverage),
    }


@pytest.mark.asyncio
async def test_generic_plan_keeps_both_bound_objects_and_fresh_full_sources(tmp_path, monkeypatch):
    rig = await arrange_generic_plan(tmp_path, monkeypatch)
    try:
        monkeypatch.setenv("CORRECTIVE_RAG_ENABLED", "true")  # A retired branch cannot bypass the canonical plan.
        bundle = await retrieve_bound(rig)
        assert rig.searches == [QUERY, *(name + "\n" + QUERY for name in NAMES)]
        assert bundle["abstained"] is False
        assert {row["document_id"] for row in bundle["results"]} == {row["id"] for row in rig.documents}
        assert {packet["original_body"] for packet in bundle["original_source_packets"]} == set(BODIES)
        buckets = bundle["task_candidate_coverage"]
        assert [bucket["query"] for bucket in buckets] == [QUERY, *(name + "\n" + QUERY for name in NAMES)]
        assert all(bucket["reference_binding"] == rig.binding for bucket in buckets[1:])
        receipts = [row["original_source_receipt"] for row in bundle["source_coverage"]]
        assert len(receipts) == 2
        assert all(receipt["authority"] == "fresh_knowledge_document_read" and receipt["candidate_included"] for receipt in receipts)
        assert {receipt["authority_revision"] for receipt in receipts} == {knowledge._get_rebuild_revision()}
        rig.forbidden_curated.assert_not_called()
    finally:
        rig.database.close_connection()


@pytest.mark.asyncio
async def test_generic_index_failure_still_stops_api_with_503_before_answer(tmp_path, monkeypatch, semantic_plan):
    rig = await arrange_generic_plan(tmp_path, monkeypatch)
    try:
        monkeypatch.setenv("CORRECTIVE_RAG_ENABLED", "true")
        planning = semantic_plan(QUERY, dict(public_knowledge=[0]))
        actual_binding = public_question_binding.resolve_question_binding

        async def resolve(*args, **kwargs):
            async def controlled(messages):
                payload = json.loads(messages[-1]["content"])
                assert payload["query"] == QUERY
                assert payload["reference_context"]["history"] == history_fixture.HISTORY
                task, = payload["public_tasks"]
                assert task["text"] == QUERY
                return open_scope_response(messages, [{"task_id": task["id"], "objects": list(NAMES)}])
            return await actual_binding(*args, **kwargs, reviewer=controlled)

        monkeypatch.setattr(public_question_binding, "resolve_question_binding", resolve)
        # A real SQLite mutation invalidates the previously published index.
        rig.database.save_knowledge_document(dict(title="无关新规程", content="仅离线核对"), chunks=["仅离线核对"])
        failed_publication = Mock(side_effect=OSError("synthetic index publication failure"))
        monkeypatch.setattr(rig.vector, "clear_all", failed_publication)

        async def forbidden_answer(**kwargs):
            pytest.fail("A failed required knowledge index must not reach the answer model")

        with pytest.raises(HTTPException) as caught:
            await generate._generate_with_retrieval(
                MessageRequest(message=QUERY, history=history_fixture.HISTORY), None,
                runtime_config=dict(useKnowledgeBase=True, maxTokens=256), model_generate=forbidden_answer,
            )
        assert caught.value.status_code == 503 and isinstance(caught.value.__cause__, RuntimeError)
        assert "index is not ready" in str(caught.value.__cause__)
        assert len(planning) == 1 and not rig.searches and not rig.helper._query_cache
        failed_publication.assert_called_once()
        assert knowledge._read_rebuild_status()[0] != "complete" and knowledge._vector_index_built is False
    finally:
        rig.database.close_connection()
