"""Same-object businesses need independent candidates from complete source data."""

from knowledge import index_lifecycle

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from scope_fixtures import open_scope_response

from api import generate, knowledge
from db.database import SQLiteDB
from inference.provider_context import get_provider_context_budget
from knowledge import rag_helper, vector_db
from knowledge.multiscale_rag import runtime
from knowledge.public_obligations import parse_public_partitions
from knowledge.public_question_binding import resolve_question_binding
from knowledge.retrieval_core.registry import KnowledgeDomainConfig
from knowledge.turn_dependencies import KINDS, parse_dependencies

NAME = "松檐馆"
QUERY = "核对松檐馆新卡办理的费用和材料，另核对松檐馆押金退回的费用和材料。保留公众假期不受理和预约不免原件的限制。"
BODIES = (
    "松檐馆新卡办理：周二09:00至11:00受理，收费12元；携带本人申请卡正本，公众假期不受理，预约不能免原件。",
    "松檐馆新卡办理照片要求：一寸白底近照一张，照片须为三个月内拍摄；没有快照服务，不收照相费用。",
    "松檐馆新卡办理代领材料：仅限已办结的新卡，须持本人和代领人身份证明正本；预约不能代替授权书。",
    "松檐馆保证金返还：每周四14:00至16:00办理，返还80元、手续费0元；须本人付款凭证正本，公众假期不办理，预约不能免原件。",
)

pytestmark = pytest.mark.usefixtures("no_live_context_review")


async def arrange(tmp_path, monkeypatch):
    groups = {kind: [0] if kind == "public_knowledge" else [1] if kind == "control" else [] for kind in KINDS}
    dependencies = parse_dependencies(groups, QUERY)
    obligations = parse_public_partitions([dict(segment_id=0, cuts=["另核对松檐馆"])], dependencies, QUERY)

    async def review(messages):
        payload = json.loads(messages[-1]["content"])
        assert payload["query"] == QUERY and len(payload["public_tasks"]) == 2
        return open_scope_response(messages, [dict(task_id=task["id"], objects=[NAME]) for task in payload["public_tasks"]])

    binding = await resolve_question_binding(dependencies, QUERY, window_tokens=65536,
        public_obligations=obligations, reviewer=review)
    database = SQLiteDB(tmp_path / "same-object.sqlite")
    base = database.create_knowledge_base("虚构松檐馆规程", "仅隔离链路控制")
    documents = [database.save_knowledge_document(dict(title=body.split("：")[0], content=body,
        category="规程", knowledge_base_id=base["id"]), chunks=[body]) for body in BODIES]
    monkeypatch.setenv("RERANKER_ENABLED", "false")
    monkeypatch.setattr(knowledge, "db", database)
    monkeypatch.setattr(index_lifecycle, "db", database)
    monkeypatch.setattr(generate, "db", database)
    monkeypatch.setattr(index_lifecycle, "_vector_index_revision", None)
    monkeypatch.setattr(vector_db.VectorDatabase, "_check_gpu_availability", lambda _: False)
    vector = vector_db.VectorDatabase(str(tmp_path / "vectors"), vector_db.IndexConfig(index_type="flat"))

    def encode(texts, **kwargs):
        # The controlled model can resolve either single business, including
        # the deposit-return synonym. The mixed question/name remains broad.
        values = []
        for text in texts:
            refund_only = "保证金返还" in text or ("押金退回" in text and "新卡办理" not in text)
            item = np.zeros(384, dtype=np.float32)
            if refund_only:
                item[1] = 1
            elif "照片要求" in text or "代领材料" in text:
                item[0], item[2] = .25, np.sqrt(1 - .25 ** 2)
            elif "新卡办理" in text and "押金退回" not in text:
                item[0] = 1
            else:
                item[2] = 1
            values.append(item)
        return np.asarray(values, dtype=np.float32)

    monkeypatch.setattr(vector, "_model", object())
    monkeypatch.setattr(vector_db, "sentence_transformer_embeddings", lambda _model, texts, **options: encode(texts, **options))
    monkeypatch.setattr(vector_db, "get_vector_db", lambda: vector)
    monkeypatch.setattr(rag_helper, "get_vector_db", lambda: vector)
    assert index_lifecycle._ensure_vector_index()
    helper = rag_helper.RAGHelper()
    monkeypatch.setattr(rag_helper, "get_rag_helper", lambda: helper)
    for task, document in zip(obligations, (documents[0], documents[-1]), strict=True):
        rows = helper.retrieve_context(NAME + "\n" + task["query"], top_k=1, enable_rerank=False, use_cache=False)
        assert rows[0]["document_id"] == document["id"], "Atomic business must be retrievable before testing the chain"
    vector.clear_cache()
    budget = get_provider_context_budget(manager=SimpleNamespace(_current_provider=SimpleNamespace(value="openai_compat")),
        env={"OPENAI_COMPAT_CONTEXT_WINDOW_TOKENS": "65536"})
    monkeypatch.setattr(generate, "get_provider_context_budget", lambda: budget)
    curated = Mock(side_effect=AssertionError("These bound businesses are not curated character evidence"))
    monkeypatch.setattr(runtime, "get_multiscale_rag_service", lambda: SimpleNamespace(
        config=KnowledgeDomainConfig("fixture-curated", tmp_path, lambda _: []), retrieve_with_citations=curated, is_warm=lambda: True))
    searches = []
    actual = vector.hybrid_search

    def search(question, **kwargs):
        searches.append(question)
        return actual(question, **kwargs)

    monkeypatch.setattr(vector, "hybrid_search", search)
    return SimpleNamespace(database=database, binding=binding, obligations=obligations,
        documents=documents, searches=searches, curated=curated)


@pytest.mark.parametrize("top_k", [1, 3])
async def test_same_object_businesses_retain_both_full_sources(tmp_path, monkeypatch, top_k):
    rig = await arrange(tmp_path, monkeypatch)
    try:
        bundle = await generate._retrieve_rag_bundle(QUERY, top_k, None, question_binding=rig.binding)
        retained = {packet["original_body"] for packet in bundle["original_source_packets"]}
        assert BODIES[0] in retained and BODIES[-1] in retained
        assert rig.searches[0] == QUERY
        assert any("押金退回" in text and "新卡办理" not in text for text in rig.searches[1:])
        assert any("新卡办理" in text and "押金退回" not in text for text in rig.searches[1:])
        assert all(record["original_source_receipt"]["authority"] == "fresh_knowledge_document_read"
                   for record in bundle["source_coverage"])
        rig.curated.assert_not_called()
    finally:
        rig.database.close_connection()


@pytest.mark.parametrize("defect", ["object_only", "missing", "proof", "order"])
async def test_task_receipts_preserve_complete_queries_and_their_binding(tmp_path, monkeypatch, defect):
    from inference.task_evidence_coverage import validate_task_candidates

    rig = await arrange(tmp_path, monkeypatch)
    try:
        bundle = await generate._retrieve_rag_bundle(QUERY, 3, None, question_binding=rig.binding)
        records = bundle["task_candidate_coverage"]
        checked = validate_task_candidates(records, query=QUERY)
        assert len(checked) == 3 and all(row["reference_binding"] == rig.binding for row in checked[1:])
        assert all(row["semantic_coverage"] == "unverified" for row in checked)
        changed = deepcopy(records)
        if defect == "object_only":
            changed[1]["query"] = NAME
        elif defect == "missing":
            changed = changed[:-1]
        elif defect == "proof":
            changed[1]["reference_binding"]["input_sha256"] = "0" * 64
        else:
            changed[1]["query"], changed[2]["query"] = changed[2]["query"], changed[1]["query"]
        with pytest.raises(ValueError):
            validate_task_candidates(changed, query=QUERY)
    finally:
        rig.database.close_connection()


async def binding_for(query):
    from knowledge.turn_dependencies import query_segments

    dependencies = parse_dependencies({kind: list(range(len(query_segments(query)))) if kind == "public_knowledge" else []
                                      for kind in KINDS}, query)

    async def review(messages):
        payload = json.loads(messages[-1]["content"])
        return open_scope_response(messages, [dict(task_id=task["id"], objects=[NAME]) for task in payload["public_tasks"]])

    return await resolve_question_binding(dependencies, query, window_tokens=65536, reviewer=review)


@pytest.mark.parametrize("count", [4, 5])
async def test_complete_business_query_set_fits_or_fails_without_trimming(count):
    from knowledge.public_question_binding import QuestionBindingCapacityError, bound_task_search_views

    query = "".join(f"核对松檐馆业务{i}的费用和全部例外。" for i in range(count))
    binding = await binding_for(query)
    if count == 5:
        with pytest.raises(QuestionBindingCapacityError):
            bound_task_search_views(binding, query)
    else:
        views = bound_task_search_views(binding, query)
        assert len(views) == 4
        assert all(view == NAME + "\n" + task["text"] for view, task in zip(views, binding["input"]["public_tasks"], strict=True))


async def test_long_bound_business_query_keeps_its_final_negation():
    from inference.task_evidence_coverage import validate_task_candidates
    from knowledge.public_question_binding import bound_task_search_views
    from knowledge.task_retrieval import collect_task_candidates, literal_task_views

    query = "核对松檐馆公开规程中的全部条件" + "完整背景" * 300 + "末尾要求：未付款不能办理且预约不免原件。"
    binding = await binding_for(query)
    views = bound_task_search_views(binding, query)
    assert len(views) == 1 and len(views[0]) > 1024 and views[0].endswith("预约不免原件。")
    with pytest.raises(ValueError):
        literal_task_views(query, views)
    row = dict(id="doc_1_chunk_0", document_id=1, title="完整合成规程", content="完整原文控制", score=.9)
    plan = collect_task_candidates(query, views, question_binding=binding, top_k=1, filters=None,
        retrieve=lambda *a, **kw: [dict(row)], confidence=lambda rows: .9, stable_key=lambda item: item["id"], snapshot=lambda: 1)
    assert validate_task_candidates(plan.coverage, query=query)[1]["query"] == NAME + "\n" + query


async def test_bound_businesses_reject_separate_query_hints_before_search(tmp_path, monkeypatch):
    rig = await arrange(tmp_path, monkeypatch)
    try:
        with pytest.raises(ValueError):
            rag_helper.get_rag_helper().retrieve_with_citations(QUERY, additional_queries=(NAME,), question_binding=rig.binding)
        assert not rig.searches
    finally:
        rig.database.close_connection()


async def test_capacity_error_reaches_generation_as_actionable_failure(monkeypatch, semantic_plan):
    from unittest.mock import AsyncMock

    from fastapi import HTTPException

    from db.schemas import MessageRequest
    from knowledge import public_question_binding
    from knowledge.public_question_binding import QuestionBindingCapacityError

    query = "核对松檐馆的全部公共业务。"
    semantic_plan(query, dict(public_knowledge=[0]))
    failure = QuestionBindingCapacityError("Complete public task views exceed retrieval capacity")
    retrieve = AsyncMock(side_effect=failure)
    model = AsyncMock()
    monkeypatch.setattr(public_question_binding, "resolve_question_binding", AsyncMock(return_value=None))
    monkeypatch.setattr(generate, "_retrieve_rag_bundle", retrieve)
    monkeypatch.setattr(generate, "_rag_retrieval_timeout", lambda: 30.0)
    with pytest.raises(HTTPException) as caught:
        await generate._generate_with_retrieval(MessageRequest(message=query), None,
            runtime_config={"useKnowledgeBase": True}, model_generate=model)
    assert caught.value.status_code == 503 and caught.value.detail == failure.public_detail()
    assert caught.value.__cause__ is failure
    model.assert_not_awaited()
