"""Declared planner failures with real private-source reads, no cloud or vector claims."""

from services import chat_answer

import asyncio
import json
from datetime import datetime, timezone
from html import unescape

import pytest
from repositories.character_memory import DatabaseCharacterMemoryRepository
from services.character_context import PreparedCharacterTurn

from character.models import CompiledCharacterContext, RelationshipState, UserScope
from character.source_memory import SourceMemoryService, attach_sources
from db.database import SQLiteDB
from knowledge.retrieval_query_plan import QueryPlanReviewError

QUERY = "读取我保存的私人饮品喜好原话；同时查询桐屿续签的费用和完整材料规则；只读，不新增或删除记忆，缺少依据保持未知。"
PRIVATE = "虚构测试原话：我不喜欢薄荷水，只在周日午间吃过午饭且没有牙痛时喜欢；缺少一项条件仍不喜欢。这不证明我已经喝过或完成现实审核。"
PUBLIC = "虚构公共规程：桐屿续签收费31元，需要本人申请卡原件，公众假期不办理，预约不能免原件。"


async def exercise(tmp_path, monkeypatch, mode):
    from db.schemas import MessageRequest
    from knowledge import public_question_binding, retrieval_query_plan

    monkeypatch.setattr(chat_answer, '_rag_retrieval_timeout', lambda: 30.0)

    database = SQLiteDB(tmp_path / "native-source.sqlite")
    scope = UserScope("web", "failed-plan-unit", "fiction-owner", "fiction-owner", "private")
    fields = dict(
        character_id="fiction-role",
        platform=scope.platform,
        adapter=scope.adapter,
        sender_id=scope.sender_id,
        conversation_type=scope.conversation_type,
        conversation_id=scope.conversation_id,
    )
    assert (
        database.capture_memory_source(
            **fields, source_message_id="fiction-source", body=PRIVATE, observed_at=datetime.now(timezone.utc)
        )
        == "recorded"
    )
    repository = DatabaseCharacterMemoryRepository(database)
    recalled = await SourceMemoryService(repository, defer_budget=True).recall("fiction-role", scope, QUERY)
    context = attach_sources(
        CompiledCharacterContext("虚构角色", "", "", memory_status="no_match"), recalled
    )
    assert PRIVATE in context.episodic_reference_context
    base = database.create_knowledge_base("虚构公共规则", "声明的单元候选库")
    document = database.save_knowledge_document(
        dict(title="桐屿续签规程", content=PUBLIC, knowledge_base_id=base["id"]), chunks=[PUBLIC]
    )
    before = database.list_memory_sources(**fields)
    seen = dict(retrieval=0, model=0, binding=0)
    original_plan = retrieval_query_plan.plan_retrieval
    monkeypatch.setenv("RAG_TASK_PLANNER_ENABLED", "false" if mode == "disabled" else "true")
    monkeypatch.setenv("CORRECTIVE_RAG_ENABLED", "false")
    monkeypatch.setenv("CHARACTER_INDEPENDENT_TASKS_ENABLED", "false")
    monkeypatch.setenv("RAG_CITATIONS_ENABLED", "false")

    async def reviewer(messages):
        assert json.loads(messages[-1]["content"])["query"] == QUERY
        if mode == "provider_error":
            raise RuntimeError("Declared unit provider failure")
        if mode == "timeout":
            raise asyncio.TimeoutError("Declared unit timeout")
        if mode == "invalid_json":
            return "{"
        dependencies = dict(
            private_memory=[0], public_knowledge=[1], control=[2], current_input=[], unresolved_source=[]
        )
        if mode == "incomplete_dependencies":
            dependencies["private_memory"] = []
        return json.dumps(
            dict(
                dependencies=dependencies,
                **({"search_views": [["not in original query"]]} if mode == "invalid_views" else {}),
                public_partitions=[dict(segment_id=1, cuts=["不存在的业务"] if mode == "invalid_partition" else [])],
            )
        )

    async def planner(query, **kwargs):
        assert query == QUERY
        try:
            plan = await original_plan(query, reviewer=reviewer, **kwargs)
        except QueryPlanReviewError as exc:
            seen["status"] = exc.status
            raise
        seen["status"] = plan.status
        return plan

    async def binding(*args, **kwargs):
        # No object/source permission is minted by this control stub.
        seen["binding"] += 1
        return None

    async def retrieve(*args, **kwargs):
        seen["retrieval"] += 1
        if mode in {"valid"}:
            # Control verifies dispatch only; it supplies no approved evidence.
            return dict(results=[], citations=[], abstained=True)
        doc = database.get_knowledge_document(document["id"])
        return dict(
            results=[
                dict(
                    id=f"doc_{doc['id']}_chunk_0",
                    document_id=doc["id"],
                    title=doc["title"],
                    content=doc["content"],
                    knowledge_base_id=doc["knowledge_base_id"],
                )
            ],
            citations=[],
            source_coverage=[],
            abstained=False,
        )

    actual_generate = chat_answer.generate_character_response

    async def capture(request, model):
        result = await actual_generate(request, model)
        seen["retrieval_plan"] = result.plan.retrieval
        return result

    async def model(**kwargs):
        seen["model"] += 1
        wire = unescape("\n".join(m["content"] for m in kwargs["messages"]))
        seen["private_complete"] = PRIVATE in wire
        seen["public_visible"] = PUBLIC in wire
        assert QUERY in wire and PRIVATE in wire
        return "明确零云单元检查占位，私人来源保留，公共未知；不是实际模型答案。"

    monkeypatch.setattr(retrieval_query_plan, "plan_retrieval", planner)
    monkeypatch.setattr(public_question_binding, "resolve_question_binding", binding)
    monkeypatch.setattr(chat_answer, 'retrieve_chat_knowledge', retrieve)
    monkeypatch.setattr(chat_answer, 'get_lora_system_prompt', lambda _: "虚构角色依据可见来源回答")
    monkeypatch.setattr(chat_answer, 'generate_character_response', capture)
    try:
        pending = chat_answer.generate_chat_answer(
            MessageRequest(message=QUERY, characterId="fiction-role"),
            None,
            runtime_config=dict(maxTokens=2048, useKnowledgeBase=True),
            prepared_character_turn=PreparedCharacterTurn(
                character_id="fiction-role",
                user_scope=scope,
                compiled=context,
                history=(),
                relationship=RelationshipState(),
                memory_candidates=0,
                interaction_count=0,
                reply_guard=None,
            ),
            model_generate=model,
            message_db=database,
        )
        if mode not in {"valid"}:
            with pytest.raises(chat_answer.ChatRetrievalError) as caught:
                await pending
            error = caught.value
            assert error.http_status == 503 and isinstance(error.__cause__, QueryPlanReviewError)
            assert error.__cause__.__cause__ is not None
            assert seen["retrieval"] == seen["binding"] == seen["model"] == 0
            assert database.list_memory_sources(**fields) == before
            assert database.get_knowledge_document(document["id"])["content"] == PUBLIC
            return seen, error
        answer = await pending
        used, metadata = answer.used_rag, answer.metadata
        assert database.list_memory_sources(**fields) == before
        assert database.get_knowledge_document(document["id"])["content"] == PUBLIC
        assert used and seen["model"] == 1 and seen["private_complete"] and not seen["public_visible"]
        return seen, metadata
    finally:
        database.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["provider_error", "timeout", "invalid_json", "incomplete_dependencies", "invalid_views", "invalid_partition"])
async def test_failed_plan_stops_retrieval_and_answer_without_mutating_sources(tmp_path, monkeypatch, mode):
    seen, error = await exercise(tmp_path, monkeypatch, mode)
    expected = mode if mode in {"provider_error", "timeout"} else "invalid"
    assert seen["status"] == expected and error.__cause__.status == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["valid"])
async def test_complete_semantic_plan_preserves_normal_dispatch(tmp_path, monkeypatch, mode):
    seen, metadata = await exercise(tmp_path, monkeypatch, mode)
    assert seen["retrieval"] == 1 and metadata["abstained"]
    assert seen["status"] == "applied"
    assert seen["binding"] == 1
    assert not any(w.startswith("query_plan_") for w in (metadata.get("warnings") or []))
