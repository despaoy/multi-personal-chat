"""Declared planner failures with real private-source reads, no cloud or vector claims."""

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

QUERY = "读取我保存的私人饮品喜好原话；同时查询桐屿续签的费用和完整材料规则；只读，不新增或删除记忆，缺少依据保持未知。"
PRIVATE = "虚构测试原话：我不喜欢薄荷水，只在周日午间吃过午饭且没有牙痛时喜欢；缺少一项条件仍不喜欢。这不证明我已经喝过或完成现实审核。"
PUBLIC = "虚构公共规程：桐屿续签收费31元，需要本人申请卡原件，公众假期不办理，预约不能免原件。"


async def exercise(tmp_path, monkeypatch, mode):
    from api import generate
    from db.schemas import MessageRequest
    from knowledge import intent_detector, public_question_binding, retrieval_query_plan

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
        CompiledCharacterContext("虚构角色", "", "", memory_status="no_match"), recalled, complete_evidence=True
    )
    assert PRIVATE in context.episodic_reference_context
    base = database.create_knowledge_base("虚构公共规则", "声明的单元候选库")
    document = database.add_knowledge_document(
        dict(title="桐屿续签规程", content=PUBLIC, knowledge_base_id=base["id"], chunkCount=1)
    )
    database.add_knowledge_chunk(dict(documentId=document["id"], chunkIndex=0, content=PUBLIC))
    before = database.list_memory_sources(**fields)
    seen = dict(retrieval=0, model=0, binding=0)
    original_plan = retrieval_query_plan.plan_retrieval_views
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
                search_views=[["not in original query"]],
                public_partitions=[dict(segment_id=1, cuts=[])],
            )
        )

    async def planner(query):
        assert query == QUERY
        plan = await original_plan(query, reviewer=reviewer)
        seen["status"] = plan.status
        return plan

    async def binding(*args, **kwargs):
        # No object/source permission is minted by this control stub.
        seen["binding"] += 1
        return None

    async def retrieve(*args, **kwargs):
        seen["retrieval"] += 1
        if mode in {"disabled", "invalid_views"}:
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

    actual_generate = generate.generate_character_response

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

    monkeypatch.setattr(retrieval_query_plan, "plan_retrieval_views", planner)
    monkeypatch.setattr(intent_detector, "needs_rag", lambda _: (True, "declared-unit", None))
    monkeypatch.setattr(public_question_binding, "resolve_question_binding", binding)
    monkeypatch.setattr(generate, "_retrieve_rag_bundle", retrieve)
    monkeypatch.setattr(generate, "_get_system_prompt", lambda _: "虚构角色依据可见来源回答")
    monkeypatch.setattr(generate, "generate_character_response", capture)
    try:
        _, used, metadata = await generate._generate_with_retrieval(
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
        assert database.list_memory_sources(**fields) == before
        assert database.get_knowledge_document(document["id"])["content"] == PUBLIC
        assert used and seen["model"] == 1 and seen["private_complete"] and not seen["public_visible"]
        return seen, metadata
    finally:
        database.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["provider_error", "timeout", "invalid_json", "incomplete_dependencies"])
async def test_failed_dependency_plan_never_grants_public_candidates_but_keeps_private(tmp_path, monkeypatch, mode):
    seen, metadata = await exercise(tmp_path, monkeypatch, mode)
    assert seen["retrieval"] == seen["binding"] == 0
    assert metadata["abstained"] and metadata["answerMode"] == "abstention"
    assert not metadata["citations"]
    assert seen["retrieval_plan"].reason == "query_plan_" + seen["status"]
    assert "query_plan_" + seen["status"] in metadata["warnings"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["invalid_views", "disabled"])
async def test_valid_dependencies_or_disabled_planner_preserve_normal_dispatch(tmp_path, monkeypatch, mode):
    seen, metadata = await exercise(tmp_path, monkeypatch, mode)
    assert seen["retrieval"] == 1 and metadata["abstained"]
    assert seen["status"] == ("dependencies_only_invalid_views" if mode == "invalid_views" else "disabled")
    assert seen["binding"] == (1 if mode == "invalid_views" else 0)
    assert not any(w.startswith("query_plan_") for w in (metadata.get("warnings") or []))
