"""New component race with real SQLite source capture/recall/authorized erase.

Classifier, public planner, public transport failure and main provider are mocks;
this file makes no paid provider request and claims no native auth/vector test.
"""

from datetime import datetime, timezone
from html import unescape

import pytest
from repositories.character_memory import DatabaseCharacterMemoryRepository

from character.models import CompiledCharacterContext, UserScope
from character.source_memory import SourceMemoryService, attach_sources
from db.database import SQLiteDB

QUERY = "读取我已保存的私人茶饮偏好原话；同时核对朔湾改签的费用与材料。偏好仅作为本人自述，不能证明现实审核完成，缺少公共规则保持未知。"
BODY = "虚构评测原话：我偏好青柚茶；只在周五晚间阅读时选择，强度较弱，未确定会持续；出差时例外，地点仅限旧书房。这是历史自述，不证明审核完成。"


async def exercise(tmp_path, monkeypatch, *, erase_during_review):
    from services.character_context import PreparedCharacterTurn

    from api import generate
    from character.models import RelationshipState
    from db.schemas import MessageRequest
    from knowledge import intent_detector, public_question_binding, retrieval_query_plan
    from knowledge.retrieval_query_plan import RetrievalQueryPlan
    from knowledge.turn_dependencies import parse_dependencies

    db = SQLiteDB(tmp_path / "real-source.sqlite")
    scope = UserScope("web", "authority-race", "fiction-owner", "fiction-owner", "private")
    fields = dict(
        character_id="fiction-role",
        platform=scope.platform,
        adapter=scope.adapter,
        sender_id=scope.sender_id,
        conversation_type=scope.conversation_type,
        conversation_id=scope.conversation_id,
    )
    assert (
        db.capture_memory_source(
            **fields, source_message_id="fiction-original", body=BODY, observed_at=datetime.now(timezone.utc)
        )
        == "recorded"
    )
    repo = DatabaseCharacterMemoryRepository(db)
    recalled = await SourceMemoryService(repo, defer_budget=True).recall("fiction-role", scope, QUERY)
    context = attach_sources(CompiledCharacterContext("虚构评测画像", "", "", memory_status="no_match"), recalled)
    assert BODY in context.episodic_reference_context or BODY in context.source_candidate_context
    assert not context.used_memory_ids and not context.memory_packets
    deps = parse_dependencies(
        dict(private_memory=[0], public_knowledge=[1], control=[2], current_input=[], unresolved_source=[]), QUERY
    )
    observed = {}
    original_resolve = public_question_binding.resolve_question_binding

    async def planner(query):
        assert query == QUERY
        return RetrievalQueryPlan(status="applied", dependencies=deps)

    async def binding(dependencies, query, **kwargs):
        async def review(messages):
            import json

            assert json.loads(messages[-1]["content"])["query"] == QUERY
            if erase_during_review:
                db.clear_character_memories(**fields)
                assert db.list_memory_sources(**fields) == []
            # Actual strict resolver failure retains complete unavailable public
            # tasks; this is a declared component failure, not a cloud fault.
            return "{"

        return await original_resolve(dependencies, query, reviewer=review, **kwargs)

    async def public_unavailable(*args, **kwargs):
        raise RuntimeError("declared component public transport failure")

    async def model(**kwargs):
        wire = unescape("\n".join(m["content"] for m in kwargs["messages"]))
        observed["wire"] = wire
        observed["model_calls"] = observed.get("model_calls", 0) + 1
        assert QUERY in wire
        return "仅依据仍获授权的资料回答，未知保持未知。"

    monkeypatch.setattr(retrieval_query_plan, "plan_retrieval_views", planner)
    monkeypatch.setattr(intent_detector, "needs_rag", lambda _: (True, "declared-component", None))
    monkeypatch.setattr(public_question_binding, "resolve_question_binding", binding)
    monkeypatch.setattr(generate, "_retrieve_rag_bundle", public_unavailable)
    monkeypatch.setattr(generate, "_get_system_prompt", lambda _: "虚构评测角色规则")
    _, used, meta = await generate._generate_with_retrieval(
        MessageRequest(message=QUERY, characterId="fiction-role"),
        None,
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
        runtime_config=dict(maxTokens=2048, useKnowledgeBase=True),
        model_generate=model,
        message_db=db,
    )
    assert used and observed["model_calls"] == 1
    assert not meta.get("generationError")
    assert (BODY in observed["wire"]) is (not erase_during_review)
    assert meta["answerMode"] == ("abstention" if erase_during_review else "partial_answer")
    return observed, meta


@pytest.mark.asyncio
async def test_complete_authorized_source_remains_visible_without_revocation(tmp_path, monkeypatch):
    await exercise(tmp_path, monkeypatch, erase_during_review=False)


@pytest.mark.asyncio
async def test_source_revoked_during_public_review_is_absent_from_final_model(tmp_path, monkeypatch):
    await exercise(tmp_path, monkeypatch, erase_during_review=True)


SIBLING = "虚构独立原话：我另一项茶饮偏好是白桃水，只在周二午间读报时选择；外出时例外，不代表已经审核。"
PUBLIC = "虚构当前公共规则：朔湾改签收费17元，只规定费用，不规定材料。"


async def complete_request(tmp_path, *, deferred=False, history=()):
    from services.character_context import PreparedCharacterTurn

    from character.models import RelationshipState
    from inference.generation_request import GenerationRequest, RetrievalResult
    from inference.private_context_authority import make_private_context_revalidator

    db = SQLiteDB(tmp_path / "complete-source-grants.sqlite")
    scope = UserScope("web", "final-authority", "fiction-owner", "fiction-owner", "private")
    fields = dict(
        character_id="fiction-role",
        platform=scope.platform,
        adapter=scope.adapter,
        sender_id=scope.sender_id,
        conversation_type=scope.conversation_type,
        conversation_id=scope.conversation_id,
    )
    for i, body in enumerate((BODY, SIBLING)):
        assert (
            db.capture_memory_source(
                **fields, source_message_id=f"fiction-source-{i}", body=body, observed_at=datetime.now(timezone.utc)
            )
            == "recorded"
        )
    repository = DatabaseCharacterMemoryRepository(db)
    sources = await SourceMemoryService(repository, max_chars=40 if deferred else 2400, defer_budget=True).recall(
        "fiction-role", scope, QUERY
    )
    context = attach_sources(CompiledCharacterContext("虚构评测画像", "", "", memory_status="no_match"), sources)
    packet = context.source_candidate_context if deferred else context.episodic_reference_context
    import json

    assert {row["text"] for row in json.loads(packet)["records"]} == {BODY, SIBLING}
    prepared = PreparedCharacterTurn(
        character_id="fiction-role",
        user_scope=scope,
        compiled=context,
        history=history,
        relationship=RelationshipState(),
        memory_candidates=0,
        interaction_count=0,
        reply_guard=None,
    )
    request = GenerationRequest(
        message=QUERY,
        character_context=context,
        history=history,
        retrieval=RetrievalResult(status="ok", evidence=PUBLIC),
        private_context_revalidator=make_private_context_revalidator(prepared, db),
    )
    return request, prepared, repository, db, fields


def model_wire(messages):
    return unescape("\n".join(message["content"] for message in messages))


@pytest.mark.asyncio
async def test_selected_erasure_keeps_whole_sibling_and_independent_public_source(tmp_path):
    from inference.generation_request import generate_character_response

    request, _prepared, _repository, db, fields = await complete_request(tmp_path)
    assert db.erase_unlinked_memory_sources(**fields, source_message_ids=("fiction-source-0",)) == 1
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        return "按仍获授权的完整资料核对，缺项未知。"

    result = await generate_character_response(request, model)
    assert len(calls) == 1
    wire = model_wire(calls[0]["messages"])
    assert BODY not in wire and SIBLING in wire and PUBLIC in wire and QUERY in wire
    assert result.plan.character_context.memory_source_status == "available"


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["body", "observed_at"])
async def test_changed_source_snapshot_never_substitutes_new_unreviewed_content(tmp_path, field):
    from inference.generation_request import generate_character_response

    request, _prepared, _repository, db, _fields = await complete_request(tmp_path)
    replacement = "虚构变更原话：我的茶饮偏好已经改为梨汁，尚未重新审核，不能借旧选择授予依据。"
    value = replacement if field == "body" else "2030-01-01T00:00:00+00:00"
    db._get_connection().execute(
        f"UPDATE memory_sources SET {field}=? WHERE source_message_id=?", (value, "fiction-source-0")
    )
    db._get_connection().commit()
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        return "依当前仍获授权的来源回答。"

    await generate_character_response(request, model)
    assert len(calls) == 1
    wire = model_wire(calls[0]["messages"])
    assert BODY not in wire and replacement not in wire and SIBLING in wire and PUBLIC in wire


@pytest.mark.asyncio
async def test_failed_fresh_source_read_does_not_grant_cached_private_packet(tmp_path, monkeypatch):
    from inference.generation_request import generate_character_response

    request, _prepared, _repository, _db, _fields = await complete_request(tmp_path)

    async def unavailable(*args, **kwargs):
        raise RuntimeError("declared scoped read failure")

    monkeypatch.setattr(DatabaseCharacterMemoryRepository, "list_sources", unavailable)
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        return "公共费用有依据，私人原话本轮不可核验。"

    result = await generate_character_response(request, model)
    wire = model_wire(calls[0]["messages"])
    assert len(calls) == 1 and BODY not in wire and SIBLING not in wire and PUBLIC in wire
    assert result.plan.character_context.memory_source_status == "authority_unavailable"


@pytest.mark.asyncio
async def test_identical_source_id_in_other_owner_scope_cannot_restore_original(tmp_path):
    from dataclasses import replace

    from inference.generation_request import generate_character_response
    from inference.private_context_authority import make_private_context_revalidator

    request, prepared, _repository, db, _fields = await complete_request(tmp_path)
    other = replace(prepared.user_scope, sender_id="other-fiction-owner")
    prepared = replace(prepared, user_scope=other)
    request = replace(request, private_context_revalidator=make_private_context_revalidator(prepared, db))
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        return "没有该范围的可用私人原话。"

    await generate_character_response(request, model)
    wire = model_wire(calls[0]["messages"])
    assert BODY not in wire and SIBLING not in wire and PUBLIC in wire


@pytest.mark.asyncio
async def test_revoked_deferred_whole_packet_cannot_return_through_budget_admission(tmp_path):
    from inference.generation_request import generate_character_response

    request, _prepared, _repository, db, fields = await complete_request(tmp_path, deferred=True)
    assert (
        not request.character_context.episodic_reference_context and request.character_context.source_candidate_context
    )
    db.clear_character_memories(**fields)
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        return "保留公共依据，私人字段未知。"

    result = await generate_character_response(request, model)
    wire = model_wire(calls[0]["messages"])
    assert BODY not in wire and SIBLING not in wire and PUBLIC in wire
    assert not result.plan.character_context.source_candidate_context
    assert not result.plan.character_context.episodic_reference_context


@pytest.mark.asyncio
async def test_revoked_original_and_its_history_turn_do_not_survive_context_copy(tmp_path):
    from dataclasses import replace

    from inference.generation_request import generate_character_response

    history = (
        dict(role="user", content=BODY),
        dict(role="assistant", content="已收到上面的历史茶饮自述。"),
        dict(role="user", content="虚构无关轮次：刚才讨论的是书页装订。"),
        dict(role="assistant", content="书页装订的话题保持。"),
    )
    request, _prepared, _repository, db, fields = await complete_request(tmp_path, history=history)
    request = replace(
        request, character_context=replace(request.character_context, conversation_reference_context=BODY)
    )
    db.erase_unlinked_memory_sources(**fields, source_message_ids=("fiction-source-0",))
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        return "仍有效的其他资料保持。"

    result = await generate_character_response(request, model)
    wire = model_wire(calls[0]["messages"])
    assert BODY not in wire and "已收到上面的历史茶饮自述。" not in wire
    assert "书页装订" in wire and SIBLING in wire and PUBLIC in wire
    assert not result.plan.character_context.conversation_reference_context


@pytest.mark.asyncio
async def test_guard_retry_rechecks_grant_after_first_reply(tmp_path):
    from dataclasses import replace

    from character.output_guard import ReplyGuard, validate_reply
    from inference.generation_request import generate_character_response

    request, _prepared, _repository, db, fields = await complete_request(tmp_path)
    guard = ReplyGuard(forbid_laughter=True)
    first_reply = "哈哈，资料已经核对。"
    assert validate_reply(first_reply, guard)
    request = replace(request, reply_guard=guard, reply_guard_mode="strict")
    calls = []

    async def model(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            db.clear_character_memories(**fields)
            return first_reply
        return "本轮仅依据当前仍可核验的公共资料回答。"

    result = await generate_character_response(request, model)
    assert result.guard_retried and len(calls) == 2 and not result.guard_fallback
    assert BODY in model_wire(calls[0]["messages"]) and BODY not in model_wire(calls[1]["messages"])
    assert SIBLING not in model_wire(calls[1]["messages"]) and PUBLIC in model_wire(calls[1]["messages"])
    assert first_reply not in model_wire(calls[1]["messages"])
    assert not result.plan.character_context.episodic_reference_context


@pytest.mark.asyncio
async def test_request_callback_never_changes_serialized_completion_schema(tmp_path):
    import json

    from services.delivery_memory import freeze_completion

    request, prepared, _repository, _db, _fields = await complete_request(tmp_path)
    assert callable(request.private_context_revalidator)
    snapshot = freeze_completion(prepared)
    encoded = json.dumps(snapshot)
    assert "private_context_revalidator" not in encoded
    assert set(snapshot) == {
        "version",
        "character_id",
        "user_scope",
        "received_at",
        "relationship",
        "history",
        "used_memory_ids",
        "memory_operation_receipt",
    }
