"""Complete synthetic administrative validity window; normal rule storage.

Physical UTC time advances; original stored versions do not change. Main is a
spy, source recall is off, and no native auth/vector/cloud result is claimed.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from html import unescape

import pytest
from repositories.character_memory import DatabaseCharacterMemoryRepository
from services.character_context import PreparedCharacterTurn

from character.context_builder import compile_reference_context
from character.memory_extractor import extract_memories
from character.memory_read_authority import record_version
from character.memory_service import CharacterMemoryService
from character.models import CompiledCharacterContext, RelationshipState, UserScope
from character.rule_memory_writer import write_rule_memory
from db.database import SQLiteDB
from inference.generation_request import GenerationRequest, generate_character_response
from inference.private_context_authority import make_private_context_revalidator

BODY = "我喜欢乌龙茶。"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["before", "after", "historical"])
async def test_effective_time_rechecked_between_preparation_and_send(tmp_path, mode):
    db = SQLiteDB(tmp_path / "actual-expiry.sqlite")
    repo = DatabaseCharacterMemoryRepository(db)
    scope = UserScope("web", "validity-boundary", "fiction-owner", "fiction-owner", "private")
    stamp = datetime.now(timezone.utc)
    deadline = stamp + timedelta(seconds=1)
    assert (
        await repo.capture_source(
            "fiction-role", scope, source_message_id="fiction-window-original", body=BODY, observed_at=stamp
        )
        == "recorded"
    )
    extracted = extract_memories(BODY, reference_time=stamp)
    assert len(extracted) == 1 and await write_rule_memory(
        repo, "fiction-role", scope, extracted[0], "fiction-window-original", observed_at=stamp
    )
    rows = await repo.list_memory_records("fiction-role", scope, limit=None)
    assert len(rows) == 1
    memory_id = rows[0]["id"]
    # An explicitly declared isolated administrative window, not an inferred
    # model endpoint. This is the authoritative time information in the case.
    connection = db._get_connection()
    connection.execute(
        "UPDATE character_memories SET valid_from=?,valid_to=? WHERE id=?",
        ((stamp - timedelta(days=1)).isoformat(), deadline.isoformat(), memory_id),
    )
    connection.commit()
    query = (
        "我在过去这条记录的有效期内，保存的乌龙茶偏好是什么？只依据本人历史记录，不推断当前仍保持。"
        if mode == "historical"
        else "我当前保存的乌龙茶偏好是什么？只依据当前时间仍处于所存有效区间的本人记录回答，到期偏好保持未知，不推断当前仍保持。"
    )
    if mode == "historical":
        await asyncio.sleep(max(0, (deadline - datetime.now(timezone.utc)).total_seconds()) + 0.02)
    service = CharacterMemoryService(repo, semantic_enabled=False)
    memories, count = await service.load_relevant_memories("fiction-role", scope, query)
    assert count == 1 and len(memories) == 1
    selected = memories[0]
    assert selected.historical is (mode == "historical") and selected.valid_to == deadline.isoformat()
    assert selected.storage_versions
    reference, ids = compile_reference_context(memories, complete_evidence=True, max_chars=None)
    assert BODY.rstrip("。") in reference
    context = CompiledCharacterContext(
        "虚构组件画像", "", reference, used_memory_ids=ids, memory_status="available", memory_packets=memories
    )
    prepared = PreparedCharacterTurn(
        character_id="fiction-role",
        user_scope=scope,
        compiled=context,
        history=(),
        relationship=RelationshipState(),
        memory_candidates=count,
        interaction_count=0,
        reply_guard=None,
    )
    request = GenerationRequest(
        message=query,
        persona_prompt="虚构组件规则",
        context_window_tokens=65536,
        character_context=context,
        private_context_revalidator=make_private_context_revalidator(prepared, db),
    )
    before = record_version(await repo.get_memory_record(memory_id, "fiction-role", scope))
    if mode == "after":
        await asyncio.sleep(max(0, (deadline - datetime.now(timezone.utc)).total_seconds()) + 0.02)
        assert datetime.now(timezone.utc) >= deadline
        current, _count = await service.load_relevant_memories("fiction-role", scope, query)
        assert current == ()
    if mode == "before":
        assert datetime.now(timezone.utc) < deadline
    assert record_version(await repo.get_memory_record(memory_id, "fiction-role", scope)) == before
    seen = []

    async def model(**kwargs):
        seen.append(unescape("\n".join(message["content"] for message in kwargs["messages"])))
        return "只依据本轮仍符合所问时间的记录回答。"

    result = await generate_character_response(request, model)
    assert len(seen) == 1 and result.reply and query in seen[0]
    assert (BODY.rstrip("。") in seen[0]) is (mode != "after")
    assert (selected.memory_id in result.plan.character_context.used_memory_ids) is (mode != "after")


async def time_request(tmp_path, monkeypatch, *, mixed=False, future=False):
    from inference.generation_request import RetrievalResult

    db = SQLiteDB(tmp_path / "more-effective-time.sqlite")
    repo = DatabaseCharacterMemoryRepository(db)
    scope = UserScope("web", "validity-boundary", "fiction-owner", "fiction-owner", "private")
    stamp = datetime.now(timezone.utc)
    statements = [(BODY, stamp - timedelta(days=2)), ("我喜欢豆浆。", stamp - timedelta(days=2))]
    if mixed:
        statements.append(("我不喜欢乌龙茶。", stamp - timedelta(days=1)))
    for index, (body, observed) in enumerate(statements):
        assert (
            await repo.capture_source(
                "fiction-role", scope, source_message_id=f"fiction-effective-{index}", body=body, observed_at=observed
            )
            == "recorded"
        )
        items = extract_memories(body, reference_time=observed)
        assert len(items) == 1 and await write_rule_memory(
            repo, "fiction-role", scope, items[0], f"fiction-effective-{index}", observed_at=observed
        )
    rows = await repo.list_memory_records("fiction-role", scope, limit=None)
    target = next(row for row in rows if "乌龙茶" in row["content"])
    deadline = datetime.now(timezone.utc) + timedelta(seconds=0.6)
    connection = db._get_connection()
    if future:
        connection.execute(
            "UPDATE character_memories SET valid_from=? WHERE id=?",
            ((stamp + timedelta(days=1)).isoformat(), target["id"]),
        )
    else:
        connection.execute("UPDATE character_memories SET valid_to=? WHERE id=?", (deadline.isoformat(), target["id"]))
    connection.commit()
    query = (
        "我过去保存的乌龙茶偏好是什么？我当前的乌龙茶偏好是什么？我当前的豆浆偏好是什么？只依据各自时间的本人记录，不证明现实审核完成。"
        if mixed
        else "我当前保存的乌龙茶和豆浆偏好分别是什么？只依据当前时间仍有效的本人记录，到期或尚未生效的偏好保持未知。"
    )
    service = CharacterMemoryService(repo, semantic_enabled=False)
    memories, count = await service.load_relevant_memories(
        "fiction-role", scope, query, reference_time=stamp + timedelta(days=2) if future else None
    )
    assert len(memories) == (3 if mixed else 2) and count == len(memories)
    selected = next(item for item in memories if item.memory_id == str(target["id"]))
    assert not selected.historical
    historical = next((item for item in memories if item.historical), None)
    if mixed:
        assert historical and historical.content != selected.content
    if future:
        # Explicit clock rewind control: preparation used an advanced clock;
        # generation uses real current UTC and must reject not-yet-valid facts.
        class FutureClock(datetime):
            @classmethod
            def now(cls, tz=None):
                return stamp + timedelta(days=2)

        with monkeypatch.context() as temporary:
            temporary.setattr("character.context_builder.datetime", FutureClock)
            reference, ids = compile_reference_context(memories, complete_evidence=True, max_chars=None)
    else:
        reference, ids = compile_reference_context(memories, complete_evidence=True, max_chars=None)
    assert set(ids) == {item.memory_id for item in memories}
    context = CompiledCharacterContext(
        "虚构组件画像", "", reference, used_memory_ids=ids, memory_status="available", memory_packets=memories
    )
    prepared = PreparedCharacterTurn(
        character_id="fiction-role",
        user_scope=scope,
        compiled=context,
        history=(),
        relationship=RelationshipState(),
        memory_candidates=count,
        interaction_count=0,
        reply_guard=None,
    )
    public = "独立虚构公共规则：书页登记费用13元。"
    request = GenerationRequest(
        message=query,
        persona_prompt="虚构组件规则",
        context_window_tokens=65536,
        character_context=context,
        retrieval=RetrievalResult(status="ok", evidence=public),
        private_context_revalidator=make_private_context_revalidator(prepared, db),
    )
    return request, repo, scope, selected, historical, deadline, public


async def send_wire(request):
    seen = []

    async def model(**kwargs):
        seen.append(unescape("\n".join(message["content"] for message in kwargs["messages"])))
        return "仅依据仍符合所问时间的资料回答。"

    result = await generate_character_response(request, model)
    assert len(seen) == 1
    return seen[0], result


async def pass_deadline(deadline):
    await asyncio.sleep(max(0, (deadline - datetime.now(timezone.utc)).total_seconds()) + 0.02)
    assert datetime.now(timezone.utc) >= deadline


@pytest.mark.asyncio
async def test_expiry_keeps_independent_current_claim_and_public_evidence(tmp_path, monkeypatch):
    request, _repo, _scope, selected, _historical, deadline, public = await time_request(tmp_path, monkeypatch)
    await pass_deadline(deadline)
    wire, result = await send_wire(request)
    assert selected.content not in wire and BODY.rstrip("。") not in wire
    assert "我喜欢豆浆" in wire and public in wire
    assert selected.memory_id not in result.plan.character_context.used_memory_ids


@pytest.mark.asyncio
async def test_clock_rewind_excludes_prepared_future_fact_without_losing_independent_current_fact(
    tmp_path, monkeypatch
):
    request, _repo, _scope, selected, _historical, _deadline, public = await time_request(
        tmp_path, monkeypatch, future=True
    )
    wire, result = await send_wire(request)
    assert selected.content not in wire and "我喜欢豆浆" in wire and public in wire
    assert selected.memory_id not in result.plan.character_context.used_memory_ids


@pytest.mark.asyncio
async def test_expiry_checked_after_async_storage_reads_complete(tmp_path, monkeypatch):
    from inference.structured_context_authority import revalidate_private_memories

    request, repo, scope, selected, _historical, deadline, _public = await time_request(tmp_path, monkeypatch)
    read = repo.get_memory_record

    async def waited(memory_id, *args):
        row = await read(memory_id, *args)
        if str(memory_id) == selected.memory_id:
            await pass_deadline(deadline)
        return row

    monkeypatch.setattr(repo, "get_memory_record", waited)
    refreshed = await revalidate_private_memories(request, repo, "fiction-role", scope)
    wire, result = await send_wire(refreshed)
    assert selected.content not in wire and "我喜欢豆浆" in wire
    assert selected.memory_id not in result.plan.character_context.used_memory_ids


@pytest.mark.asyncio
async def test_guard_retry_rechecks_time_after_first_reply(tmp_path, monkeypatch):
    from dataclasses import replace

    from character.output_guard import ReplyGuard

    request, _repo, _scope, selected, _historical, deadline, public = await time_request(tmp_path, monkeypatch)
    request = replace(request, reply_guard=ReplyGuard(forbid_laughter=True), reply_guard_mode="strict")
    seen = []

    async def model(**kwargs):
        seen.append(unescape("\n".join(message["content"] for message in kwargs["messages"])))
        if len(seen) == 1:
            await pass_deadline(deadline)
            return "哈哈，已核对。"
        return "只依据当前仍有效的记录回答。"

    result = await generate_character_response(request, model)
    assert len(seen) == 2 and result.guard_retried and not result.guard_fallback
    assert selected.content in seen[0] and selected.content not in seen[1]
    assert "我喜欢豆浆" in seen[1] and public in seen[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("future_plan", [False, True])
async def test_unverified_model_endpoints_do_not_override_source_based_effective_view(tmp_path, future_plan):
    import json
    from types import SimpleNamespace

    from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig, parse_llm_proposals

    db = SQLiteDB(tmp_path / "semantic-view.sqlite")
    repo = DatabaseCharacterMemoryRepository(db)
    scope = UserScope("web", "semantic-boundary", "fiction-owner", "fiction-owner", "private")
    stamp = datetime.now(timezone.utc) - timedelta(days=1)
    source = (
        "从2099年1月1日起，我喜欢乌龙茶；只有天气晴朗时才选择。这是未来计划，不证明当前偏好或实际饮用。"
        if future_plan
        else BODY
    )
    assert (
        await repo.capture_source(
            "fiction-role", scope, source_message_id="fiction-semantic-view", body=source, observed_at=stamp
        )
        == "recorded"
    )
    raw = dict(
        kind="like",
        value="乌龙茶",
        content="用户喜欢乌龙茶",
        evidence=source,
        confidence=0.95,
        operation="ADD",
        attributed_to="user",
        valid_from="2099-01-01T00:00:00+00:00",
        valid_to="2100-01-01T00:00:00+00:00",
        qualifiers=dict(time="2099年1月1日", condition="只有天气晴朗时才选择", certainty="不证明当前偏好或实际饮用")
        if future_plan
        else {},
    )
    proposals = parse_llm_proposals(
        json.dumps(dict(memories=[raw]), ensure_ascii=False), source_message=source, existing_memories=()
    )
    assert len(proposals) == 1
    scheduler = MemoryEnrichmentScheduler(
        config=MemoryLlmConfig(True, "component-only", "component-only"), completion=None
    )
    job = SimpleNamespace(
        repository=repo,
        character_id="fiction-role",
        user_scope=scope,
        message=source,
        observed_at=stamp,
        source_message_id="fiction-semantic-view",
        write_mode="hot",
        feedback_target_ids=(),
    )
    assert await scheduler._persist_proposal(job, proposals[0]) == "saved"
    query = (
        "读取我保存的乌龙茶未来计划原话，完整保留条件与句尾不证明当前偏好的说明，不将未来计划当成当前事实。"
        if future_plan
        else "我当前保存的乌龙茶偏好是什么？只根据本人可核验记录，不推断实际饮用。"
    )
    memories, count = await CharacterMemoryService(repo, semantic_enabled=False).load_relevant_memories(
        "fiction-role", scope, query
    )
    assert count == len(memories) == 1
    selected = memories[0]
    assert selected.storage_versions and selected.source_versions and not selected.valid_to
    assert selected.temporal_mode == ("observation" if future_plan else "asserted_state")
    reference, ids = compile_reference_context(
        memories, complete_evidence=True, observation_semantics=True, max_chars=None
    )
    assert source in reference
    context = CompiledCharacterContext(
        "虚构组件画像", "", reference, used_memory_ids=ids, memory_status="available", memory_packets=memories
    )
    prepared = PreparedCharacterTurn(
        character_id="fiction-role",
        user_scope=scope,
        compiled=context,
        history=(),
        relationship=RelationshipState(),
        memory_candidates=count,
        interaction_count=0,
        reply_guard=None,
    )
    request = GenerationRequest(
        message=query,
        persona_prompt="虚构组件规则",
        context_window_tokens=65536,
        character_context=context,
        private_context_revalidator=make_private_context_revalidator(prepared, db),
    )
    wire, result = await send_wire(request)
    assert source in wire and selected.memory_id in result.plan.character_context.used_memory_ids


@pytest.mark.asyncio
async def test_expired_current_packet_does_not_remove_independent_historical_version(tmp_path, monkeypatch):
    request, _repo, _scope, selected, historical, deadline, public = await time_request(
        tmp_path, monkeypatch, mixed=True
    )
    await pass_deadline(deadline)
    wire, result = await send_wire(request)
    assert selected.content not in wire and historical.content in wire
    assert historical.memory_id in result.plan.character_context.used_memory_ids
    assert selected.memory_id not in result.plan.character_context.used_memory_ids
    assert "我喜欢豆浆" in wire and public in wire


@pytest.mark.parametrize(
    "query,expected",
    [
        (
            "我过去保存的乌龙茶偏好是什么？我当前的乌龙茶偏好是什么？我当前的豆浆偏好是什么？只依据各自时间的本人记录，不证明现实审核完成。",
            [("乌龙茶", "过去"), ("乌龙茶", "当前"), ("豆浆", "当前")],
        ),
        (
            "读取我当前仍在所存有效区间的乌龙茶偏好，返回expired_value与expired_quote。不要从过去缓存补齐，也不推断曾经缺失的事实。",
            [("乌龙茶", "当前")],
        ),
        ("我的当前乌龙茶和豆浆偏好分别是什么？", [("乌龙茶", "当前"), ("豆浆", "当前")]),
        ("以前我的乌龙茶偏好是什么？现在我的豆浆偏好是什么？", [("乌龙茶", "以前"), ("豆浆", "现在")]),
        ("我朋友过去的乌龙茶偏好是什么？", []),
        ("他说“我当前的乌龙茶偏好是什么”，这是什么意思？", []),
        ("我当前的乌龙茶偏好是已经喝过。", []),
        ("不是当前的乌龙茶偏好，不要把过去当作现在。", []),
    ],
)
def test_preference_time_views_use_explicit_own_lookup_not_unrelated_history_words(query, expected):
    from character.preference_time_views import preference_time_tasks

    tasks = preference_time_tasks(query)
    actual = [(subject, task.time_expression) for task in tasks for subject in task.subjects]
    assert actual == expected
    for task in tasks:
        assert task.query in query
        assert task.matches(dict(memory_key="preference_" + task.subjects[0]))
        assert not task.matches(dict(memory_key="user_name"))
        assert not task.matches(dict(memory_key="preference_无关完整对象"))
