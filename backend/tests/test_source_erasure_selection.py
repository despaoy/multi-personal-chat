import json
from datetime import datetime, timezone

import pytest
from repositories.character_memory import DatabaseCharacterMemoryRepository

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig
from character.models import MemoryItem, UserScope
from character.source_erasure_selection import candidates, selected_ids
from db.database import SQLiteDB
from test_memory_source_inheritance import source_database as source_database
from inference.token_counting import text_token_counter

SCOPE = UserScope('web', 'test', 'user', 'user', 'private')


def test_known_source_requires_erasure_authority():
    with pytest.raises(ValueError):
        selected_ids({'erase_source_ids': ['known']}, {'known'}, authorized=False)


async def test_source_only_target_uses_same_writer_call_and_committed_receipt(tmp_path):
    db = SQLiteDB(tmp_path / 'source.sqlite')
    repo = DatabaseCharacterMemoryRepository(db)
    for source, body in [('pet', '我的狗叫松糕。'), ('other', '我周末喜欢参观美术馆。')]:
        await repo.capture_source('role', SCOPE, source_message_id=source, body=body,
                                  observed_at=datetime.now(timezone.utc))

    class Completion:
        calls = 0

        async def complete(self, messages):
            self.calls += 1
            data = json.loads(messages[1]['content'])
            assert data['existing_memories'] == []
            assert any(r['source_id'] == 'pet' and r['text'] == '我的狗叫松糕。'
                       for r in data['source_erasure_candidates'])
            return '{"memories":[],"erase_source_ids":["pet"]}'

        async def close(self):
            pass

    completion = Completion()
    scheduler = MemoryEnrichmentScheduler(config=MemoryLlmConfig(enabled=True, base_url='http://unused',
        model='test', context_window_tokens=65536), completion=completion)
    try:
        receipt = await scheduler.schedule_and_wait(repository=repo, character_id='role', user_scope=SCOPE,
            message='请把我的狗的名字从记忆里删除。', rule_hints=(), history=(), source_message_id='delete')
        assert receipt['status'] == 'erased' and receipt['source_erased'] == 1
        from character.memory_operation import render_operation_response

        reply = render_operation_response('请把我的狗的名字从记忆里删除。', receipt)
        assert '已删除本次匹配的记录' in reply and '未逐一核对全部历史' in reply
        assert completion.calls == 1
        assert [r['source_message_id'] for r in await repo.list_sources('role', SCOPE)] == ['other']
        assert await repo.list_memory_records('role', SCOPE) == []
    finally:
        await scheduler.shutdown(timeout=2)


async def test_candidate_budget_keeps_full_sources_and_reports_omissions():
    class Repo:
        async def search_sources(self, *args, **kwargs):
            return [dict(source_message_id='long', body='甲' * 5000)]

        async def list_sources(self, *args, **kwargs):
            return [dict(source_message_id='short', body='完整短原话')]

    rows, coverage = await candidates(
        Repo(),
        "role",
        SCOPE,
        "删除",
        context_window_tokens=8192,
        count_tokens=text_token_counter("openai_compat", "http://unused", "test"),
    )
    assert rows == (dict(source_id='short', text='完整短原话', observed_at=None),)
    assert coverage['omitted'] == 1 and coverage['complete'] is False


@pytest.mark.parametrize('value', [None, False, 0, '', {}, 'known', ['foreign'], [1]])
def test_invalid_source_selection_is_not_empty_success(value):
    with pytest.raises(ValueError):
        selected_ids({'erase_source_ids': value}, {'known'}, authorized=True)


@pytest.mark.parametrize('payload', [{}, {'erase_source_ids': []}])
def test_explicit_or_omitted_empty_selection_needs_no_erasure_authority(payload):
    assert selected_ids(payload, set(), authorized=False) == ()


@pytest.mark.parametrize('message,selection', [
    ('请记住，我的专业是物理学。', ['foreign']),
    ('请记住，我的专业是物理学。', None),
    ('请把我的狗的名字从记忆里删除。', ['foreign']),
    ('请把我的狗的名字从记忆里删除。', False),
], ids=['ordinary-foreign', 'ordinary-null', 'erase-foreign', 'erase-false'])
async def test_scheduler_rejects_invalid_selection_even_without_candidates(tmp_path, message, selection):
    database = SQLiteDB(tmp_path / 'invalid-selection.sqlite')
    repository = DatabaseCharacterMemoryRepository(database)

    class Completion:
        async def complete(self, messages):
            payload = json.loads(messages[1]['content'])
            assert not payload.get('source_erasure_candidates')
            memories = [] if '删除' in message else [dict(kind='major', value='物理学',
                evidence='我的专业是物理学', operation='ADD', confidence=0.99, attributed_to='user')]
            return json.dumps(dict(memories=memories, erase_source_ids=selection))

        async def close(self):
            pass

    scheduler = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True, 'unit', 'unit'), completion=Completion())
    try:
        receipt = await scheduler.schedule_and_wait(repository=repository, character_id='role', user_scope=SCOPE,
            message=message, rule_hints=(), source_message_id='request')
        assert receipt['status'] == 'failed' and receipt['stage'] == 'proposal_validation'
        assert receipt['accepted'] == receipt['persisted'] == 0
        assert await repository.list_memory_records('role', SCOPE) == []
        assert 'foreign' not in json.dumps(receipt)
        sources = await repository.list_sources('role', SCOPE)
        assert len(sources) == (0 if '删除' in message else 1)
    finally:
        await scheduler.shutdown(timeout=1)
        database.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["source", "claim", "claim_mixed", "protected", "foreign", "missing"])
async def test_scoped_erasure_retains_original_speech(source_database, mode):
    import uuid
    from dataclasses import replace
    from types import SimpleNamespace
    from datetime import timedelta
    from character.memory_subject import is_source_observation

    database = source_database
    repository = DatabaseCharacterMemoryRepository(database)
    owner = "synthetic-" + uuid.uuid4().hex
    scope = UserScope("web", "retention-fixture", owner, owner, "private")
    foreign = replace(scope, sender_id=owner + "-foreign", conversation_id=owner + "-foreign")
    observed = datetime(2026, 10, 1, tzinfo=timezone.utc)
    red, green = "我喜欢测试红茶。", "我喜欢测试绿茶。"
    kept_body = red + green if mode == "claim_mixed" else green
    for user_scope, identity, body in [(scope, "old", red), (scope, "other", kept_body), (foreign, "foreign", green)]:
        await repository.capture_source("role", user_scope, source_message_id=identity, body=body, observed_at=observed)
    claim = None
    if mode in {"claim", "claim_mixed"}:
        claim = await repository.append_claim(
            "role",
            scope,
            MemoryItem("", "user_fact", "用户喜欢测试红茶", 0.9),
            memory_key="preference_测试红茶",
            source_message_id="old",
            source_message_ids=("old",),
            evidence=(red,),
            observed_at=observed.isoformat(),
        )

    class Completion:
        calls = 0

        async def complete(self, messages):
            self.calls += 1
            payload = json.loads(messages[1]["content"])
            assert payload["current_user_message"] == message
            assert any(
                row["source_id"] == "other" and row["text"] == kept_body for row in payload["source_erasure_candidates"]
            )
            assert all(row["source_id"] != "foreign" for row in payload["source_erasure_candidates"])
            result = dict(memories=[], erase_source_ids=[] if claim else ["other" if mode == "protected" else "old"])
            if claim:
                result["memories"] = [
                    dict(
                        kind="like",
                        value="测试红茶",
                        content="",
                        evidence="请从长期记忆中删除我喜欢测试红茶的记录",
                        confidence=0.99,
                        operation="ERASE",
                        target_memory_id=str(claim["id"]),
                        target_memory_key=claim["memory_key"],
                        attributed_to="user",
                    )
                ]
            if mode != "missing":
                result["source_retention_bindings"] = [
                    dict(clause="保留测试绿茶的记录", source_ids=["foreign" if mode == "foreign" else "other"])
                ]
            return json.dumps(result, ensure_ascii=False)

        async def close(self):
            pass

    message = "请从长期记忆中删除我喜欢测试红茶的记录，保留测试绿茶的记录，同时计算12+8。"
    completion = Completion()
    scheduler = MemoryEnrichmentScheduler(
        config=MemoryLlmConfig(True, "unit", "unit", context_window_tokens=65536),
        completion=completion,
        embedding_provider=SimpleNamespace(embed_texts=lambda texts: [[1.0, 0.0] for _ in texts]),
    )
    try:
        receipt = await scheduler.schedule_and_wait(
            repository=repository,
            character_id="role",
            user_scope=scope,
            message=message,
            rule_hints=(),
            source_message_id="request",
            observed_at=observed + timedelta(days=1),
        )
        if mode in {"source", "claim", "claim_mixed"}:
            assert receipt["status"] == "erased" and receipt["persisted"] == 1, receipt
            if mode == "source":
                remaining = await repository.list_sources("role", scope)
                assert [(row["source_message_id"], row["body"]) for row in remaining] == [("other", green)]
            else:
                records = await repository.list_memory_records("role", scope, limit=None)
                assert len(records) == 1 and is_source_observation(records[0])
                assert records[0]["evidence"] == [green] and records[0]["source_message_ids"] == ["other"]
                assert datetime.fromisoformat(records[0]["observed_at"]) == observed
                assert not await repository.get_memory_record(claim["id"], "role", scope)
                from character.memory_service import CharacterMemoryService
                from character.context_builder import compile_reference_context

                memory = CharacterMemoryService(
                    repository,
                    embedding_provider=SimpleNamespace(embed_texts=lambda texts: [[1.0, 0.0] for _ in texts]),
                )
                selected, _ = await memory.load_relevant_memories(
                    "role", scope, "根据保存的原话，我喜欢什么测试茶？", for_contextual_selection=True
                )
                assert len(selected) == 1 and selected[0].source_observation
                assert selected[0].source_message_ids == ("other",) and selected[0].complete_original_source is False
                reference, used = compile_reference_context(selected, complete_evidence=True)
                assert green in reference and red not in reference and used == (str(records[0]["id"]),)
        else:
            assert receipt["status"] == "failed" and receipt["persisted"] == receipt["accepted"] == 0
            assert {row["source_message_id"] for row in await repository.list_sources("role", scope)} == {
                "old",
                "other",
            }
        assert completion.calls == 1
        assert [(row["source_message_id"], row["body"]) for row in await repository.list_sources("role", foreign)] == [
            ("foreign", green)
        ]
    finally:
        await scheduler.shutdown(timeout=2)


@pytest.mark.asyncio
async def test_unlinked_source_filter_uses_actual_links_before_limit(source_database):
    import uuid
    from datetime import timedelta

    repository = DatabaseCharacterMemoryRepository(source_database)
    owner = "query-" + uuid.uuid4().hex
    scope = UserScope("web", "unlinked-fixture", owner, owner, "private")
    now = datetime.now(timezone.utc)
    for identity, body, clock in [
        ("free", "合成测试：独立保存的原话。", now - timedelta(days=1)),
        ("old", "合成测试：已关联的旧原话。", now),
        ("new", "合成测试：已关联的新原话。", now + timedelta(seconds=1)),
    ]:
        await repository.capture_source("role", scope, source_message_id=identity, body=body, observed_at=clock)
    old = await repository.append_claim(
        "role",
        scope,
        MemoryItem("", "user_fact", "合成测试旧记录", 0.5),
        memory_key="fixture",
        source_message_id="old",
        evidence=("合成测试：已关联的旧原话。",),
        observed_at=now.isoformat(),
    )
    await repository.append_claim(
        "role",
        scope,
        MemoryItem("", "user_fact", "合成测试新记录", 0.5),
        memory_key="fixture",
        relation_type="SUPERSEDE",
        supersedes_memory_id=old["id"],
        source_message_id="new",
        source_message_ids=("old", "new"),
        evidence=("合成测试：已关联的旧原话。", "合成测试：已关联的新原话。"),
        observed_at=now.isoformat(),
    )
    assert [
        row["source_message_id"] for row in await repository.list_sources("role", scope, limit=1, unlinked_only=True)
    ] == ["free"]
    assert [
        row["source_message_id"]
        for row in await repository.search_sources("role", scope, query="合成测试", limit=1, unlinked_only=True)
    ] == ["free"]
    selected, _ = await candidates(repository, "role", scope, "删除合成测试", context_window_tokens=65536)
    assert [row["source_id"] for row in selected] == ["free"]


@pytest.mark.asyncio
async def test_retained_speech_and_erasure_share_transaction_rollback(source_database, monkeypatch):
    import uuid
    from db import retained_source_observation

    repository = DatabaseCharacterMemoryRepository(source_database)
    owner = "rollback-" + uuid.uuid4().hex
    scope = UserScope("web", "retain-rollback", owner, owner, "private")
    observed = datetime(2026, 10, 1, tzinfo=timezone.utc)
    for identity, body in [("old", "我喜欢测试红茶。"), ("other", "我喜欢测试绿茶。")]:
        await repository.capture_source("role", scope, source_message_id=identity, body=body, observed_at=observed)
    claim = await repository.append_claim(
        "role",
        scope,
        MemoryItem("", "user_fact", "用户喜欢测试红茶", 0.5),
        memory_key="preference_测试红茶",
        source_message_id="old",
        source_message_ids=("old",),
        evidence=("我喜欢测试红茶。",),
        observed_at=observed.isoformat(),
    )
    original = retained_source_observation.apply_observations
    inserted = []

    def fail_after_insert(plans):
        plans = list(plans)
        assert len(plans) == 1
        yield from original(plans)
        inserted.append(True)
        raise RuntimeError("injected preservation failure")

    monkeypatch.setattr(retained_source_observation, "apply_observations", fail_after_insert)
    with pytest.raises(RuntimeError, match="injected preservation failure"):
        await repository.erase_memory("role", scope, memory_id=claim["id"], protected_source_message_ids=("other",))
    assert inserted == [True]
    records = await repository.list_memory_records("role", scope, limit=None)
    assert [row["id"] for row in records] == [claim["id"]]
    assert {row["source_message_id"]: row["body"] for row in await repository.list_sources("role", scope)} == {
        "old": "我喜欢测试红茶。",
        "other": "我喜欢测试绿茶。",
    }
