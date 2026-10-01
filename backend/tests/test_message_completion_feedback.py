"""Only trusted archive receipt feedback, actual SQLite/PG and its consumers."""

import os
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from character.context_builder import build_user_scope
from db.database import SQLiteDB
from db.schemas import MessageRequest

WARNING = "这条信息的长期记忆保存失败；本次回复不代表已保存，请稍后重试。"
BODY = "合成测试：朋友陆溪收到北岚青石工作室纸浆浮雕课确认，私人回执YP-406-R，仍有效，未参加、未出发。这是朋友的记录，不是我本人的预约。"


@pytest.fixture(scope="module", params=["sqlite", "pg"])
def database(request, tmp_path_factory):
    if request.param == "sqlite":
        db = SQLiteDB(tmp_path_factory.mktemp("feedback") / "records.sqlite")
    else:
        url = os.environ["STAGE31_FEEDBACK_PG_URL"]
        assert "/stage3_stage31_feedback_guards?" in url and "port=25433" in url
        # This legacy module also constructs an unused default adapter during
        # import. Bind that constructor to the same verified disposable DB.
        with pytest.MonkeyPatch.context() as scoped_env:
            scoped_env.setenv("DATABASE_URL", url)
            from db.pg_database import PgDatabase, SyncPgAdapter
        db = SyncPgAdapter(PgDatabase(url))
    yield db
    (db.close if request.param == "pg" else db.close_connection)()


def archive(database, **changes):
    owner = "owner-" + uuid.uuid4().hex
    record = dict(
        platform="web",
        adapter="web-character",
        senderId=owner,
        conversationType="private",
        conversationId=owner,
        characterId="tsukiyashiro_kisaki",
        sessionId=owner,
        sessionType="private",
        sourceMessageId="new-" + uuid.uuid4().hex,
        traceId="trace-" + uuid.uuid4().hex,
        message=BODY,
        reply="真实测试模型原文",
        branchId=None,
    )
    record.update(changes)
    return database.add_message(record)


def row(database, receipt):
    return next(r for r in database.get_messages(limit=1000) if str(r["id"]) == str(receipt["id"]))


def history(database, receipt):
    return database.list_conversation_history(
        receipt["platform"],
        receipt["adapter"],
        receipt["senderId"],
        receipt["conversationType"],
        receipt["conversationId"],
        character_id=receipt["characterId"],
    )


def test_exact_receipt_finalizes_once_and_history_reloads(database):
    from db.message_feedback import reply_with_warning

    receipt = archive(database)
    old = row(database, receipt)
    assert database.update_message_feedback(receipt, warning=WARNING)
    assert database.update_message_feedback(receipt, warning=WARNING)
    final = row(database, receipt)
    assert final["reply"] == reply_with_warning(receipt["reply"], WARNING) and final["reply"].count("保存提示：") == 1
    assert {k: v for k, v in final.items() if k != "reply"} == {k: v for k, v in old.items() if k != "reply"}
    assert history(database, receipt)[-1]["content"] == final["reply"]
    scope = build_user_scope(
        receipt["platform"],
        receipt["adapter"],
        receipt["senderId"],
        receipt["conversationId"],
        receipt["conversationType"],
    )
    page = database.list_scoped_conversation_turns(scope, receipt["characterId"])
    assert page.turns[-1].utterances[-1].text == final["reply"]


@pytest.mark.parametrize(
    "field",
    [
        "id",
        "platform",
        "adapter",
        "senderId",
        "conversationType",
        "conversationId",
        "characterId",
        "sessionId",
        "sessionType",
        "sourceMessageId",
        "traceId",
        "message",
        "reply",
    ],
)
def test_mismatched_receipt_cannot_change_any_row(database, field):
    receipt = archive(database)
    other = archive(database, sourceMessageId=receipt["sourceMessageId"], message=BODY, reply=receipt["reply"])
    before = [row(database, r) for r in [receipt, other]]
    wrong = {
        **receipt,
        field: other["id"]
        if field == "id"
        else "group"
        if field in {"conversationType", "sessionType"}
        else receipt[field] + "-wrong",
    }
    assert not database.update_message_feedback(wrong, warning=WARNING)
    assert [row(database, r) for r in [receipt, other]] == before


def test_deleted_receipt_is_not_recreated(database):
    receipt = archive(database)
    assert database.delete_message(int(receipt["id"]))
    assert not database.update_message_feedback(receipt, warning=WARNING)
    assert not any(str(r["id"]) == receipt["id"] for r in database.get_messages(limit=1000))


def test_intervening_reply_is_not_overwritten(database):
    receipt = archive(database)
    assert database.update_message_feedback(receipt, warning="先前合法保存提示")
    before = row(database, receipt)
    assert not database.update_message_feedback(receipt, warning=WARNING)
    assert row(database, receipt) == before


@pytest.mark.parametrize("changes", [dict(branchId="untrusted-branch"), dict(adapter="narrative"), dict(senderId="")])
def test_missing_or_branch_authority_never_writes(database, changes):
    receipt = archive(database, **changes)
    before = row(database, receipt)
    with pytest.raises(ValueError):
        database.update_message_feedback(receipt, warning=WARNING)
    assert row(database, receipt) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["vllm", "openai_compat"])
@pytest.mark.parametrize("mode", ["failed", "recorded", "feedback_store_failed"])
async def test_both_consumers_use_actual_saved_row_and_keep_visible_failure(monkeypatch, database, provider, mode):
    import asyncio

    from api import generate as gen
    from inference import model_manager as mm

    owner = "owner-" + uuid.uuid4().hex
    request = MessageRequest(
        message=BODY,
        characterId="tsukiyashiro_kisaki",
        platform="web",
        adapter="web-character",
        senderId=owner,
        sourceMessageId="new-" + uuid.uuid4().hex,
        sessionId=owner,
        conversationId=owner,
        sessionType="private",
        traceId="trace-" + uuid.uuid4().hex,
    )
    manager = SimpleNamespace(
        _current_provider=SimpleNamespace(value=provider),
        set_lora_adapter=lambda _: None,
        get_status=lambda: dict(currentProvider=provider, providers={provider: dict(modelName="unit-model")}),
    )
    monkeypatch.setattr(mm, "get_model_manager", lambda: manager)
    monkeypatch.setattr(gen, "INPUT_VALIDATOR_AVAILABLE", False)
    monkeypatch.setattr(gen, "response_cache", None)
    monkeypatch.setattr(gen, "circuit_breaker_registry", None)
    monkeypatch.setattr(gen, "db", database)
    monkeypatch.setattr(gen, "get_llm_semaphore", lambda: asyncio.Semaphore(1))
    monkeypatch.setattr(gen, "_ensure_vllm", AsyncMock(return_value=provider == "vllm"))
    monkeypatch.setattr(gen, "_vllm_client", object())
    meta = dict(warnings=["原有检索提示"], answerMode="unit", confidence=0.3, citations=[])
    monkeypatch.setattr(gen, "_generate_with_vllm", AsyncMock(return_value=("原始模型内容", False, meta)))
    monkeypatch.setattr(gen, "_generate_with_retrieval", AsyncMock(return_value=("原始模型内容", False, meta)))
    prepared = SimpleNamespace(
        character_id=request.characterId,
        compiled=SimpleNamespace(profile_context="", dynamic_context="", reference_context=""),
        history=(),
    )
    monkeypatch.setattr(gen, "_prepare_character_turn", AsyncMock(return_value=prepared))
    service = SimpleNamespace(
        complete_turn=AsyncMock(
            return_value=SimpleNamespace(
                source_capture="recorded" if mode == "recorded" else "failed", memory_enrichment_status="skipped"
            )
        )
    )
    writer = database.update_message_feedback
    receipts = []

    def observe(receipt, *, warning):
        receipts.append(dict(receipt))
        if mode == "feedback_store_failed":
            raise RuntimeError("private-store-error")
        return writer(receipt, warning=warning)

    monkeypatch.setattr(database, "update_message_feedback", observe)
    result = await gen._generate_reply_impl(
        request,
        persist_message=True,
        enable_rag=False,
        record_invocation=False,
        character_service=service,
        message_db=database,
    )
    stored = [r for r in database.get_messages(limit=1000) if r["traceId"] == request.traceId]
    assert len(stored) == 1 and stored[0]["message"] == BODY and result.warnings[0] == "原有检索提示"
    assert result.confidence == 0.3 and result.citations == [] and meta["warnings"] == ["原有检索提示"]
    if mode == "recorded":
        assert result.reply == stored[0]["reply"] == "原始模型内容" and receipts == []
    else:
        assert (
            receipts[0]["id"] == str(stored[0]["id"])
            and receipts[0]["senderId"] == owner
            and receipts[0]["reply"] == "原始模型内容"
        )
        assert "保存失败" in result.reply and result.warnings[-1] in result.reply
        if mode == "failed":
            assert stored[0]["reply"] == result.reply and history(database, stored[0])[-1]["content"] == result.reply
        else:
            assert (
                stored[0]["reply"] == "原始模型内容"
                and "刷新后可能看不到" in result.reply
                and "private-store-error" not in result.reply
            )
