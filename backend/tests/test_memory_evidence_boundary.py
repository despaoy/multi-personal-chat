"""Model quote bounds precede admission; derived complete observations preserve sources."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from repositories.character_memory import DatabaseCharacterMemoryRepository

from character.memory_llm import MemoryEnrichmentScheduler, MemoryLlmConfig, MemoryResponseError, parse_llm_proposals
from character.models import UserScope
from db.database import SQLiteDB

HISTORY = ({"role": "user", "content": "我最近在准备保研面试。"},)
CURRENT = "还是上次那个方向。"


def candidate(evidence, **changes):
    return dict(
        dict(kind="goal", value="保研面试", evidence=evidence, operation="ADD", confidence=0.95, attributed_to="user"),
        **changes,
    )


def parse(raw, source=CURRENT, history=HISTORY):
    return parse_llm_proposals(json.dumps(dict(memories=[raw])), source_message=source, history=history)


def test_copied_historical_quote_is_not_rewritten_into_new_evidence():
    with pytest.raises(MemoryResponseError) as caught:
        parse(candidate(HISTORY[0]["content"]))
    assert caught.value.field == "evidence"


def test_current_quote_can_still_use_user_history_to_resolve_value():
    (proposal,) = parse(candidate(CURRENT))
    assert proposal.evidence == CURRENT and proposal.memory.content == "用户正在进行或准备：保研面试"
    with pytest.raises(MemoryResponseError, match="value"):
        parse(candidate(CURRENT), history=())
    with pytest.raises(MemoryResponseError, match="value"):
        parse(candidate(CURRENT), history=({"role": "assistant", "content": HISTORY[0]["content"]},))


@pytest.mark.parametrize("confidence", [0.1, 0.95])
def test_oversized_model_quote_fails_before_business_filtering(confidence):
    quote = "我计划学习陶艺，" + "甲" * 113
    assert len(quote) == 121
    with pytest.raises(MemoryResponseError) as caught:
        parse(candidate(quote, kind="shared_event", value="陶艺", confidence=confidence), source=quote, history=())
    assert caught.value.field == "evidence" and quote not in str(caught.value)


def test_exact_quote_limit_and_complete_long_source_observation_remain_supported():
    quote = "我计划学习陶艺，" + "甲" * 112
    assert len(quote) == 120
    source = quote + "；当前只是计划，只有收到书面录取后才考虑出发，目前尚未收到。"
    (proposal,) = parse(candidate(quote, kind="shared_event", value="陶艺"), source=source, history=())
    assert proposal.evidence == source and proposal.source_observation
    assert "完整内容见证据" in proposal.memory.content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case,expected,reason",
    [
        ("copied", "failed", "invalid_model_response"),
        ("empty", "no_change", "no_new_information"),
        ("current", "saved", None),
        ("oversized", "failed", "invalid_model_response"),
    ],
)
async def test_receipt_distinguishes_quote_failure_admission_and_empty_extraction(tmp_path, case, expected, reason):
    database = SQLiteDB(tmp_path / "evidence.sqlite")
    repository = DatabaseCharacterMemoryRepository(database)
    scope = UserScope("web", "evidence-boundary", "reader", "reader", "private")
    memories = [] if case == "empty" else [candidate(HISTORY[0]["content"] if case == "copied" else CURRENT)]
    if case == "oversized":
        memories.append(candidate("甲" * 121))
    completion = SimpleNamespace(
        complete=AsyncMock(return_value=json.dumps(dict(memories=memories))), close=AsyncMock()
    )
    worker = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True, "unused", "fixture"), completion=completion)
    try:
        receipt = await worker.schedule_and_wait(
            repository=repository,
            character_id="role",
            user_scope=scope,
            message=CURRENT,
            history=HISTORY,
            rule_hints=(),
            source_message_id="current",
        )
        assert receipt["status"] == expected and receipt["source_capture"] == "recorded"
        records = await repository.list_memory_records("role", scope)
        if case == "current":
            assert receipt["persisted"] == len(records) == 1
        else:
            assert receipt["reason"] == reason and receipt["accepted"] == receipt["persisted"] == 0 and not records
        if case in {"copied", "oversized"}:
            assert receipt["error"] == "MemoryResponseError" and receipt["field"] == "evidence"
            assert receipt["stage"] == "proposal_validation" and receipt["retryable"] is False
        assert len(await repository.list_sources("role", scope)) == 1
        completion.complete.assert_awaited_once()
    finally:
        await worker.shutdown(timeout=1)
        database.close_connection()


def unit_payload(source):
    from character.memory_llm import build_memory_llm_messages
    payload = json.loads(build_memory_llm_messages(source, (), (), (), 2000, .85)[-1]["content"])
    assert payload["current_user_message"] == source
    assert "qualifier_sources" not in payload
    return payload["source_units"]


@pytest.mark.parametrize("source", ["我叫秋禾？", "我的专业是考古学，你还记得我叫什么吗？",
    "  每天09:30以后，我喜欢紫茶。\n只有预约确认才饮用。"])
def test_reference_units_preserve_complete_original_punctuation(source):
    assert "".join(unit_payload(source).values()) == source


def test_explicit_reference_selects_the_statement_without_rewriting_it():
    source = "我的专业是考古学，你还记得我叫什么吗？"
    units = unit_payload(source)
    key = next(key for key, text in units.items() if "专业是考古学" in text)
    raw = candidate({"source_units": [key]}, kind="major", value="考古学", content="用户记录考古学")
    (proposal,) = parse(raw, source=source, history=())
    assert proposal.evidence == units[key] and proposal.memory.content == "用户记录考古学"
    assert "？" not in proposal.evidence


@pytest.mark.parametrize("references", [[], ["unknown"], ["q1", "q0"], ["q0", "q0"], ["q0", "q2"]])
def test_reference_evidence_cannot_guess_skip_repeat_or_reorder_sources(references):
    source = "我的专业是考古学。我的目标是保研。我的狗叫年糕。"
    with pytest.raises(MemoryResponseError) as caught:
        parse(candidate({"source_units": references}, kind="major", value="考古学"), source=source, history=())
    assert caught.value.field == "evidence"


def test_original_overwide_quote_is_an_error_instead_of_a_rule_repair():
    source = "我的专业是考古学，你还记得我叫什么吗？"
    with pytest.raises(MemoryResponseError) as caught:
        parse(candidate(source, kind="major", value="考古学"), source=source, history=())
    assert caught.value.field == "evidence"


def test_question_reference_is_not_an_assertion_after_reference_resolution():
    source = "我叫秋禾？"
    with pytest.raises(MemoryResponseError) as caught:
        parse(candidate({"source_units": list(unit_payload(source))}, kind="name", value="秋禾"), source=source, history=())
    assert caught.value.field == "evidence"


async def test_bad_question_proposal_rejects_the_actual_batch_before_fact_writes(tmp_path):
    source = "我的专业是考古学。我叫秋禾？"
    units = unit_payload(source)
    assert units["q0"] == "我的专业是考古学。"
    assert units["q1"] == "我叫秋禾？"
    proposals = [dict(kind="major", value="考古学", evidence={"source_units": ["q0"]}, content="用户记录考古学",
        operation="ADD", attributed_to="user", confidence=.99),
        dict(kind="name", value="秋禾", evidence={"source_units": ["q1"]}, content="",
        operation="ADD", attributed_to="user", confidence=.99)]
    database = SQLiteDB(tmp_path / "reference-batch.sqlite")
    repository = DatabaseCharacterMemoryRepository(database)
    scope = UserScope("test", "reference", "fiction", "fiction", "private")
    completion = SimpleNamespace(complete=AsyncMock(return_value=json.dumps({"memories": proposals}, ensure_ascii=False)), close=AsyncMock())
    worker = MemoryEnrichmentScheduler(config=MemoryLlmConfig(True, "unused", "fixture"), completion=completion)
    try:
        receipt = await worker.schedule_and_wait(repository=repository, character_id="role", user_scope=scope,
            message=source, rule_hints=(), source_message_id="reference-source")
        assert receipt["status"] == "failed" and receipt["field"] == "evidence"
        assert receipt["accepted"] == receipt["persisted"] == 0
        assert await repository.list_memory_records("role", scope, include_inactive=True) == []
        assert (await repository.list_sources("role", scope))[0]["body"] == source
        completion.complete.assert_awaited_once()
    finally:
        await worker.shutdown(timeout=1)
        database.close_connection()


def test_contiguous_evidence_and_qualifiers_share_complete_original_units():
    source = "我喜欢紫茶，只有预约确认才饮用；我还没有喝。你记得吗？"
    units = unit_payload(source)
    raw = candidate({"source_units": ["q0", "q1", "q2"]}, kind="like", value="紫茶", content="",
        qualifiers={"condition": {"source_unit": "q1"}, "certainty": {"source_unit": "q2"}})
    (proposal,) = parse(raw, source=source, history=())
    assert proposal.evidence == "".join(units[key] for key in ("q0", "q1", "q2"))
    assert dict(proposal.qualifiers) == {"condition": "只有预约确认才饮用", "certainty": "我还没有喝"}
    assert "只有预约确认才饮用" in proposal.memory.content and "我还没有喝" in proposal.memory.content
    assert "？" not in proposal.evidence
