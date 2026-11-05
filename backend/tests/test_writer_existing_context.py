"""A writer must receive the stored basis and conditions, not a clipped display."""

import json
from copy import deepcopy
from datetime import datetime, timezone

import pytest

from character.memory_llm import build_memory_llm_messages
from character.models import MemoryItem, UserScope
from db.database import SQLiteDB
from inference.context_budget import CONTEXT_SAFETY_MARGIN_TOKENS, InputBudgetError
from repositories.character_memory import DatabaseCharacterMemoryRepository

SOURCE = "我只在工作日早晨喝咖啡。" + "这是当时完整的背景说明。" * 180 + "周末和夜间我不喝咖啡；休假期间也不喝。"
QUALIFIERS = {"condition": "只在工作日早晨", "context": "周末和夜间我不喝咖啡；休假期间也不喝"}
CONTENT = "用户的咖啡习惯：" + "完整背景。" * 30 + "只限工作日早晨，周末、夜间及休假期间不喝。"
CURRENT = "请记住：出差期间我周六早晨也喝咖啡，只补充这个例外，以前的日常限制不变。"


@pytest.fixture
async def stored(tmp_path):
    database = SQLiteDB(tmp_path / "writer-context.sqlite")
    repository = DatabaseCharacterMemoryRepository(database)
    scope = UserScope("web", "wire-context", "owner", "room", "private")
    observed = datetime(2026, 11, 4, 8, tzinfo=timezone.utc)
    assert (
        await repository.capture_source(
            "role", scope, source_message_id="existing-complete-source", body=SOURCE, observed_at=observed
        )
        == "recorded"
    )
    memory = MemoryItem(
        memory_id="",
        memory_type="user_fact",
        content=CONTENT,
        evidence=(SOURCE,),
        source_message_ids=("existing-complete-source",),
    )
    result = await repository.append_claim(
        "role",
        scope,
        memory,
        memory_key="fact_喝咖啡习惯",
        relation_type="ADD",
        evidence=(SOURCE,),
        source_message_id="existing-complete-source",
        source_message_ids=("existing-complete-source",),
        observed_at=observed,
        metadata={"qualifiers": QUALIFIERS, "fact_kind": "other_user_fact"},
    )
    assert result["persisted"] is True
    records = await repository.list_memory_records("role", scope, limit=None)
    assert len(records) == 1 and records[0]["metadata"]["qualifiers"] == QUALIFIERS
    try:
        yield repository, scope, records, observed
    finally:
        database.close_connection()


def messages(records, observed, window=65536):
    return build_memory_llm_messages(
        CURRENT,
        (),
        (),
        tuple(records),
        2000,
        0.85,
        observed_at=observed,
        context_window_tokens=window,
        count_tokens=len,
    )


@pytest.mark.parametrize("raw_columns", [False, True])
async def test_old_claim_keeps_complete_original_conditions_and_identity(stored, raw_columns):
    repository, scope, records, observed = stored
    before = deepcopy(records)
    supplied = deepcopy(records)
    if raw_columns:
        del supplied[0]["evidence"]
        del supplied[0]["source_message_ids"]
    request = messages(supplied, observed)
    payload = json.loads(request[1]["content"])
    (old,) = payload["existing_memories"]
    assert payload["current_user_message"] == CURRENT
    assert old["memory_id"] == str(records[0]["id"]) and old["memory_key"] == "fact_喝咖啡习惯"
    assert old["evidence"] == [SOURCE] and old["qualifiers"] == QUALIFIERS
    assert old["source_message_ids"] == ["existing-complete-source"]
    assert old["content"] == CONTENT
    assert SOURCE not in request[0]["content"] and CURRENT not in request[0]["content"]
    assert supplied[0]["metadata"]["qualifiers"] == QUALIFIERS
    assert await repository.list_memory_records("role", scope, limit=None) == before


async def test_complete_existing_basis_must_fit_before_model_request(stored):
    _, _, records, observed = stored
    complete = messages(records, observed)
    payload = json.loads(complete[1]["content"])
    # This allowance admits the old clipped-summary representation. It cannot
    # admit the full old basis and must fail rather than clip or drop that basis.
    for row in payload["existing_memories"]:
        for field in ["evidence", "qualifiers", "source_message_ids"]:
            row.pop(field, None)
        row["content"] = row["content"][:120]
    old_size = len(complete[0]["content"]) + len(json.dumps(payload, ensure_ascii=False)) + 8
    window = old_size + 768 + CONTEXT_SAFETY_MARGIN_TOKENS + 64
    with pytest.raises(InputBudgetError, match="complete memory request"):
        messages(records, observed, window)
