"""Summary provenance, prompt compaction and rewind boundaries."""

import asyncio
import json
from dataclasses import replace
from functools import partial

import pytest

from character.library import CharacterSettings
from character.memory_llm import MemoryLlmConfig
from db.database import SQLiteDB
from inference.context_budget import CONTEXT_SAFETY_MARGIN_TOKENS, estimated_tokens
from inference.generation_request import build_generation_request
from repositories.character_chat import CharacterChatRepository
from services.character_chat import prepare_request
from services.character_chat_enrichment import process_enrichment
from services.character_chat_summary import enrich_summary, summaries_for_prompt, valid_summaries


@pytest.fixture
def conversation(tmp_path, monkeypatch):
    monkeypatch.setenv("SUMMARY_LLM_ENABLED", "true")
    db = SQLiteDB(tmp_path / "summary.db")
    repo = CharacterChatRepository(db)
    char = repo.characters.create("1", CharacterSettings(name="旅人", persona="谨慎", greeting="你好"))
    session = repo.create("1", char["id"], "测试")
    turns = []
    for index in range(12):
        turns.append(repo.commit_turn(session, str(index), f"第{index}个计划", f"讨论计划{index}", "test",
                                      enqueue_enrichment=index == 11))
        session = repo.get("1", session["id"])
    yield db, repo, session, turns
    db.close_connection()


class Writer:
    def __init__(self, first, invalid=False):
        self.first, self.invalid, self.calls = first, invalid, 0

    async def complete(self, messages):
        self.calls += 1
        return json.dumps({"summary": "双方讨论了几个计划，尚未实施。", "evidence": [
            {"turn_id": self.first["id"], "role": "user", "quote": "杜撰" if self.invalid else self.first["message"]}
        ]}, ensure_ascii=False)


def summarizer(writer):
    return partial(enrich_summary, config=MemoryLlmConfig(True, "https://test.invalid", "test"), completion=writer)


def test_summary_activity_compacts_context_and_rewind_removes_summary(conversation):
    db, repo, session, turns = conversation
    writer = Writer(turns[0])

    async def keep(*args):
        return args[-2]["state"]

    result = asyncio.run(process_enrichment(turns[-1]["id"], database=db, enrich=keep, relationship=keep,
                                           summary=summarizer(writer), expected_revision=session["revision"]))
    assert result == "completed" and writer.calls == 1
    current = repo.get("1", session["id"])
    assert current["state"]["summary_processing"] == "completed"
    assert len(current["state"]["summaries"][0]["sources"]) == 6
    request = prepare_request(current, turns, "继续")
    assert len(request.history) == 13  # Greeting plus six uncompressed recent turns.
    assert "尚未实施" in request.character_context.conversation_reference_context
    assert turns[0]["id"] not in request.character_context.conversation_reference_context
    assert "fingerprint" not in request.character_context.conversation_reference_context
    branch = repo.create("1", session["character_id"], "早期分支", source_session_id=session["id"],
                         source_revision=current["revision"], source_turn_id=turns[4]["id"], hypothesis="另一走向")
    assert not branch["state"].get("summaries")
    continued = repo.create("1", session["character_id"], "不带历史", source_session_id=session["id"],
                            source_revision=current["revision"], inherit_history=False)
    assert not continued["state"].get("summaries")
    rolled = repo.retract("1", session["id"], turns[-1]["id"], current["revision"])
    assert not rolled["state"].get("summaries")


def test_summarized_history_uses_actual_remaining_window(conversation):
    _, _, session, turns = conversation
    state = asyncio.run(summarizer(Writer(turns[0]))(session, turns, "summary"))
    request = prepare_request({**session, "state": state}, turns, "继续")
    full = build_generation_request(replace(request, context_window_tokens=1_000_000))
    assert [m["content"] for m in full.messages[:-1] if m["role"] == "user"] == [t["message"] for t in turns]
    assert "尚未实施" in full.messages[-1]["content"]
    limit = (sum(estimated_tokens(m["content"]) + 4 for m in full.messages)
             + request.max_tokens + CONTEXT_SAFETY_MARGIN_TOKENS)
    assert build_generation_request(replace(request, context_window_tokens=limit)).messages == full.messages
    tight = replace(request, context_window_tokens=limit - 1)
    compact = build_generation_request(tight)
    assert compact.messages == (*full.messages[:2], *full.messages[4:])
    assert "尚未实施" in compact.messages[-1]["content"]


def test_partial_compaction_preserves_unsummarized_turn_with_repeated_wording(conversation):
    _, _, session, turns = conversation
    turns[1] = {**turns[1], "message": turns[0]["message"]}
    # The first turn was skipped by the summarizer. The next turn repeats it.
    state = asyncio.run(summarizer(Writer(turns[1]))(session, turns, "summary"))
    summary = state["summaries"][0]
    summary["sources"] = summary["sources"][1:]
    request = prepare_request({**session, "state": state}, turns, "继续")
    full = build_generation_request(replace(request, context_window_tokens=1_000_000))
    limit = (sum(estimated_tokens(m["content"]) + 4 for m in full.messages)
             + request.max_tokens + CONTEXT_SAFETY_MARGIN_TOKENS - 1)
    partial = build_generation_request(replace(request, context_window_tokens=limit))
    assert partial.messages == (*full.messages[:4], *full.messages[6:])


def test_one_restricted_folded_turn_does_not_hide_other_visible_turns(conversation):
    _, _, session, turns = conversation
    turns[2] = {**turns[2], "reply": "这一轮不要保存，也不要记住。"}
    state = asyncio.run(summarizer(Writer(turns[0]))(session, turns, "summary"))
    request = prepare_request({**session, "state": state}, turns, "第一个计划是什么？")
    plan = build_generation_request(replace(request, context_window_tokens=1_000_000))
    history = plan.messages[:-1]
    assert [m["content"] for m in history if m["role"] == "user"] == [
        t["message"] for t in turns if t["id"] != turns[2]["id"]
    ]
    assert all(m["content"] != turns[2]["reply"] for m in history)


def test_invalid_evidence_does_not_create_summary(conversation):
    _, _, session, turns = conversation
    state = asyncio.run(summarizer(Writer(turns[0], invalid=True))(session, turns, "summary"))
    assert state["summary_processing"] == "failed" and not state.get("summaries")


def test_summary_prompt_keeps_relevant_quotes_without_storage_metadata():
    evidence = [
        {"role": "user", "quote": "桌上的纸袋是空的。"},
        {"role": "assistant", "quote": "知道了，我会等。"},
        {"role": "user", "quote": "借书卡是灰色的。"},
        {"role": "assistant", "quote": "借书卡颜色已更正。"},
    ]
    stored = {"summary": "整理旧书与杂物。", "source_session_id": "private-lineage",
              "evidence": [{**item, "turn_id": str(index)} for index, item in enumerate(evidence)]}
    assert summaries_for_prompt([stored], "我说借书卡是什么颜色？") == [
        {"summary": stored["summary"], "evidence": evidence[2:]},
    ]
    assert len(stored["evidence"]) == 4


def test_two_summary_segments_restore_only_the_retained_checkpoint(conversation):
    db, repo, original, _ = conversation
    session = repo.create("1", original["character_id"], "跨段回退")
    turns = []

    async def keep(*args):
        return args[-2]["state"]

    for index in range(19):
        message = "这一轮不要保存，也不要记住。" if index == 6 else f"用户原话 {index}"
        checkpoint = index in {11, 18}
        turn = repo.commit_turn(session, str(index), message, f"角色回复 {index}", "test", enqueue_enrichment=checkpoint)
        turns.append(turn)
        session = repo.get("1", session["id"])
        if checkpoint:
            writer = Writer(turns[0 if index == 11 else 7])
            result = asyncio.run(process_enrichment(
                turn["id"], database=db, enrich=keep, relationship=keep,
                summary=summarizer(writer), expected_revision=session["revision"],
            ))
            assert result == "completed" and writer.calls == 1
            session = repo.get("1", session["id"])

    first, second = session["state"]["summaries"]
    assert [source["turn_id"] for source in first["sources"]] == [turn["id"] for turn in turns[:6]]
    assert [source["turn_id"] for source in second["sources"]] == [turn["id"] for turn in turns[7:13]]
    branch = repo.create("1", session["character_id"], "未来保留分支", source_session_id=session["id"],
                         source_revision=session["revision"], source_turn_id=turns[-1]["id"])
    rolled = repo.retract("1", session["id"], turns[12]["id"], session["revision"])
    assert rolled["state"]["summaries"] == [first]
    request = prepare_request(rolled, repo.turns("1", session["id"]), "回忆以前")
    summaries = json.loads(request.character_context.conversation_reference_context)["conversation_summaries"]
    assert len(summaries) == 1
    assert repo.get("1", branch["id"])["state"]["summaries"] == [first, second]


def test_summary_sources_must_still_exist_and_match(conversation):
    _, _, session, turns = conversation
    state = asyncio.run(summarizer(Writer(turns[0]))(session, turns, "summary"))
    assert len(valid_summaries(state, turns)) == 1
    assert valid_summaries(state, turns[1:]) == []
    altered = [{**turns[0], "message": "替换后内容"}, *turns[1:]]
    assert valid_summaries(state, altered) == []


def test_memory_correction_clears_summary_and_blocks_rebuilding_its_sources(conversation):
    _, repo, session, turns = conversation
    state = asyncio.run(summarizer(Writer(turns[0]))(session, turns, "summary"))
    repo.database.revision_transaction([repo.cas(session, state)])
    current = repo.get("1", session["id"])
    current = repo.manage_memory("1", session["id"], current["revision"], content="旧偏好")
    current = repo.manage_memory("1", session["id"], current["revision"],
                                memory_id=current["state"]["memories"][0]["id"], content="新偏好")
    assert not current["state"].get("summaries")
    writer = Writer(turns[0])
    state = asyncio.run(summarizer(writer)(current, turns, "summary"))
    assert state["summary_processing"] == "skipped" and writer.calls == 0
