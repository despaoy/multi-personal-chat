"""Relationship extraction failures and transactional conversation boundaries."""

import asyncio
import copy
import json
from functools import partial

import pytest

from character.library import CharacterSettings
from character.memory_llm import MemoryLlmConfig
from db.database import SQLiteDB
from db.errors import RevisionConflictError
from repositories.character_chat import CharacterChatRepository
from repositories.character_library import CharacterNotFoundError
from services.character_chat import CharacterChatService, prepare_request
from services.character_chat_relationship import enrich_relationship


class Completion:
    def __init__(self, proposal):
        self.proposal = proposal
        self.calls = 0

    async def complete(self, messages):
        self.calls += 1
        return json.dumps(self.proposal, ensure_ascii=False)


def test_relationship_guidance_uses_only_the_committed_stage(tmp_path):
    db = SQLiteDB(tmp_path / "relationship-prompt.db")
    try:
        repo = CharacterChatRepository(db)
        character = repo.characters.create("1", CharacterSettings(name="迟禾", persona="谨慎", greeting="你好"))
        session = repo.create("1", character["id"], "关系表达")
        message = "又见面了。"
        initial = prepare_request(session, [], message)
        assert "礼貌而不假定熟悉" in initial.persona_prompt
        familiar = repo.update_relationship("1", session["id"], 1, "familiar", "此前多次讨论修钟")
        familiar["state"]["relationship_proposal"] = {"stage": "close", "summary": "待确认内容"}
        prepared = prepare_request(familiar, [], message)
        assert "仅在相关时引用共同记忆" in prepared.persona_prompt
        assert "越过边界" in prepared.persona_prompt
        assert "关心不等于追问" not in prepared.persona_prompt
        assert "待确认内容" not in prepared.character_context.conversation_reference_context
    finally:
        db.close_connection()


@pytest.fixture
def proposal(monkeypatch):
    monkeypatch.setenv("RELATIONSHIP_LLM_ENABLED", "true")
    return {
        "action": "update",
        "stage": "acquaintance",
        "summary": "双方约定一起旅行",
        "confidence": 0.99,
        "evidence": [{"role": "user", "quote": "一起旅行吧"}, {"role": "assistant", "quote": "好，一起出发"}],
    }


def extractor(writer):
    return partial(enrich_relationship, config=MemoryLlmConfig(True, "https://test.invalid", "test"), completion=writer)


@pytest.mark.parametrize("fault", ["invented", "one_sided", "invalid", "low_confidence"])
def test_untrusted_proposals_preserve_relation_and_memory(proposal, fault):
    if fault == "invented":
        proposal["evidence"][0]["quote"] = "不存在的原话"
    elif fault == "one_sided":
        proposal["evidence"].pop()
    elif fault == "invalid":
        proposal["stage"] = "married"
    else:
        proposal["confidence"] = 0.5
    state = {"relationship": {"stage": "stranger", "summary": ""}, "memories": [{"content": "喜欢茶"}]}
    session = {"id": "session", "mode": "initial", "state": copy.deepcopy(state)}
    result = asyncio.run(extractor(Completion(proposal))("一起旅行吧", "好，一起出发", session, "request"))
    assert result["relationship"] == state["relationship"]
    assert result["memories"] == state["memories"]
    assert result["relationship_processing"] == ("completed" if fault == "low_confidence" else "failed")
    assert session["state"] == state


def test_disabled_and_opt_out_never_call_provider(proposal, monkeypatch):
    writer = Completion(proposal)
    session = {"id": "session", "mode": "initial", "state": {"relationship": {"stage": "stranger"}, "memories": []}}
    result = asyncio.run(extractor(writer)("不要记住这件事", "好的", session, "request"))
    assert result["relationship_processing"] == "skipped"
    monkeypatch.setenv("RELATIONSHIP_LLM_ENABLED", "false")
    result = asyncio.run(extractor(writer)("一起旅行吧", "好的", session, "request"))
    assert result["relationship_processing"] == "disabled"
    assert writer.calls == 0


def test_provider_timeout_preserves_successful_memory(proposal):
    class SlowCompletion:
        async def complete(self, messages):
            await asyncio.sleep(1)

    config = MemoryLlmConfig(True, "https://test.invalid", "test", timeout_seconds=0.01)
    state = {"relationship": {"stage": "familiar", "summary": "旧经历"}, "memories": [{"content": "喜欢茶"}]}
    result = asyncio.run(
        enrich_relationship(
            "一起旅行吧",
            "好，一起出发",
            {"id": "session", "mode": "initial", "state": state},
            "request",
            config=config,
            completion=SlowCompletion(),
        )
    )
    assert result["relationship_processing"] == "failed"
    assert result["relationship"] == state["relationship"]
    assert result["memories"] == state["memories"]


def test_automatic_relation_follows_edit_retract_and_branch(tmp_path, proposal):
    db = SQLiteDB(tmp_path / "relationship.db")
    try:
        repo = CharacterChatRepository(db)
        character = repo.characters.create("1", CharacterSettings(name="旅人", persona="谨慎", greeting="你好"))
        session = repo.create("1", character["id"], "初识")
        writer = Completion(proposal)

        async def complete(*args):
            return "好，一起出发", "test"

        chat = CharacterChatService(db, complete, relationship=extractor(writer))
        first = asyncio.run(chat.generate("1", session["id"], revision=1, request_id="first", message="一起旅行吧"))
        saved = repo.get("1", session["id"])
        assert saved["state"]["relationship"]["source_turn_id"] == first["id"]
        assert saved["state"]["relationship"]["stage"] == "acquaintance"
        branch = repo.create(
            "1",
            character["id"],
            "分支",
            source_session_id=session["id"],
            source_revision=2,
            source_turn_id=first["id"],
            hypothesis="去海边",
        )
        writer.proposal = {"action": "keep"}
        replacement = asyncio.run(
            chat.generate(
                "1", session["id"], revision=2, request_id="edit", message="再见", replace_turn_id=first["id"]
            )
        )
        saved = repo.get("1", session["id"])
        assert saved["state"]["relationship"]["stage"] == "stranger"
        repo.retract("1", session["id"], replacement["id"], saved["revision"])
        assert repo.get("1", branch["id"])["state"]["relationship"]["stage"] == "acquaintance"
    finally:
        db.close_connection()


@pytest.mark.parametrize("accept", [True, False])
def test_manual_relationship_requires_explicit_decision(tmp_path, proposal, accept):
    db = SQLiteDB(tmp_path / "manual-relationship.db")
    try:
        repo = CharacterChatRepository(db)
        character = repo.characters.create("1", CharacterSettings(name="旅人", persona="谨慎", greeting="你好"))
        session = repo.create("1", character["id"], "初识")
        manual = repo.update_relationship("1", session["id"], 1, "familiar", "用户确认的旧关系")

        async def complete(*args):
            return "好，一起出发", "test"

        chat = CharacterChatService(db, complete, relationship=extractor(Completion(proposal)))
        turn = asyncio.run(chat.generate("1", session["id"], revision=2, request_id="proposal", message="一起旅行吧"))
        saved = repo.get("1", session["id"])
        assert saved["state"]["relationship"] == manual["state"]["relationship"]
        assert saved["state"]["relationship_proposal"]["source_turn_id"] == turn["id"]
        branch = repo.create(
            "1", character["id"], "分支", source_session_id=session["id"],
            source_revision=3, source_turn_id=turn["id"], hypothesis="去海边",
        )
        with pytest.raises(CharacterNotFoundError):
            repo.resolve_relationship_proposal("other", session["id"], 3, "proposal", accept)
        with pytest.raises(RevisionConflictError):
            repo.resolve_relationship_proposal("1", session["id"], 2, "proposal", accept)
        with pytest.raises(RevisionConflictError):
            repo.resolve_relationship_proposal("1", session["id"], 3, "wrong", accept)
        resolved = repo.resolve_relationship_proposal("1", session["id"], 3, "proposal", accept)
        assert "relationship_proposal" not in resolved["state"]
        assert resolved["state"]["relationship"]["origin"] == "user"
        assert resolved["state"]["relationship"]["stage"] == ("acquaintance" if accept else "familiar")
        assert repo.get("1", branch["id"])["state"]["relationship_proposal"]["id"] == "proposal"
        with pytest.raises(RevisionConflictError):
            repo.resolve_relationship_proposal("1", session["id"], 4, "proposal", accept)
        repo.retract("1", session["id"], turn["id"], 4)
        restored = repo.get("1", session["id"])
        assert restored["state"]["relationship"] == manual["state"]["relationship"]
        assert "relationship_proposal" not in restored["state"]
        repo.update_relationship("1", branch["id"], 1, "close", "重新设定")
        assert "relationship_proposal" not in repo.get("1", branch["id"])["state"]
    finally:
        db.close_connection()
