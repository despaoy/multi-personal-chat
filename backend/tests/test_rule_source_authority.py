"""Full normal rule-write sources must stay authoritative through final send.

Source mutations are explicit administrative fault injection in isolated
SQLite records created by normal capture and rule writing. Main is a spy;
these controls do not claim native auth, vector, or real model evaluation.
"""

import hashlib

import pytest
from tests.test_structured_memory_authority import BODY, structured_request, wire_response

from character.memory_read_authority import record_version
from db.memory_source import source_identity, source_scope


@pytest.mark.parametrize("mutation", ["unchanged", "changed", "revoked"])
async def test_ordinary_rule_source_change_is_rechecked_before_final_send(tmp_path, record_property, mutation):
    request, prepared, repo, db, scope, target, public = await structured_request(tmp_path)
    before = await repo.get_memory_record(int(target.memory_id), "fiction-role", scope)
    assert before["metadata"]["origin"] == "rule_v2"
    assert target.source_message_ids == ("fiction-structured-0",)
    sources = await repo.list_sources("fiction-role", scope, source_message_ids=target.source_message_ids)
    assert len(sources) == 1 and sources[0]["body"] == BODY
    record_property("actual_source_version_binding_count", len(target.source_record_versions))
    replacement = "我喜欢薄荷茶。"
    if mutation != "unchanged":
        identity = source_identity(
            source_scope(
                "fiction-role",
                scope.platform,
                scope.adapter,
                scope.sender_id,
                scope.conversation_type,
                scope.conversation_id,
            ),
            target.source_message_ids[0],
        )
        connection = db._get_connection()
        if mutation == "changed":
            cursor = connection.execute(
                "UPDATE memory_sources SET body=?, body_digest=? WHERE source_key=? AND body=? AND state='recorded'",
                (replacement, hashlib.sha256(replacement.encode()).hexdigest(), identity["source_key"], BODY),
            )
        else:
            cursor = connection.execute(
                "UPDATE memory_sources SET body=NULL, state='revoked' WHERE source_key=? AND body=? AND state='recorded'",
                (identity["source_key"], BODY),
            )
        assert cursor.rowcount == 1
        connection.commit()
        after = await repo.get_memory_record(int(target.memory_id), "fiction-role", scope)
        assert record_version(after) == record_version(before)
        fresh = await repo.list_sources("fiction-role", scope, source_message_ids=target.source_message_ids)
        assert (fresh[0]["body"] == replacement) if mutation == "changed" else not fresh
    wire, result = await wire_response(request)
    assert public in wire and "用户说喜欢豆浆" in wire
    assert (target.content in wire) is (mutation == "unchanged")
    assert (BODY.rstrip("。") in wire) is (mutation == "unchanged")
    assert (target.memory_id in result.plan.character_context.used_memory_ids) is (mutation == "unchanged")
    assert replacement not in wire


async def test_source_revision_reader_returns_no_plaintext_and_preserves_owner_fence(tmp_path):
    request, _prepared, repo, _db, scope, target, public = await structured_request(tmp_path)
    other = next(item for item in request.character_context.memory_packets if item.memory_id != target.memory_id)
    pairs = tuple((int(key), source) for key, source in other.source_record_pairs)
    revisions = await repo.linked_source_revisions("fiction-role", scope, claim_sources=pairs)
    assert len(revisions) == 1 and set(revisions[0]) == {"memory_id", "source_message_id", "observed_at", "body_sha256"}
    assert revisions[0]["body_sha256"] == hashlib.sha256("我喜欢豆浆。".encode()).hexdigest()
    assert await repo.erase_memory("fiction-role", scope, memory_id=int(target.memory_id)) == 1
    assert await repo.list_sources("fiction-role", scope, source_message_ids=other.source_message_ids) == []
    assert await repo.linked_source_revisions("fiction-role", scope, claim_sources=pairs) == revisions
    wire, result = await wire_response(request)
    assert target.content not in wire and "用户说喜欢豆浆" in wire and public in wire
    assert len(result.plan.character_context.memory_packets) == 1


async def test_source_revision_reader_does_not_borrow_another_owner_or_claim_link(tmp_path):
    from character.models import UserScope

    request, _prepared, repo, _db, scope, target, _public = await structured_request(tmp_path)
    pair = ((int(target.memory_id), target.source_message_ids[0]),)
    other_owner = UserScope(
        scope.platform, scope.adapter, "other-fiction-owner", "other-fiction-owner", scope.conversation_type
    )
    assert await repo.linked_source_revisions("fiction-role", other_owner, claim_sources=pair) == []
    assert await repo.linked_source_revisions("other-fiction-role", scope, claim_sources=pair) == []
    assert await repo.linked_source_revisions("fiction-role", scope, claim_sources=((999999, pair[0][1]),)) == []
    assert (
        await repo.linked_source_revisions(
            "fiction-role", scope, claim_sources=((pair[0][0], "nonexistent-fiction-source"),)
        )
        == []
    )


async def test_source_revision_fingerprints_do_not_enter_model_selection_or_completion(tmp_path):
    import json

    from services.delivery_memory import freeze_completion

    from character.evidence_selector import selection_messages

    request, prepared, _repo, _db, _scope, target, _public = await structured_request(tmp_path)
    assert target.source_record_pairs and target.source_record_versions
    wire, _result = await wire_response(request)
    for output in (
        wire,
        json.dumps(selection_messages(request.message, request.character_context.memory_packets)),
        json.dumps(freeze_completion(prepared)),
    ):
        assert (
            "source_record_pairs" not in output
            and "source_record_versions" not in output
            and "body_sha256" not in output
        )
        assert all(version not in output for _key, _source_id, version in target.source_record_versions)


async def test_source_revision_read_failure_removes_only_affected_normal_fact(tmp_path, monkeypatch):
    from repositories.character_memory import DatabaseCharacterMemoryRepository

    request, _prepared, _repo, _db, _scope, target, public = await structured_request(tmp_path)
    original = DatabaseCharacterMemoryRepository.linked_source_revisions

    async def fail_target(repository, character_id, scope, *, claim_sources):
        if any(memory_id == int(target.memory_id) for memory_id, _source_id in claim_sources):
            raise RuntimeError("Controlled isolated revision read unavailable")
        return await original(repository, character_id, scope, claim_sources=claim_sources)

    monkeypatch.setattr(DatabaseCharacterMemoryRepository, "linked_source_revisions", fail_target)
    wire, _result = await wire_response(request)
    assert target.content not in wire and BODY.rstrip("。") not in wire
    assert "用户说喜欢豆浆" in wire and public in wire
