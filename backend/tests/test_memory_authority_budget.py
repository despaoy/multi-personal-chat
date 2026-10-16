"""Prepared budgets preserve independently authorized claims after revocation.

SQLite persistence, source fences, scoped recall, context preparation and final
generation are real; semantic selection and model output are explicit controls.
"""

import json
from datetime import datetime, timedelta, timezone
from html import unescape
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from repositories.character_memory import DatabaseCharacterMemoryRepository
from repositories.messages import DatabaseMessageRepository
from services.character_context import CharacterContextService, TurnInput

from character.contextual_policy import PolicyOutcome
from character.evidence_selector import ContextualEvidenceSelector
from character.memory_service import CharacterMemoryService
from character.models import CharacterProfile, MemoryItem, UserScope
from db.database import SQLiteDB
from inference.generation_request import GenerationRequest, generate_character_response
from inference.private_context_authority import make_private_context_revalidator
from inference.provider_context import get_provider_context_budget

ROLE = "fiction-authority-budget"
QUERY = "请逐项列出我已经保存的全部纸张偏好和每条完整限定，只依据当前仍可核验的记录，不补写现实授权。"


async def arrange_authority_budget_case(tmp_path, monkeypatch, path):
    """Return real prepared inputs; make no assertion about repaired output.

    The private_sources case installs a one-shot deletion at the first source
    reader after preparation, which is after structured authority validation.
    """
    if path not in {"structured", "private_sources"}:
        raise ValueError("Unknown controlled revocation path")
    database = SQLiteDB(tmp_path / "authority-budget.sqlite")
    repository = DatabaseCharacterMemoryRepository(database)
    scope = UserScope("web", "authority-budget", "fiction-owner", "fiction-owner", "private")
    stamp = datetime.now(timezone.utc) - timedelta(days=1)
    budget = get_provider_context_budget(
        manager=SimpleNamespace(_current_provider=SimpleNamespace(value="openai_compat")),
        env={"OPENAI_COMPAT_CONTEXT_WINDOW_TOKENS": "65536"},
    )
    bodies_by_id = {}
    target_id = None
    try:
        # Each complete claim/source is small and valid. Their admitted union
        # crosses the OLD 6000 fallback without any oversized single record.
        for index in range(24):
            body = (f"我本人喜欢第{index:02}号纸张；"
                    + "仅用于离线整理，须保留编号和原始限定；" * 3
                    + f"末尾限定{index:02}：不代表现实授权。")
            assert len(body) <= 120
            source_id = f"fiction-budget-source-{index:02}"
            assert await repository.capture_source(
                ROLE, scope, source_message_id=source_id, body=body, observed_at=stamp,
            ) == "recorded"
            saved = await repository.append_claim(
                ROLE, scope, MemoryItem("", "user_fact", "用户明确表示：" + body, .9),
                memory_key=f"preference_paper_{index:02}", confidence=.99,
                evidence=(body,), source_message_id=source_id, observed_at=stamp.isoformat(),
                metadata={"qualifiers": {"condition": "仅用于离线整理"}},
            )
            identity = str(saved["id"])
            bodies_by_id[identity] = body
            if target_id is None:
                target_id = identity

        selection_calls = []

        async def select_complete_input(messages):
            payload = json.loads(messages[1]["content"])
            selection_calls.append(payload)
            assert payload["query"] == QUERY and len(payload["required_ids"]) == 24
            return json.dumps({"decisions": [
                {"id": identity, "label": "use"} for identity in payload["required_ids"]
            ]})

        profile = CharacterProfile(ROLE, "虚构纸张记录角色", identity="只解释可核验的合成记录")
        policy = SimpleNamespace(refine=AsyncMock(side_effect=lambda baseline, **kwargs: PolicyOutcome(baseline)))
        service = CharacterContextService(
            SimpleNamespace(get_profile=lambda _identity: profile), repository,
            DatabaseMessageRepository(database),
            memory_service=CharacterMemoryService(repository, semantic_enabled=False),
            memory_selector=ContextualEvidenceSelector(select_complete_input, context_budget=budget.review),
            contextual_policy=policy, source_recall_enabled=True,
            history_limit=budget.history_limit, history_max_chars=budget.history_max_chars,
            source_max_chars=budget.source_max_chars, defer_source_budget=budget.defer_source_budget,
            reference_max_chars=budget.reference_max_chars,
        )
        prepared = await service.prepare_turn(TurnInput(
            message=QUERY, platform=scope.platform, adapter=scope.adapter,
            sender_id=scope.sender_id, conversation_id=scope.conversation_id,
            conversation_type=scope.conversation_type,
        ), ROLE)
        original = prepared.compiled
        assert len(selection_calls) == 1
        assert prepared.memory_budget["budget_chars"] == budget.reference_max_chars == 32768
        assert set(original.used_memory_ids) == set(bodies_by_id)
        assert tuple(item.memory_id for item in original.memory_packets) == original.used_memory_ids
        assert 6000 < len(original.reference_context) < budget.reference_max_chars
        assert original.episodic_reference_context and not original.source_candidate_context

        deleted = []
        if path == "structured":
            assert await repository.erase_memory(ROLE, scope, memory_id=int(target_id)) == 1
            deleted.append(target_id)
        else:
            original_reader = DatabaseCharacterMemoryRepository.list_sources

            async def erase_between_authority_passes(instance, *args, **kwargs):
                # Structured validation has finished by the time the source
                # validator performs this first reader call. Delete once there,
                # exercising its own claim-removal/rebuild branch in production.
                if not deleted:
                    assert await repository.get_memory_record(int(target_id), ROLE, scope) is not None
                    assert await repository.erase_memory(ROLE, scope, memory_id=int(target_id)) == 1
                    deleted.append(target_id)
                return await original_reader(instance, *args, **kwargs)

            monkeypatch.setattr(DatabaseCharacterMemoryRepository, "list_sources", erase_between_authority_passes)

        request = GenerationRequest(
            message=QUERY, persona_prompt="虚构组件角色，只按实际可见完整记录回答。",
            history=prepared.history, character_context=original,
            context_window_tokens=budget.window_tokens,
            private_context_revalidator=make_private_context_revalidator(prepared, database),
        )
        expected_ids = tuple(identity for identity in original.used_memory_ids if identity != target_id)
        return SimpleNamespace(
            request=request, prepared=prepared, repository=repository, database=database,
            scope=scope, target_id=target_id, bodies_by_id=bodies_by_id,
            expected_ids=expected_ids, original=original, budget=budget,
            selection_calls=selection_calls, deleted=deleted, path=path,
        )
    except BaseException:
        database.close_connection()
        raise


async def capture_authority_budget_wire(arranged):
    """Capture actual final input, without asserting the fix's expected result."""
    wires = []

    async def controlled_answer(**kwargs):
        wires.append(unescape("\n".join(message["content"] for message in kwargs["messages"])))
        return "只依据本轮仍可核验的纸张偏好和完整限定作答。"

    result = await generate_character_response(arranged.request, controlled_answer)
    return result, wires


@pytest.mark.parametrize("path", ["structured", "private_sources"])
async def test_authority_subset_preserves_prepared_cloud_reference_budget(tmp_path, monkeypatch, path):
    arranged = await arrange_authority_budget_case(tmp_path, monkeypatch, path)
    try:
        result, wires = await capture_authority_budget_wire(arranged)
        assert len(wires) == 1 and arranged.deleted == [arranged.target_id]
        final = result.plan.character_context
        assert final.used_memory_ids == arranged.expected_ids
        assert tuple(item.memory_id for item in final.memory_packets) == arranged.expected_ids
        assert final.memory_status == "available"
        assert not final.episodic_reference_context and not final.source_candidate_context
        assert final.memory_source_status == "authority_changed"
        assert 6000 < len(final.reference_context) < arranged.budget.reference_max_chars
        assert arranged.bodies_by_id[arranged.target_id] not in wires[0]
        assert f'"id":"{arranged.target_id}"' not in wires[0]
        for identity in arranged.expected_ids:
            assert arranged.bodies_by_id[identity] in final.reference_context
            assert arranged.bodies_by_id[identity] in wires[0]
            assert f'"id":"{identity}"' in wires[0]
        assert await arranged.repository.get_memory_record(int(arranged.target_id), ROLE, arranged.scope) is None
        assert len(await arranged.repository.list_memory_records(ROLE, arranged.scope, limit=None)) == len(arranged.expected_ids)
        assert len(arranged.selection_calls) == 1  # No second review or recall.
    finally:
        arranged.database.close_connection()
