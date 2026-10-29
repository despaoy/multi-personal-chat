"""Live source attachment remains independent; archived frozen inputs still replay."""

import json
from dataclasses import asdict, replace

import pytest
from evaluation.guard_pipeline_replay import replay
from evaluation.recorded_context import restore_recorded_context

from character.context_builder import compile_reference_context
from character.models import CompiledCharacterContext, MemoryItem
from character.output_guard import ReplyGuard
from character.source_memory import SourceRecall, attach_sources, compile_sources
from inference.generation_request import RetrievalResult

pytestmark = pytest.mark.usefixtures("no_live_context_review")
STAMP = "2026-01-01T00:00:00+00:00"
TEXT = "我同事参加陶艺课，我自己没有参加。"


def context():
    item = MemoryItem("1", "shared_event", TEXT, evidence=(TEXT,), source_message_ids=("s",),
                      observed_at=STAMP, temporal_mode="observation", source_observation=True)
    reference, ids = compile_reference_context((item,))
    return CompiledCharacterContext("", "", reference, used_memory_ids=ids, memory_packets=(item,))


def source(body=TEXT, identity="s"):
    return compile_sources([dict(source_message_id=identity, observed_at=STAMP, body=body)])


def test_runtime_keeps_memory_reference_and_attaches_complete_original_sources():
    before = context()
    sources = source()
    after = attach_sources(before, sources)
    assert after.reference_context == before.reference_context and TEXT in after.reference_context
    assert after.used_memory_ids == before.used_memory_ids and after.memory_packets == before.memory_packets
    assert after.episodic_reference_context == sources.context and after.memory_source_status == "available"


def test_changing_or_clearing_sources_preserves_independent_memory_reference():
    before = context()
    attached = attach_sources(before, source())
    updated = attach_sources(attached, source("新原话：同事课程取消，我本人仍未参加。", "new"))
    assert updated.reference_context == before.reference_context and updated.memory_packets == before.memory_packets
    assert updated.episodic_reference_context == source("新原话：同事课程取消，我本人仍未参加。", "new").context
    cleared = attach_sources(updated, SourceRecall(diagnostics={"status": "not_checked"}))
    assert not cleared.episodic_reference_context and cleared.reference_context == before.reference_context
    assert cleared.used_memory_ids == before.used_memory_ids


@pytest.mark.parametrize("legacy", [False, True])
def test_archived_context_decoding_preserves_record_and_known_evidence(legacy):
    original = context()
    expected = replace(original, reference_context="", episodic_reference_context=source().context,
                       memory_source_status="available") if legacy else original
    record = asdict(expected)
    if legacy:
        record.update(source_shared_memory_ids=["1"], source_reference_backup=original.reference_context)
    encoded = json.dumps(record)
    decoded = restore_recorded_context(record)
    assert json.dumps(record) == encoded
    assert decoded == expected and decoded.memory_packets[0].source_observation
    assert not hasattr(decoded, "source_shared_memory_ids") and not hasattr(decoded, "source_reference_backup")


def test_archived_context_unknown_fields_are_not_silently_discarded():
    record = dict(asdict(context()), future_unknown_field="unrecognized")
    with pytest.raises(TypeError, match="future_unknown_field"):
        restore_recorded_context(record)


@pytest.mark.asyncio
async def test_archived_retired_metadata_keeps_the_actual_frozen_request():
    compiled = asdict(context())
    backup = compiled["reference_context"]
    compiled.update(reference_context="", episodic_reference_context=source().context, memory_source_status="available",
                    source_shared_memory_ids=["1"], source_reference_backup=backup)
    messages = [dict(role="system", content="历史冻结应用约束。"), dict(role="user", content=source().context+"\n请回应当前描述。")]
    row = dict(case="archived-context", message="请回应当前描述。", prepared=dict(
        compiled=compiled, history=[], reply_guard=asdict(ReplyGuard())), generation=dict(plan=dict(
            retrieval=asdict(RetrievalResult()), prompt_policy_version="archived-test")))
    cloud = dict(request=dict(messages=messages, temperature=0.0, max_tokens=128, top_p=1.0, frequency_penalty=0.0),
                 response=dict(choices=[dict(message=dict(content="我们可以讨论当前描述。"))]), guard={})
    before = json.dumps(row)
    result = await replay(row, cloud)
    assert json.dumps(row) == before and result["reply_preserved"]
    assert result["recorded_model_calls"] == 1 and result["new_model_calls"] == 0
    assert result["upstream_frozen"] and result["result"]["plan"]["messages"] == tuple(messages)
    assert backup not in str(result["result"]["plan"]["messages"]) and TEXT in messages[-1]["content"]
