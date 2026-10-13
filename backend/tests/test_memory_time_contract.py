"""Malformed model validity dates fail explicitly instead of dropping a fact."""
import json

import pytest

from character.memory_llm import parse_llm_proposals


def parse(**fields):
    row=dict(operation='ADD',kind='study_stage',value='大三',evidence='今年刚升大三',
             confidence=.96,attributed_to='user',**fields)
    return parse_llm_proposals(json.dumps({'memories':[row]},ensure_ascii=False),source_message='今年刚升大三')

@pytest.mark.parametrize('field',['valid_from','valid_to','valid_at','invalid_at'])
@pytest.mark.parametrize('bad',['not-a-date','2026-02-30',0,False,[],{}])
def test_invalid_date_or_structure_is_not_a_normal_empty_result(field,bad):
    with pytest.raises(ValueError,match='valid_from|valid_to'):
        parse(**{field:bad})

def test_inverted_validity_interval_is_explicit_error():
    with pytest.raises(ValueError,match='valid_from.*valid_to'):
        parse(valid_from='2026-10-09',valid_to='2026-10-08')

@pytest.mark.parametrize('empty',[None,''])
def test_unspecified_optional_times_are_still_valid(empty):
    proposal=parse(valid_from=empty,valid_to=empty)[0]
    assert proposal.valid_from==proposal.valid_to==''


async def recall_stored_time(**changes):
    from datetime import datetime, timezone

    from character.memory_service import CharacterMemoryService
    from character.models import UserScope

    row = dict(id="source", memory_type="user_fact", memory_key="preference_咖啡",
               content="用户说喜欢咖啡", importance=0.6, confidence=0.98, status="active",
               updated_at="2026-10-13T12:00:00+00:00", evidence=["我喜欢咖啡。"], metadata={})
    historical = changes.pop("historical", False)
    row.update(changes)

    class Repository:
        async def list_memory_records(self, *args, **kwargs):
            return [row]

    return await CharacterMemoryService(Repository(), semantic_enabled=False).recall_with_diagnostics(
        "role", UserScope("web", "time", "owner", "owner", "private"),
        "我2026年十月喜欢咖啡吗？" if historical else "我喜欢咖啡吗？",
        reference_time=datetime(2026, 10, 13, 12, tzinfo=timezone.utc),
    )


@pytest.mark.parametrize("field", ["valid_from", "valid_to", "valid_at", "invalid_at"])
@pytest.mark.parametrize("bad", ["not-a-date", "2026-02-30", 0, False, [], {}])
@pytest.mark.parametrize("historical", [False, True])
async def test_stored_time_corruption_is_not_unbounded_validity(field, bad, historical):
    with pytest.raises(ValueError, match="Memory valid_from|Memory valid_to"):
        await recall_stored_time(**{field: bad}, historical=historical)


@pytest.mark.parametrize("bad", ["not-a-date", False, {}])
async def test_stored_time_recency_corruption_is_not_zero_score(bad):
    with pytest.raises(ValueError, match="Memory updated_at"):
        await recall_stored_time(updated_at=bad)


@pytest.mark.parametrize("field", ["valid_from", "valid_to", "observed_at"])
@pytest.mark.parametrize("bad", ["not-a-date", False, {}])
def test_reference_time_corruption_cannot_become_injectable_memory(field, bad):
    from character.context_builder import compile_reference_context
    from character.models import MemoryItem

    kwargs = {field: bad}
    if field == "observed_at":
        kwargs["temporal_mode"] = "observation"
    memory = MemoryItem("source", "user_fact", "用户说喜欢咖啡", evidence=("我喜欢咖啡。",), **kwargs)
    with pytest.raises(ValueError, match="Memory " + field):
        compile_reference_context((memory,), complete_evidence=True)


@pytest.mark.parametrize("empty", [None, ""])
async def test_stored_time_optional_bounds_remain_valid(empty):
    items, total, _ = await recall_stored_time(valid_from=empty, valid_to=empty)
    assert total == 1 and len(items) == 1 and items[0].memory_id == "source"


@pytest.mark.parametrize("start,stop,expected", [
    ("2026-10-13T20:00:00+08:00", None, True),
    (None, "2026-10-13T20:00:00+08:00", False),
    ("2026-10-14T00:00:00+00:00", None, False),
])
async def test_stored_time_current_half_open_interval_respects_clock(start, stop, expected):
    items, total, _ = await recall_stored_time(valid_from=start, valid_to=stop)
    assert total == 1 and bool(items) is expected


async def test_stored_time_legacy_alias_still_works_when_canonical_field_is_empty():
    items, _, _ = await recall_stored_time(valid_from="", valid_to=None,
                                         valid_at="2026-10-01", invalid_at="2026-11-01")
    assert len(items) == 1 and items[0].valid_from == "2026-10-01" and items[0].valid_to == "2026-11-01"
