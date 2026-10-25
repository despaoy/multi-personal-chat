"""Optional admin filters share one query path without interpolating values."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from api import preferences, retrieval_eval
from db.schemas import SampleFromHistoryRequest


@pytest.mark.asyncio
@pytest.mark.parametrize("filtered", [False, True])
@pytest.mark.parametrize("kind", ["preferences", "retrieval"])
async def test_list_queries_bind_same_filter_for_rows_and_count(kind, filtered):
    database = SimpleNamespace(execute_sql=Mock(side_effect=[[], [{"cnt": 0}]]))
    value = "reviewed" if filtered else None
    if kind == "preferences":
        result = await preferences.list_preferences(
            review_status=value,
            limit=12,
            offset=3,
            current_user={},
            database=database,
        )
        field, parameter = "review_status", "rs"
    else:
        result = await retrieval_eval.list_questions(
            category=value,
            limit=12,
            offset=3,
            current_user={},
            database=database,
        )
        field, parameter = "category", "cat"
    assert result["total"] == 0
    rows, count = database.execute_sql.call_args_list
    expected = {parameter: value} if filtered else {}
    assert rows.args[1] == {**expected, "lim": 12, "off": 3}
    assert count.args[1] == expected
    for call in (rows, count):
        assert (f" WHERE {field}=:{parameter}" in call.args[0]) is filtered
        assert "reviewed" not in call.args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("session", ["", "literal-session-' OR 1=1"])
async def test_sampling_preserves_branch_exclusion_and_binds_session(session):
    database = SimpleNamespace(execute_sql=Mock(return_value=[]))
    result = await preferences.sample_from_history(
        SampleFromHistoryRequest(session_id=session, min_length=8, limit=7),
        current_user={},
        database=database,
    )
    assert result == {"success": True, "candidates": [], "total": 0}
    query, params = database.execute_sql.call_args.args
    assert '"branchId" IS NULL' in query
    assert params == {"min_len": 8, "lim": 7, **({"sid": session} if session else {})}
    assert ('"sessionId"=:sid' in query) is bool(session)
    if session:
        assert session not in query
