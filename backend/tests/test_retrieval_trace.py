import asyncio

import pytest

from evaluation.retrieval_trace import RecordedRetrieval


@pytest.mark.asyncio
@pytest.mark.parametrize('override, expected', [(None, 3), (8, 8)])
async def test_trace_preserves_bundle_and_records_effective_budget(override, expected):
    current = {}
    bundle = {'context_text': '完整证据，但仅在特定条件下成立。'}

    async def retrieve(query, top_k, filters):
        assert (query, top_k, filters) == ('独立问题', expected, {'knowledge_base_id': 7})
        return bundle

    wrapped = RecordedRetrieval(retrieve, current, top_k=override)
    assert await wrapped('独立问题', 3, {'knowledge_base_id': 7}) is bundle
    call = current['retrieval_calls'][0]
    assert call['requested_top_k'] == 3 and call['effective_top_k'] == expected
    assert call['bundle'] is bundle and call['status'] == 'completed'
    assert call['seconds'] >= 0


@pytest.mark.asyncio
@pytest.mark.parametrize('error', [RuntimeError, asyncio.CancelledError])
async def test_trace_never_turns_failure_or_cancellation_into_empty_success(error):
    current = {}

    async def retrieve(*args):
        raise error()

    with pytest.raises(error):
        await RecordedRetrieval(retrieve, current)('问题', 3, None)
    call = current['retrieval_calls'][0]
    assert call['status'] == 'failed_or_cancelled' and 'bundle' not in call


@pytest.mark.parametrize('value', [0, 33, True, 1.5])
def test_invalid_experimental_count_rejected(value):
    with pytest.raises(ValueError):
        RecordedRetrieval(None, {}, top_k=value)


def test_coverage_fixture_is_valid_for_actual_web_replay():
    from pathlib import Path

    from evaluation.temporal_memory_live_replay import load_cases

    cases = load_cases(Path(__file__).parents[1] / 'evaluation/fixtures/rag_coverage_transfer_20260927.json')
    assert len(cases) == 2
