import json

import pytest

from evaluation.frozen_memory_writer import FrozenMemoryWriter


def messages(source):
    return [dict(role='user', content=json.dumps(dict(current_user_message=source)))]


def writer(candidate):
    return FrozenMemoryWriter([dict(case='a', index=0, writer_calls=[dict(
        messages=messages('我学陶艺。'), output=json.dumps(dict(memories=[candidate])))] )],
        dict(case='a', index=0))


@pytest.mark.asyncio
async def test_replay_logs_current_request_without_calling_provider():
    instance = writer(dict(operation='ADD'))
    result = await instance.complete(messages('我学陶艺。'))
    assert json.loads(result)['memories'] == [dict(operation='ADD')]
    assert instance.calls[0]['recorded']
    with pytest.raises(IndexError):
        await instance.complete(messages('我学陶艺。'))


@pytest.mark.asyncio
async def test_replay_rejects_changed_source_or_dependent_operations():
    with pytest.raises(ValueError, match='source mismatch'):
        await writer(dict(operation='ADD')).complete(messages('我不学陶艺。'))
    for candidate in (dict(operation='MERGE'), dict(operation='ADD', target_memory_id='1')):
        with pytest.raises(ValueError, match='target-dependent'):
            await writer(candidate).complete(messages('我学陶艺。'))
