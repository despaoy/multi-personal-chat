from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from db.schemas import MessageRequest


@pytest.mark.parametrize('character,adapter', [('role', 'nonebot'), (None, 'web-character')])
async def test_web_character_history_is_server_owned(character, adapter):
    from api.generate import generate_reply

    request = MessageRequest(message='继续', characterId=character, adapter=adapter,
        history=[dict(role='user', content='已撤销的旧消息'), dict(role='assistant', content='旧回答')])
    service = SimpleNamespace(generate_queued=AsyncMock(return_value='result'))
    assert await generate_reply(request, {'id': 'account'}, service) == 'result'
    passed = service.generate_queued.call_args.args[0]
    assert passed.history == []
    assert passed.senderId == 'account'


async def test_generic_stateless_management_history_remains_supported():
    from api.generate import generate_reply

    history = [dict(role='user', content='导入的上下文')]
    request = MessageRequest(message='继续', history=history)
    service = SimpleNamespace(generate_queued=AsyncMock(return_value='result'))
    await generate_reply(request, {'id': 'account'}, service)
    assert request.history == history
