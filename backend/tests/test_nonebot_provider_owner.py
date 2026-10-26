"""NoneBot uses the shared model owner and records tool completion usage."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

from api import generate
from db.database import SQLiteDB
from inference import model_manager, vllm_client
from test_compiled_cloud_transport import installed as installed, ANSWER, QUERY
from test_redis_message_boundary import (
    bot_module as bot_module, generation_control as generation_control,
    message_handler as message_handler, private_event as private_event,
)


@pytest.fixture
def lora_backend(bot_module, monkeypatch):
    rows = [dict(id='adapter-1', name='hutao_lora_7b', status='inactive')]
    reader = Mock(return_value=rows)
    monkeypatch.setattr(bot_module, 'db', SimpleNamespace(get_loras=reader))
    monkeypatch.setattr(bot_module, 'config', SimpleNamespace(SUPERUSERS=['1001'], BOT_NAME='合成机器人', REPLY_DELAY=-100))
    provider = SimpleNamespace(name='vllm')
    monkeypatch.setattr(model_manager, 'get_model_manager', lambda *, database=None: SimpleNamespace(get_current_provider=lambda: provider))
    client = SimpleNamespace(list_loras=AsyncMock(return_value=['hutao']))
    getter = AsyncMock(return_value=client)
    monkeypatch.setattr(vllm_client, 'get_enabled_vllm_client', getter)
    return SimpleNamespace(rows=rows, reader=reader, provider=provider, client=client, getter=getter)


@pytest.mark.parametrize('name', ['hutao_lora_7b', 'hutao'])
async def test_registered_or_explicit_alias_selection_reaches_normal_generation(
    bot_module, private_event, generation_control, lora_backend, name,
):
    assert await bot_module._select_lora(name) == 'hutao_lora_7b'
    await bot_module.process_message(private_event)
    request = generation_control[2].await_args.args[0]
    assert request.loraName == 'hutao_lora_7b'
    assert request.senderId == '1001' and request.message == str(private_event.message)
    lora_backend.client.list_loras.assert_awaited_once()


async def test_selection_is_snapshotted_before_admission_wait(
    bot_module, private_event, generation_control, monkeypatch,
):
    bot_module._selected_lora = 'default'
    async def change_selection(*args):
        bot_module._selected_lora = 'changed-after-admission'
    generation_control[0].check_rate_limits.side_effect = change_selection
    await bot_module.process_message(private_event)
    assert generation_control[2].await_args.args[0].loraName == 'default'


async def test_initial_selection_follows_backend_without_hutao_override(
    bot_module, private_event, generation_control,
):
    await bot_module.process_message(private_event)
    assert generation_control[2].await_args.args[0].loraName == ''


@pytest.mark.parametrize('case', ['missing', 'ambiguous', 'unloaded', 'disabled', 'cloud', 'database', 'vllm'])
async def test_failed_switch_preserves_selection_and_never_loads_a_private_model(bot_module, lora_backend, case):
    bot_module._selected_lora = 'default'
    target = 'hutao'
    error_type = bot_module.LoraSelectionError
    if case == 'missing':
        target = 'hutao-prefix-guess'
    elif case == 'ambiguous':
        lora_backend.rows.append(dict(id='adapter-2', name='other_hutao', status='inactive'))
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, LORA_SERVED_NAME_MAP='{"other_hutao":"hutao"}'):
            with pytest.raises(error_type):
                await bot_module._select_lora(target)
        assert bot_module._selected_lora == 'default'
        lora_backend.getter.assert_not_awaited()
        return
    elif case == 'unloaded':
        lora_backend.client.list_loras.return_value = []
    elif case == 'disabled':
        lora_backend.getter.return_value = None
    elif case == 'cloud':
        lora_backend.provider.name = 'openai_compat'
    elif case == 'database':
        lora_backend.reader.side_effect = OSError('controlled database failure')
        error_type = OSError
    elif case == 'vllm':
        lora_backend.client.list_loras.side_effect = ConnectionError('controlled vllm failure')
        error_type = ConnectionError
    with pytest.raises(error_type):
        await bot_module._select_lora(target)
    assert bot_module._selected_lora == 'default'
    if case in {'missing', 'cloud', 'database'}:
        lora_backend.getter.assert_not_awaited()


async def test_exact_database_name_wins_over_an_ambiguous_alias(bot_module, lora_backend):
    lora_backend.rows.append(dict(id='adapter-2', name='hutao', status='inactive'))
    assert await bot_module._select_lora('hutao') == 'hutao'


@pytest.mark.parametrize('present', [False, True])
async def test_native_selection_uses_the_shared_registered_path(bot_module, lora_backend, monkeypatch, tmp_path, present):
    from db import database
    lora_backend.provider.name = 'transformers_peft'
    adapter = tmp_path/'adapter'
    adapter.mkdir()
    (adapter/'adapter_config.json').write_text('{"peft_type":"LORA"}')
    monkeypatch.setattr(database, 'refresh_lora_dir_map', lambda: {'hutao_lora_7b':str(adapter)} if present else {})
    if present:
        assert await bot_module._select_lora('hutao') == 'hutao_lora_7b'
    else:
        with pytest.raises(bot_module.LoraSelectionError, match='文件不存在'):
            await bot_module._select_lora('hutao')
        assert bot_module._selected_lora == ''
    lora_backend.getter.assert_not_awaited()


async def test_cloud_base_selection_is_explicit_and_uses_bot_display_name(bot_module, lora_backend):
    lora_backend.provider.name = 'openai_compat'
    assert await bot_module._select_lora('default') == 'default'
    assert bot_module._get_char_name('default') == '合成机器人'
    lora_backend.reader.assert_not_called()
    lora_backend.getter.assert_not_awaited()


@pytest.mark.parametrize('authorized,failure', [(False, False), (True, False), (True, True)])
async def test_lora_command_checks_global_permission_and_finishes_once_without_error_details(
    bot_module, message_handler, lora_backend, authorized, failure,
):
    from nonebot.adapters.onebot.v11 import Message
    from nonebot.exception import FinishedException
    matcher, event, bot = message_handler
    command = matcher.commands['lora']
    event.message = Message('/lora hutao')
    bot_module.config.SUPERUSERS = ['1001'] if authorized else []
    if failure:
        lora_backend.reader.side_effect = OSError('synthetic-private-credential-detail')
    logs = []
    sink = bot_module.logger.add(lambda message: logs.append(str(message)))
    try:
        with pytest.raises(FinishedException):
            await command.handlers[0](bot, event)
    finally:
        bot_module.logger.remove(sink)
    command.finish.assert_awaited_once()
    reply = command.finish.await_args.args[0]
    assert 'synthetic-private' not in reply + ''.join(logs)
    if not authorized:
        assert '超级用户' in reply and bot_module._selected_lora == ''
        lora_backend.reader.assert_not_called()
    elif failure:
        assert '失败' in reply and bot_module._selected_lora == ''
    else:
        assert '已选择 hutao_lora_7b' in reply and bot_module._selected_lora == 'hutao_lora_7b'


async def test_lora_command_lists_actual_registered_adapters(bot_module, message_handler, lora_backend):
    from nonebot.adapters.onebot.v11 import Message
    from nonebot.exception import FinishedException
    matcher, event, bot = message_handler
    event.message = Message('/lora')
    command = matcher.commands['lora']
    with pytest.raises(FinishedException):
        await command.handlers[0](bot, event)
    command.finish.assert_awaited_once()
    assert 'default, hutao_lora_7b' in command.finish.await_args.args[0]
    assert '跟随后端配置' in command.finish.await_args.args[0]


async def test_all_tool_steps_share_the_callers_selection_and_session(bot_module, private_event, monkeypatch):
    bot = SimpleNamespace(send=AsyncMock())
    bot_module._selected_lora = 'unrelated-global-choice'
    monkeypatch.setattr(bot_module, 'config', SimpleNamespace(BOT_NAME='合成机器人'))
    completion = AsyncMock(side_effect=['{"tool":"lookup","args":{"query":"周五安排"}}', '查询结果：周五执行，出差顺延。'])
    execute = AsyncMock(return_value='合成档案：周五执行，出差顺延。')
    monkeypatch.setattr(bot_module, '_unified_generate', completion)
    monkeypatch.setattr(bot_module, 'execute_tool', execute)
    await bot_module.handle_claw(bot, '查询周五安排及出差例外', private_event, 'default')
    assert len(completion.await_args_list) == 2
    assert all(call.args[1:] == ('1001', 'default') for call in completion.await_args_list)
    execute.assert_awaited_once_with('lookup', {'query':'周五安排'}, bot=bot, event=private_event)


@pytest.fixture
def tool_rig(bot_module, installed, monkeypatch, tmp_path):
    database = SQLiteDB(tmp_path/'tool.sqlite')
    owner = SimpleNamespace(config=installed.config, loras=[], add_model_invocation=database.add_model_invocation)
    async def submit(factory, **kwargs):
        return await factory()
    runtime = SimpleNamespace(submit=AsyncMock(side_effect=submit), priority_for=Mock(return_value=10))
    monkeypatch.setattr(bot_module, 'db', owner)
    monkeypatch.setattr(bot_module, 'inference_runtime', runtime)
    yield SimpleNamespace(database=database, owner=owner, provider=installed, runtime=runtime)
    database.close_connection()


async def test_actual_cloud_tool_completion_records_exact_usage_and_keeps_chat_totals_separate(bot_module, tool_rig):
    assert await bot_module._unified_generate(QUERY, 'fiction-tool-session', 'default') == ANSWER
    rows = tool_rig.database.execute_sql('SELECT * FROM model_invocations')
    assert len(rows) == len(tool_rig.provider.calls) == 1
    row = rows[0]
    assert (row['purpose'], row['totalTokens'], row['usageSource']) == ('tool_generation', 120, 'provider')
    assert row['conversationId'] == 'fiction-tool-session' and row['loraName'] == 'default' and row['traceId']
    assert row['errorType'] == ''
    assert tool_rig.database.execute_sql("SELECT COUNT(*) AS n FROM model_invocations WHERE purpose='chat'")[0]['n'] == 0
    await generate.generate_reply_core(generate.MessageRequest(message=QUERY, loraName='default'), persist_message=False, message_db=tool_rig.owner)
    rows = tool_rig.database.execute_sql('SELECT * FROM model_invocations')
    assert sorted((row['purpose'], row['totalTokens']) for row in rows) == [('chat',120), ('tool_generation',120)]


async def test_tool_cache_hit_does_not_create_another_completion_record(bot_module, tool_rig, monkeypatch):
    from cache.response_cache import ResponseCache
    monkeypatch.setattr(generate, 'response_cache', ResponseCache(max_size=2, default_ttl=60))
    assert await bot_module._unified_generate(QUERY, 'fiction-tool-session', 'default') == ANSWER
    assert await bot_module._unified_generate(QUERY, 'fiction-tool-session', 'default') == ANSWER
    rows = tool_rig.database.execute_sql('SELECT * FROM model_invocations')
    assert len(rows) == len(tool_rig.provider.calls) == 1
    assert rows[0]['purpose'] == 'tool_generation' and rows[0]['totalTokens'] == 120


@pytest.mark.parametrize('cancelled', [False, True])
async def test_failed_or_cancelled_tool_completion_is_recorded_once_and_not_replaced(bot_module, tool_rig, cancelled):
    tool_rig.provider.state.failure = asyncio.CancelledError() if cancelled else ConnectionError('synthetic-private-upstream-detail')
    with pytest.raises(asyncio.CancelledError if cancelled else HTTPException):
        await bot_module._unified_generate(QUERY, 'fiction-tool-session', 'default')
    rows = tool_rig.database.execute_sql('SELECT * FROM model_invocations')
    assert len(rows) == 1 and rows[0]['purpose'] == 'tool_generation'
    assert rows[0]['totalTokens'] is None and rows[0]['errorType']
    assert tool_rig.provider.calls == []
    tool_rig.provider.manager.async_generate.assert_not_awaited()


async def test_tool_record_failure_cannot_report_success(bot_module, tool_rig, monkeypatch):
    monkeypatch.setattr(tool_rig.owner, 'add_model_invocation', Mock(side_effect=OSError('synthetic-private-ledger-detail')))
    with pytest.raises(HTTPException) as caught:
        await bot_module._unified_generate(QUERY, 'fiction-tool-session', 'default')
    assert caught.value.status_code == 503 and caught.value.detail['stage'] == 'model_invocation_record'
    assert 'synthetic-private' not in str(caught.value.detail)
    assert len(tool_rig.provider.calls) == 1
    assert tool_rig.database.execute_sql('SELECT * FROM model_invocations') == []
