"""Actual tool protocol and bounded script execution; synthetic files only."""

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bot import tools
from services.tool_execution import ToolCodeError, run_tool_script
from test_redis_message_boundary import bot_module as bot_module, message_handler as message_handler, private_event as private_event


@pytest.fixture
def tool_state(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, 'CODE_ROOT', tmp_path/'generated')
    return SimpleNamespace(bot=SimpleNamespace(send=AsyncMock(), upload_private_file=AsyncMock()), event=SimpleNamespace(user_id=1001), root=tmp_path)


@pytest.mark.parametrize('payload', [
    'not json', '```json\n{"tool":"status","args":{}}\n```',
    'before {"tool":"status","args":{}} after', '[]',
    '{"tool":"status"}', '{"tool":2,"args":{}}',
    '{"tool":"status","args":null}', '{"tool":"status","args":{},"extra":1}',
])
async def test_invalid_model_call_stops_before_execution_and_completion(bot_module, private_event, tool_state, monkeypatch, payload):
    monkeypatch.setattr(bot_module, 'config', SimpleNamespace(BOT_NAME='合成机器人'))
    completion=AsyncMock(side_effect=[payload])
    execution=AsyncMock()
    monkeypatch.setattr(bot_module,'_unified_generate',completion)
    monkeypatch.setattr(bot_module,'execute_tool',execution)
    with pytest.raises(tools.ToolError, match='格式无效'):
        await bot_module.handle_claw(tool_state.bot,'查询系统状态',private_event,'default')
    execution.assert_not_awaited()
    assert completion.await_count==1
    tool_state.bot.send.assert_not_awaited()


@pytest.mark.parametrize('name,args', [('missing',{}),('write_code',{}),('status',{'bot':'untrusted'}),('status',{'unexpected':1})])
async def test_unknown_tool_and_invalid_arguments_raise_before_side_effects(tool_state, name, args):
    with pytest.raises(tools.ToolError):
        await tools.execute_tool(name,args,bot=tool_state.bot,event=tool_state.event)
    assert not (tool_state.root/'generated').exists()
    tool_state.bot.send.assert_not_awaited()


async def test_actual_status_measurement_does_not_block_the_event_loop(tool_state, monkeypatch):
    def sample(*, interval):
        time.sleep(.1)
        return 17
    monkeypatch.setattr(tools.psutil,'cpu_percent',sample)
    monkeypatch.setattr(tools.psutil,'virtual_memory',lambda:SimpleNamespace(percent=23))
    monkeypatch.setattr(tools.psutil,'disk_usage',lambda _:SimpleNamespace(percent=31))
    task=asyncio.create_task(tools.execute_tool('status',{},bot=tool_state.bot,event=tool_state.event))
    await asyncio.sleep(.01)
    assert not task.done()
    assert await task=='CPU: 17%, MEM: 23%, DISK: 31%'


async def test_script_uses_current_interpreter_and_retains_standalone_imports_without_parent_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv('CLAW_SYNTHETIC_PARENT_SENTINEL','synthetic-private-parent')
    (tmp_path/'helper.py').write_text('value=42\n')
    script=tmp_path/'main.py'
    script.write_text('import helper, os, sys\nprint(sys.executable)\nprint(helper.value)\nprint(os.getenv("CLAW_SYNTHETIC_PARENT_SENTINEL"))\nprint("周五执行，出差顺延。")\n',encoding='utf-8')
    result=await run_tool_script(script)
    assert result.splitlines()==[sys.executable,'42','None','周五执行，出差顺延。']


async def test_successful_script_preserves_stderr_without_turning_it_into_failure(tmp_path):
    script=tmp_path/'notice.py'
    script.write_text('import sys\nprint("完成")\nprint("合成诊断",file=sys.stderr)\n',encoding='utf-8')
    assert await run_tool_script(script)=='完成\n标准错误输出:\n合成诊断'


async def test_nonzero_exit_is_an_exception_with_internal_diagnostics(tmp_path):
    script=tmp_path/'failure.py'
    script.write_text('import sys\nprint("synthetic-private-diagnostic",file=sys.stderr)\nsys.exit(7)\n')
    with pytest.raises(subprocess.CalledProcessError) as caught:
        await run_tool_script(script)
    assert caught.value.returncode==7 and 'synthetic-private' in caught.value.stderr


@pytest.mark.parametrize("size,stream",[(65000,"stdout"),(1000000,"stdout"),(1000000,"stderr")])
async def test_script_output_limit_fails_instead_of_truncating_a_success(tmp_path,size,stream):
    script=tmp_path/'output.py'
    script.write_text(f'import sys\nprint("x"*{size},file=sys.{stream})\n')
    with pytest.raises(ToolCodeError,match='output limit'):
        await asyncio.wait_for(run_tool_script(script),2)


@pytest.mark.skipif(os.name!='posix',reason='Process-group lifetime proof uses POSIX process state')
@pytest.mark.parametrize('cancelled',[False,True])
async def test_timeout_or_cancellation_terminates_the_owned_process_group(tmp_path, cancelled):
    script=tmp_path/'wait.py'
    child_file=tmp_path/'child.pid'
    parent_file=tmp_path/'parent.pid'
    script.write_text('import os, pathlib, subprocess, sys, time\n'
        'child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"])\n'
        f'pathlib.Path({str(child_file)!r}).write_text(str(child.pid))\n'
        f'pathlib.Path({str(parent_file)!r}).write_text(str(os.getpid()))\n'
        'time.sleep(60)\n')
    task=asyncio.create_task(run_tool_script(script,timeout=5 if cancelled else .7))
    async def started():
        while not parent_file.exists():
            if task.done():
                await task
            await asyncio.sleep(.01)
    await asyncio.wait_for(started(),2)
    if cancelled:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancelled else asyncio.TimeoutError):
        await task
    for path in [parent_file,child_file]:
        pid=int(path.read_text())
        state=Path('/proc')/str(pid)/'stat'
        if state.exists():
            assert state.read_text().split(') ',1)[1].split()[0]=='Z'


@pytest.mark.parametrize('success',[False,True])
async def test_write_code_has_honest_partial_and_success_receipts(tool_state, success):
    code='print("周五执行，出差顺延。")' if success else 'raise ValueError("synthetic-private-code-detail")'
    logs=[]
    sink=tools.logger.add(lambda message:logs.append(str(message)))
    try:
        if success:
            result=await tools.execute_tool('write_code',{'code':code,'filename':'case.py'},bot=tool_state.bot,event=tool_state.event)
            assert '已保存并运行成功' in result and '周五执行，出差顺延。' in result
        else:
            with pytest.raises(tools.ToolError,match='已保存，但执行未完成') as caught:
                await tools.execute_tool('write_code',{'code':code,'filename':'case.py'},bot=tool_state.bot,event=tool_state.event)
            assert 'synthetic-private' not in str(caught.value)
    finally:
        tools.logger.remove(sink)
    assert [file.read_text(encoding='utf-8') for file in (tool_state.root/'generated').rglob('case.py')]==[code]
    assert 'synthetic-private' not in ''.join(logs)


@pytest.mark.parametrize('filename,folder',[('../escape.py',None),('safe.py','../outside')])
async def test_generated_code_cannot_escape_its_working_directory(tool_state, filename, folder):
    with pytest.raises(tools.ToolError,match='工作目录'):
        await tools.execute_tool('write_code',{'code':'print(42)','filename':filename,'folder_path':folder},bot=tool_state.bot,event=tool_state.event)
    assert list(tool_state.root.rglob('*.py'))==[]


async def test_send_file_distinguishes_absence_from_directory_and_transport_failure(tool_state, monkeypatch):
    directory=tool_state.root/'files'
    monkeypatch.setenv('SEND_FILE_DIR',str(directory))
    with pytest.raises(tools.ToolError,match='目录不可用'):
        await tools.execute_tool('send_file',{'filename':'case.txt'},bot=tool_state.bot,event=tool_state.event)
    directory.mkdir()
    assert await tools.execute_tool('send_file',{'filename':'case.txt'},bot=tool_state.bot,event=tool_state.event)=='未找到匹配的文件'
    (directory/'case.txt').write_text('synthetic content')
    failure=OSError('synthetic-private-platform-error')
    tool_state.bot.upload_private_file.side_effect=failure
    with pytest.raises(OSError) as caught:
        await tools.execute_tool('send_file',{'filename':'case.txt'},bot=tool_state.bot,event=tool_state.event)
    assert caught.value is failure
    tool_state.bot.upload_private_file.side_effect=None
    assert await tools.execute_tool('send_file',{'filename':'case.txt'},bot=tool_state.bot,event=tool_state.event)=='文件 case.txt 已发送'


@pytest.mark.parametrize('failure',[False,True])
async def test_real_tool_failure_never_reaches_completion_narration(bot_module, private_event, tool_state, monkeypatch, failure):
    monkeypatch.setattr(bot_module,'config',SimpleNamespace(BOT_NAME='合成机器人'))
    code='raise ValueError("synthetic-private-error")' if failure else 'print("周五执行，出差顺延。")'
    import json
    completion=AsyncMock(side_effect=[json.dumps({'tool':'write_code','args':{'filename':'case.py','code':code}}),'任务完成。'])
    monkeypatch.setattr(bot_module,'_unified_generate',completion)
    if failure:
        with pytest.raises(tools.ToolError):
            await bot_module.handle_claw(tool_state.bot,'编写代码输出周五安排及例外',private_event,'default')
        assert completion.await_count==1
        tool_state.bot.send.assert_not_awaited()
    else:
        await bot_module.handle_claw(tool_state.bot,'编写代码输出周五安排及例外',private_event,'default')
        assert completion.await_count==2
        assert any('已保存并运行成功' in str(call.args) for call in tool_state.bot.send.await_args_list)


async def test_message_entry_reports_tool_partial_failure_once_without_private_details(bot_module, message_handler, tool_state, monkeypatch):
    from nonebot.exception import FinishedException
    matcher,event,bot=message_handler
    bot_module.claw_sessions[str(event.user_id)]=True
    monkeypatch.setattr(bot_module,'_is_duplicate_message',AsyncMock(return_value=False))
    monkeypatch.setattr(bot_module,'handle_claw',AsyncMock(side_effect=tools.ToolError('代码文件已保存，但执行未完成。')))
    with pytest.raises(FinishedException):
        await matcher.handlers[0](bot,event)
    matcher.finish.assert_awaited_once()
    assert '已保存，但执行未完成' in matcher.finish.await_args.args[0]
