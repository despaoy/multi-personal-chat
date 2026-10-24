"""NoneBot tool protocol, registry and platform operations."""

import asyncio
import inspect
import os
import platform
import subprocess
import uuid
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Callable

import psutil
from nonebot import logger
from nonebot.adapters.onebot.v11 import Bot, MessageEvent
from pydantic import BaseModel, ConfigDict, Field

from services.tool_execution import ToolCodeError, run_tool_script

TOOLS: dict[str, dict[str, Any]] = {}
CODE_ROOT = Path(__file__).parent / "code"


class ToolError(RuntimeError):
    """Public tool failure containing no code, credentials or server paths."""


class ToolCall(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    tool: str = Field(min_length=1)
    args: dict[str, Any]


def register_tool(name: str, description: str, handler: Callable):
    TOOLS[name] = {"description": description, "handler": handler}


async def _status_handler(bot: Bot, event: MessageEvent) -> str:
    cpu = await asyncio.to_thread(psutil.cpu_percent, interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("C:\\" if platform.system() == "Windows" else "/")
    return f"CPU: {cpu}%, MEM: {memory.percent}%, DISK: {disk.percent}%"


async def _show_tools_handler(bot: Bot, event: MessageEvent) -> str:
    return "\n".join(f"{name}: {tool['description']}" for name, tool in TOOLS.items())


async def _send_file_handler(filename: str, bot: Bot, event: MessageEvent) -> str:
    base = Path.home() / "Desktop" if platform.system() == "Windows" else Path.home()
    directory = Path(os.getenv("SEND_FILE_DIR", str(base)))
    if not directory.is_dir():
        raise ToolError("发送目录不可用，请检查文件目录配置。")
    files = [file.name for file in directory.iterdir() if file.is_file()]
    matches = get_close_matches(filename, files, n=5, cutoff=0.3)
    if not matches:
        return "未找到匹配的文件"
    if len(matches) > 1:
        return f"找到多个匹配的文件: {matches}"
    name = matches[0]
    await bot.upload_private_file(user_id=event.user_id, file=str(directory/name), name=name)
    return f"文件 {name} 已发送"


async def _get_coderesult_handler(filename: str, filepath: str, bot=None, event: MessageEvent = None) -> str:
    try:
        result = await run_tool_script(Path(filepath)/filename)
    except (OSError, subprocess.CalledProcessError, asyncio.TimeoutError, ToolCodeError, UnicodeError) as exc:
        raise ToolError("代码执行未完成，请检查代码、执行环境及运行时限。") from exc
    return result or "代码执行成功（无输出）"


async def _write_code_handler(code: str, filename: str, bot=None, folder_path: str | None = None, event: MessageEvent = None) -> str:
    root = CODE_ROOT.resolve()
    directory = (root/(folder_path if folder_path is not None else uuid.uuid4().hex)).resolve()
    path = (directory/filename).resolve()
    if not directory.is_relative_to(root) or path.parent != directory:
        raise ToolError("代码文件必须位于工具工作目录，filename须为文件名。")
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(code, encoding="utf-8")
    try:
        result = await _get_coderesult_handler(path.name, str(directory))
    except ToolError as exc:
        raise ToolError("代码文件已保存，但执行未完成，请检查代码及执行环境。") from exc
    return f"代码文件已保存并运行成功。\n运行结果:\n{result}"


register_tool("status", "获取系统状态，包括CPU、内存、磁盘占用率", _status_handler)
register_tool("show_tools", "显示所有可使用工具", _show_tools_handler)
register_tool("send_file", "发送桌面上指定名称的文件。参数: filename(文件名关键部分)", _send_file_handler)
register_tool("get_coderesult", "获取代码执行结果。参数: filename(文件名)、filepath(文件所在目录)", _get_coderesult_handler)
register_tool("write_code", "编写并执行Python代码。参数: code(代码)、filename(文件名)，可选folder_path(工作目录内子目录)", _write_code_handler)


async def execute_tool(name: str, args: dict, *, bot: Bot, event: MessageEvent) -> str:
    tool = TOOLS.get(name)
    if tool is None:
        raise ToolError("模型选择的工具不存在，请检查工具配置。")
    handler = tool["handler"]
    try:
        inspect.signature(handler).bind(bot=bot, event=event, **args)
    except TypeError as exc:
        raise ToolError("工具参数缺失或无效，本次未执行工具。") from exc
    logger.info("[tool] 执行: {}", name)
    return await handler(bot=bot, event=event, **args)
