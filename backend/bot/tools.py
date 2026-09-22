"""
QQ机器人工具注册模块
基于NoneBot的工具注册与分发系统，提供系统状态查询、文件发送、代码执行等能力。
通过装饰器模式注册工具，运行时按名称动态分发执行。
"""

from typing import Dict, Callable, Any
import psutil
from typing import Optional
from nonebot import logger
import os
from pathlib import Path
from nonebot.adapters.onebot.v11 import Bot,MessageEvent
TOOLS: Dict[str, Dict[str, Any]] = {}
def register_tool(name: str, description: str, handler: Callable):
    """注册一个工具到全局工具表。

    Args:
        name: 工具名称（唯一标识）
        description: 工具功能描述
        handler: 工具处理函数
    """
    TOOLS[name] = {
        "description": description,
        "handler": handler,
    }

async def _status_handler(bot: Bot, event: MessageEvent) ->str:
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    import platform as _platform
    path = 'C:\\' if _platform.system() == 'Windows' else '/'
    disk = psutil.disk_usage(path)
    logger.info(f"[status] CPU: {cpu}%, MEM: {mem.percent}%, DISK: {disk.percent}%")
    return f"CPU: {cpu}%, MEM: {mem.percent}%, DISK: {disk.percent}%"
register_tool("status", "获取系统状态，包括CPU、内存、磁盘占用率", _status_handler)

async def _show_tools_handler(bot: Bot, event: MessageEvent)->str:
    lines = []
    for name, tool in TOOLS.items():
        lines.append(f"{name}: {tool['description']}")
        logger.info(f"[show_tools] 显示工具 {name}: {tool['description']}")
    logger.info(f"[show_tools] 可使用工具: {lines}")
    return "\n".join(lines)
register_tool("show_tools", "显示所有可使用工具", _show_tools_handler)

async def _send_file_handler(filename: str,bot: Bot, event: MessageEvent) -> str:
    """发送文件"""
    import platform as _platform
    base = Path.home() / "Desktop" if _platform.system() == "Windows" else Path.home()
    # 支持通过环境变量自定义发送目录
    send_dir = Path(os.getenv("SEND_FILE_DIR", str(base)))
    if not send_dir.exists():
        return f"发送目录不存在: {send_dir}"
    all_files=[f for f in os.listdir(send_dir) if Path(send_dir/f).is_file()]
    
    from difflib import get_close_matches
    matchs=get_close_matches(filename,all_files,n=5,cutoff=0.3)
    if not matchs:
        return f"未找到匹配的文件"
    if(len(matchs)==1):
        file_path=str(send_dir/matchs[0])
        file_name=matchs[0]
        await bot.upload_private_file(user_id=event.user_id,
        file=file_path,name=file_name)
        logger.info(f"[send_file] 发送文件 {file_name}")
        return f"文件 {file_name} 已发送"
    else:
        return f"找到多个匹配的文件: {matchs[:5]}"
register_tool("send_file", "发送桌面上指定名称的文件。参数: filename(文件名关键部分)", _send_file_handler)

async def _get_coderesult_handler(filename: str, filepath: str, bot=None, event: MessageEvent = None):
    code_path = Path(filepath) / filename
    import subprocess
    result = subprocess.run(["python", str(code_path)], capture_output=True, text=True, timeout=30, cwd=str(filepath))
    if result.returncode != 0:
        return result.stderr.strip() or "代码执行失败"
    else:
        return result.stdout.strip() or "代码执行成功"
        
register_tool("get_coderesult", "获取代码执行结果。参数: filename(文件名)、filepath(文件所在目录)", _get_coderesult_handler)


async def _write_code_handler(code: str , filename: str,bot=None ,folder_path:str=None, event: MessageEvent = None) -> str:
    logger.info(f"代码名称：{filename}")
    logger.info(f"代码：{code}")
    from datetime import datetime
    
    if folder_path is None:
        folder_name= datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        folder_name=folder_path
    task_dir = Path(__file__).parent / "code" / folder_name
    task_dir.mkdir(parents=True, exist_ok=True)
    with open(task_dir/filename,"w")as f:
        f.write(code)
    
    logger.info(f"代码 {filename} 已写入 {task_dir/filename}")
    result= await _get_coderesult_handler(filename, filepath=str(task_dir))
    return f"📝 已保存 → {folder_name}/{filename}\n⏳ 执行完成\n📤 结果:\n{result}"

register_tool("write_code", "当找不到解决问题的工具时，自己编写代码解决问题。参数:code(代码)、filename(文件名)", _write_code_handler)

def get_tools(name:str)->Optional[Dict[str, Any]]:
    """按名称获取已注册的工具。

    Args:
        name: 工具名称

    Returns:
        工具信息字典（含description和handler），不存在返回None
    """
    return TOOLS.get(name, None)

async def execute_tool(name:str, args:dict = None,bot: Bot = None, event: MessageEvent = None)->str:
    """按名称动态分发执行工具。

    Args:
        name: 工具名称
        args: 传递给工具处理函数的参数字典
        bot: NoneBot Bot实例
        event: NoneBot消息事件

    Returns:
        工具执行结果字符串，工具不存在时返回错误提示
    """
    logger.info(f"[tool] 执行: {name}, 参数: {args}")
    tool = get_tools(name)
    if tool is None:
        return f"工具 {name} 不存在"
    handler = tool["handler"]
    kwargs = {"bot": bot, "event": event}
    if args:
        for k, v in args.items():
            if k not in kwargs:
                kwargs[k] = v
    return await handler(**kwargs)
