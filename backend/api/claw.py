"""Claw 工具管理 API"""
import ast
import asyncio
import datetime
import io
import json
import logging
import multiprocessing
import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from db.adapter import db
from app.dependencies import get_current_admin
from infra.db_executor import run_db

logger = logging.getLogger(__name__)
router = APIRouter()

# ── 内置工具列表（来自 bot/tools.py）──
BUILTIN_TOOLS = [
    {"name": "status", "description": "获取系统状态，包括CPU、内存、磁盘占用率", "builtin": True},
    {"name": "show_tools", "description": "显示所有可使用工具", "builtin": True},
    {"name": "send_file", "description": "发送桌面上指定名称的文件", "builtin": True},
    {"name": "get_coderesult", "description": "获取代码执行结果", "builtin": True},
    {"name": "write_code", "description": "编写并执行 Python 代码解决问题", "builtin": True},
]

# ── 请求/响应模型 ──

class ToolSaveRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=2000)
    code: str = Field(default="", max_length=100_000)
    enabled: bool = True


class ToolExecuteRequest(BaseModel):
    code: str = Field(min_length=1, max_length=20_000)
    args: dict = Field(default_factory=dict)


# ── 工具列表 ──

@router.get("/api/claw/tools")
async def list_tools(current_user: dict = Depends(get_current_admin)):
    """列出所有 Claw 工具（内置 + 自定义）"""
    custom_tools = await run_db(db.get_claw_tools)
    result = []
    # 内置工具在前
    for t in BUILTIN_TOOLS:
        result.append({
            "name": t["name"],
            "description": t["description"],
            "code": "",
            "enabled": True,
            "builtin": True,
        })
    # 自定义工具在后
    for t in custom_tools:
        result.append({
            "name": t["name"],
            "description": t["description"],
            "code": t["code"],
            "enabled": bool(t["enabled"]),
            "builtin": False,
            "created_at": t["created_at"],
            "updated_at": t["updated_at"],
        })
    return {"success": True, "tools": result, "total": len(result)}


# ── 保存（创建/更新）工具 ──

@router.post("/api/claw/tools")
async def save_tool(req: ToolSaveRequest, current_user: dict = Depends(get_current_admin)):
    """创建或更新自定义 Claw 工具

    s2 fix: 工具包含可执行 Python 代码，限定 admin。
    """
    if not req.name or not req.name.strip():
        raise HTTPException(status_code=400, detail="工具名称不能为空")
    
    # 检查是否与内置工具重名
    for bt in BUILTIN_TOOLS:
        if bt["name"] == req.name.strip():
            raise HTTPException(status_code=400, detail=f"工具名称 '{req.name}' 与内置工具冲突")
    
    try:
        await run_db(db.save_claw_tool, req.name.strip(), req.description.strip(), req.code, req.enabled)
        return {"success": True, "message": "工具已保存"}
    except Exception as e:
        logger.error(f"保存 Claw 工具失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── 删除工具 ──

@router.delete("/api/claw/tools/{name}")
async def delete_tool(name: str, current_user: dict = Depends(get_current_admin)):
    """删除自定义 Claw 工具

    s2 fix: 工具包含可执行代码，删除影响全局，限定 admin。
    """
    for bt in BUILTIN_TOOLS:
        if bt["name"] == name:
            raise HTTPException(status_code=400, detail="不能删除内置工具")
    
    success = await run_db(db.delete_claw_tool, name)
    if not success:
        raise HTTPException(status_code=404, detail="工具不存在")
    return {"success": True, "message": "工具已删除"}


# ── 测试执行工具代码 ──

# This executor is a convenience tool, not a security boundary. Production use
# is opt-in and runs in a short-lived child process with a hard timeout.
_SAFE_MODULE_NAMES = {
    "json", "datetime", "math", "re", "collections", "itertools", "statistics",
}
_FORBIDDEN_TOKENS = (
    "import os", "from os", "import subprocess", "from subprocess",
    "import sys", "from sys", "__import__", "builtins",
    "open(", "exec(", "eval(", "compile(", "globals(", "locals(",
    "socket.", "shutil.", "pathlib", "ctypes", "pickle",
)
_MAX_OUTPUT_CHARS = 64_000


class _CappedWriter(io.StringIO):
    def write(self, value):
        remaining = _MAX_OUTPUT_CHARS - self.tell()
        if remaining <= 0:
            return len(value)
        super().write(str(value)[:remaining])
        return len(value)


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    top_level = name.split(".", 1)[0]
    if level != 0 or top_level not in _SAFE_MODULE_NAMES:
        raise ImportError(f"module '{name}' is not allowed")
    return __import__(name, globals, locals, fromlist, level)


def _safe_builtins():
    return {
        "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
        "range": range, "enumerate": enumerate, "zip": zip, "map": map,
        "filter": filter, "sorted": sorted, "reversed": reversed,
        "round": round, "isinstance": isinstance, "type": type,
        "list": list, "dict": dict, "tuple": tuple, "set": set,
        "str": str, "int": int, "float": float, "bool": bool,
        "True": True, "False": False, "None": None,
        "print": print, "repr": repr, "format": format,
        "any": any, "all": all,
        "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
        "__import__": _safe_import,
    }


def _validate_tool_code(code: str) -> None:
    lowered = code.lower()
    for token in _FORBIDDEN_TOKENS:
        if token in lowered:
            raise ValueError(f"code contains forbidden operation: {token}")

    wrapped = "def _validate_claw_code():\n" + "\n".join(
        "    " + line for line in code.splitlines()
    )
    try:
        tree = ast.parse(wrapped, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"invalid Python syntax: {exc.msg}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = {alias.name.split(".", 1)[0] for alias in node.names}
            if not names.issubset(_SAFE_MODULE_NAMES):
                raise ValueError("code imports a module outside the allowlist")
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module or node.module.split(".", 1)[0] not in _SAFE_MODULE_NAMES:
                raise ValueError("code imports a module outside the allowlist")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("private and dunder attributes are not allowed")
        elif isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("dunder names are not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            raise ValueError("global and nonlocal statements are not allowed")


def _sandbox_worker(code: str, args: dict, result_channel) -> None:
    stdout = _CappedWriter()
    stderr = _CappedWriter()
    try:
        try:
            import resource
            memory_limit = int(os.getenv("CLAW_MEMORY_LIMIT_MB", "4096")) * 1024 * 1024
            cpu_limit = max(1, int(float(os.getenv("CLAW_EXECUTION_TIMEOUT", "3"))))
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit + 1))
        except (ImportError, OSError, ValueError):
            pass

        import contextlib
        # exec-defined functions resolve names through the globals mapping.
        # Use one restricted namespace so _claw_main and args remain visible
        # to the generated module-level invocation.
        restricted_globals = {"__builtins__": _safe_builtins(), "args": args, "result": None}
        wrapped_code = (
            "def _claw_main(**kwargs):\n"
            + "\n".join("    " + line for line in code.splitlines())
            + "\n    return None\n"
            "_result = _claw_main(**args)\n"
        )
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(wrapped_code, restricted_globals, restricted_globals)  # noqa: S102
        value = restricted_globals.get("_result")
        result_channel.send({
            "success": True,
            "output": stdout.getvalue(),
            "error": stderr.getvalue(),
            "result": "" if value is None else str(value)[:_MAX_OUTPUT_CHARS],
        })
    except BaseException as exc:
        result_channel.send({
            "success": False,
            "output": stdout.getvalue(),
            "error": f"{type(exc).__name__}: {exc}"[:_MAX_OUTPUT_CHARS],
            "result": "",
        })


def _run_in_sandbox_process(code: str, args: dict) -> dict:
    timeout = min(10.0, max(0.5, float(os.getenv("CLAW_EXECUTION_TIMEOUT", "3"))))
    # ``spawn`` re-imports the active test runner on POSIX, which can make a
    # healthy sandbox worker exit before it writes to the result pipe. ``fork``
    # keeps the same process boundary and timeout semantics without that import path.
    start_method = "fork" if os.name == "posix" else "spawn"
    context = multiprocessing.get_context(start_method)
    result_reader, result_writer = context.Pipe(duplex=False)
    process = context.Process(target=_sandbox_worker, args=(code, args, result_writer), daemon=True)
    process.start()
    result_writer.close()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(1)
        return {"success": False, "output": "", "error": "execution timed out", "result": ""}

    try:
        if result_reader.poll(0.5):
            try:
                return result_reader.recv()
            except EOFError:
                pass
        return {
            "success": False,
            "output": "",
            "error": f"executor exited without a result (exit code {process.exitcode})",
            "result": "",
        }
    finally:
        result_reader.close()


@router.post("/api/claw/tools/execute")
async def execute_tool_code(req: ToolExecuteRequest, current_user: dict = Depends(get_current_admin)):
    """Execute a small tool in an isolated child process.

    s2 fix: 执行任意用户代码（子进程沙箱），限定 admin。

    Production execution is disabled unless CLAW_CODE_EXECUTION_ENABLED=true.
    This process isolation limits hangs and accidental damage, but it is not a
    replacement for a locked-down container or microVM.
    """
    production = os.getenv("ENVIRONMENT", "development").strip().lower() == "production"
    explicitly_enabled = os.getenv("CLAW_CODE_EXECUTION_ENABLED", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    if production and not explicitly_enabled:
        raise HTTPException(status_code=403, detail="Claw code execution is disabled in production")

    try:
        _validate_tool_code(req.code)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    try:
        return await asyncio.to_thread(_run_in_sandbox_process, req.code, req.args)
    except Exception:
        logger.exception("Claw sandbox process failed")
        raise HTTPException(status_code=503, detail="Claw executor is temporarily unavailable")