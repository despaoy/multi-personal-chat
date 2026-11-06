"""Claw 工具管理 API"""

import asyncio
import logging
import os

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from services.tool_execution import ToolCodeError, run_tool_code

from app.dependencies import get_current_admin, get_request_database
from infra.db_executor import run_db

logger = logging.getLogger(__name__)
router = APIRouter()

BUILTIN_TOOLS = [
    {"name": "status", "description": "获取系统状态，包括CPU、内存、磁盘占用率", "builtin": True},
    {"name": "show_tools", "description": "显示所有可使用工具", "builtin": True},
    {"name": "send_file", "description": "发送桌面上指定名称的文件", "builtin": True},
    {"name": "get_coderesult", "description": "获取代码执行结果", "builtin": True},
    {"name": "write_code", "description": "编写并执行 Python 代码解决问题", "builtin": True},
]


class ToolSaveRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=2000)
    code: str = Field(default="", max_length=100_000)
    enabled: bool = True


class ToolExecuteRequest(BaseModel):
    code: str = Field(min_length=1, max_length=20_000)
    args: dict = Field(default_factory=dict)


@router.get("/api/claw/tools")
async def list_tools(current_user: dict = Depends(get_current_admin), database=Depends(get_request_database)):
    """列出所有 Claw 工具（内置 + 自定义）"""
    custom_tools = await run_db(database.get_claw_tools)
    result = [{**tool, "code": "", "enabled": True} for tool in BUILTIN_TOOLS]
    # 自定义工具在后
    for t in custom_tools:
        result.append(
            {
                "name": t["name"],
                "description": t["description"],
                "code": t["code"],
                "enabled": bool(t["enabled"]),
                "builtin": False,
                "created_at": t["created_at"],
                "updated_at": t["updated_at"],
            }
        )
    return {"success": True, "tools": result, "total": len(result)}


@router.post("/api/claw/tools")
async def save_tool(
    req: ToolSaveRequest, current_user: dict = Depends(get_current_admin), database=Depends(get_request_database)
):
    """Save custom tool code verbatim in the current application store."""
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="工具名称不能为空")
    if any(tool["name"] == name for tool in BUILTIN_TOOLS):
        raise HTTPException(status_code=400, detail=f"工具名称 '{name}' 与内置工具冲突")
    await run_db(database.save_claw_tool, name, req.description.strip(), req.code, req.enabled)
    return {"success": True, "message": "工具已保存"}


@router.delete("/api/claw/tools/{name}")
async def delete_tool(
    name: str, current_user: dict = Depends(get_current_admin), database=Depends(get_request_database)
):
    """Delete a custom registration from the current application store."""
    if any(tool["name"] == name for tool in BUILTIN_TOOLS):
        raise HTTPException(status_code=400, detail="不能删除内置工具")
    if not await run_db(database.delete_claw_tool, name):
        raise HTTPException(status_code=404, detail="工具不存在")
    return {"success": True, "message": "工具已删除"}


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
        "1",
        "true",
        "yes",
        "on",
    }
    if production and not explicitly_enabled:
        raise HTTPException(status_code=403, detail="Claw code execution is disabled in production")

    try:
        return await asyncio.to_thread(run_tool_code, req.code, req.args)
    except ToolCodeError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except Exception:
        logger.exception("Claw sandbox process failed")
        raise HTTPException(status_code=503, detail="Claw executor is temporarily unavailable")
