"""Claw 工具管理 API"""
import asyncio
import logging
import os

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from db.adapter import db
from app.dependencies import get_current_admin
from infra.db_executor import run_db
from services.tool_execution import ToolCodeError, run_tool_code

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
        return await asyncio.to_thread(run_tool_code, req.code, req.args)
    except ToolCodeError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except Exception:
        logger.exception("Claw sandbox process failed")
        raise HTTPException(status_code=503, detail="Claw executor is temporarily unavailable")