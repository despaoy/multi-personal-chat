"""用户数据持久化API"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_current_user
from app.providers import get_user_data_repository
from db.schemas import UserDataRequest
from repositories.user_data import UserDataRepository, UserDataUserNotFoundError

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/user/data")
async def get_user_data(
    page_key: str = "",
    current_user: dict = Depends(get_current_user),
    repository: UserDataRepository = Depends(get_user_data_repository),
):
    """获取用户表单数据"""
    # get_current_user 返回 {"user_id": ..., "username": ...}（无 "sub" 键）
    username = current_user.get("username") or "unknown"

    try:
        data = await repository.load(username, page_key or None)
        if page_key and not data:
            return {"success": True, "data": None}
        return {"success": True, "data": data}
    except UserDataUserNotFoundError:
        raise HTTPException(status_code=404, detail="用户不存在") from None
    except HTTPException:
        raise
    except Exception:
        logger.error("Failed to load user data", exc_info=True)
        raise HTTPException(status_code=500, detail="获取数据失败") from None


@router.put("/api/user/data")
async def save_user_data(
    request: UserDataRequest,
    current_user: dict = Depends(get_current_user),
    repository: UserDataRepository = Depends(get_user_data_repository),
):
    """保存用户表单数据"""
    # get_current_user 返回 {"user_id": ..., "username": ...}（无 "sub" 键）
    username = current_user.get("username") or "unknown"

    try:
        await repository.save(username, request.page_key, request.data_json)
        return {"success": True, "message": "数据保存成功"}
    except UserDataUserNotFoundError:
        raise HTTPException(status_code=404, detail="用户不存在") from None
    except HTTPException:
        raise
    except Exception:
        logger.error("Failed to save user data", exc_info=True)
        raise HTTPException(status_code=500, detail="保存失败") from None
