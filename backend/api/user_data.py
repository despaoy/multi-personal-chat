"""用户数据持久化API"""

from fastapi import APIRouter, HTTPException, Depends

from db.schemas import UserDataRequest
from db.adapter import db
from app.dependencies import get_current_user

router = APIRouter()


@router.get("/api/user/data")
async def get_user_data(page_key: str = "", current_user: dict = Depends(get_current_user)):
    """获取用户表单数据"""
    # get_current_user 返回 {"user_id": ..., "username": ...}（无 "sub" 键）
    username = current_user.get("username") or "unknown"

    try:
        user = db.get_user_by_username(username)
        if not user:
            raise HTTPException(status_code=404, detail="用户不存在")
        user_id = user['id']

        if page_key:
            data = db.get_user_data(user_id, page_key)
            if not data:
                return {"success": True, "data": None}
            return {
                "success": True,
                "data": data
            }
        else:
            data = db.get_user_data(user_id)
            return {"success": True, "data": data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取数据失败: {str(e)}")


@router.put("/api/user/data")
async def save_user_data(request: UserDataRequest, current_user: dict = Depends(get_current_user)):
    """保存用户表单数据"""
    # get_current_user 返回 {"user_id": ..., "username": ...}（无 "sub" 键）
    username = current_user.get("username") or "unknown"

    try:
        user = db.get_user_by_username(username)
        if not user:
            raise HTTPException(status_code=404, detail="用户不存在")
        user_id = user['id']

        db.save_user_data(user_id, request.page_key, request.data_json)
        return {"success": True, "message": "数据保存成功"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存失败: {str(e)}")
