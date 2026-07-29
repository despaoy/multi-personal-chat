"""模型管理API"""
import logging

from fastapi import APIRouter, HTTPException, Depends
from app.dependencies import get_current_user, get_current_admin

from db.models import ModelDownloadRequest

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/models")
async def list_models(current_user: dict = Depends(get_current_user)):
    """列出所有可用的模型"""
    try:
        from inference.model_manager import get_model_manager
        manager = get_model_manager()

        models = manager.list_available_models()

        return {
            "success": True,
            "models": models
        }
    except Exception as e:
        logger.error(f"列出模型失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/models/check/{model_name}")
async def check_model(model_name: str, current_user: dict = Depends(get_current_user)):
    """检查模型是否已下载"""
    try:
        from inference.model_manager import get_model_manager
        manager = get_model_manager()

        exists = manager.check_model_exists(model_name)

        return {
            "success": True,
            "model_name": model_name,
            "downloaded": exists
        }
    except Exception as e:
        logger.error(f"检查模型失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/models/download")
async def download_model(request: ModelDownloadRequest, current_user: dict = Depends(get_current_admin)):
    """下载模型

    C-S1 fix: 模型下载占用大量磁盘/带宽，限定 admin。
    """
    try:
        from inference.model_manager import get_model_manager
        manager = get_model_manager()

        result = manager.download_model_from_hf(
            model_name=request.model_name,
            force=request.force
        )

        return result
    except Exception as e:
        logger.error(f"下载模型失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/models/{model_name}")
async def delete_model(model_name: str, current_user: dict = Depends(get_current_admin)):
    """删除模型

    C-S1 fix: 模型删除为不可逆操作，限定 admin。
    """
    try:
        from inference.model_manager import get_model_manager
        manager = get_model_manager()

        success = manager.delete_model(model_name)

        if success:
            return {
                "success": True,
                "message": "模型已删除"
            }
        else:
            raise HTTPException(status_code=400, detail="删除模型失败")
    except HTTPException:
        # C1 fix: HTTPException 是 Exception 子类，必须单独捕获并 re-raise，
        # 否则下方 except Exception 会把 400 改写成 500，丢失原始状态码与语义。
        # 对比 api/router.py:185、api/auth.py:116 均使用此模式。
        raise
    except Exception as e:
        logger.error(f"删除模型失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/models/check-7b")
async def check_and_download_7b_model(current_user: dict = Depends(get_current_user)):
    """检查并自动下载7B模型（如果不存在）"""
    try:
        from inference.model_manager import get_model_manager
        manager = get_model_manager()

        model_name = "qwen3-8b"

        if manager.check_model_exists(model_name):
            return {
                "success": True,
                "message": "7B模型已存在",
                "model_name": model_name,
                "downloaded": True
            }

        logger.info("7B模型不存在，开始自动下载...")
        result = manager.download_model_from_hf(model_name=model_name)

        return result
    except Exception as e:
        logger.error(f"检查/下载7B模型失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
