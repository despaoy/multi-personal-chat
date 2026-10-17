"""模型管理API"""
import logging

from fastapi import APIRouter, HTTPException, Depends
from app.dependencies import get_current_admin
from app.providers import get_model_management_service

from db.schemas import ModelDownloadRequest
from services.model_management import ModelManagementService

logger = logging.getLogger(__name__)
router = APIRouter()


def _require_download_success(result):
    """Convert manager-level failures into stable HTTP error semantics."""

    if isinstance(result, dict) and result.get("success") is True:
        return result
    error = result.get("error", "") if isinstance(result, dict) else ""
    status_code = 400 if str(error).startswith("未知模型:") else 502
    raise HTTPException(status_code=status_code, detail="模型下载失败")

@router.get("/api/models")
async def list_models(
    current_user: dict = Depends(get_current_admin),
    service: ModelManagementService = Depends(get_model_management_service),
):
    """列出所有可用的模型"""
    try:
        models = await service.list_available_models()

        return {
            "success": True,
            "models": models
        }
    except Exception:
        logger.exception("列出模型失败")
        raise HTTPException(status_code=500, detail="列出模型失败")


@router.get("/api/models/check/{model_name}")
async def check_model(
    model_name: str,
    current_user: dict = Depends(get_current_admin),
    service: ModelManagementService = Depends(get_model_management_service),
):
    """检查模型是否已下载"""
    try:
        exists = await service.check_model_exists(model_name)

        return {
            "success": True,
            "model_name": model_name,
            "downloaded": exists
        }
    except Exception:
        logger.exception("检查模型失败")
        raise HTTPException(status_code=500, detail="检查模型失败")


@router.post("/api/models/download")
async def download_model(
    request: ModelDownloadRequest,
    current_user: dict = Depends(get_current_admin),
    service: ModelManagementService = Depends(get_model_management_service),
):
    """下载模型

    C-S1 fix: 模型下载占用大量磁盘/带宽，限定 admin。
    """
    try:
        result = await service.download_model(request.model_name, force=request.force)
        return _require_download_success(result)
    except HTTPException:
        raise
    except Exception:
        logger.exception("下载模型失败")
        raise HTTPException(status_code=500, detail="下载模型失败")


@router.delete("/api/models/{model_name}")
async def delete_model(
    model_name: str,
    current_user: dict = Depends(get_current_admin),
    service: ModelManagementService = Depends(get_model_management_service),
):
    """Delete model files; an I/O failure may leave a partially deleted directory."""
    try:
        if not await service.delete_model(model_name):
            raise HTTPException(status_code=400, detail="删除模型失败")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("模型删除失败: %s", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail=f"模型删除未完成（{type(exc).__name__}），可能已删除部分文件，请刷新模型列表并检查目录权限后重试",
        ) from exc
    return {"success": True, "message": "模型已删除"}


@router.post("/api/models/check-7b")
async def check_and_download_7b_model(
    current_user: dict = Depends(get_current_admin),
    service: ModelManagementService = Depends(get_model_management_service),
):
    """检查并自动下载7B模型（如果不存在）"""
    try:
        model_name = "qwen3-8b"

        if await service.check_model_exists(model_name):
            return {
                "success": True,
                "message": "7B模型已存在",
                "model_name": model_name,
                "downloaded": True
            }

        logger.info("7B模型不存在，开始自动下载...")
        result = await service.download_model(model_name)
        return _require_download_success(result)
    except HTTPException:
        raise
    except Exception:
        logger.exception("检查/下载7B模型失败")
        raise HTTPException(status_code=500, detail="检查/下载7B模型失败")
