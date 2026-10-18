"""知识库API - 知识库/文件夹/文档管理 + ZIP上传 + 文件夹扫描 + 搜索"""

import asyncio
import io
import logging
import os
import threading
import zipfile
from collections.abc import Awaitable
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from knowledge.text_decode import KnowledgeTextEncoding, decode_knowledge_text
from infra.input_validator import KNOWLEDGE_DOCUMENT_SCHEMA, InputValidator
from app.dependencies import get_current_admin
from app.runtime import get_runtime_container
from db.adapter import db
from db.schemas import (
    IntentSampleAddRequest,
    IntentSampleGenerateRequest,
    IntentSamplesSaveRequest,
    IntentSampleUpdateRequest,
    IntentTrainRequest,
    KnowledgeBaseCreate,
    KnowledgeBaseUpdate,
    KnowledgeDocumentCreate,
    KnowledgeDocumentUpdate,
    KnowledgeFolderCreate,
    KnowledgeSearchRequest,
)
from infra.db_executor import run_db
from infra.bounded_executor import BlockingWorkTimeout

logger = logging.getLogger(__name__)

KnowledgeRetrievalMode = Literal["evidence", "hybrid", "keyword", "empty"]

_LEGACY_SEARCH_TYPE: dict[KnowledgeRetrievalMode, str] = {
    "evidence": "rag_pipeline",
    "hybrid": "hybrid",
    "keyword": "keyword",
    "empty": "empty",
}


def _knowledge_search_response(
    query: str,
    results: list[dict[str, Any]],
    retrieval_mode: KnowledgeRetrievalMode,
) -> dict[str, Any]:
    """Build the versioned search contract while preserving legacy clients."""
    return {
        "success": True,
        "query": query,
        "results": results,
        "searchType": _LEGACY_SEARCH_TYPE[retrieval_mode],
        "retrievalMode": retrieval_mode,
    }


router = APIRouter()

_ZIP_MAX_BYTES = int(os.getenv("KNOWLEDGE_ZIP_MAX_BYTES", str(100 * 1024 * 1024)))
_ZIP_MAX_FILES = int(os.getenv("KNOWLEDGE_ZIP_MAX_FILES", "1000"))
_ZIP_MAX_ENTRY_BYTES = int(os.getenv("KNOWLEDGE_ZIP_MAX_ENTRY_BYTES", str(10 * 1024 * 1024)))
_ZIP_MAX_UNCOMPRESSED_BYTES = int(os.getenv("KNOWLEDGE_ZIP_MAX_UNCOMPRESSED_BYTES", str(200 * 1024 * 1024)))
_ZIP_MAX_COMPRESSION_RATIO = float(os.getenv("KNOWLEDGE_ZIP_MAX_COMPRESSION_RATIO", "200"))


_intent_task: asyncio.Task[Any] | None = None
_intent_task_lock: asyncio.Lock | None = None
_intent_task_lock_loop: asyncio.AbstractEventLoop | None = None


def _get_intent_task_lock() -> asyncio.Lock:
    global _intent_task_lock, _intent_task_lock_loop
    loop = asyncio.get_running_loop()
    if _intent_task_lock is None or _intent_task_lock_loop is not loop:
        _intent_task_lock = asyncio.Lock()
        _intent_task_lock_loop = loop
    return _intent_task_lock


def _intent_task_finished(completed: asyncio.Task[Any]) -> None:
    global _intent_task
    if _intent_task is completed:
        _intent_task = None
    if completed.cancelled():
        return
    try:
        error = completed.exception()
    except asyncio.CancelledError:
        return
    if error is not None:
        logger.error("意图任务后台协程异常: %s", error)


def _schedule_intent_task(work: Awaitable[Any], *, name: str) -> asyncio.Task[Any]:
    global _intent_task
    task = asyncio.create_task(work, name=name)
    _intent_task = task
    task.add_done_callback(_intent_task_finished)
    return task


async def shutdown_intent_tasks(timeout: float | None = None) -> None:
    """Cooperatively stop the single intent generation/training job."""
    global _intent_task
    wait_timeout = (
        float(os.getenv("INTENT_TASK_SHUTDOWN_TIMEOUT", "10")) if timeout is None else max(float(timeout), 0.0)
    )
    async with _get_intent_task_lock():
        task = _intent_task
    if task is None or task.done():
        _intent_task = None
        return

    from knowledge.intent_trainer import cancel_training

    cancellation_signalled = cancel_training()
    if not cancellation_signalled and not task.done():
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if _intent_task is task:
            _intent_task = None
        return

    _, pending = await asyncio.wait({task}, timeout=wait_timeout)
    if pending:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    if _intent_task is task:
        _intent_task = None


def _validated_zip_entries(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """Validate archive metadata before reading or writing any document."""
    entries = [entry for entry in archive.infolist() if not entry.is_dir()]
    if len(entries) > _ZIP_MAX_FILES:
        raise HTTPException(status_code=413, detail="ZIP contains too many files")

    total_uncompressed = 0
    seen_names: set[str] = set()
    for entry in entries:
        name = entry.filename
        path = PurePosixPath(name)
        if (
            not name
            or chr(92) in name
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
            or ":" in path.parts[0]
        ):
            raise HTTPException(status_code=400, detail="ZIP contains an unsafe path")
        if name in seen_names:
            raise HTTPException(status_code=400, detail="ZIP contains duplicate file names")
        seen_names.add(name)

        if entry.flag_bits & 0x1:
            raise HTTPException(status_code=400, detail="Encrypted ZIP entries are not supported")
        if entry.file_size > _ZIP_MAX_ENTRY_BYTES:
            raise HTTPException(status_code=413, detail="A ZIP entry exceeds the size limit")

        total_uncompressed += entry.file_size
        if total_uncompressed > _ZIP_MAX_UNCOMPRESSED_BYTES:
            raise HTTPException(status_code=413, detail="ZIP expands beyond the allowed size")
        if entry.file_size and (
            entry.compress_size == 0 or entry.file_size / entry.compress_size > _ZIP_MAX_COMPRESSION_RATIO
        ):
            raise HTTPException(status_code=413, detail="ZIP compression ratio is suspicious")

    return entries


# ============================================
# 知识库管理
# ============================================


@router.get("/api/knowledge/bases")
async def list_knowledge_bases(current_user: dict = Depends(get_current_admin)):
    """获取所有知识库"""
    bases = await run_db(db.get_knowledge_bases)
    return {"success": True, "bases": bases}


@router.post("/api/knowledge/bases")
async def create_knowledge_base(request: KnowledgeBaseCreate, current_user: dict = Depends(get_current_admin)):
    """创建知识库"""
    result = await run_db(db.create_knowledge_base, request.name, request.description)
    if result is None:
        raise HTTPException(status_code=409, detail="知识库名称已存在")
    return {"success": True, "base": result}


@router.put("/api/knowledge/bases/{kb_id}")
async def update_knowledge_base(
    kb_id: int, request: KnowledgeBaseUpdate, current_user: dict = Depends(get_current_admin)
):
    """更新知识库"""
    existing = await run_db(db.get_knowledge_base, kb_id)
    if not existing:
        raise HTTPException(status_code=404, detail="知识库不存在")
    data = {}
    if request.name is not None:
        data["name"] = request.name
    if request.description is not None:
        data["description"] = request.description
    result = await run_db(db.update_knowledge_base, kb_id, data)
    if result is None:
        raise HTTPException(status_code=404, detail="知识库不存在")
    if result['name'] != existing['name']:
        await run_db(_invalidate_local_knowledge_index)
    return {"success": True, "base": result}


@router.delete("/api/knowledge/bases/{kb_id}")
async def delete_knowledge_base(kb_id: int, current_user: dict = Depends(get_current_admin)):
    """删除知识库（级联删除文件夹和文档）

    C-S1 fix: 级联删除不可逆，限定 admin。
    """
    existing = await run_db(db.get_knowledge_base, kb_id)
    if not existing:
        raise HTTPException(status_code=404, detail="知识库不存在")
    if not await run_db(db.delete_knowledge_base, kb_id):
        raise HTTPException(status_code=404, detail="知识库不存在")
    await run_db(_invalidate_local_knowledge_index)
    return {"success": True, "message": "知识库已删除"}


# ============================================
# 文件夹管理
# ============================================


@router.get("/api/knowledge/bases/{kb_id}/folders")
async def list_knowledge_folders(kb_id: int, current_user: dict = Depends(get_current_admin)):
    """获取知识库下的文件夹"""
    existing = await run_db(db.get_knowledge_base, kb_id)
    if not existing:
        raise HTTPException(status_code=404, detail="知识库不存在")
    folders = await run_db(db.get_knowledge_folders, kb_id)
    return {"success": True, "folders": folders}


@router.post("/api/knowledge/bases/{kb_id}/folders")
async def create_knowledge_folder(
    kb_id: int, request: KnowledgeFolderCreate, current_user: dict = Depends(get_current_admin)
):
    """创建文件夹"""
    existing = await run_db(db.get_knowledge_base, kb_id)
    if not existing:
        raise HTTPException(status_code=404, detail="知识库不存在")
    result = await run_db(db.create_knowledge_folder, kb_id, request.name, request.description)
    if result is None:
        raise HTTPException(status_code=409, detail="文件夹名称已存在")
    return {"success": True, "folder": result}


@router.delete("/api/knowledge/folders/{folder_id}")
async def delete_knowledge_folder(folder_id: int, current_user: dict = Depends(get_current_admin)):
    """删除文件夹并将保留的文档移至未分类，限定 admin。"""
    existing = await run_db(db.get_knowledge_folder, folder_id)
    if not existing:
        raise HTTPException(status_code=404, detail="文件夹不存在")
    if not await run_db(db.delete_knowledge_folder, folder_id):
        raise HTTPException(status_code=404, detail="文件夹不存在")
    await run_db(_invalidate_local_knowledge_index)
    return {"success": True, "message": "文件夹已删除"}


# ============================================
# ZIP上传
# ============================================


def _knowledge_import_error(entry: str, error: Exception) -> str:
    logger.warning("知识导入条目处理失败: %s", type(error).__name__)
    if isinstance(error, UnicodeError):
        return f"{entry}：文件编码与所选文本格式不匹配，请选择正确的文本格式后重试。"
    return f"{entry}：处理未完成（{type(error).__name__}），请核对文件权限、数据服务及已保存文档。"


def _knowledge_import_result(
    created_folders: dict, created_docs: int, errors: list[str], *, unconfirmed: bool = False
) -> dict[str, Any]:
    if unconfirmed:
        status = "unconfirmed"
        message = f"导入结果未确认：已确认保存 {created_docs} 个文档，请刷新核对后再决定是否重试。"
    elif errors:
        status = "partial" if created_docs else "failed"
        message = (
            f"部分导入：已保存 {created_docs} 个文档，{len(errors)} 项处理未完成。"
            if created_docs
            else f"导入失败：{len(errors)} 项处理未完成。"
        )
    else:
        status = "complete"
        message = f"已导入 {created_docs} 个文档。" if created_docs else "未找到可导入的文档。"
    logger.log(
        logging.INFO if status == "complete" else logging.WARNING,
        "知识导入结束: status=%s saved=%d errors=%d",
        status,
        created_docs,
        len(errors),
    )
    return {
        "success": status == "complete",
        "status": status,
        "message": message,
        "createdFolders": list(created_folders),
        "createdDocs": created_docs,
        "errors": errors,
    }


@router.post("/api/knowledge/bases/{kb_id}/upload-zip")
async def upload_zip(
    kb_id: int, file: UploadFile = File(...), current_user: dict = Depends(get_current_admin),
    text_encoding: KnowledgeTextEncoding = "unicode",
):
    """上传ZIP文件，自动按目录结构创建文件夹和文档

    ZIP结构要求：
    - 顶层目录名作为文件夹名
    - 顶层目录下的.txt文件作为文档
    - 例: 角色/胡桃.txt, 事件/活动剧情.txt
    """
    existing = await run_db(db.get_knowledge_base, kb_id)
    if not existing:
        raise HTTPException(status_code=404, detail="知识库不存在")

    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="请上传ZIP文件")

    content = await file.read(_ZIP_MAX_BYTES + 1)
    if len(content) > _ZIP_MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"文件大小超过限制 ({_ZIP_MAX_BYTES // 1024 // 1024}MB)")
    created_folders = {}
    created_docs = 0
    errors = []
    unconfirmed = False
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            zip_entries = _validated_zip_entries(zf)
            for zip_entry in zip_entries:
                entry = zip_entry.filename
                # 跳过目录条目和隐藏文件
                if entry.endswith("/") or entry.startswith(".") or "__MACOSX" in entry:
                    continue

                # 解析路径：folder_name/filename.txt
                parts = PurePosixPath(entry).parts
                if len(parts) < 2:
                    # 根目录下的文件，归入"未分类"
                    folder_name = "未分类"
                    filename = parts[0]
                else:
                    folder_name = parts[-2]
                    filename = parts[-1]

                # 只处理文本文件
                if not filename.lower().endswith((".txt", ".md", ".json")):
                    continue

                try:
                    # 获取或创建文件夹
                    if folder_name not in created_folders:
                        folder = await run_db(db.create_knowledge_folder, kb_id, folder_name)
                        if folder is None:
                            # 文件夹已存在，查找它
                            folders = await run_db(db.get_knowledge_folders, kb_id)
                            folder = next((f for f in folders if f["name"] == folder_name), None)
                        if folder:
                            created_folders[folder_name] = folder["id"]
                        else:
                            errors.append(f"无法创建文件夹: {folder_name}")
                            continue

                    folder_id = created_folders[folder_name]

                    # 读取文件内容
                    file_bytes = zf.read(zip_entry)
                    file_content = decode_knowledge_text(file_bytes, text_encoding)

                    if not file_content.strip():
                        continue

                    # 文档标题 = 文件名（去掉扩展名）
                    doc_title = filename.rsplit(".", 1)[0]

                    # 创建文档 - 注入文件夹路径到category
                    document_data = {
                        "title": doc_title,
                        "content": file_content,
                        "category": folder_name,
                        "knowledge_base_id": kb_id,
                        "folder_id": folder_id,
                        "sourceType": "file",
                        "fileType": filename.rsplit(".", 1)[-1] if "." in filename else "txt",
                        "fileSize": len(file_bytes),
                    }

                    # 分块处理
                    from knowledge.text_splitter import simple_text_split

                    chunks = simple_text_split(file_content)
                    await run_db(db.save_knowledge_document, document_data, chunks=chunks)
                    created_docs += 1

                    # 数据与 dirty 修订已原子提交；此处仅清除当前进程的加速状态
                    await run_db(_invalidate_local_knowledge_index)

                except BlockingWorkTimeout:
                    raise
                except Exception as exc:
                    errors.append(_knowledge_import_error(entry, exc))
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="无效的ZIP文件") from exc
    except BlockingWorkTimeout:
        unconfirmed = True
        errors.append("导入操作超时，结果未确认，请刷新核对后再决定是否重试。")
    except HTTPException:
        raise
    except Exception as exc:
        errors.append(_knowledge_import_error(file.filename, exc))
    return _knowledge_import_result(created_folders, created_docs, errors, unconfirmed=unconfirmed)


# ============================================
# 文件夹扫描
# ============================================

KNOWLEDGE_BASES_DIR = Path(__file__).parent.parent / "knowledge_bases"
SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".csv", ".html", ".xml"}


def _knowledge_source_path(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(KNOWLEDGE_BASES_DIR.resolve()):
        raise HTTPException(status_code=400, detail="来源路径超出知识库目录范围")
    return resolved


def _scan_directory(directory: Path, ancestors: frozenset[Path] = frozenset()) -> dict:
    """扫描目录结构，返回树形结构"""
    resolved = _knowledge_source_path(directory)
    if resolved in ancestors:
        raise HTTPException(status_code=400, detail="知识库来源目录存在循环链接")
    ancestors = ancestors | {resolved}
    result = {"name": directory.name, "type": "folder", "children": [], "fileCount": 0, "totalSize": 0}
    if not directory.exists():
        return result

    for item in sorted(directory.iterdir()):
        if item.name.startswith(".") or item.name == "__pycache__":
            continue
        if item.is_dir():
            sub = _scan_directory(item, ancestors)
            result["children"].append(sub)
            result["fileCount"] += sub["fileCount"]
            result["totalSize"] += sub["totalSize"]
        elif item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
            _knowledge_source_path(item)
            file_size = item.stat().st_size
            result["children"].append(
                {
                    "name": item.name,
                    "type": "file",
                    "size": file_size,
                    "extension": item.suffix.lower(),
                }
            )
            result["fileCount"] += 1
            result["totalSize"] += file_size
    return result


def _scan_knowledge_dirs() -> dict:
    """扫描 knowledge_bases 目录，返回所有可用的知识库文件夹结构

    扫描 backend/knowledge_bases/ 下的所有子目录，
    每个顶层子目录被视为一个知识库候选项。
    """
    if not KNOWLEDGE_BASES_DIR.exists():
        return {"success": True, "directories": [], "message": "知识库目录不存在"}

    directories = []
    for item in sorted(KNOWLEDGE_BASES_DIR.iterdir()):
        if not item.is_dir() or item.name.startswith("."):
            continue
        tree = _scan_directory(item)
        directories.append(tree)

    return {"success": True, "directories": directories}


@router.get("/api/knowledge/scan")
async def scan_knowledge_dirs(request: Request, current_user: dict = Depends(get_current_admin)):
    """Scan configured sources without blocking the service event loop."""
    try:
        executor = get_runtime_container(request.app).knowledge_scan_executor
        if executor is None:
            raise RuntimeError("Knowledge scan executor is not initialized")
        return await executor.run(_scan_knowledge_dirs)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("知识库目录扫描失败: %s", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail=f"知识库扫描未完成（{type(exc).__name__}），请检查来源目录及服务负载后重试",
        ) from exc


@router.post("/api/knowledge/scan/import")
async def import_scanned_directory(
    directory_name: str, kb_id: int = None, current_user: dict = Depends(get_current_admin),
    text_encoding: KnowledgeTextEncoding = "unicode",
):
    """将扫描到的目录导入到知识库

    读取 knowledge_bases/<directory_name> 下的所有文件，
    自动按子目录创建文件夹，按文件创建文档。

    如果 kb_id 为空，则自动创建新知识库。
    """
    target_dir = _knowledge_source_path(KNOWLEDGE_BASES_DIR / directory_name)
    if not target_dir.exists() or not target_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"目录不存在: {directory_name}")

    created_folders = {}
    created_docs = 0
    errors = []
    unconfirmed = False
    kb = None
    try:
        # 获取或创建知识库
        if kb_id:
            kb = await run_db(db.get_knowledge_base, kb_id)
            if not kb:
                raise HTTPException(status_code=404, detail="知识库不存在")
        else:
            kb = await run_db(db.create_knowledge_base, directory_name)
            if kb is None:
                # 已存在同名知识库，查找它
                all_bases = await run_db(db.get_knowledge_bases)
                kb = next((b for b in all_bases if b["name"] == directory_name), None)
                if not kb:
                    raise HTTPException(status_code=500, detail="无法创建或找到知识库")

        kb_id = kb["id"]

        # 遍历子目录
        for sub_dir in sorted(target_dir.iterdir()):
            if sub_dir.name.startswith(".") or not sub_dir.is_dir():
                continue

            try:
                _knowledge_source_path(sub_dir)
            except HTTPException as exc:
                errors.append(_knowledge_import_error(sub_dir.name, exc))
                continue
            folder_name = sub_dir.name

            # 创建文件夹
            folder = await run_db(db.create_knowledge_folder, kb_id, folder_name)
            if folder is None:
                folders = await run_db(db.get_knowledge_folders, kb_id)
                folder = next((f for f in folders if f["name"] == folder_name), None)
            if folder:
                created_folders[folder_name] = folder["id"]
            else:
                errors.append(f"无法创建文件夹: {folder_name}")
                continue

            folder_id = created_folders[folder_name]

            # 遍历文件夹中的文件
            for file_path in sorted(sub_dir.iterdir()):
                if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue

                try:
                    _knowledge_source_path(file_path)
                    file_bytes = await asyncio.to_thread(file_path.read_bytes)
                    content = decode_knowledge_text(file_bytes, text_encoding)

                    if not content.strip():
                        continue

                    doc_title = file_path.stem
                    file_size = len(file_bytes)

                    # 创建文档
                    document_data = {
                        "title": doc_title,
                        "content": content,
                        "category": folder_name,
                        "knowledge_base_id": kb_id,
                        "folder_id": folder_id,
                        "sourceType": "file",
                        "fileType": file_path.suffix.lstrip("."),
                        "fileSize": file_size,
                    }

                    # 原子发布文档、完整分块和索引修订
                    from knowledge.text_splitter import simple_text_split

                    chunks = simple_text_split(content)
                    await run_db(db.save_knowledge_document, document_data, chunks=chunks)
                    created_docs += 1

                    # 数据与 dirty 修订已原子提交；此处仅清除当前进程的加速状态
                    await run_db(_invalidate_local_knowledge_index)


                except BlockingWorkTimeout:
                    raise
                except Exception as exc:
                    errors.append(_knowledge_import_error(file_path.name, exc))

        # 也处理根目录下的文件（不属于任何子文件夹）
        for file_path in sorted(target_dir.iterdir()):
            if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            try:
                _knowledge_source_path(file_path)
                file_bytes = await asyncio.to_thread(file_path.read_bytes)
                content = decode_knowledge_text(file_bytes, text_encoding)
                if not content.strip():
                    continue

                doc_title = file_path.stem
                document_data = {
                    "title": doc_title,
                    "content": content,
                    "category": "未分类",
                    "knowledge_base_id": kb_id,
                    "folder_id": None,
                    "sourceType": "file",
                    "fileType": file_path.suffix.lstrip("."),
                    "fileSize": len(file_bytes),
                }

                from knowledge.text_splitter import simple_text_split

                chunks = simple_text_split(content)
                await run_db(db.save_knowledge_document, document_data, chunks=chunks)
                created_docs += 1

                # 数据与 dirty 修订已原子提交；此处仅清除当前进程的加速状态
                await run_db(_invalidate_local_knowledge_index)
            except BlockingWorkTimeout:
                raise
            except Exception as exc:
                errors.append(_knowledge_import_error(file_path.name, exc))

    except BlockingWorkTimeout:
        unconfirmed = True
        errors.append("导入操作超时，结果未确认，请刷新核对后再决定是否重试。")
    except HTTPException:
        raise
    except Exception as exc:
        errors.append(_knowledge_import_error(directory_name, exc))
    result = _knowledge_import_result(created_folders, created_docs, errors, unconfirmed=unconfirmed)
    if kb is not None:
        result["knowledgeBase"] = kb
    return result


# ============================================
# 文档管理
# ============================================


@router.get("/api/knowledge/documents")
async def get_knowledge_documents(
    limit: int = 100,
    offset: int = 0,
    category: str = None,
    knowledge_base_id: int = None,
    folder_id: int = None,
    current_user: dict = Depends(get_current_admin),
):
    """获取知识库文档列表，支持按分类/知识库/文件夹筛选"""
    documents = await run_db(
        db.get_knowledge_documents,
        limit=limit,
        offset=offset,
        category=category,
        knowledge_base_id=knowledge_base_id,
        folder_id=folder_id,
    )
    stats = await run_db(db.get_knowledge_stats)
    return {"success": True, "documents": documents, "stats": stats}


@router.get("/api/knowledge/documents/{doc_id}")
async def get_knowledge_document(doc_id: int, current_user: dict = Depends(get_current_admin)):
    """获取单个知识库文档"""
    document = await run_db(db.get_knowledge_document, doc_id)
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在")
    chunks = await run_db(db.get_knowledge_chunks, doc_id)
    return {"success": True, "document": document, "chunks": chunks}


@router.post("/api/knowledge/documents")
async def create_knowledge_document(request: KnowledgeDocumentCreate, current_user: dict = Depends(get_current_admin)):
    """创建知识库文档"""
    try:
        # 输入验证

        document_data = request.model_dump()
        is_valid, errors = InputValidator.validate(document_data, KNOWLEDGE_DOCUMENT_SCHEMA)
        if not is_valid:
            raise HTTPException(status_code=422, detail={"message": "输入验证失败", "errors": errors})

        # 使用文件夹元数据确定文档分类
        if document_data["folder_id"]:
            folder = await run_db(db.get_knowledge_folder, document_data["folder_id"])
            if folder:
                document_data["category"] = folder["name"]

        # 原子发布文档、完整分块和索引修订
        from knowledge.text_splitter import simple_text_split

        chunks = simple_text_split(document_data["content"])
        document = await run_db(db.save_knowledge_document, document_data, chunks=chunks)
        chunk_count = len(chunks)

        # 数据与 dirty 修订已原子提交；此处仅清除当前进程的加速状态
        await run_db(_invalidate_local_knowledge_index)
        logger.info("创建知识库文档: %s, 分块数: %d", document["title"], chunk_count)
        return {"success": True, "message": "文档创建成功", "document": document, "chunkCount": chunk_count}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("创建知识库文档失败: %s", type(e).__name__)
        raise HTTPException(status_code=500, detail="知识文档创建未完成，请检查数据库及输入内容后重试") from e


@router.put("/api/knowledge/documents/{doc_id}")
async def update_knowledge_document(
    doc_id: int, request: KnowledgeDocumentUpdate, current_user: dict = Depends(get_current_admin)
):
    """更新知识库文档"""
    try:
        existing_doc = await run_db(db.get_knowledge_document, doc_id)
        if not existing_doc:
            raise HTTPException(status_code=404, detail="文档不存在")

        # 更新文档
        update_data = request.model_dump(exclude_none=True)

        from knowledge.text_splitter import simple_text_split

        replacement_chunks = simple_text_split(update_data["content"]) if "content" in update_data else None
        updated_doc = await run_db(db.save_knowledge_document, update_data, doc_id=doc_id, chunks=replacement_chunks)
        if updated_doc is None:
            raise HTTPException(status_code=404, detail="文档不存在")

        await run_db(_invalidate_local_knowledge_index)

        logger.info("更新知识库文档: %s", doc_id)
        return {"success": True, "message": "文档更新成功", "document": updated_doc}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("更新知识库文档失败: %s", type(e).__name__)
        raise HTTPException(status_code=500, detail="知识文档更新未完成，请检查数据库及输入内容后重试") from e


@router.delete("/api/knowledge/documents/{doc_id}")
async def delete_knowledge_document(doc_id: int, current_user: dict = Depends(get_current_admin)):
    """删除知识库文档

    s2 fix: 删除文档及关联向量索引，限定 admin。
    """
    try:
        existing_doc = await run_db(db.get_knowledge_document, doc_id)
        if not existing_doc:
            raise HTTPException(status_code=404, detail="文档不存在")

        if not await run_db(db.delete_knowledge_document, doc_id):
            raise HTTPException(status_code=404, detail="文档不存在")
        await run_db(_invalidate_local_knowledge_index)

        logger.info("删除知识库文档: %s", doc_id)
        return {"success": True, "message": "文档删除成功"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("删除知识库文档失败: %s", type(e).__name__)
        raise HTTPException(status_code=500, detail="知识文档删除未完成，请检查数据库及输入内容后重试") from e


# ============================================
# 搜索
# ============================================

_vector_index_built = False
_vector_index_revision: int | None = None
_vector_index_lock = threading.Lock()
# 只保护本进程完成标志与缓存失效；持久化版本由数据库事务和 CAS 管理。
# 嵌套锁顺序固定为 _vector_index_lock → _revision_lock。
_revision_lock = threading.Lock()

# 重建状态键，存储在 config 表中。
# 状态格式：
#   "building:{expected}"                          - 重建进行中（中断后会触发重建）
#   "complete:{count}:{fingerprint}:{revision}"    - 重建完成，count + 指纹 + revision 必须同时匹配
#   "dirty"                                        - 文档 CUD 后标记，下次搜索必须重建
# fingerprint 基于 chunk 内容+位置哈希，覆盖"数量不变但内容已更新"场景。
# revision 是单调递增计数器，每次文档 CUD 自增；重建记录 start_revision，
# 完成时仅当 current_revision == start_revision 才写入 complete，避免
# 重建期间并发 CRUD 被旧重建任务覆盖。
_VECTOR_REBUILD_STATUS_KEY = "vector_index_rebuild_status"
_VECTOR_REBUILD_REVISION_KEY = "vector_index_rebuild_revision"
_EMPTY_FINGERPRINT = "empty"


def _get_rebuild_revision() -> int:
    """读取当前重建修订号（单调递增）。默认 0。"""
    from db.knowledge_index_state import parse_revision

    return parse_revision(db.get_config_value(_VECTOR_REBUILD_REVISION_KEY, "0"))


def _chunk_fingerprint_record(row: dict) -> bytes:
    parts = (
        str(row.get("documentId")),
        str(row.get("chunkIndex")),
        row.get("content", "") or "",
        row.get("doc_title") or "",
        row.get("doc_category") or "",
        str(row.get("doc_kb_id") or ""),
    )
    return ("\x1f".join(parts) + "\x1e").encode("utf-8")


def _compute_chunk_fingerprint() -> str:
    """计算 chunk 内容指纹，用于检测内容变更（即使 chunk 数量不变）。

    指纹覆盖每个 chunk 的 (documentId, chunkIndex, content, doc_title,
    doc_category, doc_kb_id)。使用稳定的串接顺序确保跨后端一致。

    跳过孤儿 chunk（doc_title IS NULL），与 _ensure_vector_index 重建遍历、
    _get_expected_chunk_count 的 INNER JOIN 数据集完全一致，避免"指纹含孤儿
    但重建跳过孤儿"导致重建后指纹永远不匹配、反复重建。

    为避免大库一次性加载导致 OOM，使用流式哈希：每读一批更新一次 md5。
    """
    import hashlib

    h = hashlib.md5()
    for row in db.iter_chunks_with_document(batch_size=500):
        if row.get("doc_title") is None:
            continue  # 孤儿 chunk，与重建遍历保持一致
        h.update(_chunk_fingerprint_record(row))
    return h.hexdigest()[:16]


def _get_expected_chunk_count() -> int:
    """获取数据库中"有效" chunk 的预期总数（与重建遍历范围一致）。

    使用 INNER JOIN：与 _ensure_vector_index 重建时跳过孤儿 chunk 的逻辑一致，
    避免"预期数量含孤儿但遍历跳过孤儿"导致永久不匹配、每次搜索都重建。
    """
    rows = db.execute_sql(
        'SELECT COUNT(*) AS cnt FROM knowledge_chunks c INNER JOIN knowledge_documents d ON c."documentId" = d.id',
        {},
    )
    return rows[0]["cnt"]


def _read_rebuild_status() -> tuple[str, int, str, int]:
    """读取重建状态，返回 (status, count, fingerprint, revision)。

    status 为 'building'/'complete'/'dirty'/''。
    count 为整数，fingerprint 为字符串，revision 为整数（仅 complete 时有意义）。
    旧格式 "complete:{count}" / "complete:{count}:{fp}" 视为 revision 缺失（-1），
    会触发重建。
    """
    raw = db.get_config_value(_VECTOR_REBUILD_STATUS_KEY, "")
    if not raw:
        return ("", 0, "", -1)
    # dirty 单独处理
    if raw == "dirty":
        return ("dirty", 0, "", -1)
    parts = raw.split(":")
    if len(parts) < 2:
        return ("", 0, "", -1)
    status = parts[0]
    try:
        count = int(parts[1])
    except (ValueError, IndexError):
        return ("", 0, "", -1)
    fingerprint = parts[2] if len(parts) >= 3 else ""
    try:
        revision = int(parts[3]) if len(parts) >= 4 else -1
    except (ValueError, IndexError):
        revision = -1
    return (status, count, fingerprint, revision)


def _invalidate_local_knowledge_index() -> None:
    """Reset local acceleration only after the durable document transaction."""
    global _vector_index_built, _vector_index_revision
    with _revision_lock:
        _vector_index_built = False
        _vector_index_revision = None


def _vector_chunk_document(row: dict, kb_name_map: dict) -> dict:
    """Use the same authoritative fields for build and loaded-index validation."""
    folder_name = row.get("doc_category") or ""
    kb_id = row.get("doc_kb_id")
    kb_name = kb_name_map.get(kb_id, "") if kb_id else ""
    path_prefix = f"[{kb_name}/{folder_name}]" if kb_name else f"[{folder_name}]"
    return {
        "id": f"doc_{row['documentId']}_chunk_{row['chunkIndex']}",
        "chunk_index": row["chunkIndex"],
        "title": row["doc_title"],
        "content": f"{path_prefix} {row['doc_title']}: {row['content']}",
        "document_id": row["documentId"],
        "category": folder_name,
        "knowledge_base_id": kb_id,
    }


def _loaded_metadata_matches_database(vector_db) -> bool:
    # A global complete marker may belong to another worker/index directory.
    # Check this process's loaded evidence rather than authorizing it by count.
    with vector_db._lock:
        if not getattr(vector_db, "snapshot_validated", False):
            return False
        actual = {record.get("id"): record for record in vector_db.metadata}
        if len(actual) != len(vector_db.metadata):
            return False
        kb_names = {kb["id"]: kb["name"] for kb in db.get_knowledge_bases()}
        for row in db.iter_chunks_with_document(batch_size=500):
            if row.get("doc_title") is None:
                continue
            expected = _vector_chunk_document(row, kb_names)
            loaded = actual.pop(expected["id"], None)
            if loaded is None or any(loaded.get(key) != value for key, value in expected.items()):
                return False
        return not actual


def _ensure_vector_index():
    """延迟重建向量索引：首次搜索时从数据库加载chunks并构建Faiss索引。
    避免在启动时阻塞服务（尤其是多worker场景下GPU显存竞争）。

    C7 fix: 用 threading.Lock 保护 check-build-set 序列，防止多个并发搜索请求
    同时观察到 _vector_index_built == False 并重复构建索引。

    可靠性 fix: 重建状态 = building/complete/dirty + 数量 + 内容指纹 + revision CAS。
    - building: 上次重建中断，下次必须重建
    - dirty: 文档 CUD 后标记，下次必须重建（即使数量不变）
    - complete:{count}:{fingerprint}:{revision}: 仅当 count、fingerprint、revision
      都匹配，且 metadata 数量、FAISS ntotal、BM25 corpus 数量一致才跳过
    - expected_count == 0 时仍调用 clear_all() 持久化空索引，防止旧磁盘文件残留
    - revision CAS: 重建记录 start_revision，完成时仅当 current_revision ==
      start_revision 才写入 complete，避免重建期间并发 CRUD 被旧重建任务覆盖
    - 落盘失败（clear_all/add_documents 抛异常）不会标记 complete，状态保持 building
    """
    global _vector_index_built, _vector_index_revision
    # The boolean belongs to this process; the revision belongs to the DB.
    # A failed authority read must not authorize old cached evidence.
    try:
        if _vector_index_built and _vector_index_revision == _get_rebuild_revision():
            return True
    except Exception as exc:
        logger.warning("读取知识索引修订号失败: %s", type(exc).__name__)
        raise

    with _vector_index_lock:
        try:
            if _vector_index_built and _vector_index_revision == _get_rebuild_revision():
                return True
            _vector_index_built = False
            _vector_index_revision = None
            from knowledge.vector_db import get_vector_db

            vector_db = get_vector_db()
            stats = vector_db.get_stats()
            expected_count = _get_expected_chunk_count()
            status, status_count, status_fp, status_rev = _read_rebuild_status()
            current_revision = _get_rebuild_revision()

            # expected_count == 0：必须清空并持久化空索引，防止旧磁盘文件残留
            if expected_count == 0:
                needs_clear = (
                    not getattr(vector_db, "snapshot_validated", False)
                    or stats["total_documents"] != 0
                    or stats["index_size"] != 0
                    or stats["bm25_corpus_size"] != 0
                    or status != "complete"
                    or status_count != 0
                    or status_rev != current_revision
                )
                if needs_clear:
                    logger.info("数据库无有效 chunk，清空向量索引并持久化空索引")
                    vector_db.clear_all()  # 失败时抛异常，不会写入 complete
                # commit 临界区：revision 校验 + 写 complete + 设 _vector_index_built
                # 必须原子，避免并发 CRUD 在窗口内被覆盖
                with _revision_lock:
                    if _get_rebuild_revision() != current_revision:
                        logger.warning("空库清理期间检测到并发 CRUD，不标记 complete，等待下次重建")
                        return False
                    if not db.commit_knowledge_index_revision(current_revision, 0, _EMPTY_FINGERPRINT):
                        return False
                    logger.info("向量索引已清空并标记 complete:0:empty:%s", current_revision)
                    _vector_index_revision = current_revision
                    _vector_index_built = True
                return True

            # 跳过重建的多重校验：status + count + metadata + FAISS ntotal +
            # BM25 corpus + fingerprint + revision 必须全部一致。
            # 指纹计算放在数量校验通过后，避免每次搜索都全表扫描。
            if (
                status == "complete"
                and status_count == expected_count
                and status_rev == current_revision
                and stats["total_documents"] == expected_count
                and stats["index_size"] == expected_count
                and stats["bm25_corpus_size"] == expected_count
            ):
                current_fp = _compute_chunk_fingerprint()
                if status_fp and status_fp == current_fp and _loaded_metadata_matches_database(vector_db):
                    # commit 临界区：指纹计算期间可能发生 CRUD（已自增 revision），
                    # 必须在锁内重新校验 revision 才能设置 _vector_index_built
                    with _revision_lock:
                        if _get_rebuild_revision() != current_revision:
                            logger.info("跳过检查期间检测到并发 CRUD，放弃跳过，进入重建")
                            # 落入下方重建分支：重新读取最新 revision
                        else:
                            if not db.commit_knowledge_index_revision(current_revision, expected_count, current_fp):
                                return False
                            logger.info(
                                "向量索引已完整: %d 个文档（complete，数量+指纹+revision 匹配），跳过重建",
                                stats["total_documents"],
                            )
                            vector_db.clear_cache()
                            _vector_index_revision = current_revision
                            _vector_index_built = True
                            return True
                else:
                    logger.info(
                        "向量索引指纹不匹配: 状态=%s, 实际=%s，需要重建",
                        status_fp or "(空)",
                        current_fp,
                    )
            else:
                logger.info(
                    "向量索引需要重建: 状态=%s:%d:%s:%d, 实际(meta/faiss/bm25)=%d/%d/%d, 预期=%d, revision=%d/%d",
                    status or "none",
                    status_count,
                    status_fp or "(空)",
                    status_rev,
                    stats["total_documents"],
                    stats["index_size"],
                    stats["bm25_corpus_size"],
                    expected_count,
                    status_rev,
                    current_revision,
                )

            # 需要重建：重新读取最新 revision 作为 start_revision。
            # 顶部读取的 current_revision 可能已过期（跳过分支检测到并发 CRUD
            # 后落入此路径，或指纹计算期间发生 CRUD）。
            start_revision = _get_rebuild_revision()
            db.set_config_value(_VECTOR_REBUILD_STATUS_KEY, f"building:{expected_count}")
            vector_db.clear_all()  # 失败时抛异常，状态保持 building

            # 使用 JOIN 分批读取 chunk + document，避免 N+1 查询。
            # 预加载知识库名称映射（知识库数量通常很少）。
            kb_name_map: dict = {}
            for kb in db.get_knowledge_bases():
                kb_name_map[kb["id"]] = kb["name"]

            batch_vector_docs = []
            vector_batch_size = 200
            total_chunks_indexed = 0
            # 同时累积指纹，确保索引内容与指纹一致
            import hashlib

            fp_hash = hashlib.md5()

            for row in db.iter_chunks_with_document(batch_size=500):
                doc_title = row.get("doc_title")
                if doc_title is None:
                    continue  # 孤儿 chunk（文档已删除），跳过

                batch_vector_docs.append(_vector_chunk_document(row, kb_name_map))

                # 同步累积指纹（与 _compute_chunk_fingerprint 一致，均跳过孤儿）
                fp_hash.update(_chunk_fingerprint_record(row))

                # 攒够一批立即写入，释放内存
                if len(batch_vector_docs) >= vector_batch_size:
                    vector_db.add_documents(batch_vector_docs)  # 落盘失败时抛异常
                    total_chunks_indexed += len(batch_vector_docs)
                    batch_vector_docs = []

            # 写入最后一批
            if batch_vector_docs:
                vector_db.add_documents(batch_vector_docs)
                total_chunks_indexed += len(batch_vector_docs)

            # 完整性校验：写入数量必须 == 预期数量（孤儿 chunk 已在 INNER JOIN 排除）
            if total_chunks_indexed != expected_count:
                logger.error(
                    "向量索引重建数量不匹配: 已写入 %d, 预期 %d，保持 building 状态等待下次重建",
                    total_chunks_indexed,
                    expected_count,
                )
                return False

            # 落盘保证：当 save_on_every_n_adds > 0 且未达阈值时，内存索引完整
            # 但磁盘仍为空。flush() 强制持久化，异常向上传播（不标记 complete）。
            vector_db.flush()

            # commit 临界区：revision 校验 + 写 complete + 设 _vector_index_built
            # 必须在同一 _revision_lock 内原子完成。否则 CRUD 可能在
            # final_revision 读取后、complete 写入前发生，旧重建会覆盖 dirty。
            final_fp = fp_hash.hexdigest()[:16]
            with _revision_lock:
                final_revision = _get_rebuild_revision()
                if final_revision != start_revision:
                    logger.warning(
                        "重建期间检测到并发 CRUD: start_revision=%d, current=%d，"
                        "不标记 complete，保持 building 等待下次重建",
                        start_revision,
                        final_revision,
                    )
                    return False
                if not db.commit_knowledge_index_revision(start_revision, total_chunks_indexed, final_fp):
                    return False
                logger.info(
                    "向量索引重建完成: %s 个 chunks（数量+指纹+revision CAS 校验通过，revision=%s）",
                    total_chunks_indexed,
                    start_revision,
                )
                _vector_index_revision = start_revision
                _vector_index_built = True
            return True
        except Exception as e:
            logger.warning("向量索引重建失败: %s", type(e).__name__)
            raise


@router.post("/api/knowledge/search")
async def search_knowledge(
    request: KnowledgeSearchRequest,
    current_user: dict = Depends(get_current_admin),
):
    """搜索知识库 - 使用 RAGHelper 两阶段检索（向量+BM25混合 → Cross-Encoder精排）

    支持通过 knowledgeBaseName 按指定知识库过滤检索结果，
    用于意图分类器路由后的精准检索。
    """
    try:
        query = request.query
        top_k = request.topK

        # 构造检索过滤器：如果指定了知识库名称，则按 knowledge_base_id 过滤
        filters = None
        if request.knowledgeBaseName:
            # 查询知识库名称对应的ID
            all_bases = await run_db(db.get_knowledge_bases)
            matched = [b for b in all_bases if b["name"] == request.knowledgeBaseName]
            if matched:
                filters = {"knowledge_base_id": matched[0]["id"]}
                logger.info("搜索过滤: 知识库「%s」(id=%s)", request.knowledgeBaseName, matched[0]["id"])
            else:
                # fail-closed: 用户明确指定的知识库不存在时不应退化为全库搜索，
                # 否则会返回其他知识库的内容。返回空结果。
                logger.warning("未找到知识库「%s」，返回空结果（不退化为全库搜索）", request.knowledgeBaseName)
                return _knowledge_search_response(query, [], "empty")

        if not await asyncio.to_thread(_ensure_vector_index):
            raise RuntimeError("Knowledge index rebuild did not complete")

        from knowledge.rag_helper import get_rag_helper

        results = await asyncio.to_thread(
            get_rag_helper().retrieve_context, query, top_k=top_k,
            enable_rerank=True, filters=filters, use_cache=False,
        )
        formatted = [
            {
                "documentId": row.get("id"),
                "documentTitle": row.get("title", ""),
                "chunkIndex": row.get("chunk_index", 0),
                "content": row.get("content", ""),
                "score": row.get("normalized_score", row.get("score", 0)),
                "searchType": "rag_pipeline",
            }
            for row in results
        ]
        return _knowledge_search_response(query, formatted, "evidence" if formatted else "empty")
    except Exception as exc:
        logger.error("知识库搜索失败: %s", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail="知识库检索未完成，请检查索引、数据库及检索模型后重试",
        ) from exc


@router.get("/api/knowledge/stats")
async def get_knowledge_stats(current_user: dict = Depends(get_current_admin)):
    """获取知识库统计数据"""
    stats = await run_db(db.get_knowledge_stats)
    return {"success": True, "stats": stats}


@router.get("/api/vector/stats")
async def get_vector_stats(current_user: dict = Depends(get_current_admin)):
    """获取向量数据库统计数据"""
    try:
        from knowledge.vector_db import get_vector_db

        vector_db = get_vector_db()
        stats = await asyncio.to_thread(vector_db.get_stats)
        return {"success": True, "stats": stats}
    except Exception as exc:
        logger.error("获取向量数据库统计失败: %s", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail="向量数据库统计未完成，请检查索引及向量数据库依赖后重试",
        ) from exc


# ============================================
# 意图分类器训练
# ============================================


@router.post("/api/knowledge/train-intent/generate")
async def generate_intent_samples(
    request: IntentSampleGenerateRequest | None = None,
    current_user: dict = Depends(get_current_admin),
):
    """生成训练样本（LLM基于知识库文档生成，不训练）"""
    from knowledge.intent_trainer import (
        generate_samples,
        get_generation_status,
        get_training_status,
    )

    params = request or IntentSampleGenerateRequest()
    async with _get_intent_task_lock():
        if _intent_task is not None and not _intent_task.done():
            raise HTTPException(status_code=409, detail="已有意图任务正在运行")
        if get_generation_status()["running"] or get_training_status()["running"]:
            raise HTTPException(status_code=409, detail="已有意图任务正在运行")
        _schedule_intent_task(
            generate_samples(
                params.kb_ids,
                params.samples_per_kb,
                params.negative_count,
                params.lora_name,
            ),
            name="intent-sample-generation",
        )
    return {"success": True, "message": "样本生成已启动"}


@router.get("/api/knowledge/train-intent/generate/status")
async def get_generation_status(current_user: dict = Depends(get_current_admin)):
    """查询样本生成进度"""
    try:
        from knowledge.intent_trainer import get_generation_status as get_status

        return {"success": True, "status": get_status()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def _intent_samples_error(exc: Exception) -> HTTPException:
    logger.error("意图样本读写失败: %s", type(exc).__name__)
    return HTTPException(
        status_code=503,
        detail=f"意图样本读写未完成（{type(exc).__name__}），请检查样本文件、存储空间和权限后重试",
    )


@router.get("/api/knowledge/train-intent/samples")
async def get_intent_samples(current_user: dict = Depends(get_current_admin)):
    """获取当前所有训练样本"""
    try:
        from knowledge.intent_trainer import get_samples

        return {"success": True, **get_samples()}
    except Exception as exc:
        raise _intent_samples_error(exc) from exc


@router.put("/api/knowledge/train-intent/samples")
async def update_intent_sample(request: IntentSampleUpdateRequest, current_user: dict = Depends(get_current_admin)):
    """编辑单条样本"""
    try:
        from knowledge.intent_trainer import update_sample

        result = update_sample(request.label, request.index, request.text)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise _intent_samples_error(exc) from exc


@router.delete("/api/knowledge/train-intent/samples")
async def delete_intent_sample(label: str, index: int, current_user: dict = Depends(get_current_admin)):
    """删除单条样本"""
    try:
        from knowledge.intent_trainer import delete_sample

        result = delete_sample(label, index)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise _intent_samples_error(exc) from exc


@router.patch("/api/knowledge/train-intent/samples")
async def batch_save_intent_samples(request: IntentSamplesSaveRequest, current_user: dict = Depends(get_current_admin)):
    """批量保存样本（覆盖写入）"""
    try:
        from knowledge.intent_trainer import save_samples

        return save_samples(request.samples)
    except Exception as exc:
        raise _intent_samples_error(exc) from exc


@router.post("/api/knowledge/train-intent/samples")
async def add_intent_sample(request: IntentSampleAddRequest, current_user: dict = Depends(get_current_admin)):
    """添加单条样本"""
    try:
        from knowledge.intent_trainer import add_sample

        return add_sample(request.label, request.text)
    except Exception as exc:
        raise _intent_samples_error(exc) from exc


@router.post("/api/knowledge/train-intent")
async def train_intent_classifier(
    request: IntentTrainRequest | None = None,
    current_user: dict = Depends(get_current_admin),
):
    """使用已审查的样本训练多分类模型。"""
    from knowledge.intent_trainer import (
        get_generation_status,
        get_training_status,
    )
    from knowledge.intent_trainer import (
        train_intent_classifier as do_train,
    )

    params = request or IntentTrainRequest()
    async with _get_intent_task_lock():
        if _intent_task is not None and not _intent_task.done():
            raise HTTPException(status_code=409, detail="已有意图任务正在运行")
        if get_generation_status()["running"] or get_training_status()["running"]:
            raise HTTPException(status_code=409, detail="已有意图任务正在运行")
        _schedule_intent_task(
            do_train(kb_ids=params.kb_ids),
            name="intent-classifier-training",
        )
    return {"success": True, "message": "训练已启动"}


@router.get("/api/knowledge/train-intent/status")
async def get_intent_training_status(current_user: dict = Depends(get_current_admin)):
    """查询训练进度"""
    try:
        from knowledge.intent_trainer import get_training_status

        return {"success": True, "status": get_training_status()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/api/knowledge/train-intent/cancel")
async def cancel_intent_training(current_user: dict = Depends(get_current_admin)):
    """取消训练/生成"""
    try:
        from knowledge.intent_trainer import cancel_training

        result = cancel_training()
        return {"success": result, "message": "已发送取消请求" if result else "没有正在进行的任务"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/api/knowledge/train-intent/model")
async def get_intent_model_info(current_user: dict = Depends(get_current_admin)):
    """获取当前模型信息"""
    try:
        from knowledge.intent_trainer import get_model_info

        return {"success": True, "model": get_model_info()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/api/knowledge/train-intent/active-kbs")
async def get_active_knowledge_bases(current_user: dict = Depends(get_current_admin)):
    """获取参与检索的知识库"""
    try:
        from knowledge.intent_trainer import get_active_knowledge_bases as get_kbs

        return {"success": True, "active_kbs": get_kbs()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/api/knowledge/train-intent/active-kbs")
async def set_active_knowledge_bases(request: dict, current_user: dict = Depends(get_current_admin)):
    """设置参与检索的知识库"""
    try:
        from knowledge.intent_trainer import set_active_knowledge_bases as set_kbs

        result = set_kbs(request.get("kb_ids", []))
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return {"success": True, "active_kbs": result.get("active_kbs", [])}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
