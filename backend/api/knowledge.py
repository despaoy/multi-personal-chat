"""知识库API - 知识库/文件夹/文档管理 + ZIP上传 + 文件夹扫描 + 搜索"""
import asyncio
import logging
import threading
import io
import os
import zipfile
import re
from pathlib import Path, PurePosixPath

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from app.dependencies import get_current_user, get_current_admin

from db.adapter import db
from db.schemas import (
    KnowledgeBaseCreate, KnowledgeBaseUpdate,
    KnowledgeFolderCreate,
    KnowledgeDocumentCreate, KnowledgeDocumentUpdate,
    KnowledgeSearchRequest
)
from app.config import INPUT_VALIDATOR_AVAILABLE, KNOWLEDGE_DOCUMENT_SCHEMA, VECTOR_DB_AVAILABLE

logger = logging.getLogger(__name__)
router = APIRouter()

_ZIP_MAX_BYTES = int(os.getenv("KNOWLEDGE_ZIP_MAX_BYTES", str(100 * 1024 * 1024)))
_ZIP_MAX_FILES = int(os.getenv("KNOWLEDGE_ZIP_MAX_FILES", "1000"))
_ZIP_MAX_ENTRY_BYTES = int(os.getenv("KNOWLEDGE_ZIP_MAX_ENTRY_BYTES", str(10 * 1024 * 1024)))
_ZIP_MAX_UNCOMPRESSED_BYTES = int(
    os.getenv("KNOWLEDGE_ZIP_MAX_UNCOMPRESSED_BYTES", str(200 * 1024 * 1024))
)
_ZIP_MAX_COMPRESSION_RATIO = float(os.getenv("KNOWLEDGE_ZIP_MAX_COMPRESSION_RATIO", "200"))


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
            entry.compress_size == 0
            or entry.file_size / entry.compress_size > _ZIP_MAX_COMPRESSION_RATIO
        ):
            raise HTTPException(status_code=413, detail="ZIP compression ratio is suspicious")

    return entries

# ============================================
# 知识库管理
# ============================================

@router.get("/api/knowledge/bases")
async def list_knowledge_bases(current_user: dict = Depends(get_current_user)):
    """获取所有知识库"""
    bases = db.get_knowledge_bases()
    return {"success": True, "bases": bases}


@router.post("/api/knowledge/bases")
async def create_knowledge_base(request: KnowledgeBaseCreate, current_user: dict = Depends(get_current_user)):
    """创建知识库"""
    result = db.create_knowledge_base(request.name, request.description)
    if result is None:
        raise HTTPException(status_code=409, detail="知识库名称已存在")
    return {"success": True, "base": result}


@router.put("/api/knowledge/bases/{kb_id}")
async def update_knowledge_base(kb_id: int, request: KnowledgeBaseUpdate, current_user: dict = Depends(get_current_user)):
    """更新知识库"""
    existing = db.get_knowledge_base(kb_id)
    if not existing:
        raise HTTPException(status_code=404, detail="知识库不存在")
    data = {}
    if request.name is not None:
        data["name"] = request.name
    if request.description is not None:
        data["description"] = request.description
    result = db.update_knowledge_base(kb_id, data)
    return {"success": True, "base": result}


@router.delete("/api/knowledge/bases/{kb_id}")
async def delete_knowledge_base(kb_id: int, current_user: dict = Depends(get_current_admin)):
    """删除知识库（级联删除文件夹和文档）

    C-S1 fix: 级联删除不可逆，限定 admin。
    """
    existing = db.get_knowledge_base(kb_id)
    if not existing:
        raise HTTPException(status_code=404, detail="知识库不存在")
    db.delete_knowledge_base(kb_id)
    return {"success": True, "message": "知识库已删除"}


# ============================================
# 文件夹管理
# ============================================

@router.get("/api/knowledge/bases/{kb_id}/folders")
async def list_knowledge_folders(kb_id: int, current_user: dict = Depends(get_current_user)):
    """获取知识库下的文件夹"""
    existing = db.get_knowledge_base(kb_id)
    if not existing:
        raise HTTPException(status_code=404, detail="知识库不存在")
    folders = db.get_knowledge_folders(kb_id)
    return {"success": True, "folders": folders}


@router.post("/api/knowledge/bases/{kb_id}/folders")
async def create_knowledge_folder(kb_id: int, request: KnowledgeFolderCreate, current_user: dict = Depends(get_current_user)):
    """创建文件夹"""
    existing = db.get_knowledge_base(kb_id)
    if not existing:
        raise HTTPException(status_code=404, detail="知识库不存在")
    result = db.create_knowledge_folder(kb_id, request.name, request.description)
    if result is None:
        raise HTTPException(status_code=409, detail="文件夹名称已存在")
    return {"success": True, "folder": result}


@router.delete("/api/knowledge/folders/{folder_id}")
async def delete_knowledge_folder(folder_id: int, current_user: dict = Depends(get_current_admin)):
    """删除文件夹

    s2 fix: 级联删除文件夹下所有文档与向量索引，限定 admin。
    """
    existing = db.get_knowledge_folder(folder_id)
    if not existing:
        raise HTTPException(status_code=404, detail="文件夹不存在")
    db.delete_knowledge_folder(folder_id)
    return {"success": True, "message": "文件夹已删除"}


# ============================================
# ZIP上传
# ============================================

@router.post("/api/knowledge/bases/{kb_id}/upload-zip")
async def upload_zip(kb_id: int, file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    """上传ZIP文件，自动按目录结构创建文件夹和文档

    ZIP结构要求：
    - 顶层目录名作为文件夹名
    - 顶层目录下的.txt文件作为文档
    - 例: 角色/胡桃.txt, 事件/活动剧情.txt
    """
    existing = db.get_knowledge_base(kb_id)
    if not existing:
        raise HTTPException(status_code=404, detail="知识库不存在")

    if not file.filename or not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="请上传ZIP文件")

    try:
        content = await file.read(_ZIP_MAX_BYTES + 1)
        if len(content) > _ZIP_MAX_BYTES:
            raise HTTPException(status_code=413, detail=f"文件大小超过限制 ({_ZIP_MAX_BYTES // 1024 // 1024}MB)")
        zf = zipfile.ZipFile(io.BytesIO(content))
        zip_entries = _validated_zip_entries(zf)
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="无效的ZIP文件") from exc

    kb_name = existing["name"]
    created_folders = {}
    created_docs = 0
    errors = []

    for zip_entry in zip_entries:
        entry = zip_entry.filename
        # 跳过目录条目和隐藏文件
        if entry.endswith('/') or entry.startswith('.') or '__MACOSX' in entry:
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
        if not filename.lower().endswith(('.txt', '.md', '.json')):
            continue

        # 安全检查
        if '..' in entry or entry.startswith('/'):
            continue

        # 获取或创建文件夹
        if folder_name not in created_folders:
            folder = db.create_knowledge_folder(kb_id, folder_name)
            if folder is None:
                # 文件夹已存在，查找它
                folders = db.get_knowledge_folders(kb_id)
                folder = next((f for f in folders if f["name"] == folder_name), None)
            if folder:
                created_folders[folder_name] = folder["id"]
            else:
                errors.append(f"无法创建文件夹: {folder_name}")
                continue

        folder_id = created_folders[folder_name]

        # 读取文件内容
        try:
            file_content = zf.read(zip_entry).decode('utf-8')
        except UnicodeDecodeError:
            try:
                file_content = zf.read(zip_entry).decode('gbk')
            except UnicodeDecodeError:
                errors.append(f"文件编码不支持: {entry}")
                continue

        if not file_content.strip():
            continue

        # 文档标题 = 文件名（去掉扩展名）
        doc_title = re.sub(r'\.(txt|md|json)$', '', filename)

        # 创建文档 - 注入文件夹路径到category
        document_data = {
            "title": doc_title,
            "content": file_content,
            "category": folder_name,
            "knowledge_base_id": kb_id,
            "folder_id": folder_id,
            "sourceType": "file",
            "fileType": filename.rsplit('.', 1)[-1] if '.' in filename else "txt",
            "fileSize": len(file_content.encode('utf-8')),
            "chunkCount": 0
        }
        document = db.add_knowledge_document(document_data)

        # 分块处理
        from knowledge.text_splitter import simple_text_split
        chunks = simple_text_split(file_content)
        chunk_count = 0
        vector_docs = []

        for i, chunk_content in enumerate(chunks):
            chunk = {
                "documentId": document["id"],
                "chunkIndex": i,
                "content": chunk_content,
                "embedding": None
            }
            db.add_knowledge_chunk(chunk)
            chunk_count += 1

            # 注入文件夹路径到检索文本：知识库名/文件夹名/文档名 + 内容
            enriched_content = f"[{kb_name}/{folder_name}] {doc_title}: {chunk_content}"

            vector_docs.append({
                "id": f"doc_{document['id']}_chunk_{i}",
                "chunk_index": i,
                "title": doc_title,
                "content": enriched_content,
                "source_type": "file",
                "document_id": document["id"],
                "category": folder_name,
                "knowledge_base_id": kb_id,
            })

        # 更新文档的chunkCount
        db.update_knowledge_document(document["id"], {"chunkCount": chunk_count})

        # 添加到向量数据库
        if VECTOR_DB_AVAILABLE and vector_docs:
            try:
                from app.config import get_vector_db
                vector_db = get_vector_db()
                await asyncio.to_thread(vector_db.add_documents, vector_docs)
            except Exception as ve:
                logger.error(f"添加到向量数据库失败: {ve}")

        created_docs += 1

    zf.close()

    logger.info(f"ZIP上传完成: 知识库={kb_name}, 文件夹={len(created_folders)}, 文档={created_docs}")
    return {
        "success": True,
        "message": f"成功导入 {created_docs} 个文档到 {len(created_folders)} 个文件夹",
        "createdFolders": list(created_folders.keys()),
        "createdDocs": created_docs,
        "errors": errors
    }


# ============================================
# 文件夹扫描
# ============================================

KNOWLEDGE_BASES_DIR = Path(__file__).parent.parent / "knowledge_bases"
SUPPORTED_EXTENSIONS = {'.txt', '.md', '.json', '.csv', '.html', '.xml'}


def _scan_directory(directory: Path) -> dict:
    """扫描目录结构，返回树形结构"""
    result = {
        "name": directory.name,
        "type": "folder",
        "children": [],
        "fileCount": 0,
        "totalSize": 0
    }
    if not directory.exists():
        return result
    
    for item in sorted(directory.iterdir()):
        if item.name.startswith('.') or item.name == '__pycache__':
            continue
        if item.is_dir():
            sub = _scan_directory(item)
            result["children"].append(sub)
            result["fileCount"] += sub["fileCount"]
            result["totalSize"] += sub["totalSize"]
        elif item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
            file_size = item.stat().st_size
            result["children"].append({
                "name": item.name,
                "type": "file",
                "size": file_size,
                "extension": item.suffix.lower(),
            })
            result["fileCount"] += 1
            result["totalSize"] += file_size
    return result


@router.get("/api/knowledge/scan")
async def scan_knowledge_dirs(current_user: dict = Depends(get_current_user)):
    """扫描 knowledge_bases 目录，返回所有可用的知识库文件夹结构
    
    扫描 backend/knowledge_bases/ 下的所有子目录，
    每个顶层子目录被视为一个知识库候选项。
    """
    if not KNOWLEDGE_BASES_DIR.exists():
        return {"success": True, "directories": [], "message": "知识库目录不存在"}
    
    directories = []
    for item in sorted(KNOWLEDGE_BASES_DIR.iterdir()):
        if not item.is_dir() or item.name.startswith('.'):
            continue
        tree = _scan_directory(item)
        directories.append(tree)
    
    return {"success": True, "directories": directories}


@router.post("/api/knowledge/scan/import")
async def import_scanned_directory(directory_name: str, kb_id: int = None, current_user: dict = Depends(get_current_user)):
    """将扫描到的目录导入到知识库
    
    读取 knowledge_bases/<directory_name> 下的所有文件，
    自动按子目录创建文件夹，按文件创建文档。
    
    如果 kb_id 为空，则自动创建新知识库。
    """
    target_dir = KNOWLEDGE_BASES_DIR / directory_name
    if not target_dir.exists() or not target_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"目录不存在: {directory_name}")
    
    # 获取或创建知识库
    if kb_id:
        kb = db.get_knowledge_base(kb_id)
        if not kb:
            raise HTTPException(status_code=404, detail="知识库不存在")
    else:
        kb = db.create_knowledge_base(directory_name)
        if kb is None:
            # 已存在同名知识库，查找它
            all_bases = db.get_knowledge_bases()
            kb = next((b for b in all_bases if b["name"] == directory_name), None)
            if not kb:
                raise HTTPException(status_code=500, detail="无法创建或找到知识库")
    
    kb_id = kb["id"]
    kb_name = kb["name"]
    created_folders = {}
    created_docs = 0
    errors = []
    
    # 遍历子目录
    for sub_dir in sorted(target_dir.iterdir()):
        if sub_dir.name.startswith('.') or not sub_dir.is_dir():
            continue
        
        folder_name = sub_dir.name
        
        # 创建文件夹
        folder = db.create_knowledge_folder(kb_id, folder_name)
        if folder is None:
            folders = db.get_knowledge_folders(kb_id)
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
                # 尝试多种编码读取
                content = None
                for encoding in ['utf-8', 'gbk', 'gb2312', 'utf-16']:
                    try:
                        content = file_path.read_text(encoding=encoding)
                        break
                    except (UnicodeDecodeError, UnicodeError):
                        continue
                
                if content is None:
                    errors.append(f"文件编码不支持: {file_path.name}")
                    continue
                
                if not content.strip():
                    continue
                
                doc_title = file_path.stem
                file_size = file_path.stat().st_size
                
                # 创建文档
                document_data = {
                    "title": doc_title,
                    "content": content,
                    "category": folder_name,
                    "knowledge_base_id": kb_id,
                    "folder_id": folder_id,
                    "sourceType": "file",
                    "fileType": file_path.suffix.lstrip('.'),
                    "fileSize": file_size,
                    "chunkCount": 0
                }
                document = db.add_knowledge_document(document_data)
                
                # 分块 + 路径注入
                from knowledge.text_splitter import simple_text_split
                chunks = simple_text_split(content)
                chunk_count = 0
                vector_docs = []
                
                for i, chunk_content in enumerate(chunks):
                    chunk = {
                        "documentId": document["id"],
                        "chunkIndex": i,
                        "content": chunk_content,
                        "embedding": None
                    }
                    db.add_knowledge_chunk(chunk)
                    chunk_count += 1
                    
                    enriched_content = f"[{kb_name}/{folder_name}] {doc_title}: {chunk_content}"
                    vector_docs.append({
                        "id": f"doc_{document['id']}_chunk_{i}",
                        "chunk_index": i,
                        "title": doc_title,
                        "content": enriched_content,
                        "source_type": "file",
                        "document_id": document["id"],
                        "category": folder_name,
                        "knowledge_base_id": kb_id,
                    })
                
                db.update_knowledge_document(document["id"], {"chunkCount": chunk_count})
                
                if VECTOR_DB_AVAILABLE and vector_docs:
                    try:
                        from app.config import get_vector_db
                        vector_db = get_vector_db()
                        await asyncio.to_thread(vector_db.add_documents, vector_docs)
                    except Exception as ve:
                        logger.error(f"添加到向量数据库失败: {ve}")

                created_docs += 1

            except Exception as e:
                errors.append(f"处理文件 {file_path.name} 失败: {str(e)}")
    
    # 也处理根目录下的文件（不属于任何子文件夹）
    for file_path in sorted(target_dir.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        
        try:
            content = None
            for encoding in ['utf-8', 'gbk', 'gb2312', 'utf-16']:
                try:
                    content = file_path.read_text(encoding=encoding)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            if content is None or not content.strip():
                continue
            
            doc_title = file_path.stem
            document_data = {
                "title": doc_title,
                "content": content,
                "category": "未分类",
                "knowledge_base_id": kb_id,
                "folder_id": None,
                "sourceType": "file",
                "fileType": file_path.suffix.lstrip('.'),
                "fileSize": file_path.stat().st_size,
                "chunkCount": 0
            }
            document = db.add_knowledge_document(document_data)
            
            from knowledge.text_splitter import simple_text_split
            chunks = simple_text_split(content)
            chunk_count = 0
            vector_docs = []
            
            for i, chunk_content in enumerate(chunks):
                chunk = {
                    "documentId": document["id"],
                    "chunkIndex": i,
                    "content": chunk_content,
                    "embedding": None
                }
                db.add_knowledge_chunk(chunk)
                chunk_count += 1
                
                enriched_content = f"[{kb_name}] {doc_title}: {chunk_content}"
                vector_docs.append({
                    "id": f"doc_{document['id']}_chunk_{i}",
                    "chunk_index": i,
                    "title": doc_title,
                    "content": enriched_content,
                    "source_type": "file",
                    "document_id": document["id"],
                    "category": "未分类",
                    "knowledge_base_id": kb_id,
                })
            
            db.update_knowledge_document(document["id"], {"chunkCount": chunk_count})
            
            if VECTOR_DB_AVAILABLE and vector_docs:
                try:
                    from app.config import get_vector_db
                    vector_db = get_vector_db()
                    await asyncio.to_thread(vector_db.add_documents, vector_docs)
                except Exception as ve:
                    logger.error(f"添加到向量数据库失败: {ve}")

            created_docs += 1
        except Exception as e:
            errors.append(f"处理根目录文件 {file_path.name} 失败: {str(e)}")
    
    logger.info(f"扫描导入完成: 知识库={kb_name}, 文件夹={len(created_folders)}, 文档={created_docs}")
    return {
        "success": True,
        "message": f"成功导入 {created_docs} 个文档到 {len(created_folders)} 个文件夹",
        "knowledgeBase": kb,
        "createdFolders": list(created_folders.keys()),
        "createdDocs": created_docs,
        "errors": errors
    }


# ============================================
# 文档管理
# ============================================

@router.get("/api/knowledge/documents")
async def get_knowledge_documents(limit: int = 100, offset: int = 0, category: str = None, knowledge_base_id: int = None, folder_id: int = None, current_user: dict = Depends(get_current_user)):
    """获取知识库文档列表，支持按分类/知识库/文件夹筛选"""
    documents = db.get_knowledge_documents(
        limit=limit, offset=offset,
        category=category,
        knowledge_base_id=knowledge_base_id,
        folder_id=folder_id
    )
    stats = db.get_knowledge_stats()
    return {
        "success": True,
        "documents": documents,
        "stats": stats
    }


@router.get("/api/knowledge/documents/{doc_id}")
async def get_knowledge_document(doc_id: int, current_user: dict = Depends(get_current_user)):
    """获取单个知识库文档"""
    document = db.get_knowledge_document(doc_id)
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在")
    chunks = db.get_knowledge_chunks(doc_id)
    return {
        "success": True,
        "document": document,
        "chunks": chunks
    }


@router.post("/api/knowledge/documents")
async def create_knowledge_document(request: KnowledgeDocumentCreate, current_user: dict = Depends(get_current_user)):
    """创建知识库文档"""
    try:
        # 输入验证
        if INPUT_VALIDATOR_AVAILABLE:
            from infra.input_validator import InputValidator
            is_valid, errors = InputValidator.validate(request.model_dump(), KNOWLEDGE_DOCUMENT_SCHEMA)
            if not is_valid:
                raise HTTPException(status_code=422, detail={"message": "输入验证失败", "errors": errors})

        # 获取知识库和文件夹信息（用于路径注入）
        kb_name = ""
        folder_name = request.category
        if request.knowledge_base_id:
            kb = db.get_knowledge_base(request.knowledge_base_id)
            if kb:
                kb_name = kb["name"]
        if request.folder_id:
            folder = db.get_knowledge_folder(request.folder_id)
            if folder:
                folder_name = folder["name"]

        # 创建文档
        document_data = {
            "title": request.title,
            "content": request.content,
            "category": folder_name,
            "knowledge_base_id": request.knowledge_base_id,
            "folder_id": request.folder_id,
            "sourceType": request.sourceType,
            "sourceUrl": request.sourceUrl,
            "fileType": request.fileType,
            "fileSize": request.fileSize,
            "chunkCount": 0
        }
        document = db.add_knowledge_document(document_data)

        # 分块处理 - 注入路径到检索文本
        from knowledge.text_splitter import simple_text_split
        chunks = simple_text_split(request.content)
        chunk_count = 0
        vector_docs = []

        for i, chunk_content in enumerate(chunks):
            chunk = {
                "documentId": document["id"],
                "chunkIndex": i,
                "content": chunk_content,
                "embedding": None
            }
            db.add_knowledge_chunk(chunk)
            chunk_count += 1

            # 注入文件夹路径到检索文本
            path_prefix = f"[{kb_name}/{folder_name}]" if kb_name else f"[{folder_name}]"
            enriched_content = f"{path_prefix} {request.title}: {chunk_content}"

            vector_docs.append({
                "id": f"doc_{document['id']}_chunk_{i}",
                "chunk_index": i,
                "title": request.title,
                "content": enriched_content,
                "source_type": request.sourceType,
                "document_id": document["id"],
                "category": folder_name,
                "knowledge_base_id": request.knowledge_base_id,
            })

        # 更新文档的chunkCount
        db.update_knowledge_document(document["id"], {"chunkCount": chunk_count})

        # 添加到向量数据库
        if VECTOR_DB_AVAILABLE and vector_docs:
            try:
                from app.config import get_vector_db
                vector_db = get_vector_db()
                await asyncio.to_thread(vector_db.add_documents, vector_docs)
                logger.info(f"文档已添加到向量数据库: {document['title']}")
            except Exception as ve:
                logger.error(f"添加到向量数据库失败: {ve}")

        logger.info(f"创建知识库文档: {document['title']}, 分块数: {chunk_count}")
        return {
            "success": True,
            "message": "文档创建成功",
            "document": document,
            "chunkCount": chunk_count
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建知识库文档失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/knowledge/documents/{doc_id}")
async def update_knowledge_document(doc_id: int, request: KnowledgeDocumentUpdate, current_user: dict = Depends(get_current_user)):
    """更新知识库文档"""
    try:
        existing_doc = db.get_knowledge_document(doc_id)
        if not existing_doc:
            raise HTTPException(status_code=404, detail="文档不存在")

        # 更新文档
        update_data = {}
        if request.title is not None:
            update_data["title"] = request.title
        if request.content is not None:
            update_data["content"] = request.content
        if request.category is not None:
            update_data["category"] = request.category
        if request.knowledge_base_id is not None:
            update_data["knowledge_base_id"] = request.knowledge_base_id
        if request.folder_id is not None:
            update_data["folder_id"] = request.folder_id
        if request.sourceType is not None:
            update_data["sourceType"] = request.sourceType
        if request.sourceUrl is not None:
            update_data["sourceUrl"] = request.sourceUrl
        if request.fileType is not None:
            update_data["fileType"] = request.fileType
        if request.fileSize is not None:
            update_data["fileSize"] = request.fileSize

        updated_doc = db.update_knowledge_document(doc_id, update_data)

        # 如果内容更新了，重新分块
        if "content" in update_data:
            db.execute_sql('DELETE FROM knowledge_chunks WHERE documentId = :doc_id', {"doc_id": doc_id})

            # 获取路径信息用于注入
            kb_name = ""
            folder_name = update_data.get("category", existing_doc.get("category", "未分类"))
            kb_id = update_data.get("knowledge_base_id", existing_doc.get("knowledge_base_id"))
            folder_id = update_data.get("folder_id", existing_doc.get("folder_id"))
            if kb_id:
                kb = db.get_knowledge_base(kb_id)
                if kb:
                    kb_name = kb["name"]
            if folder_id:
                folder = db.get_knowledge_folder(folder_id)
                if folder:
                    folder_name = folder["name"]

            from knowledge.text_splitter import simple_text_split
            chunks = simple_text_split(update_data["content"])
            chunk_count = 0
            vector_docs = []

            for i, chunk_content in enumerate(chunks):
                chunk = {
                    "documentId": doc_id,
                    "chunkIndex": i,
                    "content": chunk_content,
                    "embedding": None
                }
                db.add_knowledge_chunk(chunk)
                chunk_count += 1

                path_prefix = f"[{kb_name}/{folder_name}]" if kb_name else f"[{folder_name}]"
                doc_title = update_data.get("title", existing_doc.get("title", ""))
                enriched_content = f"{path_prefix} {doc_title}: {chunk_content}"

                vector_docs.append({
                    "id": f"doc_{doc_id}_chunk_{i}",
                    "chunk_index": i,
                    "title": doc_title,
                    "content": enriched_content,
                    "source_type": update_data.get("sourceType", existing_doc.get("sourceType", "text")),
                    "document_id": doc_id,
                    "category": folder_name,
                    "knowledge_base_id": kb_id,
                })

            db.update_knowledge_document(doc_id, {"chunkCount": chunk_count})

            if VECTOR_DB_AVAILABLE and vector_docs:
                try:
                    old_chunk_ids = []
                    for i in range(existing_doc.get("chunkCount", 0)):
                        old_chunk_ids.append(f"doc_{doc_id}_chunk_{i}")
                    if old_chunk_ids:
                        from app.config import get_vector_db
                        vector_db = get_vector_db()
                        await asyncio.to_thread(vector_db.delete_documents, old_chunk_ids)
                    await asyncio.to_thread(vector_db.add_documents, vector_docs)
                    logger.info(f"文档 {doc_id} 向量数据库已更新")
                except Exception as ve:
                    logger.warning(f"更新向量数据库失败: {ve}")

        logger.info(f"更新知识库文档: {doc_id}")
        return {
            "success": True,
            "message": "文档更新成功",
            "document": updated_doc
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新知识库文档失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/knowledge/documents/{doc_id}")
async def delete_knowledge_document(doc_id: int, current_user: dict = Depends(get_current_admin)):
    """删除知识库文档

    s2 fix: 删除文档及关联向量索引，限定 admin。
    """
    try:
        existing_doc = db.get_knowledge_document(doc_id)
        if not existing_doc:
            raise HTTPException(status_code=404, detail="文档不存在")

        if VECTOR_DB_AVAILABLE:
            try:
                from app.config import get_vector_db
                vector_db = get_vector_db()
                chunk_ids = []
                chunks = db.get_knowledge_chunks(doc_id)
                for chunk in chunks:
                    chunk_id = f"doc_{doc_id}_chunk_{chunk.get('chunkIndex', chunk.get('id', 0))}"
                    chunk_ids.append(chunk_id)
                if chunk_ids:
                    await asyncio.to_thread(vector_db.delete_documents, chunk_ids)
            except Exception as ve:
                logger.warning(f"从向量数据库删除文档失败: {ve}")

        db.delete_knowledge_document(doc_id)
        logger.info(f"删除知识库文档: {doc_id}")
        return {"success": True, "message": "文档删除成功"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除知识库文档失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# 搜索
# ============================================

_vector_index_built = False
_vector_index_lock = threading.Lock()


def _ensure_vector_index():
    """延迟重建向量索引：首次搜索时从数据库加载chunks并构建Faiss索引。
    避免在启动时阻塞服务（尤其是多worker场景下GPU显存竞争）。

    C7 fix: 用 threading.Lock 保护 check-build-set 序列，防止多个并发搜索请求
    同时观察到 _vector_index_built == False 并重复构建索引（导致重复向量写入与资源浪费）。
    """
    global _vector_index_built
    # 双重检查：已构建时直接返回，避免每次搜索都获取锁
    if _vector_index_built:
        return True

    with _vector_index_lock:
        # 再次检查：可能已被其他线程构建
        if _vector_index_built:
            return True

        try:
            from app.config import VECTOR_DB_AVAILABLE
            if not VECTOR_DB_AVAILABLE:
                _vector_index_built = True
                return True

            from knowledge.vector_db import get_vector_db
            vector_db = get_vector_db()
            stats = vector_db.get_stats()

            if stats["total_documents"] > 0:
                logger.info(f"向量索引已存在: {stats['total_documents']} 个文档，跳过重建")
                _vector_index_built = True
                return True

            logger.info("向量索引为空，从数据库重建...")
            chunks = db.get_all_knowledge_chunks()
            if not chunks:
                logger.info("数据库中无知识库chunks，跳过向量索引重建")
                _vector_index_built = True
                return True

            all_docs = {doc["id"]: doc for doc in db.get_knowledge_documents(limit=10000)}
            vector_docs = []
            for chunk in chunks:
                doc = all_docs.get(chunk.get("documentId"))
                if not doc:
                    continue
                kb_name = ""
                folder_name = doc.get("category", "")
                if doc.get("knowledge_base_id"):
                    kb = db.get_knowledge_base(doc["knowledge_base_id"])
                    if kb:
                        kb_name = kb["name"]
                path_prefix = f"[{kb_name}/{folder_name}]" if kb_name else f"[{folder_name}]"
                enriched = f"{path_prefix} {doc['title']}: {chunk['content']}"
                vector_docs.append({
                    "id": f"doc_{chunk['documentId']}_chunk_{chunk['chunkIndex']}",
                    "chunk_index": chunk["chunkIndex"],
                    "title": doc["title"],
                    "content": enriched,
                    "document_id": chunk["documentId"],
                    "category": folder_name,
                    "knowledge_base_id": doc.get("knowledge_base_id"),
                })

            if vector_docs:
                vector_db.add_documents(vector_docs)
                logger.info(f"向量索引重建完成: {len(vector_docs)} 个chunks")
            _vector_index_built = True
            return True
        except Exception as e:
            logger.warning(f"向量索引重建失败: {e}")
            return False


@router.post("/api/knowledge/search")
async def search_knowledge(request: KnowledgeSearchRequest):
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
            all_bases = db.get_knowledge_bases()
            matched = [b for b in all_bases if b["name"] == request.knowledgeBaseName]
            if matched:
                filters = {"knowledge_base_id": matched[0]["id"]}
                logger.info(f"搜索过滤: 知识库「{request.knowledgeBaseName}」(id={matched[0]['id']})")
            else:
                logger.warning(f"未找到知识库「{request.knowledgeBaseName}」，不应用过滤")

        # 首次搜索时确保向量索引已构建（同步操作，放线程池避免阻塞事件循环）
        await asyncio.to_thread(_ensure_vector_index)

        # 优先使用 RAGHelper 完整管线（retrieve_context 是同步阻塞的 CPU 密集型操作）
        try:
            from knowledge.rag_helper import get_rag_helper
            rag = get_rag_helper()
            results = await asyncio.to_thread(rag.retrieve_context, query, top_k, True, filters, None)
            if results:
                formatted = []
                for r in results:
                    formatted.append({
                        "documentId": r.get("id"),
                        "documentTitle": r.get("title", ""),
                        "chunkIndex": r.get("chunk_index", 0),
                        "content": r.get("content", ""),
                        "score": r.get("normalized_score", r.get("score", 0)),
                        "searchType": "rag_pipeline"
                    })
                return {"success": True, "query": query, "results": formatted, "searchType": "rag_pipeline"}
        except Exception as e:
            logger.warning(f"RAGHelper检索失败，回退向量检索: {e}")

        # 回退：向量检索（hybrid_search 同步阻塞，放线程池）
        if VECTOR_DB_AVAILABLE:
            try:
                from app.config import get_vector_db
                vector_db = get_vector_db()
                vector_results = await asyncio.to_thread(
                    lambda: vector_db.hybrid_search(query, top_k=top_k, filters=filters)
                )
                if vector_results:
                    formatted = []
                    for r in vector_results:
                        formatted.append({
                            "documentId": r.get("id"),
                            "documentTitle": r.get("title", ""),
                            "chunkIndex": r.get("chunk_index", r.get("chunk_id", 0)),
                            "content": r.get("content", ""),
                            "score": r.get("score", 0),
                            "searchType": "hybrid"
                        })
                    return {"success": True, "query": query, "results": formatted, "searchType": "hybrid"}
            except Exception as ve:
                logger.warning(f"向量检索失败: {ve}")

        # 最终回退：关键词匹配（支持分词匹配，提高召回率）
        logger.info("回退到关键词匹配")
        query_lower = query.lower()
        # 提取查询中的关键词（中文单字+英文单词）
        import re as _re
        query_keywords = _re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+', query_lower)
        all_chunks = db.get_all_knowledge_chunks()
        all_docs = {doc["id"]: doc for doc in db.get_knowledge_documents(limit=1000)}
        results = []
        for chunk in all_chunks:
            content = chunk["content"].lower()
            doc = all_docs.get(chunk["documentId"])
            if not doc:
                continue
            # 完整匹配
            score = content.count(query_lower) * 0.5
            if query_lower in doc["title"].lower():
                score += 1.0
            # 分词匹配：每个关键词命中加分
            for kw in query_keywords:
                if len(kw) >= 2 or (len(kw) == 1 and '\u4e00' <= kw <= '\u9fff'):
                    score += content.count(kw) * 0.2
                    if kw in doc["title"].lower():
                        score += 0.5
            if score > 0:
                results.append({
                    "documentId": chunk["documentId"], "documentTitle": doc["title"],
                    "chunkIndex": chunk["chunkIndex"], "content": chunk["content"],
                    "score": round(score, 2), "searchType": "keyword"
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:top_k]
        return {"success": True, "query": query, "results": results, "searchType": "keyword"}
    except Exception as e:
        logger.error(f"搜索知识库失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/knowledge/stats")
async def get_knowledge_stats(current_user: dict = Depends(get_current_user)):
    """获取知识库统计数据"""
    stats = db.get_knowledge_stats()
    return {"success": True, "stats": stats}


@router.get("/api/vector/stats")
async def get_vector_stats(current_user: dict = Depends(get_current_user)):
    """获取向量数据库统计数据"""
    if not VECTOR_DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="向量数据库不可用")

    try:
        from app.config import get_vector_db
        vector_db = get_vector_db()
        stats = await asyncio.to_thread(vector_db.get_stats)
        return {"success": True, "stats": stats}
    except Exception as e:
        logger.error(f"获取向量数据库统计失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# 意图分类器训练
# ============================================

@router.post("/api/knowledge/train-intent/generate")
async def generate_intent_samples(request: dict = None, current_user: dict = Depends(get_current_user)):
    """生成训练样本（LLM基于知识库文档生成，不训练）"""
    try:
        from knowledge.intent_trainer import generate_samples, get_generation_status

        status = get_generation_status()
        if status["running"]:
            # C5 fix: 冲突状态应用 409，而非 200+success=False
            raise HTTPException(status_code=409, detail="样本生成正在进行中")

        params = request or {}
        kb_ids = params.get("kb_ids", [])
        samples_per_kb = params.get("samples_per_kb", 100)
        negative_count = params.get("negative_count", 200)
        lora_name = params.get("lora_name")

        asyncio.create_task(generate_samples(kb_ids, samples_per_kb, negative_count, lora_name))
        return {"success": True, "message": "样本生成已启动"}
    except HTTPException:
        # C9 fix: C5 引入的 409 必须在 except Exception 之前 re-raise，
        # 否则会被改写为 500，前端无法识别"冲突"语义
        raise
    except Exception as e:
        logger.error(f"启动样本生成失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/knowledge/train-intent/generate/status")
async def get_generation_status(current_user: dict = Depends(get_current_user)):
    """查询样本生成进度"""
    try:
        from knowledge.intent_trainer import get_generation_status as get_status
        return {"success": True, "status": get_status()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/knowledge/train-intent/samples")
async def get_intent_samples(current_user: dict = Depends(get_current_user)):
    """获取当前所有训练样本"""
    try:
        from knowledge.intent_trainer import get_samples
        return {"success": True, **get_samples()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/knowledge/train-intent/samples")
async def update_intent_sample(request: dict, current_user: dict = Depends(get_current_user)):
    """编辑单条样本"""
    try:
        from knowledge.intent_trainer import update_sample
        result = update_sample(request.get("label"), request.get("index"), request.get("text"))
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/knowledge/train-intent/samples")
async def delete_intent_sample(label: str, index: int, current_user: dict = Depends(get_current_user)):
    """删除单条样本"""
    try:
        from knowledge.intent_trainer import delete_sample
        result = delete_sample(label, index)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/api/knowledge/train-intent/samples")
async def batch_save_intent_samples(request: dict, current_user: dict = Depends(get_current_user)):
    """批量保存样本（覆盖写入）"""
    try:
        from knowledge.intent_trainer import save_samples
        result = save_samples(request.get("samples", {}))
        return {"success": True, "stats": result.get("stats", {})}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/knowledge/train-intent/samples")
async def add_intent_sample(request: dict, current_user: dict = Depends(get_current_user)):
    """添加单条样本"""
    try:
        from knowledge.intent_trainer import add_sample
        result = add_sample(request.get("label"), request.get("text"))
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/knowledge/train-intent")
async def train_intent_classifier(request: dict = None, current_user: dict = Depends(get_current_admin)):
    """使用已审查的样本训练多分类模型

    s2 fix: 触发模型训练消耗计算资源，限定 admin。
    """
    try:
        from knowledge.intent_trainer import train_intent_classifier as do_train, get_training_status

        status = get_training_status()
        if status["running"]:
            # C5 fix: 冲突状态应用 409，而非 200+success=False
            raise HTTPException(status_code=409, detail="训练正在进行中")

        params = request or {}
        kb_ids = params.get("kb_ids")

        asyncio.create_task(do_train(kb_ids=kb_ids))
        return {"success": True, "message": "训练已启动"}
    except HTTPException:
        # C9 fix: C5 引入的 409 必须在 except Exception 之前 re-raise
        raise
    except Exception as e:
        logger.error(f"启动意图训练失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/knowledge/train-intent/status")
async def get_intent_training_status(current_user: dict = Depends(get_current_user)):
    """查询训练进度"""
    try:
        from knowledge.intent_trainer import get_training_status
        return {"success": True, "status": get_training_status()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/knowledge/train-intent/cancel")
async def cancel_intent_training(current_user: dict = Depends(get_current_user)):
    """取消训练/生成"""
    try:
        from knowledge.intent_trainer import cancel_training
        result = cancel_training()
        return {"success": result, "message": "已发送取消请求" if result else "没有正在进行的任务"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/knowledge/train-intent/model")
async def get_intent_model_info(current_user: dict = Depends(get_current_user)):
    """获取当前模型信息"""
    try:
        from knowledge.intent_trainer import get_model_info
        return {"success": True, "model": get_model_info()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/knowledge/train-intent/active-kbs")
async def get_active_knowledge_bases(current_user: dict = Depends(get_current_user)):
    """获取参与检索的知识库"""
    try:
        from knowledge.intent_trainer import get_active_knowledge_bases as get_kbs
        return {"success": True, "active_kbs": get_kbs()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/knowledge/train-intent/active-kbs")
async def set_active_knowledge_bases(request: dict, current_user: dict = Depends(get_current_user)):
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
        raise HTTPException(status_code=500, detail=str(e))
