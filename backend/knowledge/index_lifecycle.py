"""Own generic knowledge index publication and durable revision checks."""

import logging
import threading

from db.adapter import db

logger = logging.getLogger(__name__)

_vector_index_revision: int | None = None
_vector_index_owner: tuple[object, object] | None = None
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
    global _vector_index_revision
    with _revision_lock:
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
    同时观察到 _vector_index_revision is None 并重复构建索引。

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
    global _vector_index_revision, _vector_index_owner
    with _vector_index_lock:
        try:
            from knowledge.vector_db import get_vector_db

            vector_db = get_vector_db()
            # Revisions only identify content inside one database/index binding.
            if (_vector_index_owner is not None
                    and _vector_index_owner[0] is db
                    and _vector_index_owner[1] is vector_db
                    and _vector_index_revision is not None
                    and _vector_index_revision == _get_rebuild_revision()):
                return True
            _vector_index_revision = None
            _vector_index_owner = (db, vector_db)
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
                # commit 临界区：revision 校验 + 写 complete + 发布本地版本号
                # 必须原子，避免并发 CRUD 在窗口内被覆盖
                with _revision_lock:
                    if _get_rebuild_revision() != current_revision:
                        logger.warning("空库清理期间检测到并发 CRUD，不标记 complete，等待下次重建")
                        return False
                    if not db.commit_knowledge_index_revision(current_revision, 0, _EMPTY_FINGERPRINT):
                        return False
                    logger.info("向量索引已清空并标记 complete:0:empty:%s", current_revision)
                    _vector_index_revision = current_revision
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
                    # 必须在锁内重新校验 revision 才能发布本地版本号
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

            # commit 临界区：revision 校验 + 写 complete + 发布本地版本号
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
            return True
        except Exception as e:
            logger.warning("向量索引重建失败: %s", type(e).__name__)
            raise
