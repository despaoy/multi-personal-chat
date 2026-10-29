"""Knowledge database, index publication and retrieval caches share one owner."""

import hashlib
import logging
import os
import threading
from functools import cache, partial
from pathlib import Path

from db.adapter import db
from knowledge import rag_helper as rag_resources, vector_db as vector_resources

logger = logging.getLogger(__name__)
_VECTOR_REBUILD_STATUS_KEY = "vector_index_rebuild_status"
_VECTOR_REBUILD_REVISION_KEY = "vector_index_rebuild_revision"
_EMPTY_FINGERPRINT = "empty"


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


class KnowledgeIndex:
    """Own one source database and its index, publication locks and RAG cache."""

    _VECTOR_REBUILD_STATUS_KEY = _VECTOR_REBUILD_STATUS_KEY
    _VECTOR_REBUILD_REVISION_KEY = _VECTOR_REBUILD_REVISION_KEY
    _vector_chunk_document = staticmethod(_vector_chunk_document)
    _chunk_fingerprint_record = staticmethod(_chunk_fingerprint_record)

    def __init__(self, database, vector_provider, rag_provider=None):
        self.db = database
        self._vector_provider = vector_provider
        self._rag_provider = rag_provider
        self._rag_helper = None
        self._resources_lock = threading.RLock()
        self._vector_index_revision = None
        self._vector_index_owner = None
        self._vector_index_lock = threading.Lock()
        self._revision_lock = threading.Lock()

    @property
    def vector_db(self):
        with self._resources_lock:
            return self._vector_provider()

    @property
    def rag_helper(self):
        with self._resources_lock:
            if self._rag_provider is not None:
                return self._rag_provider()
            if self._rag_helper is None:
                self._rag_helper = rag_resources.RAGHelper(vector_provider=lambda: self.vector_db)
            return self._rag_helper

    def _get_rebuild_revision(self) -> int:
        """读取当前重建修订号（单调递增）。默认 0。"""
        from db.knowledge_index_state import parse_revision

        return parse_revision(self.db.get_config_value(_VECTOR_REBUILD_REVISION_KEY, "0"))

    def _compute_chunk_fingerprint(self) -> str:
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
        for row in self.db.iter_chunks_with_document(batch_size=500):
            if row.get("doc_title") is None:
                continue
            h.update(_chunk_fingerprint_record(row))
        return h.hexdigest()[:16]

    def _get_expected_chunk_count(self) -> int:
        """获取数据库中"有效" chunk 的预期总数（与重建遍历范围一致）。

        使用 INNER JOIN：与 _ensure_vector_index 重建时跳过孤儿 chunk 的逻辑一致，
        避免"预期数量含孤儿但遍历跳过孤儿"导致永久不匹配、每次搜索都重建。
        """
        rows = self.db.execute_sql(
            'SELECT COUNT(*) AS cnt FROM knowledge_chunks c INNER JOIN knowledge_documents d ON c."documentId" = d.id',
            {},
        )
        return rows[0]["cnt"]

    def _read_rebuild_status(self) -> tuple[str, int, str, int]:
        """Read this owner's persisted status using the shared format contract."""
        from db.knowledge_index_state import parse_status

        return parse_status(self.db.get_config_value(_VECTOR_REBUILD_STATUS_KEY, ""))

    def _invalidate_local_knowledge_index(self) -> None:
        """Reset local acceleration only after the durable document transaction."""
        with self._revision_lock:
            self._vector_index_revision = None

    def _loaded_metadata_matches_database(self, vector_db) -> bool:
        with vector_db._lock:
            if not getattr(vector_db, "snapshot_validated", False):
                return False
            actual = {record.get("id"): record for record in vector_db.metadata}
            if len(actual) != len(vector_db.metadata):
                return False
            kb_names = {kb["id"]: kb["name"] for kb in self.db.get_knowledge_bases()}
            for row in self.db.iter_chunks_with_document(batch_size=500):
                if row.get("doc_title") is None:
                    continue
                expected = _vector_chunk_document(row, kb_names)
                loaded = actual.pop(expected["id"], None)
                if loaded is None or any((loaded.get(key) != value for key, value in expected.items())):
                    return False
            return not actual

    def _ensure_vector_index(self):
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
        with self._vector_index_lock:
            try:
                vector_db = self.vector_db
                if (
                    self._vector_index_owner is not None
                    and self._vector_index_owner[0] is self.db
                    and (self._vector_index_owner[1] is vector_db)
                    and (self._vector_index_revision is not None)
                    and (self._vector_index_revision == self._get_rebuild_revision())
                ):
                    return True
                self._vector_index_revision = None
                self._vector_index_owner = (self.db, vector_db)
                stats = vector_db.get_stats()
                expected_count = self._get_expected_chunk_count()
                status, status_count, status_fp, status_rev = self._read_rebuild_status()
                current_revision = self._get_rebuild_revision()
                if expected_count == 0:
                    needs_clear = (
                        not getattr(vector_db, "snapshot_validated", False)
                        or stats["total_documents"] != 0
                        or stats["index_size"] != 0
                        or (stats["bm25_corpus_size"] != 0)
                        or (status != "complete")
                        or (status_count != 0)
                        or (status_rev != current_revision)
                    )
                    if needs_clear:
                        logger.info("数据库无有效 chunk，清空向量索引并持久化空索引")
                        vector_db.clear_all()
                    with self._revision_lock:
                        if self._get_rebuild_revision() != current_revision:
                            logger.warning("空库清理期间检测到并发 CRUD，不标记 complete，等待下次重建")
                            return False
                        if not self.db.commit_knowledge_index_revision(current_revision, 0, _EMPTY_FINGERPRINT):
                            return False
                        logger.info("向量索引已清空并标记 complete:0:empty:%s", current_revision)
                        self._vector_index_revision = current_revision
                    return True
                if (
                    status == "complete"
                    and status_count == expected_count
                    and (status_rev == current_revision)
                    and (stats["total_documents"] == expected_count)
                    and (stats["index_size"] == expected_count)
                    and (stats["bm25_corpus_size"] == expected_count)
                ):
                    current_fp = self._compute_chunk_fingerprint()
                    if status_fp and status_fp == current_fp and self._loaded_metadata_matches_database(vector_db):
                        with self._revision_lock:
                            if self._get_rebuild_revision() != current_revision:
                                logger.info("跳过检查期间检测到并发 CRUD，放弃跳过，进入重建")
                            else:
                                if not self.db.commit_knowledge_index_revision(
                                    current_revision, expected_count, current_fp
                                ):
                                    return False
                                logger.info(
                                    "向量索引已完整: %d 个文档（complete，数量+指纹+revision 匹配），跳过重建",
                                    stats["total_documents"],
                                )
                                vector_db.clear_cache()
                                self._vector_index_revision = current_revision
                                return True
                    else:
                        logger.info("向量索引指纹不匹配: 状态=%s, 实际=%s，需要重建", status_fp or "(空)", current_fp)
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
                start_revision = self._get_rebuild_revision()
                self.db.set_config_value(_VECTOR_REBUILD_STATUS_KEY, f"building:{expected_count}")
                vector_db.clear_all()
                kb_name_map: dict = {}
                for kb in self.db.get_knowledge_bases():
                    kb_name_map[kb["id"]] = kb["name"]
                batch_vector_docs = []
                vector_batch_size = 200
                total_chunks_indexed = 0
                import hashlib

                fp_hash = hashlib.md5()
                for row in self.db.iter_chunks_with_document(batch_size=500):
                    doc_title = row.get("doc_title")
                    if doc_title is None:
                        continue
                    batch_vector_docs.append(_vector_chunk_document(row, kb_name_map))
                    fp_hash.update(_chunk_fingerprint_record(row))
                    if len(batch_vector_docs) >= vector_batch_size:
                        vector_db.add_documents(batch_vector_docs)
                        total_chunks_indexed += len(batch_vector_docs)
                        batch_vector_docs = []
                if batch_vector_docs:
                    vector_db.add_documents(batch_vector_docs)
                    total_chunks_indexed += len(batch_vector_docs)
                if total_chunks_indexed != expected_count:
                    logger.error(
                        "向量索引重建数量不匹配: 已写入 %d, 预期 %d，保持 building 状态等待下次重建",
                        total_chunks_indexed,
                        expected_count,
                    )
                    return False
                vector_db.flush()
                final_fp = fp_hash.hexdigest()[:16]
                with self._revision_lock:
                    final_revision = self._get_rebuild_revision()
                    if final_revision != start_revision:
                        logger.warning(
                            "重建期间检测到并发 CRUD: start_revision=%d, current=%d，不标记 complete，保持 building 等待下次重建",
                            start_revision,
                            final_revision,
                        )
                        return False
                    if not self.db.commit_knowledge_index_revision(start_revision, total_chunks_indexed, final_fp):
                        return False
                    logger.info(
                        "向量索引重建完成: %s 个 chunks（数量+指纹+revision CAS 校验通过，revision=%s）",
                        total_chunks_indexed,
                        start_revision,
                    )
                    self._vector_index_revision = start_revision
                return True
            except Exception as e:
                logger.warning("向量索引重建失败: %s", type(e).__name__)
                raise


# The process-default binding remains for standalone evaluation commands.
default_knowledge_index = KnowledgeIndex(
    db, lambda: vector_resources.get_vector_db(), lambda: rag_resources.get_rag_helper()
)
_owner_init_lock = threading.Lock()


def get_knowledge_index(database):
    """Reuse resources owned by this adapter, never by another application's store."""
    with _owner_init_lock:
        owner = getattr(database, "_knowledge_index", None)
        if owner is not None:
            return owner
        if database is db:
            owner = default_knowledge_index
        else:
            from db.database import SQLiteDB

            if isinstance(database, SQLiteDB):
                source = Path(database.db_path).expanduser().resolve()
                path = source.with_name(source.name + ".knowledge")
            else:
                from db.pg_database import SyncPgAdapter
                from sqlalchemy.engine import make_url

                if not isinstance(database, SyncPgAdapter):
                    raise TypeError("Knowledge resources require a configured database adapter")
                identity = make_url(database._pg.database_url).set(password=None).render_as_string(hide_password=True)
                namespace = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
                configured = os.environ.get("VECTOR_DB_PATH", "").strip()
                root = Path(configured).expanduser() if configured else Path(__file__).parent / "data/vector_db"
                if not root.is_absolute():
                    root = Path(__file__).parent.parent / root
                path = root / "stores" / namespace
            vector = cache(partial(vector_resources.VectorDatabase, db_path=str(path)))
            owner = KnowledgeIndex(database, vector)
        database._knowledge_index = owner
        return owner


# Historical evaluation scripts import these readers directly. They all resolve
# the same default owner; production callers select an owner by database.
_get_rebuild_revision = default_knowledge_index._get_rebuild_revision
_compute_chunk_fingerprint = default_knowledge_index._compute_chunk_fingerprint
_get_expected_chunk_count = default_knowledge_index._get_expected_chunk_count
_read_rebuild_status = default_knowledge_index._read_rebuild_status
_invalidate_local_knowledge_index = default_knowledge_index._invalidate_local_knowledge_index
_loaded_metadata_matches_database = default_knowledge_index._loaded_metadata_matches_database
_ensure_vector_index = default_knowledge_index._ensure_vector_index
