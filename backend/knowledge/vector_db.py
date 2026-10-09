"""
向量数据库系统 - 用于RAG检索
基于Faiss的向量存储与检索核心模块，负责将知识库文档向量化并支持高效语义搜索。
提供多种索引类型（Flat/IVF/HNSW）、BM25关键词检索、混合搜索、元数据过滤、批量操作等功能。
"""

import hashlib
import json
import logging
import math
import os
import pickle
import re
import time
from collections import Counter, OrderedDict, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from knowledge.retrieval_core.embedding import resolve_local_model_path
from knowledge.snapshot_store import SNAPSHOT_NAME, document_ids, load_snapshot, save_snapshot

logger = logging.getLogger(__name__)


@dataclass
class IndexConfig:
    index_type: str = "auto"
    nlist: int = 100
    nprobe: int = 10
    m_hnsw: int = 32
    ef_construction: int = 200
    ef_search: int = 64
    auto_switch_threshold: int = 10000
    batch_encode_size: int = 64
    save_on_every_n_adds: int = 0


class BM25Retriever:
    """BM25关键词检索器，基于词频-逆文档频率对文档进行关键词匹配排序。

    用于与向量检索互补，提升混合搜索的召回率。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """初始化BM25检索器。

        Args:
            k1: 词频饱和度参数，默认1.5
            b: 文档长度归一化参数，默认0.75
        """
        self.k1 = k1
        self.b = b
        self.corpus: List[str] = []
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.doc_lens: List[int] = []
        self.avgdl: float = 0.0
        self.idf: Dict[str, float] = {}
        self.tokenized_corpus: List[List[str]] = []
        self._built = False

    def _tokenize(self, text: str) -> List[str]:
        """中文用jieba分词，英文按单词切分"""
        tokens = re.findall(r'[\w]+', text.lower())
        try:
            import jieba
            cn_tokens = list(jieba.cut(text))
            tokens.extend(t for t in cn_tokens if t.strip() and not t.isascii())
        except ImportError:
            cn_chars = re.findall(r'[\u4e00-\u9fff]', text)
            tokens.extend(cn_chars)
        return tokens

    def add_documents(self, documents: List[Dict[str, Any]]):
        """将文档加入BM25索引，对标题和内容分词后统计词频和文档频率。

        Args:
            documents: 文档列表，每个文档需包含title和content字段
        """
        for doc in documents:
            text = f"{doc.get('title', '')} {doc.get('content', '')}"
            self.corpus.append(text)
            tokens = self._tokenize(text)
            self.tokenized_corpus.append(tokens)
            self.doc_lens.append(len(tokens))

            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1

        self.avgdl = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0
        self._compute_idf()
        self._built = True

    def _compute_idf(self):
        n_docs = len(self.corpus)
        self.idf = {}
        for token, freq in self.doc_freqs.items():
            self.idf[token] = math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 10, threshold: float = 0.0) -> List[Tuple[int, float]]:
        """BM25关键词搜索。

        Args:
            query: 搜索查询文本
            top_k: 返回结果数量
            threshold: 最低分数阈值

        Returns:
            按BM25分数降序排列的 (文档索引, 分数) 列表
        """
        if not self._built or not self.corpus:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = []
        for doc_idx, doc_tokens in enumerate(self.tokenized_corpus):
            score = 0.0
            token_counts = Counter(doc_tokens)
            doc_len = self.doc_lens[doc_idx]

            for token in query_tokens:
                if token not in self.idf:
                    continue
                tf = token_counts.get(token, 0)
                if tf == 0:
                    continue
                idf = self.idf[token]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1))
                score += idf * numerator / denominator

            if score >= threshold:
                scores.append((doc_idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_docs": len(self.corpus),
            "vocab_size": len(self.doc_freqs),
            "avg_doc_len": round(self.avgdl, 1),
            "built": self._built,
        }


class VectorDatabase:
    """向量数据库管理类，封装Faiss索引的创建、加载、查询和混合检索。

    支持三种索引类型：Flat（小数据集）、IVF（中等数据集，自动迁移阈值默认1万）、
    HNSW（大数据集，自动迁移阈值默认10万）。集成BM25检索器实现混合搜索。
    """

    EMBEDDING_DIM = 384
    EMBEDDING_CACHE_SIZE = 2000

    def __init__(self, db_path: str = "data/vector_db", index_config: Optional[IndexConfig] = None):
        """初始化向量数据库。

        Args:
            db_path: 向量数据存储目录路径
            index_config: 索引配置对象，默认使用IndexConfig()
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.index_path = self.db_path / "faiss_index.bin"
        self.metadata_path = self.db_path / "metadata.pkl"
        self.bm25_path = self.db_path / "bm25_state.pkl"
        self.config_path = self.db_path / "db_config.json"
        self.snapshot_path = self.db_path / SNAPSHOT_NAME
        self.snapshot_validated = False
        self._snapshot_bm25 = None

        self.config = index_config or IndexConfig()
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self._lock = RLock()
        self._id_to_index: Dict[int, int] = {}
        self._model = None
        self._device = None
        self._use_gpu = self._check_gpu_availability()
        self._add_count = 0
        self._dirty = False

        # 查询结果LRU缓存
        self._query_cache: OrderedDict = OrderedDict()
        self._query_cache_max_size = 100
        self._query_cache_ttl = 300  # 查询缓存TTL（秒）
        self._cache_generation = 0

        self.bm25 = BM25Retriever()

        self._ensure_index()
        self._rebuild_id_mapping()
        self._load_bm25()

        logger.info(f"向量数据库初始化完成，使用GPU: {self._use_gpu}，"
                    f"文档数: {len(self.metadata)}")

    def _check_gpu_availability(self) -> bool:
        import torch

        return bool(torch.cuda.is_available())

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._device = "cuda" if self._use_gpu else "cpu"
            model_path = resolve_local_model_path()
            self._model = SentenceTransformer(model_path, device=self._device)
            logger.info("向量嵌入模型加载完成，设备: %s", self._device)

    def _determine_index_type(self) -> str:
        if self.config.index_type != "auto":
            return self.config.index_type

        n_docs = len(self.metadata)
        if n_docs < self.config.auto_switch_threshold:
            return "flat"
        elif n_docs < 100000:
            return "ivf"
        else:
            return "hnsw"

    def _ensure_index(self):
        if self.index is None:
            if any(path.exists() for path in (self.snapshot_path, self.index_path, self.metadata_path, self.bm25_path)):
                self._load_index()
            else:
                self._create_index()

    def _create_index(self, index_type: Optional[str] = None):
        idx_type = self._determine_index_type() if index_type is None else index_type
        dim = self.EMBEDDING_DIM

        if idx_type == "flat":
            base_index = faiss.IndexFlatIP(dim)
            logger.info("创建FAISS平面索引（适合小数据集）")

        elif idx_type == "ivf":
            nlist = min(self.config.nlist, max(1, int(math.sqrt(len(self.metadata)))))
            quantizer = faiss.IndexFlatIP(dim)
            base_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            base_index.nprobe = self.config.nprobe
            logger.info(f"创建FAISS IVF索引（nlist={nlist}，适合中等数据集）")

        elif idx_type == "hnsw":
            base_index = faiss.IndexHNSWFlat(dim, self.config.m_hnsw, faiss.METRIC_INNER_PRODUCT)
            base_index.hnsw.efConstruction = self.config.ef_construction
            base_index.hnsw.efSearch = self.config.ef_search
            logger.info(f"创建FAISS HNSW索引（M={self.config.m_hnsw}，适合大数据集）")

        else:
            raise ValueError(f"不支持的 index_type: {idx_type!r}；请选择 flat、ivf、hnsw 或配置 auto")

        self.index = faiss.IndexIDMap(base_index)

    def _load_index(self):
        if self.snapshot_path.exists():
            self.index, self.metadata, self._snapshot_bm25 = load_snapshot(
                self.db_path, self.EMBEDDING_DIM, self._to_faiss_id
            )
            self.snapshot_validated = True
        else:
            # Legacy triples remain readable but cannot authorize API reuse.
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        logger.info(f"加载FAISS索引完成，共 {len(self.metadata)} 条记录")

    def _load_bm25(self):
        if self.snapshot_path.exists():
            bm25_data = self._snapshot_bm25
        elif self.bm25_path.exists() or self.metadata:
            with open(self.bm25_path, 'rb') as f:
                bm25_data = pickle.load(f)
        else:
            return
        self.bm25.corpus = bm25_data["corpus"]
        self.bm25.doc_freqs = defaultdict(int, bm25_data["doc_freqs"])
        self.bm25.doc_lens = bm25_data["doc_lens"]
        self.bm25.avgdl = bm25_data["avgdl"]
        self.bm25.idf = bm25_data["idf"]
        self.bm25.tokenized_corpus = bm25_data["tokenized_corpus"]
        self.bm25._built = bm25_data["built"]

    def _bm25_state(self):
        return {
            "corpus": self.bm25.corpus,
            "doc_freqs": dict(self.bm25.doc_freqs),
            "doc_lens": self.bm25.doc_lens,
            "avgdl": self.bm25.avgdl,
            "idf": self.bm25.idf,
            "tokenized_corpus": self.bm25.tokenized_corpus,
            "built": self.bm25._built,
        }

    def _rebuild_id_mapping(self):
        self._id_to_index = {key: i for i, key in enumerate(document_ids(self.metadata, self._to_faiss_id))}

    def _save_index(self):
        """Publish the complete triple together; failures remain dirty."""
        with self._lock:
            self._dirty = True
            save_snapshot(self.db_path, self.index, self.metadata, self._bm25_state(),
                          self.EMBEDDING_DIM, self._to_faiss_id)
            self.snapshot_validated = True
            self._dirty = False
            logger.info("FAISS、来源与BM25完整快照保存完成")

    def _to_faiss_id(self, id_value: Any) -> Optional[int]:
        try:
            if isinstance(id_value, int):
                if id_value < -(1 << 63) or id_value >= (1 << 63):
                    id_value = id_value % (1 << 63)
                return id_value
            elif isinstance(id_value, str):
                try:
                    value = int(id_value)
                    if value < -(1 << 63) or value >= (1 << 63):
                        value = value % (1 << 63)
                    return value
                except ValueError:
                    pass
                try:
                    if len(id_value) > 16:
                        hex_str = id_value[:16]
                        if all(c in '0123456789abcdefABCDEF' for c in hex_str):
                            value = int(hex_str, 16)
                            if value < -(1 << 63) or value >= (1 << 63):
                                value = value % (1 << 63)
                            return value
                    if all(c in '0123456789abcdefABCDEF' for c in id_value):
                        value = int(id_value, 16)
                        if value < -(1 << 63) or value >= (1 << 63):
                            value = value % (1 << 63)
                        return value
                except ValueError:
                    pass
                hash_obj = hashlib.md5(id_value.encode('utf-8'))
                digest_bytes = hash_obj.digest()[:8]
                value = int.from_bytes(digest_bytes, byteorder='big', signed=True)
                return value
            else:
                value = int(id_value)
                if value < -(1 << 63) or value >= (1 << 63):
                    value = value % (1 << 63)
                return value
        except Exception as e:
            logger.warning(f"无法转换ID '{id_value}' 为整数: {e}")
            return None

    def _get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        self._load_model()
        batch_size = self.config.batch_encode_size
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self._model.encode(batch, normalize_embeddings=True, batch_size=len(batch))
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings).astype('float32')

    def _get_embedding(self, text: str) -> np.ndarray:
        return self._get_embeddings_batch([text])[0]

    def _maybe_migrate_index(self):
        if self.config.index_type != "auto":
            return
        target = self._determine_index_type()
        expected = {"flat": faiss.IndexFlatIP, "ivf": faiss.IndexIVFFlat, "hnsw": faiss.IndexHNSWFlat}[target]
        if not isinstance(faiss.downcast_index(self.index.index), expected):
            self._migrate_index(target)

    def _migrate_index(self, index_type: str):
        if not self.metadata:
            return
        old_index, old_metadata, old_mapping = self.index, self.metadata, self._id_to_index
        try:
            self._create_index(index_type)
            self._re_add_all_documents(old_metadata.copy())
        except Exception:
            self.index, self.metadata, self._id_to_index = old_index, old_metadata, old_mapping
            logger.error("索引迁移失败，已恢复旧索引")
            raise

    def _re_add_all_documents(self, documents: List[Dict[str, Any]]):
        ids = np.array(document_ids(documents, self._to_faiss_id), dtype=np.int64)
        texts = [f"{doc.get('title', '')} {doc.get('content', '')}" for doc in documents]
        embeddings = self._get_embeddings_batch(texts)

        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            logger.info("训练IVF索引...")
            train_size = min(len(embeddings), max(256, self.config.nlist * 39))
            train_data = embeddings[:train_size]
            self.index.train(train_data)

        self.index.add_with_ids(embeddings, ids)
        self.metadata = documents
        self._rebuild_id_mapping()
        self._save_index()

    def add_documents(self, documents: List[Dict[str, Any]], kb_revision: Optional[str] = None):
        """添加文档到向量数据库，自动生成嵌入向量并写入Faiss索引和BM25索引。

        如果文档数量超过自动切换阈值，会自动迁移索引类型（Flat -> IVF -> HNSW）。
        同时为每个文档注入 RAG 2.0 证据化元数据（content_hash/kb_revision/section/version/import_time），
        使用 setdefault 向后兼容已有字段。

        Args:
            documents: 文档列表，每个文档需包含id、title、content字段
            kb_revision: 知识库版本标识，用于引用溯源。默认使用当日日期。
        """
        if not documents:
            return

        rev = kb_revision or datetime.now().strftime("%Y%m%d")
        now_iso = datetime.now().isoformat()
        for doc in documents:
            chunk_text = doc.get("content", "")
            doc.setdefault("content_hash", hashlib.md5(chunk_text.encode("utf-8")).hexdigest()[:12])
            doc.setdefault("kb_revision", rev)
            doc.setdefault("section", doc.get("category", ""))
            doc.setdefault("version", "1.0")
            doc.setdefault("import_time", now_iso)

        with self._lock:
            self._ensure_index()
            keys = document_ids(documents, self._to_faiss_id)
            if self._id_to_index.keys() & set(keys):
                raise ValueError("Document identity already exists in vector index")
            ids = np.array(keys, dtype=np.int64)
            self._load_model()

            texts = []
            for doc in documents:
                full_text = f"{doc.get('title', '')} {doc.get('content', '')}"
                texts.append(full_text)

            logger.info(f"正在生成 {len(texts)} 个向量...")
            embeddings = self._get_embeddings_batch(texts)

            if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                logger.info("训练IVF索引...")
                train_size = min(len(embeddings), max(256, self.config.nlist * 39))
                self.index.train(embeddings[:train_size])

            # 修改开始后即使失败，也不能授权旧缓存或把内存状态视为已落盘。
            self._dirty = True
            self.clear_cache()
            self.index.add_with_ids(embeddings, ids)

            start_idx = len(self.metadata)
            self.metadata.extend(documents)
            for i, faiss_id in enumerate(ids):
                self._id_to_index[int(faiss_id)] = start_idx + i

            self.bm25.add_documents(documents)

            self._add_count += len(documents)

            if self.config.save_on_every_n_adds <= 0 or self._add_count >= self.config.save_on_every_n_adds:
                self._save_index()
                self._add_count = 0
            else:
                logger.info(f"延迟保存：已添加 {self._add_count} 个文档（阈值: {self.config.save_on_every_n_adds}）")

            self._maybe_migrate_index()

            logger.info(f"成功添加 {len(documents)} 个文档到向量数据库（总计: {len(self.metadata)}）")

    def flush(self):
        if self._dirty:
            self._save_index()

    def _normalize_query(self, query: str) -> str:
        """规范化查询文本，用于缓存key生成"""
        return " ".join(query.lower().split())

    def _get_query_cache_key(self, query: str, top_k: int, threshold: float, filters: Optional[Dict[str, Any]]) -> str:
        """生成查询缓存key"""
        normalized = self._normalize_query(query)
        filter_str = json.dumps(filters, sort_keys=True) if filters else ""
        return f"{normalized}|{top_k}|{threshold}|{filter_str}"

    def _get_cached_query(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """从查询缓存中获取结果，过期则返回None"""
        if cache_key not in self._query_cache:
            return None
        cached_entry = self._query_cache[cache_key]
        cached_time, cached_results = cached_entry
        if time.time() - cached_time > self._query_cache_ttl:
            # 缓存过期，移除
            self._query_cache.pop(cache_key, None)
            return None
        # LRU：移到末尾
        self._query_cache.move_to_end(cache_key)
        return cached_results

    def _set_query_cache(self, cache_key: str, results: List[Dict[str, Any]]) -> None:
        """将查询结果存入缓存"""
        # LRU淘汰
        while len(self._query_cache) >= self._query_cache_max_size:
            self._query_cache.popitem(last=False)
        self._query_cache[cache_key] = (time.time(), results)

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.15,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """纯向量语义搜索，将查询文本向量化后在Faiss索引中进行最近邻检索。
        支持查询结果缓存，相同查询参数在TTL内直接返回缓存结果。

        Args:
            query: 搜索查询文本
            top_k: 返回结果数量
            threshold: 最低相似度阈值，低于此分数的结果会被过滤
            filters: 元数据过滤条件字典，支持等值、列表、范围等过滤

        Returns:
            按相似度降序排列的搜索结果列表，每项包含完整元数据及score字段
        """
        with self._lock:
            self._ensure_index()
            if len(self.metadata) == 0:
                return []

            # 查询缓存检查
            cache_key = self._get_query_cache_key(query, top_k, threshold, filters)
            cached = self._get_cached_query(cache_key)
            if cached is not None:
                logger.debug(f"查询缓存命中: query='{query[:30]}...'")
                return deepcopy(cached)

            query_embedding = self._get_embedding(query)
            query_embedding = np.array([query_embedding]).astype('float32')

            search_k = top_k
            if filters:
                search_k = min(top_k * 5, len(self.metadata))

            scores, faiss_ids = self.index.search(query_embedding, search_k)

            results = []
            for score, faiss_id in zip(scores[0], faiss_ids[0]):
                if score < threshold:
                    continue
                meta_idx = self._id_to_index.get(int(faiss_id))
                if meta_idx is None or meta_idx < 0 or meta_idx >= len(self.metadata):
                    continue
                result = {**self.metadata[meta_idx], "score": float(score)}
                if filters and not self._match_filters(result, filters):
                    continue
                results.append(result)
                if len(results) >= top_k:
                    break

            # 存入查询缓存
            self._set_query_cache(cache_key, results)

            return deepcopy(results)

    def _match_filters(self, doc: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        for key, value in filters.items():
            doc_value = doc.get(key)
            if isinstance(value, list):
                if doc_value not in value:
                    return False
            elif isinstance(value, dict):
                if "$contains" in value and value["$contains"] not in str(doc_value):
                    return False
                if "$gt" in value and not (doc_value is not None and doc_value > value["$gt"]):
                    return False
                if "$lt" in value and not (doc_value is not None and doc_value < value["$lt"]):
                    return False
                if "$in" in value and doc_value not in value["$in"]:
                    return False
            else:
                if doc_value != value:
                    return False
        return True

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.15,
        keyword_weight: float = 0.3,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """混合搜索：融合向量语义检索和BM25关键词检索的结果。

        先分别进行向量检索和BM25检索，然后对分数进行归一化融合，
        融合权重为 keyword_weight * BM25分数 + (1 - keyword_weight) * 向量分数。

        Args:
            query: 搜索查询文本
            top_k: 返回结果数量
            threshold: 向量检索的最低相似度阈值
            keyword_weight: BM25关键词权重（0~1），默认0.3
            filters: 元数据过滤条件

        Returns:
            按融合分数降序排列的搜索结果，每项包含vector_score、bm25_score、fused_score
        """
        with self._lock:
            recall_k = min(top_k * 3, len(self.metadata))
            vector_results = self.search(query, top_k=recall_k, threshold=threshold, filters=filters)

            doc_scores: Dict[int, Dict[str, float]] = {}

            for doc in vector_results:
                key = self._id_to_index[self._to_faiss_id(doc["id"])]
                doc_scores[key] = {"vector_score": doc["score"], "bm25_score": 0.0, "doc": doc}

            if self.bm25._built:
                for key, score in self.bm25.search(query, top_k=recall_k, threshold=0.0):
                    doc = self.metadata[key]
                    if filters and not self._match_filters(doc, filters):
                        continue
                    if key in doc_scores:
                        doc_scores[key]["bm25_score"] = score
                    else:
                        doc_scores[key] = {"vector_score": 0.0, "bm25_score": score, "doc": doc}

            max_bm25 = max((s["bm25_score"] for s in doc_scores.values()), default=1.0) or 1.0
            max_vector = max((s["vector_score"] for s in doc_scores.values()), default=1.0) or 1.0

            fused_results = []
            for key, scores in doc_scores.items():
                norm_vector = scores["vector_score"] / max_vector if max_vector > 0 else 0
                norm_bm25 = scores["bm25_score"] / max_bm25 if max_bm25 > 0 else 0
                fused_score = (1 - keyword_weight) * norm_vector + keyword_weight * norm_bm25

                doc = dict(scores["doc"])
                doc["vector_score"] = scores["vector_score"]
                doc["bm25_score"] = scores["bm25_score"]
                doc["fused_score"] = fused_score
                fused_results.append(doc)

            fused_results.sort(key=lambda x: x["fused_score"], reverse=True)
            return deepcopy(fused_results[:top_k])

    def clear_all(self):
        """清空所有向量索引和元数据，并立即持久化空索引到磁盘。

        用于向量索引重建前的清理：确保上次失败的部分批次不会残留，
        也避免删除全部知识后旧磁盘文件被下次启动加载。
        重建状态由调用方（_ensure_vector_index）在 config 表中管理。

        失败时抛出异常（不吞错），确保调用方不会在落盘失败时误标记 complete。
        """
        with self._lock:
            self.index = None
            self.metadata = []
            self._id_to_index = {}
            self._query_cache.clear()
            self._cache_generation += 1
            self._create_index()
            # 重置 BM25：旧文档词频与新文档混合会导致 hybrid/BM25 结果错位
            self.bm25 = BM25Retriever()
            self._dirty = True
            # 立即持久化空索引（原子替换），覆盖磁盘上的旧文件
            # 防止"内存清空但磁盘残留"导致重启后旧内容被加载
            self._save_index()  # 内部使用原子写入，失败时抛出异常
        logger.info("向量索引已清空并持久化空索引，准备重建")

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            index_type = "unknown"
            if isinstance(self.index, faiss.IndexIDMap):
                inner = faiss.downcast_index(self.index.index)
                if isinstance(inner, faiss.IndexFlatIP):
                    index_type = "flat"
                elif isinstance(inner, faiss.IndexIVFFlat):
                    index_type = "ivf"
                elif isinstance(inner, faiss.IndexHNSWFlat):
                    index_type = "hnsw"

            return {
                "total_documents": len(self.metadata),
                "index_size": self.index.ntotal if self.index else 0,
                "index_type": index_type,
                "embedding_dim": self.EMBEDDING_DIM,
                "use_gpu": self._use_gpu,
                "bm25_built": self.bm25._built,
                "bm25_vocab_size": len(self.bm25.doc_freqs),
                "bm25_corpus_size": len(self.bm25.corpus),
                "dirty": self._dirty,
            }

    @property
    def cache_generation(self) -> int:
        """Monotonic in-process revision used by upper retrieval cache keys."""
        return self._cache_generation

    def clear_cache(self):
        """清除查询结果缓存并使上层 RAG 查询缓存自然失效。"""
        with self._lock:
            self._query_cache.clear()
            self._cache_generation += 1
        logger.info("向量查询缓存已清除")

_vector_db: Optional[VectorDatabase] = None


def get_vector_db() -> VectorDatabase:
    """获取向量数据库全局单例。

    首次调用时自动初始化，后续调用返回同一实例。

    Returns:
        VectorDatabase: 全局唯一的向量数据库实例
    """
    global _vector_db
    if _vector_db is None:
        configured_path = os.getenv("VECTOR_DB_PATH", "").strip()
        if configured_path:
            candidate = Path(configured_path).expanduser()
            db_path = str(candidate if candidate.is_absolute() else Path(__file__).parent.parent / candidate)
        else:
            db_path = str(Path(__file__).parent / "data" / "vector_db")
        _vector_db = VectorDatabase(db_path=db_path)
    return _vector_db
