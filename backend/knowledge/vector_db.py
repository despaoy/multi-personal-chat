"""
向量数据库系统 - 用于RAG检索
基于Faiss的向量存储与检索核心模块，负责将知识库文档向量化并支持高效语义搜索。
提供多种索引类型（Flat/IVF/HNSW）、BM25关键词检索、混合搜索、元数据过滤、批量操作等功能。
"""

import os
import re
import math
import json
import hashlib
import numpy as np
import faiss
import pickle
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
from threading import RLock
from collections import Counter, defaultdict, OrderedDict
from dataclasses import dataclass, field

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

    EMBEDDING_MODELS = {
        "default": os.getenv("EMBEDDING_MODEL_PATH", ""),
        "bge-small-zh": "BAAI/bge-small-zh-v1.5",
        "bge-base-zh": "BAAI/bge-base-zh-v1.5",
    }
    EMBEDDING_DIM = 384
    EMBEDDING_CACHE_SIZE = 2000

    _LOCAL_MODEL_SEARCH_PATHS = [
        Path(__file__).parent.parent / "RAG" / "paraphrase-multilingual-MiniLM-L12-v2",
        Path(__file__).parent.parent / "models" / "paraphrase-multilingual-MiniLM-L12-v2",
        Path.home() / ".cache" / "huggingface" / "hub" / "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2",
        # 国内镜像/ModelScope 缓存路径
        Path.home() / ".cache" / "modelscope" / "hub" / "iic" / "nlp_corom_sentence-embedding_chinese-base",
        Path(os.getenv("QQCHAT_LAB_ROOT", "")) / "models" / "paraphrase-multilingual-MiniLM-L12-v2",
    ]

    @classmethod
    def _find_local_embedding_model(cls) -> str:
        env_path = os.getenv("EMBEDDING_MODEL_PATH", "")
        if env_path and Path(env_path).exists():
            return env_path
        for candidate in cls._LOCAL_MODEL_SEARCH_PATHS:
            if candidate.exists() and (candidate / "config.json").exists():
                return str(candidate)
        if os.getenv("ALLOW_REMOTE_EMBEDDING_MODEL", "false").lower() in {"1", "true", "yes", "on"}:
            return "paraphrase-multilingual-MiniLM-L12-v2"
        raise FileNotFoundError(
            "Embedding model not found locally. Set EMBEDDING_MODEL_PATH or download "
            "paraphrase-multilingual-MiniLM-L12-v2; set ALLOW_REMOTE_EMBEDDING_MODEL=true "
            "only when outbound HuggingFace access is available."
        )

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
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"检测到GPU: {gpu_name}")
                return True
            return False
        except ImportError:
            logger.warning("PyTorch未安装，使用CPU")
            return False

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("正在加载向量嵌入模型...")
                self._device = "cuda" if self._use_gpu else "cpu"
                model_path = self._find_local_embedding_model()
                logger.info(f"使用嵌入模型路径: {model_path}")
                try:
                    self._model = SentenceTransformer(model_path, device=self._device)
                    logger.info(f"向量嵌入模型加载完成，设备: {self._device}")
                except (RuntimeError, Exception) as gpu_err:
                    if "CUDA out of memory" in str(gpu_err) or "out of memory" in str(gpu_err).lower():
                        logger.warning(f"GPU显存不足，回退到CPU加载嵌入模型: {gpu_err}")
                        self._device = "cpu"
                        self._model = SentenceTransformer(model_path, device="cpu")
                        logger.info(f"向量嵌入模型加载完成，设备: cpu（GPU回退）")
                    else:
                        raise
            except Exception as e:
                logger.error(f"加载模型失败: {e}")
                raise

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
            if self.index_path.exists():
                self._load_index()
            else:
                self._create_index()

    def _create_index(self, index_type: Optional[str] = None):
        idx_type = index_type or self._determine_index_type()
        dim = self.EMBEDDING_DIM

        if idx_type == "flat":
            base_index = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIDMap(base_index)
            logger.info("创建FAISS平面索引（适合小数据集）")

        elif idx_type == "ivf":
            nlist = min(self.config.nlist, max(1, int(math.sqrt(len(self.metadata)))))
            quantizer = faiss.IndexFlatIP(dim)
            base_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index = faiss.IndexIDMap(base_index)
            self.index.nprobe = self.config.nprobe
            logger.info(f"创建FAISS IVF索引（nlist={nlist}，适合中等数据集）")

        elif idx_type == "hnsw":
            base_index = faiss.IndexHNSWFlat(dim, self.config.m_hnsw, faiss.METRIC_INNER_PRODUCT)
            base_index.hnsw.efConstruction = self.config.ef_construction
            base_index.hnsw.efSearch = self.config.ef_search
            self.index = faiss.IndexIDMap(base_index)
            logger.info(f"创建FAISS HNSW索引（M={self.config.m_hnsw}，适合大数据集）")

        else:
            base_index = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIDMap(base_index)
            logger.info("创建FAISS平面索引（默认回退）")

    def _load_index(self):
        try:
            self.index = faiss.read_index(str(self.index_path))
            if self.metadata_path.exists():
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
            logger.info(f"加载FAISS索引完成，共 {len(self.metadata)} 条记录")
        except Exception as e:
            logger.error(f"加载索引失败: {e}")
            self._create_index("flat")

    def _load_bm25(self):
        if self.bm25_path.exists():
            try:
                with open(self.bm25_path, 'rb') as f:
                    bm25_data = pickle.load(f)
                self.bm25.corpus = bm25_data.get("corpus", [])
                self.bm25.doc_freqs = defaultdict(int, bm25_data.get("doc_freqs", {}))
                self.bm25.doc_lens = bm25_data.get("doc_lens", [])
                self.bm25.avgdl = bm25_data.get("avgdl", 0.0)
                self.bm25.idf = bm25_data.get("idf", {})
                self.bm25.tokenized_corpus = bm25_data.get("tokenized_corpus", [])
                self.bm25._built = bm25_data.get("built", False)
                logger.info(f"BM25状态加载完成，{len(self.bm25.corpus)} 个文档")
            except Exception as e:
                logger.warning(f"BM25状态加载失败: {e}")

    def _save_bm25(self):
        """持久化 BM25 状态，使用原子替换。失败时抛出异常，不吞错。"""
        bm25_data = {
            "corpus": self.bm25.corpus,
            "doc_freqs": dict(self.bm25.doc_freqs),
            "doc_lens": self.bm25.doc_lens,
            "avgdl": self.bm25.avgdl,
            "idf": self.bm25.idf,
            "tokenized_corpus": self.bm25.tokenized_corpus,
            "built": self.bm25._built,
        }
        tmp_path = str(self.bm25_path) + ".tmp"
        with open(tmp_path, 'wb') as f:
            pickle.dump(bm25_data, f)
        os.replace(tmp_path, str(self.bm25_path))

    def _rebuild_id_mapping(self):
        self._id_to_index = {}
        for i, meta in enumerate(self.metadata):
            doc_id = meta.get("id")
            if doc_id is not None:
                faiss_id = self._to_faiss_id(doc_id)
                if faiss_id is not None:
                    self._id_to_index[faiss_id] = i

    def _save_index(self):
        """持久化 FAISS 索引、metadata 和 BM25 状态，使用原子替换。

        失败时抛出异常（不吞错），确保调用方（如 _ensure_vector_index）
        不会在落盘失败时误标记 complete。
        """
        with self._lock:
            # FAISS: 写入临时文件后原子替换
            tmp_index = str(self.index_path) + ".tmp"
            faiss.write_index(self.index, tmp_index)
            os.replace(tmp_index, str(self.index_path))
            # metadata: 同样原子替换
            tmp_meta = str(self.metadata_path) + ".tmp"
            with open(tmp_meta, 'wb') as f:
                pickle.dump(self.metadata, f)
            os.replace(tmp_meta, str(self.metadata_path))
            # BM25
            self._save_bm25()
            self._dirty = False
            logger.info("FAISS索引和BM25状态保存完成")

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

    def _maybe_migrate_index(self, new_doc_count: int):
        current_type = self._determine_index_type()
        if current_type == "flat" and new_doc_count >= self.config.auto_switch_threshold:
            logger.info(f"文档数({new_doc_count})超过阈值({self.config.auto_switch_threshold})，迁移到IVF索引")
            self._migrate_to_ivf()
        elif current_type == "ivf" and new_doc_count >= 100000:
            logger.info(f"文档数({new_doc_count})超过10万，迁移到HNSW索引")
            self._migrate_to_hnsw()

    def _migrate_to_ivf(self):
        if len(self.metadata) == 0:
            return
        old_index = self.index
        old_metadata = self.metadata
        old_id_to_index = self._id_to_index
        try:
            documents = self.metadata.copy()
            self.index = None
            self.metadata = []
            self._id_to_index = {}
            self._create_index("ivf")
            self._re_add_all_documents(documents)
        except Exception:
            self.index = old_index
            self.metadata = old_metadata
            self._id_to_index = old_id_to_index
            logger.error("IVF索引迁移失败，已恢复旧索引")
            raise

    def _migrate_to_hnsw(self):
        if len(self.metadata) == 0:
            return
        old_index = self.index
        old_metadata = self.metadata
        old_id_to_index = self._id_to_index
        try:
            documents = self.metadata.copy()
            self.index = None
            self.metadata = []
            self._id_to_index = {}
            self._create_index("hnsw")
            self._re_add_all_documents(documents)
        except Exception:
            self.index = old_index
            self.metadata = old_metadata
            self._id_to_index = old_id_to_index
            logger.error("HNSW索引迁移失败，已恢复旧索引")
            raise

    def _re_add_all_documents(self, documents: List[Dict[str, Any]]):
        texts = [f"{doc.get('title', '')} {doc.get('content', '')}" for doc in documents]
        embeddings = self._get_embeddings_batch(texts)

        ids = []
        for i, doc in enumerate(documents):
            doc_id = doc.get("id")
            faiss_id = self._to_faiss_id(doc_id) if doc_id is not None else i
            if faiss_id is None:
                faiss_id = i
            ids.append(faiss_id)

        ids = np.array(ids, dtype=np.int64)

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
            self._load_model()

            texts = []
            for doc in documents:
                full_text = f"{doc.get('title', '')} {doc.get('content', '')}"
                texts.append(full_text)

            logger.info(f"正在生成 {len(texts)} 个向量...")
            embeddings = self._get_embeddings_batch(texts)

            ids = []
            start_id = len(self.metadata)
            for i, doc in enumerate(documents):
                doc_id = doc.get("id")
                faiss_id = None
                if doc_id is not None:
                    faiss_id = self._to_faiss_id(doc_id)
                if faiss_id is None:
                    faiss_id = start_id + i
                ids.append(faiss_id)

            ids = np.array(ids, dtype=np.int64)

            if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                logger.info("训练IVF索引...")
                train_size = min(len(embeddings), max(256, self.config.nlist * 39))
                self.index.train(embeddings[:train_size])

            self.index.add_with_ids(embeddings, ids)

            start_idx = len(self.metadata)
            self.metadata.extend(documents)
            for i, faiss_id in enumerate(ids):
                self._id_to_index[int(faiss_id)] = start_idx + i

            self.bm25.add_documents(documents)

            self._add_count += len(documents)
            self._dirty = True

            if self.config.save_on_every_n_adds <= 0 or self._add_count >= self.config.save_on_every_n_adds:
                self._save_index()
                self._add_count = 0
            else:
                logger.info(f"延迟保存：已添加 {self._add_count} 个文档（阈值: {self.config.save_on_every_n_adds}）")

            self._maybe_migrate_index(len(self.metadata))

            # 数据变更后清除查询缓存，防止脏读过期结果
            self.clear_cache()

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
        self._ensure_index()
        if len(self.metadata) == 0:
            return []

        # 查询缓存检查
        cache_key = self._get_query_cache_key(query, top_k, threshold, filters)
        cached = self._get_cached_query(cache_key)
        if cached is not None:
            logger.debug(f"查询缓存命中: query='{query[:30]}...'")
            return cached

        with self._lock:
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

            return results

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

    def _find_metadata_index(self, doc: Dict[str, Any]) -> Optional[int]:
        doc_id = doc.get("id")
        if doc_id is not None:
            faiss_id = self._to_faiss_id(doc_id)
            if faiss_id is not None and faiss_id in self._id_to_index:
                return self._id_to_index[faiss_id]
        doc_content = doc.get("content", "")
        doc_title = doc.get("title", "")
        for i, meta in enumerate(self.metadata):
            if meta.get("content") == doc_content and meta.get("title") == doc_title:
                return i
        # Use content hash as stable fallback key
        content_hash = hashlib.md5(str(doc_content).encode()).hexdigest()[:8]
        for i, meta in enumerate(self.metadata):
            meta_hash = hashlib.md5(str(meta.get("content", "")).encode()).hexdigest()[:8]
            if meta_hash == content_hash:
                return i
        return None

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
        recall_k = min(top_k * 3, len(self.metadata))
        with self._lock:
            vector_results = self.search(query, top_k=recall_k, threshold=threshold, filters=filters)

            if not vector_results and not self.bm25._built:
                return []

            bm25_results = []
            if self.bm25._built:
                bm25_hits = self.bm25.search(query, top_k=recall_k, threshold=0.0)
                for doc_idx, score in bm25_hits:
                    if doc_idx < len(self.metadata):
                        result = {**self.metadata[doc_idx], "bm25_score": score}
                        if filters and not self._match_filters(result, filters):
                            continue
                        bm25_results.append(result)

            if not vector_results and not bm25_results:
                return []

            doc_scores: Dict[int, Dict[str, float]] = {}

            for doc in vector_results:
                key = self._find_metadata_index(doc)
                if key is None:
                    continue
                doc_scores[key] = {"vector_score": doc.get("score", 0), "bm25_score": 0.0, "doc": doc}

            for doc in bm25_results:
                key = self._find_metadata_index(doc)
                if key is None:
                    continue
                if key in doc_scores:
                    doc_scores[key]["bm25_score"] = doc.get("bm25_score", 0)
                else:
                    doc_scores[key] = {"vector_score": 0.0, "bm25_score": doc.get("bm25_score", 0), "doc": doc}

            max_bm25 = max((s["bm25_score"] for s in doc_scores.values()), default=1.0) or 1.0
            max_vector = max((s["vector_score"] for s in doc_scores.values()), default=1.0) or 1.0

            fused_results = []
            for key, scores in doc_scores.items():
                norm_vector = scores["vector_score"] / max_vector if max_vector > 0 else 0
                norm_bm25 = scores["bm25_score"] / max_bm25 if max_bm25 > 0 else 0
                fused_score = (1 - keyword_weight) * norm_vector + keyword_weight * norm_bm25

                doc = scores["doc"]
                doc["vector_score"] = scores["vector_score"]
                doc["bm25_score"] = scores["bm25_score"]
                doc["fused_score"] = fused_score
                fused_results.append(doc)

            fused_results.sort(key=lambda x: x["fused_score"], reverse=True)
            return fused_results[:top_k]

    def batch_search(
        self,
        queries: List[str],
        top_k: int = 5,
        threshold: float = 0.15,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[List[Dict[str, Any]]]:
        self._ensure_index()
        if len(self.metadata) == 0:
            return [[] for _ in queries]

        self._load_model()
        query_embeddings = self._get_embeddings_batch(queries)

        with self._lock:
            results = []
            for i, query_emb in enumerate(query_embeddings):
                query_emb = np.array([query_emb]).astype('float32')
                scores, faiss_ids = self.index.search(query_emb, top_k)

                query_results = []
                for score, faiss_id in zip(scores[0], faiss_ids[0]):
                    if score < threshold:
                        continue
                    meta_idx = self._id_to_index.get(int(faiss_id))
                    if meta_idx is None or meta_idx < 0 or meta_idx >= len(self.metadata):
                        continue
                    result = {**self.metadata[meta_idx], "score": float(score)}
                    if filters and not self._match_filters(result, filters):
                        continue
                    query_results.append(result)

                results.append(query_results)

            return results

    def delete_documents(self, doc_ids: List[Any]) -> int:
        if not doc_ids:
            return 0

        with self._lock:
            faiss_ids_to_remove = []
            indices_to_remove = set()

            for doc_id in doc_ids:
                faiss_id = self._to_faiss_id(doc_id)
                if faiss_id is not None and faiss_id in self._id_to_index:
                    meta_idx = self._id_to_index[faiss_id]
                    indices_to_remove.add(meta_idx)
                    faiss_ids_to_remove.append(faiss_id)

            if not indices_to_remove:
                return 0

            try:
                faiss_ids_array = np.array(faiss_ids_to_remove, dtype=np.int64)
                if hasattr(self.index, 'remove_ids'):
                    n_removed = self.index.remove_ids(faiss_ids_array)
                    logger.info(f"从FAISS索引中删除 {n_removed} 个向量")
                else:
                    logger.warning("当前索引不支持remove_ids，需要重建索引")
                    self._rebuild_index_excluding(indices_to_remove)
                    return len(indices_to_remove)

                sorted_indices = sorted(indices_to_remove, reverse=True)
                for idx in sorted_indices:
                    if idx < len(self.metadata):
                        self.metadata.pop(idx)

                self._rebuild_id_mapping()
                self._rebuild_bm25()
                self._save_index()

                # 数据变更后清除查询缓存
                self.clear_cache()

                return len(indices_to_remove)

            except Exception as e:
                logger.error(f"删除文档失败: {e}")
                return 0

    def _rebuild_index_excluding(self, exclude_indices: Set[int]):
        documents = [meta for i, meta in enumerate(self.metadata) if i not in exclude_indices]
        self.index = None
        self.metadata = []
        self._id_to_index = {}
        self._create_index()
        if documents:
            self._re_add_all_documents(documents)

    def _rebuild_bm25(self):
        self.bm25 = BM25Retriever()
        self.bm25.add_documents(self.metadata)

    def rebuild_index(self):
        """重建整个向量索引：清空现有索引后重新添加所有文档。

        用于索引损坏后恢复或强制切换索引类型。
        """
        if len(self.metadata) == 0:
            return
        with self._lock:
            documents = self.metadata.copy()
            self.index = None
            self.metadata = []
            self._id_to_index = {}
            self._create_index()
        self.add_documents(documents)

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
            if self.index:
                if isinstance(self.index, faiss.IndexIDMap):
                    inner = self.index.index
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
