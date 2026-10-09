"""
Cross-Encoder重排模块 - 优化版
基于BGE-Reranker模型的文档精排模块，作为RAG检索的第二阶段，
对粗排召回的候选文档进行精确的相关性打分和排序。
改进：分数归一化、模型预热、批量优化、GPU显存管理、
默认离线加载（本地模型目录优先，未显式开启下载时不联网）、输入候选保护
"""

import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).parent


def _resolve_path(p: str) -> str:
    if os.path.isabs(p):
        return p
    return str(_BACKEND_DIR / p)


def _env_flag(name: str, default: str = "false") -> bool:
    """读取布尔型环境变量（支持1/true/yes/on）。"""
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class RerankConfig:
    model_name: str = os.getenv("RERANKER_MODEL_PATH", _resolve_path("bge-reranker-base"))
    device: str = "cuda:0"
    batch_size: int = 8
    max_length: int = 512
    enable_quantization: bool = False
    warmup_on_init: bool = False
    score_normalize: bool = True
    # 默认离线：仅当显式设置 RERANKER_ALLOW_DOWNLOAD=true 时才允许联网下载
    allow_download: bool = field(default_factory=lambda: _env_flag("RERANKER_ALLOW_DOWNLOAD"))

    def __post_init__(self):
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError(f"RerankConfig.batch_size 必须为正整数，当前值: {self.batch_size!r}")
        if isinstance(self.max_length, bool) or not isinstance(self.max_length, int) or self.max_length < 1:
            raise ValueError(f"RerankConfig.max_length 必须为正整数，当前值: {self.max_length!r}")


class CrossEncoderReranker:
    """Cross-Encoder重排器，使用BGE-Reranker-Base模型对候选文档进行精细化相关性评分。

    支持4bit量化、模型预热、GPU/CPU自动切换、分数归一化。
    默认离线加载本地模型目录；加载或推理失败直接抛出异常。
    """

    def __init__(self, config: RerankConfig | None = None):
        """初始化重排器。

        Args:
            config: 重排配置，默认使用RerankConfig()
        """
        self.config = config or RerankConfig()
        self.device = self.config.device
        self.tokenizer = None
        self.model = None
        self._model_loaded = False
        self._warmup_done = False

        # GPU不可用时立即回退CPU，避免后续加载/推理走CUDA专属路径
        if not self._check_gpu_available():
            if self.device != "cpu":
                logger.warning(f"GPU不可用，设备从 {self.device} 回退到CPU")
            self.device = "cpu"

        logger.info(f"Cross-Encoder重排器初始化完成，设备: {self.device}")

    def _check_gpu_available(self) -> bool:
        try:
            if not torch.cuda.is_available():
                return False
            torch.cuda.device_count()
            return True
        except Exception:
            return False

    def _candidate_model_paths(self) -> list[str]:
        """返回去重后的候选模型路径（原始值 + 相对backend/knowledge目录解析的绝对路径）。"""
        paths: list[str] = []
        seen = set()
        for p in (self.config.model_name, _resolve_path(self.config.model_name)):
            if not p:
                continue
            key = os.path.normcase(os.path.normpath(os.path.abspath(p)))
            if key in seen:
                continue
            seen.add(key)
            paths.append(p)
        return paths

    def _load_model(self) -> bool:
        if self._model_loaded:
            return True
        try:
            logger.info(
                f"正在加载Cross-Encoder模型: {self.config.model_name} (允许联网下载={self.config.allow_download})"
            )

            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            start_time = time.time()

            # 默认离线加载：local_files_only=True保证不会静默联网下载
            load_kwargs: dict[str, Any] = {"local_files_only": not self.config.allow_download}
            if self.device != "cpu":
                # CPU路径不使用CUDA专属dtype
                load_kwargs["torch_dtype"] = torch.float16

            last_error: Exception | None = None
            for path in self._candidate_model_paths():
                if not Path(path).exists() and not self.config.allow_download:
                    logger.warning(f"本地模型路径不存在（离线模式，不联网下载）: {path}")
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(path, **load_kwargs)
                    self.model = AutoModelForSequenceClassification.from_pretrained(path, **load_kwargs)
                    break
                except Exception as e:
                    last_error = e
                    self.tokenizer = None
                    self.model = None

            if self.model is None or self.tokenizer is None:
                raise RuntimeError("CrossEncoder 模型加载失败，请检查 RERANKER_MODEL_PATH 和本地模型文件") from last_error

            self.model.to(self.device)
            self.model.eval()

            for param in self.model.parameters():
                param.requires_grad = False

            load_time = time.time() - start_time

            model_size = sum(p.numel() for p in self.model.parameters())
            model_mem = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024**2
            logger.info(
                f"Cross-Encoder模型加载完成: "
                f"参数量={model_size:,}, "
                f"模型大小={model_mem:.1f}MB, "
                f"设备={self.device}, "
                f"加载时间={load_time:.2f}s"
            )

            if self.config.warmup_on_init:
                self._warmup()

            self._model_loaded = True
            return True

        except Exception:
            self.tokenizer = None
            self.model = None
            raise

    def _warmup(self):
        if self._warmup_done:
            return

        logger.info("Cross-Encoder模型预热中...")
        dummy_input = self.tokenizer(
            "预热查询",
            "预热文档内容",
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            _ = self.model(**dummy_input)

        if self.device != "cpu":
            torch.cuda.empty_cache()

        self._warmup_done = True
        logger.info("Cross-Encoder模型预热完成")


    def _normalize_scores(self, scores: list[float]) -> list[float]:
        if not scores or not self.config.score_normalize:
            return scores

        import numpy as np

        scores_arr = np.array(scores)

        min_s = scores_arr.min()
        max_s = scores_arr.max()
        score_range = max_s - min_s

        if score_range < 1e-6:
            return [1.0 if s > 0 else 0.0 for s in scores]

        normalized = (scores_arr - min_s) / score_range
        return normalized.tolist()

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        """对候选文档列表进行Cross-Encoder精确相关性重排。

        将查询与每个候选文档配对输入模型打分，支持批量推理以提升效率。
        默认离线加载本地模型目录，加载或推理失败直接抛出异常。

        不修改调用方传入的candidate字典：返回复制后的候选副本，
        并在副本上附加rerank_score与rerank_normalized_score字段。
        空候选直接返回；非空候选必须具有有效文本，并全部获得模型分数。

        Args:
            query: 用户查询文本
            candidates: 候选文档列表，每个文档需包含content字段
            top_k: 返回的文档数量，必须为正整数

        Returns:
            按模型分数降序排列的前top_k个文档副本，
            每项额外包含rerank_score和rerank_normalized_score

        Raises:
            ValueError: top_k不是正整数时
        """
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError(f"top_k 必须为正整数，当前值: {top_k!r}")

        if not candidates:
            return []
        candidate_texts = []
        for candidate in candidates:
            if not isinstance(candidate, dict) or not isinstance(candidate.get("content"), str) or not candidate["content"].strip():
                raise ValueError("CrossEncoder candidate content must be non-empty text")
            candidate_texts.append(candidate["content"])

        self._load_model()

        start_time = time.time()

        scores = []
        batch_size = self.config.batch_size

        for i in range(0, len(candidates), batch_size):
            batch_end = min(i + batch_size, len(candidates))
            batch_texts = candidate_texts[i:batch_end]

            inputs = self.tokenizer(
                [query] * len(batch_texts),
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_scores = outputs.logits[:, 0].cpu().tolist()
                scores.extend(batch_scores)

            if self.device != "cpu" and i + batch_size < len(candidates):
                torch.cuda.empty_cache()

        if len(scores) != len(candidates) or not all(math.isfinite(score) for score in scores):
            raise RuntimeError("CrossEncoder returned incomplete or non-finite scores")
        normalized_scores = self._normalize_scores(scores)

        sorted_pairs = sorted(zip(candidates, scores, normalized_scores), key=lambda x: x[1], reverse=True)

        reranked = []
        for doc, raw_score, norm_score in sorted_pairs[:top_k]:
            # 复制候选字典，不原地修改调用方传入的原始数据
            doc_copy = dict(doc)
            doc_copy["rerank_score"] = raw_score
            doc_copy["rerank_normalized_score"] = norm_score
            reranked.append(doc_copy)

        process_time = time.time() - start_time
        logger.info(f"重排完成: {len(candidates)} -> {len(reranked)} 文档, 耗时={process_time:.3f}s")

        return reranked



_reranker_instance: CrossEncoderReranker | None = None


def get_reranker(config: RerankConfig | None = None) -> CrossEncoderReranker:
    """获取Cross-Encoder重排器全局单例。

    首次调用时自动初始化，后续调用返回同一实例。

    Args:
        config: 可选的配置，仅在首次初始化时生效

    Returns:
        CrossEncoderReranker: 全局唯一的重排器实例
    """
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker(config)
    return _reranker_instance


def rerank_documents(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    reranker = get_reranker()
    if os.getenv("RERANKER_ENABLED", "false").strip().lower() not in {"1", "true", "yes", "on"}:
        return candidates[:top_k]

    return reranker.rerank(query, candidates, top_k)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Cross-Encoder重排器测试（优化版）")
    print("=" * 60)

    test_query = "原神是什么类型的游戏？"
    test_candidates = [
        {"title": "游戏介绍", "content": "原神是一款开放世界动作角色扮演游戏，由米哈游开发。"},
        {"title": "游戏类型", "content": "原神属于ARPG类型，包含探索、战斗、解谜等元素。"},
        {"title": "角色系统", "content": "玩家可以收集和使用各种角色进行战斗。"},
        {"title": "开放世界", "content": "游戏拥有广阔的世界供玩家探索。"},
        {"title": "多人游戏", "content": "原神支持多人联机合作游戏。"},
    ]

    config = RerankConfig(model_name="./bge-reranker-base")
    reranker = CrossEncoderReranker(config)

    print(f"查询: {test_query}")
    print(f"候选文档数量: {len(test_candidates)}")
    print()

    reranked = reranker.rerank(test_query, test_candidates, top_k=3)

    print("重排结果（前3个）:")
    for i, doc in enumerate(reranked, 1):
        title = doc.get("title", "无标题")
        content = doc.get("content", "")[:100]
        raw_score = doc.get("rerank_score", 0)
        norm_score = doc.get("rerank_normalized_score", 0)
        print(f"{i}. {title}: raw={raw_score:.4f}, normalized={norm_score:.4f}")
        print(f"   {content}...")

    print("=" * 60)
    print("测试完成")
