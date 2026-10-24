"""
RAG意图检测模块
负责智能分析用户消息，判断是否需要触发RAG知识库检索。

训练好的 ML 分类器严格加载并保留自身判定；不可用时直接报错。
主聊天使用独立语义规划；本模块只负责显式使用的已训练分类器。
"""

import os
import json
import logging
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# 全局 ML 模型缓存
_ml_encoder = None
_ml_classifier = None
_ml_config = None
_ML_AVAILABLE = False


def _set_ml_model(artifacts):
    """Publish or invalidate one complete in-process classifier state."""
    global _ml_encoder, _ml_classifier, _ml_config, _ML_AVAILABLE
    _ml_encoder, _ml_classifier, _ml_config = artifacts if artifacts is not None else (None, None, None)
    _ML_AVAILABLE = artifacts is not None


def _read_ml_model(model_dir=None):
    """Read one artifact set without changing availability or routing state."""
    if model_dir is None:
        configured = os.getenv("INTENT_MODEL_PATH", "").strip()
        model_dir = Path(configured).expanduser() if configured else Path("intent_classifier_model")
    model_dir = Path(model_dir)
    if not model_dir.is_absolute():
        model_dir = Path(__file__).parent.parent / model_dir

    import joblib
    from sentence_transformers import SentenceTransformer

    with (model_dir / "config.json").open(encoding="utf-8") as source:
        config = json.load(source)
    if not isinstance(config, dict) or not isinstance(config.get("embedding_model"), str) or not config["embedding_model"].strip():
        raise ValueError("Intent classifier config requires embedding_model")
    classifier = joblib.load(model_dir / "classifier.joblib")
    scaler = joblib.load(model_dir / "scaler.joblib")
    if not callable(getattr(classifier, "predict_proba", None)) or not callable(getattr(scaler, "transform", None)):
        raise TypeError("Invalid intent classifier or scaler artifact")
    encoder = SentenceTransformer(config["embedding_model"])
    config["_scaler"] = scaler

    return encoder, classifier, config


def _load_ml_model(model_dir=None):
    """An explicit directory always reloads the requested artifact set."""
    if model_dir is None and _ML_AVAILABLE:
        return
    _set_ml_model(None)
    _set_ml_model(_read_ml_model(model_dir))
    logger.info("Intent classifier artifact set loaded")


def _ml_predict(message: str) -> Tuple[bool, str, float, Optional[str]]:
    """多分类预测：返回 (是否需要RAG, 原因, 置信度, 预测的KB名称)

    使用多分类模型预测消息应路由到哪个知识库。
    - 预测类别为 "none" 时不需要RAG
    - 预测类别为某个KB名称时需要RAG，并返回该KB名称供RAG过滤使用
    """
    global _ml_encoder, _ml_classifier, _ml_config
    threshold = _ml_config.get("threshold", 0.5)
    embedding = _ml_encoder.encode([message])
    scaler = _ml_config.get("_scaler")
    if scaler is not None:
        embedding = scaler.transform(embedding)

    # 多分类预测：取概率最高的类别
    proba = _ml_classifier.predict_proba(embedding)[0]
    label_names = _ml_config.get("label_names", [])
    if not label_names:
        # 兼容旧版二分类模型
        score = proba[1] if len(proba) > 1 else proba[0]
        pred = score >= threshold
        return pred, f"ML分类器(旧版二分类, 置信度: {score:.2%})", score, None

    best_idx = int(proba.argmax())
    best_label = label_names[best_idx]
    best_score = float(proba[best_idx])

    # 预测为 "none" 或置信度不足时不需要RAG
    if best_label == "none":
        return False, f"ML分类器: 不需要RAG (置信度: {best_score:.2%})", best_score, None

    if best_score < threshold:
        return False, f"ML分类器: 置信度不足 ({best_score:.2%} < {threshold})", best_score, None

    return True, f"ML分类器: 路由到「{best_label}」(置信度: {best_score:.2%})", best_score, best_label



def predict_rag_intent(message: str) -> Tuple[bool, float, Optional[str]]:
    """Return the trained classifier score and knowledge target for routing."""
    _load_ml_model()
    pred, _reason, confidence, kb_name = _ml_predict(message)
    return pred, confidence, kb_name
