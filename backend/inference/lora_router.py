"""离线多 LoRA 路由研究基线。

在线聊天明确选择角色，路由决策仅用于研究实验。
intent 模式依赖已训练分类器；依赖或配置失败直接报错。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class RouteTarget(str, Enum):
    """路由目标类型。"""
    BASE_CHAT = "base_chat"
    RAG_REQUIRED = "rag_required"
    PERSONA_ADAPTER = "persona_adapter"


@dataclass
class RoutingDecision:
    """路由决策结果。"""
    target: str  # RouteTarget value
    adapter_name: str = "default"
    confidence: float = 0.0
    reason: str = ""
    fallback: bool = False
    requires_rag: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# 默认角色关键词映射
DEFAULT_PERSONA_KEYWORDS: Dict[str, List[str]] = {
    "kisaki": ["月社妃", "妃", "琉璃", "夜子", "彼方", "理央", "纸上的魔法使"],
    "minamo": ["神白水菜萌", "水菜萌", "Minamo", "ATRI"],
    "hutao": ["胡桃", "往生堂", "堂主", "蝶引来生"],
    "zhongli": ["钟离", "岩王帝君", "摩拉克斯"],
    "qiqi": ["七七", "不卜庐", "僵尸"],
}

DEFAULT_PERSONA_ADAPTERS: Dict[str, str] = {
    "kisaki": "kisaki",
    "minamo": "minamo",
    "hutao": "hutao",
    "zhongli": "zhongli",
    "qiqi": "qiqi",
}


class LoRARouter:
    """多 LoRA 路由器。

    根据用户查询内容选择路由目标：
    - 含知识/百科类问题 → RAG_REQUIRED
    - 含角色关键词 → PERSONA_ADAPTER（选择对应适配器）
    - 其他 → BASE_CHAT
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.enabled = config.get("enabled", False)
        self.default_adapter = config.get("default_adapter", "default")
        self.mode = config.get("mode", "intent")
        self.persona_adapters: Dict[str, str] = config.get("persona_adapters", DEFAULT_PERSONA_ADAPTERS)
        self.rag_confidence_threshold = config.get("rag_confidence_threshold", 0.5)
        self.persona_keywords: Dict[str, List[str]] = config.get(
            "persona_keywords", DEFAULT_PERSONA_KEYWORDS
        )
        self._intent_detector = None
        self._routing_logs: List[Dict[str, Any]] = []
        self._max_logs = 500

    def _get_intent_detector(self):
        """延迟加载意图检测器。"""
        if self._intent_detector is None:
            from knowledge.intent_detector import predict_rag_intent
            self._intent_detector = predict_rag_intent
        return self._intent_detector

    def route(self, query: str, intent_result: Optional[tuple] = None) -> RoutingDecision:
        """路由查询到合适的适配器。

        Args:
            query: 用户查询文本
            intent_result: 预计算的意图检测结果 (need_rag, confidence, kb_name)

        Returns:
            RoutingDecision
        """
        if not self.enabled or self.mode == "manual":
            return RoutingDecision(
                target=RouteTarget.BASE_CHAT.value,
                adapter_name=self.default_adapter,
                confidence=1.0,
                reason="路由器未启用，使用默认适配器",
                fallback=True,
            )

        # 1. 检查是否需要 RAG（知识检索）
        need_rag = False
        rag_confidence = 0.0
        if self.mode != "rule":
            detected = intent_result if intent_result is not None else self._get_intent_detector()(query)
            need_rag, rag_confidence, _kb_name = detected
            need_rag = bool(need_rag)
            rag_confidence = float(rag_confidence)

        # 2. 检查角色关键词匹配
        best_persona = None
        best_match_count = 0
        for persona, keywords in self.persona_keywords.items():
            match_count = sum(1 for kw in keywords if kw in query)
            if match_count > best_match_count:
                best_match_count = match_count
                best_persona = persona

        if best_persona and best_match_count > 0:
            confidence = min(1.0, best_match_count / 3.0)
            adapter_name = self.persona_adapters[best_persona]
            # 验证适配器名称安全性
            if not adapter_name or not all(c.isalnum() or c in "_-" for c in adapter_name):
                raise ValueError("Invalid configured persona adapter name")
            return RoutingDecision(
                target=RouteTarget.PERSONA_ADAPTER.value,
                adapter_name=adapter_name,
                confidence=confidence,
                reason=f"匹配角色 {best_persona} ({best_match_count} 个关键词)",
                requires_rag=need_rag and rag_confidence >= self.rag_confidence_threshold,
            )

        if need_rag and rag_confidence >= self.rag_confidence_threshold:
            return RoutingDecision(
                target=RouteTarget.RAG_REQUIRED.value,
                adapter_name=self.default_adapter,
                confidence=rag_confidence,
                reason=f"意图检测需要 RAG (confidence={rag_confidence:.2f})",
                requires_rag=True,
            )
        # 3. 默认基础聊天
        return RoutingDecision(
            target=RouteTarget.BASE_CHAT.value,
            adapter_name=self.default_adapter,
            confidence=0.5,
            reason="未匹配 RAG 或角色关键词，使用基础聊天",
        )

    def log_routing(self, decision: RoutingDecision, trace_id: str = ""):
        """记录路由日志（Python logger + 内存存储）。"""
        from datetime import datetime, timezone
        logger.info(
            f"LoRA路由: target={decision.target}, adapter={decision.adapter_name}, "
            f"confidence={decision.confidence:.2f}, reason={decision.reason}, traceId={trace_id}"
        )
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trace_id": trace_id,
            **decision.to_dict(),
        }
        self._routing_logs.append(entry)
        if len(self._routing_logs) > self._max_logs:
            self._routing_logs = self._routing_logs[-self._max_logs:]

    def get_routing_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """返回最近的路由日志（按时间倒序）。"""
        return list(reversed(self._routing_logs[-limit:]))


# 默认配置
DEFAULT_ROUTER_CONFIG = {
    "enabled": os.getenv("LORA_ROUTER_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"},
    "default_adapter": "default",
    "mode": os.getenv("LORA_ROUTER_MODE", "intent"),
    "persona_adapters": DEFAULT_PERSONA_ADAPTERS,
    "rag_confidence_threshold": 0.5,
    "persona_keywords": DEFAULT_PERSONA_KEYWORDS,
}

_router: Optional[LoRARouter] = None
_router_config_hash = ""


def get_lora_router(config: Optional[Dict[str, Any]] = None) -> LoRARouter:
    """Return the singleton and refresh it only when persisted config changes."""
    global _router, _router_config_hash
    effective = config or DEFAULT_ROUTER_CONFIG
    config_hash = hashlib.sha256(
        json.dumps(effective, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if _router is None or config_hash != _router_config_hash:
        previous_logs = _router._routing_logs if _router is not None else []
        _router = LoRARouter(effective)
        _router._routing_logs = previous_logs[-_router._max_logs:]
        _router_config_hash = config_hash
    return _router
