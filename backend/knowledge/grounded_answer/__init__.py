"""P7 grounded-answer 系统：检索结果 → 有证据约束的回答 → 引用返回。

分层：
- models      数据契约（EvidencePacket / AnswerMode / GroundedAnswerResult）
- packet      P6 bundle → 证据包（citation key 分配）
- modes       回答模式识别与 abstention 策略（确定性规则）
- prompt      grounded prompt 契约（persona 分层 + 不可信证据边界）
- validator   citation 解析/校验/绑定 + 轻量回答后检查
- corrective  P6 纠正性检索适配（改写 → 重试 → 合并）
- cache       有限回答缓存（绑定 index/prompt/model 版本）
- service     编排入口（非流式 / 流式 / bundle 复用）
"""

from .models import (
    AnswerMode,
    AnswerStreamEvent,
    AnswerTimings,
    EvidenceItem,
    EvidencePacket,
    FailureKind,
    GroundedAnswerResult,
)
from .packet import EvidencePacketBuilder, is_p6_bundle

__all__ = [
    "AnswerMode",
    "AnswerStreamEvent",
    "AnswerTimings",
    "EvidenceItem",
    "EvidencePacket",
    "EvidencePacketBuilder",
    "FailureKind",
    "GroundedAnswerResult",
    "is_p6_bundle",
]
