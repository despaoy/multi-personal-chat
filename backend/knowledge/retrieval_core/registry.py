"""Shared knowledge-domain registry.

知识域（namespace）通过 KnowledgeDomainConfig 声明：
- domain_id 与数据源位置
- 文档类型集合
- loader（把源数据转成 canonical 文档）
- 别名表（alias → 规范名，服务查询归一与实体命中）
- 叙事层取值与默认权重策略
- 默认检索策略与索引版本
- 是否启用

通用检索代码不写死任何作品内容；作品专属配置只存在于
domains/ 子包的数据配置中，通过 register_defaults() 注册。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .documents import KnowledgeIndexDocument

# Loader 协议：source_root → canonical 文档列表。
# 领域 loader 只做数据转换，不读环境密钥、不联网、不初始化模型。
SourceLoader = Callable[[Path], list[KnowledgeIndexDocument]]


@dataclass
class NarrativeLayerPolicy:
    """叙事层通用权重策略（domain 可覆盖取值集合与权重）。

    核心代码只读取"取值 → 权重"映射；objective 优先等规则由
    domain 配置给出，避免把作品叙事规则写死在核心层。
    """

    reality_boost: dict[str, float] = field(default_factory=lambda: {"objective": 1.0})
    temporal_boost: dict[str, float] = field(default_factory=lambda: {"current": 1.0})
    scope_boost: dict[str, float] = field(default_factory=lambda: {"main_story": 1.0})


@dataclass
class RetrievalDefaults:
    """domain 默认检索策略。"""

    top_k: int = 5
    recall_k: int = 20
    rrf_k: int = 60
    # 精确实体命中加权（通用，全局可调）
    entity_boost: float = 0.02
    entity_boost_cap: float = 0.03
    # 叙事层权重乘数区间（温和范围：压过相邻名次但不压制整页结果；
    # 防止某层完全压制其他层，多层结果共存）
    layer_boost_range: tuple[float, float] = (0.85, 1.15)


@dataclass
class KnowledgeDomainConfig:
    """单个知识域的注册配置。"""

    domain_id: str
    source_root: Path
    loader: SourceLoader
    document_types: list[str] = field(default_factory=lambda: ["fact", "relation", "event"])
    # 别名表：alias（含规范名本身）→ 规范名
    aliases: dict[str, str] = field(default_factory=dict)
    # 故事/卷名等标题词（查询理解时识别 story 命中）
    story_titles: list[str] = field(default_factory=list)
    narrative_policy: NarrativeLayerPolicy = field(default_factory=NarrativeLayerPolicy)
    retrieval_defaults: RetrievalDefaults = field(default_factory=RetrievalDefaults)
    # 域专属回答表达规则（如层级的固定说法）只能放这里，
    # 核心回答代码不得写死任何作品内容；缺省为空（无补充规则）
    prompt_supplement: str = ""
    index_version: str = "v1"
    enabled: bool = True
    display_name: str = ""
    description: str = ""
    # 索引产物目录；缺省为 source_root 同级的 rag_index/
    index_root: Path | None = None

    def resolve_index_root(self) -> Path:
        if self.index_root is not None:
            return Path(self.index_root)
        return self.source_root.parent / "rag_index"

    def canonical_entity(self, token: str) -> str | None:
        """把查询中的别名/规范名映射到规范实体名。"""
        return self.aliases.get(token)

    def known_entities(self) -> list[str]:
        return sorted(set(self.aliases.values()))


class DomainRegistry:
    """知识域注册表：新增域 = 注册一个 KnowledgeDomainConfig。"""

    def __init__(self) -> None:
        self._domains: dict[str, KnowledgeDomainConfig] = {}

    def register(self, config: KnowledgeDomainConfig) -> None:
        if not config.domain_id or not config.domain_id.strip():
            raise ValueError("domain_id 不能为空")
        if config.domain_id in self._domains:
            raise ValueError(f"知识域已注册: {config.domain_id}")
        self._domains[config.domain_id] = config

    def get(self, domain_id: str) -> KnowledgeDomainConfig | None:
        return self._domains.get(domain_id)

    def require(self, domain_id: str) -> KnowledgeDomainConfig:
        config = self._domains.get(domain_id)
        if config is None:
            raise KeyError(f"未注册的知识域: {domain_id}")
        return config

    def list_domains(self, enabled_only: bool = True) -> list[KnowledgeDomainConfig]:
        return [cfg for cfg in self._domains.values() if (cfg.enabled or not enabled_only)]

    def domain_ids(self, enabled_only: bool = True) -> list[str]:
        return [cfg.domain_id for cfg in self.list_domains(enabled_only)]

    def __len__(self) -> int:
        return len(self._domains)

    def snapshot(self) -> dict[str, Any]:
        """注册表摘要（用于日志与诊断）。"""
        return {
            cfg.domain_id: {
                "enabled": cfg.enabled,
                "source_root": str(cfg.source_root),
                "document_types": list(cfg.document_types),
                "index_version": cfg.index_version,
                "alias_count": len(cfg.aliases),
            }
            for cfg in self._domains.values()
        }


_default_registry: DomainRegistry | None = None


def build_default_registry() -> DomainRegistry:
    """构建默认注册表：注册所有内置 domain 配置。

    每个 domain 的专属配置从 domains.<name> 模块导入，注册表本身
    不包含任何作品词表。
    """
    registry = DomainRegistry()
    from .domains import builtin_domain_factories

    for factory in builtin_domain_factories():
        registry.register(factory())
    return registry


def get_default_registry() -> DomainRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = build_default_registry()
    return _default_registry
