"""Shared canonical knowledge-document contract.

Canonical Document：所有知识域的源数据（approved 卡、角色记忆、
用户文档等）经 loader 转换为本模型的实例后再进入 embedding、
索引与检索层。通用检索代码只依赖本契约，不依赖任何作品专属字段；
作品专属内容（人物别名、卷名、叙事层级词表）全部收敛在对应
domain 配置（registry + domains/*）中。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


def _fingerprint(text: str) -> str:
    """确定性文本指纹（embedding 缓存与增量构建的比对依据）。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class SourceReference:
    """来源引用：文件、行号、卡片 ID 等原始定位信息。"""

    source_path: str = ""
    line_start: int | None = None
    line_end: int | None = None
    card_id: str = ""
    # 额外定位信息（如 story_unit_id、场景 ID 等）
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "card_id": self.card_id,
            **({"extra": self.extra} if self.extra else {}),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SourceReference:
        if not data:
            return cls()
        return cls(
            source_path=str(data.get("source_path", "")),
            line_start=data.get("line_start"),
            line_end=data.get("line_end"),
            card_id=str(data.get("card_id", "")),
            extra=dict(data.get("extra") or {}),
        )


@dataclass
class KnowledgeIndexDocument:
    """统一索引文档模型。

    字段语义：
    - content：用于展示或上下文组装的正文（可含完整 evidence）。
    - embedding_text：确定性生成的向量化文本（loader 保证稳定）。
    - keywords：精确词汇召回用的关键词列表。
    - entities：人物/实体规范名列表，服务实体过滤与加权。
    - relations：文档表达的关系描述（如 "理央-主人-夜子"）。
    - source：原始引用定位。
    - metadata：知识域自定义扩展字段（story 信息等）。
    - reality_status / temporal_scope / content_scope：叙事层通用字段，
      取值由各 domain 约定（如 objective/fictional/...），核心代码只做
      通用过滤与权重策略，不写死任何作品的层级语义。
    """

    id: str
    domain_id: str
    document_type: str
    title: str
    summary: str
    content: str
    embedding_text: str
    keywords: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)
    source: SourceReference = field(default_factory=SourceReference)
    metadata: dict[str, Any] = field(default_factory=dict)
    reality_status: str = "unknown"
    temporal_scope: str = "unknown"
    content_scope: str = "unknown"
    review_status: str = "approved"
    index_version: str = "v1"

    @property
    def embedding_text_fingerprint(self) -> str:
        return _fingerprint(self.embedding_text)

    @property
    def content_fingerprint(self) -> str:
        return _fingerprint(f"{self.id}\n{self.embedding_text}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "domain_id": self.domain_id,
            "document_type": self.document_type,
            "title": self.title,
            "summary": self.summary,
            "content": self.content,
            "embedding_text": self.embedding_text,
            "keywords": list(self.keywords),
            "entities": list(self.entities),
            "relations": list(self.relations),
            "source": self.source.to_dict(),
            "metadata": dict(self.metadata),
            "reality_status": self.reality_status,
            "temporal_scope": self.temporal_scope,
            "content_scope": self.content_scope,
            "review_status": self.review_status,
            "index_version": self.index_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeIndexDocument:
        return cls(
            id=str(data["id"]),
            domain_id=str(data["domain_id"]),
            document_type=str(data["document_type"]),
            title=str(data.get("title", "")),
            summary=str(data.get("summary", "")),
            content=str(data.get("content", "")),
            embedding_text=str(data.get("embedding_text", "")),
            keywords=[str(k) for k in data.get("keywords", [])],
            entities=[str(e) for e in data.get("entities", [])],
            relations=[str(r) for r in data.get("relations", [])],
            source=SourceReference.from_dict(data.get("source")),
            metadata=dict(data.get("metadata") or {}),
            reality_status=str(data.get("reality_status", "unknown")),
            temporal_scope=str(data.get("temporal_scope", "unknown")),
            content_scope=str(data.get("content_scope", "unknown")),
            review_status=str(data.get("review_status", "approved")),
            index_version=str(data.get("index_version", "v1")),
        )

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)
