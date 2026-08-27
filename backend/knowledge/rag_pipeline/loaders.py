"""Source Loader（P6）：approved 知识卡 → canonical 索引文档。

ApprovedCardsLoader 面向 P5 产出的通用知识卡契约
（fact / relation / event + review_status），不包含任何作品专属
词表——实体规范化所需的词表由 domain 配置注入。其他来源
（角色记忆、用户文档）可实现同签名的 loader 接入同一管线。

embedding_text 构造原则：
- 事实卡：主体 + 谓词 + 值 + 标题 + 摘要 + 关键证据 + 故事信息
- 关系卡：主体 + 关系类型 + 对象 + 自然语言方向表达 + 摘要 + 证据 + 故事信息
- 事件卡：标题 + 摘要 + 参与者 + 起因 + 结果 + 证据 + 故事信息
- 不只 embedding 标题，也不序列化整份 JSON
- 完整 evidence 保留在 content 与 source 中，不丢失原文引用能力
- 单张知识卡对应一个索引文档（不按字符数拆碎）
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .documents import KnowledgeIndexDocument, SourceReference

logger = logging.getLogger(__name__)

# approved 卡文件名（P5 契约）
FACTS_FILE = "facts_approved.jsonl"
RELATIONS_FILE = "relations_approved.jsonl"
EVENTS_FILE = "events_approved.jsonl"

# embedding_text 中保留的关键证据长度上限（字符）
EVIDENCE_EXCERPT_CHARS = 240


class UnapprovedDataError(ValueError):
    """输入数据未通过 approved 门禁时抛出（构建入口必须失败）。"""


def _clean_sentence(text: str) -> str:
    """去掉尾部重复句号（拼接时统一加句号）。"""
    text = (text or "").rstrip()
    while text.endswith("。"):
        text = text[:-1].rstrip()
    return text


def _evidence_excerpt(evidence: str, limit: int = EVIDENCE_EXCERPT_CHARS) -> str:
    text = (evidence or "").strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _story_info_text(story: dict[str, Any] | None) -> str:
    if not story:
        return ""
    parts: list[str] = []
    title = story.get("story_title")
    if title:
        parts.append(f"出自《{title}》")
    volume = story.get("volume_number")
    if volume is not None:
        parts.append(f"第{volume}卷")
    scope = story.get("content_scope")
    if scope:
        parts.append(str(scope))
    temporal = story.get("temporal_scope")
    if temporal:
        parts.append(str(temporal))
    viewpoint = story.get("viewpoint")
    if viewpoint:
        parts.append(f"视角:{viewpoint}")
    return "，".join(parts)


def _story_layers(story: dict[str, Any] | None) -> dict[str, str]:
    story = story or {}
    return {
        "temporal_scope": str(story.get("temporal_scope") or "unknown"),
        "content_scope": str(story.get("content_scope") or "unknown"),
    }


class AliasEntityNormalizer:
    """基于 domain alias 表的实体归一器。

    tokens() 返回 (alias, canonical) 全量序对，供 loader 做子串
    识别；canonical(token) 供查询分析做精确归一。
    """

    def __init__(self, aliases: dict[str, str]):
        # 确保规范名本身也在表内（自映射）
        self._aliases: dict[str, str] = dict(aliases)
        for canonical in set(aliases.values()):
            self._aliases.setdefault(canonical, canonical)
        # 按长度降序排列，保证"遊行寺夜子"优先于"夜子"命中
        self._sorted_tokens = sorted(self._aliases.keys(), key=len, reverse=True)

    def canonical(self, token: str) -> str | None:
        return self._aliases.get(token)

    def tokens(self):
        for token in self._sorted_tokens:
            yield token, self._aliases[token]

    def scan_text(self, text: str) -> list[str]:
        """返回文本中出现的规范实体（长词优先，互不重叠）。"""
        result: list[str] = []
        consumed: list[tuple[int, int]] = []
        for token, canonical in self.tokens():
            start = 0
            while True:
                idx = text.find(token, start) if token else -1
                if idx < 0:
                    break
                span = (idx, idx + len(token))
                if not any(s <= span[0] < e or s < span[1] <= e for s, e in consumed):
                    consumed.append(span)
                    if canonical not in result:
                        result.append(canonical)
                start = idx + 1
        return result


class ApprovedCardsLoader:
    """P5 approved 知识卡 loader（通用卡契约，作品词表由注入决定）。

    review_status != approved 的卡片直接抛 UnapprovedDataError，
    构建入口不会静默跳过未批准数据。
    """

    def __init__(
        self,
        domain_id: str,
        index_version: str,
        entity_normalizer: AliasEntityNormalizer | None = None,
        source_root: Path | None = None,
    ):
        self.domain_id = domain_id
        self.index_version = index_version
        self._normalizer = entity_normalizer
        self._source_root = source_root

    # -- loader 协议入口 -------------------------------------------------
    def __call__(self, source_root: Path) -> list[KnowledgeIndexDocument]:
        return self.load(source_root)

    def load(self, source_root: Path | None = None) -> list[KnowledgeIndexDocument]:
        root = source_root or self._source_root
        if root is None:
            raise ValueError("ApprovedCardsLoader 需要 source_root")
        root = Path(root)
        documents: list[KnowledgeIndexDocument] = []
        documents.extend(self._load_facts(root))
        documents.extend(self._load_relations(root))
        documents.extend(self._load_events(root))

        ids = [doc.id for doc in documents]
        if len(ids) != len(set(ids)):
            duplicated = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"文档 ID 重复: {duplicated[:5]}")
        return documents

    # -- JSONL 读取与门禁 -------------------------------------------------
    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            logger.warning("approved 卡文件不存在，跳过: %s", path)
            return []
        records: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    raise UnapprovedDataError(f"{path.name}:{line_no} JSON 解析失败: {e}") from e
                status = record.get("review_status")
                if status != "approved":
                    raise UnapprovedDataError(f"{path.name}:{line_no} review_status={status!r}，未通过 approved 门禁")
                records.append(record)
        return records

    def _entities(self, *text_values: str) -> list[str]:
        """核心实体：结构化字段（subject/target/participants + 标题/
        摘要/值）扫描结果。evidence 提及的实体不进入 entities——
        实体过滤/加权需要"关于谁"的精确语义，"提到谁"存放在
        metadata.mentioned_entities 中保留可用。"""
        if self._normalizer is None:
            return []
        return self._normalizer.scan_text(" ".join(v for v in text_values if v))

    def _mentioned_entities(self, evidence: str) -> list[str]:
        """evidence 原文中提及的实体（降级存放，不参与实体通道）。"""
        if self._normalizer is None or not evidence:
            return []
        return self._normalizer.scan_text(evidence)

    def _base_layers(self, card: dict[str, Any]) -> dict[str, str]:
        layers = _story_layers(card.get("story"))
        reality = str(card.get("reality_status") or "unknown")
        return {"reality_status": reality, **layers}

    def _source(self, card: dict[str, Any]) -> SourceReference:
        src = card.get("source") or {}
        story = card.get("story") or {}
        extra = {k: story[k] for k in ("story_unit_id", "story_title") if story.get(k)}
        return SourceReference(
            source_path=str(src.get("source_path", "")),
            line_start=src.get("line_start"),
            line_end=src.get("line_end"),
            card_id=str(card.get("id", "")),
            extra=extra,
        )

    # -- 事实卡 ------------------------------------------------------------
    def _load_facts(self, root: Path) -> list[KnowledgeIndexDocument]:
        docs: list[KnowledgeIndexDocument] = []
        for card in self._read_jsonl(root / FACTS_FILE):
            subject = str(card.get("subject") or "")
            predicate = str(card.get("predicate") or "")
            value = str(card.get("value") or "")
            title = str(card.get("title") or "")
            summary = str(card.get("summary") or "")
            evidence = str(card.get("evidence_text") or "")
            story = card.get("story") or {}
            layers = self._base_layers(card)
            story_info = _story_info_text(story)

            # 三元组 + 标题 + 摘要 + 关键证据 + 故事信息
            embedding_parts = [
                f"{subject}的{predicate}是{value}。" if subject or predicate or value else "",
                f"标题：{title}。" if title else "",
                f"摘要：{_clean_sentence(summary)}。" if summary else "",
                f"关键证据：{_evidence_excerpt(evidence)}。" if evidence else "",
                f"故事信息：{story_info}。" if story_info else "",
            ]
            embedding_text = "".join(p for p in embedding_parts if p)

            content_parts = [
                f"{subject}的{predicate}是{value}。" if subject or predicate or value else "",
                f"（{summary}）" if summary else "",
                f"\n证据：{evidence}" if evidence else "",
            ]
            content = "".join(p for p in content_parts if p)

            keywords = [v for v in (subject, predicate, value, title, *summary.split("，")) if v]

            docs.append(
                KnowledgeIndexDocument(
                    id=str(card["id"]),
                    domain_id=self.domain_id,
                    document_type="fact",
                    title=title or summary[:40],
                    summary=summary,
                    content=content,
                    embedding_text=embedding_text,
                    keywords=keywords,
                    entities=self._entities(subject, value, title, summary),
                    relations=[],
                    source=self._source(card),
                    metadata={
                        "subject": subject,
                        "predicate": predicate,
                        "value": value,
                        "mentioned_entities": self._mentioned_entities(evidence),
                        **{
                            k: story.get(k)
                            for k in ("volume_number", "story_unit_id", "story_title", "viewpoint", "route")
                        },
                    },
                    reality_status=layers["reality_status"],
                    temporal_scope=layers["temporal_scope"],
                    content_scope=layers["content_scope"],
                    review_status="approved",
                    index_version=self.index_version,
                )
            )
        return docs

    # -- 关系卡 ------------------------------------------------------------
    def _load_relations(self, root: Path) -> list[KnowledgeIndexDocument]:
        docs: list[KnowledgeIndexDocument] = []
        for card in self._read_jsonl(root / RELATIONS_FILE):
            subject = str(card.get("subject") or "")
            relation = str(card.get("relation") or "")
            target = str(card.get("target") or "")
            title = str(card.get("title") or "")
            summary = str(card.get("summary") or "")
            evidence = str(card.get("evidence_text") or "")
            story = card.get("story") or {}
            layers = self._base_layers(card)
            story_info = _story_info_text(story)

            direction = summary or f"{subject}与{target}是{relation}关系"
            embedding_parts = [
                f"{subject}与{target}的关系是{relation}：{_clean_sentence(direction)}。"
                if relation
                else _clean_sentence(direction) + "。",
                f"标题：{title}。" if title else "",
                f"摘要：{_clean_sentence(summary)}。" if summary else "",
                f"关键证据：{_evidence_excerpt(evidence)}。" if evidence else "",
                f"故事信息：{story_info}。" if story_info else "",
            ]
            embedding_text = "".join(p for p in embedding_parts if p)

            content = f"{direction}\n（{subject} —{relation}→ {target}）" + (f"\n证据：{evidence}" if evidence else "")
            keywords = [v for v in (subject, relation, target, title, *summary.split("，")) if v]
            relation_expr = f"{subject}-{relation}-{target}"

            docs.append(
                KnowledgeIndexDocument(
                    id=str(card["id"]),
                    domain_id=self.domain_id,
                    document_type="relation",
                    title=title or relation_expr,
                    summary=summary or direction,
                    content=content,
                    embedding_text=embedding_text,
                    keywords=keywords,
                    entities=self._entities(subject, target, title, summary),
                    relations=[relation_expr],
                    source=self._source(card),
                    metadata={
                        "subject": subject,
                        "relation": relation,
                        "target": target,
                        "mentioned_entities": self._mentioned_entities(evidence),
                        **{
                            k: story.get(k)
                            for k in ("volume_number", "story_unit_id", "story_title", "viewpoint", "route")
                        },
                    },
                    reality_status=layers["reality_status"],
                    temporal_scope=layers["temporal_scope"],
                    content_scope=layers["content_scope"],
                    review_status="approved",
                    index_version=self.index_version,
                )
            )
        return docs

    # -- 事件卡 ------------------------------------------------------------
    def _load_events(self, root: Path) -> list[KnowledgeIndexDocument]:
        docs: list[KnowledgeIndexDocument] = []
        for card in self._read_jsonl(root / EVENTS_FILE):
            title = str(card.get("title") or "")
            summary = str(card.get("summary") or "")
            participants = [str(p) for p in (card.get("participants") or []) if p]
            causes = [str(c) for c in (card.get("causes") or []) if c]
            outcomes = [str(o) for o in (card.get("outcomes") or []) if o]
            evidence = str(card.get("evidence_text") or "")
            story = card.get("story") or {}
            layers = self._base_layers(card)
            story_info = _story_info_text(story)

            def _join(items: list[str]) -> str:
                return "；".join(i for i in items if i)

            embedding_parts = [
                f"事件：{title}。{_clean_sentence(summary)}。" if title else f"{_clean_sentence(summary)}。",
                f"参与者：{_join(participants)}。" if participants else "",
                f"起因：{_join(causes)}。" if causes else "",
                f"结果：{_join(outcomes)}。" if outcomes else "",
                f"关键证据：{_evidence_excerpt(evidence)}。" if evidence else "",
                f"故事信息：{story_info}。" if story_info else "",
            ]
            embedding_text = "".join(p for p in embedding_parts if p)

            content_parts = [
                f"【{title}】{summary}",
                f"\n参与者：{_join(participants)}" if participants else "",
                f"\n起因：{_join(causes)}" if causes else "",
                f"\n结果：{_join(outcomes)}" if outcomes else "",
                f"\n证据：{evidence}" if evidence else "",
            ]
            content = "".join(p for p in content_parts if p)
            keywords = [title, *participants, *causes, *outcomes]
            keywords = [k for k in keywords if k]

            docs.append(
                KnowledgeIndexDocument(
                    id=str(card["id"]),
                    domain_id=self.domain_id,
                    document_type="event",
                    title=title or summary[:40],
                    summary=summary,
                    content=content,
                    embedding_text=embedding_text,
                    keywords=keywords,
                    entities=self._entities(" ".join(participants), title, summary, _join(causes), _join(outcomes)),
                    relations=[],
                    source=self._source(card),
                    metadata={
                        "participants": participants,
                        "causes": causes,
                        "outcomes": outcomes,
                        "mentioned_entities": self._mentioned_entities(evidence),
                        **{
                            k: story.get(k)
                            for k in ("volume_number", "story_unit_id", "story_title", "viewpoint", "route")
                        },
                    },
                    reality_status=layers["reality_status"],
                    temporal_scope=layers["temporal_scope"],
                    content_scope=layers["content_scope"],
                    review_status="approved",
                    index_version=self.index_version,
                )
            )
        return docs
