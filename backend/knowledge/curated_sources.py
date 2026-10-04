"""Fresh declared source spans retain native curated identities and scope."""

import hashlib
import re
from pathlib import Path

from knowledge.multiscale_rag.source_text import OriginalTextExtractor
from knowledge.multiscale_rag.visibility import visible_to
from knowledge.retrieval_core.documents import SourceReference


def _extractor():
    return OriginalTextExtractor(Path(__file__).resolve().parents[2])


def _sha(data):
    return hashlib.sha256(data).hexdigest()


class CuratedSourceChangedError(ValueError):
    """A valid original snapshot no longer matches the current source file."""


def collect_curated_sources(candidates, evidence_by_parent, extractor, boundary):
    """Read every selected visible evidence reference; never invent a source."""
    sources, unavailable, seen = [], [], set()
    for candidate in candidates:
        card = candidate.document
        evidence = evidence_by_parent.get(card.id)
        if extractor is None or evidence is None or not visible_to(evidence, boundary):
            unavailable.append(dict(parent_id=card.id, status="no_visible_source"))
            continue
        if evidence.id in seen:
            raise ValueError("Ambiguous native curated source identity")
        seen.add(evidence.id)
        try:
            path = extractor.resolve_path(evidence.source.source_path)
            before = path.read_bytes()
            excerpt = extractor.extract(evidence.source).to_dict()
            if path.read_bytes() != before:
                raise ValueError("Curated source changed during original read")
            sources.append(
                dict(
                    source_id=evidence.id,
                    parent_id=card.id,
                    domain_id=card.domain_id,
                    authority="fresh_curated_declared_span",
                    declared_source=evidence.source.to_dict(),
                    excerpt=excerpt,
                    source_file_sha256=_sha(before),
                    body_sha256=_sha(excerpt["text"].encode()),
                    extraction_limits=dict(max_lines=extractor.max_lines, max_chars=extractor.max_chars),
                    indexed_claim=card.content,
                    indexed_claim_status="not_independently_verified",
                    indexed_annotations=dict(
                        metadata=card.metadata,
                        reality_status=card.reality_status,
                        temporal_scope=card.temporal_scope,
                        content_scope=card.content_scope,
                    ),
                )
            )
        except (ValueError, OSError, UnicodeError) as exc:
            unavailable.append(dict(parent_id=card.id, status="source_unavailable", error_type=type(exc).__name__))
    return dict(sources=sources, unavailable=unavailable)


def validate_curated_source(source, *, verify_original=True):
    """Re-read the same declared span and original file before final admission."""
    required = {
        "source_id",
        "parent_id",
        "domain_id",
        "authority",
        "declared_source",
        "excerpt",
        "source_file_sha256",
        "body_sha256",
        "extraction_limits",
        "indexed_claim",
        "indexed_claim_status",
        "indexed_annotations",
    }
    if (
        not isinstance(source, dict)
        or set(source) != required
        or source["authority"] != "fresh_curated_declared_span"
        or source["indexed_claim_status"] != "not_independently_verified"
    ):
        raise ValueError("Invalid curated source receipt")
    if any(not isinstance(source[key], str) or not source[key] for key in ("source_id", "parent_id", "domain_id")):
        raise ValueError("Invalid native curated identity")
    limits = source["extraction_limits"]
    if (
        not isinstance(limits, dict)
        or set(limits) != {"max_lines", "max_chars"}
        or any(type(limits[key]) is not int or limits[key] < 1 for key in limits)
    ):
        raise ValueError("Invalid curated extraction bounds")
    extractor = _extractor()
    if limits != {"max_lines": extractor.max_lines, "max_chars": extractor.max_chars}:
        raise ValueError("Curated source changed its extraction profile")
    reference = SourceReference.from_dict(source["declared_source"])
    excerpt = source["excerpt"]
    if (
        not isinstance(excerpt, dict)
        or set(excerpt) != {"source_path", "line_start", "line_end", "text", "truncated"}
        or not isinstance(excerpt["text"], str)
        or not excerpt["text"].strip()
        or type(excerpt["truncated"]) is not bool
        or excerpt["source_path"] != reference.source_path
        or any(type(excerpt[key]) is not int for key in ("line_start", "line_end"))
        or excerpt["line_start"] < 1
        or excerpt["line_end"] < excerpt["line_start"]
        or type(reference.line_start) is not int
        or type(reference.line_end) is not int
        or reference.line_start != excerpt["line_start"]
        or reference.line_end < excerpt["line_end"]
        or _sha(excerpt["text"].encode()) != source["body_sha256"]
        or not re.fullmatch(r"[a-f0-9]{64}", str(source["source_file_sha256"]))
    ):
        raise ValueError("Invalid original curated span identity")
    if not isinstance(source["indexed_claim"], str) or not isinstance(source["indexed_annotations"], dict):
        raise ValueError("Invalid unverified curated annotations")
    if not verify_original:
        return source
    path = extractor.resolve_path(reference.source_path)
    before = path.read_bytes()
    excerpt = extractor.extract(reference).to_dict()
    if (
        path.read_bytes() != before
        or _sha(before) != source["source_file_sha256"]
        or excerpt != source["excerpt"]
        or _sha(excerpt["text"].encode()) != source["body_sha256"]
    ):
        raise CuratedSourceChangedError("Curated original span changed its actual source")
    return source


def curated_source_packet(source):
    excerpt = source["excerpt"]
    location = f"{excerpt['source_path']} L{excerpt['line_start']}-{excerpt['line_end']}"
    scope = "declared_span_partial" if excerpt["truncated"] else "complete_declared_span"
    return dict(
        kind="evidence",
        document_ids=[source["source_id"]],
        curated_source_id=source["source_id"],
        original_excerpt=excerpt["text"],
        original_excerpt_sha256=source["body_sha256"],
        text=f"【角色原作来源片段: {location}】\n来源范围：{scope}，不代表原作全文完整。\n{excerpt['text']}",
    )
