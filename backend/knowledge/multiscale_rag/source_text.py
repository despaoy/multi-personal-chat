"""Safe, exact source-text extraction for character knowledge retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from knowledge.retrieval_core.documents import SourceReference


@dataclass(frozen=True)
class RawExcerpt:
    """An exact excerpt read from a registered source file."""

    source_path: str
    line_start: int
    line_end: int
    text: str
    truncated: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "source_path": self.source_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "text": self.text,
            "truncated": self.truncated,
        }


class OriginalTextExtractor:
    """Resolve portable source paths beneath one explicit corpus root.

    ``corpus_root`` is normally the repository root because approved records
    use paths such as ``gametext/纸上魔法使/1翡翠的排挤原理.txt``.
    Path traversal and unbounded excerpts are rejected before any file read.
    """

    def __init__(
        self,
        corpus_root: Path,
        *,
        max_lines: int = 5000,
        max_chars: int = 100_000,
    ) -> None:
        if type(max_lines) is not int or max_lines < 1:
            raise ValueError("max_lines must be a positive integer")
        if type(max_chars) is not int or max_chars < 100:
            raise ValueError("max_chars must be an integer >= 100")
        self.corpus_root = Path(corpus_root).resolve()
        self.max_lines = max_lines
        self.max_chars = max_chars

    def resolve_path(self, source_path: str) -> Path:
        if not source_path:
            raise ValueError("source_path 不能为空")
        candidate = (self.corpus_root / source_path).resolve()
        try:
            candidate.relative_to(self.corpus_root)
        except ValueError as exc:
            raise ValueError("source_path 超出语料根目录") from exc
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate

    def extract(self, source: SourceReference, *, snapshot: bytes | None = None) -> RawExcerpt:
        """Use caller-owned bytes when source hashes must describe the same snapshot."""
        start, end = source.line_start, source.line_end
        if type(start) is not int or type(end) is not int or start < 1 or end < start:
            raise ValueError(f"非法来源行号: {start}-{end}")
        if end - start + 1 > self.max_lines:
            raise ValueError("Complete source span exceeds max_lines; narrow the declared span or increase the limit")

        path = self.resolve_path(source.source_path)
        if snapshot is None:
            snapshot = path.read_bytes()
        lines = snapshot.decode("utf-8-sig").splitlines()
        if end > len(lines):
            raise ValueError(f"来源行号超出文件范围: {start}-{end}>{len(lines)}")
        text = "\n".join(lines[start - 1 : end])
        if len(text) > self.max_chars:
            raise ValueError("Complete source span exceeds max_chars; narrow the declared span or increase the limit")
        return RawExcerpt(
            source_path=source.source_path,
            line_start=start,
            line_end=end,
            text=text,
        )

    def tighten_to_evidence(
        self,
        source: SourceReference,
        evidence_text: str,
    ) -> tuple[SourceReference, str]:
        """Narrow a broad approved span when evidence is an exact substring.

        Returns ``(source, status)``.  Failure to match is a safe fallback,
        never a reason to rewrite the approved evidence or widen its source.
        """
        evidence = (evidence_text or "").strip().replace("\r\n", "\n")
        if not evidence:
            return source, "empty_evidence"
        excerpt = self.extract(source)

        haystack = excerpt.text.replace("\r\n", "\n")
        offset = haystack.find(evidence)
        if offset < 0:
            return source, "approved_evidence_fallback"

        start_delta = haystack[:offset].count("\n")
        end_delta = start_delta + evidence.count("\n")
        narrowed = SourceReference(
            source_path=source.source_path,
            line_start=excerpt.line_start + start_delta,
            line_end=excerpt.line_start + end_delta,
            card_id=source.card_id,
            extra={**source.extra, "evidence_match": "exact"},
        )
        return narrowed, "exact_source_match"
