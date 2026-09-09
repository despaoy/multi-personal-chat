#!/usr/bin/env python3
"""Build the retired fixed-size text-chunk baseline for an isolated RAG test.

This script intentionally does not touch the production multi-scale index.  It
recreates the old generic splitter so that the legacy baseline can be measured
independently on another machine.

Example:
    python backend/scripts/build_legacy_chunk_baseline.py \
        --source-dir gametext/纸上魔法使 \
        --output /tmp/kisaki_chunks_600_100.jsonl

The historical generic splitter used a 600-character target size.  The
repository had both 100- and 150-character overlap call sites, so the overlap
is an explicit CLI option rather than an undocumented assumption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

PAIR_DELIMITERS = (
    ("「", "」"),
    ("『", "』"),
    ("【", "】"),
    ("《", "》"),
    ("（", "）"),
    ('"', '"'),
    ("'", "'"),
)
SENTENCE_ENDS = set("。！？；.!?;\n")


def _inside_pair(text: str, position: int) -> bool:
    """Return whether ``position`` is inside a configured quote pair."""

    prefix = text[:position]
    return any(prefix.count(left) > prefix.count(right) for left, right in PAIR_DELIMITERS)


def _sentences(paragraph: str) -> list[str]:
    result: list[str] = []
    start = 0
    for index, char in enumerate(paragraph):
        if char in SENTENCE_ENDS and not _inside_pair(paragraph, index):
            sentence = paragraph[start : index + 1].strip()
            if sentence:
                result.append(sentence)
            start = index + 1
    tail = paragraph[start:].strip()
    if tail:
        result.append(tail)
    return result


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Reproduce the old paragraph/句子/定长 splitter."""

    if not text or chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("要求 chunk_size > 0 且 0 <= overlap < chunk_size")

    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    def flush() -> None:
        nonlocal current, current_length
        if current:
            chunks.append(" ".join(current))
            current = []
            current_length = 0

    def start_with_overlap(next_text: str) -> None:
        nonlocal current, current_length
        if not current or overlap == 0:
            current = [next_text]
            current_length = len(next_text)
            return
        tail = " ".join(current[-2:]) if len(current) >= 2 else current[-1]
        tail = tail[-overlap:]
        current = [tail, next_text]
        current_length = len(tail) + len(next_text)

    for paragraph in paragraphs:
        units = _sentences(paragraph) if len(paragraph) > chunk_size else [paragraph]
        expanded_units: list[str] = []
        for unit in units or [paragraph]:
            if len(unit) > chunk_size:
                expanded_units.extend(unit[i : i + chunk_size] for i in range(0, len(unit), chunk_size))
            else:
                expanded_units.append(unit)

        for unit in expanded_units:
            if current_length + len(unit) <= chunk_size:
                current.append(unit)
                current_length += len(unit)
            else:
                flush()
                start_with_overlap(unit)

    flush()
    return chunks


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def build_documents(source_dir: Path, chunk_size: int, overlap: int) -> list[dict[str, object]]:
    """Read source text files and return portable JSONL documents."""

    if not source_dir.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {source_dir}")

    documents: list[dict[str, object]] = []
    for source_path in sorted(source_dir.rglob("*.txt")):
        text = source_path.read_text(encoding="utf-8")
        chunks = split_text(text, chunk_size, overlap)
        relative = _relative_path(source_path, source_dir)
        for chunk_index, content in enumerate(chunks):
            digest = hashlib.sha1(f"{relative}\n{chunk_index}\n{content}".encode()).hexdigest()[:16]
            documents.append(
                {
                    "id": f"legacy_chunk_{digest}",
                    "document_type": "legacy_chunk",
                    "title": source_path.stem,
                    "content": content,
                    "source_file": relative,
                    "chunk_index": chunk_index,
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                }
            )
    return documents


def _write_jsonl(path: Path, documents: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="构建旧版固定长度 RAG 分块基线")
    parser.add_argument("--source-dir", type=Path, required=True, help="原始 TXT 文件目录")
    parser.add_argument("--output", type=Path, required=True, help="输出 JSONL 路径")
    parser.add_argument("--chunk-size", type=int, default=600, help="目标分块字符数，默认 600")
    parser.add_argument(
        "--overlap",
        type=int,
        default=100,
        help="相邻分块重叠字符数，默认 100；旧导入器也曾使用 150",
    )
    args = parser.parse_args()

    documents = build_documents(args.source_dir, args.chunk_size, args.overlap)
    _write_jsonl(args.output, documents)

    lengths = [len(str(document["content"])) for document in documents]
    file_count = len({document["source_file"] for document in documents})
    print(f"source_files={file_count}")
    print(f"chunks={len(documents)}")
    print(f"chunk_size={args.chunk_size} overlap={args.overlap}")
    if lengths:
        print(
            "length_chars="
            f"min:{min(lengths)} mean:{statistics.mean(lengths):.1f} "
            f"p95:{sorted(lengths)[int(len(lengths) * 0.95) - 1]} max:{max(lengths)}"
        )
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
