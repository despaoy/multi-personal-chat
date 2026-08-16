#!/usr/bin/env python3
"""Attach one continuous, reproducible source window to every Kisaki RAG document."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_PATH = (
    ROOT
    / "backend/data/character_dialogues/experiments/research/character_rag_seed_documents.json"
)
RAW_PATH = ROOT / "backend/data/character_dialogues/tsukiyashiro_kisaki_raw.jsonl"
GAME_ROOT = ROOT / "gametext/纸上魔法使"
MAX_EVIDENCE_WINDOW_LINES = 40
MAX_SEGMENT_PHYSICAL_LINES = 8

CONTENT_OVERRIDES = {
    "tsukiyashiro_kisaki_doc_010": (
        "[琉璃] 「就算不做这种事，我也一直——」\n"
        "一直，无时无刻，\n"
        "都没有忘记你哦。\n"
        "[妃] 「……为了确认这点，是需要时间的。」"
    ),
    "tsukiyashiro_kisaki_doc_015": (
        "[妃] 「今后，假若发生同样的事情。」\n"
        "[妃] 「我避免再妒忌，也不会再多嘴啰嗦。」\n"
        "[妃] 「我不想让你看到我丑陋的地方。妒忌可是无谓的感情。」\n"
        "[妃] 「我相信你不会变心。因此，我不会再妒忌。」\n"
        "[妃] 「即使不能，我也做到不表现出来。我会让你体会到这份爱的宽大。」"
    ),
}

SUPPLEMENTAL_DOCUMENTS = [
    {
        "id": "tsukiyashiro_kisaki_doc_031",
        "title": "tsukiyashiro_kisaki held-out evidence",
        "category": "tsukiyashiro_kisaki",
        "content": "我想和理央还有夜子穿着同样的校服，一起到这间学校上学。可是，只有我年纪小一岁呢。",
        "metadata": {"persona": "tsukiyashiro_kisaki", "held_out": True},
    },
    {
        "id": "tsukiyashiro_kisaki_doc_032",
        "title": "tsukiyashiro_kisaki held-out evidence",
        "category": "tsukiyashiro_kisaki",
        "content": "想快点办完事情，回去尝理央的晚餐啊……",
        "metadata": {"persona": "tsukiyashiro_kisaki", "held_out": True},
    },
    {
        "id": "tsukiyashiro_kisaki_doc_033",
        "title": "tsukiyashiro_kisaki held-out evidence",
        "category": "tsukiyashiro_kisaki",
        "content": "怎么办呢？这书是夜子借我的，要不你自己去问她拿许可？",
        "metadata": {"persona": "tsukiyashiro_kisaki", "held_out": True},
    },
]

SPEAKER_PREFIX_RE = re.compile(r"^\[[^\]]+\]\s*")
NORMALIZE_RE = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalized(value: str) -> str:
    return NORMALIZE_RE.sub("", value.casefold())


def semantic_line(value: str) -> str:
    value = SPEAKER_PREFIX_RE.sub("", value.strip())
    if value[:1] in {"「", "『", "“"}:
        value = value[1:]
    if value[-1:] in {"」", "』", "”"}:
        value = value[:-1]
    return normalized(value)


def content_segments(content: str) -> list[str]:
    segments = [semantic_line(line) for line in content.splitlines() if line.strip()]
    if not segments or any(not segment for segment in segments):
        raise ValueError("RAG evidence contains an empty normalized segment")
    return segments


def matching_spans(lines: list[str], segment: str) -> list[tuple[int, int]]:
    matches: list[tuple[int, int]] = []
    for start in range(len(lines)):
        combined = ""
        for end in range(start, min(len(lines), start + MAX_SEGMENT_PHYSICAL_LINES)):
            combined += semantic_line(lines[end])
            if not combined:
                continue
            if combined == segment:
                matches.append((start + 1, end + 1))
                break
            if len(combined) >= len(segment) or not segment.startswith(combined):
                break
    return matches


def _ordered_paths(
    segment_matches: list[list[tuple[int, int]]],
) -> list[tuple[tuple[int, int], ...]]:
    paths = [(match,) for match in segment_matches[0]]
    for matches in segment_matches[1:]:
        next_paths = []
        for path in paths:
            first_start = path[0][0]
            previous_end = path[-1][1]
            for match in matches:
                if match[0] <= previous_end:
                    continue
                if match[1] - first_start + 1 > MAX_EVIDENCE_WINDOW_LINES:
                    continue
                next_paths.append((*path, match))
        paths = sorted(
            next_paths,
            key=lambda path: (path[-1][1] - path[0][0], path[0][0], path[-1][1]),
        )[:2000]
        if not paths:
            break
    return paths


def source_window_candidates(
    content: str, source_files: dict[str, list[str]]
) -> list[tuple[str, tuple[tuple[int, int], ...]]]:
    segments = content_segments(content)
    candidates: list[tuple[int, int, str, tuple[tuple[int, int], ...]]] = []
    for source_file, lines in source_files.items():
        segment_matches = [matching_spans(lines, segment) for segment in segments]
        if any(not matches for matches in segment_matches):
            continue
        for path in _ordered_paths(segment_matches):
            window_size = path[-1][1] - path[0][0] + 1
            gap_size = window_size - sum(end - start + 1 for start, end in path)
            candidates.append((window_size, gap_size, source_file, path))
    if not candidates:
        raise ValueError(
            "all evidence segments must match in order inside one "
            f"{MAX_EVIDENCE_WINDOW_LINES}-line source window"
        )
    return [
        (source_file, path)
        for _, _, source_file, path in sorted(candidates)
    ]


def match_source_window(
    content: str, source_files: dict[str, list[str]]
) -> tuple[str, tuple[tuple[int, int], ...]]:
    return source_window_candidates(content, source_files)[0]


def raw_events_in_window(
    raw: list[dict[str, Any]], source_file: str, line_start: int, line_end: int
) -> list[str]:
    return [
        event["id"]
        for event in raw
        if event.get("source_file") == source_file
        and int(event.get("source_line_end", event.get("source_line_start", -1))) >= line_start
        and int(event.get("source_line_start", -1)) <= line_end
    ]


def enrich_document(
    document: dict[str, Any], source_files: dict[str, list[str]], raw: list[dict[str, Any]]
) -> None:
    selected = None
    for source_file, spans in source_window_candidates(str(document["content"]), source_files):
        line_start = spans[0][0]
        line_end = spans[-1][1]
        event_ids = raw_events_in_window(raw, source_file, line_start, line_end)
        if event_ids:
            selected = source_file, spans, event_ids
            break
    if selected is None:
        raise ValueError(f"{document['id']} source windows have no Kisaki raw event")
    source_file, spans, event_ids = selected
    line_start = spans[0][0]
    line_end = spans[-1][1]
    source_path = f"gametext/纸上魔法使/{source_file}"

    document["source_path"] = source_path
    document["source_line_start"] = line_start
    document["source_line_end"] = line_end
    document["source_event_ids"] = event_ids
    document["source_lineage"] = [
        {
            "segment_index": index,
            "source_path": source_path,
            "source_line_start": start,
            "source_line_end": end,
        }
        for index, (start, end) in enumerate(spans, 1)
    ]

    for key in ("source_line", "content_hash", "kb_revision"):
        document.pop(key, None)
    metadata = document.setdefault("metadata", {})
    for key in ("source_lineage", "content_hash", "kb_revision"):
        metadata.pop(key, None)


def build() -> dict[str, Any]:
    payload = json.loads(DOCUMENTS_PATH.read_text(encoding="utf-8"))
    raw = load_jsonl(RAW_PATH)
    source_files = {
        path.name: path.read_text(encoding="utf-8").splitlines()
        for path in sorted(GAME_ROOT.glob("*.txt"))
    }
    documents = payload["documents"]
    by_id = {document["id"]: document for document in documents}
    for document_id, content in CONTENT_OVERRIDES.items():
        by_id[document_id]["content"] = content
    for document in SUPPLEMENTAL_DOCUMENTS:
        if document["id"] not in by_id:
            documents.append(document)
            by_id[document["id"]] = document

    errors = []
    for document in documents:
        try:
            enrich_document(document, source_files, raw)
        except ValueError as exc:
            errors.append(f"{document['id']}: {exc}")
    if errors:
        raise ValueError("RAG lineage failed:\n" + "\n".join(errors))

    revision_payload = [
        {
            "id": document["id"],
            "content": document["content"],
            "source_path": document["source_path"],
            "source_line_start": document["source_line_start"],
            "source_line_end": document["source_line_end"],
            "source_event_ids": document["source_event_ids"],
        }
        for document in documents
    ]
    payload["kb_revision"] = canonical_hash(revision_payload)
    return payload


def main() -> int:
    payload = build()
    DOCUMENTS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"documents": len(payload["documents"]), "kb_revision": payload["kb_revision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
