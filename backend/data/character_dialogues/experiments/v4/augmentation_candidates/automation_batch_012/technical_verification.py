from __future__ import annotations

import json
import re
from pathlib import Path


BATCH_DIR = Path(__file__).resolve().parent


def load_parser_class():
    sessions = json.loads((BATCH_DIR / "candidates.json").read_text(encoding="utf-8"))
    session = next(item for item in sessions if item["session_id"] == "incremental_jsonl_byte_parser")
    answer = session["messages"][-1]["content"]
    match = re.search(r"```python\n(.*?)\n```", answer, flags=re.DOTALL)
    if match is None:
        raise AssertionError("embedded Python implementation is missing")
    namespace: dict[str, object] = {}
    exec(compile(match.group(1), "<embedded-jsonl-parser>", "exec"), namespace)
    return namespace["JsonlStreamParser"]


def verify() -> None:
    parser_class = load_parser_class()

    payload = '{"text":"月社妃"}\r\n   \n{"n":2}\n{"tail":true}'.encode("utf-8")
    parser = parser_class(max_line_bytes=64)
    values = []
    for byte in payload:
        values.extend(parser.feed(bytes([byte])))
    values.extend(parser.finalize())
    assert values == [{"text": "月社妃"}, {"n": 2}, {"tail": True}]

    parser = parser_class(max_line_bytes=64)
    assert parser.feed(b'{"a":1}\n{"b":2}\n') == [{"a": 1}, {"b": 2}]
    assert parser.finalize() == []
    try:
        parser.finalize()
    except RuntimeError:
        pass
    else:
        raise AssertionError("finalize must not be repeatable")

    parser = parser_class(max_line_bytes=8)
    try:
        parser.feed(b"123456789")
    except ValueError as exc:
        assert "max_line_bytes" in str(exc)
    else:
        raise AssertionError("oversized lines must fail")

    parser = parser_class()
    try:
        parser.feed(b'{"broken":}\n')
    except ValueError as exc:
        assert "line 1" in str(exc)
    else:
        raise AssertionError("invalid JSON must fail")

    parser = parser_class()
    try:
        parser.feed(b'"\xe6\x9c"\n')
    except ValueError as exc:
        assert "line 1" in str(exc)
    else:
        raise AssertionError("invalid UTF-8 must fail")


if __name__ == "__main__":
    verify()
    print("technical verification passed")
