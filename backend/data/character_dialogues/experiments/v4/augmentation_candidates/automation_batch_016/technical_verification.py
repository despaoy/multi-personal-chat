from __future__ import annotations

import json
import re
import stat
import tempfile
import zipfile
from pathlib import Path


BATCH_DIR = Path(__file__).resolve().parent


def load_extractor():
    approval = json.loads(
        (BATCH_DIR / "approved_sessions.json").read_text(encoding="utf-8")
    )
    session = next(
        item for item in approval["sessions"] if item["session_id"] == "safe_zip_extraction"
    )
    answer = session["messages"][-1]["content"]
    match = re.search(r"```python\n(.*?)\n```", answer, flags=re.DOTALL)
    if match is None:
        raise AssertionError("embedded Python implementation is missing")
    namespace: dict[str, object] = {}
    exec(compile(match.group(1), "<embedded-safe-zip-extractor>", "exec"), namespace)
    return namespace["safe_extract_zip"]


def write_zip(path: Path, entries: list[tuple[str, bytes]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, content in entries:
            archive.writestr(name, content)


def assert_rejected(extractor, archive: Path, destination: Path, **limits) -> None:
    try:
        extractor(archive, destination, **limits)
    except (ValueError, zipfile.BadZipFile):
        pass
    else:
        raise AssertionError(f"unsafe archive was accepted: {archive.name}")
    assert not destination.exists()
    assert not list(destination.parent.glob(f".{destination.name}.staging.*"))


def corrupt_member(path: Path, member: str) -> None:
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
    payload = bytearray(path.read_bytes())
    offset = info.header_offset
    name_length = int.from_bytes(payload[offset + 26 : offset + 28], "little")
    extra_length = int.from_bytes(payload[offset + 28 : offset + 30], "little")
    data_offset = offset + 30 + name_length + extra_length
    payload[data_offset] ^= 0xFF
    path.write_bytes(payload)


def verify() -> None:
    extractor = load_extractor()

    with tempfile.TemporaryDirectory() as raw_dir:
        root = Path(raw_dir)

        valid = root / "valid.zip"
        write_zip(
            valid,
            [("folder/", b""), ("folder/ok.txt", b"hello"), ("月.txt", "妃".encode())],
        )
        output = root / "valid-output"
        extractor(valid, output, max_entries=10, max_file_size=32, max_total_size=64)
        assert (output / "folder" / "ok.txt").read_bytes() == b"hello"
        assert (output / "月.txt").read_text(encoding="utf-8") == "妃"

        traversal = root / "traversal.zip"
        write_zip(traversal, [("../escape.txt", b"bad")])
        assert_rejected(extractor, traversal, root / "traversal-output")
        assert not (root / "escape.txt").exists()

        backslash = root / "backslash.zip"
        write_zip(backslash, [("..\\escape.txt", b"bad")])
        assert_rejected(extractor, backslash, root / "backslash-output")

        absolute = root / "absolute.zip"
        write_zip(absolute, [("/absolute.txt", b"bad")])
        assert_rejected(extractor, absolute, root / "absolute-output")

        reserved = root / "reserved.zip"
        write_zip(reserved, [("CON.txt", b"bad")])
        assert_rejected(extractor, reserved, root / "reserved-output")

        reserved_spaced = root / "reserved-spaced.zip"
        write_zip(reserved_spaced, [("CON .txt", b"bad")])
        assert_rejected(extractor, reserved_spaced, root / "reserved-spaced-output")

        duplicate = root / "duplicate.zip"
        write_zip(duplicate, [("A.txt", b"one"), ("a.txt", b"two")])
        assert_rejected(extractor, duplicate, root / "duplicate-output")

        conflict = root / "conflict.zip"
        write_zip(conflict, [("node", b"file"), ("node/child.txt", b"child")])
        assert_rejected(extractor, conflict, root / "conflict-output")

        symlink = root / "symlink.zip"
        link = zipfile.ZipInfo("link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(symlink, "w") as archive:
            archive.writestr(link, "target")
        assert_rejected(extractor, symlink, root / "symlink-output")

        oversized = root / "oversized.zip"
        write_zip(oversized, [("large.bin", b"x" * 33)])
        assert_rejected(
            extractor,
            oversized,
            root / "oversized-output",
            max_file_size=32,
            max_total_size=64,
        )

        too_many = root / "too-many.zip"
        write_zip(too_many, [("one", b"1"), ("two", b"2")])
        assert_rejected(extractor, too_many, root / "too-many-output", max_entries=1)

        corrupt = root / "corrupt.zip"
        write_zip(corrupt, [("first.txt", b"first"), ("second.txt", b"second")])
        corrupt_member(corrupt, "second.txt")
        assert_rejected(extractor, corrupt, root / "corrupt-output")

        existing = root / "existing"
        existing.mkdir()
        try:
            extractor(valid, existing)
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing destination must fail")


if __name__ == "__main__":
    verify()
    print("technical verification passed")
