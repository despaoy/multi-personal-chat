"""collect_kisaki_v5_simulation_decisions.py 的单元测试。

覆盖：
- 勾选解析：keep/exclude/revise 三种合法勾选（x/X/✓）
- 拒绝：未选择、重复选择
- revise → exclude + needs_revision
- ID 覆盖：未知 ID / 缺失 ID 拒绝
- 产出 review_status=draft（不算批准），--decisions 只认 approved
"""

import importlib.util
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _module():
    spec = importlib.util.spec_from_file_location(
        "collect_kisaki_v5_simulation_decisions",
        PROJECT_ROOT / "scripts/collect_kisaki_v5_simulation_decisions.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_batch(path: Path, records: list[tuple[str, str]]) -> None:
    """records: [(id, choice_line)]，choice_line 如 '[x] keep  [ ] exclude  [ ] revise'"""
    lines = ["# 测试批次", ""]
    for rid, choice in records:
        lines += [
            "## 场景",
            f"- ID: `{rid}`",
            f"- **人工选择**: {choice}",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_packet(path: Path, ids: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "packet_id": "TEST-PACKET",
                "generated_at": "2026-01-01T00:00:00+00:00",
                "entries": [{"id": i} for i in ids],
            }
        ),
        encoding="utf-8",
    )


KEEP = "[x] keep  [ ] exclude  [ ] revise"
EXCLUDE = "[ ] keep  [X] exclude  [ ] revise"
REVISE = "[ ] keep  [ ] exclude  [✓] revise"
NONE = "[ ] keep  [ ] exclude  [ ] revise"
DOUBLE = "[x] keep  [x] exclude  [ ] revise"


def test_parse_all_three_choices(tmp_path):
    mod = _module()
    batch = tmp_path / "batch_01.md"
    _write_batch(batch, [("a", KEEP), ("b", EXCLUDE), ("c", REVISE)])
    choices = mod.parse_batches(tmp_path)
    assert choices == {"a": "keep", "b": "exclude", "c": "revise"}


def test_parse_rejects_unselected(tmp_path):
    mod = _module()
    _write_batch(tmp_path / "batch_01.md", [("a", KEEP), ("b", NONE)])
    with pytest.raises(SystemExit, match="未勾选任何选项"):
        mod.parse_batches(tmp_path)


def test_parse_rejects_double_selection(tmp_path):
    mod = _module()
    _write_batch(tmp_path / "batch_01.md", [("a", DOUBLE)])
    with pytest.raises(SystemExit, match="只能选一项"):
        mod.parse_batches(tmp_path)


def test_parse_rejects_choice_without_id(tmp_path):
    mod = _module()
    (tmp_path / "batch_01.md").write_text("- **人工选择**: [x] keep  [ ] exclude  [ ] revise\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="没有前置 ID 行"):
        mod.parse_batches(tmp_path)


def test_parse_rejects_trailing_id_without_choice(tmp_path):
    mod = _module()
    (tmp_path / "batch_01.md").write_text("- ID: `a`\n- 对话\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="没有选择行"):
        mod.parse_batches(tmp_path)


def test_coverage_rejects_unknown_and_missing(tmp_path):
    mod = _module()
    # unknown
    with pytest.raises(SystemExit, match="未知 ID"):
        mod.validate_coverage({"a": "keep", "ghost": "keep"}, {"a"})
    # missing
    with pytest.raises(SystemExit, match="未出现在勾选"):
        mod.validate_coverage({"a": "keep"}, {"a", "b"})


def test_draft_document_converts_revise_and_marks_draft(tmp_path):
    mod = _module()
    batch = tmp_path / "batch_01.md"
    _write_batch(batch, [("a", KEEP), ("b", EXCLUDE), ("c", REVISE)])
    packet = tmp_path / "packet.json"
    _write_packet(packet, ["a", "b", "c"])

    choices = mod.parse_batches(tmp_path)
    mod.validate_coverage(choices, {"a", "b", "c"})
    doc = mod.build_draft_document(choices, tmp_path, json.loads(packet.read_text(encoding="utf-8")))

    assert doc["review_status"] == "draft"
    assert doc["reviewed_by"] is None
    assert doc["decisions"] == {"a": "keep", "b": "exclude", "c": "exclude"}
    assert doc["needs_revision"] == ["c"]
    assert doc["stats"] == {"total": 3, "keep": 1, "exclude": 2, "needs_revision": 1}
    assert "approved" in doc["note"]


def test_draft_is_not_accepted_by_decision_gate():
    """draft 文档必须被 build_kisaki_v5_candidate 的决定门禁拒绝"""
    builder_spec = importlib.util.spec_from_file_location(
        "build_kisaki_v5_candidate",
        PROJECT_ROOT / "scripts/build_kisaki_v5_candidate.py",
    )
    builder = importlib.util.module_from_spec(builder_spec)
    builder_spec.loader.exec_module(builder)

    draft_doc = {
        "review_status": "draft",
        "reviewed_by": "owner",
        "decisions": {"a": "keep"},
    }
    with pytest.raises(SystemExit, match="review_status"):
        builder.validate_decision_document(draft_doc, {"a"})


def test_end_to_end_collect_main(tmp_path, monkeypatch, capsys):
    mod = _module()
    batch = tmp_path / "batch_01.md"
    _write_batch(batch, [("a", KEEP), ("b", REVISE)])
    packet = tmp_path / "packet.json"
    _write_packet(packet, ["a", "b"])
    output = tmp_path / "decisions.json"

    rc = (
        mod.main_with_args(packet=packet, batches_dir=tmp_path, output=output)
        if hasattr(mod, "main_with_args")
        else None
    )
    if rc is None:
        # 通过 argv 驱动 main
        import sys as _sys

        monkeypatch.setattr(
            _sys,
            "argv",
            ["collect", "--packet", str(packet), "--batches-dir", str(tmp_path), "--output", str(output)],
        )
        rc = mod.main()
    assert rc == 0
    doc = json.loads(output.read_text(encoding="utf-8"))
    assert doc["review_status"] == "draft"
    assert doc["decisions"] == {"a": "keep", "b": "exclude"}
    assert doc["needs_revision"] == ["b"]
