"""Repository links cover source documents rather than ignored working copies."""

import importlib.util
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def checker(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts/check_repository_integrity.py"
    spec = importlib.util.spec_from_file_location("repository_integrity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    return module


def test_ignored_readme_copies_do_not_invalidate_repository_links(checker):
    root = checker.ROOT
    (root / ".gitignore").write_text(".tmp/\nnode_modules/\nlocal-notes/\n", encoding="utf-8")
    (root / "README.md").write_text("[说明](guide.md)", encoding="utf-8")
    (root / "guide.md").write_text("有效文档。", encoding="utf-8")
    subprocess.run(["git", "add", "--", "README.md"], cwd=root, check=True, capture_output=True)
    for directory in [".tmp/snapshot", "node_modules/fixture", "local-notes"]:
        path = root / directory / "README.md"
        path.parent.mkdir(parents=True)
        path.write_text("[副本缺少的文件](missing.md)", encoding="utf-8")
    errors = []
    checker._check_readme_links(errors)
    assert errors == []


@pytest.mark.parametrize("tracked", [False, True])
def test_real_links_are_checked_in_tracked_and_new_documents(checker, tracked):
    readme = checker.ROOT / "新文档 说明" / "README.md"
    readme.parent.mkdir()
    readme.write_text("[需要的文档](missing.md)", encoding="utf-8")
    if tracked:
        subprocess.run(["git", "add", "--", str(readme.relative_to(checker.ROOT))],
                       cwd=checker.ROOT, check=True, capture_output=True)
    errors = []
    checker._check_readme_links(errors)
    assert errors == [f"{readme.relative_to(checker.ROOT)} broken link: missing.md"]
    (readme.parent / "missing.md").write_text("已补齐。", encoding="utf-8")
    errors = []
    checker._check_readme_links(errors)
    assert errors == []
