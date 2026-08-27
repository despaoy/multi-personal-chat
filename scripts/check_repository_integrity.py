"""Verify repository structure, entrypoints, frozen data, and documentation links."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl_count(path: Path) -> int:
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())


def _check_frozen_dataset(errors: list[str]) -> None:
    base = ROOT / "backend/data/character_dialogues/experiments/v4"
    manifest = json.loads((base / "canonical_dataset_manifest.json").read_text(encoding="utf-8"))
    for split in ("train", "validation"):
        path = ROOT / manifest[split]["path"]
        actual_hash = _sha256(path)
        if actual_hash != manifest[split]["sha256"]:
            errors.append(f"{split} hash mismatch: {actual_hash}")
        actual_count = _jsonl_count(path)
        if actual_count != manifest[split]["count"]:
            errors.append(f"{split} count mismatch: {actual_count}")


def _check_api_mounts(errors: list[str]) -> None:
    api_modules = {
        path.stem
        for path in (ROOT / "backend/api").glob("*.py")
        if path.name != "__init__.py" and "APIRouter(" in path.read_text(encoding="utf-8")
    }
    main_text = (ROOT / "backend/app/main.py").read_text(encoding="utf-8")
    mounted = set(re.findall(r"^from api\.([a-zA-Z0-9_]+) import router as ", main_text, re.MULTILINE))
    if missing := sorted(api_modules - mounted):
        errors.append(f"API modules not mounted: {', '.join(missing)}")
    if stale := sorted(mounted - api_modules):
        errors.append(f"mounted API modules missing on disk: {', '.join(stale)}")


def _check_frontend_navigation(errors: list[str]) -> None:
    app_root = ROOT / "src/app"
    page_routes: set[str] = set()
    for path in app_root.rglob("page.tsx"):
        relative = path.parent.relative_to(app_root).as_posix()
        route = "/" if relative == "." else f"/{relative}"
        if route != "/login" and "[" not in route:
            page_routes.add(route)
    sidebar = (ROOT / "src/components/layout/Sidebar.tsx").read_text(encoding="utf-8")
    linked = set(re.findall(r"href:\s*['\"]([^'\"]+)['\"]", sidebar))
    if missing := sorted(page_routes - linked):
        errors.append(f"frontend pages missing from navigation: {', '.join(missing)}")


def _check_script_indexes(errors: list[str]) -> None:
    indexes = (
        (ROOT / "scripts", ROOT / "scripts/README.md", {"__init__.py", "check_repository_integrity.py"}),
        (ROOT / "backend/scripts", ROOT / "backend/scripts/README.md", set()),
        (ROOT / "backend/benchmarks", ROOT / "backend/benchmarks/README.md", {"__init__.py"}),
    )
    for directory, readme_path, excluded in indexes:
        readme = readme_path.read_text(encoding="utf-8")
        active = {
            path.name
            for path in directory.iterdir()
            if path.is_file() and path.suffix in {".py", ".ps1", ".sh"} and path.name not in excluded
        }
        if missing := sorted(name for name in active if f"`{name}`" not in readme):
            errors.append(f"{readme_path.relative_to(ROOT)} missing scripts: {', '.join(missing)}")


def _check_readme_links(errors: list[str]) -> None:
    pattern = re.compile(r"!?(?:\[[^\]]*\])\(([^)]+)\)")
    for readme in ROOT.rglob("README.md"):
        if any(part in {"node_modules", ".git"} for part in readme.parts):
            continue
        text = readme.read_text(encoding="utf-8")
        for raw_target in pattern.findall(text):
            target = raw_target.strip().strip("<>").split("#", 1)[0]
            if not target or re.match(r"^(?:https?://|mailto:)", target):
                continue
            target = unquote(target)
            resolved = (readme.parent / target).resolve()
            try:
                resolved.relative_to(ROOT)
            except ValueError:
                errors.append(f"{readme.relative_to(ROOT)} link escapes repository: {raw_target}")
                continue
            if not resolved.exists():
                errors.append(f"{readme.relative_to(ROOT)} broken link: {raw_target}")


def _check_archive_index(errors: list[str]) -> None:
    index_path = ROOT / "scripts/archive/INDEX.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    entries = index.get("entries", [])
    if index.get("entry_count") != len(entries):
        errors.append("scripts/archive/INDEX.json entry_count mismatch")
    indexed_paths = {entry["archived_path"] for entry in entries}
    actual_paths = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "scripts/archive").rglob("*")
        if path.is_file() and path != index_path
    }
    if missing := sorted(indexed_paths - actual_paths):
        errors.append(f"archive entries missing on disk: {', '.join(missing)}")
    if unindexed := sorted(actual_paths - indexed_paths):
        errors.append(f"archive files not indexed: {', '.join(unindexed)}")
    for entry in entries:
        path = ROOT / entry["archived_path"]
        if not path.exists():
            continue
        if entry.get("bytes") != path.stat().st_size:
            errors.append(f"archive size mismatch: {entry['archived_path']}")
        if entry.get("sha256") != _sha256(path):
            errors.append(f"archive hash mismatch: {entry['archived_path']}")


def _check_tracked_artifacts(errors: list[str]) -> None:
    result = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8"
    )
    forbidden = re.compile(r"(?:^|/)(?:__pycache__|\.pytest_cache|\.next|node_modules)(?:/|$)|\.pyc$")
    tracked = [path for path in result.stdout.splitlines() if forbidden.search(path)]
    if tracked:
        errors.append(f"tracked generated artifacts: {', '.join(tracked[:10])}")


def main() -> int:
    errors: list[str] = []
    _check_frozen_dataset(errors)
    _check_api_mounts(errors)
    _check_frontend_navigation(errors)
    _check_script_indexes(errors)
    _check_readme_links(errors)
    _check_archive_index(errors)
    _check_tracked_artifacts(errors)
    if errors:
        print("Repository integrity check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Repository integrity check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
