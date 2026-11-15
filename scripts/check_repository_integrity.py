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
    return sum(
        bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines()
    )


def _check_frozen_dataset(errors: list[str]) -> None:
    base = ROOT / "backend/data/character_dialogues/experiments/v4"
    manifest = json.loads(
        (base / "canonical_dataset_manifest.json").read_text(encoding="utf-8")
    )
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
        if path.name != "__init__.py"
        and "APIRouter(" in path.read_text(encoding="utf-8")
    }
    main_text = (ROOT / "backend/app/main.py").read_text(encoding="utf-8")
    mounted = set(
        re.findall(
            r"^from api\.([a-zA-Z0-9_]+) import router as ", main_text, re.MULTILINE
        )
    )
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
    # Navigation declarations moved out of the rendering component.
    navigation = (ROOT / "src/lib/navigation.ts").read_text(encoding="utf-8")
    linked = set(re.findall(r"href:\s*['\"]([^'\"]+)['\"]", navigation))
    if missing := sorted(page_routes - linked):
        errors.append(f"frontend pages missing from navigation: {', '.join(missing)}")


def _check_script_indexes(errors: list[str]) -> None:
    published = set(filter(None, subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT,
    ).decode('utf-8').split('\0')))
    indexes = (
        (
            ROOT / "scripts",
            ROOT / "scripts/README.md",
            {"__init__.py", "check_repository_integrity.py"},
        ),
        (ROOT / "backend/scripts", ROOT / "backend/scripts/README.md", set()),
        (
            ROOT / "backend/benchmarks",
            ROOT / "backend/benchmarks/README.md",
            {"__init__.py"},
        ),
    )
    for directory, readme_path, excluded in indexes:
        readme = readme_path.read_text(encoding="utf-8")
        active = {
            path.name
            for path in directory.iterdir()
            if path.is_file()
            and path.relative_to(ROOT).as_posix() in published
            and path.suffix in {".py", ".ps1", ".sh"}
            and path.name not in excluded
        }
        if missing := sorted(name for name in active if f"`{name}`" not in readme):
            errors.append(
                f"{readme_path.relative_to(ROOT)} missing scripts: {', '.join(missing)}"
            )


def _check_readme_links(errors: list[str]) -> None:
    pattern = re.compile(r"!?(?:\[[^\]]*\])\(([^)]+)\)")
    tracked = set(filter(None, subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT,
    ).decode('utf-8').split('\0')))
    paths = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "README.md", "**/README.md"],
        cwd=ROOT,
    ).decode("utf-8").split("\0")
    for name in filter(None, paths):
        readme = ROOT / name
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
                errors.append(
                    f"{readme.relative_to(ROOT)} link escapes repository: {raw_target}"
                )
                continue
            relative = resolved.relative_to(ROOT).as_posix()
            available = relative in tracked or any(name.startswith(relative.rstrip('/') + '/') for name in tracked)
            if not resolved.exists() or not available:
                errors.append(f"{readme.relative_to(ROOT)} broken link: {raw_target}")






def _check_tracked_artifacts(errors: list[str]) -> None:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    forbidden = re.compile(
        r"(?:^|/)(?:__pycache__|\.pytest_cache|\.next|node_modules)(?:/|$)"
        r"|\.pyc$"
        r"|^local-notes/"
        r"|^archive/p6_rag_pipeline/artifacts/"
        r"|^backend/data/eval/rag_retrieval/reports/archive/"
        r"|^backend/data/eval/rag_retrieval/reports/[^/]*_multiscale_[^/]*\.json$"
        r"|^docs/research/POSTGRADUATE_[^/]*\.md$"
    )
    tracked = [
        path
        for path in result.stdout.splitlines()
        if (ROOT / path).exists() and forbidden.search(path)
    ]
    if tracked:
        errors.append(f"tracked generated artifacts: {', '.join(tracked[:10])}")


def main() -> int:
    errors: list[str] = []
    _check_frozen_dataset(errors)
    _check_api_mounts(errors)
    _check_frontend_navigation(errors)
    _check_script_indexes(errors)
    _check_readme_links(errors)
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
