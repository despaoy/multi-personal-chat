"""Backend environment loading shared by every application entry point."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

_BACKEND_ROOT = Path(__file__).resolve().parents[1]


def load_backend_env(
    env_path: str | Path | None = None,
    *,
    override: bool = False,
) -> Path:
    """Load one dotenv file without replacing explicitly injected variables."""

    resolved = Path(env_path) if env_path is not None else _BACKEND_ROOT / ".env"
    load_dotenv(resolved, override=override)
    return resolved