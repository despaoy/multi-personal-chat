"""Application runtime dependency container.

The container is intentionally small.  It gives the composition root one
explicit place to select process-wide dependencies while existing API modules
continue to use their compatibility imports during the incremental migration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


@dataclass(slots=True)
class RuntimeContainer:
    """Dependencies owned by one FastAPI application instance."""

    db: Any
    is_pg_mode: Callable[[], bool]
    inference_runtime: Any | None = None
    startup_env: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))

    def __post_init__(self) -> None:
        # Keep startup validation deterministic even if tests or callers mutate
        # os.environ after constructing the application.
        self.startup_env = dict(self.startup_env)

    @classmethod
    def default(
        cls,
        *,
        startup_env: Mapping[str, str] | None = None,
    ) -> "RuntimeContainer":
        """Build the production-compatible dependency set lazily."""

        from db.adapter import db, is_pg_mode
        from infra.concurrency_control import inference_runtime

        return cls(
            db=db,
            is_pg_mode=is_pg_mode,
            inference_runtime=inference_runtime,
            startup_env=dict(os.environ if startup_env is None else startup_env),
        )


def get_runtime_container(app: Any) -> RuntimeContainer:
    """Return the container attached by ``create_app``."""

    container = getattr(app.state, "runtime_container", None)
    if not isinstance(container, RuntimeContainer):
        raise RuntimeError("FastAPI runtime container is not configured")
    return container
