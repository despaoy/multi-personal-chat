from __future__ import annotations

from types import SimpleNamespace

from starlette.middleware.cors import CORSMiddleware

from app.main import create_app
from app.runtime import RuntimeContainer


def _runtime(startup_env: dict[str, str]) -> RuntimeContainer:
    return RuntimeContainer(
        db=SimpleNamespace(),
        is_pg_mode=lambda: False,
        inference_runtime=None,
        startup_env=startup_env,
    )


def _cors_origins(application) -> list[str]:
    middleware = next(
        item for item in application.user_middleware if item.cls is CORSMiddleware
    )
    return middleware.kwargs["allow_origins"]


def test_create_app_uses_its_runtime_environment_for_middleware(monkeypatch) -> None:
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://ambient.example")
    monkeypatch.setenv("SECURITY_MIDDLEWARE_ENABLED", "true")

    application = create_app(
        _runtime(
            {
                "ALLOWED_ORIGINS": "https://one.example, https://two.example",
                "SECURITY_MIDDLEWARE_ENABLED": "false",
            }
        )
    )

    assert _cors_origins(application) == [
        "https://one.example",
        "https://two.example",
    ]
    assert [item.cls for item in application.user_middleware] == [CORSMiddleware]


def test_application_factories_do_not_share_cors_environment() -> None:
    first = create_app(
        _runtime(
            {
                "ALLOWED_ORIGINS": "https://first.example",
                "SECURITY_MIDDLEWARE_ENABLED": "false",
            }
        )
    )
    second = create_app(
        _runtime(
            {
                "ALLOWED_ORIGINS": "https://second.example",
                "SECURITY_MIDDLEWARE_ENABLED": "false",
            }
        )
    )

    assert _cors_origins(first) == ["https://first.example"]
    assert _cors_origins(second) == ["https://second.example"]
