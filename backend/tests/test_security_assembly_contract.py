"""Enabled security assembly must fail instead of silently disabling itself."""

import builtins
from types import SimpleNamespace

import httpx
import pytest
from app.main import create_app
from app.runtime import RuntimeContainer


def _runtime(environment="test", flag=None):
    env = {"ENVIRONMENT": environment, "ALLOWED_ORIGINS": "http://localhost:5000"}
    if flag is not None:
        env["SECURITY_MIDDLEWARE_ENABLED"] = flag
    return RuntimeContainer(db=SimpleNamespace(), is_pg_mode=lambda: False, startup_env=env)


@pytest.mark.parametrize("environment", ["development", "test", "production"])
def test_enabled_security_import_failure_stops_assembly_in_every_environment(monkeypatch, environment):
    original = builtins.__import__
    failure = ImportError("synthetic security dependency unavailable")

    def blocked(name, *args, **kwargs):
        if name == "middleware.security":
            raise failure
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError) as raised:
        create_app(_runtime(environment, "true"))
    assert raised.value is failure


@pytest.mark.parametrize("flag", ["false", "0"])
def test_explicitly_disabled_security_never_resolves_its_dependency(monkeypatch, flag):
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "middleware.security":
            pytest.fail("Explicitly disabled middleware must not import its dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    app = create_app(_runtime(flag=flag))
    assert [row.cls.__name__ for row in app.user_middleware] == ["CORSMiddleware"]


@pytest.mark.parametrize("environment", ["test", "production"])
@pytest.mark.parametrize("flag", ["tru", ""])
def test_invalid_flag_cannot_disable_security_by_accident(environment, flag):
    with pytest.raises(ValueError, match="SECURITY_MIDDLEWARE_ENABLED must be an explicit boolean"):
        create_app(_runtime(environment, flag))


@pytest.mark.parametrize("flag", [None, "true", " true ", "1"])
async def test_enabled_security_rejects_anonymous_requests_and_keeps_cors_outermost(flag):
    app = create_app(_runtime(flag=flag))
    names = {row.cls.__name__ for row in app.user_middleware}
    assert {
        "SecurityMiddleware",
        "InputValidationMiddleware",
        "RateLimitMiddleware",
        "SecurityHeadersMiddleware",
        "AuditLogMiddleware",
    } <= names
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://isolated") as client:
        response = await client.get("/api/stats", headers={"Origin": "http://localhost:5000"})
    assert response.status_code == 401
    assert response.headers["access-control-allow-origin"] == "http://localhost:5000"
    assert response.headers["x-content-type-options"] == "nosniff"
