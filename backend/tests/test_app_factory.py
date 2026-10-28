from __future__ import annotations

from infra.resource_pool import HttpClientPool

from types import SimpleNamespace

import pytest


def test_create_app_uses_the_supplied_runtime_container() -> None:
    from app.main import create_app
    from app.runtime import RuntimeContainer, get_runtime_container
    runtime = RuntimeContainer(db=SimpleNamespace(), is_pg_mode=lambda: False, inference_runtime=None, startup_env={}, http_client_pool=HttpClientPool(max_connections=1))
    application = create_app(runtime)
    assert get_runtime_container(application) is runtime
    paths = set(application.openapi()['paths'])
    assert {'/health', '/ready', '/api/generate'} <= paths


def test_runtime_default_preserves_an_explicit_empty_environment(monkeypatch) -> None:
    from app.runtime import RuntimeContainer

    monkeypatch.setenv("QQCHAT_RUNTIME_CONTAINER_TEST", "must-not-leak")

    runtime = RuntimeContainer.default(startup_env={})

    assert runtime.startup_env == {}


@pytest.mark.asyncio
async def test_generate_endpoint_uses_runtime_container_inference_runtime() -> None:
    from fastapi import Request
    from api.generate import generate_reply, get_request_chat_generation_service
    from app.main import create_app
    from app.runtime import RuntimeContainer
    from db.schemas import GenerateResponse, MessageRequest
    calls: list[tuple] = []
    expected = GenerateResponse(reply='container-runtime', model='fake', costTime=0.0)

    class Runtime:

        def __bool__(self) -> bool:
            return False

        def priority_for(self, platform: str, session_type: str) -> int:
            calls.append(('priority', platform, session_type))
            return 5

        async def check_rate_limits(self, platform: str, session_id: str, identity: str) -> None:
            calls.append(('rate-limit', platform, session_id, identity))

        async def submit(self, _job, *, session_id: str, priority: int):
            calls.append(('submit', session_id, priority))
            return expected
    runtime = Runtime()
    container = RuntimeContainer(db=SimpleNamespace(), is_pg_mode=lambda: False, inference_runtime=runtime, startup_env={}, http_client_pool=HttpClientPool(max_connections=1))
    application = create_app(container)
    http_request = Request({'type': 'http', 'app': application})
    service = get_request_chat_generation_service(http_request)
    response = await generate_reply(MessageRequest(message='hello', sessionType='group'), {'user_id': 17}, service)
    assert response is expected
    assert calls == [('priority', 'admin', 'group'), ('rate-limit', 'admin', '["manual","17",""]', '17'), ('submit', '["manual","17",""]', 5)]
def test_shared_environment_loader_preserves_injected_values(tmp_path, monkeypatch) -> None:
    from app.env import load_backend_env

    env_file = tmp_path / ".env"
    env_file.write_text(
        'QQCHAT_EXISTING="from-file"\nQQCHAT_QUOTED="value with spaces"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("QQCHAT_EXISTING", "injected")
    monkeypatch.delenv("QQCHAT_QUOTED", raising=False)

    assert load_backend_env(env_file) == env_file
    assert __import__("os").environ["QQCHAT_EXISTING"] == "injected"
    assert __import__("os").environ["QQCHAT_QUOTED"] == "value with spaces"


def test_run_entrypoint_does_not_delete_bytecode_or_reparse_dotenv() -> None:
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "run.py").read_text(encoding="utf-8")

    assert "load_backend_env()" in source
    assert "shutil.rmtree" not in source
    assert "def _load_env" not in source

def test_generate_service_binds_character_context_to_container_db() -> None:
    """生成链路的角色上下文服务必须绑定当前应用容器的数据库。

    管理接口已走容器数据库；若生成链路（prepare/complete_turn）
    仍用全局服务，create_app(custom_container) 的实例会把人物
    记忆/关系读写到全局数据库，与管理接口脱节。
    """
    from fastapi import Request
    from api.generate import get_request_chat_generation_service
    from app.main import create_app
    from app.runtime import RuntimeContainer
    from infra.concurrency_control import inference_runtime
    container = RuntimeContainer(db=SimpleNamespace(), is_pg_mode=lambda: False, inference_runtime=inference_runtime, startup_env={}, http_client_pool=HttpClientPool(max_connections=1))
    application = create_app(container)
    http_request = Request({'type': 'http', 'app': application})
    service = get_request_chat_generation_service(http_request)
    handler = service._generate_handler
    assert handler.keywords['message_db'] is container.db
    assert handler.keywords['character_service'] is None


def test_lifespan_resource_references_are_cleared_between_application_runs() -> None:
    from app.main import _clear_lifespan_resource_references

    config = SimpleNamespace(
        http_client_pool=object(),
        backup_mgr=object(),
        access_control_mgr=object(),
        unrelated="preserved",
    )

    _clear_lifespan_resource_references(config)

    assert config.http_client_pool is None
    assert config.backup_mgr is None
    assert config.access_control_mgr is None
    assert config.unrelated == "preserved"


@pytest.mark.parametrize('missing', ['database', 'scheduler', 'both'])
def test_generate_service_rejects_missing_dependencies(missing) -> None:
    from fastapi import HTTPException, Request
    from api.generate import get_request_chat_generation_service
    from app.main import create_app
    from app.runtime import RuntimeContainer
    container = RuntimeContainer(db=None if missing in {'database', 'both'} else SimpleNamespace(), is_pg_mode=lambda: False, inference_runtime=None if missing in {'scheduler', 'both'} else SimpleNamespace(), startup_env={}, http_client_pool=HttpClientPool(max_connections=1))
    request = Request({'type': 'http', 'app': create_app(container)})
    with pytest.raises(HTTPException) as error:
        get_request_chat_generation_service(request)
    assert error.value.status_code == 503
    assert error.value.detail['stage'] == 'dependencies'
