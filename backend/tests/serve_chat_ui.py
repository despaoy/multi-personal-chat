"""Local-only UI acceptance server with isolated data and a synthetic model.

Run from backend: python -m tests.serve_chat_ui (or execute this file directly).
This harness is never used by the production entry point.
"""

import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
RUNTIME = ROOT / ".codex-test-runtime-ui"
RUNTIME.mkdir(exist_ok=True)
os.environ.update({
    "ENVIRONMENT": "test", "USE_POSTGRESQL": "false", "DATABASE_URL": "",
    "DATABASE_PATH": str(RUNTIME / "bootstrap.sqlite"),
    "JWT_SECRET": "isolated-ui-acceptance-only-secret-20261115",
    "ENCRYPTION_KEY": "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8=",
    "ALLOW_PUBLIC_REGISTRATION": "true", "MEMORY_LLM_ENABLED": "false",
    "RELATIONSHIP_LLM_ENABLED": "false", "MODEL_PROVIDER": "openai_compat",
})


async def main():
    import httpx
    import pytest
    import uvicorn
    from infra.concurrency_control import InferenceRuntime

    spec = importlib.util.spec_from_file_location("ui_fixture", Path(__file__).with_name("test_application_lifespan_ownership.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    patch = pytest.MonkeyPatch()
    fixture = module.applications.__wrapped__(RUNTIME, patch)
    state = await anext(fixture)
    state.containers[0].inference_runtime = InferenceRuntime()
    try:
        async with state.running(0):
            class SlowBody(httpx.AsyncByteStream):
                async def __aiter__(self):
                    for text in ["这是慢速验收回复。", "第一段已经显示，", "刷新后仍应继续，", "最后只保存一条完整消息。"]:
                        event = {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}
                        yield ("data: " + json.dumps(event) + "\n\n").encode()
                        await asyncio.sleep(5)
                    event = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                             "usage": {"prompt_tokens": 40, "completion_tokens": 20, "total_tokens": 60}}
                    yield ("data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n").encode()

            def respond(request):
                return httpx.Response(200, stream=SlowBody(), headers={"Content-Type": "text/event-stream"})

            await state.pools[0]._client.aclose()
            state.pools[0]._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
            print("Synthetic-model UI acceptance server: http://127.0.0.1:8100", flush=True)
            server = uvicorn.Server(uvicorn.Config(state.apps[0], host="127.0.0.1", port=8100, lifespan="off"))
            await server.serve()
    finally:
        await fixture.aclose()
        patch.undo()


if __name__ == "__main__":
    asyncio.run(main())
