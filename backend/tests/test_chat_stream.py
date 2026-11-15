import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest

from inference.chat_stream import chat_deltas
from inference.model_manager import OpenAICompatProvider
from inference.openai_protocol import ProviderResponseError
from inference.usage import CompletionUsage, collect_usage


def chunk(content=None, finish=None, **extra):
    return {"choices": [{"index": 0, "delta": {"content": content}, "finish_reason": finish}], **extra}


def consume(events):
    usage = CompletionUsage()

    async def lines():
        yield ": keepalive"
        yield ""
        for item in events:
            yield "data: " + (item if isinstance(item, str) else json.dumps(item))
            yield ""

    async def run():
        return [part async for part in chat_deltas(lines(), usage)]

    return asyncio.run(run()), usage.tokens


def test_content_and_final_chunk_usage():
    parts, usage = consume([
        chunk("你"), chunk("好"),
        chunk(finish="stop", usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}), "[DONE]",
    ])
    assert parts == ["你", "好"] and usage == (2, 3, 5)


def test_separate_usage_chunk():
    assert consume([chunk("你好", "stop"), {"choices": [], "usage": {
        "prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5,
    }}, "[DONE]"])[1] == (2, 3, 5)


@pytest.mark.parametrize("events", [
    [chunk("partial")], [chunk("partial", "length"), "[DONE]"],
    [chunk("partial"), "[DONE]"], [chunk(finish="stop"), "[DONE]"],
    [chunk("ok", "stop"), chunk("extra"), "[DONE]"],
    ["invalid JSON"], [{"error": {"message": "private upstream detail"}}],
])
def test_incomplete_or_invalid_stream_is_not_success(events):
    with pytest.raises(ValueError):
        consume(events)


@pytest.mark.parametrize("mode", ["complete", "disconnect", "cancel", "http_error"])
def test_provider_stream_delivers_before_end_and_closes_transport(mode):
    async def exercise():
        release = asyncio.Event()
        wire = []

        class Body(httpx.AsyncByteStream):
            closed = False

            async def __aiter__(self):
                # Split a UTF-8 event across transport chunks too.
                first = ("data: " + json.dumps(chunk("你好"), ensure_ascii=False) + "\n\n").encode()
                for part in [first[:3], first[3:]]:
                    yield part
                await release.wait()
                if mode == "disconnect":
                    return
                final = chunk(finish="stop", usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5})
                yield ("data: " + json.dumps(final) + "\n\ndata: [DONE]\n\n").encode()

            async def aclose(self):
                self.closed = True

        body = Body()

        def respond(request):
            wire.append(json.loads(request.content))
            return httpx.Response(429 if mode == "http_error" else 200, stream=body)

        provider = OpenAICompatProvider({"openaiCompatBaseUrl": "https://api.deepseek.com",
                                        "openaiCompatApiKey": "synthetic-test", "openaiCompatModel": "deepseek-flash"})

        @asynccontextmanager
        async def client(**kwargs):
            async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as value:
                yield value

        provider._acquire_http_client = client
        with collect_usage() as ledger:
            stream = provider.async_stream([{"role": "user", "content": "你好"}], temperature=0.7, max_tokens=128)
            if mode == "http_error":
                with pytest.raises(ProviderResponseError):
                    await anext(stream)
            else:
                assert await asyncio.wait_for(anext(stream), 2) == "你好"
                assert not release.is_set() and ledger.attempts[0].tokens is None
                if mode == "cancel":
                    await stream.aclose()
                else:
                    release.set()
                    if mode == "disconnect":
                        with pytest.raises(ValueError, match="disconnected"):
                            await anext(stream)
                    else:
                        with pytest.raises(StopAsyncIteration):
                            await anext(stream)
                        assert ledger.attempts[0].tokens == (2, 3, 5)
            assert body.closed
        assert len(wire) == 1
        assert wire[0]["stream"] is True
        assert wire[0]["thinking"] == {"type": "disabled"}

    asyncio.run(exercise())


def test_generation_bridge_resets_drafts_and_never_claims_persistence():
    from services.character_chat import collect_stream

    async def exercise():
        events = []
        closed = []

        async def sink(event):
            events.append(event)

        async def source(fail=False):
            try:
                yield "第一段"
                if fail:
                    raise ValueError("interrupted")
                yield "第二段"
            finally:
                closed.append(True)

        assert await collect_stream(source(), sink) == "第一段第二段"
        with pytest.raises(ValueError):
            await collect_stream(source(True), sink)
        assert sum(event["type"] == "draft_reset" for event in events) == 2
        assert all(event["type"] in {"draft_reset", "draft_delta"} for event in events)
        assert len(closed) == 2

    asyncio.run(exercise())


def test_stream_bridge_batches_small_tokens_without_losing_text():
    from services.character_chat import collect_stream

    async def exercise():
        events = []

        async def sink(event):
            events.append(event)

        async def source():
            for index in range(100):
                yield str(index % 10)

        result = await collect_stream(source(), sink, clock=lambda: 1.0)
        deltas = [event["text"] for event in events if event["type"] == "draft_delta"]
        assert "".join(deltas) == result == "0123456789" * 10
        assert len(deltas) == 2

    asyncio.run(exercise())
