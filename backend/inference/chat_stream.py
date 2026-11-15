"""Validate SSE completion boundaries while exposing provisional text deltas."""

import json


async def chat_deltas(lines, usage=None):
    data = []
    size = 0
    finished = False
    visible = False
    async for line in lines:
        if line.startswith(":"):
            continue
        if line:
            if line.startswith("data:"):
                value = line[5:].lstrip(" ")
                size += len(value)
                if size > 1_000_000:
                    raise ValueError("Completion event exceeds limit")
                data.append(value)
            continue
        if not data:
            continue
        raw = "\n".join(data)
        data, size = [], 0
        if raw == "[DONE]":
            if not finished or not visible:
                raise ValueError("Completion stream has no complete answer")
            return
        try:
            payload = json.loads(raw)
        except ValueError:
            raise ValueError("Malformed completion event") from None
        if not isinstance(payload, dict) or "error" in payload:
            raise ValueError("Invalid completion event")
        if payload.get("usage") is not None and usage is not None:
            usage.read(payload)
        choices = payload.get("choices")
        if choices == [] and payload.get("usage") is not None:
            continue
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ValueError("Invalid completion choices")
        choice = choices[0]
        if choice.get("index", 0) != 0 or finished:
            raise ValueError("Unexpected completion choice")
        delta = choice.get("delta")
        if not isinstance(delta, dict) or delta.get("tool_calls") or delta.get("function_call"):
            raise ValueError("Unsupported completion delta")
        content = delta.get("content")
        if content is not None:
            if not isinstance(content, str):
                raise ValueError("Invalid completion content")
            visible = visible or bool(content.strip())
            if content:
                yield content
        reason = choice.get("finish_reason")
        if reason is not None:
            if reason != "stop":
                raise ValueError("Completion stream did not finish normally")
            finished = True
    raise ValueError("Completion stream disconnected before DONE")
