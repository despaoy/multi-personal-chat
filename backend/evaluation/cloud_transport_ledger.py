"""Persist every actual evaluation send before awaiting its provider response."""

import json


async def record_cloud_send(
    root, calls, payload, budget, client, request, sender, state, send_with_balance_stop, **kwargs
):
    if state.get("blocked_http_status") == 402:
        return await send_with_balance_stop(client, request, sender, state, **kwargs)

    def save():
        path = root / "cloud-calls.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes((json.dumps(calls, ensure_ascii=False, indent=2) + chr(10)).encode())
        temporary.replace(path)

    record = dict(request=payload, http_status=None, response=None, budget=budget, request_started=True)
    calls.append(record)
    save()
    try:
        response = await send_with_balance_stop(client, request, sender, state, **kwargs)
        record["http_status"] = response.status_code
        save()
        await response.aread()
        try:
            record["response"] = response.json()
        except ValueError:
            record["response_text"] = response.text
            raise
        save()
        return response
    except BaseException as error:
        record["transport_error"] = type(error).__name__
        save()
        raise
    finally:
        (root / "provider-access-state.json").write_bytes((json.dumps(state, indent=2) + chr(10)).encode())
