from __future__ import annotations

import json
import re
from datetime import datetime, time, timezone
from pathlib import Path


BATCH_DIR = Path(__file__).resolve().parent


def load_scheduler():
    approval = json.loads(
        (BATCH_DIR / "approved_sessions.json").read_text(encoding="utf-8")
    )
    session = next(
        item
        for item in approval["sessions"]
        if item["session_id"] == "timezone_daily_scheduler"
    )
    answer = session["messages"][-1]["content"]
    match = re.search(r"```python\n(.*?)\n```", answer, flags=re.DOTALL)
    if match is None:
        raise AssertionError("embedded Python implementation is missing")
    namespace: dict[str, object] = {}
    exec(compile(match.group(1), "<embedded-timezone-scheduler>", "exec"), namespace)
    return namespace["next_daily_run"]


def verify() -> None:
    next_daily_run = load_scheduler()

    assert next_daily_run(
        datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        time(9, 0),
        "Asia/Shanghai",
    ) == datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)

    assert next_daily_run(
        datetime(2026, 3, 8, 6, 0, tzinfo=timezone.utc),
        time(2, 30),
        "America/New_York",
    ) == datetime(2026, 3, 8, 7, 0, tzinfo=timezone.utc)

    assert next_daily_run(
        datetime(2026, 11, 1, 4, 0, tzinfo=timezone.utc),
        time(1, 30),
        "America/New_York",
    ) == datetime(2026, 11, 1, 5, 30, tzinfo=timezone.utc)

    assert next_daily_run(
        datetime(2026, 11, 1, 5, 45, tzinfo=timezone.utc),
        time(1, 30),
        "America/New_York",
    ) == datetime(2026, 11, 2, 6, 30, tzinfo=timezone.utc)

    assert next_daily_run(
        datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc),
        time(9, 0),
        "Asia/Shanghai",
    ) == datetime(2026, 1, 2, 1, 0, tzinfo=timezone.utc)

    try:
        next_daily_run(datetime(2026, 1, 1), time(9, 0), "Asia/Shanghai")
    except ValueError as exc:
        assert "timezone-aware" in str(exc)
    else:
        raise AssertionError("naive after_utc must fail")

    try:
        next_daily_run(
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            time(9, 0, tzinfo=timezone.utc),
            "Asia/Shanghai",
        )
    except ValueError as exc:
        assert "local_time" in str(exc)
    else:
        raise AssertionError("timezone-aware local_time must fail")

    try:
        next_daily_run(
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            time(9, 0, 1),
            "Asia/Shanghai",
        )
    except ValueError as exc:
        assert "minute precision" in str(exc)
    else:
        raise AssertionError("second-level local_time must fail")


if __name__ == "__main__":
    verify()
    print("technical verification passed")
