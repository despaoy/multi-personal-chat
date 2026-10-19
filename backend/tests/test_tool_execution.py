"""Real isolated tool behavior; no network, credentials or model calls."""

import os
from unittest.mock import patch

import pytest

from services.tool_execution import run_tool_code

pytestmark = pytest.mark.skipif(os.name != "posix", reason="Native execution resource limits require POSIX")


def test_combined_outputs_are_complete_without_pipe_backpressure(monkeypatch):
    monkeypatch.setenv("CLAW_EXECUTION_TIMEOUT", "2")
    result = run_tool_code('print("x" * 40000)\nreturn "y" * 40000', {})
    assert result == {"success": True, "output": "x" * 40000 + "\n", "error": "", "result": "y" * 40000}


@pytest.mark.parametrize("code", ['print("x" * 64001)', 'return "y" * 64001'])
def test_exceeded_output_is_failure_instead_of_truncated_success(code):
    result = run_tool_code(code, {})
    assert result["success"] is False and result["result"] == ""
    assert "exceeds output limit" in result["error"]


@pytest.mark.parametrize("key,value", [("CLAW_MEMORY_LIMIT_MB", "invalid"), ("CLAW_EXECUTION_TIMEOUT", "0.1")])
def test_invalid_limits_fail_before_starting_code(monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with patch("services.tool_execution.subprocess.run", side_effect=AssertionError("must not start")):
        with pytest.raises(ValueError):
            run_tool_code("return 42", {})


@pytest.mark.parametrize("field", [
    "_sys.modules[os].environ[CLAW_SYNTHETIC_PARENT_SENTINEL]",
    "_sys.modules[api.claw]",
])
def test_worker_has_no_parent_secret_or_api_modules(monkeypatch, field):
    monkeypatch.setenv("CLAW_SYNTHETIC_PARENT_SENTINEL", "synthetic-parent-marker")
    code = 'import collections\nreturn "{0.' + field + '}".format(collections)'
    result = run_tool_code(code, {})
    assert result["success"] is False and result["result"] == ""
    assert "KeyError" in result["error"] and "synthetic-parent-marker" not in result["error"]



def test_failed_resource_enforcement_stops_before_code():
    from services.tool_execution import _execute

    with patch("resource.setrlimit", side_effect=OSError("synthetic limit unavailable")):
        result = _execute('raise ValueError("must not run")', {}, 64 * 1024 * 1024, 1)
    assert result == {"success": False, "output": "", "error": "OSError: synthetic limit unavailable", "result": ""}
