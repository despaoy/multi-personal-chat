"""Short-lived administrative Python tools with explicit execution limits.

The code policy is a convenience restriction, not a replacement for an OS
container. The worker owns no application imports, credentials or live resources.
"""

import ast
import asyncio
import signal
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_SAFE_MODULE_NAMES = {
    "json", "datetime", "math", "re", "collections", "itertools", "statistics",
}
_FORBIDDEN_TOKENS = (
    "import os", "from os", "import subprocess", "from subprocess",
    "import sys", "from sys", "__import__", "builtins",
    "open(", "exec(", "eval(", "compile(", "globals(", "locals(",
    "socket.", "shutil.", "pathlib", "ctypes", "pickle",
)
_MAX_OUTPUT_CHARS = 64_000



class ToolCodeError(ValueError):
    """Source rejected by the existing tool policy."""


class _CappedWriter(io.StringIO):
    def write(self, value):
        if self.tell() + len(value) > _MAX_OUTPUT_CHARS:
            raise ValueError("output exceeds output limit")
        return super().write(value)


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    top_level = name.split(".", 1)[0]
    if level != 0 or top_level not in _SAFE_MODULE_NAMES:
        raise ImportError(f"module '{name}' is not allowed")
    return __import__(name, globals, locals, fromlist, level)


def _safe_builtins():
    return {
        "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
        "range": range, "enumerate": enumerate, "zip": zip, "map": map,
        "filter": filter, "sorted": sorted, "reversed": reversed,
        "round": round, "isinstance": isinstance, "type": type,
        "list": list, "dict": dict, "tuple": tuple, "set": set,
        "str": str, "int": int, "float": float, "bool": bool,
        "True": True, "False": False, "None": None,
        "print": print, "repr": repr, "format": format,
        "any": any, "all": all,
        "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
        "__import__": _safe_import,
    }


def validate_tool_code(code: str) -> None:
    lowered = code.lower()
    for token in _FORBIDDEN_TOKENS:
        if token in lowered:
            raise ToolCodeError(f"code contains forbidden operation: {token}")

    wrapped = "def _validate_claw_code():\n" + "\n".join(
        "    " + line for line in code.splitlines()
    )
    try:
        tree = ast.parse(wrapped, mode="exec")
    except SyntaxError as exc:
        raise ToolCodeError(f"invalid Python syntax: {exc.msg}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = {alias.name.split(".", 1)[0] for alias in node.names}
            if not names.issubset(_SAFE_MODULE_NAMES):
                raise ToolCodeError("code imports a module outside the allowlist")
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module or node.module.split(".", 1)[0] not in _SAFE_MODULE_NAMES:
                raise ToolCodeError("code imports a module outside the allowlist")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ToolCodeError("private and dunder attributes are not allowed")
        elif isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ToolCodeError("dunder names are not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            raise ToolCodeError("global and nonlocal statements are not allowed")


def _execute(code: str, args: dict, memory_limit: int, cpu_limit: int) -> dict:
    stdout = _CappedWriter()
    stderr = _CappedWriter()
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit + 1))
        restricted_globals = {"__builtins__": _safe_builtins(), "args": args, "result": None}
        wrapped_code = (
            "def _claw_main(**kwargs):\n"
            + "\n".join("    " + line for line in code.splitlines())
            + "\n    return None\n"
            "_result = _claw_main(**args)\n"
        )
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(wrapped_code, restricted_globals, restricted_globals)  # noqa: S102
        value = restricted_globals.get("_result")
        result = "" if value is None else str(value)
        if len(result) > _MAX_OUTPUT_CHARS:
            raise ValueError("result exceeds output limit")
        return {"success": True, "output": stdout.getvalue(), "error": stderr.getvalue(), "result": result}
    except BaseException as exc:
        return {"success": False, "output": stdout.getvalue(),
                "error": f"{type(exc).__name__}: {exc}"[:_MAX_OUTPUT_CHARS], "result": ""}


def _worker_environment() -> dict[str, str]:
    environment = {"PATH": os.defpath}
    environment.update({key: os.environ[key] for key in ("SYSTEMROOT", "TZ", "LANG", "LC_ALL") if key in os.environ})
    return environment


async def run_tool_script(path: Path, *, timeout: float = 30.0) -> str:
    """Run a trusted standalone script; own output, deadline and child lifetime.

    Unlike function tools, standalone scripts retain Python script semantics.
    This executor does not provide an operating-system security sandbox.
    """
    path = path.resolve(strict=True)
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-X", "utf8", str(path), cwd=path.parent,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=_worker_environment(), start_new_session=os.name == "posix",
    )

    async def read_output(stream):
        chunks, size = [], 0
        while chunk := await stream.read(4096):
            size += len(chunk)
            if size > _MAX_OUTPUT_CHARS:
                raise ToolCodeError("script output exceeds output limit")
            chunks.append(chunk)
        return b"".join(chunks).decode("utf-8")

    tasks = [asyncio.create_task(read_output(process.stdout)),
             asyncio.create_task(read_output(process.stderr)),
             asyncio.create_task(process.wait())]
    try:
        output, error, returncode = await asyncio.wait_for(asyncio.gather(*tasks), timeout)
    finally:
        if os.name == "posix":
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        elif process.returncode is None:
            process.kill()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await process.communicate()
    if returncode:
        raise subprocess.CalledProcessError(returncode, [sys.executable, str(path)], output, error)
    output, error = output.strip(), error.strip()
    return "\n".join(part for part in (output, f"标准错误输出:\n{error}" if error else "") if part)


def run_tool_code(code: str, args: dict) -> dict:
    validate_tool_code(code)
    timeout = float(os.getenv("CLAW_EXECUTION_TIMEOUT", "3"))
    if not 0.5 <= timeout <= 10.0:
        raise ValueError("CLAW_EXECUTION_TIMEOUT must be between 0.5 and 10 seconds")
    memory_limit = int(os.getenv("CLAW_MEMORY_LIMIT_MB", "4096")) * 1024 * 1024
    if memory_limit <= 0:
        raise ValueError("CLAW_MEMORY_LIMIT_MB must be positive")
    request = json.dumps({"code": code, "args": args, "memory_limit": memory_limit,
                          "cpu_limit": max(1, int(timeout))}, ensure_ascii=False)
    environment = _worker_environment()
    try:
        with tempfile.TemporaryDirectory(prefix="claw-") as directory:
            completed = subprocess.run(
                [sys.executable, "-I", "-S", "-X", "utf8", str(Path(__file__).resolve())],
                input=request, capture_output=True, encoding="utf-8", timeout=timeout,
                env=environment, cwd=directory,
            )
    except subprocess.TimeoutExpired:
        return {"success": False, "output": "", "error": "execution timed out", "result": ""}
    if completed.returncode:
        return {"success": False, "output": "", "result": "",
                "error": f"executor exited without a result (exit code {completed.returncode}): {completed.stderr.strip()}"[:_MAX_OUTPUT_CHARS]}
    return json.loads(completed.stdout)


def _worker_main() -> None:
    request = json.load(sys.stdin)
    response = _execute(request["code"], request["args"], request["memory_limit"], request["cpu_limit"])
    json.dump(response, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    _worker_main()
