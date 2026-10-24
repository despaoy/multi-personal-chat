"""Shared standalone Ollama transport for offline candidate generation.

Task scripts own model parameters; transport owns requests, errors and statistics.
"""

import json
import time
import urllib.error
import urllib.request


class OllamaCandidateClient:

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        temperature: float,
        top_p: float,
        num_predict: int,
        num_ctx: int,
        timeout_s: float,
        progress_prefix: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.num_predict = num_predict
        self.num_ctx = num_ctx
        self.timeout_s = timeout_s
        self.progress_prefix = progress_prefix
        self.stats: dict[str, float | int] = {
            "calls": 0,
            "failures": 0,
            "prompt_tokens": 0,
            "eval_tokens": 0,
            "total_latency_s": 0.0,
            "max_latency_s": 0.0,
        }

    def __call__(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "num_predict": self.num_predict,
                    "num_ctx": self.num_ctx,
                },
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        self.stats["calls"] = int(self.stats["calls"]) + 1
        call_no = int(self.stats["calls"])
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            self.stats["failures"] = int(self.stats["failures"]) + 1
            raise RuntimeError(f"ollama http status {exc.code}") from exc
        except urllib.error.URLError as exc:
            self.stats["failures"] = int(self.stats["failures"]) + 1
            if isinstance(exc.reason, TimeoutError):
                raise TimeoutError("ollama request timeout") from exc
            raise RuntimeError(f"ollama connection error: {type(exc.reason).__name__}") from exc
        except TimeoutError as exc:
            self.stats["failures"] = int(self.stats["failures"]) + 1
            raise TimeoutError("ollama request timeout") from exc
        latency = time.monotonic() - started
        self.stats["total_latency_s"] = float(self.stats["total_latency_s"]) + latency
        self.stats["max_latency_s"] = max(float(self.stats["max_latency_s"]), latency)
        try:
            data = json.loads(body.decode("utf-8"))
            content = data["message"]["content"]
            self.stats["prompt_tokens"] = int(self.stats["prompt_tokens"]) + int(data.get("prompt_eval_count") or 0)
            self.stats["eval_tokens"] = int(self.stats["eval_tokens"]) + int(data.get("eval_count") or 0)
        except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError) as exc:
            self.stats["failures"] = int(self.stats["failures"]) + 1
            raise RuntimeError("ollama response parse error") from exc
        if not isinstance(content, str):
            self.stats["failures"] = int(self.stats["failures"]) + 1
            raise RuntimeError("ollama response content is not a string")
        if self.progress_prefix is not None:
            print(
                f"{self.progress_prefix} call#{call_no} ok {latency:.1f}s eval={int(data.get('eval_count') or 0)}tok",
                flush=True,
            )
        return content
