"""Direct vLLM capacity benchmark; application API load uses http_load_benchmark.

Run from backend: python -m benchmarks.vllm_benchmark
VLLM_URL and VLLM_BENCHMARK_MODEL_ID select the target explicitly.
"""

import asyncio
import os
import time

import httpx

from benchmarks.http_load_benchmark import RequestOutcome, StepResult
from inference.openai_protocol import chat_completions_endpoint, completed_chat_content

MESSAGES = [
    "你好",
    "在吗",
    "早上好",
    "晚安",
    "哈哈",
    "原神怎么抽卡概率高一点",
    "雷电将军的技能是什么",
    "钟离的护盾怎么算",
    "帮助",
    "你是谁",
    "你能做什么",
    "请帮我分析一下原神4.0版本枫丹地区的全新角色林尼的强度",
    "1",
    "666",
    "？",
]


async def test_vllm_direct() -> list[StepResult]:
    """Measure completed responses with one client across all concurrency steps."""
    endpoint = chat_completions_endpoint(os.getenv("VLLM_URL", "http://localhost:8001"))
    model_id = os.getenv("VLLM_BENCHMARK_MODEL_ID", "Qwen3-8B-Instruct-AWQ")
    results = []
    async with httpx.AsyncClient(timeout=120.0) as client:

        async def one_request(index: int, semaphore: asyncio.Semaphore) -> RequestOutcome:
            async with semaphore:
                started = time.monotonic()
                status = 0
                error = ""
                try:
                    response = await client.post(
                        endpoint,
                        json={
                            "model": model_id,
                            "messages": [{"role": "user", "content": MESSAGES[index % len(MESSAGES)]}],
                            "max_tokens": 256,
                            "temperature": 0.7,
                        },
                    )
                    status = response.status_code
                    response.raise_for_status()
                    completed_chat_content(response.json())
                except (httpx.HTTPError, ValueError) as exc:
                    error = type(exc).__name__ if isinstance(exc, httpx.HTTPError) else str(exc)
                return RequestOutcome(not error, status, (time.monotonic() - started) * 1000, error)

        for concurrency in (1, 3, 5, 8, 10):
            semaphore = asyncio.Semaphore(concurrency)

            started = time.monotonic()
            outcomes = await asyncio.gather(*(one_request(index, semaphore) for index in range(concurrency * 3)))
            result = StepResult("vllm", concurrency, time.monotonic() - started, list(outcomes))
            results.append(result)
            print(result.summary())
            if result.error_counts:
                print(f"  errors={result.error_counts}")
            if concurrency != 10:
                await asyncio.sleep(2)
    return results


async def main() -> int:
    results = await test_vllm_direct()
    return int(any(result.success_count != result.total_count for result in results))


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
