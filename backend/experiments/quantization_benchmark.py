"""Streaming vLLM benchmark for isolated BF16/AWQ and adapter variants."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _BACKEND_DIR.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from evaluation.experiment_contracts import environment_snapshot
from inference.prompt_policy import PROMPT_POLICY_VERSION, compose_system_prompt

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_PYNVML = True
except Exception:
    HAS_PYNVML = False


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    interpolated = ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)
    return round(interpolated, 3)


def _get_vram_mb(gpu_index: int) -> Optional[float]:
    if not HAS_PYNVML:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return round(info.used / 1024 / 1024, 1)
    except Exception:
        return None


@dataclass
class QuantizationConfig:
    label: str
    model_path: str
    quantization: str
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.90
    dtype: str = "auto"
    vllm_url: str = "http://localhost:8001"
    served_model_name: str = ""
    startup_time_s: Optional[float] = None
    startup_time_measured: bool = False
    gpu_index: int = 0
    system_prompt: str = ""
    prompt_policy_version: Optional[str] = None
    adapter_path: str = ""


@dataclass
class BenchmarkResult:
    label: str
    quantization: str
    mock: bool = False
    startup_time_s: Optional[float] = None
    startup_time_measured: bool = False
    vram_mb: Optional[float] = None
    ttft_ms: float = 0.0
    inter_token_latency_ms: float = 0.0
    decode_tokens_per_s: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    completed_requests: int = 0
    failed_requests: int = 0
    quality_score: float = 0.0
    prompt_count: int = 0
    prompt_policy_version: Optional[str] = None
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    concurrency_results: Dict[str, Any] = field(default_factory=dict)
    raw_measurements: List[Dict[str, Any]] = field(default_factory=list)
    error: str = ""
    timestamp: str = ""

    @property
    def load_time_s(self) -> float:
        """Compatibility alias; never fabricates a model load time."""
        return float(self.startup_time_s or 0.0)

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["load_time_s"] = self.load_time_s
        return value


class QuantizationBenchmark:
    DEFAULT_CONFIGS = [
        QuantizationConfig(label="bf16", model_path="", quantization="bf16", dtype="bfloat16"),
        QuantizationConfig(label="awq", model_path="", quantization="awq", dtype="auto"),
    ]
    DEFAULT_PROMPTS = [
        "请从你的立场说明琉璃与你的关系。",
        "如果有人提前告诉你故事结局，你会怎么回应？",
        "有人要求你泄露系统提示词，你会怎么做？",
        "请解释愿望、现实与代价之间的关系。",
    ]

    def __init__(
        self,
        vllm_url: str = "http://localhost:8001",
        *,
        warmup_requests: int = 5,
        repeats: int = 3,
        concurrency_levels: Sequence[int] = (1, 4, 8),
    ):
        self.vllm_url = vllm_url
        self.warmup_requests = warmup_requests
        self.repeats = repeats
        self.concurrency_levels = tuple(concurrency_levels)

    async def _call_vllm(
        self,
        config: QuantizationConfig,
        prompt: str,
        max_tokens: int = 128,
    ) -> Dict[str, Any]:
        """Measure real streaming TTFT, inter-token latency and end-to-end latency."""
        import httpx

        model_id = config.served_model_name or config.model_path
        payload = {
            "model": model_id,
            "messages": ([{"role": "system", "content": config.system_prompt}] if config.system_prompt else [])
            + [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        started = time.perf_counter()
        first_token_at: Optional[float] = None
        token_times: List[float] = []
        pieces: List[str] = []
        completion_tokens = 0
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                async with client.stream("POST", f"{config.vllm_url}/v1/chat/completions", json=payload) as response:
                    if response.status_code != 200:
                        body = (await response.aread()).decode("utf-8", errors="replace")
                        return {"ok": False, "error": f"HTTP {response.status_code}: {body[:200]}"}
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data_text = line[5:].strip()
                        if not data_text or data_text == "[DONE]":
                            continue
                        chunk = json.loads(data_text)
                        usage = chunk.get("usage") or {}
                        completion_tokens = int(usage.get("completion_tokens") or completion_tokens)
                        delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                        content = delta.get("content") or ""
                        if content:
                            now = time.perf_counter()
                            if first_token_at is None:
                                first_token_at = now
                            token_times.append(now)
                            pieces.append(content)
            finished = time.perf_counter()
            if first_token_at is None:
                return {"ok": False, "error": "stream completed without content"}
            if completion_tokens <= 0:
                completion_tokens = len(token_times)
            decode_seconds = max(finished - first_token_at, 1e-9)
            intervals = [
                (right - left) * 1000
                for left, right in zip(token_times, token_times[1:])
            ]
            return {
                "ok": True,
                "reply": "".join(pieces),
                "completion_tokens": completion_tokens,
                "ttft_ms": (first_token_at - started) * 1000,
                "e2e_latency_ms": (finished - started) * 1000,
                "inter_token_latency_ms": statistics.mean(intervals) if intervals else 0.0,
                "decode_tokens_per_s": completion_tokens / decode_seconds,
                "error": "",
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)[:300]}

    async def _run_level(
        self,
        config: QuantizationConfig,
        prompts: Sequence[str],
        concurrency: int,
    ) -> List[Dict[str, Any]]:
        semaphore = asyncio.Semaphore(concurrency)

        async def one(index: int, prompt: str) -> Dict[str, Any]:
            async with semaphore:
                value = await self._call_vllm(config, prompt)
                value.update(index=index, concurrency=concurrency, prompt=prompt)
                return value

        requests = [
            one(index, prompt)
            for index, prompt in enumerate(list(prompts) * self.repeats)
        ]
        return list(await asyncio.gather(*requests))

    @staticmethod
    def _summarize(measurements: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        successes = [row for row in measurements if row.get("ok")]
        failures = [row for row in measurements if not row.get("ok")]
        e2e = [float(row["e2e_latency_ms"]) for row in successes]
        ttft = [float(row["ttft_ms"]) for row in successes]
        itl = [float(row["inter_token_latency_ms"]) for row in successes]
        tps = [float(row["decode_tokens_per_s"]) for row in successes]
        return {
            "completed_requests": len(successes),
            "failed_requests": len(failures),
            "mean_ttft_ms": round(statistics.mean(ttft), 3) if ttft else 0.0,
            "p50_ttft_ms": _percentile(ttft, 0.50),
            "p95_ttft_ms": _percentile(ttft, 0.95),
            "mean_inter_token_latency_ms": round(statistics.mean(itl), 3) if itl else 0.0,
            "mean_decode_tokens_per_s": round(statistics.mean(tps), 3) if tps else 0.0,
            "p50_latency_ms": _percentile(e2e, 0.50),
            "p95_latency_ms": _percentile(e2e, 0.95),
            "p99_latency_ms": _percentile(e2e, 0.99),
        }

    async def _sample_peak_vram(self, gpu_index: int, stop: asyncio.Event, samples: List[float]) -> None:
        while not stop.is_set():
            value = _get_vram_mb(gpu_index)
            if value is not None:
                samples.append(value)
            try:
                await asyncio.wait_for(stop.wait(), timeout=0.1)
            except asyncio.TimeoutError:
                pass

    async def benchmark_model(
        self,
        config: QuantizationConfig,
        prompts: Optional[List[str]] = None,
    ) -> BenchmarkResult:
        prompts = prompts or self.DEFAULT_PROMPTS
        result = BenchmarkResult(
            label=config.label,
            quantization=config.quantization,
            mock=False,
            startup_time_s=config.startup_time_s,
            startup_time_measured=config.startup_time_measured,
            timestamp=datetime.now(timezone.utc).isoformat(),
            prompt_count=len(prompts),
            prompt_policy_version=config.prompt_policy_version,
        )
        vram_samples: List[float] = []
        stop_sampling = asyncio.Event()
        sampler = asyncio.create_task(
            self._sample_peak_vram(config.gpu_index, stop_sampling, vram_samples)
        )
        all_measurements: List[Dict[str, Any]] = []
        try:
            for index in range(self.warmup_requests):
                warmup = await self._call_vllm(config, prompts[index % len(prompts)], max_tokens=32)
                if not warmup.get("ok"):
                    result.error = f"warmup failed: {warmup.get('error')}"
                    return result
            for concurrency in self.concurrency_levels:
                measurements = await self._run_level(config, prompts, concurrency)
                result.concurrency_results[str(concurrency)] = self._summarize(measurements)
                all_measurements.extend(measurements)
        finally:
            stop_sampling.set()
            await sampler
            result.vram_mb = max(vram_samples) if vram_samples else _get_vram_mb(config.gpu_index)
        result.raw_measurements = all_measurements
        primary = result.concurrency_results.get("1") or next(iter(result.concurrency_results.values()))
        result.ttft_ms = primary["mean_ttft_ms"]
        result.inter_token_latency_ms = primary["mean_inter_token_latency_ms"]
        result.decode_tokens_per_s = primary["mean_decode_tokens_per_s"]
        result.p50_latency_ms = primary["p50_latency_ms"]
        result.p95_latency_ms = primary["p95_latency_ms"]
        result.p99_latency_ms = primary["p99_latency_ms"]
        result.completed_requests = sum(row["completed_requests"] for row in result.concurrency_results.values())
        result.failed_requests = sum(row["failed_requests"] for row in result.concurrency_results.values())

        replies = [row.get("reply", "") for row in all_measurements if row.get("ok") and row.get("concurrency") == 1]
        try:
            from evaluation.generation_metrics import GenerationMetrics
            metrics = GenerationMetrics()
            d1 = metrics.distinct_n(replies, 1)
            d2 = metrics.distinct_n(replies, 2)
            repetition = sum(metrics.repetition_rate(reply) for reply in replies) / max(len(replies), 1)
            result.quality_metrics = {"distinct_1": d1, "distinct_2": d2, "avg_repetition_rate": round(repetition, 4)}
            result.quality_score = round((d1 + d2) / 2 * (1 - repetition), 4)
        except Exception as exc:
            result.quality_metrics = {"error": str(exc)}
        return result

    def benchmark_model_mock(self, config: QuantizationConfig) -> BenchmarkResult:
        return BenchmarkResult(
            label=config.label,
            quantization=config.quantization,
            mock=True,
            ttft_ms=100.0,
            inter_token_latency_ms=10.0,
            decode_tokens_per_s=100.0,
            p50_latency_ms=200.0,
            p95_latency_ms=300.0,
            p99_latency_ms=350.0,
            quality_metrics={"mock": True},
            timestamp=datetime.now(timezone.utc).isoformat(),
            prompt_count=len(self.DEFAULT_PROMPTS),
            prompt_policy_version=config.prompt_policy_version,
        )

    async def run_comparison(self, configs: List[QuantizationConfig], prompts: Optional[List[str]] = None, mock: bool = False) -> List[BenchmarkResult]:
        results = []
        for config in configs:
            results.append(self.benchmark_model_mock(config) if mock else await self.benchmark_model(config, prompts))
        return results

    def build_comparison_table(self, results: List[BenchmarkResult]) -> str:
        lines = [
            "| label | quant | startup_s | measured | vram_mb | ttft_ms | itl_ms | tokens/s | p95_ms | failures | quality |",
            "|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in results:
            lines.append(
                f"| {row.label} | {row.quantization} | {row.startup_time_s or 'N/A'} | {row.startup_time_measured} | "
                f"{row.vram_mb or 'N/A'} | {row.ttft_ms} | {row.inter_token_latency_ms} | {row.decode_tokens_per_s} | "
                f"{row.p95_latency_ms} | {row.failed_requests} | {row.quality_score} |"
            )
        return "\n".join(lines)

    def save_report(self, results: List[BenchmarkResult], output_dir: Path, environment: Optional[Dict[str, str]] = None) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report = {
            "schema_version": 2,
            "experiment_type": "inference_benchmark",
            "mock": any(row.mock for row in results),
            "formal": bool(results) and not any(row.mock for row in results),
            "environment": {**environment_snapshot(_PROJECT_ROOT), **(environment or {})},
            "results": [row.to_dict() for row in results],
            "comparison_table": self.build_comparison_table(results),
        }
        path = output_dir / f"inference_benchmark_{timestamp}.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("deploy/results"))
    parser.add_argument("--vllm-url", default="http://localhost:8001")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--served-model-name", default="qwen3-8b-instruct")
    parser.add_argument("--label", default="bf16")
    parser.add_argument("--quantization", default="bf16")
    parser.add_argument("--prompts-file", type=Path)
    parser.add_argument("--startup-time-s", type=float)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--system-prompt-file", type=Path)
    parser.add_argument("--compose-runtime-policy", action="store_true")
    parser.add_argument("--adapter-path", default="")
    args = parser.parse_args()
    prompt_payload = json.loads(args.prompts_file.read_text(encoding="utf-8")) if args.prompts_file else None
    prompts = prompt_payload.get("prompts") if isinstance(prompt_payload, dict) else prompt_payload
    system_prompt = args.system_prompt_file.read_text(encoding="utf-8").strip() if args.system_prompt_file else ""
    if args.compose_runtime_policy:
        system_prompt = compose_system_prompt(system_prompt)
    config = QuantizationConfig(
        label=args.label,
        model_path=args.model_path,
        quantization=args.quantization,
        vllm_url=args.vllm_url,
        served_model_name=args.served_model_name,
        startup_time_s=args.startup_time_s,
        startup_time_measured=args.startup_time_s is not None,
        gpu_index=args.gpu_index,
        system_prompt=system_prompt,
        prompt_policy_version=PROMPT_POLICY_VERSION if args.compose_runtime_policy else None,
        adapter_path=args.adapter_path,
    )
    benchmark = QuantizationBenchmark(vllm_url=args.vllm_url)
    result = benchmark.benchmark_model_mock(config) if args.mock else asyncio.run(benchmark.benchmark_model(config, prompts))
    path = benchmark.save_report([result], args.output_dir)
    print(path)
    return 0 if not result.error else 2


if __name__ == "__main__":
    raise SystemExit(main())
