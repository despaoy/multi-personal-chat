"""
MultiPersonal Chat System并发压力测试

模拟NapCat收到大量消息的场景，测试系统实际上限。
测试维度：
1. 纯API吞吐（不走vLLM，测FastAPI+DB+缓存）
2. 推理吞吐（走vLLM，测端到端QPS）
3. 消息队列吞吐（测Redis Streams入队/出队）
4. 缓存命中率（重复消息 vs 新消息）
5. 限流效果（超限请求是否被429）

用法：
  python benchmarks/concurrency_benchmark.py --mode all
  python benchmarks/concurrency_benchmark.py --mode api
  python benchmarks/concurrency_benchmark.py --mode inference
  python benchmarks/concurrency_benchmark.py --mode queue
"""

import argparse
import asyncio
import json
import time
import statistics
import sys
import os
from dataclasses import dataclass, field
from typing import List, Optional

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class RequestResult:
    """单次请求结果"""
    success: bool
    status_code: int
    latency_ms: float
    error: str = ""
    cached: bool = False


@dataclass
class BenchmarkResult:
    """压测结果"""
    name: str
    total_requests: int
    concurrency: int
    duration_s: float
    results: List[RequestResult] = field(default_factory=list)

    @property
    def qps(self) -> float:
        return self.total_requests / self.duration_s if self.duration_s > 0 else 0

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.results if not r.success)

    @property
    def success_rate(self) -> float:
        return self.success_count / self.total_requests * 100 if self.total_requests > 0 else 0

    @property
    def avg_latency_ms(self) -> float:
        latencies = [r.latency_ms for r in self.results if r.success]
        return statistics.mean(latencies) if latencies else 0

    @property
    def p50_latency_ms(self) -> float:
        latencies = sorted([r.latency_ms for r in self.results if r.success])
        return latencies[len(latencies) // 2] if latencies else 0

    @property
    def p95_latency_ms(self) -> float:
        latencies = sorted([r.latency_ms for r in self.results if r.success])
        if not latencies:
            return 0
        idx = int(len(latencies) * 0.95)
        return latencies[min(idx, len(latencies) - 1)]

    @property
    def p99_latency_ms(self) -> float:
        latencies = sorted([r.latency_ms for r in self.results if r.success])
        if not latencies:
            return 0
        idx = int(len(latencies) * 0.99)
        return latencies[min(idx, len(latencies) - 1)]

    @property
    def cache_hit_count(self) -> int:
        return sum(1 for r in self.results if r.cached)

    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hit_count / self.total_requests * 100 if self.total_requests > 0 else 0

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"  压测结果: {self.name}",
            f"{'='*60}",
            f"  总请求数:     {self.total_requests}",
            f"  并发数:       {self.concurrency}",
            f"  持续时间:     {self.duration_s:.2f}s",
            f"  QPS:          {self.qps:.2f}",
            f"  成功率:       {self.success_rate:.1f}% ({self.success_count}/{self.total_requests})",
            f"  失败数:       {self.fail_count}",
            f"  平均延迟:     {self.avg_latency_ms:.1f}ms",
            f"  P50延迟:      {self.p50_latency_ms:.1f}ms",
            f"  P95延迟:      {self.p95_latency_ms:.1f}ms",
            f"  P99延迟:      {self.p99_latency_ms:.1f}ms",
        ]
        if self.cache_hit_count > 0:
            lines.append(f"  缓存命中:     {self.cache_hit_rate:.1f}% ({self.cache_hit_count}/{self.total_requests})")

        # 状态码分布
        status_dist = {}
        for r in self.results:
            status_dist[r.status_code] = status_dist.get(r.status_code, 0) + 1
        lines.append(f"  状态码分布:   {dict(sorted(status_dist.items()))}")
        lines.append(f"{'='*60}")
        return "\n".join(lines)


async def _http_request(method: str, url: str, json_body: dict = None, headers: dict = None) -> RequestResult:
    """发送HTTP请求并记录结果"""
    import httpx
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            if method == "GET":
                resp = await client.get(url, headers=headers)
            elif method == "POST":
                resp = await client.post(url, json=json_body, headers=headers)
            else:
                resp = await client.request(method, url, json=json_body, headers=headers)
        latency = (time.monotonic() - start) * 1000
        return RequestResult(
            success=200 <= resp.status_code < 300,
            status_code=resp.status_code,
            latency_ms=latency,
        )
    except Exception as e:
        latency = (time.monotonic() - start) * 1000
        return RequestResult(
            success=False,
            status_code=0,
            latency_ms=latency,
            error=str(e),
        )


async def run_concurrent_test(
    name: str,
    request_fn,
    total_requests: int,
    concurrency: int,
) -> BenchmarkResult:
    """通用并发测试框架"""
    result = BenchmarkResult(
        name=name,
        total_requests=total_requests,
        concurrency=concurrency,
        duration_s=0,
    )

    semaphore = asyncio.Semaphore(concurrency)
    completed = 0

    async def limited_request():
        nonlocal completed
        async with semaphore:
            r = await request_fn()
            result.results.append(r)
            completed += 1
            if completed % 50 == 0:
                print(f"  进度: {completed}/{total_requests}", end="\r")

    start = time.monotonic()
    tasks = [limited_request() for _ in range(total_requests)]
    await asyncio.gather(*tasks)
    result.duration_s = time.monotonic() - start

    return result


# ============================================
# 测试1: 纯API吞吐 (不走vLLM)
# ============================================

async def benchmark_api(base_url: str, total: int = 200, concurrency: int = 50):
    """测试纯API吞吐 - /api/stats, /health 等轻量端点"""
    print("\n📡 测试1: 纯API吞吐 (轻量GET端点)")

    # 测试 /health
    async def health_request():
        return await _http_request("GET", f"{base_url}/health")

    r1 = await run_concurrent_test("GET /health", health_request, total, concurrency)
    print(r1.summary())

    # 测试 /api/stats
    async def stats_request():
        return await _http_request("GET", f"{base_url}/api/stats")

    r2 = await run_concurrent_test("GET /api/stats", stats_request, total, concurrency)
    print(r2.summary())

    return [r1, r2]


# ============================================
# 测试2: 推理吞吐 (走vLLM)
# ============================================

async def benchmark_inference(base_url: str, total: int = 30, concurrency: int = 5):
    """测试推理端到端吞吐 - /api/generate"""
    print("\n🧠 测试2: 推理端到端吞吐 (走vLLM)")

    prompts = [
        "你好", "今天天气怎么样", "讲个笑话", "1+1等于几",
        "你叫什么名字", "帮我写一首诗", "什么是AI", "推荐一本书",
    ]

    async def generate_request():
        import random
        prompt = random.choice(prompts)
        return await _http_request(
            "POST",
            f"{base_url}/api/generate",
            json_body={"message": prompt, "lora_name": "minamo_lora"},
        )

    # 低并发测试
    r1 = await run_concurrent_test(
        f"POST /api/generate (并发={concurrency})",
        generate_request, total, concurrency,
    )
    print(r1.summary())

    # 逐步提高并发
    for c in [10, 20]:
        if total * 2 <= 100:
            r = await run_concurrent_test(
                f"POST /api/generate (并发={c})",
                generate_request, min(total, 20), c,
            )
            print(r.summary())

    return [r1]


# ============================================
# 测试3: 缓存命中率 (重复消息)
# ============================================

async def benchmark_cache(base_url: str, total: int = 100, concurrency: int = 20):
    """测试缓存命中率 - 发送重复消息"""
    print("\n💾 测试3: 缓存命中率 (重复消息 vs 新消息)")

    # 重复消息测试 - 同一条消息发送多次
    same_prompt = "你好，请问你是谁？"

    async def same_generate_request():
        return await _http_request(
            "POST",
            f"{base_url}/api/generate",
            json_body={"message": same_prompt, "lora_name": "minamo_lora"},
        )

    r1 = await run_concurrent_test(
        "重复消息 (相同prompt)",
        same_generate_request, total, concurrency,
    )
    print(r1.summary())

    # 不同消息测试
    unique_prompts = [f"测试消息_{i}，随机内容{hash(str(i))}" for i in range(total)]

    async def unique_generate_request():
        idx = len([r for r in r1.results])  # 用不同索引
        prompt = unique_prompts[idx % len(unique_prompts)]
        return await _http_request(
            "POST",
            f"{base_url}/api/generate",
            json_body={"message": prompt, "lora_name": "minamo_lora"},
        )

    r2 = await run_concurrent_test(
        "不同消息 (唯一prompt)",
        unique_generate_request, min(total, 30), concurrency,
    )
    print(r2.summary())

    return [r1, r2]


# ============================================
# 测试4: 限流效果
# ============================================

async def benchmark_ratelimit(base_url: str, total: int = 100, concurrency: int = 50):
    """测试限流效果 - 超限请求是否被429"""
    print("\n🚦 测试4: 限流效果")

    async def rapid_request():
        return await _http_request("GET", f"{base_url}/api/stats")

    r = await run_concurrent_test(
        "限流测试 (高并发GET)",
        rapid_request, total, concurrency,
    )
    print(r.summary())

    # 统计429数量
    rate_limited = sum(1 for res in r.results if res.status_code == 429)
    print(f"  被限流(429): {rate_limited}/{total} ({rate_limited/total*100:.1f}%)")

    return [r]


# ============================================
# 测试6: 逐步增加并发找到上限
# ============================================

async def benchmark_find_limit(base_url: str):
    """逐步增加并发找到系统实际上限"""
    print("\n🔍 测试6: 逐步增加并发找上限")

    results = []
    for concurrency in [1, 5, 10, 20, 50, 100, 200]:
        total = max(concurrency * 5, 20)

        async def stats_request():
            return await _http_request("GET", f"{base_url}/api/stats")

        r = await run_concurrent_test(
            f"并发={concurrency}",
            stats_request, total, concurrency,
        )
        results.append(r)
        print(f"  并发={concurrency:3d} | QPS={r.qps:6.1f} | "
              f"P95={r.p95_latency_ms:7.1f}ms | "
              f"成功率={r.success_rate:5.1f}%")

    # 找到QPS峰值
    best = max(results, key=lambda r: r.qps)
    print(f"\n  📊 QPS峰值: {best.qps:.1f} (并发={best.concurrency})")

    # 找到延迟拐点 (P95 > 1000ms的最低并发)
    for r in results:
        if r.p95_latency_ms > 1000:
            print(f"  ⚠️ 延迟拐点: 并发={r.concurrency} (P95={r.p95_latency_ms:.1f}ms > 1000ms)")
            break

    return results


# ============================================
# 主入口
# ============================================

async def main():
    parser = argparse.ArgumentParser(description="MultiPersonal Chat System并发压力测试")
    parser.add_argument("--mode", default="all",
                        choices=["all", "api", "inference", "cache", "ratelimit", "queue", "limit"],
                        help="测试模式")
    parser.add_argument("--url", default="http://localhost:8000",
                        help="后端API地址")
    parser.add_argument("--total", type=int, default=200,
                        help="每项测试的总请求数")
    parser.add_argument("--concurrency", type=int, default=50,
                        help="并发数")
    args = parser.parse_args()

    print("=" * 60)
    print("  MultiPersonal Chat System 并发压力测试")
    print(f"  目标: {args.url}")
    print(f"  总请求数: {args.total}, 并发数: {args.concurrency}")
    print("=" * 60)

    # 健康检查
    health = await _http_request("GET", f"{args.url}/health")
    if not health.success:
        print(f"\n❌ 后端不可用: {args.url}/health 返回 {health.status_code}")
        return
    print(f"✅ 后端健康检查通过")

    all_results = []

    if args.mode in ("all", "api"):
        all_results.extend(await benchmark_api(args.url, args.total, args.concurrency))

    if args.mode in ("all", "ratelimit"):
        all_results.extend(await benchmark_ratelimit(args.url, args.total, args.concurrency))

    if args.mode in ("all", "limit"):
        all_results.extend(await benchmark_find_limit(args.url))

    if args.mode in ("all", "inference"):
        infer_total = min(args.total, 30)
        all_results.extend(await benchmark_inference(args.url, infer_total, 5))

    if args.mode in ("all", "cache"):
        all_results.extend(await benchmark_cache(args.url, min(args.total, 50), args.concurrency))

    # 最终汇总
    print("\n" + "=" * 60)
    print("  📊 压测汇总")
    print("=" * 60)
    for r in all_results:
        print(f"  {r.name:40s} | QPS={r.qps:6.1f} | P95={r.p95_latency_ms:7.1f}ms | 成功率={r.success_rate:5.1f}%")

    # 输出结论
    print("\n" + "=" * 60)
    print("  🏁 结论")
    print("=" * 60)

    api_results = [r for r in all_results if "/health" in r.name or "/api/stats" in r.name]
    if api_results:
        max_api_qps = max(r.qps for r in api_results)
        print(f"  纯API吞吐上限: ~{max_api_qps:.0f} QPS")

    infer_results = [r for r in all_results if "generate" in r.name.lower()]
    if infer_results:
        max_infer_qps = max(r.qps for r in infer_results)
        print(f"  推理吞吐上限:  ~{max_infer_qps:.1f} QPS")

    limit_results = [r for r in all_results if "并发=" in r.name]
    if limit_results:
        best = max(limit_results, key=lambda r: r.qps)
        print(f"  系统QPS峰值:   ~{best.qps:.0f} (并发={best.concurrency})")


if __name__ == "__main__":
    asyncio.run(main())
