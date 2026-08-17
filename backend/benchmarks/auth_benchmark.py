"""带认证的推理压力测试"""
import asyncio
import sys
import os
import time
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = os.getenv("BENCH_URL", "http://localhost:8000")


async def get_auth_cookie():
    """登录获取httpOnly Cookie"""
    import httpx
    async with httpx.AsyncClient(timeout=10.0) as client:
        # 先尝试注册bench_tester
        await client.post(f"{BASE_URL}/api/auth/register", json={
            "username": "bench_tester",
            "password": "bench123456"
        })
        # 登录
        r = await client.post(f"{BASE_URL}/api/auth/login", json={
            "username": "bench_tester",
            "password": "bench123456"
        })
        if r.status_code != 200:
            # 尝试已知用户
            r = await client.post(f"{BASE_URL}/api/auth/login", json={
                "username": "lihemu",
                "password": "lihemu"
            })
        if r.status_code != 200:
            r = await client.post(f"{BASE_URL}/api/auth/login", json={
                "username": "testuser999",
                "password": "test123456"
            })
        if r.status_code != 200:
            print(f"登录失败: {r.status_code} {r.text[:200]}")
            return None
        # 提取Set-Cookie
        cookies = r.cookies
        print(f"登录成功, cookies: {dict(cookies)}")
        return cookies


async def benchmark_with_auth(cookies, total=20, concurrency=5):
    """带认证的推理压测"""
    import httpx

    prompts = ["你好", "今天天气怎么样", "讲个笑话", "1+1等于几",
               "你叫什么名字", "帮我写一首诗", "什么是AI", "推荐一本书"]

    results = []
    semaphore = asyncio.Semaphore(concurrency)

    async def one_request():
        import random
        prompt = random.choice(prompts)
        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=60.0, cookies=cookies) as client:
                r = await client.post(f"{BASE_URL}/api/generate", json={
                    "message": prompt,
                    "lora_name": "minamo_lora"
                })
            latency = (time.monotonic() - start) * 1000
            return {"success": 200 <= r.status_code < 300,
                    "status": r.status_code,
                    "latency_ms": latency,
                    "body": r.text[:100]}
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return {"success": False, "status": 0, "latency_ms": latency, "error": str(e)}

    async def limited():
        async with semaphore:
            return await one_request()

    print(f"\n推理压测: total={total}, concurrency={concurrency}")
    start = time.monotonic()
    tasks = [limited() for _ in range(total)]
    results = await asyncio.gather(*tasks)
    duration = time.monotonic() - start

    success = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"  QPS: {total/duration:.2f}")
    print(f"  成功: {len(success)}/{total} ({len(success)/total*100:.1f}%)")
    print(f"  失败: {len(failed)}")

    if success:
        latencies = sorted([r["latency_ms"] for r in success])
        print(f"  平均延迟: {statistics.mean(latencies):.0f}ms")
        print(f"  P50: {latencies[len(latencies)//2]:.0f}ms")
        print(f"  P95: {latencies[int(len(latencies)*0.95)]:.0f}ms")
        print(f"  P99: {latencies[int(len(latencies)*0.99)]:.0f}ms")

    # 状态码分布
    status_dist = {}
    for r in results:
        s = r["status"]
        status_dist[s] = status_dist.get(s, 0) + 1
    print(f"  状态码: {dict(sorted(status_dist.items()))}")

    # 显示前3个失败详情
    for r in failed[:3]:
        print(f"  失败样例: status={r['status']}, {r.get('error', r.get('body', ''))[:80]}")


async def benchmark_api_with_auth(cookies, total=500, concurrency=100):
    """带认证的API压测"""
    import httpx

    endpoints = [
        ("GET", "/api/loras"),
        ("GET", "/api/config"),
        ("GET", "/api/messages"),
    ]

    for method, path in endpoints:
        results = []
        semaphore = asyncio.Semaphore(concurrency)

        async def one_request():
            start = time.monotonic()
            try:
                async with httpx.AsyncClient(timeout=10.0, cookies=cookies) as client:
                    r = await client.get(f"{BASE_URL}{path}")
                latency = (time.monotonic() - start) * 1000
                return {"success": 200 <= r.status_code < 300,
                        "status": r.status_code,
                        "latency_ms": latency}
            except Exception as e:
                latency = (time.monotonic() - start) * 1000
                return {"success": False, "status": 0, "latency_ms": latency, "error": str(e)}

        async def limited():
            async with semaphore:
                return await one_request()

        start = time.monotonic()
        tasks = [limited() for _ in range(total)]
        results = await asyncio.gather(*tasks)
        duration = time.monotonic() - start

        success = [r for r in results if r["success"]]
        latencies = sorted([r["latency_ms"] for r in success]) if success else [0]

        print(f"\n{method} {path}: QPS={total/duration:.1f}, "
              f"成功={len(success)}/{total} ({len(success)/total*100:.1f}%), "
              f"P50={latencies[len(latencies)//2]:.0f}ms, "
              f"P95={latencies[int(len(latencies)*0.95)]:.0f}ms")


async def main():
    print("=" * 60)
    print("  MultiPersonal Chat System 带认证并发压力测试")
    print("=" * 60)

    # 获取认证Cookie
    cookies = await get_auth_cookie()
    if not cookies:
        print("无法获取认证Cookie，退出")
        return

    # 1. API压测 (带认证)
    print("\n📡 测试1: 带认证API吞吐")
    await benchmark_api_with_auth(cookies, total=500, concurrency=100)

    # 2. 推理压测 (低并发)
    print("\n🧠 测试2: 推理吞吐 (并发=3)")
    await benchmark_with_auth(cookies, total=10, concurrency=3)

    # 3. 推理压测 (中并发)
    print("\n🧠 测试3: 推理吞吐 (并发=5)")
    await benchmark_with_auth(cookies, total=15, concurrency=5)

    # 4. 推理压测 (高并发)
    print("\n🧠 测试4: 推理吞吐 (并发=10)")
    await benchmark_with_auth(cookies, total=20, concurrency=10)

    # 5. 缓存测试 - 相同消息
    print("\n💾 测试5: 缓存效果 (相同消息x20)")
    import httpx
    same_results = []
    for i in range(20):
        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=60.0, cookies=cookies) as client:
                r = await client.post(f"{BASE_URL}/api/generate", json={
                    "message": "你好，请问你是谁？",
                    "lora_name": "minamo_lora"
                })
            latency = (time.monotonic() - start) * 1000
            same_results.append({"success": 200 <= r.status_code < 300, "latency_ms": latency})
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            same_results.append({"success": False, "latency_ms": latency})

    success = [r for r in same_results if r["success"]]
    if success:
        latencies = [r["latency_ms"] for r in success]
        print(f"  平均延迟: {statistics.mean(latencies):.0f}ms")
        print(f"  首次: {latencies[0]:.0f}ms, 后续平均: {statistics.mean(latencies[1:]):.0f}ms")
        if len(latencies) > 1 and latencies[0] > 0:
            speedup = latencies[0] / statistics.mean(latencies[1:]) if statistics.mean(latencies[1:]) > 0 else 0
            print(f"  缓存加速比: {speedup:.1f}x")


if __name__ == "__main__":
    asyncio.run(main())
