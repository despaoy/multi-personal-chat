"""
vLLM推理极限压测 - 精确测量真实推理QPS

直接测试vLLM推理能力，不经过熔断器降级。
逐步增加并发，找到vLLM推理的最佳吞吐量。
"""
import asyncio
import httpx
import time
import json
import sys
import os

BASE_URL = os.getenv("BENCH_URL", "http://localhost:8000")

# 真实QQ消息样本
MESSAGES = [
    "你好", "在吗", "早上好", "晚安", "哈哈",
    "原神怎么抽卡概率高一点", "雷电将军的技能是什么", "钟离的护盾怎么算",
    "帮助", "你是谁", "你能做什么",
    "请帮我分析一下原神4.0版本枫丹地区的全新角色林尼的强度",
    "1", "666", "？",
]

DEGRADED_REPLY = "[系统提示] 推理服务暂时不可用，请稍后再试"


async def login():
    """登录获取认证Cookie"""
    async with httpx.AsyncClient(timeout=10.0) as c:
        await c.post(f"{BASE_URL}/api/auth/register", json={
            "username": "bench_v2", "password": "bench123456"
        })
        r = await c.post(f"{BASE_URL}/api/auth/login", json={
            "username": "bench_v2", "password": "bench123456"
        })
        if r.status_code == 200:
            print(f"✅ 登录成功")
            return r.cookies
        print(f"❌ 登录失败: {r.status_code}")
        return None


async def test_vllm_direct():
    """直接测试vLLM推理能力"""
    print("\n" + "=" * 60)
    print("  🔬 测试1: vLLM直接推理 (绕过后端)")
    print("=" * 60)

    vllm_url = os.getenv("VLLM_URL", "http://localhost:8001")
    model_id = os.getenv("VLLM_BENCHMARK_MODEL_ID", "Qwen3-8B-Instruct-AWQ")

    for concurrency in [1, 3, 5, 8, 10]:
        total = concurrency * 3
        semaphore = asyncio.Semaphore(concurrency)
        results = []

        async def one_req():
            msg = MESSAGES[len(results) % len(MESSAGES)]
            start = time.monotonic()
            try:
                async with httpx.AsyncClient(timeout=120.0) as c:
                    r = await c.post(f"{vllm_url}/v1/chat/completions", json={
                        "model": model_id,
                        "messages": [{"role": "user", "content": msg}],
                        "max_tokens": 256,
                        "temperature": 0.7,
                    })
                latency = (time.monotonic() - start) * 1000
                reply = ""
                if r.status_code == 200:
                    data = r.json()
                    reply = data.get("choices", [{}])[0].get("message", {}).get("content", "")[:50]
                return {"ok": r.status_code == 200, "latency": latency, "reply": reply, "status": r.status_code}
            except Exception as e:
                latency = (time.monotonic() - start) * 1000
                return {"ok": False, "latency": latency, "error": str(e)[:50], "status": 0}

        async def limited_req():
            async with semaphore:
                return await one_req()

        start = time.monotonic()
        tasks = [limited_req() for _ in range(total)]
        results = await asyncio.gather(*tasks)
        duration = time.monotonic() - start

        success = [r for r in results if r["ok"]]
        qps = len(success) / duration if duration > 0 else 0
        latencies = sorted([r["latency"] for r in success]) if success else [0]
        p50 = latencies[len(latencies) // 2] if latencies else 0
        p95 = latencies[int(len(latencies) * 0.95)] if len(latencies) > 1 else 0

        print(f"  并发={concurrency:2d} | QPS={qps:5.2f} | P50={p50:7.0f}ms | P95={p95:7.0f}ms | 成功={len(success)}/{total}")

        await asyncio.sleep(2)


async def test_backend_inference(cookies):
    """测试后端推理端点"""
    print("\n" + "=" * 60)
    print("  🧠 测试2: 后端推理端点 (含熔断器)")
    print("=" * 60)

    for concurrency in [1, 2, 3, 5, 8]:
        total = concurrency * 3
        semaphore = asyncio.Semaphore(concurrency)
        results = []

        async def one_req(idx):
            msg = MESSAGES[idx % len(MESSAGES)]
            start = time.monotonic()
            try:
                async with httpx.AsyncClient(timeout=120.0, cookies=cookies) as c:
                    r = await c.post(f"{BASE_URL}/api/generate", json={
                        "message": msg,
                        "sessionId": f"bench-{idx}-{int(time.time())}"
                    })
                latency = (time.monotonic() - start) * 1000
                is_real = False
                reply = ""
                if r.status_code == 200:
                    data = r.json()
                    reply = data.get("reply", "")[:50]
                    is_real = reply != DEGRADED_REPLY
                return {
                    "ok": r.status_code == 200,
                    "is_real": is_real,
                    "latency": latency,
                    "reply": reply,
                    "status": r.status_code,
                }
            except Exception as e:
                latency = (time.monotonic() - start) * 1000
                return {"ok": False, "is_real": False, "latency": latency, "error": str(e)[:50], "status": 0}

        async def limited_req(idx):
            async with semaphore:
                return await one_req(idx)

        start = time.monotonic()
        tasks = [limited_req(i) for i in range(total)]
        results = await asyncio.gather(*tasks)
        duration = time.monotonic() - start

        success = [r for r in results if r["ok"]]
        real_inference = [r for r in results if r.get("is_real")]
        degraded = [r for r in success if not r.get("is_real")]
        qps_total = len(success) / duration if duration > 0 else 0
        qps_real = len(real_inference) / duration if duration > 0 else 0

        if real_inference:
            latencies = sorted([r["latency"] for r in real_inference])
            p50 = latencies[len(latencies) // 2]
            p95 = latencies[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[-1]
        else:
            p50 = p95 = 0

        print(f"  并发={concurrency:2d} | 真实QPS={qps_real:5.2f} | 总QPS={qps_total:5.2f} | P50={p50:7.0f}ms | P95={p95:7.0f}ms | 真实推理={len(real_inference)}/{total} | 降级={len(degraded)}")

        # 如果真实推理成功率<30%，停止
        if len(real_inference) / total < 0.3 and concurrency > 1:
            print(f"  ⚠️ 真实推理比例过低，停止增加并发")
            break

        await asyncio.sleep(5)


async def test_sustained(cookies, concurrency=3, duration_s=30):
    """持续压测：固定并发运行一段时间，测量稳态QPS"""
    print(f"\n" + "=" * 60)
    print(f"  ⏱️ 测试3: 持续压测 (并发={concurrency}, 持续{duration_s}秒)")
    print("=" * 60)

    results = []
    stop_time = time.monotonic() + duration_s
    req_count = 0

    async def worker():
        nonlocal req_count
        while time.monotonic() < stop_time:
            msg = MESSAGES[req_count % len(MESSAGES)]
            req_count += 1
            start = time.monotonic()
            try:
                async with httpx.AsyncClient(timeout=120.0, cookies=cookies) as c:
                    r = await c.post(f"{BASE_URL}/api/generate", json={
                        "message": msg,
                        "sessionId": f"sustained-{req_count}"
                    })
                latency = (time.monotonic() - start) * 1000
                is_real = False
                if r.status_code == 200:
                    data = r.json()
                    is_real = data.get("reply", "") != DEGRADED_REPLY
                results.append({"ok": r.status_code == 200, "is_real": is_real, "latency": latency})
            except Exception as e:
                latency = (time.monotonic() - start) * 1000
                results.append({"ok": False, "is_real": False, "latency": latency})

    workers = [worker() for _ in range(concurrency)]
    start = time.monotonic()
    await asyncio.gather(*workers)
    actual_duration = time.monotonic() - start

    success = [r for r in results if r["ok"]]
    real = [r for r in results if r.get("is_real")]
    degraded = [r for r in success if not r.get("is_real")]

    qps_real = len(real) / actual_duration if actual_duration > 0 else 0
    qps_total = len(success) / actual_duration if actual_duration > 0 else 0

    if real:
        latencies = sorted([r["latency"] for r in real])
        import statistics
        avg = statistics.mean(latencies)
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[-1]
        p99 = latencies[int(len(latencies) * 0.99)] if len(latencies) > 2 else latencies[-1]
    else:
        avg = p50 = p95 = p99 = 0

    print(f"  总请求: {len(results)}")
    print(f"  真实推理: {len(real)} ({len(real)/len(results)*100:.1f}%)")
    print(f"  降级回复: {len(degraded)} ({len(degraded)/len(results)*100:.1f}%)")
    print(f"  真实QPS: {qps_real:.2f}")
    print(f"  总QPS: {qps_total:.2f}")
    if real:
        print(f"  延迟 - 平均:{avg:.0f}ms P50:{p50:.0f}ms P95:{p95:.0f}ms P99:{p99:.0f}ms")


async def main():
    print("=" * 60)
    print("  vLLM推理极限压测 (Qwen3-8B-Instruct)")
    print(f"  后端: {BASE_URL}")
    print("=" * 60)

    # 健康检查
    async with httpx.AsyncClient(timeout=5.0) as c:
        try:
            r = await c.get(f"{BASE_URL}/health")
            print(f"✅ 后端健康: {r.status_code}")
        except Exception as e:
            print(f"❌ 后端不可用: {e}")
            return

    # 测试1: vLLM直接推理
    await test_vllm_direct()

    # 登录
    cookies = await login()
    if not cookies:
        return

    # 测试2: 后端推理端点
    await test_backend_inference(cookies)

    # 测试3: 持续压测
    await test_sustained(cookies, concurrency=3, duration_s=30)

    print("\n" + "=" * 60)
    print("  🏁 压测完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
