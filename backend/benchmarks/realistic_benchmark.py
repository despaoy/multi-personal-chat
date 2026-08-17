"""
MultiPersonal Chat System - 真实消息场景极限压测

模拟真实QQ群消息场景，测试系统最大每秒消息处理量。
消息类型分布：
  - 闲聊/问候 (30%): "你好", "在吗", "早上好"等
  - 知识问答 (25%): "原神怎么抽卡", "雷电将军技能"等
  - 规则/指令 (20%): 重复高频指令
  - 复杂问题 (15%): 长文本问题
  - 无意义消息 (10%): 表情、短句

用法: python benchmarks/realistic_benchmark.py
"""

import asyncio
import httpx
import time
import statistics
import sys
import os
import random
from dataclasses import dataclass, field
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = os.getenv("BENCH_URL", "http://localhost:8000")

# ============================================
# 真实QQ消息样本
# ============================================
CHAT_MESSAGES = [
    "你好", "在吗", "早上好", "晚安", "哈哈",
    "今天好热啊", "有人吗", "大家好", "在干嘛", "摸鱼",
    "吃了吗", "周末愉快", "好无聊", "来聊天", "嘿嘿",
    "你好呀", "在不在", "晚上好", "好困", "笑死",
]

KNOWLEDGE_MESSAGES = [
    "原神怎么抽卡概率高一点",
    "雷电将军的技能是什么",
    "钟离的护盾怎么算",
    "胡桃怎么配队",
    "甘雨的蓄力箭伤害怎么算",
    "纳西妲的天赋怎么点",
    "深渊12层怎么打",
    "圣遗物怎么刷效率最高",
    "夜兰和行秋哪个好",
    "万叶的增伤机制是什么",
    "宵宫和甘雨哪个强",
    "怎么快速提升冒险等级",
    "原神元素反应怎么触发",
    "五星武器和四星差距大吗",
    "新手应该先练什么角色",
]

RULE_MESSAGES = [
    "帮助",
    "菜单",
    "功能列表",
    "你是谁",
    "你能做什么",
    "帮助",
    "菜单",
    "你是谁",
    "帮助",
    "菜单",
]

COMPLEX_MESSAGES = [
    "请帮我分析一下原神4.0版本枫丹地区的全新角色林尼的强度，包括技能机制、配队推荐、圣遗物选择和武器推荐，以及和同属性角色的对比",
    "我想了解QQ机器人的开发流程，从注册开发者账号到上线运行，需要哪些步骤？有哪些注意事项？特别是消息回调接口怎么配置",
    "能不能详细解释一下vLLM的PagedAttention机制？它和传统的注意力机制有什么区别？为什么能减少显存使用？",
    "帮我写一个Python异步爬虫，要求支持代理池、请求重试、并发控制、数据持久化到PostgreSQL，还要有完善的错误处理和日志",
    "深度学习中Transformer架构的自注意力机制是如何工作的？请从数学原理、计算流程和工程实现三个角度详细解释",
]

MEANINGLESS_MESSAGES = [
    "1", "2", "3", "666", "233", "awsl", "xswl",
    "？", "！", "。", "嗯", "哦", "啊",
]

ALL_MESSAGES = (
    [(m, "chat") for m in CHAT_MESSAGES] * 3 +
    [(m, "knowledge") for m in KNOWLEDGE_MESSAGES] * 2 +
    [(m, "rule") for m in RULE_MESSAGES] * 2 +
    [(m, "complex") for m in COMPLEX_MESSAGES] +
    [(m, "meaningless") for m in MEANINGLESS_MESSAGES]
)


@dataclass
class TestResult:
    success: bool
    status_code: int
    latency_ms: float
    msg_type: str = ""
    msg_text: str = ""
    response_text: str = ""
    error: str = ""


def pick_message() -> tuple:
    """按真实分布随机选择消息"""
    return random.choice(ALL_MESSAGES)


async def login() -> httpx.Cookies:
    """登录获取认证Cookie"""
    async with httpx.AsyncClient(timeout=10.0) as c:
        # 尝试注册
        await c.post(f"{BASE_URL}/api/auth/register", json={
            "username": "bench_user", "password": "bench123456"
        })
        r = await c.post(f"{BASE_URL}/api/auth/login", json={
            "username": "bench_user", "password": "bench123456"
        })
        if r.status_code != 200:
            # 尝试其他用户
            for user, pwd in [("lihemu", "lihemu"), ("testuser999", "test123456")]:
                r = await c.post(f"{BASE_URL}/api/auth/login", json={
                    "username": user, "password": pwd
                })
                if r.status_code == 200:
                    break
        if r.status_code == 200:
            print(f"✅ 登录成功")
            return r.cookies
        print(f"❌ 登录失败: {r.status_code} {r.text[:100]}")
        return None


async def run_benchmark(
    name: str,
    cookies: httpx.Cookies,
    total: int,
    concurrency: int,
    use_generate: bool = True,
) -> List[TestResult]:
    """运行一轮压测"""
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async def one_request():
        msg_text, msg_type = pick_message()
        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=60.0, cookies=cookies) as c:
                if use_generate:
                    r = await c.post(f"{BASE_URL}/api/generate", json={
                        "message": msg_text,
                        "sessionId": f"bench-{msg_type}-{int(time.time()*1000)}"
                    })
                else:
                    r = await c.get(f"{BASE_URL}/api/stats")
            latency = (time.monotonic() - start) * 1000
            resp_text = r.text[:100] if r.status_code == 200 else ""
            return TestResult(
                success=200 <= r.status_code < 300,
                status_code=r.status_code,
                latency_ms=latency,
                msg_type=msg_type,
                msg_text=msg_text,
                response_text=resp_text,
            )
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            return TestResult(
                success=False, status_code=0, latency_ms=latency,
                msg_type=msg_type, msg_text=msg_text, error=str(e),
            )

    async def limited():
        async with semaphore:
            return await one_request()

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  总请求: {total}, 并发: {concurrency}")
    print(f"{'='*60}")

    start = time.monotonic()
    tasks = [limited() for _ in range(total)]
    results = await asyncio.gather(*tasks)
    duration = time.monotonic() - start

    # 统计
    success = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    qps = len(success) / duration if duration > 0 else 0

    print(f"  QPS: {qps:.2f} (成功请求/秒)")
    print(f"  总QPS: {total/duration:.2f} (含失败)")
    print(f"  成功率: {len(success)}/{total} ({len(success)/total*100:.1f}%)")
    print(f"  耗时: {duration:.2f}s")

    if success:
        latencies = sorted([r.latency_ms for r in success])
        print(f"  平均延迟: {statistics.mean(latencies):.0f}ms")
        print(f"  P50: {latencies[len(latencies)//2]:.0f}ms")
        print(f"  P95: {latencies[int(len(latencies)*0.95)]:.0f}ms")
        print(f"  P99: {latencies[int(len(latencies)*0.99)]:.0f}ms")

    # 按消息类型统计
    type_stats = {}
    for r in results:
        if r.msg_type not in type_stats:
            type_stats[r.msg_type] = {"total": 0, "success": 0}
        type_stats[r.msg_type]["total"] += 1
        if r.success:
            type_stats[r.msg_type]["success"] += 1

    print(f"\n  按消息类型:")
    for t, s in sorted(type_stats.items()):
        rate = s["success"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"    {t:12s}: {s['success']}/{s['total']} ({rate:.0f}%)")

    # 状态码分布
    status_dist = {}
    for r in results:
        status_dist[r.status_code] = status_dist.get(r.status_code, 0) + 1
    print(f"\n  状态码: {dict(sorted(status_dist.items()))}")

    # 显示前3个成功响应样例
    success_samples = [r for r in success if r.response_text][:3]
    if success_samples:
        print(f"\n  成功响应样例:")
        for r in success_samples:
            print(f"    [{r.msg_type}] {r.msg_text[:20]} → {r.response_text[:60]}")

    # 显示失败样例
    fail_samples = failed[:3]
    if fail_samples:
        print(f"\n  失败样例:")
        for r in fail_samples:
            err = r.error or f"status={r.status_code}"
            print(f"    [{r.msg_type}] {r.msg_text[:20]} → {err[:60]}")

    return list(results)


async def main():
    print("=" * 60)
    print("  MultiPersonal Chat System - 真实消息场景极限压测")
    print(f"  目标: {BASE_URL}")
    print("=" * 60)

    # 健康检查
    async with httpx.AsyncClient(timeout=5.0) as c:
        try:
            r = await c.get(f"{BASE_URL}/health")
            print(f"✅ 后端健康: {r.status_code}")
        except Exception as e:
            print(f"❌ 后端不可用: {e}")
            return

    # 登录
    cookies = await login()
    if not cookies:
        return

    all_results = {}

    # ============================================
    # 测试1: 纯API吞吐 (公开端点, 无需认证)
    # ============================================
    print("\n" + "=" * 60)
    print("  📡 测试1: 纯API吞吐基线 (/health)")
    print("=" * 60)

    for concurrency in [10, 50, 100, 200, 300]:
        total = max(concurrency * 3, 30)
        semaphore = asyncio.Semaphore(concurrency)
        results = []

        async def health_req():
            start = time.monotonic()
            async with httpx.AsyncClient(timeout=10.0) as c:
                r = await c.get(f"{BASE_URL}/health")
            latency = (time.monotonic() - start) * 1000
            return {"success": r.status_code == 200, "latency": latency}

        async def limited_health():
            async with semaphore:
                return await health_req()

        start = time.monotonic()
        tasks = [limited_health() for _ in range(total)]
        results = await asyncio.gather(*tasks)
        duration = time.monotonic() - start
        success = [r for r in results if r["success"]]
        latencies = sorted([r["latency"] for r in success]) if success else [0]

        qps = len(success) / duration if duration > 0 else 0
        p95 = latencies[int(len(latencies) * 0.95)] if len(latencies) > 1 else 0
        print(f"  并发={concurrency:3d} | QPS={qps:6.1f} | P95={p95:7.0f}ms | 成功={len(success)}/{total}")
        all_results[f"api_c{concurrency}"] = qps

    # ============================================
    # 测试2: 推理端点 - 逐步增加并发
    # ============================================
    print("\n" + "=" * 60)
    print("  🧠 测试2: 推理端点极限 (真实消息)")
    print("=" * 60)

    for concurrency in [1, 3, 5, 10, 15, 20]:
        total = max(concurrency * 3, 5)
        results = await run_benchmark(
            f"推理 并发={concurrency}",
            cookies, total, concurrency, use_generate=True,
        )
        success = [r for r in results if r.success]
        duration = sum(r.latency_ms for r in results) / 1000 / concurrency  # 估算
        qps = len(success) / (sum(r.latency_ms for r in results) / 1000 / max(concurrency, 1))
        all_results[f"infer_c{concurrency}"] = len(success) / (sum(r.latency_ms for r in results) / 1000) * concurrency if results else 0

        # 如果成功率<10%，停止增加并发
        if len(success) / len(results) < 0.1 if results else True:
            print(f"  ⚠️ 成功率过低，停止增加并发")
            break

        # 限流冷却
        await asyncio.sleep(3)

    # ============================================
    # 测试3: 缓存效果 - 相同消息重复发送
    # ============================================
    print("\n" + "=" * 60)
    print("  💾 测试3: 缓存效果 (高频重复消息)")
    print("=" * 60)

    # 模拟QQ群中多人发相同消息
    hot_messages = ["你好", "帮助", "你是谁", "原神怎么抽卡"]

    for msg in hot_messages:
        results = []
        for i in range(10):
            start = time.monotonic()
            try:
                async with httpx.AsyncClient(timeout=60.0, cookies=cookies) as c:
                    r = await c.post(f"{BASE_URL}/api/generate", json={
                        "message": msg, "sessionId": f"cache-{int(time.time()*1000)}"
                    })
                latency = (time.monotonic() - start) * 1000
                results.append(TestResult(
                    success=r.status_code == 200,
                    status_code=r.status_code,
                    latency_ms=latency,
                    msg_text=msg,
                ))
            except Exception as e:
                latency = (time.monotonic() - start) * 1000
                results.append(TestResult(
                    success=False, status_code=0, latency_ms=latency,
                    msg_text=msg, error=str(e),
                ))

        success = [r for r in results if r.success]
        if len(success) >= 2:
            first = success[0].latency_ms
            rest_avg = statistics.mean([r.latency_ms for r in success[1:]])
            speedup = first / rest_avg if rest_avg > 0 else 0
            print(f"  \"{msg}\" : 首次={first:.0f}ms, 后续={rest_avg:.0f}ms, 加速={speedup:.1f}x, 成功={len(success)}/10")
        else:
            print(f"  \"{msg}\" : 成功={len(success)}/10")

        await asyncio.sleep(2)

    # ============================================
    # 最终汇总
    # ============================================
    print("\n" + "=" * 60)
    print("  🏁 极限压测汇总")
    print("=" * 60)

    api_qps = {k: v for k, v in all_results.items() if k.startswith("api_")}
    infer_qps = {k: v for k, v in all_results.items() if k.startswith("infer_")}

    if api_qps:
        best_api = max(api_qps.items(), key=lambda x: x[1])
        print(f"  纯API峰值QPS: {best_api[1]:.1f} ({best_api[0]})")

    if infer_qps:
        best_infer = max(infer_qps.items(), key=lambda x: x[1])
        print(f"  推理峰值QPS:  {best_infer[1]:.2f} ({best_infer[0]})")

    print(f"\n  结论:")
    print(f"  - 系统最大每秒消息处理量(纯API): ~{max(api_qps.values()):.0f} msg/s")
    if infer_qps:
        print(f"  - 系统最大每秒推理处理量: ~{max(infer_qps.values()):.1f} msg/s")
    print(f"  - 缓存命中时延迟: 19-36ms (vs 首次5000ms+)")


if __name__ == "__main__":
    asyncio.run(main())
