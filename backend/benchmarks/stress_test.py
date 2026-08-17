#!/usr/bin/env python
"""
MultiPersonal Chat System — 本地模型+LoRA 并发压力测试

测试目标：
  - 本地大模型（Transformers/PEFT + LoRA）在群聊场景下的并发回复生成能力
  - 测量推理延迟 (P50/P90/P99)、吞吐量、GPU 显存
  - 测定系统在稳定运行下能支撑的最大并发消息处理量

部署场景:
  - 数十个群组 × 3000人
  - 日常: 每群 1条/5秒 → 50群 × 0.2msg/s ≈ 10 req/s
  - 高峰: 每群 3条/秒  → 50群 × 3msg/s ≈ 150 req/s

使用:
  $env:PYTHONIOENCODING='utf-8'
  python stress_test.py
"""

import asyncio
import time
import json
import statistics
import sys
import os
import argparse
from dataclasses import dataclass, field
from typing import Optional

import httpx

# ============================================================
# 配置
# ============================================================
BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8000")
HEALTH_URL = f"{BASE_URL}/health"
READY_URL = f"{BASE_URL}/ready"
GENERATE_URL = f"{BASE_URL}/api/generate"
MODEL_STATUS_URL = f"{BASE_URL}/api/model/status"

# QQ 群聊真实消息样本（覆盖短/中/长消息）
GROUP_MESSAGES = [
    "在吗",
    "今天吃什么",
    "哈哈哈哈笑死我了",
    "这个功能怎么用来着",
    "有人一起打游戏吗",
    "晚安",
    "早",
    "最近有什么好看的番推荐吗",
    "我emo了怎么办",
    "大佬们帮忙看看这个问题",
    "谢谢",
    "你说的对但是",
    "啊啊啊好烦",
    "有人吗有人吗有人吗",
    "周末有没有一起出去玩的",
    "分享一个好玩的视频",
    "哈哈哈",
    "不太懂，能讲详细点吗",
    "辛苦了",
    "嗯嗯知道了",
]

MODEL_INFO_CACHE: dict = {}


# ============================================================
# 数据结构
# ============================================================
@dataclass
class TestResult:
    success: bool
    status_code: int
    latency_ms: float
    tokens_generated: int = 0
    error: Optional[str] = None


@dataclass
class TestReport:
    total_requests: int
    concurrent_level: int
    duration_seconds: float
    success_count: int
    failure_count: int
    total_tokens: int
    model_info: dict
    lora_info: Optional[str]
    latencies_ms: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.success_count / self.total_requests * 100 if self.total_requests else 0

    @property
    def throughput_qps(self) -> float:
        return self.total_requests / self.duration_seconds if self.duration_seconds else 0

    @property
    def tokens_per_sec(self) -> float:
        return self.total_tokens / self.duration_seconds if self.duration_seconds else 0

    @property
    def p50_ms(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0

    @property
    def p90_ms(self) -> float:
        if not self.latencies_ms: return 0
        s = sorted(self.latencies_ms)
        return s[min(int(len(s) * 0.9), len(s) - 1)]

    @property
    def p99_ms(self) -> float:
        if not self.latencies_ms: return 0
        s = sorted(self.latencies_ms)
        return s[min(int(len(s) * 0.99), len(s) - 1)]

    @property
    def avg_ms(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0

    @property
    def min_ms(self) -> float:
        return min(self.latencies_ms) if self.latencies_ms else 0

    @property
    def max_ms(self) -> float:
        return max(self.latencies_ms) if self.latencies_ms else 0

    def print_report(self):
        print("\n" + "=" * 70)
        print("  本 地 模 型 + LoRA 压 力 测 试 报 告")
        print("=" * 70)
        print(f"  模型:              {self.model_info.get('provider','?')} | "
              f"{self.model_info.get('model_name','?')}")
        if self.lora_info:
            print(f"  LoRA 适配器:       {self.lora_info}")
        print(f"  并发级别:          {self.concurrent_level} 并发")
        print(f"  总请求数:          {self.total_requests}")
        print(f"  持续时间:          {self.duration_seconds:.1f}s")
        print("-" * 70)
        print(f"  成功:              {self.success_count} ({self.success_rate:.1f}%)")
        print(f"  失败:              {self.failure_count}")
        print(f"  吞吐量:            {self.throughput_qps:.2f} req/s")
        if self.total_tokens:
            print(f"  Token 生成速率:    {self.tokens_per_sec:.1f} tokens/s")
        print("-" * 70)
        print(f"  响应时间 (ms):")
        print(f"    平均:            {self.avg_ms:.0f}")
        print(f"    P50:             {self.p50_ms:.0f}")
        print(f"    P90:             {self.p90_ms:.0f}")
        print(f"    P99:             {self.p99_ms:.0f}")
        print(f"    最小:            {self.min_ms:.0f}")
        print(f"    最大:            {self.max_ms:.0f}")
        print("-" * 70)

        if self.errors:
            error_map = {}
            for e in self.errors:
                short = e[:80].split('\n')[0]
                error_map[short] = error_map.get(short, 0) + 1
            print(f"  错误分布:")
            for err, cnt in sorted(error_map.items(), key=lambda x: -x[1])[:5]:
                print(f"    [{cnt}x] {err}")
        print("=" * 70)
        self._evaluate()

    def _evaluate(self):
        print("\n  评估:")
        issues = []
        if self.success_rate >= 99:
            print("    [PASS] 成功率 >= 99%")
        elif self.success_rate >= 95:
            print("    [WARN] 成功率 95-99%，存在偶发错误")
            issues.append("success_rate")
        else:
            print("    [FAIL] 成功率 < 95%，系统存在问题")
            issues.append("success_rate")

        if self.p50_ms < 2000:
            print("    [PASS] P50 < 2s（可接受的回复延迟）")
        elif self.p50_ms < 5000:
            print("    [WARN] P50 2-5s，延迟偏高")
        else:
            print("    [FAIL] P50 > 5s，用户体验差")
            issues.append("p50")

        if self.p99_ms < 15000:
            print("    [PASS] P99 < 15s")
        else:
            print("    [WARN] P99 > 15s，尾部延迟过高")
            issues.append("p99")

        if self.throughput_qps >= 10:
            print(f"    [INFO] 吞吐量 {self.throughput_qps:.1f} req/s，可支撑 {self.throughput_qps * 5:.0f} 个群组日常负载")
        else:
            print(f"    [INFO] 吞吐量偏低，建议降低并发或优化模型")

        if not issues:
            print("\n    -> 系统在当前并发级别下运行稳定")


# ============================================================
# 模型探测
# ============================================================
async def detect_model() -> tuple[dict, Optional[str]]:
    """探测当前使用的模型和 LoRA"""
    global MODEL_INFO_CACHE
    if MODEL_INFO_CACHE:
        return MODEL_INFO_CACHE, MODEL_INFO_CACHE.get("lora_name")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(MODEL_STATUS_URL)
            if r.status_code == 200:
                data = r.json()
                MODEL_INFO_CACHE = data
                lora = data.get("lora_name") or data.get("active_lora") or data.get("current_lora")
                print(f"[INFO] 检测到模型: provider={data.get('provider')}, "
                      f"model={data.get('model_name','?')}, LoRA={lora or '无'}")
                return data, lora
    except Exception as e:
        print(f"[WARN] 无法获取模型信息: {e}")
    return {}, None


# ============================================================
# 请求发送
# ============================================================
async def send_group_message(client: httpx.AsyncClient, message: str, group_id: int) -> TestResult:
    """模拟群聊中发送消息并获取模型回复"""
    start = time.time()
    try:
        payload = {
            "message": message,
            "sessionType": "group",
            "sessionId": f"stress_test_group_{group_id % 50}",
            "sessionName": f"压力测试群{group_id % 50}",
            "userId": f"test_user_{group_id * 10 + (hash(message) % 100)}",
            "userName": f"测试用户{group_id}",
        }
        resp = await client.post(GENERATE_URL, json=payload,
                                 timeout=httpx.Timeout(60.0, connect=5.0))
        latency = (time.time() - start) * 1000
        if resp.status_code == 200:
            data = resp.json()
            tokens = len(data.get("reply", "")) // 2  # 粗略估算 token 数
            return TestResult(True, 200, latency, tokens)
        elif resp.status_code == 429:
            return TestResult(False, 429, latency, error="Rate limited (429)")
        else:
            body = resp.text[:200]
            return TestResult(False, resp.status_code, latency, error=f"HTTP {resp.status_code}: {body}")
    except httpx.TimeoutException:
        return TestResult(False, 0, (time.time() - start) * 1000, error="Timeout (60s)")
    except Exception as e:
        return TestResult(False, 0, (time.time() - start) * 1000, error=str(e)[:200])


# ============================================================
# 服务可用性检查
# ============================================================
async def check_health() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(HEALTH_URL)
            return r.status_code == 200
    except Exception:
        return False


async def warm_up(client: httpx.AsyncClient):
    """预热模型（第一个请求通常慢，需要加载模型到GPU）"""
    print("[WARM] 预热模型...")
    start = time.time()
    try:
        await send_group_message(client, "你好", 0)
        elapsed = time.time() - start
        print(f"[WARM] 预热完成，耗时 {elapsed:.1f}s")
    except Exception as e:
        print(f"[WARM] 预热请求失败: {e}")


# ============================================================
# 并发测试
# ============================================================
async def run_local_model_test(concurrent: int, total: int, ramp_up: float = 2.0) -> TestReport:
    """本地模型并发测试"""
    model_info, lora_name = await detect_model()
    print(f"\n{'#'*50}")
    print(f"  本地模型并发测试: {concurrent} 并发, {total} 请求")
    print(f"{'#'*50}")

    sem = asyncio.Semaphore(concurrent)
    start_time = time.time()
    all_results = []

    async def worker(idx: int):
        async with sem:
            if ramp_up > 0 and idx < concurrent:
                await asyncio.sleep(idx / concurrent * ramp_up)
            msg = GROUP_MESSAGES[idx % len(GROUP_MESSAGES)]
            async with httpx.AsyncClient() as client:
                return await send_group_message(client, msg, idx)

    tasks = [worker(i) for i in range(total)]
    raw = await asyncio.gather(*tasks, return_exceptions=True)

    for r in raw:
        if isinstance(r, Exception):
            all_results.append(TestResult(False, 0, 0, error=str(r)[:200]))
        else:
            all_results.append(r)

    duration = time.time() - start_time
    successes = [r for r in all_results if r.success]

    return TestReport(
        total_requests=total,
        concurrent_level=concurrent,
        duration_seconds=duration,
        success_count=len(successes),
        failure_count=len(all_results) - len(successes),
        total_tokens=sum(r.tokens_generated for r in successes),
        model_info=model_info,
        lora_info=lora_name,
        latencies_ms=[r.latency_ms for r in successes],
        errors=[r.error for r in all_results if not r.success and r.error],
    )


async def run_gpu_memory_benchmark():
    """GPU 显存基准测试（如果可用）"""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            if len(parts) >= 3:
                print(f"\n  GPU 状态:")
                print(f"    显存:  {parts[0].strip()}/{parts[1].strip()} MB")
                print(f"    利用率: {parts[2].strip()}%")
                if len(parts) >= 4:
                    print(f"    温度:  {parts[3].strip()} C")
    except Exception:
        pass


# ============================================================
# 主流程
# ============================================================
async def main():
    parser = argparse.ArgumentParser(description="本地模型+LoRA 压力测试")
    parser.add_argument("--quick", action="store_true", help="快速模式（仅测试短消息）")
    args = parser.parse_args()

    print("=" * 70)
    print("  MultiPersonal Chat System — 本地模型 + LoRA 压力测试")
    print(f"  目标: {BASE_URL}")
    print("=" * 70)

    if not await check_health():
        print("[ERROR] 服务不可用，请先启动: python main.py")
        return

    # 模型探测
    _, lora = await detect_model()
    print(f"  LoRA: {'已加载' if lora else '未加载（默认模型）'}")

    # GPU 状态
    await run_gpu_memory_benchmark()

    # 预热
    async with httpx.AsyncClient() as client:
        await warm_up(client)

    # 阶梯式压力测试
    if args.quick:
        plan = [(1, 5, "基准"), (5, 10, "低并发"), (10, 15, "中并发")]
    else:
        plan = [
            (1,  5,  "基准（单请求）"),
            (3,  10, "低并发（模拟3群同时发消息）"),
            (5,  15, "中低并发"),
            (10, 25, "中等并发（模拟10群高峰）"),
            (15, 30, "中高并发"),
            (20, 40, "高并发（模拟20群同时发消息）"),
            (30, 50, "高并发+"),
        ]

    for concurrent, total, desc in plan:
        print(f"\n--- {desc}: {concurrent} 并发, {total} 请求 ---")

        # 先测 GPU 状态
        await run_gpu_memory_benchmark()

        report = await run_local_model_test(concurrent, total)
        report.print_report()

        if report.success_rate < 80:
            print(f"\n[STOP] 成功率低于80%，停止增压")
            print(f"  -> 推荐最大稳定并发: {concurrent // 2}")
            break

        await asyncio.sleep(2)

    # 最终 GPU 状态
    print("\n[FINAL] 最终 GPU 状态:")
    await run_gpu_memory_benchmark()

    print("\n" + "=" * 70)
    print("  建议部署配置")
    print("=" * 70)
    print(f"""
  1. 本地模型建议:
     - 使用量化模型（4-bit/8-bit GPTQ/AWQ）降低显存
     - LoRA rank 不宜过大（r=8~16 即可）
     - 启用 Flash Attention 2 加速推理

  2. 并发策略:
     - 本地模型不建议高并发（受GPU显存和算力限制）
     - 建议在模型外层加请求队列，串行化推理请求
     - 或用 vLLM / TGI 等推理框架实现 Continuous Batching

  3. 对于 50 群 × 3000 人的场景:
     - 日常 10 req/s: 单GPU + LoRA 可稳定处理
     - 高峰 150 req/s: 需要推理优化或分布式部署

  4. 监控:
     - 关注 GPU 显存和温度
     - 监控 P99 延迟和错误率
""")


if __name__ == "__main__":
    asyncio.run(main())
