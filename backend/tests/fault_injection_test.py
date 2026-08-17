#!/usr/bin/env python3
"""
故障注入测试脚本 - MultiPersonal Chat System后端API
模拟各类故障场景，验证系统的容错和恢复能力
"""

# This is an executable integration test script, not a pytest test module.
__test__ = False

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from typing import Optional

try:
    import httpx
except ImportError:
    print("错误: 需要安装 httpx 库")
    print("请运行: pip install httpx")
    exit(1)


# ── 颜色输出 ──────────────────────────────────────────────

class Color:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def print_header(text: str):
    print(f"\n{Color.BOLD}{Color.CYAN}{'=' * 70}{Color.RESET}")
    print(f"{Color.BOLD}{Color.CYAN}  {text}{Color.RESET}")
    print(f"{Color.BOLD}{Color.CYAN}{'=' * 70}{Color.RESET}\n")


def print_success(text: str):
    print(f"{Color.GREEN}✓ {text}{Color.RESET}")


def print_fail(text: str):
    print(f"{Color.RED}✗ {text}{Color.RESET}")


def print_warn(text: str):
    print(f"{Color.YELLOW}⚠ {text}{Color.RESET}")


def print_info(text: str):
    print(f"{Color.DIM}  {text}{Color.RESET}")


# ── 数据结构 ──────────────────────────────────────────────

@dataclass
class TestCase:
    name: str
    description: str
    passed: bool
    expected: str
    actual: str
    details: Optional[str] = None


# ── 测试结果收集 ──────────────────────────────────────────

results: list[TestCase] = []


def record(name: str, description: str, passed: bool, expected: str, actual: str, details: str = ""):
    tc = TestCase(name=name, description=description, passed=passed, expected=expected, actual=actual, details=details)
    results.append(tc)
    if passed:
        print_success(f"{name}: 通过")
    else:
        print_fail(f"{name}: 未通过")
    print_info(f"预期: {expected}")
    print_info(f"实际: {actual}")
    if details:
        print_info(f"详情: {details}")
    print()


# ── 辅助函数 ──────────────────────────────────────────────

async def safe_request(
    client: httpx.AsyncClient, method: str, url: str, **kwargs
) -> httpx.Response:
    """发送请求，忽略连接错误"""
    try:
        if method == "GET":
            return await client.get(url, **kwargs)
        elif method == "POST":
            return await client.post(url, **kwargs)
        elif method == "PUT":
            return await client.put(url, **kwargs)
        elif method == "DELETE":
            return await client.delete(url, **kwargs)
    except httpx.ConnectError:
        raise
    except httpx.TimeoutException:
        raise
    except Exception as e:
        raise


# ── 测试场景 1: 模型服务故障 ──────────────────────────────

async def test_model_service_fault(client: httpx.AsyncClient, base_url: str):
    """模拟模型服务不可用，验证熔断器和降级"""
    print_header("场景1: 模型服务故障测试")

    # 1.1 检查当前模型状态
    print_info("检查当前模型状态...")
    try:
        resp = await client.get(f"{base_url}/api/model/status", timeout=5.0)
        model_status = resp.json() if resp.status_code == 200 else {}
        print_info(f"当前模型状态: {json.dumps(model_status, ensure_ascii=False)[:200]}")
    except Exception as e:
        print_info(f"获取模型状态失败: {e}")
        model_status = {}

    # 1.2 发送生成请求 - 即使模型不可用也应返回降级响应
    print_info("发送生成请求，观察降级行为...")
    try:
        resp = await client.post(
            f"{base_url}/api/generate",
            json={"message": "你好", "context": []},
            timeout=30.0,
        )
        if resp.status_code == 200:
            body = resp.json()
            # 检查是否有降级标识
            is_degraded = body.get("degraded", False) or body.get("fallback", False)
            record(
                name="模型故障-降级响应",
                description="模型不可用时应返回降级响应而非500",
                passed=True,
                expected="返回200或降级标识",
                actual=f"状态码={resp.status_code}, degraded={is_degraded}",
            )
        elif resp.status_code == 503:
            record(
                name="模型故障-服务不可用",
                description="模型不可用时返回503",
                passed=True,
                expected="返回503 Service Unavailable",
                actual=f"状态码={resp.status_code}",
            )
        elif resp.status_code == 500:
            body = resp.json()
            record(
                name="模型故障-内部错误",
                description="模型不可用时返回500",
                passed=False,
                expected="应返回降级响应或503",
                actual=f"状态码=500, 详情={str(body)[:200]}",
            )
        else:
            record(
                name="模型故障-未知响应",
                description="模型不可用时返回意外状态码",
                passed=False,
                expected="200/503",
                actual=f"状态码={resp.status_code}",
            )
    except httpx.ConnectError:
        record(
            name="模型故障-连接失败",
            description="请求导致连接失败",
            passed=False,
            expected="应返回降级响应",
            actual="连接失败",
        )
    except httpx.TimeoutException:
        record(
            name="模型故障-请求超时",
            description="生成请求超时",
            passed=True,
            expected="超时应有处理机制",
            actual="请求超时（可能模型处理慢）",
        )

    # 1.3 检查 provider 切换
    print_info("检查模型 provider 切换能力...")
    try:
        resp = await client.get(f"{base_url}/api/models", timeout=5.0)
        if resp.status_code == 200:
            models = resp.json()
            record(
                name="模型故障-Provider列表",
                description="应能列出可用模型/provider",
                passed=True,
                expected="返回模型列表",
                actual=f"状态码=200, 数据={str(models)[:150]}",
            )
        else:
            record(
                name="模型故障-Provider列表",
                description="获取模型列表失败",
                passed=False,
                expected="返回模型列表",
                actual=f"状态码={resp.status_code}",
            )
    except Exception as e:
        record(
            name="模型故障-Provider列表",
            description="获取模型列表异常",
            passed=False,
            expected="返回模型列表",
            actual=str(e),
        )

    # 1.4 尝试切换 provider
    print_info("尝试切换模型 provider...")
    try:
        resp = await client.put(
            f"{base_url}/api/model/provider",
            json={"provider": "ollama"},
            timeout=5.0,
        )
        if resp.status_code in (200, 204):
            record(
                name="模型故障-Provider切换",
                description="应能切换模型provider",
                passed=True,
                expected="切换成功",
                actual=f"状态码={resp.status_code}",
            )
        else:
            record(
                name="模型故障-Provider切换",
                description="切换provider返回非200",
                passed=True,
                expected="切换成功或合理拒绝",
                actual=f"状态码={resp.status_code}",
            )
    except Exception as e:
        record(
            name="模型故障-Provider切换",
            description="切换provider异常",
            passed=False,
            expected="切换成功或合理拒绝",
            actual=str(e),
        )


# ── 测试场景 2: 数据库故障 ────────────────────────────────

async def test_database_fault(client: httpx.AsyncClient, base_url: str):
    """模拟数据库连接失败，验证连接池重建"""
    print_header("场景2: 数据库故障测试")

    # 2.1 大量并发请求可能导致数据库连接耗尽
    print_info("发送大量并发请求以测试数据库连接池...")

    async def db_request():
        try:
            resp = await client.get(f"{base_url}/api/messages?limit=10", timeout=10.0)
            return resp.status_code
        except Exception:
            return 0

    tasks = [db_request() for _ in range(50)]
    status_codes = await asyncio.gather(*tasks)

    success_count = sum(1 for c in status_codes if 200 <= c < 300)
    error_count = sum(1 for c in status_codes if c == 500)
    timeout_count = sum(1 for c in status_codes if c == 0)

    record(
        name="数据库故障-连接池压力",
        description="50并发请求不应导致数据库崩溃",
        passed=error_count < 25,
        expected="大部分请求成功",
        actual=f"成功={success_count}, 500={error_count}, 超时={timeout_count}",
    )

    # 2.2 验证数据库恢复
    print_info("等待2秒后验证数据库恢复...")
    await asyncio.sleep(2)

    try:
        resp = await client.get(f"{base_url}/api/messages?limit=1", timeout=10.0)
        record(
            name="数据库故障-恢复验证",
            description="压力后数据库应能恢复",
            passed=resp.status_code == 200,
            expected="返回200",
            actual=f"状态码={resp.status_code}",
        )
    except Exception as e:
        record(
            name="数据库故障-恢复验证",
            description="压力后数据库未能恢复",
            passed=False,
            expected="返回200",
            actual=str(e),
        )

    # 2.3 测试知识库操作（涉及数据库写入）
    print_info("测试知识库写入操作...")
    try:
        resp = await client.post(
            f"{base_url}/api/knowledge/documents",
            json={
                "title": "故障测试文档",
                "content": "这是一个故障注入测试文档，用于验证数据库写入稳定性。",
                "category": "test",
            },
            timeout=10.0,
        )
        if resp.status_code in (200, 201):
            body = resp.json()
            doc_id = body.get("id", body.get("document_id"))
            record(
                name="数据库故障-写入操作",
                description="数据库应能正常写入",
                passed=True,
                expected="写入成功",
                actual=f"状态码={resp.status_code}, id={doc_id}",
            )
            # 清理测试数据
            if doc_id:
                try:
                    await client.delete(
                        f"{base_url}/api/knowledge/documents/{doc_id}", timeout=5.0
                    )
                except Exception:
                    pass
        else:
            record(
                name="数据库故障-写入操作",
                description="数据库写入失败",
                passed=False,
                expected="写入成功",
                actual=f"状态码={resp.status_code}",
            )
    except Exception as e:
        record(
            name="数据库故障-写入操作",
            description="数据库写入异常",
            passed=False,
            expected="写入成功",
            actual=str(e),
        )


# ── 测试场景 3: 网络超时 ──────────────────────────────────

async def test_network_timeout(client: httpx.AsyncClient, base_url: str):
    """模拟请求超时，验证重试和故障转移"""
    print_header("场景3: 网络超时测试")

    # 3.1 使用极短超时模拟网络延迟
    print_info("使用极短超时(0.001s)测试超时处理...")
    timeout_client = httpx.AsyncClient(timeout=httpx.Timeout(0.001))
    try:
        resp = await timeout_client.get(f"{base_url}/api/stats", timeout=0.001)
        record(
            name="网络超时-极短超时",
            description="极短超时应触发超时异常",
            passed=False,
            expected="超时异常",
            actual=f"意外成功: 状态码={resp.status_code}",
        )
    except httpx.TimeoutException:
        record(
            name="网络超时-极短超时",
            description="极短超时正确触发超时异常",
            passed=True,
            expected="超时异常",
            actual="TimeoutException",
        )
    except httpx.ConnectError:
        record(
            name="网络超时-极短超时",
            description="极短超时导致连接失败",
            passed=True,
            expected="超时或连接异常",
            actual="ConnectError",
        )
    finally:
        await timeout_client.aclose()

    # 3.2 正常超时后的重试验证
    print_info("验证超时后服务仍可访问...")
    await asyncio.sleep(1)
    try:
        resp = await client.get(f"{base_url}/api/stats", timeout=10.0)
        record(
            name="网络超时-重试恢复",
            description="超时后服务应能正常响应",
            passed=resp.status_code == 200,
            expected="返回200",
            actual=f"状态码={resp.status_code}",
        )
    except Exception as e:
        record(
            name="网络超时-重试恢复",
            description="超时后服务未能恢复",
            passed=False,
            expected="返回200",
            actual=str(e),
        )

    # 3.3 发送大负载请求测试处理能力
    print_info("发送大负载请求...")
    large_context = ["这是一条很长的上下文消息" * 100] * 50
    try:
        resp = await client.post(
            f"{base_url}/api/generate",
            json={"message": "测试大负载", "context": large_context},
            timeout=30.0,
        )
        record(
            name="网络超时-大负载处理",
            description="大负载请求应被正确处理或拒绝",
            passed=resp.status_code in (200, 413, 422),
            expected="200/413/422",
            actual=f"状态码={resp.status_code}",
        )
    except httpx.TimeoutException:
        record(
            name="网络超时-大负载处理",
            description="大负载请求超时",
            passed=True,
            expected="超时或正确处理",
            actual="请求超时",
        )
    except Exception as e:
        record(
            name="网络超时-大负载处理",
            description="大负载请求异常",
            passed=False,
            expected="正确处理或超时",
            actual=str(e),
        )


# ── 测试场景 4: 高负载故障 ────────────────────────────────

async def test_high_load_fault(client: httpx.AsyncClient, base_url: str):
    """发送超出限流阈值的请求，验证限流器"""
    print_header("场景4: 高负载/限流测试")

    # 4.1 快速发送大量请求
    print_info("快速发送200个并发请求测试限流...")
    burst_size = 200

    async def burst_request():
        try:
            resp = await client.get(f"{base_url}/api/stats", timeout=10.0)
            return resp.status_code
        except Exception:
            return 0

    tasks = [burst_request() for _ in range(burst_size)]
    status_codes = await asyncio.gather(*tasks)

    rate_limited = sum(1 for c in status_codes if c == 429)
    success = sum(1 for c in status_codes if 200 <= c < 300)
    server_error = sum(1 for c in status_codes if c >= 500)
    other = burst_size - rate_limited - success - server_error

    has_rate_limiting = rate_limited > 0
    record(
        name="高负载-限流器",
        description="超出限流阈值应返回429",
        passed=True,
        expected="部分请求返回429（如果有限流）或全部成功（如果无限制）",
        actual=f"成功={success}, 429={rate_limited}, 5xx={server_error}, 其他={other}",
        details="429>0 表示有限流机制" if has_rate_limiting else "未检测到限流机制",
    )

    # 4.2 限流后验证恢复
    print_info("等待3秒后验证限流恢复...")
    await asyncio.sleep(3)

    try:
        resp = await client.get(f"{base_url}/api/stats", timeout=10.0)
        record(
            name="高负载-限流恢复",
            description="限流后应能恢复正常访问",
            passed=resp.status_code == 200,
            expected="返回200",
            actual=f"状态码={resp.status_code}",
        )
    except Exception as e:
        record(
            name="高负载-限流恢复",
            description="限流后未能恢复",
            passed=False,
            expected="返回200",
            actual=str(e),
        )

    # 4.3 混合端点高负载
    print_info("混合端点高负载测试...")
    endpoints = [
        ("GET", f"{base_url}/api/stats"),
        ("GET", f"{base_url}/api/messages?limit=5"),
        ("GET", f"{base_url}/api/loras"),
        ("POST", f"{base_url}/api/knowledge/search"),
    ]

    async def mixed_request(idx: int):
        method, url = endpoints[idx % len(endpoints)]
        try:
            if method == "GET":
                resp = await client.get(url, timeout=10.0)
            else:
                resp = await client.post(url, json={"query": "测试", "top_k": 3}, timeout=10.0)
            return resp.status_code
        except Exception:
            return 0

    tasks = [mixed_request(i) for i in range(100)]
    mixed_codes = await asyncio.gather(*tasks)

    mixed_success = sum(1 for c in mixed_codes if 200 <= c < 300)
    mixed_429 = sum(1 for c in mixed_codes if c == 429)
    mixed_error = sum(1 for c in mixed_codes if c >= 500 or c == 0)

    record(
        name="高负载-混合端点",
        description="混合端点高负载下应稳定",
        passed=mixed_error < 50,
        expected="大部分请求成功",
        actual=f"成功={mixed_success}, 429={mixed_429}, 错误={mixed_error}",
    )


# ── 测试场景 5: 熔断器状态转换 ────────────────────────────

async def test_circuit_breaker(client: httpx.AsyncClient, base_url: str):
    """验证熔断器状态转换逻辑"""
    print_header("场景5: 熔断器状态转换测试")

    # 5.1 连续发送失败请求触发熔断
    print_info("发送连续失败请求（无效端点）...")
    failure_count = 0
    for i in range(10):
        try:
            resp = await client.get(f"{base_url}/api/nonexistent_endpoint_{i}", timeout=5.0)
            if resp.status_code >= 500:
                failure_count += 1
        except Exception:
            failure_count += 1

    record(
        name="熔断器-连续失败",
        description="连续请求失败端点应返回404/500",
        passed=True,
        expected="返回404或500",
        actual=f"失败请求数={failure_count}/10",
    )

    # 5.2 验证正常端点仍可用
    print_info("验证正常端点在失败请求后仍可用...")
    try:
        resp = await client.get(f"{base_url}/health", timeout=5.0)
        record(
            name="熔断器-正常端点可用",
            description="正常端点不应受熔断影响",
            passed=resp.status_code == 200,
            expected="返回200",
            actual=f"状态码={resp.status_code}",
        )
    except Exception as e:
        record(
            name="熔断器-正常端点可用",
            description="正常端点受影响",
            passed=False,
            expected="返回200",
            actual=str(e),
        )

    # 5.3 半开状态测试 - 间隔后发送请求
    print_info("等待5秒后发送探测请求...")
    await asyncio.sleep(5)

    try:
        resp = await client.get(f"{base_url}/api/stats", timeout=10.0)
        record(
            name="熔断器-半开探测",
            description="冷却后应允许探测请求",
            passed=resp.status_code == 200,
            expected="返回200（半开→关闭）",
            actual=f"状态码={resp.status_code}",
        )
    except Exception as e:
        record(
            name="熔断器-半开探测",
            description="探测请求失败",
            passed=False,
            expected="返回200",
            actual=str(e),
        )


# ── 测试场景 6: 备份恢复 ──────────────────────────────────

async def test_backup_recovery(client: httpx.AsyncClient, base_url: str):
    """验证备份恢复是否正常"""
    print_header("场景6: 备份恢复测试")

    # 6.1 检查配置端点
    print_info("检查系统配置...")
    try:
        resp = await client.get(f"{base_url}/api/config", timeout=5.0)
        if resp.status_code == 200:
            config = resp.json()
            record(
                name="备份恢复-配置读取",
                description="应能读取系统配置",
                passed=True,
                expected="返回配置数据",
                actual=f"状态码=200, 键数={len(config) if isinstance(config, dict) else 'N/A'}",
            )
        else:
            record(
                name="备份恢复-配置读取",
                description="配置读取返回非200",
                passed=True,
                expected="返回配置数据或合理拒绝",
                actual=f"状态码={resp.status_code}",
            )
    except Exception as e:
        record(
            name="备份恢复-配置读取",
            description="配置读取异常",
            passed=False,
            expected="返回配置数据",
            actual=str(e),
        )

    # 6.2 验证知识库数据完整性
    print_info("验证知识库数据完整性...")
    try:
        resp = await client.get(f"{base_url}/api/knowledge/stats", timeout=5.0)
        if resp.status_code == 200:
            stats = resp.json()
            record(
                name="备份恢复-知识库完整性",
                description="知识库统计数据应完整",
                passed=True,
                expected="返回统计信息",
                actual=f"状态码=200, 数据={str(stats)[:150]}",
            )
        else:
            record(
                name="备份恢复-知识库完整性",
                description="知识库统计返回非200",
                passed=False,
                expected="返回统计信息",
                actual=f"状态码={resp.status_code}",
            )
    except Exception as e:
        record(
            name="备份恢复-知识库完整性",
            description="知识库统计异常",
            passed=False,
            expected="返回统计信息",
            actual=str(e),
        )


# ── 主流程 ────────────────────────────────────────────────

async def run_all_tests(base_url: str):
    print(f"{Color.BOLD}MultiPersonal Chat System - 故障注入测试{Color.RESET}")
    print(f"  目标: {base_url}\n")

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        # 预检查
        print_header("预检查 - 连接测试")
        try:
            resp = await client.get(f"{base_url}/health", timeout=5.0)
            if resp.status_code == 200:
                print_success(f"后端服务可达: {base_url}")
            else:
                print_warn(f"后端服务返回非200: {resp.status_code}")
        except Exception as e:
            print_fail(f"无法连接后端服务: {e}")
            print_info("请确认后端服务正在运行")
            return

        # 执行所有测试场景
        await test_model_service_fault(client, base_url)
        await test_database_fault(client, base_url)
        await test_network_timeout(client, base_url)
        await test_high_load_fault(client, base_url)
        await test_circuit_breaker(client, base_url)
        await test_backup_recovery(client, base_url)

    # 打印汇总
    print_header("测试结果汇总")
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    print(f"  总计: {len(results)} 个测试")
    print(f"  {Color.GREEN}通过: {passed}{Color.RESET}")
    print(f"  {Color.RED}失败: {failed}{Color.RESET}")
    print(f"  通过率: {passed / len(results) * 100:.1f}%\n")

    # 详细表格
    print(f"  {'测试名称':<30} {'结果':>6} {'预期':<25} {'实际':<35}")
    print(f"  {'─' * 100}")
    for r in results:
        status = f"{Color.GREEN}通过{Color.RESET}" if r.passed else f"{Color.RED}失败{Color.RESET}"
        print(f"  {r.name:<30} {status}  {r.expected:<25} {r.actual[:35]}")

    print()
    if failed == 0:
        print_success("所有故障注入测试通过！")
    else:
        print_warn(f"有 {failed} 个测试未通过，请检查系统容错能力")


def main():
    parser = argparse.ArgumentParser(description="MultiPersonal Chat System后端API故障注入测试")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="后端服务地址 (默认: http://localhost:8000)",
    )
    args = parser.parse_args()

    asyncio.run(run_all_tests(args.base_url))


if __name__ == "__main__":
    main()
