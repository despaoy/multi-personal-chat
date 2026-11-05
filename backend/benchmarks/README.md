# Backend benchmarks

本目录的工具只在显式运行时产生负载，不参与服务启动。正式报告应保留提交、硬件、配置、命令和原始输出；隔离测试结果不能作为真实模型容量结论。

| 入口 | 职责 |
| --- | --- |
| `http_load_benchmark.py` | 统一的认证、连接复用、API／推理负载、分步并发和结果门槛 |
| `vllm_benchmark.py` | 绕过应用的直接 vLLM 请求；只把正常完成且非空的回复计为成功 |
| `realistic_benchmark.py` | 既有混合消息工作负载；独立待审查工具，其历史数字不代表当前性能 |
| `stress_test.py` | 本地模型／LoRA 与 GPU 观测；运行前按独立部署配置核对 |

从仓库的 `backend` 目录运行：

```bash
python -m benchmarks.http_load_benchmark --mode health --url http://127.0.0.1:8000
python -m benchmarks.http_load_benchmark --mode api --url http://127.0.0.1:8000 --api-path /api/stats --concurrency-steps 1,10,50 --json-out api-report.json
python -m benchmarks.http_load_benchmark --mode inference --url http://127.0.0.1:8000 --concurrency-steps 1,2,4 --requests-per-step 12 --json-out inference-report.json
python -m benchmarks.vllm_benchmark
```

受保护的 API 模式需要预先设置一种认证方式：`BENCH_BEARER_TOKEN`、`BENCH_API_KEY` 或 `BENCH_USERNAME`／`BENCH_PASSWORD`。工具不会尝试其他账号或自动注册。LoRA 通过统一入口的 `--lora-name` 指定，默认使用当前配置的基础模型。直接 vLLM 工具通过 `VLLM_URL` 和 `VLLM_BENCHMARK_MODEL_ID` 选择服务；部分、空白、非法响应及传输错误均计入失败，存在失败时进程返回非零。

旧 `auth_benchmark.py`、`inference_benchmark.py`、`concurrency_benchmark.py` 已删除，旧命令需要改为上述统一入口。原有健康检查、受保护 API、推理负载、并发阶梯及429／503统计继续提供；旧队列开关没有实现，旧缓存速度推断没有记录真实命中，不再作为评测能力或容量证据。

真实负载可能消耗模型配额或 GPU 资源，只对隔离测试部署显式运行。统一 HTTP 入口测量应用请求；直接 vLLM 入口测量配置服务的已完成回复，两者均不证明回答质量或生产容量。
