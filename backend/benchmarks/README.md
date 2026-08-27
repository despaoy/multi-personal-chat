# Backend benchmarks

本目录是显式运行的性能与容量测试，不参与服务启动，也不作为单元测试自动执行。正式报告必须记录提交、硬件、配置、命令和原始输出。

| 脚本 | 边界 |
| --- | --- |
| `http_load_benchmark.py` | 经过认证的生产 API HTTP 负载 |
| `auth_benchmark.py` | 认证路径与受保护推理压力 |
| `concurrency_benchmark.py` | 并发消息与系统回压 |
| `realistic_benchmark.py` | 群聊消息分布模拟 |
| `inference_benchmark.py` | 经应用入口的推理性能 |
| `vllm_benchmark.py` | 直接测量 vLLM 服务能力 |
| `stress_test.py` | 本地模型与 LoRA 并发压力 |

基准脚本可能消耗 GPU、模型配额或本地服务资源，不能作为发布检查的默认步骤。mock 结果不得写成真实性能结论。
