# Backend benchmarks

工具只在显式运行时产生负载。保留提交、硬件、配置、命令和原始报告；隔离测试不能证明真实模型质量或生产容量。

| 入口 | 职责 |
| --- | --- |
| `http_load_benchmark.py` | 统一认证、连接复用、预热、API／推理工作负载、分步并发和结果门槛 |
| `vllm_benchmark.py` | 直接请求配置的vLLM服务，只统计完整非空回复 |
| `workloads/*.json` | 纯请求数据，经正式MessageRequest契约读取，无独立认证或执行逻辑 |

从`backend`目录运行：

```bash
python -m benchmarks.http_load_benchmark --mode health --url http://127.0.0.1:8000
python -m benchmarks.http_load_benchmark --mode api --url http://127.0.0.1:8000 --api-path /api/stats --concurrency-steps 1,10,50 --json-out api-report.json
python -m benchmarks.http_load_benchmark --mode inference --url http://127.0.0.1:8000 --workload benchmarks/workloads/mixed_messages.json --concurrency-steps 1,2,4 --requests-per-step 12 --json-out mixed-report.json
python -m benchmarks.http_load_benchmark --mode inference --url http://127.0.0.1:8000 --workload benchmarks/workloads/group_messages.json --concurrency-steps 1,3,5 --requests-per-step 50 --json-out group-report.json
python -m benchmarks.vllm_benchmark
```

受保护模式预先配置一种认证：`BENCH_BEARER_TOKEN`、`BENCH_API_KEY`或`BENCH_USERNAME`／`BENCH_PASSWORD`。不会注册账号或尝试其他账号。直接vLLM入口使用`VLLM_URL`和`VLLM_BENCHMARK_MODEL_ID`。

推理可用`--message`发送单条原文，或用`--workload`读取JSON请求数组，两者互斥。文件只读取和校验一次，各并发阶梯按数组顺序循环；非法、不可读或空文件在联网前报错。请求中的message、history、人物、用户、群会话及LoRA字段原样传递；工具分配每次唯一traceId，未指定的sessionId使用独立会话。工作负载中的LoRA只在各请求的loraName设置；不与全局`--lora-name`混用。单消息模式仍可用`--lora-name`。

混合文件保留旧工具58种不同消息；群文件保留20种群消息，并明确50个群会话和用户。它们是公开的合成负载，部分问题有年代或信息限制，不是回答质量金标，也不声称符合真实QQ消息比例。群会话重复是有意的，不能据其吞吐量推算真实群组容量。

推理沿用严格GenerateResponse契约，只有modelInvoked=true的完整非空响应计入成功。缓存及非模型回复属于未完成新推理，预热遇到此类结果会终止；这是测量边界，不表示正常应用缓存有错。需要新推理时，在隔离部署显式准备缓存配置和完整工作负载，工具不改消息、不改服务配置。报告model_counts只记录成功响应实际声明的模型，不猜模型名称、适配器加载状态或token数。请求失败和阈值不达标返回非零。

在实际GPU主机的另一终端直接观测驱动：

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv --loop=1
```

驱动工具故障按其退出状态排查，不写成零指标或跳过后声称成功。该命令观测本机所有GPU；使用DeepSeek云接口时，本机GPU数据不反映云模型容量。显式选择LoRA前应确认隔离服务使用支持适配器的本地提供商，工具不会切换提供商。

旧auth／inference／concurrency、realistic_benchmark及stress_test入口已删除，改用上述统一命令。旧自动注册、多账号尝试、每请求连接、随机比例／分类统计、猜测模型字段、字符估算token、缓存加速常数和群组容量建议不再保留。原健康/API/推理负载、并发阶梯、429／503分类、混合消息、群会话和显式LoRA继续提供，GPU观测使用原生驱动工具；未实现的队列开关不属于保留能力。

真实负载可能消耗模型配额或GPU资源，只对隔离测试部署显式运行。
