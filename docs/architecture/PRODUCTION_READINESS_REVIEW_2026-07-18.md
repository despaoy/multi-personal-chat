# MultiPersonal Chat System 生产准备审查

> 审查日期：2026-07-18
> 范围：个人研究项目、单机 RTX 3090 部署、多平台 LLM 演示
> 结论：适合受控实验与保研展示；在公网长期运行前仍需完成真实账号、PostgreSQL 和进程监督验收。

## 1. 总体判断

项目已经具备完整的 LLM 应用闭环：AstrBot 多平台入口、FastAPI 核心编排、vLLM/LoRA 推理、RAG、数据库、管理台、评测和实验追踪。对于个人项目，继续引入 Kubernetes、服务网格、CQRS 或复杂微服务会增加维护成本，当前应优先提高可复现性、真实评测质量和部署稳定性。

## 2. 已验证能力

| 类别 | 状态 | 证据 |
| --- | --- | --- |
| 后端回归 | 通过 | Windows 本地 101 passed、1 skipped；实验室服务器 102 passed |
| GPU 运行时 | 通过 | PyTorch 2.8.0+cu128 可识别 RTX 3090 |
| vLLM 依赖 | 通过 | vLLM 0.10.2、Transformers 4.57.6，`pip check` 无冲突 |
| 前后端契约 | 已实现 | 统一 API 客户端、FastAPI Schema、集成测试 |
| 多平台消息模型 | 已实现 | platform/adapter/conversation/sourceMessageId/traceId |
| 幂等与限流 | 已实现 | 消息唯一键、会话/发送者/全局限流 |
| RAG 与 LoRA | 已实现 | 混合检索、适配器管理、训练与评测 |
| 真实 IM 账号 | 待验收 | 需要 AstrBot 平台适配器和账号侧验证 |
| PostgreSQL 生产链路 | 待验收 | SQLite 已验证，PostgreSQL 需迁移与故障演练 |
| 长期进程监督 | 待完善 | 当前主要依赖脚本，建议 systemd --user 或 supervisord |

## 3. 主要风险

| 优先级 | 风险 | 影响 | 建议 |
| --- | --- | --- | --- |
| P0 | 服务器源码存在未提交实验修改 | 拉取或覆盖时可能丢失实验成果 | 同步前提交或归档，禁止直接 reset/clean |
| P0 | 密钥与个人平台账号 | 泄露后可导致账户和数据风险 | 轮换历史暴露密钥，`.env` 权限 600，日志脱敏 |
| P1 | SQLite 多进程写入 | 锁冲突和统计不一致 | SQLite 保持单 worker；正式部署迁移 PostgreSQL |
| P1 | GPU 训练与推理竞争 | OOM、延迟抖动、服务退出 | 单 GPU 单主要任务，启动前检查可用显存 |
| P1 | AstrBot 真实链路未持续验证 | 平台收发能力可能与后端状态不一致 | 分平台做登录、收消息、重复事件和断线重连测试 |
| P1 | 旧 Qwen2.5 与当前 Qwen3 资产混杂 | LoRA 不兼容或报告误读 | 旧资产统一归档，加载前强制校验基座 |
| P2 | Redis 缺失时使用进程内状态 | 多 worker 下限流/队列不共享 | 单进程允许降级，多进程必须部署 Redis |
| P2 | 实验脚本端口不统一 | 调错 vLLM 实例 | 默认 8001，实验 8002，统一由环境变量覆盖 |
| P2 | 文档中的历史数据可能过期 | 展示时产生自相矛盾 | 历史报告加状态标签，当前事实集中在 operations 文档 |

## 4. 调用链检查重点

### 消息生成

```text
AstrBot event
 -> integration authentication/signature
 -> input validation
 -> idempotency
 -> conversation policy/cache
 -> rate limit and priority queue
 -> RAG intent/retrieval
 -> vLLM + optional LoRA
 -> message/model invocation persistence
 -> structured response with traceId
 -> AstrBot send
```

每个阶段必须有超时或明确降级。模型成功而入库失败时，不得自动重复推理；应记录 traceId 并返回已有回复或降级结果。

### 管理台

```text
Browser -> Next.js same-origin proxy -> auth/CSRF -> FastAPI route -> service/database
```

重点验证 Cookie 到 Authorization 转换、FormData 上传、长请求超时和错误响应结构。管理台不得获得 AstrBot token、JWT secret 或平台密码。

## 5. 个人项目合理边界

当前不建议引入：

- Kubernetes、服务网格和跨地域多活。
- CQRS、事件溯源或分库分表。
- 为展示而拆分大量微服务。
- 未经真实负载证据支持的复杂无锁设计。

当前更有展示价值的工作：

1. 完成 Qwen3 LoRA/DoRA/RSLoRA/NEFTune 受控消融。
2. 建立无训练泄漏的 Gold Set 和人工盲评。
3. 对 AWQ/FP16/动态 LoRA 测量 TTFT、吞吐、P95 和显存。
4. 对 vector/BM25/hybrid/reranker 做 RAG 对比与失败分析。
5. 录制 AstrBot 跨平台消息的 traceId 全链路演示。

## 6. 部署门禁

部署前必须满足：

- `git status` 中没有未归档的重要实验文件。
- `python -m pip check` 通过。
- 后端测试、TypeScript 检查和前端构建通过。
- `.env` 不含默认密钥且权限正确。
- 模型、LoRA、数据库和 RAG 索引均在 `runtime/`。
- `/health`、`/ready`、vLLM `/v1/models` 和前端健康检查通过。
- 至少完成一次真实生成、历史入库、LoRA 切换、RAG 检索和 AstrBot 幂等测试。
- 明确备份和回滚路径。

## 7. 结论

当前架构对个人 LLM 研究项目是合理的，主要风险不在“缺少更多企业组件”，而在实验资产治理、真实链路验收、历史与当前配置分离，以及可复现证据。完成上述 P0/P1 项后，可作为实验室单机部署和保研现场演示版本。