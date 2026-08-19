# 研究与学习路线图

> 将 MultiPersonal Chat System 转化为证据驱动的 LLM 系统研究平台，用于研究生保研面试展示。
> 合并自原 LLM_RESEARCH_ENHANCEMENT_ROADMAP、PERSONAL_ACTION_AND_LEARNING_ROADMAP、PROJECT_STATUS_AND_NEXT_STEPS。
> 当前主线：KISAKI-LLM-RESEARCH-V4。旧 E1/E2/E2'/E2'' 已统一归档，不再作为当前结论。

## 1. 项目定位

最强叙事不是"我调用了一个 LLM API"，而是：**"我构建并测量了一个可控、高效、检索接地、多平台的 LLM 系统；我能解释设计权衡并复现结果。"**

用本项目展示四种能力：

1. **模型适配**：LoRA 变体、数据治理、对齐、训练后评估
2. **高效推理**：AWQ、vLLM 调度、KV cache 行为、适配器生命周期、吞吐/延迟权衡
3. **接地生成**：混合 RAG、重排序、引用、置信度、失败分析
4. **系统工程**：可观测性、安全边界、平台网关、可复现部署、消融实验

## 2. 当前已完成状态

### 核心系统

- AstrBot 作为薄多平台网关；FastAPI 负责 LLM 生成、RAG、LoRA、历史、配置、指标
- 多平台消息携带 platform、adapter、conversation、sender、source message ID、trace ID
- 请求限流、队列、幂等、熔断、缓存边界、结构化日志已实现
- 管理页面覆盖：history、LoRA、training、knowledge、experiments、preferences、monitoring、settings、integrations

### LLM 能力

- vLLM 服务 `qwen3-8b-instruct-awq`（已从 Qwen2.5-7B 迁移，2026-07-15）
- LoRA 训练、运行时切换、多 LoRA 路由、混合 RAG、BM25、FAISS、重排序钩子、Gold Set 评估、偏好对管理、实验报告已实现
- 数据生成器记录生成溯源；非 mock SFT 生成在 vLLM 不可用时显式失败
- 生成的偏好对为 `pending` 状态；仅人工 approved 的对才被 DPO/ORPO 训练接受

### 当前研究状态

- R0V4 已完成人物提示词、原作覆盖、构造训练候选、validation 和 Gold 审核；V4.1 的 276 条五轮会话已逐批审核晋升。
- V4 train 当前 948 条；canonical 状态为 `frozen`，Game Train 上下文质量复审和 Gold v3 冻结均已完成，可在正式门禁复验后推进 R1V4。
- 旧 R1 结果只保留为 `legacy_exploratory_non_comparable`。
- R2/R3 实现可用于契约和工具验证，正式结果仍需冻结数据与隔离的真实服务。
- 每次发布前的实际测试计数以 CI 或本次验证报告为准，不在路线图中维护易过期数字。

## 3. 当前限制（不得声称的事项）

- 旧 DPO 报告的 `eval_accuracy=0.65` 是历史占位值，**非真实偏好胜率**
- 旧 E1/E2/E2'/E2'' 指标来自不同数据或评测契约，只能作为问题发现记录，不能声称受控提升
- 当前 AWQ 结果是热服务单模型基准，**不是** FP16/AWQ/NF4/INT8 对比，也**不是**真实 startup-load/streaming-TTFT 测量
- 当前 RAG 集是回归基准，声明泛化前需单独编写 held-out 集
- SQLite + 手动 Redis **不是**生产部署
- 月社妃是唯一正式研究角色；胡桃和 Minamo 只用于外部部署与路由演示，不据此声称跨角色泛化

## 4. 最高价值研究方向

### A. 训练效率与 LoRA 变体

将 `docs/architecture/OPTIMIZATION_STRATEGY.md` 的建议实现为训练配置的可控开关：

- **NEFTune**：仅训练时添加嵌入噪声，暴露 `neftune_noise_alpha`，对比 loss、人类偏好、鲁棒性
- **Sequence packing**：打包短对话样本，报告有效 tokens/s、GPU 利用率、padding 比例
- **DoRA 和 RSLoRA**：暴露互斥开关，在相同 rank、数据、seed、epoch、学习率下对比
- **QA-LoRA**：将 AWQ 训练视为实验而非默认，对比 FP16/BF16 base-model LoRA、QLoRA、QA-LoRA

**交付物**：版本化训练配置 JSON、可复现数据切分与固定随机种子、TensorBoard 曲线、一页消融表（质量/速度/VRAM/可训练参数/adapter 大小）

### B. 数据为中心的微调

- 添加 dataset card：来源、license、语言、规模、persona/domain、已知风险、预期用途
- 去重、长度过滤、畸形对话检测、不安全内容标记、按对话（而非随机行）切分 train/val/test
- LLM 辅助合成数据生成仅配审核队列，存储生成 prompt、源模型、温度、审核决策
- 为 persona 一致性、拒绝行为、RAG-vs-chat 意图路由添加 hard negative 和 counterfactual
- 构建 100-300 prompt 的 Gold 评测集（比大量未测量语料更有价值）

### C. 偏好对齐

SFT 工作后，添加小型偏好学习实验：

- 从生成回复收集成对偏好：有用性、persona 一致性、事实接地、风格、安全
- 在小型策展偏好集上从 DPO 或 ORPO 开始；**不声称 RLHF** 除非真正训练奖励模型并运行策略优化
- 用盲评人类判断和相同 Gold prompt 对比 SFT-only 与 SFT+偏好对齐
- 偏好样本存储为不可变记录，含 annotator rubric 和分歧元数据

### D. 评估框架

| 层 | 指标 | 用途 |
| --- | --- | --- |
| 训练 | eval loss、perplexity、throughput、VRAM | 检测优化与过拟合 |
| 生成 | 长度、Distinct-1/2、重复率 | 测量流畅性与多样性 |
| 角色 | rubric 得分、风格一致性、矛盾率 | 测量角色保真度 |
| 安全 | prompt-injection 成功率、秘密提取拒绝、有害请求拒绝 | 测量边界鲁棒性 |
| RAG | 上下文 precision/recall、回答 faithfulness、引用正确性 | 测量接地生成 |
| 服务 | TTFT、tokens/s、P50/P95/P99、队列拒绝率 | 测量用户体验 |

LLM-as-a-judge 仅用固定 rubric、held-out judge 模型或盲人工抽样，并声明局限。始终保留人工检查子集。

### E. RAG 2.0：接地且可验证的生成

- 添加 chunk 元数据：源文档、章节、版本、导入时间、content hash、KB revision
- 每个 RAG 回答返回引用，UI 显示源标题和证据摘录
- 添加基于检索分数分布、引用覆盖、拒绝阈值的回答置信度；低置信度触发弃答
- 添加 corrective RAG：低置信时重述一次查询、再检索、然后引用回答或弃答（重试上限 1-2 次）
- 构建检索评估数据集（问题、期望文档 ID、gold answer），测量 recall@k、MRR、nDCG、faithfulness
- 对比 vector-only、BM25-only、hybrid、hybrid+reranker，同时报告延迟与准确率

### F. 高效推理与 AWQ 实验

- 构建量化基准：FP16/BF16、AWQ、NF4/QLoRA、8-bit baseline
- 测量模型加载时间、VRAM、TTFT、decode tokens/s、P50/P95 延迟、固定 prompt 质量分
- 对比动态 LoRA 加载与合并 adapter（merge 对固定 persona 更快但失去热切换）
- 用证据解释 vLLM 概念：PagedAttention、continuous batching、prefix caching、KV-cache memory
- 每个实验报告保留 model、tokenizer、vLLM、CUDA、driver、prompt set、command line

**避免声称 AWQ 普遍最优**。结论应为条件性："AWQ 在 24GB VRAM 上达到质量阈值同时支持目标并发。"

### G. 多 LoRA 路由

- 训练或策展数据集明显分离的 domain/persona adapter
- 用轻量路由器选择 base chat、RAG-only、或某个 adapter；记录路由置信度和 fallback 决策
- 对比 hard routing、top-2 candidate routing with clarification、手动选择 baseline
- 添加 adapter 兼容性检查：base-model id、tokenizer revision、target modules、rank、PEFT version
- 演示 adapter 加载失败时的安全 fallback，而非静默使用错误 persona

### H. AstrBot 与 Agentic Workflow

- 保留 AstrBot 作为平台网关，在窄且可审计的工具边界后添加 LLM 智能
- 定义 typed tools：知识搜索、会话偏好查询、批准的信息检索
- 添加 tool-call tracing：tool name、sanitized arguments、result summary、latency、failure type、traceId
- 添加 planner-vs-direct-answer 实验：仅当意图置信度或检索置信度需要时允许工具调用
- MCP 仅用于小型 allow-listed 演示工具集，**绝不**暴露 shell 或无限制文件系统访问

### I. 在线反馈闭环

- 添加 thumbs-up/down + 简短原因分类：incorrect、ungrounded、style mismatch、too slow、unsafe、irrelevant
- 反馈关联 traceId、model/adapter、KB revision、prompt version
- 定期将负反馈抽样到审核队列，用于偏好数据和检索评估更新
- **绝不**在未经同意、隐私过滤、人工审核的情况下对原始生产消息自动重训

## 5. 论文级实验包

一个紧凑但可信的包包含四个实验：

1. **LoRA 消融**：LoRA vs DoRA vs RSLoRA，可选 NEFTune 和 packing
2. **高效推理**：FP16 vs AWQ + 动态 vs 合并 adapter 服务
3. **RAG 消融**：vector vs BM25 vs hybrid vs hybrid+reranker，含引用和弃答评估
4. **对齐**：SFT-only vs 偏好优化模型，在 persona、helpfulness、safety rubric 上

每个实验包含：假设、受控变量、硬件/软件版本、数据集切分、指标、结果表、错误案例、结论。

## 6. 实施计划（10 周）

每个阶段含四部分：**Learn**（应能解释的概念）/ **Do**（需完成的工作）/ **Deliver**（保留的文件/图表/结果）/ **Pass**（前进的客观标准）

### 第 0 周：建立可信基线

**Learn**：训练/验证/held-out 区别；随机种子作为实验控制；mock/smoke/benchmark/研究结论的区别；Git 工作流

**Do**：
1. 阅读 `README.md`、`docs/data/human-scoring-rubric.md` 和本文件
2. 创建实验日志，每次运行记录：日期、Git commit、命令、GPU、模型、数据版本、结果路径
3. 每次实验前检查服务器：`nvidia-smi`、`curl /health`、`curl /ready`、`redis-cli ping`
4. 轮换暴露的密码和 token，QQ/微信集成使用测试账号

**Deliver**：`experiment-log.md` 首条基线条目；数据集版本标识（如 `hutao-sft-v1`）；服务器健康截图

**Pass**：能回答"用了哪些数据、哪个代码 revision、什么硬件、结果文件在哪"

### 第 1 周：数据与评估基础

**Learn**：数据溯源、license、去重、污染、数据泄露；SFT 对话格式与按对话切分；Gold Set 设计；人工评估

**Do**：
1. 以 canonical manifest 和人物数据 README 记录训练语料边界
2. 审核生成 SFT 样本，仅保留事实可接受、persona 一致、不重复的输出
3. 构建 ≥100 prompt 的 held-out Gold Set（30 persona + 20 factual + 20 RAG + 15 safety + 15 multiturn）
4. 确保 Gold Set prompt 不出现在 SFT 训练文件
5. 用 `docs/data/human-scoring-rubric.md` 标注 20 条基线回复，记录原因码

**Deliver**：版本化 Gold Set JSON、完成的 dataset card、`gold-set-v1-review.md`、≥20 人工评分基线

**Pass**：能解释为何 test set 是 held-out，并展示至少一个失败案例

### 第 2 周：偏好数据人工审核

**Learn**：SFT vs 偏好优化；DPO/ORPO 直觉（chosen/rejected 对、参考行为、beta、为何不是 RLHF）；偏好数据失败模式

**Do**：
1. 打开生成的偏好对文件或管理页面
2. 每对判断 `chosen` 是否真正优于 `rejected`（按 rubric）
3. 仅将有效对标记为 `approved`；拒绝畸形、错误、不安全、模糊的对
4. 添加 ≥10 手写 hard negative（persona 崩塌、过度通用、无支撑 RAG、不安全指令跟随、过度拒绝）
5. pilot 保留 ≥20 approved 对，结论前目标 ≥50

**Deliver**：approved 偏好 JSONL、审核日志（approved/rejected/uncertain 计数）、bias note

**Pass**：无自动生成的对未经人工审核进入 DPO/ORPO

### 第 3-4 周：LoRA、DoRA、RSLoRA 实验

**Learn**：低秩适配、rank、alpha、target modules、dropout、adapter 大小；DoRA/RSLoRA 设计目标；过拟合信号；TensorBoard 曲线

**Do**：
1. 训练 LoRA baseline（固定数据、seed、rank、lr、epochs、max seq len）
2. 训练 DoRA（仅 `use_dora=true` 改变）
3. 训练 RSLoRA（仅 `use_rslora=true` 改变）
4. GPU 显存紧张时训练期间停止 vLLM，推理评估时重启
5. 在相同 Gold Set 评估每个 adapter：验证 loss/perplexity、adapter 大小、可训练参数、生成多样性、20 盲评对比
6. **不要使用旧 ablation 报告**，迁移到 Qwen3-8B 后重跑

**Deliver**：每个 run 一个配置 JSON、TensorBoard 日志和截图、`lora-ablation-v1.md`（含表和 3 个代表性错误案例）

**Pass**：能辩护一个结论，如"DoRA 在此数据集上未足够提升 held-out persona 得分以证明额外训练时间合理"

### 第 5 周：RAG 评估与接地

**Learn**：dense retrieval、BM25、hybrid、reranking、chunking、Recall@k、MRR、nDCG；检索正确性 vs 回答 faithfulness；引用覆盖与弃答

**Do**：
1. 创建与 seed-document 构建过程分离的 held-out 检索评估集
2. 导入带源元数据的版本化知识库
3. 运行 vector-only、BM25-only、hybrid、hybrid+reranker
4. 记录 Recall@5、MRR、nDCG、平均延迟、失败查询
5. UI 添加回答引用，手动检查 20 条引用回答
6. 定义低置信度阈值触发弃答

**Deliver**：`retrieval-eval-v1.json`、`rag-ablation-v1.md`（含指标表和失败分析）、引用和低置信度弃答截图

**Pass**：能区分"文档被检索到"和"最终回答被文档支撑"

### 第 6 周：vLLM 与 AWQ 服务研究

**Learn**：AWQ、FP16/BF16、NF4、量化误差、memory-quality 权衡；vLLM PagedAttention、continuous batching、KV cache、TTFT

**Do**：
1. 每个模型变体运行一个**隔离的 vLLM 进程**，绝不将热 AWQ 进程标记为 FP16 或 NF4
2. 每个变体使用相同 prompt、context limit、采样参数、并发
3. 测量 VRAM、启动时间、streaming TTFT、端到端 P50/P95、tokens/s、失败率
4. 对比动态 LoRA 加载与合并 adapter 服务（固定 persona）
5. 每个报告记录 model path、driver、CUDA、PyTorch、vLLM、command line

**Deliver**：`quantization-benchmark-v1.md`、CSV/JSON 原始测量、一图（VRAM vs P95 延迟 vs 质量分）

**Pass**：能做条件性陈述，如"AWQ 在 24GB GPU 上达到质量阈值同时为目标并发留足 VRAM"

### 第 7 周：DPO 或 ORPO Pilot

**Learn**：DPO 目标、参考模型行为、beta、为何偏好准确率需 held-out log-probability 评分；QLoRA、NF4、gradient checkpointing；为何小偏好集证明 pipeline 但不证明广泛对齐

**Do**：
1. 仅在 approved 对上训练
2. 在 3090 上从短 QLoRA DPO pilot 开始
3. 保留 SFT baseline adapter 的不可变副本
4. 在相同 held-out Gold Set 上评估 baseline 和 DPO adapter
5. **不报告历史 `eval_accuracy=0.65`**（占位值）
6. 用显式 chosen/rejected log probability 或盲评人类对比评分 held-out 偏好集

**Deliver**：DPO config、adapter、训练日志、内存记录；小型 baseline-vs-DPO 表；limitations 段落（数据规模和标注 bias）

**Pass**：能准确说"这是小型 DPO pilot，不是 RLHF，结论受审核偏好集规模限制"

### 第 8 周：AstrBot 与生产演示

**Learn**：webhook/gateway 边界、幂等、重试、限流、trace ID、失败隔离；认证、token 轮换、replay 保护、平台隐私约束

**Do**：
1. 配置 QQ/NapCat 连接 AstrBot
2. 安装配置 `multipersonal_gateway`（backend URL + shared token）
3. 测试私聊、群 @、prefix trigger、重复 message ID、缺失 token、超时
4. 个人微信用测试账号，视为可选（稳定性和合规风险）
5. 捕获相同 trace ID 贯穿 AstrBot 日志、FastAPI 日志、消息历史
6. demo 路径稳定后从 SQLite/手动 Redis 迁移到 PostgreSQL/持久 Redis/supervised services

**Deliver**：一个跨平台 trace 截图；一个失败处理截图或日志；部署图和服务健康截图

**Pass**：老师能发一条真实消息，你能解释它如何到达 AstrBot、FastAPI、RAG/LoRA/vLLM、数据库、并返回平台

### 第 9-10 周：演示与报告

**Do**：
1. 构建 dashboard 整合所有实验结果
2. 撰写完整实验报告
3. 准备 5 分钟 demo + 8 分钟技术讲解
4. 整理最终作品集

## 7. 立即行动

### A. 安全与服务器控制

1. 轮换每个出现在聊天、日志、截图中的密码、SSH 凭证、token、平台密钥
2. 使用 SSH key，限制防火墙仅暴露 SSH 和 Nginx 80/443
3. vLLM、Redis、PostgreSQL、FastAPI 保持 loopback 或内部 Docker 网络
4. `.env` 设为 mode `600`，**绝不**提交

### B. 生产基础

1. 用 `deploy/docker-compose.yml` 运行 PostgreSQL、持久 Redis、FastAPI、frontend、vLLM、Nginx 作为 supervised services
2. 设置 `ENVIRONMENT=production`、`DATABASE_URL`、`REDIS_URL`、`JWT_SECRET`、`ASTRBOT_INTEGRATION_TOKEN`、`ALLOWED_ORIGINS`
3. 通过 backup-and-verify 流程迁移 SQLite 数据，**不**覆盖原数据库
4. 添加定时 PostgreSQL 备份并测试恢复

### C. 真实平台验收

1. 配置 QQ/NapCat 连接 AstrBot（不直连旧 NoneBot 路径）
2. 在 AstrBot 中配置 `multipersonal_gateway`（与 FastAPI 相同 integration token）
3. 测试 QQ 私聊、群 @/prefix trigger、重复 message ID、超大文本、缺失 token、backend 超时
4. 个人微信用测试账号，验证私聊和群行为后才启用真实联系人
5. 每个平台验收测试保存一个 trace ID 及对应 AstrBot、FastAPI、数据库记录

## 8. 部署验收清单

| 领域 | 必需证据 |
| --- | --- |
| 认证 | login、logout、protected APIs、CSRF/same-origin 行为 |
| 模型 | 真实回复、历史记录、trace ID |
| LoRA | 加载、切换、失败 fallback、base-model 恢复 |
| RAG | 文档导入、检索、引用显示、低置信弃答 |
| AstrBot | 真实 QQ/微信私聊、群触发、去重、优雅失败 |
| 可靠性 | 队列过载不返回失控 500 |
| 可观测性 | trace ID 串联 gateway、backend、存储记录 |
| 恢复 | PostgreSQL restore 和服务重启已测试 |

## 9. 学习资源

优先官方论文和文档，读够能解释 idea，再用本项目验证。

| 主题 | 阅读理解 | 项目应用 |
| --- | --- | --- |
| LoRA | LoRA 论文；PEFT 文档 | baseline、target modules、rank/alpha 研究 |
| DoRA / RSLoRA | DoRA 和 RSLoRA 论文 | 受控消融 |
| QLoRA | QLoRA 论文；bitsandbytes 文档 | DPO 内存高效 pilot |
| DPO / ORPO | DPO 论文；TRL 文档 | 审核偏好对和 pilot |
| RAG | DPR、BM25、hybrid retrieval、RAG evaluation | vector/BM25/hybrid/reranker 基准 |
| vLLM | PagedAttention 论文；vLLM 文档 | 服务和量化研究 |
| AWQ | AWQ 论文 | memory-latency-quality 对比 |
| 评估 | Distinct-N、MRR、nDCG、盲评 | Gold Set 和评分 rubric |
| 系统 | 队列、幂等、熔断、可观测性 | AstrBot/FastAPI 生产路径 |

## 10. 最终作品集

联系老师前准备：

1. 一张架构图
2. 一张部署图和服务健康截图
3. 一张 LoRA/DoRA/RSLoRA 对比表
4. 一张 RAG 对比表（含引用和失败案例）
5. 一张 AWQ 服务基准图
6. 一张小型 DPO pilot 对比（诚实陈述局限）
7. 一张 AstrBot 跨平台 trace ID 演示
8. 5 分钟 demo + 8 分钟技术讲解

## 11. 现场演示

1. 同一领域问题 RAG 开/关，展示证据引用和置信度
2. 切换 adapter，展示运行时加载 trace、模型元数据、persona 评估结果
3. 展示 AWQ 基准 dashboard：memory、TTFT、throughput 相对 FP16
4. 发送重复和并发平台消息，展示幂等、队列优先级、trace 关联
5. 展示不安全或 prompt-injection 请求被拦截，含 sanitized 结构化审计日志

## 12. 口述叙事

> "我从可部署的多平台助手开始，然后把它当作 LLM 系统研究平台：我控制了数据和训练变体，测量了 LoRA 和 AWQ 的质量-效率权衡，让 RAG 证据接地且可测量，并用有界并发、可追踪性、安全性构建服务层。每个结论都有可复现实验支撑，而非功能声明。"

核心叙事：**"我构建了 LLM 系统，测量了其适配、检索、推理、安全的权衡，并能复现成功与失败。"**

## 13. 研究诚信 Guardrails

- **不**实现每个流行技术。四个有测量的实验胜过十个未检查的功能
- **不**在未经明确同意和隐私审核的情况下对个人平台消息训练
- **不**用不同数据、种子、prompt 模板、硬件对比方法并称之为消融
- **不**用单一 LLM judge 作为质量的唯一证据
- 保持生产可靠性功能与实验代码路径通过显式 flag 和版本化配置分离
- **不**将 mock 输出当作真实实验结果
- **不**声称 `eval_accuracy=0.65`（历史占位值）
- 生成的偏好对以 `pending` 开始，需人工审核才能用于 DPO/ORPO
- 真实量化对比需每个变体隔离 vLLM 进程
- 每次实验同时保留：数据版本、种子、模型版本、命令、硬件、报告
- 至少 50 条 approved 偏好对才可进行 DPO pilot
- DPO/ORPO **不是** RLHF（基于参考模型的对比学习，非强化学习）

## 14. 安全清理规则

**可安全删除**（验证后）：`__pycache__`、临时测试目录、mock 报告、失败尝试日志、有最终 adapter 时的训练 checkpoint

**必须保留**：final adapter、TensorBoard 日志、approved 数据集、数据库备份、知识库、模型文件、真实实验报告、部署配置

## 15. 优化指南映射

| 提供的建议 | 本路线图最佳用途 |
| --- | --- |
| NEFTune + packing | 训练效率消融 |
| DoRA 和 RSLoRA | 参数高效适配研究 |
| TensorBoard | 可复现训练遥测 |
| Post-training evaluator | 分层评估套件 |
| Quantization comparison | AWQ 推理证据 |
| QA-LoRA | 可选高级量化训练实验 |
| Adapter merge benchmark | 运行时切换 vs 固定 persona 服务权衡 |

**起步顺序**：TensorBoard、data card、Gold 评估集、DoRA/RSLoRA 开关、AWQ 基准。这些以最少架构风险创造最清晰证据。
