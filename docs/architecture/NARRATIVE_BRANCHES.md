# 正史与反事实分支隔离

实现日期：2026-09-18。实验性 Web MVP，默认关闭。自动测试验证状态隔离与事务行为，不代表真实模型已能可靠理解反事实或抵御语义注入。

## 启用和使用

在后端环境设置 `NARRATIVE_BRANCHES_ENABLED=true`，使用现有配置的真实推理服务，重启后端。在前端侧栏打开“假想分支实验室”（`/narrative`），选择角色，填写分支标题和初始假设。第一版 UI 使用基础模型加注册人物画像；分支生成 API 可显式选用映射到同一人物的 LoRA，不能自动切到其他人物。不会触发训练或改动冻结数据。

部署数据库升级使用现有 Alembic 工作流：在 backend 目录运行 `python -m alembic upgrade head`，新 head 为 `009_narrative_branches`。SQLite 开发初始化也会幂等创建三张表和消息字段；PostgreSQL 生产环境仍应先迁移。现有 Alembic 环境使用异步数据库驱动：SQLite 迁移需 aiosqlite，PostgreSQL 需 asyncpg。本轮未连接真实 PostgreSQL。

演示步骤：创建“从未相遇”分支 → 输入明确假设 → 聊天 → 查看模型提出的待确认事实 → 人工确认或拒绝 → 返回正史提问 → 创建第二个独立分支 → 验证两边历史不同。已归档分支只读。

初始假设保留用户原文，标记为未结构化前提，不假装已可靠提取三元组。“没有见过”与“没有血缘关系”是不同命题。

## 数据边界

```text
登录用户 + 角色
├── 正史：原作 RAG（只读） + 正史 Web 历史/关系 + 真实用户记忆
└── 独立分支 id
    ├── narrative_branches：归属、初始假设、状态、revision
    ├── branch_assertions：假设、待确认/有效/拒绝/撤回事实
    ├── branch_states：独立关系阶段、互动计数
    └── messages.branchId：独立公开对话历史
```

分支只读当前用户同角色的正史 Web 记忆，不调度普通记忆写入，也不调用正史 `complete_turn`。QQ 等外部身份的记忆不会自动关联到 Web 登录账号。正史 Web 使用按角色区分的 adapter，防止不同人物共享历史。所有正史角色历史查询额外过滤 `branchId IS NULL`。偏好数据从历史采样时排除分支消息。

原作索引无写入入口。分支数据以 user-role 参考区进入现有模型请求；系统只增加固定的分支语义规则。人物画像中的原作关系按原作背景解释，显式假设仅改变当前分支。对自然语言假设的全部因果后果尚不能自动核验，应通过人工和真实模型评测检查。

客户端传入的分支 history、senderId、userId、sessionId 均不作为身份或历史依据；分支归属由登录身份核验，历史从当前分支数据库读取。用户手动将假想内容粘贴为新消息仍属于新输入，不在隔离保证之内。

工作区的 `narrative` / `narrative:<character_id>` adapter 命名空间保留给内部已认证路径；普通 `/api/generate` 调用不能自行指定这些 adapter，从而防止伪造 senderId 向其他账号的正史工作区写入历史。

## 一轮生成与事务

`/api/generate` → 现有队列（用户 + 分支锁键） → 分支仓储读取权限/版本 → 加载独立历史、关系、有效事实 → 现有人物上下文编译、RAG、vLLM/模型管理器 → 解析可选提议 → revision CAS 事务提交消息、提议和关系计数。

读取完成后再次核对 revision，提交时做乐观并发比较。归档、确认、拒绝和成功生成均推进 revision。生成期间发生并发状态变更返回 409，不提交过期生成。数据库错误回滚整批写入。未持久化的模型结果不能成为有效分支状态。

分支请求绕过响应缓存。相同分支、traceId、消息和 LoRA 重试返回已提交回复，不重复更新关系或插入消息；相同 traceId 配不同内容返回 409。重放响应返回已保存正文，当前版本不保证还原原始耗时以外的全部证据元信息。无成功记录的失败请求可使用相同 traceId 重试。

模型提议使用可选 `branch_proposals` JSON fenced block，最多3条，解析失败丢弃元数据并保留正常正文；不增加第二次模型调用。提议由服务器绑定本轮已提交的 sourceMessageId 和 branch_id，模型不能指定归属；依赖 ID 必须来自本轮有效事实白名单。所有提议先为 pending。

确认需要当前 revision。同一规范化 subject/predicate 的不同 object 要求显式选择替代，撤回其传递依赖。来源缺失或依赖失效拒绝确认。不同措辞、复杂否定、因果冲突没有自动判定保证，用户须先核对。原始 source_type 保持 model_proposal，active 表示已人工确认，并不把模型来源改写为原作。

pending 不进入有效事实区，但之前的回复仍可能出现相关文本。系统明确标记历史不是权威事实，该行为仍需真实长对话验证。有效事实最多100条、参考包最多16000字符；最终请求另有 token 估算预算门禁，超出时拒绝而不是静默删掉假设。历史最多20个问答对、16000字符，从旧到新截断。

## API

所有 `/api/narrative-branches` 接口需要启用功能并登录用户账号，不接受静态或托管 API key 充当个人身份。

| 接口 | 用途 |
| --- | --- |
| GET /characters | 可用人物概要 |
| POST /canonical/generate | 当前人物正史 Web 对话，复用队列与正常记忆流程 |
| GET /canonical/messages | 当前用户、人物的正史历史 |
| POST / | 创建独立分支 |
| GET / | 分页列出自己的分支 |
| GET /{id} | 分支详情与 revision |
| POST /{id}/archive | 按 revision 归档 |
| GET /{id}/messages | 分页读取当前分支消息 |
| GET /{id}/assertions | 按状态分页读取事实 |
| POST /{id}/assertions/{assertion_id}/confirm | 确认，可提供 replace_ids |
| POST /{id}/assertions/{assertion_id}/reject | 拒绝 |

生成继续使用 `/api/generate`，传 `platform=web`、`branchId` 和稳定的 `traceId`。分支返回 `branchId`、`branchRevision` 和 `evidenceSources`。后者是本轮可用证据清单，不能解释为回复逐句蕴含验证。原作 RAG 的 `abstained` 标记仍表示原作证据不足，不应拿它直接代替对反事实正文的语义弃答评分。

AstrBot 请求对非空 branchId 返回校验失败；第一版不支持平台分支送达。既有平台送达回执流程保持使用原链路。

## 权限、删除与限制

分支无法跨用户读取，不支持共享、嵌套或合并。当前仓库没有用户账号/人物删除用例，本版本只实现归档，不提供物理删除按钮。未来实现账号数据删除必须在同一流程删除其分支消息、事实和状态，不能只删除账号。管理员原有消息管理仍可删除消息；来源被删除后，对应 pending 提议不能确认。

当前 UI 展示最近20轮、最多100条待确认提议；API 支持分页。没有自动语义冲突裁判、结构化因果图和完整的引用蕴含验证。没有部署服务，也没有在用户真实数据库上执行迁移。

## 科研开发评测

根目录执行：

```bash
python scripts/evaluate_narrative_branches.py --output output/narrative/prompts.json
python scripts/evaluate_narrative_branches.py --execute --model YOUR_MODEL --base-url http://127.0.0.1:8001/v1 --turns 20 --output output/narrative/run20.json
python scripts/evaluate_narrative_branches.py --annotations output/narrative/reviewed.json --output output/narrative/metrics.json
```

脚本默认只生成请求包；只有 `--execute` 发起真实模型请求。认证通过 `NARRATIVE_EVAL_API_KEY` 环境变量提供。输出路径不能覆盖已有文件。运行前确认模型权重版本与硬件，补充到研究记录；正式实验必须使用干净提交，脚本会记录 dirty_worktree。

内置两组自编故事、共24个10轮开发探针，不是原作 Gold，不进入训练或正式盲测。可选10/20/40轮，A共享历史、B共享历史加来源提示、C隔离历史、D隔离历史加来源证据；`D-labels` 去掉显式分支来源标记。所有臂共享相同脚本化历史、基座和生成参数，报告实际输入字符和服务端 token 使用量。隔离导致上下文长度不同，应报告这一差异。

这只是 teacher-forced 开发探针，D 是来源隔离组件测试，不代表在线完整系统的随机化实验；生命周期、冲突替代和并发用自动机制测试验证。后续还需要在线多轮实验、冲突处理消融、授权原作轨迹、更大独立盲测集、多种子及人工角色评分。

人工给每个 probe 的 labels 填布尔值或 null：correct、contaminated、cross_branch_leak、provenance_correct、abstained。正确性须匹配当前世界的 expected；正史污染须能指出采用了反事实命题；跨分支泄漏须指出采用了其他分支命题；来源正确须验证回答声称的来源；拒答包含未回答目标问题的规避表述。无法判断标 null 并保留理由，不当成正确。

指标按 arm 分组，分母是适用且该字段已裁定的探针（包括已裁定弃答），同时报告 eligible、unknown 和覆盖数量。拒答不能计入正确回答或正确来源归因。低污染率必须与正确率、弃答率一起读，缺少标签时指标为 null。模型调用失败另见 error_type，不可丢弃后只报告成功样本。暂未计算置信区间，不支持据此作显著性结论。

## 本地验证记录（2026-09-18）

- 新增分支机制测试覆盖权限、状态隔离、空历史、缓存旁路、来源约束、直接冲突、失效依赖、原子回滚、归档竞态、重复请求、保留命名空间和 Web 生成后返回正史。模型调用在测试中由固定实现替代。
- 分支、权限架构、限流、PG兼容/缓存回归组合：92 passed，11 skipped。跳过项没有计入已验证能力。
- 生成链路、模型回退、迁移、Schema 和运行容器相关测试分批验证；发现的旧上下文对象缺少 branch_context 兼容问题已修复，个人作用域写接口已纳入明确权限契约。
- 前端 `pnpm ts-check`、`pnpm lint`、`pnpm build` 通过，构建包含 `/narrative`。
- 新增 Python 模块 Ruff 检查通过；对部分旧模块执行宽范围 Ruff 时仍发现原有日志/导入风格问题，没有为此批量改写用户文件。
- 开发请求包生成成功：`output/narrative-development-20260918/prompts.json`（24个探针，prepared_prompts_only），不含真实模型结果。
- 未进行浏览器交互验收、真实 vLLM 评测、真实 PostgreSQL 事务运行或部署。下一阶段应先启用功能并完成实际模型演示，再开展盲评。
